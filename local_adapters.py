"""Offline local rerankers. Loading is explicit; importing needs no torch/transformers.

Weights must be prepared separately with a manifest for the pinned revision. No
adapter downloads files, truncates input, changes precision, or retries on CPU.
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import sys
import tempfile

from audit import INSTRUCTION, QUESTION, OPTIONS, QWEN_PREFIX, QWEN_SUFFIX

CODE_REVISION = "4066d5d5fbf08b66c6757ddeedbd797bd7655bc0"
MODEL_IDS = {
    "laya-multi": "convaiinnovations/laya-multilingual",
    "bge": "BAAI/bge-reranker-v2-m3",
    "qwen-0.6b": "Qwen/Qwen3-Reranker-0.6B",
    "qwen-8b": "Qwen/Qwen3-Reranker-8B",
}
LIMITS = {"laya-multi": 8192, "bge": 8192, "qwen-0.6b": 32768, "qwen-8b": 32768}


def finite(value, name="score") -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} is boolean")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {name}")
    return result


def yes_probability(no_logit, yes_logit, temperature=1.0) -> float:
    """Stable two-class softmax, without the four-decimal API presentation rounding."""
    no, yes = finite(no_logit, "no logit"), finite(yes_logit, "yes logit")
    temperature = finite(temperature, "temperature")
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    delta = (yes - no) / temperature
    if delta >= 0:
        return 1.0 / (1.0 + math.exp(-delta))
    value = math.exp(delta)
    return value / (1.0 + value)


def pairs_checked(pairs):
    checked = []
    for pair in pairs:
        if isinstance(pair, dict):
            query, document = pair["query"], pair["document"]
        else:
            query, document = pair
        if not isinstance(query, str) or not isinstance(document, str):
            raise TypeError("Each query/document must be a string")
        checked.append((query, document))
    return checked


def check_lengths(ids, max_length):
    lengths = [len(row) for row in ids]
    for index, length in enumerate(lengths):
        if length > max_length:
            raise ValueError(f"Input {index} has {length} tokens, limit {max_length}; truncation is forbidden")
        if length == 0:
            raise ValueError(f"Input {index} is empty after formatting")
    return lengths


def qwen_inputs(tokenizer, pairs, max_length, instruction=INSTRUCTION):
    prefix = tokenizer.encode(QWEN_PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(QWEN_SUFFIX, add_special_tokens=False)
    bodies = [f"<Instruct>: {instruction}\n<Query>: {q}\n<Document>: {d}" for q, d in pairs]
    encoded = tokenizer(bodies, padding=False, truncation=False,
                        add_special_tokens=False, return_attention_mask=False)
    ids = [prefix + row + suffix for row in encoded["input_ids"]]
    lengths = check_lengths(ids, max_length)
    return {"input_ids": ids, "attention_mask": [[1] * len(row) for row in ids]}, lengths


def bge_inputs(tokenizer, pairs, max_length):
    encoded = tokenizer([q for q, _ in pairs], text_pair=[d for _, d in pairs],
                        padding=False, truncation=False, add_special_tokens=True)
    lengths = check_lengths(encoded["input_ids"], max_length)
    return encoded, lengths


def laya_sequence(tokenizer, state, instruction, max_length, head_max_length):
    """Same Noul sequence as pinned build_sequence, but every clipping path is an error."""
    mask = tokenizer.mask_token
    if not mask or any(getattr(tokenizer, field) is None for field in
                       ("mask_token_id", "cls_token_id", "sep_token_id")):
        raise ValueError("Laya tokenizer is missing special tokens")

    def encode(text):
        return tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]

    head = encode("noul question: " + instruction.replace(mask, " "))
    options = [encode(" " + option.replace(mask, " ")) for option in OPTIONS]
    if any(len(option) > 48 for option in options):
        raise ValueError("Laya option would be truncated")
    budget = head_max_length - sum(len(option) + 1 for option in options)
    if budget < 16 or len(head) > max(8, budget):
        raise ValueError("Laya instruction/options would be truncated")
    ids = [tokenizer.cls_token_id] + head + [tokenizer.sep_token_id]
    markers = []
    for option in options:
        markers.append(len(ids))
        ids.extend([tokenizer.mask_token_id] + option)
    ids.append(tokenizer.sep_token_id)
    # The exact official state serialization and MASK escaping are retained.
    state_text = json.dumps(state, ensure_ascii=False).replace(mask, " ")
    ids.extend(encode(state_text))
    ids.append(tokenizer.sep_token_id)
    check_lengths([ids], max_length)
    return {"ids": ids, "markers": markers, "qtype": 2}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(folder, manifest_path, *, revision, repo=None, code=False):
    """Require complete hashes, not merely a directory named after a commit SHA."""
    folder, manifest_path = Path(folder).resolve(), Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("revision") != revision or (repo and manifest.get("repo") != repo):
        raise ValueError("Artifact manifest repo/revision does not match requested model")
    expected = manifest.get("files_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Artifact manifest must contain non-empty files_sha256")
    for relative, digest in expected.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Invalid artifact manifest path/hash")
        if not (folder / path).is_file() or sha256(folder / path) != digest:
            raise ValueError(f"Artifact hash mismatch/missing file: {relative}")
    present = {p.relative_to(folder).as_posix() for p in folder.rglob("*")
               if p.is_file() and p.resolve() != manifest_path
               and "__pycache__" not in p.parts and ".git" not in p.parts}
    if code:
        # Extra documentation is harmless, but all executable Python must be covered.
        present = {name for name in present if name.endswith(".py")}
    missing = present - set(expected)
    if missing:
        raise ValueError(f"Unverified files in artifact directory: {sorted(missing)[:5]}")
    if not code and not any(name.endswith(".safetensors") for name in expected):
        raise ValueError("Manifest contains no model weights; tokenizer-only audit assets cannot run inference")
    return {"manifest_sha256": sha256(manifest_path), "files_sha256": expected}


def _resolve(path, root):
    path = Path(path)
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _runtime(config):
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError("Local inference requires separately installed torch, transformers and safetensors") from exc
    dtype_name = config["dtype"]
    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if dtype_name not in dtypes:
        raise ValueError("dtype must be bf16, fp16 or fp32")
    device = torch.device(config["device"])
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Benchmark supports explicit cpu or cuda:N devices only")
    if device.type == "cpu" and dtype_name != "fp32":
        raise ValueError("CPU adapter requires explicit fp32; no implicit dtype fallback")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
        if device.index is None:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        with torch.cuda.device(device):
            if dtype_name == "bf16" and not torch.cuda.is_bf16_supported():
                raise RuntimeError("Requested CUDA device does not support bf16")
    return torch, transformers, device, dtypes[dtype_name]


class _Base:
    def __init__(self, alias, config, project_root=None):
        self.alias, self.config = alias, dict(config)
        self.root = Path(project_root or Path(__file__).resolve().parent)
        self.max_length = config["max_length"]
        if type(self.max_length) is not int or not 0 < self.max_length <= LIMITS[alias]:
            raise ValueError(f"Invalid max_length for {alias}")
        if config["repo"] != MODEL_IDS[alias] or not re.fullmatch(r"[a-f0-9]{40}", config["revision"]):
            raise ValueError("Use the expected repository and an immutable 40-character revision")
        for key in ("model_path", "model_manifest"):
            if not config.get(key):
                raise ValueError(f"{key} is required; adapters never download weights")
        self.model_path = _resolve(config["model_path"], self.root)
        integrity = verify_manifest(self.model_path, _resolve(config["model_manifest"], self.root),
                                    revision=config["revision"], repo=config["repo"])
        self.torch, self.transformers, self.device, self.dtype = _runtime(config)
        self.info = {"alias": alias, "repo": config["repo"], "revision": config["revision"],
                     "max_length": self.max_length, "dtype": config["dtype"],
                     "device": str(self.device), "model_path": str(self.model_path),
                     "model_manifest_sha256": integrity["manifest_sha256"], "truncation": False,
                     "offline": True, "fallback": False,
                     "torch": self.torch.__version__, "transformers": self.transformers.__version__}
        self.info["runtime"] = {
            "cuda": self.torch.version.cuda, "cudnn": self.torch.backends.cudnn.version(),
            "matmul_allow_tf32": self.torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": self.torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": self.torch.backends.cudnn.benchmark,
            "deterministic_algorithms": self.torch.are_deterministic_algorithms_enabled(),
        }
        if self.device.type == "cuda":
            prop = self.torch.cuda.get_device_properties(self.device)
            self.info["gpu"] = {"name": prop.name, "uuid": str(getattr(prop, "uuid", "unavailable")),
                                "total_memory": prop.total_memory, "major": prop.major, "minor": prop.minor,
                                "multiprocessor_count": prop.multi_processor_count}
            import subprocess
            try:
                self.info["nvidia_smi"] = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total", "--format=csv,noheader"],
                    text=True, timeout=10).strip()
            except (FileNotFoundError, subprocess.SubprocessError):
                self.info["nvidia_smi"] = "unavailable"

    def metadata(self):
        return dict(self.info)

    def synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def reset_peak_memory(self):
        if self.device.type == "cuda":
            self.torch.cuda.reset_peak_memory_stats(self.device)

    def memory_stats(self):
        if self.device.type != "cuda":
            return {"device": str(self.device), "peak_cuda_allocated_bytes": None}
        return {"peak_cuda_allocated_bytes": self.torch.cuda.max_memory_allocated(self.device),
                "peak_cuda_reserved_bytes": self.torch.cuda.max_memory_reserved(self.device)}

    def score_batch(self, pairs):
        return self.score(pairs_checked(pairs))

    def close(self):
        self.model = None
        if hasattr(self, "agent"):
            self.agent = None
        if hasattr(self, "_runtime_dir"):
            self._runtime_dir.cleanup()
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()


class TransformerAdapter(_Base):
    def __init__(self, alias, config, project_root=None):
        super().__init__(alias, config, project_root)
        is_qwen = alias.startswith("qwen")
        self.tokenizer = self.transformers.AutoTokenizer.from_pretrained(
            str(self.model_path), local_files_only=True, trust_remote_code=False,
            padding_side="left" if is_qwen else "right")
        cls = (self.transformers.AutoModelForCausalLM if is_qwen else
               self.transformers.AutoModelForSequenceClassification)
        attention = config.get("attn_implementation", "sdpa")
        if attention not in ("sdpa", "eager", "flash_attention_2"):
            raise ValueError("Unsupported explicit attention implementation")
        self.model = cls.from_pretrained(str(self.model_path), local_files_only=True,
                                         trust_remote_code=False, torch_dtype=self.dtype,
                                         attn_implementation=attention).to(self.device).eval()
        self.instruction = config.get("instruction", INSTRUCTION)
        self.info.update(attn_implementation=attention,
                         score_type="p_yes_given_yes_no" if is_qwen else "relevance_logit",
                         instruction=self.instruction if is_qwen else None)
        if is_qwen:
            self.no_id = self.tokenizer.convert_tokens_to_ids("no")
            self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
            if self.no_id == self.yes_id or any(
                self.tokenizer.encode(word, add_special_tokens=False) != [value]
                for word, value in (("no", self.no_id), ("yes", self.yes_id))
            ):
                raise ValueError("Qwen yes/no labels are not distinct single tokens")
            self.info.update(no_token_id=self.no_id, yes_token_id=self.yes_id)

    def score(self, pairs):
        pairs = pairs_checked(pairs)
        if not pairs:
            return []
        is_qwen = self.alias.startswith("qwen")
        encoded, lengths = (qwen_inputs(self.tokenizer, pairs, self.max_length, self.instruction)
                            if is_qwen else bge_inputs(self.tokenizer, pairs, self.max_length))
        inputs = self.tokenizer.pad(encoded, padding=True, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            # Only the final position is needed; avoid a full [batch, length, vocab] allocation.
            extra = {"use_cache": False, "logits_to_keep": 1} if is_qwen else {}
            logits = self.model(**inputs, **extra).logits
            if is_qwen:
                if logits.shape[0] != len(pairs):
                    raise ValueError("Unexpected Qwen batch dimension")
                rows = logits[:, -1, [self.no_id, self.yes_id]].float().cpu().tolist()
                return [{"score": yes_probability(no, yes),
                         "raw": {"no_logit": finite(no), "yes_logit": finite(yes)},
                         "input_tokens": length} for (no, yes), length in zip(rows, lengths)]
            if tuple(logits.shape) != (len(pairs), 1):
                raise ValueError(f"Expected one BGE logit per pair, got {tuple(logits.shape)}")
            rows = logits[:, 0].float().cpu().tolist()
            return [{"score": finite(value), "raw": {"logit": finite(value)},
                     "input_tokens": length} for value, length in zip(rows, lengths)]


def _import_verified_laya(code_path, code_manifest):
    integrity = verify_manifest(code_path, code_manifest, revision=CODE_REVISION, code=True)
    required = {"laya/__init__.py", "laya/agent.py", "laya/common.py", "laya/hooks.py", "laya/revisions.py"}
    if not required <= set(integrity["files_sha256"]):
        raise ValueError("A complete verified Laya source checkout is required, not only audit excerpts")
    for name, module in list(sys.modules.items()):
        if name == "laya" or name.startswith("laya."):
            origin = getattr(module, "__file__", None)
            if not origin or not Path(origin).resolve().is_relative_to(code_path):
                raise RuntimeError("A different Laya installation is already loaded; use a fresh process")
    sys.path.insert(0, str(code_path))
    try:
        module = importlib.import_module("laya.agent")
        if not Path(module.__file__).resolve().is_relative_to(code_path):
            raise RuntimeError("Laya import resolved outside the verified code checkout")
        return module.Agent
    finally:
        sys.path.remove(str(code_path))


class LayaAdapter(_Base):
    def __init__(self, alias, config, project_root=None):
        super().__init__(alias, config, project_root)
        code_path = _resolve(config["code_path"], self.root)
        code_manifest = _resolve(config["code_manifest"], self.root)
        Agent = _import_verified_laya(code_path, code_manifest)
        instruction = config.get("instruction", QUESTION)
        max_length = self.max_length
        torch = self.torch

        class StrictAgent(Agent):
            def _encode_state(agent, state, ids, internal, max_len=None, head_max_len=None):
                if ids != ["relevant"] or internal["relevant"] != {"t": "noul", "ins": instruction, "crit": None}:
                    raise ValueError("Unexpected Laya benchmark question format")
                return [laya_sequence(agent.tok, state, instruction,
                                      max_len if max_len is not None else max_length,
                                      head_max_len if head_max_len is not None else agent.cfg["head_max_len"])]

            def _infer(agent, batch):
                context = torch.autocast(device_type=agent.device.type, dtype=agent.dtype) if agent.amp_enabled else nullcontext()
                with context:
                    return agent.model(*(batch[key].to(agent.device) for key in
                                         ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")))

            def _decode_answers(agent, logits, act, items, ids, internal, offset, lang=None):
                result = super()._decode_answers(logits, act, items, ids, internal, offset, lang=lang)
                temperature = agent.temperature_by_options.get("noul:2", agent.temperature[2])
                no, yes = (finite(value) for value in logits[offset, :2])
                result["relevant"]["benchmark_unrounded_noul"] = yes_probability(no, yes, temperature)
                result["relevant"]["benchmark_raw_logits"] = [no, yes]
                result["relevant"]["benchmark_temperature"] = finite(temperature)
                return result

        # Agent's compatibility patch may rewrite tokenizer_config.json. Give it a
        # disposable symlink tree so the verified immutable source stays unchanged.
        for required in ("model.safetensors", "rl_agent_config.json", "encoder/config.json",
                         "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"):
            if not (self.model_path / required).is_file():
                raise ValueError(f"Incomplete Laya checkpoint: {required}")
        self._runtime_dir = tempfile.TemporaryDirectory(prefix="jev-laya-runtime-")
        runtime_path = Path(self._runtime_dir.name)
        for source in self.model_path.rglob("*"):
            if source.is_file():
                destination = runtime_path / source.relative_to(self.model_path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.symlink_to(source)
        # Loading on CPU avoids Agent's load-time GPU fallback catch. Explicit move
        # below propagates OOM/device errors, and StrictAgent never retries inference.
        self.agent = StrictAgent(str(runtime_path), device="cpu", fast=False, compile=False)
        self.agent.model.to(self.device).eval()
        self.agent.device, self.agent.dtype = self.device, self.dtype
        self.agent.amp_enabled = self.config["dtype"] != "fp32"
        self.model = self.agent.model
        self.instruction = instruction
        self.info.update(code_revision=CODE_REVISION, code_manifest_sha256=sha256(code_manifest),
                         instruction=instruction, score_type="unrounded_temperature_scaled_p_true",
                         resident_weights_dtype="fp32", precision_mode="autocast" if self.agent.amp_enabled else "fp32",
                         attn_implementation="sdpa", max_len_override=self.max_length,
                         tokenizer_runtime_config_sha256=sha256(runtime_path / "tokenizer/tokenizer_config.json"))

    def score(self, pairs):
        pairs = pairs_checked(pairs)
        if not pairs:
            return []
        states = [{"query": query, "document": doc} for query, doc in pairs]
        with self.torch.inference_mode():
            responses = self.agent.predict_batch(
                states, {"relevant": {"type": "noul", "instructions": self.instruction}},
                batch_size=len(states), max_len=self.max_length, sort_by_length=False)
        if len(responses) != len(pairs):
            raise ValueError("Laya returned wrong number of results")
        results = []
        for response in responses:
            score = finite(response["answers"]["relevant"]["benchmark_unrounded_noul"])
            count = response["usage"]["input_tokens"]
            if not 0 <= score <= 1 or type(count) is not int or not 0 < count <= self.max_length:
                raise ValueError("Invalid Laya score or token usage")
            json.dumps(response, allow_nan=False)
            results.append({"score": score, "raw": response, "input_tokens": count})
        return results


def create_local_adapter(alias, config, project_root=None):
    if alias not in MODEL_IDS:
        raise ValueError(f"Unknown local adapter {alias}")
    cls = LayaAdapter if alias == "laya-multi" else TransformerAdapter
    return cls(alias, config, project_root)
