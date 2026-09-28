"""Explicit local CUDA smoke test. No downloads, model inference or API calls."""
import argparse
import json
import platform
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    import torch
    import transformers
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; CPU fallback is forbidden")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("Select a CUDA device")
    prop = torch.cuda.get_device_properties(device)
    result = {"python": platform.python_version(), "torch": torch.__version__,
              "transformers": transformers.__version__, "cuda": torch.version.cuda,
              "gpu": prop.name, "uuid": str(getattr(prop, "uuid", "unavailable")),
              "vram_bytes": prop.total_memory, "capability": [prop.major, prop.minor],
              "native_models": {}, "matmul": {}}
    result["nvidia_smi"] = subprocess.check_output(["nvidia-smi"], text=True, timeout=15)
    for name in ("Qwen3ForCausalLM", "ModernBertModel", "XLMRobertaForSequenceClassification"):
        getattr(transformers, name)
        result["native_models"][name] = True
    with torch.inference_mode():
        for name, dtype in (("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)):
            a = torch.ones((128, 128), device=device, dtype=dtype)
            output = a @ a
            torch.cuda.synchronize(device)
            if not torch.isfinite(output).all() or not torch.all(output == 128):
                raise RuntimeError(f"Invalid CUDA matmul result for {name}")
            result["matmul"][name] = "ok"
    result["model_inference_verified"] = False
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
