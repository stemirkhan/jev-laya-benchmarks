"""Download public BEIR archives and tokenizer/config files only; no model weights."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import stat
import time
import zipfile

import requests

ROOT = Path(__file__).resolve().parent
DATASETS = {
    "scifact": "5f7d1de60b170fc8027bb7898e2efca1",
    "fiqa": "17918ed23cd04fb15047f73e6c3bd9d9",
    "nfcorpus": "a89dba18a62ef92f7d323ec890a0d38d",
    "trec-covid": "ce62140cb23feb9becf6270d0d1fe6d1",
}
MODELS = {
    "laya-en": "convaiinnovations/laya",
    "laya-multi": "convaiinnovations/laya-multilingual",
    "bge": "BAAI/bge-reranker-v2-m3",
    "qwen-0.6b": "Qwen/Qwen3-Reranker-0.6B",
    "qwen-8b": "Qwen/Qwen3-Reranker-8B",
}
# Revisions recorded by the original preparation manifests. A fresh public
# checkout must not silently resolve newer tokenizers than the experiment used.
MODEL_REVISIONS = {
    "laya-en": "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
    "laya-multi": "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67",
    "bge": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    "qwen-0.6b": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
    "qwen-8b": "77d193c791ed757ca307ee72715aa132723da912",
}
LAYA_CODE_REVISION = "4066d5d5fbf08b66c6757ddeedbd797bd7655bc0"


def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def download(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 4):
        part = path.with_name(path.name + ".part")
        try:
            with requests.get(url, stream=True, timeout=(20, 120)) as response:
                response.raise_for_status()
                with part.open("wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        output.write(chunk)
            part.replace(path)
            return
        except requests.RequestException as exc:
            print(f"download attempt {attempt}/3 failed: {url}: {exc}", flush=True)
            if attempt == 3:
                raise
            time.sleep(attempt)


def extract_archive(archive: Path, destination: Path, dataset: str) -> None:
    """Validate every member before extraction; reject traversal and symlinks."""
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        for entry in members:
            target = (root / entry.filename).resolve()
            if not target.is_relative_to(root / dataset):
                raise ValueError(f"Unsafe archive member: {entry.filename}")
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError(f"Symlink in archive: {entry.filename}")
        for entry in members:
            target = root / entry.filename
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                temp = target.with_name(target.name + ".extracting")
                with bundle.open(entry) as source, temp.open("wb") as output:
                    shutil.copyfileobj(source, output)
                temp.replace(target)


def prepare_dataset(name: str) -> dict:
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
    archive = ROOT / "data" / "archives" / f"{name}.zip"
    download(url, archive)
    actual = digest(archive, "md5")
    if actual != DATASETS[name]:
        raise ValueError(f"BEIR checksum mismatch for {name}: {actual}")
    extract_archive(archive, ROOT / "data" / "beir", name)
    folder = ROOT / "data" / "beir" / name
    files = {str(p.relative_to(folder)): digest(p) for p in sorted(folder.rglob("*")) if p.is_file()}
    result = {"name": name, "url": url, "md5": actual, "sha256": digest(archive),
              "bytes": archive.stat().st_size, "files_sha256": files}
    write_json(ROOT / "assets" / "manifests" / f"dataset-{name}.json", result)
    print(f"dataset ready: {name}, {archive.stat().st_size / 1e6:.1f} MB archive", flush=True)
    return result


def prepare_model(alias: str) -> dict:
    repo = MODELS[alias]
    manifest = ROOT / "assets" / "manifests" / f"model-{alias}.json"
    if manifest.exists():
        info = json.loads(manifest.read_text())
        if info["repo"] != repo or info["revision"] != MODEL_REVISIONS[alias]:
            raise ValueError(f"Preparation manifest differs from pinned experiment: {alias}")
    else:
        response = requests.get(f"https://huggingface.co/api/models/{repo}/revision/{MODEL_REVISIONS[alias]}", timeout=30)
        response.raise_for_status()
        remote = response.json()
        if remote["sha"] != MODEL_REVISIONS[alias]:
            raise ValueError(f"Hugging Face resolved a different pinned revision: {alias}")
        names = {item["rfilename"] for item in remote["siblings"]}
        candidates = (
            ["tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json", "rl_agent_config.json", "encoder/config.json", "README.md"]
            if alias.startswith("laya") else
            ["tokenizer.json", "tokenizer_config.json", "config.json", "special_tokens_map.json", "README.md", "chat_template.jinja"]
        )
        info = {"alias": alias, "repo": repo, "revision": remote["sha"],
                "files": [name for name in candidates if name in names]}
        write_json(manifest, info)
    folder = ROOT / "assets" / "models" / alias
    for name in info["files"]:
        download(f"https://huggingface.co/{repo}/resolve/{info['revision']}/{name}", folder / name)
        previous = info.get("files_sha256", {}).get(name)
        if previous and digest(folder / name) != previous:
            raise ValueError(f"Changed tokenizer/config file: {alias}/{name}")
    info["files_sha256"] = {name: digest(folder / name) for name in info["files"]}
    write_json(manifest, info)
    print(f"tokenizer ready: {alias} @ {info['revision']}", flush=True)
    return info


def prepare_sources() -> None:
    base = f"https://raw.githubusercontent.com/NandhaKishorM/laya/{LAYA_CODE_REVISION}"
    for name in ("laya/common.py", "laya/agent.py", "LICENSE"):
        download(f"{base}/{name}", ROOT / "assets" / "sources" / "laya" / name)
    download("https://raw.githubusercontent.com/beir-cellar/beir/main/README.md",
             ROOT / "assets" / "sources" / "beir-README.md")
    # Dataset cards are retained for license review, not treated as a redistribution grant.
    for name in DATASETS:
        try:
            download(f"https://huggingface.co/datasets/BeIR/{name}/raw/main/README.md",
                     ROOT / "assets" / "sources" / f"beir-{name}-card.md")
        except requests.RequestException:
            print(f"Dataset card unavailable: {name}; consult original dataset owner", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["all", "datasets", "models"], default="all")
    args = parser.parse_args()
    jobs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        if args.only in ("all", "datasets"):
            jobs += [pool.submit(prepare_dataset, name) for name in DATASETS]
        if args.only in ("all", "models"):
            jobs += [pool.submit(prepare_model, name) for name in MODELS]
        for future in concurrent.futures.as_completed(jobs):
            future.result()
    prepare_sources()


if __name__ == "__main__":
    main()
