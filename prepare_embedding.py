"""Download a pinned official ONNX embedding model, without remote Python code."""
import json
from pathlib import Path

from prepare import ROOT, digest, download, write_json


def main():
    config = json.loads((ROOT / "config/retrieval.json").read_text())["embedding"]
    folder = ROOT / "assets/embedding"
    manifest = ROOT / "assets/manifests/embedding.json"
    names = [config["model_file"], "tokenizer.json", "tokenizer_config.json", "config.json", "README.md"]
    previous = json.loads(manifest.read_text()) if manifest.exists() else None
    if previous and (previous["repo"], previous["revision"]) != (config["repo"], config["revision"]):
        raise ValueError("Embedding revision changed: use a new asset directory")
    for name in names:
        download(f"https://huggingface.co/{config['repo']}/resolve/{config['revision']}/{name}", folder / name)
        if previous and previous["files_sha256"][name] != digest(folder / name):
            raise ValueError(f"Embedding file changed: {name}")
        print(f"embedding asset ready: {name}", flush=True)
    write_json(manifest, {"repo": config["repo"], "revision": config["revision"],
                         "files_sha256": {name: digest(folder / name) for name in names}})


if __name__ == "__main__":
    main()
