"""Prepare pinned inference artifacts only with explicit --download.

The default is a local plan: no HTTP requests, model imports or file writes.
Inference is never run by this script. Temporary/partial downloads are preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

import requests

from local_adapters import CODE_REVISION, MODEL_IDS, verify_manifest
from prepare import digest, write_json

ROOT = Path(__file__).resolve().parent
LAYA_REPO = "NandhaKishorM/laya"
LAYA_URL = f"https://codeload.github.com/{LAYA_REPO}/zip/{CODE_REVISION}"
TOKENIZER_FILES = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "added_tokens.json", "tokenizer.model", "sentencepiece.bpe.model",
                   "vocab.json", "vocab.txt", "merges.txt", "chat_template.jinja"}


def local_path(root, value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Artifact paths must be relative to the project without '..'")
    target = (root / path).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes project")
    return target


def safe_relative(value):
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or not path.parts:
        raise ValueError(f"Unsafe relative path {value!r}")
    return path


def choose_files(alias, siblings):
    """One native safetensors representation, no ONNX, bins or sentence-transformer copies."""
    chosen = []
    for item in siblings:
        name = item["rfilename"]
        path = safe_relative(name)
        if alias == "laya-multi":
            wanted = (name in {"model.safetensors", "rl_agent_config.json", "encoder/config.json"}
                      or len(path.parts) == 2 and path.parts[0] == "tokenizer" and path.name in TOKENIZER_FILES)
        else:
            wanted = len(path.parts) == 1 and (
                name in TOKENIZER_FILES | {"config.json", "generation_config.json", "model.safetensors.index.json"}
                or name == "model.safetensors" or re.fullmatch(r"model-\d+-of-\d+\.safetensors", name))
        if wanted:
            size = item.get("size", item.get("lfs", {}).get("size"))
            lfs_sha = item.get("lfs", {}).get("sha256")
            git_sha = item.get("blobId")
            if type(size) is not int or size <= 0:
                raise ValueError(f"Missing file size in pinned HF metadata: {name}")
            if lfs_sha and re.fullmatch(r"[a-f0-9]{64}", lfs_sha):
                expected = {"sha256": lfs_sha}
            elif git_sha and re.fullmatch(r"[a-f0-9]{40}", git_sha) and not item.get("lfs"):
                expected = {"git_blob_sha1": git_sha}
            else:
                raise ValueError(f"Missing trusted content digest in pinned HF metadata: {name}")
            chosen.append({"name": name, "size": size, **expected})
    names = {item["name"] for item in chosen}
    required = ({"model.safetensors", "rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json",
                 "tokenizer/tokenizer_config.json"} if alias == "laya-multi" else {"config.json", "tokenizer.json"})
    if not required <= names or not any(name.endswith(".safetensors") for name in names):
        raise ValueError(f"Pinned repository lacks required native checkpoint files: {alias}")
    if "model.safetensors" in names and any(name.startswith("model-") for name in names):
        raise ValueError("Repository contains both single and sharded weights; choose one explicitly")
    return sorted(chosen, key=lambda row: row["name"])


def git_blob_sha1(path):
    h = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_file(path, expected):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected regular artifact file: {path}")
    if expected.get("size") is not None and path.stat().st_size != expected["size"]:
        raise ValueError(f"Artifact size mismatch: {path}")
    if expected.get("sha256") and digest(path) != expected["sha256"]:
        raise ValueError(f"Artifact SHA256 mismatch: {path}")
    if expected.get("git_blob_sha1") and git_blob_sha1(path) != expected["git_blob_sha1"]:
        raise ValueError(f"Artifact Git blob hash mismatch: {path}")


def download_file(url, path, expected, session):
    """Resume .part with HTTP Range; validate before atomic rename; never overwrite a bad final file."""
    if path.exists():
        check_file(path, expected)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    if part.is_symlink():
        raise ValueError("Partial download must not be a symlink")
    start = part.stat().st_size if part.exists() else 0
    if expected.get("size") is not None and start > expected["size"]:
        raise ValueError(f"Partial download is larger than source: {part}")
    if expected.get("size") == start:
        check_file(part, expected)
        part.replace(path)
        return
    headers = {"Accept-Encoding": "identity"}
    if start:
        headers["Range"] = f"bytes={start}-"
    with session.get(url, headers=headers, stream=True, timeout=(20, 180)) as response:
        response.raise_for_status()
        if response.status_code not in (200, 206):
            raise ValueError(f"Unexpected download status {response.status_code}")
        if response.status_code == 206:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if not match or int(match[1]) != start:
                raise ValueError("Incorrect Content-Range on resumed download")
            if expected.get("size") is not None and int(match[3]) != expected["size"]:
                raise ValueError("Content-Range size differs from pinned file metadata")
        else:
            # Some CDNs ignore Range. Restart just the partial file, never an existing final artifact.
            start = 0
        with part.open("ab" if start else "wb") as output:
            for block in response.iter_content(1024 * 1024):
                output.write(block)
    check_file(part, expected)
    part.replace(path)


def prepare_model(alias, config, root, session):
    folder = local_path(root, config["model_path"])
    manifest_path = local_path(root, config["model_manifest"])
    if manifest_path.exists():
        verify_manifest(folder, manifest_path, revision=config["revision"], repo=config["repo"])
        return
    progress = manifest_path.with_suffix(".download.json")
    if progress.exists():
        plan = json.loads(progress.read_text())
        if plan.get("repo") != config["repo"] or plan.get("revision") != config["revision"]:
            raise ValueError("Partial download plan does not match pinned configuration")
    else:
        response = session.get(f"https://huggingface.co/api/models/{config['repo']}/revision/{config['revision']}",
                               params={"blobs": "true"}, timeout=60)
        response.raise_for_status()
        info = response.json()
        if info["sha"] != config["revision"]:
            raise ValueError("Hugging Face resolved a different commit")
        plan = {"repo": config["repo"], "revision": config["revision"],
                "files": choose_files(alias, info["siblings"])}
        write_json(progress, plan)
    allowed = {row["name"] for row in plan["files"]}
    if folder.exists():
        extra = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()} - allowed - {name + ".part" for name in allowed}
        if extra:
            raise ValueError(f"Unexpected files in checkpoint directory: {sorted(extra)}")
    for row in plan["files"]:
        safe_relative(row["name"])
        if not (folder / row["name"]).resolve().is_relative_to(folder.resolve()):
            raise ValueError("Checkpoint file path escapes its directory through a symlink")
        url = f"https://huggingface.co/{config['repo']}/resolve/{config['revision']}/{row['name']}"
        print(f"prepare {alias}: {row['name']}", flush=True)
        download_file(url, folder / row["name"], row, session)
    if "model.safetensors.index.json" in allowed:
        index = json.loads((folder / "model.safetensors.index.json").read_text())
        if not set(index["weight_map"].values()) <= allowed:
            raise ValueError("Safetensors index references a missing/unselected shard")
    result = {"repo": config["repo"], "revision": config["revision"],
              "source_metadata": str(progress.relative_to(root)),
              "files_sha256": {row["name"]: digest(folder / row["name"]) for row in plan["files"]}}
    write_json(manifest_path, result)
    verify_manifest(folder, manifest_path, revision=config["revision"], repo=config["repo"])


def selected_code_members(bundle):
    """Validate the entire archive before extracting any member, including ignored files."""
    prefix = f"laya-{CODE_REVISION}"
    selected, seen = [], set()
    for entry in bundle.infolist():
        path = safe_relative(entry.filename)
        if not path.parts or path.parts[0] != prefix:
            raise ValueError("Unexpected root in pinned Laya archive")
        mode = entry.external_attr >> 16
        if stat.S_ISLNK(mode) or stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise ValueError(f"Link/special file in source archive: {entry.filename}")
        if entry.filename in seen:
            raise ValueError("Duplicate archive member")
        seen.add(entry.filename)
        if entry.is_dir() or len(path.parts) == 1:
            continue
        relative = PurePosixPath(*path.parts[1:])
        wanted = (relative.parts[0] == "laya" and relative.suffix == ".py"
                  or str(relative) in {"pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "LICENSE", "README.md"})
        if wanted:
            selected.append((entry, str(relative)))
    if not {"laya/__init__.py", "laya/agent.py", "laya/common.py", "laya/hooks.py", "laya/revisions.py"} <= {name for _, name in selected}:
        raise ValueError("Source archive lacks complete Laya package")
    return selected


def extract_code(archive, destination):
    with zipfile.ZipFile(archive) as bundle:
        members = selected_code_members(bundle)
        allowed = {relative for _, relative in members}
        if destination.exists():
            extra = {p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts} - allowed - {name + ".extracting" for name in allowed}
            if extra:
                raise ValueError(f"Unverified files already in source checkout: {sorted(extra)}")
        hashes = {}
        for entry, relative in members:
            target = destination / relative
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError("Source extraction path escapes destination")
            # Package files are small. Reading the member also validates its zip CRC.
            payload = bundle.read(entry)
            expected = hashlib.sha256(payload).hexdigest()
            if target.exists():
                if target.is_symlink() or digest(target) != expected:
                    raise ValueError(f"Existing source file differs from pinned archive: {target}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_name(target.name + ".extracting")
                if partial.is_symlink():
                    raise ValueError("Source extraction temporary file must not be a symlink")
                partial.write_bytes(payload)
                partial.replace(target)
            hashes[relative] = expected
    return hashes


def prepare_laya_code(config, root, session):
    folder = local_path(root, config["code_path"])
    manifest_path = local_path(root, config["code_manifest"])
    if manifest_path.exists():
        verify_manifest(folder, manifest_path, revision=CODE_REVISION, code=True)
        return
    archive = manifest_path.parent / f"laya-{CODE_REVISION}.zip"
    download_file(LAYA_URL, archive, {}, session)
    files = extract_code(archive, folder)
    write_json(manifest_path, {"repo": LAYA_REPO, "revision": CODE_REVISION,
                               "source_url": LAYA_URL, "archive_sha256": digest(archive), "files_sha256": files})
    verify_manifest(folder, manifest_path, revision=CODE_REVISION, code=True)


def plan(config, aliases=None):
    aliases = aliases or list(MODEL_IDS)
    result = []
    for alias in aliases:
        row = config["models"][alias]
        if row["repo"] != MODEL_IDS[alias] or not re.fullmatch(r"[a-f0-9]{40}", row["revision"]):
            raise ValueError("Expected known model repository and pinned commit")
        result.append({"alias": alias, "repo": row["repo"], "revision": row["revision"],
                       "model_path": row["model_path"], "model_manifest": row["model_manifest"]})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/rerank-pilot.json")
    parser.add_argument("--model", choices=list(MODEL_IDS), action="append")
    parser.add_argument("--download", action="store_true", help="Explicitly permit public weights/source downloads; never runs inference")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    rows = plan(config, args.model)
    print(json.dumps({"download": args.download, "models": rows,
                      "laya_code_revision": CODE_REVISION if any(row["alias"] == "laya-multi" for row in rows) else None}, indent=2))
    if not args.download:
        return
    with requests.Session() as session:
        for row in rows:
            model_config = config["models"][row["alias"]]
            prepare_model(row["alias"], model_config, ROOT, session)
            if row["alias"] == "laya-multi":
                prepare_laya_code(model_config, ROOT, session)


if __name__ == "__main__":
    main()
