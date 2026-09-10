#!/usr/bin/env python3
"""Copy, sort, and normalize a Minecraft Java resource tree into a perfect dataset folder.

This script:
- Reads a raw resource tree (like your paste.txt listing).
- Copies files into a canonical structure:
    dataset/
      raw/
        <namespace>/
          blockstates/
          models/
            block/
            item/
          textures/
            block/
            item/
            entity/
            gui/
            ...
          lang/
          sounds.json
          pack.mcmeta
          ...
      normalized/
        <namespace>/
          blockstates/
          models/
          textures/
          lang/
          sounds.json
          pack.mcmeta
      index/
        manifest.json
        by_namespace.json
        by_type.json
- Validates paths, normalizes casing, and logs issues.
- Produces a reproducible, training-ready layout.

No third-party dependencies required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

TEXT_EXTENSIONS = {".json", ".mcmeta", ".properties", ".txt", ".md", ".json5"}
ASSET_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".ogg", ".wav", ".obj", ".mtl", ".bbmodel", ".nbt"}
IGNORED_DIRS = {".git", ".gradle", "build", "out", "node_modules", "__pycache__"}
MAX_FILE_BYTES = 50_000_000

VALID_NAMESPACE_PATTERN = re.compile(r"^[a-z0-9._-]+$")
VALID_RESOURCE_LOCATION_PATTERN = re.compile(r"^[a-z0-9._:/-]+$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_copy(src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(str(src), str(dst))
        return True
    except OSError:
        return False


def relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def classify(path: Path) -> str:
    rel = path.as_posix().lower()
    name = path.name.lower()
    if name == "pack.mcmeta":
        return "pack_metadata"
    if "/blockstates/" in rel:
        return "blockstate"
    if "/models/block/" in rel:
        return "block_model"
    if "/models/item/" in rel:
        return "item_model"
    if "/textures/" in rel:
        return "texture"
    if "/lang/" in rel:
        return "language"
    if name == "sounds.json" or "/sounds/" in rel:
        return "sound_definition"
    if "/atlases/" in rel:
        return "atlas"
    if "/font/" in rel:
        return "font"
    if "/particles/" in rel:
        return "particle"
    if "/gui/" in rel:
        return "gui_asset"
    if "/optifine/" in rel:
        return "optifine_asset"
    return "resource_file"


def namespace_from(path: Path) -> str | None:
    parts = path.as_posix().split("/")
    try:
        index = parts.index("assets")
        return parts[index + 1] if len(parts) > index + 1 else None
    except ValueError:
        return None


def is_valid_namespace(ns: str | None) -> bool:
    if not ns:
        return False
    return bool(VALID_NAMESPACE_PATTERN.match(ns))


def is_valid_resource_location(rel: str) -> bool:
    return bool(VALID_RESOURCE_LOCATION_PATTERN.match(rel))


def iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in IGNORED_DIRS for part in path.parts):
            continue
        if path.name.startswith("."):
            continue
        yield path


def build_canonical_path(path: Path, root: Path) -> tuple[str, str, str]:
    """Return (namespace, kind, canonical_subpath) for a file."""
    rel = relative(path, root)
    kind = classify(path)
    ns = namespace_from(path) or "minecraft"
    if not is_valid_namespace(ns):
        ns = "unknown"
    parts = path.as_posix().split("/")
    try:
        assets_index = parts.index("assets")
        remainder = parts[assets_index + 2:]
    except ValueError:
        remainder = parts
    canonical_subpath = "/".join(remainder)
    if not is_valid_resource_location(canonical_subpath):
        canonical_subpath = re.sub(r"[^a-z0-9._:/-]", "_", canonical_subpath.lower())
    return ns, kind, canonical_subpath


def copy_and_normalize(root: Path, output: Path) -> dict[str, Any]:
    raw_dir = output / "raw"
    norm_dir = output / "normalized"
    index_dir = output / "index"

    raw_dir.mkdir(parents=True, exist_ok=True)
    norm_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    by_namespace: dict[str, list[str]] = defaultdict(list)
    by_type: dict[str, list[str]] = defaultdict(list)
    issues: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    copied = 0
    duplicates = 0

    for src in iter_files(root):
        stats["total_files"] += 1
        try:
            if src.stat().st_size > MAX_FILE_BYTES:
                issues.append({"path": relative(src, root), "issue": "file_too_large"})
                stats["skipped_large"] += 1
                continue
        except OSError:
            issues.append({"path": relative(src, root), "issue": "stat_failed"})
            stats["skipped_error"] += 1
            continue

        ns, kind, canonical = build_canonical_path(src, root)
        rel = relative(src, root)
        digest = sha256_file(src)

        if digest in seen_hashes:
            issues.append({"path": rel, "issue": "duplicate_content", "sha256": digest})
            stats["duplicates"] += 1
            duplicates += 1
            continue
        seen_hashes.add(digest)

        safe_name = src.name
        if not re.match(r"^[a-z0-9._-]+\.[a-z0-9]+$", safe_name.lower()):
            safe_name = re.sub(r"[^a-z0-9._-]", "_", safe_name.lower())

        raw_dst = raw_dir / ns / kind / safe_name
        norm_dst = norm_dir / ns
        if kind == "blockstate":
            norm_dst = norm_dst / "blockstates" / safe_name
        elif kind == "block_model":
            norm_dst = norm_dst / "models" / "block" / safe_name
        elif kind == "item_model":
            norm_dst = norm_dst / "models" / "item" / safe_name
        elif kind == "texture":
            norm_dst = norm_dst / "textures" / safe_name
        elif kind == "language":
            norm_dst = norm_dst / "lang" / safe_name
        elif kind == "sound_definition":
            norm_dst = norm_dst / "sounds.json"
        elif kind == "pack_metadata":
            norm_dst = norm_dir / "pack.mcmeta"
        else:
            norm_dst = norm_dst / kind / safe_name

        ok_raw = safe_copy(src, raw_dst)
        ok_norm = safe_copy(src, norm_dst) if raw_dst != norm_dst else True
        if not (ok_raw or ok_norm):
            issues.append({"path": rel, "issue": "copy_failed"})
            stats["copy_failed"] += 1
            continue

        entry = f"{ns}:{canonical}"
        by_namespace[ns].append(entry)
        by_type[kind].append(entry)
        stats[kind] += 1
        copied += 1

    manifest = {
        "schema": "minecraft-dataset-structure-v1",
        "source_root": str(root),
        "output_root": str(output),
        "counts": dict(stats),
        "namespaces": {k: len(v) for k, v in by_namespace.items()},
        "resource_types": {k: len(v) for k, v in by_type.items()},
        "issues_sample": issues[:200],
        "notes": [
            "Files are copied into raw/<namespace>/<type>/<file> and normalized into a standard resource-pack layout.",
            "Duplicate content (by SHA-256) is logged and skipped.",
            "Invalid namespaces and paths are normalized to lowercase with underscores.",
        ],
    }

    (index_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (index_dir / "by_namespace.json").write_text(
        json.dumps(dict(sorted(by_namespace.items())), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (index_dir / "by_type.json").write_text(
        json.dumps(dict(sorted(by_type.items())), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({"copied": copied, "duplicates": duplicates, "stats": dict(stats)}, indent=2))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Root of the raw resource tree")
    parser.add_argument("--output", type=Path, default=Path("minecraft_dataset"))
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"Root directory does not exist: {root}")
    copy_and_normalize(root, args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())