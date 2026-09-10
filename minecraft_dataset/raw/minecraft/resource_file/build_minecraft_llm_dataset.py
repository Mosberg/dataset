#!/usr/bin/env python3
"""Build a validated, deduplicated LLM dataset from a Minecraft Java resource-pack tree.

Outputs JSONL in conversational format, split into train/validation/test, plus a manifest.
No third-party dependencies are required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

TEXT_EXTENSIONS = {".json", ".mcmeta", ".properties", ".txt", ".md", ".json5"}
ASSET_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".ogg", ".wav", ".obj", ".mtl", ".bbmodel", ".nbt"}
IGNORED_DIRS = {".git", ".gradle", "build", "out", "node_modules", "__pycache__"}
MAX_FILE_BYTES = 2_000_000

SYSTEM_PROMPT = """You are an expert Minecraft Java Edition resource-pack engineer.
Return technically accurate, copy-ready files and explain assumptions briefly.
Respect namespace paths, JSON syntax, model parent references, texture paths,
blockstate variants, animation .mcmeta files, pack format compatibility, and
modded namespaces. Never invent a file's contents when the provided context is
insufficient; request the missing file or state what is unknown."""


def json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_read(path: Path) -> str | None:
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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


def parse_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def compact_json_context(value: Any) -> str:
    rendered = json.dumps(value, ensure_ascii=False, indent=2)
    return rendered[:12000]


def references(value: Any) -> list[str]:
    found: list[str] = []
    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if key in {"parent", "model", "texture", "textures", "particle", "sound", "sounds"}:
                    if isinstance(child, str):
                        found.append(child)
                    elif isinstance(child, list):
                        found.extend(str(x) for x in child if isinstance(x, str))
                    elif isinstance(child, dict):
                        found.extend(str(x) for x in child.values() if isinstance(x, str))
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)
    walk(value)
    return sorted(set(found))[:100]


def user_request(kind: str, rel: str, namespace: str | None, data: Any | None, raw: str) -> str:
    context = compact_json_context(data) if data is not None else raw[:12000]
    prompts = {
        "pack_metadata": f"Analyze this Minecraft Java Edition pack.mcmeta file at `{rel}`. Explain its compatibility metadata, description, and any schema or version concerns.\n\n```json\n{context}\n```",
        "blockstate": f"Analyze the blockstate file `{rel}` for namespace `{namespace or 'unknown'}`. Explain every variant/multipart rule, model reference, rotation, uvlock setting, and likely missing-reference or syntax issue.\n\n```json\n{context}\n```",
        "block_model": f"Analyze the Minecraft Java block model `{rel}`. Explain parent inheritance, texture variables, elements, faces, rotations, cullface, and display data. Identify references that should be checked.\n\n```json\n{context}\n```",
        "item_model": f"Analyze the Minecraft Java item model `{rel}`. Explain its parent, texture mapping, transforms, overrides, and compatibility concerns.\n\n```json\n{context}\n```",
        "language": f"Analyze this Minecraft language file `{rel}`. Explain key naming patterns, translation coverage risks, and how to add or correct entries without breaking JSON.\n\n```json\n{context}\n```",
        "sound_definition": f"Analyze this Minecraft sound definition `{rel}`. Explain event names, sound paths, volume, pitch, attenuation, streaming, and validation concerns.\n\n```json\n{context}\n```",
    }
    return prompts.get(kind, f"Analyze the Minecraft Java resource `{rel}` ({namespace or 'unknown namespace'}). Explain its purpose, structure, references, and validation risks.\n\n```text\n{context}\n```")


def assistant_answer(kind: str, rel: str, namespace: str | None, data: Any | None, raw: str) -> str:
    refs = references(data) if data is not None else []
    lines = [f"Resource type: {kind}.", f"Path: `{rel}`."]
    if namespace:
        lines.append(f"Namespace: `{namespace}`.")
    if refs:
        lines.append("Referenced identifiers detected: " + ", ".join(f"`{x}`" for x in refs) + ".")
    if data is not None:
        lines.append("The file is valid JSON and was normalized for analysis without changing its meaning.")
        if isinstance(data, dict):
            lines.append("Top-level keys: " + ", ".join(f"`{x}`" for x in data.keys()) + ".")
    else:
        lines.append("This is a text/binary-adjacent resource whose full semantics depend on the referenced asset and Minecraft/mod version.")
    lines.append("Validate referenced paths against the same namespace and preserve lowercase resource-location conventions.")
    return "\n".join(lines)


def make_example(path: Path, root: Path) -> dict[str, Any] | None:
    rel = relative(path, root)
    kind = classify(path)
    namespace = namespace_from(path)
    suffix = path.suffix.lower()
    raw = safe_read(path) if suffix in TEXT_EXTENSIONS else None
    data = parse_json(raw) if raw is not None and suffix in {".json", ".mcmeta", ".json5"} else None
    if raw is None and suffix not in ASSET_EXTENSIONS:
        return None
    if raw is not None and not normalize_text(raw):
        return None
    digest = sha256_bytes(path.read_bytes())
    prompt = user_request(kind, rel, namespace, data, normalize_text(raw or ""))
    answer = assistant_answer(kind, rel, namespace, data, normalize_text(raw or ""))
    return {
        "id": f"minecraft-resource-{digest[:16]}",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {
            "path": rel,
            "resource_type": kind,
            "namespace": namespace,
            "extension": path.suffix.lower(),
            "sha256": digest,
            "source": "local_resource_pack",
        },
    }


def iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in IGNORED_DIRS for part in path.parts):
            continue
        if path.name.startswith("."):
            continue
        yield path


def split_examples(examples: list[dict[str, Any]], seed: int, validation: float, test: float) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in examples:
        meta = item["metadata"]
        group = f"{meta.get('namespace') or 'root'}:{meta['resource_type']}"
        groups[group].append(item)
    result = {"train": [], "validation": [], "test": []}
    for group_items in groups.values():
        rng.shuffle(group_items)
        n = len(group_items)
        test_n = min(max(0, int(round(n * test))), max(0, n - 1))
        val_n = min(max(0, int(round(n * validation))), max(0, n - test_n - 1))
        result["test"].extend(group_items[:test_n])
        result["validation"].extend(group_items[test_n:test_n + val_n])
        result["train"].extend(group_items[test_n + val_n:])
    for values in result.values():
        values.sort(key=lambda x: x["id"])
    return result


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json_dump(row) + "\n")
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Resource-pack or extracted assets root")
    parser.add_argument("--output", type=Path, default=Path("minecraft_llm_dataset"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation", type=float, default=0.05)
    parser.add_argument("--test", type=float, default=0.05)
    parser.add_argument("--max-examples", type=int, default=0, help="0 means unlimited")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"Root directory does not exist: {root}")
    if args.validation < 0 or args.test < 0 or args.validation + args.test >= 1:
        parser.error("validation + test must be between 0 and 1")

    examples: list[dict[str, Any]] = []
    seen: set[str] = set()
    skipped = Counter()
    for path in iter_files(root):
        try:
            item = make_example(path, root)
        except (OSError, UnicodeError) as exc:
            skipped[type(exc).__name__] += 1
            continue
        if item is None:
            skipped["unsupported_or_empty"] += 1
            continue
        if item["metadata"]["sha256"] in seen:
            skipped["duplicate_content"] += 1
            continue
        seen.add(item["metadata"]["sha256"])
        examples.append(item)
        if args.max_examples and len(examples) >= args.max_examples:
            break

    splits = split_examples(examples, args.seed, args.validation, args.test)
    args.output.mkdir(parents=True, exist_ok=True)
    counts = {name: write_jsonl(args.output / f"{name}.jsonl", rows) for name, rows in splits.items()}
    type_counts = Counter(item["metadata"]["resource_type"] for item in examples)
    namespace_counts = Counter(item["metadata"].get("namespace") or "root" for item in examples)
    manifest = {
        "schema": "minecraft-java-resource-llm-v1",
        "format": "conversational-jsonl",
        "source_root": str(root),
        "system_prompt": SYSTEM_PROMPT,
        "seed": args.seed,
        "split_ratios": {"validation": args.validation, "test": args.test},
        "counts": counts,
        "total": len(examples),
        "resource_types": dict(sorted(type_counts.items())),
        "namespaces": dict(sorted(namespace_counts.items())),
        "skipped": dict(skipped),
        "quality_notes": [
            "Examples are deterministic and content-hash identified.",
            "JSON files are parsed when valid; malformed JSON is retained as text context for diagnosis.",
            "Binary assets receive metadata-level examples because pixel/audio semantics require a vision/audio encoder or captions.",
            "Generated answers describe structure and references; they are not a substitute for verified game-version documentation.",
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "counts": counts, "total": len(examples)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
