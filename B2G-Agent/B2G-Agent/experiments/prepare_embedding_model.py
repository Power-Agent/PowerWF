"""Download or verify the frozen local dense-retrieval model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "knowledge" / "retrieval_config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def file_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(root.rglob("*"))
        if (
            path.is_file()
            and path.name != "model_manifest.json"
            and ".cache" not in path.relative_to(root).parts
        )
    }


def main() -> int:
    args = parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    embedding = config["embedding"]
    model_root = (REPO_ROOT / embedding["local_model_path"]).resolve()
    if not args.verify_only:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "Install the retrieval extra before preparing the embedding model"
            ) from exc
        model_root.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=embedding["model_id"],
            revision=embedding["revision"],
            local_dir=model_root,
            allow_patterns=[
                "1_Pooling/*",
                "config.json",
                "config_sentence_transformers.json",
                "model.safetensors",
                "modules.json",
                "sentence_bert_config.json",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.txt",
            ],
        )
    required = {
        "config.json",
        "model.safetensors",
        "modules.json",
        "tokenizer.json",
    }
    missing = sorted(name for name in required if not (model_root / name).is_file())
    if missing:
        raise FileNotFoundError(
            "Frozen embedding model is incomplete: " + ", ".join(missing)
        )
    hashes = file_hashes(model_root)
    bundle = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": "1.0",
        "model_id": embedding["model_id"],
        "revision": embedding["revision"],
        "dimension": embedding["dimension"],
        "file_count": len(hashes),
        "file_hashes": hashes,
        "bundle_sha256": bundle,
    }
    manifest_path = model_root / "model_manifest.json"
    if args.verify_only and manifest_path.is_file():
        expected = json.loads(manifest_path.read_text(encoding="utf-8"))
        if expected != manifest:
            raise ValueError("Frozen embedding model manifest does not match local files")
    else:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "status": "PASS",
                "model_root": str(model_root),
                "bundle_sha256": bundle,
                "file_count": len(hashes),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
