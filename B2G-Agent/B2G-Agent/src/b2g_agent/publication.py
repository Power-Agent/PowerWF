"""Validate redacted publication bytes against a declared historical hash.

The acquisition hash is never replaced silently. A differing file is accepted
only when its exact original and published hashes are paired in the release
manifest. New runs and unchanged files use ordinary SHA-256 validation.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from b2g_agent.artifact_paths import ROOT, resolve_recorded_path


@lru_cache(maxsize=4)
def _redaction_map(root: Path):
    manifest = root / "publication/redaction_map.json"
    if not manifest.is_file():
        return {}
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0":
        raise ValueError("Unsupported publication hash manifest")
    return payload["artifacts"]


def recorded_hash_matches(path, expected, *, data=None, actual_sha256=None, member=None, repo_root=None):
    """Check acquired or explicitly redacted bytes, including frozen ZIP members."""
    root = Path(repo_root).resolve() if repo_root is not None else ROOT
    path = resolve_recorded_path(path, repo_root=root if repo_root is not None else None)
    if actual_sha256 is None:
        if data is None:
            if member is not None:
                raise ValueError("ZIP member bytes must be supplied")
            data = path.read_bytes()
        actual_sha256 = hashlib.sha256(data).hexdigest()
    if actual_sha256 == expected:
        return True
    try:
        key = path.resolve().relative_to(root).as_posix()
    except ValueError:
        return False
    if member is not None:
        key += "!" + member
    declaration = _redaction_map(root).get(key, {})
    return (declaration.get("original_sha256") == expected
            and declaration.get("published_sha256") == actual_sha256)
