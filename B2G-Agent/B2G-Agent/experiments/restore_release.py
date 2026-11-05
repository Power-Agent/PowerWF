"""Verify and restore downloaded publication archives into this checkout."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def io_path(path):
    """Use extended Windows paths for deeply nested archived evidence."""
    value = str(Path(path).absolute())
    if os.name == "nt" and len(value) >= 240 and not value.startswith("\\\\?\\"):
        value = "\\\\?\\UNC\\" + value[2:] if value.startswith("\\\\") else "\\\\?\\" + value
    return Path(value)


def sha256(path):
    with io_path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def member_destination(root: Path, info):
    normalized = info.filename.replace("\\", "/")
    name = PurePosixPath(normalized)
    if (name.is_absolute() or ".." in name.parts or not name.parts
            or ":" in normalized or name.parts[0] not in {"outputs", ".local-data", "publication"}
            or stat.S_ISLNK(info.external_attr >> 16)):
        raise ValueError(f"Unsafe release member: {info.filename}")
    target = (root / Path(*name.parts)).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Release member escapes project: {info.filename}")
    return target


def restore(archive_dir: Path, root: Path, include="inputs"):
    manifest = json.loads((archive_dir / "release_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "1.0":
        raise ValueError("Unsupported release manifest")
    selected = [row for row in manifest["archives"]
                if include == "all" or row["archive"] in {"b2g-input-data.zip", "b2g-publication.zip"}]
    if not selected:
        raise ValueError("No matching release archives")
    # Verify every selected archive and its member paths before extracting any.
    for row in selected:
        name = row["archive"]
        if Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError("Archive names must not contain paths")
        path = archive_dir / name
        if sha256(path) != row["sha256"]:
            raise ValueError(f"Archive SHA-256 mismatch: {name}")
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                member_destination(root, member)
        print(f"Verified {name}", flush=True)
    files = 0
    for row in selected:
        with zipfile.ZipFile(archive_dir / row["archive"]) as archive:
            for member in archive.infolist():
                target = io_path(member_destination(root, member))
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if target.exists():
                    with archive.open(member) as stream:
                        expected = hashlib.file_digest(stream, "sha256").hexdigest()
                    if sha256(target) != expected:
                        raise FileExistsError(f"Refusing to overwrite a different existing file: {member.filename}")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("xb") as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
                files += 1
        print(f"Restored {row['archive']}", flush=True)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--include", choices=("inputs", "all"), default="inputs")
    args = parser.parse_args()
    files = restore(args.archive_dir.resolve(), ROOT, args.include)
    print(json.dumps({"files_restored": files, "include": args.include, "api_calls": 0, "physics_runs": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
