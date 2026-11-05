"""Acquire and prepare local-only DOE prototype assets for the formal case.

The download cache and transitioned IDFs live under `.local-data/`, which is
ignored by git. The committed case directory contains only source URLs,
selectors, schemas, and small reproducibility artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = REPO_ROOT / "cases" / "ieee33_10bem_15min"
DEFAULT_DATA_ROOT = REPO_ROOT / ".local-data"
WEATHER_URL = "https://www.energycodes.gov/sites/default/files/2023-10/USA_NY_Buffalo.Niagara.Intl_.AP_.725280_TMY3.epw"
WEATHER_NAME = "USA_NY_Buffalo.Niagara.Intl_.AP_.725280_TMY3.epw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--energyplus-root", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-transition", action="store_true")
    parser.add_argument("--patch-only", action="store_true")
    parser.add_argument(
        "--only-bem",
        action="append",
        default=[],
        help="prepare only the named BEM while preserving other manifest records; repeatable",
    )
    return parser.parse_args()


def resolve_energyplus_root(explicit: Path | None) -> Path:
    """Resolve a private workstation path without committing it to the repository."""
    if explicit is not None:
        return explicit.expanduser().resolve()
    configured_root = os.getenv("B2G_ENERGYPLUS_ROOT", "").strip()
    if configured_root:
        return Path(configured_root).expanduser().resolve()
    configured_executable = os.getenv("B2G_ENERGYPLUS_EXE", "").strip()
    if configured_executable:
        return Path(configured_executable).expanduser().resolve().parent
    raise RuntimeError(
        "EnergyPlus root is not configured. Pass --energyplus-root or set "
        "B2G_ENERGYPLUS_ROOT/B2G_ENERGYPLUS_EXE in the untracked .env file."
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, *, force: bool) -> None:
    if destination.is_file() and not force:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 B2G-Agent/1.0"})
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    temporary.replace(destination)


def select_idf(archive: Path, required_tokens: list[str]) -> tuple[str, bytes]:
    with zipfile.ZipFile(archive) as bundle:
        candidates = [
            name
            for name in bundle.namelist()
            if name.lower().endswith(".idf")
            and all(token.lower() in name.lower() for token in required_tokens)
        ]
        if len(candidates) != 1:
            buffalo = [
                name
                for name in bundle.namelist()
                if name.lower().endswith(".idf") and "buffalo" in name.lower()
            ]
            if len(buffalo) == 1:
                candidates = buffalo
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected one Buffalo IDF in {archive.name}; candidates={candidates[:20]}"
            )
        return candidates[0], bundle.read(candidates[0])


def patch_runtime_idf(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    timestep = re.compile(r"(?is)(^\s*Timestep\s*,\s*)\d+(\s*;)", re.MULTILINE)
    if timestep.search(text):
        text = timestep.sub(r"\g<1>4\g<2>", text, count=1)
    else:
        text += "\nTimestep,4;\n"
    run_period_weekday = re.compile(
        r"(?is)(\bRunPeriod\s*,.*?\n\s*)"
        r"(Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday)"
        r"(\s*,\s*!-\s*Day of Week for Start Day)"
    )
    text, weekday_replacements = run_period_weekday.subn(
        r"\g<1>Wednesday\g<3>", text
    )
    if weekday_replacements < 1:
        raise ValueError(f"No RunPeriod start weekday was found in {path}")
    additions = []
    if "Output:Meter,Electricity:Facility,Timestep;".lower() not in re.sub(r"\s+", "", text).lower():
        additions.append("Output:Meter,Electricity:Facility,Timestep;")
    if not re.search(
        r"(?is)Output:Variable\s*,[^;]*Zone Mean Air Temperature\s*,\s*Timestep\s*;",
        text,
    ):
        additions.append("Output:Variable,*,Zone Mean Air Temperature,Timestep;")
    if additions:
        text += "\n!- B2G-Agent reproducibility outputs\n" + "\n".join(additions) + "\n"
    path.write_text(text, encoding="utf-8")


def transition_idf(source: Path, destination: Path, energyplus_root: Path) -> list[dict[str, object]]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    # EnergyPlus transition utilities create deterministic siblings and refuse
    # to overwrite stale copies from an earlier regeneration.
    for stale in (
        destination.with_name(destination.name + "old"),
        destination.with_suffix(".VCpErr"),
        destination.with_suffix(destination.suffix + "new"),
        destination.with_suffix(".idfnew"),
    ):
        if stale.is_file():
            stale.unlink()
    shutil.copy2(source, destination)
    updater = energyplus_root / "PreProcess" / "IDFVersionUpdater"
    executables = [
        updater / "Transition-V22-1-0-to-V22-2-0.exe",
        updater / "Transition-V22-2-0-to-V23-1-0.exe",
    ]
    logs: list[dict[str, object]] = []
    for executable in executables:
        if not executable.is_file():
            raise FileNotFoundError(f"Missing EnergyPlus transition executable: {executable}")
        completed = subprocess.run(
            [str(executable), str(destination)],
            # Transition executables resolve their versioned IDD files from the
            # working directory, so they must run inside IDFVersionUpdater.
            cwd=updater,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        logs.append(
            {
                "executable": executable.name,
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-2000:],
                "stderr_tail": completed.stderr[-2000:],
            }
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{executable.name} failed for {destination.name}: "
                f"stdout={completed.stdout[-1000:]!r} stderr={completed.stderr[-1000:]!r}"
            )
        candidates = [
            destination.with_suffix(destination.suffix + "new"),
            destination.with_suffix(".idfnew"),
        ]
        for candidate in candidates:
            if candidate.is_file():
                candidate.replace(destination)
                break
    patch_runtime_idf(destination)
    return logs


def main() -> None:
    load_dotenv(REPO_ROOT / ".env", override=False)
    args = parse_args()
    data_root = (args.data_root or Path(os.getenv("B2G_DATA_ROOT", DEFAULT_DATA_ROOT))).resolve()
    local_case = data_root / "ieee33_10bem_15min"
    downloads = local_case / "downloads"
    if args.patch_only:
        manifest_path = local_case / "local_asset_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Local asset manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        patched = 0
        for record in manifest.get("assets", []):
            runtime_relative = record.get("runtime_path")
            if not runtime_relative:
                continue
            runtime_path = data_root / runtime_relative
            patch_runtime_idf(runtime_path)
            record["runtime_idf_sha256"] = sha256(runtime_path)
            patched += 1
        manifest["patched_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Patched {patched} runtime IDFs; manifest={manifest_path}")
        return
    energyplus_root = resolve_energyplus_root(args.energyplus_root)
    case = json.loads((CASE_DIR / "case.json").read_text(encoding="utf-8"))
    selected_bems = set(args.only_bem) or {bem["bem_id"] for bem in case["bems"]}
    known_bems = {bem["bem_id"] for bem in case["bems"]}
    if not selected_bems <= known_bems:
        raise ValueError(f"Unknown --only-bem values: {sorted(selected_bems - known_bems)}")
    records: list[dict[str, object]] = []
    manifest_path = local_case / "local_asset_manifest.json"
    if args.only_bem and manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = [
            record
            for record in existing.get("assets", [])
            if record.get("asset_id") not in selected_bems
            and record.get("asset_id") != "weather_buffalo_tmy3"
        ]

    weather_path = local_case / "weather" / WEATHER_NAME
    download(WEATHER_URL, weather_path, force=args.force)
    records.append(
        {
            "asset_id": "weather_buffalo_tmy3",
            "source_url": WEATHER_URL,
            "local_path": str(weather_path.relative_to(data_root)).replace("\\", "/"),
            "sha256": sha256(weather_path),
            "bytes": weather_path.stat().st_size,
        }
    )

    for bem in case["bems"]:
        if bem["bem_id"] not in selected_bems:
            continue
        metadata_path = CASE_DIR / "buildings" / bem["bem_id"] / "building.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        url = metadata["source_archive"]
        archive = downloads / Path(url).name
        download(url, archive, force=args.force)
        building_dir = local_case / "buildings" / bem["bem_id"]
        building_dir.mkdir(parents=True, exist_ok=True)
        member, content = select_idf(archive, metadata["source_selector"]["required_tokens"])
        source_idf = building_dir / "source.idf"
        source_idf.write_bytes(content)
        runtime_idf = building_dir / "energyplus_23_1.idf"
        transition_logs: list[dict[str, object]] = []
        replicate_from = metadata.get("replicate_runtime_from")
        if replicate_from:
            replicate_runtime = (
                local_case / "buildings" / str(replicate_from) / "energyplus_23_1.idf"
            )
            if not replicate_runtime.is_file():
                raise FileNotFoundError(
                    f"Replicate runtime source is missing for {bem['bem_id']}: {replicate_runtime}"
                )
            shutil.copy2(replicate_runtime, runtime_idf)
            transition_logs = [
                {
                    "mode": "byte_identical_runtime_replica",
                    "replicate_runtime_from": replicate_from,
                    "source_runtime_sha256": sha256(replicate_runtime),
                }
            ]
        elif args.skip_transition:
            shutil.copy2(source_idf, runtime_idf)
            patch_runtime_idf(runtime_idf)
        else:
            transition_logs = transition_idf(source_idf, runtime_idf, energyplus_root)
        (building_dir / "transition.json").write_text(
            json.dumps(transition_logs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        records.append(
            {
                "asset_id": bem["bem_id"],
                "building_type": bem["building_type"],
                "source_url": url,
                "archive_sha256": sha256(archive),
                "archive_bytes": archive.stat().st_size,
                "archive_member": member,
                "source_idf_sha256": sha256(source_idf),
                "runtime_idf_sha256": sha256(runtime_idf),
                "runtime_path": str(runtime_idf.relative_to(data_root)).replace("\\", "/"),
                "transition_logs": str((building_dir / "transition.json").relative_to(data_root)).replace("\\", "/"),
            }
        )

    manifest = {
        "schema_version": "1.0",
        "case_id": case["case_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "energyplus_root": str(energyplus_root),
        "assets": records,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(records)} assets; manifest={manifest_path}")


if __name__ == "__main__":
    main()
