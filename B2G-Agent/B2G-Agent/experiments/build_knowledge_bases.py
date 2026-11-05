from __future__ import annotations

"""Acquire public source slices and build the three Translator knowledge bases.

Raw upstream assets remain under the ignored local data root.  Only normalized,
source-linked records and their content lock are written under ``knowledge/``.
The extractor never reads the formal case, its tasks, or its ground truth.
"""

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import shutil
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_ROOT = REPO_ROOT / "knowledge"
REGISTRY_PATH = KNOWLEDGE_ROOT / "sources" / "source_registry.json"
LOCK_PATH = KNOWLEDGE_ROOT / "sources" / "asset_lock.json"
DEFAULT_DATA_ROOT = REPO_ROOT / ".local-data"
EXTRACTOR_VERSION = "case-derived-kb-extractor-v2"
DOMAINS = ("building", "grid", "cross_domain")
TARGET_RECORDS_PER_DOMAIN = 180
FORBIDDEN_OUTPUT_PATTERNS = (
    re.compile(r"\bep-\d{3}\b", re.IGNORECASE),
    re.compile(r"\bbem(?:_|[- ]?)\d{2}\b", re.IGNORECASE),
    re.compile(r"contract_oracles|numeric_oracles|groundtruth", re.IGNORECASE),
)


@dataclass(frozen=True)
class Asset:
    source_id: str
    asset_id: str
    url: str
    upstream_locator: str
    revision: str
    local_path: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def slug(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return text or "unknown"


def unique_text(values: Iterable[object], *, limit: int | None = None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = re.sub(r"\s+", " ", str(value)).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if limit is not None and len(result) >= limit:
            break
    return result


def record(
    *,
    record_id: str,
    domain: str,
    record_type: str,
    title: str,
    summary: str,
    source_refs: list[dict[str, str]],
    keywords: Iterable[object] = (),
    entities: Iterable[object] = (),
    relations: Iterable[object] = (),
    rules: Iterable[object] = (),
    applicability: Iterable[object] = (),
    limitations: Iterable[object] = (),
    units: Iterable[object] = (),
    time_scales: Iterable[object] = (),
    spatial_levels: Iterable[object] = (),
    applicable_roles: Iterable[object] = (),
    business_types: Iterable[object] = (),
    directions: Iterable[object] = (),
    review_status: str = "source_derived",
) -> dict[str, object]:
    if domain not in DOMAINS:
        raise ValueError(f"Unsupported domain: {domain}")
    searchable = " ".join(
        [title, summary, *map(str, keywords), *map(str, entities), *map(str, rules)]
    ).lower()
    inferred_units = list(units) or _infer_metadata(
        searchable,
        (("kW", r"\bkw\b|kilowatt"), ("MW", r"\bmw\b|megawatt"),
         ("kWh", r"\bkwh\b"), ("kvar", r"\bkvar\b"), ("Mvar", r"\bmvar\b"),
         ("p.u.", r"p\.?\s*u\.?|per.unit"), ("%", r"percent|%"),
         ("USD/MWh", r"usd|price|settlement")),
        "unspecified",
    )
    inferred_time_scales = list(time_scales) or _infer_metadata(
        searchable,
        (("sub-minute", r"second|real.time|helics"),
         ("interval", r"interval|time.series|profile|schedule"),
         ("event", r"event|active.period|demand.response"),
         ("annual", r"annual|year")),
        "general",
    )
    inferred_spatial = list(spatial_levels) or _infer_metadata(
        searchable,
        (("device", r"device|battery|ev|thermostat|equipment"),
         ("building", r"building|zone|meter"),
         ("portfolio", r"district|portfolio|aggregate"),
         ("bus", r"\bbus\b|\bnode\b|connection.point"),
         ("feeder", r"feeder|line|transformer|substation|network")),
        "general",
    )
    inferred_roles = list(applicable_roles) or (
        ["building_energy_manager"] if domain == "building" else
        ["distribution_grid_operator"] if domain == "grid" else
        ["building_energy_manager", "distribution_grid_operator"]
    )
    inferred_business = list(business_types) or _infer_metadata(
        searchable,
        (("peak_shaving", r"peak|load.shedding"),
         ("congestion_management", r"congestion|overload|line.loading"),
         ("voltage_management", r"voltage|\bvm\b"),
         ("reserve_service", r"reserve|capacity"),
         ("renewable_absorption", r"renewable|solar|pv|curtail"),
         ("baseline_flexibility", r"baseline|flexibil"),
         ("market_trading", r"market|price|settlement|offer")),
        "general_coordination",
    )
    inferred_directions = list(directions) or (
        ["building_to_grid"] if domain == "building" else
        ["grid_to_building"] if domain == "grid" else
        ["building_to_grid", "grid_to_building"]
    )
    payload: dict[str, object] = {
        "record_id": slug(record_id),
        "domain": domain,
        "record_type": record_type,
        "title": re.sub(r"\s+", " ", title).strip(),
        "summary": re.sub(r"\s+", " ", summary).strip(),
        "keywords": sorted(unique_text(keywords), key=str.casefold),
        "entities": sorted(unique_text(entities), key=str.casefold),
        "relations": sorted(unique_text(relations), key=str.casefold),
        "rules": unique_text(rules),
        "applicability": unique_text(applicability),
        "limitations": unique_text(limitations),
        "units": sorted(unique_text(inferred_units), key=str.casefold),
        "time_scales": sorted(unique_text(inferred_time_scales), key=str.casefold),
        "spatial_levels": sorted(unique_text(inferred_spatial), key=str.casefold),
        "applicable_roles": sorted(unique_text(inferred_roles), key=str.casefold),
        "business_types": sorted(unique_text(inferred_business), key=str.casefold),
        "directions": sorted(unique_text(inferred_directions), key=str.casefold),
        "review_status": review_status,
        "source_refs": sorted(source_refs, key=lambda item: (item["source_id"], item["asset_id"])),
    }
    if len(payload["summary"]) < 10:  # type: ignore[arg-type]
        raise ValueError(f"Summary is too short for {record_id}")
    return payload


def _infer_metadata(
    text: str,
    patterns: Iterable[tuple[str, str]],
    fallback: str,
) -> list[str]:
    values = [label for label, pattern in patterns if re.search(pattern, text, re.IGNORECASE)]
    return values or [fallback]


def load_registry() -> dict[str, object]:
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    source_ids = [item["source_id"] for item in payload["sources"]]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Duplicate source_id in knowledge source registry")
    return payload


def build_asset_catalog(registry: dict[str, object], raw_root: Path) -> dict[str, Asset]:
    assets: dict[str, Asset] = {}
    for source in registry["sources"]:  # type: ignore[index]
        source_id = source["source_id"]
        revision = source["revision"]
        landing_page = source["landing_page"]
        github_match = re.fullmatch(r"https://github\.com/([^/]+/[^/]+)", landing_page)
        for item in source.get("assets", []):
            asset_id = item["asset_id"]
            if asset_id in assets:
                raise ValueError(f"Duplicate asset_id: {asset_id}")
            upstream_locator = item.get("path", item.get("url"))
            if "url" in item:
                url = item["url"]
            elif github_match:
                url = (
                    "https://raw.githubusercontent.com/"
                    f"{github_match.group(1)}/{revision}/"
                    f"{urllib.parse.quote(item['path'], safe='/')}"
                )
            else:
                raise ValueError(f"No download URL can be formed for {asset_id}")
            suffixes = "".join(Path(urllib.parse.urlparse(url).path).suffixes)
            suffix = suffixes[-12:] if suffixes else ".data"
            local_path = raw_root / source_id / f"{asset_id}{suffix}"
            assets[asset_id] = Asset(
                source_id=source_id,
                asset_id=asset_id,
                url=url,
                upstream_locator=upstream_locator,
                revision=revision,
                local_path=local_path,
            )
    return assets


def download_asset(asset: Asset, *, force: bool) -> None:
    if asset.local_path.is_file() and not force:
        return
    asset.local_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = asset.local_path.with_suffix(asset.local_path.suffix + ".part")
    request = urllib.request.Request(
        asset.url,
        headers={"User-Agent": "B2G-Agent-knowledge-builder/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        temporary.replace(asset.local_path)
    finally:
        temporary.unlink(missing_ok=True)


def acquire(
    assets: dict[str, Asset], *, force: bool, offline: bool, update_lock: bool
) -> dict[str, dict[str, object]]:
    existing_lock = (
        json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        if LOCK_PATH.is_file()
        else {"assets": {}}
    )
    locked_assets: dict[str, dict[str, object]] = existing_lock.get("assets", {})
    observed: dict[str, dict[str, object]] = {}
    for asset_id, asset in sorted(assets.items()):
        if not offline:
            download_asset(asset, force=force)
        if not asset.local_path.is_file():
            raise FileNotFoundError(
                f"Missing local source asset {asset_id}: {asset.local_path}. "
                "Run without --offline to acquire it."
            )
        digest = sha256_file(asset.local_path)
        prior = locked_assets.get(asset_id)
        if prior and not update_lock and prior["sha256"] != digest:
            raise ValueError(
                f"Locked hash mismatch for {asset_id}: expected {prior['sha256']}, observed {digest}"
            )
        observed[asset_id] = {
            "source_id": asset.source_id,
            "url": asset.url,
            "revision": asset.revision,
            "upstream_locator": asset.upstream_locator,
            "sha256": digest,
            "bytes": asset.local_path.stat().st_size,
        }
    if not LOCK_PATH.is_file() or update_lock:
        LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOCK_PATH.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "extractor_version": EXTRACTOR_VERSION,
                    "assets": observed,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return observed


def source_ref(asset: Asset, lock: dict[str, dict[str, object]]) -> dict[str, str]:
    return {
        "source_id": asset.source_id,
        "asset_id": asset.asset_id,
        "upstream_locator": asset.upstream_locator,
        "sha256": str(lock[asset.asset_id]["sha256"]),
    }


def read_json(asset: Asset) -> object:
    return json.loads(asset.local_path.read_text(encoding="utf-8-sig"))


def read_csv(asset: Asset) -> tuple[list[str], list[dict[str, str]]]:
    text = asset.local_path.read_text(encoding="utf-8-sig", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(text.splitlines(), dialect=dialect)
    return list(reader.fieldnames or []), [dict(row) for row in reader]


def flatten_scalars(payload: object, prefix: str = "") -> list[tuple[str, object]]:
    flattened: list[tuple[str, object]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.extend(flatten_scalars(value, path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            flattened.extend(flatten_scalars(value, f"{prefix}[{index}]"))
    elif payload is not None:
        flattened.append((prefix, payload))
    return flattened


def extract_tesp(
    assets: dict[str, Asset], lock: dict[str, dict[str, object]]
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for asset_id in sorted(key for key in assets if key.startswith("tesp-") and key != "tesp-energyplus-readme"):
        asset = assets[asset_id]
        payload = read_json(asset)
        if not isinstance(payload, dict):
            continue
        name = str(payload.get("name") or payload.get("coreName") or asset_id)
        publications = payload.get("publications", [])
        subscriptions = payload.get("subscriptions", [])
        endpoints = payload.get("endpoints", [])
        pub_keys = [str(item.get("key") or item.get("name")) for item in publications if isinstance(item, dict)]
        sub_keys = [str(item.get("key") or item.get("target") or item.get("name")) for item in subscriptions if isinstance(item, dict)]
        endpoint_keys = [str(item.get("name") or item.get("target")) for item in endpoints if isinstance(item, dict)]
        refs = [source_ref(asset, lock)]
        records.append(
            record(
                record_id=f"tesp-federate-{asset_id}",
                domain="cross_domain",
                record_type="co_simulation_case",
                title=f"TESP federate interface: {name}",
                summary=(
                    f"The pinned TESP case configures {name} with {len(pub_keys)} publications, "
                    f"{len(sub_keys)} subscriptions, and {len(endpoint_keys)} endpoints. "
                    "The declared names form an executable building-grid message interface."
                ),
                keywords=["TESP", "HELICS", "publication", "subscription", "co-simulation", *pub_keys, *sub_keys],
                entities=[name, *pub_keys, *sub_keys, *endpoint_keys],
                relations=[*(f"publishes:{key}" for key in pub_keys), *(f"subscribes:{key}" for key in sub_keys)],
                rules=[
                    "A publication is an output produced by one federate; a subscription is an input consumed from another federate.",
                    "Message names and types must be validated against the pinned HELICS configuration before execution.",
                ],
                applicability=["EnergyPlus-agent-grid co-simulation", "load shedding", "multi-federate scheduling"],
                limitations=["This record describes an interface pattern, not a physical result or an episode answer."],
                source_refs=refs,
            )
        )
        for direction, keys in (("publication", pub_keys), ("subscription", sub_keys)):
            for key in unique_text(keys, limit=40):
                records.append(
                    record(
                        record_id=f"tesp-{direction}-{asset_id}-{key}",
                        domain="cross_domain",
                        record_type="message_field",
                        title=f"TESP {direction}: {key}",
                        summary=f"The {name} federate declares {key} as a {direction} in a pinned public co-simulation case.",
                        keywords=["TESP", "HELICS", direction, key],
                        entities=[name, key],
                        relations=[f"{name}:{direction}:{key}"],
                        rules=["Resolve the publisher, subscriber, datatype, and time basis before routing this field."],
                        applicability=["cross-domain message translation"],
                        limitations=["The config proves field use but does not establish universal sign or unit conventions by itself."],
                        source_refs=refs,
                    )
                )
    return records


def extract_urbanopt(
    assets: dict[str, Asset], lock: dict[str, dict[str, object]]
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for asset_id in ("urbanopt-electric-network", "urbanopt-pv-project"):
        asset = assets[asset_id]
        payload = read_json(asset)
        features = payload.get("features", []) if isinstance(payload, dict) else []
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for feature in features:
            if not isinstance(feature, dict):
                continue
            properties = feature.get("properties", {})
            feature_type = str(properties.get("type") or feature.get("type") or "unknown")
            grouped[feature_type].append(properties)
        for feature_type, rows in sorted(grouped.items()):
            property_names = unique_text(key for row in rows for key in row.keys())
            ids = unique_text(((row.get("id") or row.get("name")) for row in rows), limit=12)
            domain = "grid" if re.search(r"electric|substation|feeder|junction|transformer|line", feature_type, re.I) else "building"
            records.append(
                record(
                    record_id=f"urbanopt-{asset_id}-{feature_type}",
                    domain=domain,
                    record_type="spatial_case_schema",
                    title=f"URBANopt feature type: {feature_type}",
                    summary=(
                        f"The pinned URBANopt GeoJSON contains {len(rows)} {feature_type} features. "
                        f"Their declared properties include {', '.join(property_names[:18])}."
                    ),
                    keywords=["URBANopt", "GeoJSON", feature_type, *property_names],
                    entities=[feature_type, *ids],
                    relations=["feature_has_properties", "geometry_locates_feature"],
                    rules=["Preserve feature IDs when joining building results to electrical-network objects."],
                    applicability=["district building models", "spatial building-grid mapping"],
                    limitations=["Feature presence does not prove that a simulation was executed or feasible."],
                    source_refs=[source_ref(asset, lock)],
                )
            )
    for asset_id in (
        "urbanopt-baseline-scenario",
        "urbanopt-thermostat-shedding",
        "urbanopt-mels-shedding",
    ):
        asset = assets[asset_id]
        fields, rows = read_csv(asset)
        values = unique_text((value for row in rows for value in row.values()), limit=25)
        records.append(
            record(
                record_id=f"urbanopt-scenario-{asset_id}",
                domain="building",
                record_type="district_control_case",
                title=f"URBANopt scenario: {asset_id.removeprefix('urbanopt-').replace('-', ' ')}",
                summary=f"This public URBANopt scenario has {len(rows)} rows and fields {', '.join(fields)}; it demonstrates a district-level control configuration rather than an episode answer.",
                keywords=["URBANopt", "scenario", *fields, *values],
                entities=values,
                relations=["scenario_applies_measure_to_feature"],
                rules=["A scenario row links a feature identifier to a named modeling or control measure."],
                applicability=["baseline", "thermostat adjustment", "miscellaneous electric load shedding"],
                limitations=["Scenario configuration alone does not quantify delivered grid service."],
                source_refs=[source_ref(asset, lock)],
            )
        )
    return records


def extract_boptest(
    assets: dict[str, Asset], lock: dict[str, dict[str, object]]
) -> list[dict[str, object]]:
    groups: dict[str, dict[str, Asset]] = defaultdict(dict)
    for asset_id, asset in assets.items():
        match = re.match(r"boptest-(.+)-(config|days|kpis)$", asset_id)
        if match:
            groups[match.group(1)][match.group(2)] = asset
    records: list[dict[str, object]] = []
    for case_name, parts in sorted(groups.items()):
        config = read_json(parts["config"])
        days = read_json(parts["days"])
        kpis = read_json(parts["kpis"])
        config_fields = unique_text((path for path, _ in flatten_scalars(config)), limit=30)
        day_names = list(days) if isinstance(days, dict) else []
        kpi_fields = unique_text((path for path, _ in flatten_scalars(kpis)), limit=40)
        refs = [source_ref(parts[key], lock) for key in ("config", "days", "kpis")]
        records.append(
            record(
                record_id=f"boptest-case-{case_name}",
                domain="building",
                record_type="building_control_benchmark",
                title=f"BOPTEST case: {case_name.replace('-', ' ')}",
                summary=(
                    f"The public BOPTEST {case_name} case declares configuration fields, {len(day_names)} named day selections, "
                    f"and KPI mappings including {', '.join(kpi_fields[:12])}."
                ),
                keywords=["BOPTEST", "building control", case_name, *config_fields, *kpi_fields],
                entities=[case_name, *day_names],
                relations=["control_input_affects_building_state", "measurement_supports_kpi"],
                rules=[
                    "Control actions must be assessed with measured outputs and declared KPI mappings.",
                    "Comfort, energy, cost, emissions, and peak metrics are distinct evidence channels when present.",
                ],
                applicability=["building control evaluation", "comfort-constrained flexibility"],
                limitations=["BOPTEST is building-side evidence and does not by itself validate feeder voltage or congestion."],
                source_refs=refs,
            )
        )
    return records


def extract_citylearn(
    assets: dict[str, Asset], lock: dict[str, dict[str, object]]
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    definition_asset = assets["citylearn-schema-definition"]
    fields, rows = read_csv(definition_asset)
    for index, row in enumerate(rows):
        values = unique_text(row.values())
        if not values:
            continue
        title = values[0]
        records.append(
            record(
                record_id=f"citylearn-schema-{index:03d}-{title}",
                domain="building",
                record_type="schema_field",
                title=f"CityLearn schema field: {title}",
                summary=f"The CityLearn schema table documents {title} with attributes: {'; '.join(values[1:8]) or ', '.join(fields)}.",
                keywords=["CityLearn", *fields, *values],
                entities=[title],
                relations=["schema_field_describes_building_or_device"],
                rules=["Interpret the field using its declared datatype, unit, observation/action role, and active status."],
                applicability=["multi-building coordination", "demand response", "storage and EV control"],
                limitations=["A schema field is not a measured value and cannot serve as physical outcome evidence."],
                source_refs=[source_ref(definition_asset, lock)],
            )
        )
    for asset_id in ("citylearn-ev-schema", "citylearn-three-phase-schema"):
        asset = assets[asset_id]
        payload = read_json(asset)
        flattened = flatten_scalars(payload)
        paths = unique_text((path for path, _ in flattened), limit=120)
        values = unique_text((value for _, value in flattened if isinstance(value, str)), limit=80)
        records.append(
            record(
                record_id=f"citylearn-case-{asset_id}",
                domain="cross_domain" if "three-phase" in asset_id else "building",
                record_type="coordination_case_schema",
                title=f"CityLearn dataset schema: {asset_id.removeprefix('citylearn-').replace('-', ' ')}",
                summary=f"The pinned CityLearn schema exposes {len(flattened)} scalar declarations across buildings, devices, observations, actions, and evaluation settings.",
                keywords=["CityLearn", "observation", "action", "building", "electric service", *paths, *values],
                entities=values,
                relations=["observation_informs_action", "building_contains_energy_system"],
                rules=["Only active observations and actions declared in the selected schema may be used by a controller."],
                applicability=["community energy coordination", "EV and storage flexibility", "electrical service representation"],
                limitations=["Dataset schema semantics do not replace independent AC power-flow validation."],
                source_refs=[source_ref(asset, lock)],
            )
        )
    return records


def extract_ochre(
    assets: dict[str, Asset], lock: dict[str, dict[str, object]]
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    variable_asset = assets["ochre-variable-units"]
    headers, rows = read_csv(variable_asset)
    for index, row in enumerate(rows):
        values = unique_text(row.values())
        searchable = " ".join(values)
        if not re.search(r"power|energy|temperature|setpoint|state of charge|voltage|current|reactive|electric", searchable, re.I):
            continue
        name = row.get("OCHRE Name") or (values[0] if values else f"variable-{index}")
        unit = row.get("OCHRE Units") or "unit not declared"
        description = row.get("Description") or "building or equipment output"
        records.append(
            record(
                record_id=f"ochre-variable-{index:03d}-{name}",
                domain="building",
                record_type="building_output_variable",
                title=f"OCHRE variable: {name}",
                summary=f"OCHRE declares {name} in {unit}: {description}. Cross-simulator aliases remain attached as metadata.",
                keywords=["OCHRE", *headers, *values],
                entities=[name],
                relations=["equipment_exposes_variable"],
                rules=["Preserve the upstream variable unit and distinguish state, rate, and cumulative-energy quantities."],
                applicability=["residential building simulation", "device flexibility"],
                limitations=["Variable metadata does not establish a feasible control range for a particular dwelling."],
                source_refs=[source_ref(variable_asset, lock)],
            )
        )
    for asset_id in (
        "ochre-battery-parameters",
        "ochre-ev-profiles",
        "ochre-water-heater-parameters",
        "ochre-building-template",
    ):
        asset = assets[asset_id]
        fields, rows = read_csv(asset)
        records.append(
            record(
                record_id=f"ochre-component-{asset_id}",
                domain="building",
                record_type="device_case_schema",
                title=f"OCHRE component data: {asset_id.removeprefix('ochre-').replace('-', ' ')}",
                summary=f"This pinned OCHRE component table contains {len(rows)} rows with fields {', '.join(fields)}.",
                keywords=["OCHRE", *fields, asset_id],
                entities=[asset_id.removeprefix("ochre-")],
                relations=["component_parameterizes_building_model"],
                rules=["Device availability, capacity, efficiency, and state constraints must be retained during aggregation."],
                applicability=["battery", "electric vehicle", "water heating", "residential model assembly"],
                limitations=["Default parameters are priors and must not be presented as calibrated site values."],
                source_refs=[source_ref(asset, lock)],
            )
        )
    return records


def extract_brick(
    assets: dict[str, Asset], lock: dict[str, dict[str, object]]
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    definitions = assets["brick-definitions"]
    fields, rows = read_csv(definitions)
    for index, row in enumerate(rows):
        values = unique_text(row.values())
        searchable = " ".join(values)
        if not re.search(r"meter|power|energy|temperature|setpoint|battery|solar|electric|HVAC|load|thermal|zone|equipment", searchable, re.I):
            continue
        term = values[0] if values else f"term-{index}"
        records.append(
            record(
                record_id=f"brick-concept-{index:04d}-{term}",
                domain="building",
                record_type="ontology_concept",
                title=f"Brick concept: {term}",
                summary=f"Brick defines {term} with ontology metadata {'; '.join(values[1:6]) or ', '.join(fields)}.",
                keywords=["Brick", "ontology", *fields, *values],
                entities=[term],
                relations=["concept_has_definition"],
                rules=["Use Brick identifiers as canonical building-side concepts and retain source-system aliases separately."],
                applicability=["building metadata normalization", "sensor and equipment identification"],
                limitations=["Ontology class membership does not supply a live value, unit conversion, or control authority."],
                source_refs=[source_ref(definitions, lock)],
            )
        )
    for asset_id in ("brick-building-meter", "brick-solar-array", "brick-evse", "brick-submeter"):
        asset = assets[asset_id]
        text = asset.local_path.read_text(encoding="utf-8", errors="replace")
        classes = unique_text(re.findall(r"\bbrick:([A-Za-z0-9_]+)", text), limit=50)
        relations = unique_text(re.findall(r"\bbrick:(has[A-Z][A-Za-z0-9_]+|is[A-Z][A-Za-z0-9_]+|feeds)\b", text), limit=30)
        records.append(
            record(
                record_id=f"brick-example-{asset_id}",
                domain="cross_domain" if "meter" in asset_id or "solar" in asset_id or "evse" in asset_id else "building",
                record_type="ontology_example",
                title=f"Brick example: {asset_id.removeprefix('brick-').replace('-', ' ')}",
                summary=f"The public Brick example instantiates {', '.join(classes[:16])} and relations {', '.join(relations[:12]) or 'declared RDF relationships'}.",
                keywords=["Brick", "RDF", *classes, *relations],
                entities=classes,
                relations=relations,
                rules=["Preserve meter/submeter, equipment/location, and source/load relationships when mapping telemetry."],
                applicability=["meter hierarchy", "DER identification", "building point mapping"],
                limitations=["Example topology is illustrative and must not be copied as a study network."],
                source_refs=[source_ref(asset, lock)],
            )
        )
    return records


def ttl_concepts(text: str) -> list[tuple[str, str, str]]:
    concepts: list[tuple[str, str, str]] = []
    blocks = re.split(r"\n(?=[A-Za-z][\w-]*:[\w-]+\s)", text)
    for block in blocks:
        if "owl:Class" not in block and "rdf:Property" not in block and "owl:ObjectProperty" not in block:
            continue
        subject_match = re.match(r"\s*([A-Za-z][\w-]*:[\w-]+)", block)
        label_match = re.search(r"rdfs:label\s+\"([^\"]+)\"", block)
        comment_match = re.search(r"rdfs:comment\s+\"\"\"(.*?)\"\"\"|rdfs:comment\s+\"([^\"]+)\"", block, re.S)
        if not subject_match:
            continue
        subject = subject_match.group(1)
        label = label_match.group(1) if label_match else subject.split(":", 1)[-1]
        comment = next((group for group in (comment_match.groups() if comment_match else ()) if group), "")
        concepts.append((subject, label, re.sub(r"\s+", " ", comment).strip()))
    return concepts


def extract_saref(
    assets: dict[str, Asset], lock: dict[str, dict[str, object]]
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for asset_id, domain in (
        ("saref4bldg-ontology", "building"),
        ("saref4grid-ontology", "grid"),
    ):
        asset = assets[asset_id]
        concepts = ttl_concepts(asset.local_path.read_text(encoding="utf-8", errors="replace"))
        for index, (identifier, label, comment) in enumerate(concepts):
            searchable = f"{identifier} {label} {comment}"
            if not re.search(r"power|energy|electric|building|device|meter|grid|voltage|current|load|profile|connection|service|storage|generator|transformer", searchable, re.I):
                continue
            records.append(
                record(
                    record_id=f"{asset_id}-{index:03d}-{identifier}",
                    domain=domain,
                    record_type="standard_ontology_concept",
                    title=f"SAREF concept: {label}",
                    summary=comment or f"The versioned ETSI ontology declares {identifier} as {label}.",
                    keywords=["SAREF", identifier, label, domain],
                    entities=[identifier, label],
                    relations=["standard_concept_has_label"],
                    rules=["Retain the versioned SAREF identifier when aligning building and grid vocabulary."],
                    applicability=["cross-system semantic normalization"],
                    limitations=["A standard concept supplies semantics, not site-specific topology or numerical state."],
                    source_refs=[source_ref(asset, lock)],
                )
            )
    return records


def extract_openleadr(
    assets: dict[str, Asset], lock: dict[str, dict[str, object]]
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    enum_asset = assets["openleadr-enums"]
    tree = ast.parse(enum_asset.local_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        values: list[str] = []
        for child in node.body:
            if isinstance(child, ast.Assign) and len(child.targets) == 1 and isinstance(child.targets[0], ast.Name):
                try:
                    literal = ast.literal_eval(child.value)
                except (ValueError, TypeError):
                    continue
                values.append(f"{child.targets[0].id}={literal}")
        if not values:
            continue
        records.append(
            record(
                record_id=f"openadr-enum-{node.name}",
                domain="cross_domain",
                record_type="interaction_vocabulary",
                title=f"OpenADR vocabulary: {node.name}",
                summary=f"The pinned OpenLEADR implementation declares {node.name} values: {', '.join(values[:24])}.",
                keywords=["OpenADR", "OpenLEADR", node.name, *values],
                entities=[node.name, *values],
                relations=["protocol_enum_allows_value"],
                rules=["Reject or clarify interaction values that are outside the pinned protocol vocabulary."],
                applicability=["demand-response events", "opt decisions", "reports and targets"],
                limitations=["Protocol vocabulary does not prove physical deliverability."],
                source_refs=[source_ref(enum_asset, lock)],
            )
        )
    for asset_id in sorted(key for key in assets if key.startswith("openleadr-") and key not in {"openleadr-enums", "openleadr-objects"}):
        asset = assets[asset_id]
        template_text = asset.local_path.read_text(encoding="utf-8", errors="replace")
        # OpenLEADR templates are XML/Jinja fragments, so namespace prefixes are
        # intentionally unbound until rendering.  Lexical tag extraction is the
        # correct deterministic operation here; parsing as standalone XML is not.
        raw_tags = re.findall(r"</?([A-Za-z_][\w.-]*(?::[A-Za-z_][\w.-]*)?)", template_text)
        tags = unique_text(tag.split(":")[-1] for tag in raw_tags)
        records.append(
            record(
                record_id=f"openadr-template-{asset_id}",
                domain="cross_domain",
                record_type="interaction_message_template",
                title=f"OpenADR message template: {asset_id.removeprefix('openleadr-').replace('-', ' ')}",
                summary=f"The OpenLEADR template declares message elements {', '.join(tags[:30])}.",
                keywords=["OpenADR", "OpenLEADR", "event", *tags],
                entities=tags,
                relations=["message_contains_protocol_element"],
                rules=["Preserve event identity, active period, signal, target, and opt/response state as separate fields."],
                applicability=["single-turn and multi-turn demand-response interaction"],
                limitations=["Template fields require domain-specific unit and feasibility validation."],
                source_refs=[source_ref(asset, lock)],
            )
        )
    return records


def extract_simbench(
    assets: dict[str, Asset], lock: dict[str, dict[str, object]]
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for asset_id in sorted(key for key in assets if key.startswith("simbench-")):
        asset = assets[asset_id]
        fields, rows = read_csv(asset)
        sample_values = unique_text((value for row in rows[:8] for value in row.values()), limit=30)
        records.append(
            record(
                record_id=f"simbench-table-{asset_id}",
                domain="grid",
                record_type="grid_data_schema",
                title=f"SimBench table: {asset_id.removeprefix('simbench-').replace('-', ' ')}",
                summary=f"The pinned SimBench table contains {len(rows)} rows with grid fields {', '.join(fields)}.",
                keywords=["SimBench", *fields, *sample_values],
                entities=[asset_id.removeprefix("simbench-"), *sample_values],
                relations=["network_table_references_grid_entities"],
                rules=["Resolve referenced node, type, profile, and study-case identifiers before constructing a power-flow model."],
                applicability=["distribution network import", "load and generation profiles", "grid constraint evaluation"],
                limitations=["The selected small upstream network is a schema-learning case, not the paper study feeder."],
                source_refs=[source_ref(asset, lock)],
            )
        )
        for field in fields:
            examples = unique_text((row.get(field, "") for row in rows[:20]), limit=8)
            records.append(
                record(
                    record_id=f"simbench-field-{asset_id}-{field}",
                    domain="grid",
                    record_type="grid_schema_field",
                    title=f"SimBench field: {field}",
                    summary=(
                        f"The pinned SimBench {asset_id.removeprefix('simbench-')} table "
                        f"declares the grid field {field}; observed source examples include "
                        f"{', '.join(examples) or 'no non-empty sample in the selected slice'}."
                    ),
                    keywords=["SimBench", field, *examples],
                    entities=[asset_id.removeprefix("simbench-"), field],
                    relations=["grid_table_declares_field"],
                    rules=["Bind this field through the pinned table schema before power-flow execution."],
                    applicability=["distribution network import", "grid evidence validation"],
                    limitations=["A schema field does not establish an episode-specific value."],
                    source_refs=[source_ref(asset, lock)],
                )
            )
    return records


def extract_citylearn_grid_fields(
    assets: dict[str, Asset], lock: dict[str, dict[str, object]]
) -> list[dict[str, object]]:
    asset = assets["citylearn-three-phase-schema"]
    payload = read_json(asset)
    paths = unique_text(path for path, _ in flatten_scalars(payload))
    selected = [
        path for path in paths
        if re.search(
            r"electric|phase|voltage|power|transformer|line|bus|node|load|solar|battery|capacity|efficiency",
            path,
            re.IGNORECASE,
        )
    ]
    return [
        record(
            record_id=f"citylearn-grid-field-{index:03d}-{path}",
            domain="grid",
            record_type="grid_coordination_schema_field",
            title=f"CityLearn three-phase grid field: {path}",
            summary=(
                f"The pinned CityLearn three-phase coordination schema declares {path} as a "
                "grid-side field used in a building-to-distribution-system case structure."
            ),
            keywords=["CityLearn", "three-phase", "distribution grid", path],
            entities=[path],
            relations=["coordination_schema_declares_grid_field"],
            rules=["Preserve phase, topology, unit, and interval semantics when binding this field."],
            applicability=["three-phase building-grid coordination"],
            limitations=["This is a public schema pattern, not the formal case topology or result."],
            source_refs=[source_ref(asset, lock)],
        )
        for index, path in enumerate(selected)
    ]


def mapping_records(
    assets: dict[str, Asset], lock: dict[str, dict[str, object]]
) -> list[dict[str, object]]:
    def refs(*asset_ids: str) -> list[dict[str, str]]:
        return [source_ref(assets[asset_id], lock) for asset_id in asset_ids]

    definitions = [
        (
            "average-power-from-interval-energy",
            "Interval energy to average active power",
            "Convert an interval energy difference in joules to average kW using delta_J / interval_seconds / 1000; never treat cumulative energy as instantaneous power.",
            ["energy", "power", "J", "kW", "interval"],
            ["ochre-variable-units", "tesp-eplus-federate"],
        ),
        (
            "kw-to-mw-boundary",
            "Building kW to grid MW",
            "At a power-flow boundary, convert active power from kW to MW by dividing by 1000 and retain the original sign convention in provenance.",
            ["kW", "MW", "active power", "pandapower"],
            ["ochre-variable-units", "simbench-load"],
        ),
        (
            "kvar-to-mvar-boundary",
            "Building kvar to grid Mvar",
            "At a power-flow boundary, convert reactive power from kvar to Mvar by dividing by 1000; active and reactive power must remain separate fields.",
            ["kvar", "Mvar", "reactive power"],
            ["saref4grid-ontology", "simbench-load"],
        ),
        (
            "net-load-composition",
            "Building net-load composition",
            "Represent net grid import as end-use consumption minus on-site generation minus discharging storage, with charging storage added as consumption after source sign conventions are normalized.",
            ["net load", "PV", "battery", "EV", "consumption"],
            ["brick-solar-array", "ochre-battery-parameters", "citylearn-ev-schema"],
        ),
        (
            "meter-to-grid-connection",
            "Building meter to electrical connection",
            "Join building meter or service identifiers to an explicit electrical-network connection identifier before aggregating demand at a bus or node.",
            ["meter", "service point", "bus", "node", "PCC"],
            ["brick-building-meter", "urbanopt-electric-network", "simbench-node"],
        ),
        (
            "submeter-aggregation",
            "Submeter hierarchy aggregation",
            "Aggregate submeters through the declared meter hierarchy exactly once; do not add a parent meter to the same children when the parent already measures their total.",
            ["meter", "submeter", "aggregation", "double counting"],
            ["brick-submeter", "brick-building-meter"],
        ),
        (
            "time-grid-alignment",
            "Building and grid interval alignment",
            "Align timezone, interval start/end convention, duration, and missing intervals before joining building and grid time series; resampling must declare whether it sums energy or averages power.",
            ["timezone", "interval", "resampling", "time series"],
            ["citylearn-ev-schema", "simbench-load-profile", "boptest-bestest-air-days"],
        ),
        (
            "thermostat-action-to-flexibility",
            "Thermostat action to electrical flexibility",
            "Translate a thermostat adjustment into electrical flexibility only after a building simulation reports the resulting power trajectory and comfort outcomes.",
            ["thermostat", "flexibility", "comfort", "power trajectory"],
            ["urbanopt-thermostat-shedding", "boptest-office-kpis", "ochre-variable-units"],
        ),
        (
            "building-kpi-to-feasibility",
            "Building KPI to feasibility evidence",
            "Energy, peak, cost, emissions, and thermal-discomfort KPIs are separate evidence; a grid service is not feasible merely because energy decreases.",
            ["KPI", "comfort", "energy", "peak", "feasibility"],
            ["boptest-bestest-air-kpis", "boptest-commercial-kpis"],
        ),
        (
            "helics-publication-subscription",
            "HELICS publication to subscription routing",
            "Route a co-simulation value from the declared publication to the matching subscription and validate name, datatype, unit, and time basis at both ends.",
            ["HELICS", "publication", "subscription", "federate"],
            ["tesp-eplus-federate", "tesp-eplus-agent"],
        ),
        (
            "openadr-active-period",
            "OpenADR active period to task window",
            "Map an OpenADR event active period to an explicit task time window while retaining event identity, modification number, notification time, and ramp or recovery periods when declared.",
            ["OpenADR", "event", "active period", "time window"],
            ["openleadr-event", "openleadr-active-period"],
        ),
        (
            "openadr-signal-target",
            "OpenADR event signal and target",
            "Keep the event signal payload, signal type, unit, target entity, and interval schedule separate; the target identifies who is addressed and the signal describes what is requested.",
            ["OpenADR", "signal", "target", "payload", "interval"],
            ["openleadr-event-signal", "openleadr-event-target"],
        ),
        (
            "openadr-opt-response",
            "OpenADR response as acceptance state",
            "Represent accept, decline, and modification or counteroffer states explicitly; protocol acceptance does not replace physical delivery verification.",
            ["OpenADR", "opt", "accept", "decline", "counteroffer"],
            ["openleadr-enums", "openleadr-distribute-event"],
        ),
        (
            "der-to-grid-element",
            "DER model to grid element",
            "Map PV, battery, EV, or other DER to a typed grid element at an explicit connection node and preserve capacity, availability, state, efficiency, and active/reactive capability.",
            ["DER", "PV", "battery", "EV", "grid connection"],
            ["urbanopt-pv-project", "brick-solar-array", "brick-evse", "simbench-node"],
        ),
        (
            "request-to-validated-delivery",
            "Requested service to validated delivery",
            "Keep requested quantity, offered quantity, scheduled trajectory, simulated delivery, and verified completion as separate states so that a partial response can be quantified without rewriting the original request.",
            ["request", "offer", "schedule", "delivery", "completion"],
            ["tesp-loadshed-agent", "openleadr-event-signal", "boptest-heat-pump-kpis"],
        ),
        (
            "precooling-and-rebound",
            "Precooling service and rebound accounting",
            "Treat precooling as a time-shifted building action and report both event-window power change and post-event rebound before declaring grid-service completion.",
            ["precooling", "thermal inertia", "rebound", "comfort"],
            ["boptest-commercial-kpis", "urbanopt-thermostat-shedding"],
        ),
        (
            "comfort-constraint-preservation",
            "Comfort constraints across domain translation",
            "Carry occupied-zone comfort bounds as hard constraints when a grid request is translated into building controls; power delivery alone is insufficient evidence.",
            ["comfort", "occupied zone", "hard constraint", "delivery"],
            ["boptest-office-kpis", "ochre-variable-units"],
        ),
        (
            "voltage-request-direction",
            "Voltage support direction and location",
            "Translate a voltage-support request with the affected bus, lower or upper bound, and required active or reactive response direction kept explicit.",
            ["voltage support", "bus", "active power", "reactive power"],
            ["saref4grid-ontology", "simbench-node", "simbench-load"],
        ),
        (
            "congestion-location",
            "Congestion request to constrained grid element",
            "Bind a congestion-management request to the affected line or transformer and its loading limit before allocating building flexibility by connection bus.",
            ["congestion", "line", "transformer", "loading limit", "bus"],
            ["simbench-line", "simbench-transformer", "urbanopt-electric-network"],
        ),
        (
            "reserve-capacity-activation",
            "Reserve capacity and activation separation",
            "Keep reserved capacity, activated trajectory, delivered trajectory, and rebound as separate quantities with aligned event intervals.",
            ["reserve", "capacity", "activation", "delivery", "rebound"],
            ["openleadr-event-signal", "tesp-loadshed-agent", "boptest-heat-pump-kpis"],
        ),
        (
            "renewable-absorption-sign",
            "Renewable absorption sign convention",
            "An increase in building consumption or storage charging can reduce export curtailment; normalize import, export, generation, charging, and discharging signs before quantifying absorption.",
            ["renewable absorption", "curtailment", "charging", "sign convention"],
            ["citylearn-ev-schema", "ochre-battery-parameters", "urbanopt-pv-project"],
        ),
        (
            "counterfactual-baseline",
            "Counterfactual baseline and delivered flexibility",
            "Compute delivered flexibility against a traceable no-event counterfactual over the same weather, occupancy, and interval grid; never substitute a requested schedule for the baseline.",
            ["counterfactual", "baseline", "weather", "occupancy", "delivery"],
            ["boptest-bestest-air-days", "boptest-bestest-air-kpis", "ochre-variable-units"],
        ),
        (
            "market-offer-settlement",
            "Market offer, acceptance, delivery, and settlement",
            "Preserve offered capacity and price, accepted capacity, simulated delivery, penalty rule, and final settlement as distinct auditable states.",
            ["market", "offer", "acceptance", "delivery", "settlement"],
            ["openleadr-event-signal", "openleadr-distribute-event", "tesp-loadshed-agent"],
        ),
        (
            "spatial-portfolio-allocation",
            "Portfolio allocation to feeder buses",
            "Allocate portfolio-level flexibility to individual building meters and their mapped feeder buses before power flow, and retain the allocation weights and residual mismatch.",
            ["portfolio", "allocation", "meter", "feeder bus", "residual"],
            ["urbanopt-electric-network", "brick-building-meter", "simbench-node"],
        ),
    ]
    return [
        record(
            record_id=f"mapping-{identifier}",
            domain="cross_domain",
            record_type="executable_mapping_rule",
            title=title,
            summary=summary,
            keywords=keywords,
            entities=keywords,
            relations=[identifier.replace("-", "_")],
            rules=[summary],
            applicability=["Translator normalization", "Actor input validation", "Reviewer evidence audit"],
            limitations=["Apply only after validating the source-specific fields and conventions cited by the record."],
            review_status="source_derived_method_frozen",
            source_refs=refs(*asset_ids),
        )
        for identifier, title, summary, keywords, asset_ids in definitions
    ]


def validate_records(
    records_by_domain: dict[str, list[dict[str, object]]],
    registry: dict[str, object],
) -> None:
    source_ids = {source["source_id"] for source in registry["sources"]}  # type: ignore[index]
    all_ids: set[str] = set()
    for domain in DOMAINS:
        records = records_by_domain[domain]
        if not records:
            raise ValueError(f"No records extracted for domain {domain}")
        for item in records:
            required = {
                "record_id", "domain", "record_type", "title", "summary", "keywords",
                "entities", "relations", "rules", "applicability", "limitations", "source_refs",
                "units", "time_scales", "spatial_levels", "applicable_roles",
                "business_types", "directions", "review_status",
            }
            if set(item) != required:
                raise ValueError(f"Unexpected record fields for {item.get('record_id')}")
            record_id = str(item["record_id"])
            if record_id in all_ids:
                raise ValueError(f"Duplicate knowledge record_id: {record_id}")
            all_ids.add(record_id)
            if item["domain"] != domain:
                raise ValueError(f"Domain mismatch for {record_id}")
            if not item["source_refs"]:
                raise ValueError(f"Missing source_refs for {record_id}")
            for ref in item["source_refs"]:  # type: ignore[union-attr]
                if ref["source_id"] not in source_ids:
                    raise ValueError(f"Unknown source {ref['source_id']} in {record_id}")
                if not re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]):
                    raise ValueError(f"Invalid source hash in {record_id}")
            serialized = json.dumps(item, ensure_ascii=False, sort_keys=True)
            for pattern in FORBIDDEN_OUTPUT_PATTERNS:
                if pattern.search(serialized):
                    raise ValueError(f"Potential benchmark leakage in {record_id}: {pattern.pattern}")


def balance_records(
    records_by_domain: dict[str, list[dict[str, object]]],
) -> tuple[dict[str, list[dict[str, object]]], dict[str, int]]:
    """Select equal-size libraries without duplicate padding or benchmark data."""

    candidate_counts = {domain: len(records_by_domain[domain]) for domain in DOMAINS}
    balanced: dict[str, list[dict[str, object]]] = {}
    for domain in DOMAINS:
        candidates = records_by_domain[domain]
        if len(candidates) < TARGET_RECORDS_PER_DOMAIN:
            raise ValueError(
                f"{domain} has {len(candidates)} unique candidates; "
                f"{TARGET_RECORDS_PER_DOMAIN} are required without duplicate padding"
            )
        buckets: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
        for item in candidates:
            sources = "+".join(sorted({ref["source_id"] for ref in item["source_refs"]}))
            buckets[(str(item["record_type"]), sources)].append(item)
        for bucket in buckets.values():
            bucket.sort(key=lambda item: str(item["record_id"]))
        selected: list[dict[str, object]] = []
        keys = sorted(buckets)
        while len(selected) < TARGET_RECORDS_PER_DOMAIN:
            progressed = False
            for key in keys:
                if buckets[key] and len(selected) < TARGET_RECORDS_PER_DOMAIN:
                    selected.append(buckets[key].pop(0))
                    progressed = True
            if not progressed:
                break
        balanced[domain] = selected
    return balanced, candidate_counts


def write_records(records_by_domain: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    outputs: dict[str, object] = {}
    for domain in DOMAINS:
        ordered = sorted(records_by_domain[domain], key=lambda item: str(item["record_id"]))
        path = KNOWLEDGE_ROOT / domain / "records.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for item in ordered
        )
        path.write_text(content, encoding="utf-8")
        type_counts: dict[str, int] = defaultdict(int)
        source_counts: dict[str, int] = defaultdict(int)
        for item in ordered:
            type_counts[str(item["record_type"])] += 1
            for ref in item["source_refs"]:  # type: ignore[union-attr]
                source_counts[ref["source_id"]] += 1
        outputs[domain] = {
            "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "record_count": len(ordered),
            "sha256": sha256_file(path),
            "record_type_counts": dict(sorted(type_counts.items())),
            "source_reference_counts": dict(sorted(source_counts.items())),
        }
    return outputs


def build(raw_root: Path, *, force: bool, offline: bool, update_lock: bool) -> dict[str, object]:
    registry = load_registry()
    assets = build_asset_catalog(registry, raw_root)
    lock = acquire(assets, force=force, offline=offline, update_lock=update_lock)
    records: dict[str, list[dict[str, object]]] = {domain: [] for domain in DOMAINS}
    extracted = [
        *extract_tesp(assets, lock),
        *extract_urbanopt(assets, lock),
        *extract_boptest(assets, lock),
        *extract_citylearn(assets, lock),
        *extract_ochre(assets, lock),
        *extract_brick(assets, lock),
        *extract_saref(assets, lock),
        *extract_openleadr(assets, lock),
        *extract_simbench(assets, lock),
        *extract_citylearn_grid_fields(assets, lock),
        *mapping_records(assets, lock),
    ]
    for item in extracted:
        records[str(item["domain"])].append(item)
    validate_records(records, registry)
    records, candidate_counts = balance_records(records)
    validate_records(records, registry)
    output_info = write_records(records)
    manifest = {
        "schema_version": "1.0",
        "knowledge_release": registry["knowledge_release"],
        "extractor_version": EXTRACTOR_VERSION,
        "source_registry_sha256": sha256_file(REGISTRY_PATH),
        "asset_lock_sha256": sha256_file(LOCK_PATH),
        "raw_asset_count": len(assets),
        "raw_asset_bytes": sum(int(item["bytes"]) for item in lock.values()),
        "leakage_boundary": registry["policy"],
        "outputs": output_info,
        "candidate_record_counts": candidate_counts,
        "balance_policy": {
            "target_per_domain": TARGET_RECORDS_PER_DOMAIN,
            "selection": "deterministic round-robin over record_type and source strata",
            "duplicate_padding": False,
        },
        "release_sha256": canonical_sha256(output_info),
    }
    manifest_path = KNOWLEDGE_ROOT / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.getenv("B2G_DATA_ROOT", DEFAULT_DATA_ROOT)),
        help="Root for ignored raw assets; defaults to B2G_DATA_ROOT or .local-data",
    )
    parser.add_argument("--offline", action="store_true", help="Use only previously downloaded assets")
    parser.add_argument("--force", action="store_true", help="Redownload every registered asset")
    parser.add_argument(
        "--update-lock",
        action="store_true",
        help="Explicitly replace the committed asset hashes after an intentional source update",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = args.data_root.resolve() / "knowledge_sources"
    manifest = build(
        raw_root,
        force=args.force,
        offline=args.offline,
        update_lock=args.update_lock,
    )
    counts = {
        domain: payload["record_count"]
        for domain, payload in manifest["outputs"].items()
    }
    print(
        json.dumps(
            {
                "knowledge_release": manifest["knowledge_release"],
                "raw_asset_count": manifest["raw_asset_count"],
                "raw_asset_bytes": manifest["raw_asset_bytes"],
                "record_counts": counts,
                "manifest": str((KNOWLEDGE_ROOT / "manifest.json").relative_to(REPO_ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
