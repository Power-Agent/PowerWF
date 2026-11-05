"""Freeze the IEEE 33-bus source network and verify the ten BEM bus choices."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import networkx as nx
import pandapower as pp
import pandapower.networks as pn


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = REPO_ROOT / "cases" / "ieee33_10bem_15min"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def electrical_distances(net: object) -> dict[int, float]:
    graph = nx.Graph()
    for index, line in net.line.iterrows():  # type: ignore[attr-defined]
        if not bool(line.in_service):
            continue
        resistance = float(line.r_ohm_per_km) * float(line.length_km)
        graph.add_edge(int(line.from_bus), int(line.to_bus), weight=resistance, line_index=int(index))
    slack = int(net.ext_grid.iloc[0].bus)  # type: ignore[attr-defined]
    return {
        int(bus): float(distance)
        for bus, distance in nx.single_source_dijkstra_path_length(
            graph, slack, weight="weight"
        ).items()
    }


def selected_quantile_buses(distances: dict[int, float], count: int = 10) -> list[int]:
    ordered = sorted((distance, bus) for bus, distance in distances.items() if bus != 0)
    selected = []
    for index in range(1, count + 1):
        # Nearest-rank deciles of the empirical electrical-distance distribution.
        # Using the distribution rather than equal distance increments prevents
        # the sparse far end of the feeder from dominating node selection.
        position = max(0, min(len(ordered) - 1, round(index * len(ordered) / count) - 1))
        selected.append(ordered[position][1])
    if len(set(selected)) != count:
        raise RuntimeError("Electrical-distance quantiles did not produce ten unique buses")
    return selected


def main() -> None:
    network_dir = CASE_DIR / "network"
    network_dir.mkdir(parents=True, exist_ok=True)
    net = pn.case33bw()
    source_path = network_dir / "source_case33bw.json"
    study_path = network_dir / "study_network.json"
    pp.to_json(net, source_path)
    study = pp.from_json(source_path)
    study["b2g_metadata"] = {
        "source": "pandapower.networks.case33bw",
        "source_pandapower_version": pp.__version__,
        "line_limits": "external experiment-specific line_limits.csv",
        "bus_numbering": "internal 0-32; paper 1-33",
    }
    pp.to_json(study, study_path)

    distances = electrical_distances(net)
    selected_internal = selected_quantile_buses(distances)
    selected_paper = [bus + 1 for bus in selected_internal]
    expected = [3, 5, 21, 27, 8, 29, 11, 33, 15, 18]
    if selected_paper != expected:
        raise RuntimeError(f"BEM bus selection changed: {selected_paper} != {expected}")

    metadata = {
        "pandapower_version": pp.__version__,
        "source_sha256": sha256(source_path),
        "study_sha256": sha256(study_path),
        "bus_count": int(len(net.bus)),
        "load_count": int(len(net.load)),
        "line_count": int(len(net.line)),
        "in_service_line_count": int(net.line.in_service.sum()),
        "out_of_service_tie_count": int((~net.line.in_service).sum()),
        "base_load_mw": float(net.load.p_mw.sum()),
        "base_reactive_load_mvar": float(net.load.q_mvar.sum()),
        "selected_buses_in_electrical_distance_order": selected_paper,
        "electrical_distance_ohm_by_paper_bus": {
            str(bus + 1): round(distance, 6) for bus, distance in sorted(distances.items())
        },
    }
    (network_dir / "source_case33bw_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
