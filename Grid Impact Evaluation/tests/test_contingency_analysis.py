"""Regression tests for truthful pandapower contingency accounting."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "agent" / "tools_pandapower.py"
spec = importlib.util.spec_from_file_location("tools_pandapower", MODULE_PATH)
module = importlib.util.module_from_spec(spec)


pytest.importorskip("pandapower")
pytest.importorskip("langchain_core")
spec.loader.exec_module(module)


def _network():
    net = module.pp.create_empty_network()
    b0 = module.pp.create_bus(net, vn_kv=110.0)
    b1 = module.pp.create_bus(net, vn_kv=110.0)
    b2 = module.pp.create_bus(net, vn_kv=110.0)
    module.pp.create_ext_grid(net, b0, vm_pu=1.02)
    module.pp.create_line_from_parameters(
        net, b0, b1, length_km=10, r_ohm_per_km=0.05,
        x_ohm_per_km=0.15, c_nf_per_km=100, max_i_ka=1.0
    )
    module.pp.create_line_from_parameters(
        net, b1, b2, length_km=5, r_ohm_per_km=0.05,
        x_ohm_per_km=0.15, c_nf_per_km=100, max_i_ka=1.0
    )
    module.pp.create_load(net, b2, p_mw=10.0, q_mvar=2.0)
    return net


def test_contingency_summary_records_every_run(monkeypatch):
    net = _network()
    module._current_net = net

    original_runpp = module.pp.runpp
    calls = {"count": 0}

    def runpp_with_one_failure(contingency_net, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("forced contingency failure")
        return original_runpp(contingency_net, *args, **kwargs)

    monkeypatch.setattr(module.pp, "runpp", runpp_with_one_failure)

    result = module.run_contingency_analysis.invoke({})

    assert result["status"] == "success"
    assert result["summary"]["tested"] == 2
    assert result["summary"]["failed"] == 1
    assert result["summary"]["converged"] == 1
    assert any(row["status"] == "failed" for row in result["results"])
    assert all("violations" in row for row in result["results"])


def test_contingency_checks_transformer_loading():
    net = _network()
    b3 = module.pp.create_bus(net, vn_kv=20.0)
    module.pp.create_transformer_from_parameters(
        net, 2, b3, sn_mva=0.01,
        vn_hv_kv=110.0, vn_lv_kv=20.0,
        vkr_percent=0.5, vk_percent=10.0,
        pfe_kw=0.0, i0_percent=0.0
    )
    module.pp.create_load(net, b3, p_mw=1.0, q_mvar=0.2)
    module._current_net = net

    result = module.run_contingency_analysis.invoke({})

    assert result["status"] == "success"
    assert all(
        "transformer_loading_violations" in row["violations"]
        for row in result["results"]
    )
