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
        if calls["count"] == 2:
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
    # Preserve supply to bus 2 after either series-line outage so the
    # transformer remains energized in at least one converged contingency.
    module.pp.create_line_from_parameters(
        net, 0, 2, length_km=12, r_ohm_per_km=0.05,
        x_ohm_per_km=0.15, c_nf_per_km=100, max_i_ka=1.0
    )
    b3 = module.pp.create_bus(net, vn_kv=20.0)
    module.pp.create_transformer_from_parameters(
        net, 2, b3, sn_mva=0.5,
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
    assert any(
        row["violations"]["transformer_loading_violations"]
        for row in result["results"]
        if row["status"] == "converged"
    )
    assert result["summary"]["converged_with_violations"] >= 1



def test_nonconverged_powerflow_is_recorded_as_failed(monkeypatch):
    net = _network()
    module._current_net = net

    calls = {"count": 0}

    def nonconverged(contingency_net, *args, **kwargs):
        calls["count"] += 1
        contingency_net.converged = calls["count"] == 1

    monkeypatch.setattr(module.pp, "runpp", nonconverged)
    result = module.run_contingency_analysis.invoke({})

    assert result["status"] == "success"
    assert result["summary"]["tested"] == 2
    assert result["summary"]["failed"] == 2
    assert result["summary"]["converged"] == 0
    assert calls["count"] == 3
    assert all(row["status"] == "failed" for row in result["results"])
    assert all("returned without convergence" in row["error"] for row in result["results"])


def test_rejects_unsupported_contingency_type():
    net = _network()
    module._current_net = net

    result = module.run_contingency_analysis.invoke({"contingency_type": "N-2"})

    assert result["status"] == "error"
    assert "Unsupported contingency type" in result["message"]



@pytest.mark.parametrize("elements", [["bus"], ["line", "bus"], ["unknown"]])
def test_rejects_unsupported_contingency_element_types(elements):
    module._current_net = _network()
    result = module.run_contingency_analysis.invoke({"elements": elements})
    assert result["status"] == "error"
    assert "Unsupported contingency element types" in result["message"]


def test_duplicate_contingency_element_types_are_deduplicated():
    module._current_net = _network()
    result = module.run_contingency_analysis.invoke({"elements": ["line", "line"]})
    assert result["status"] == "success"
    assert result["summary"]["tested"] == 2


def test_contingency_results_follow_the_requested_scope():
    module._current_net = _network()
    result = module.run_contingency_analysis.invoke({"elements": ["line"]})
    assert result["status"] == "success"
    assert result["summary"]["tested"] == 2
    assert [row["contingency"].split("_")[0] for row in result["results"]] == ["line", "line"]



def test_unsupported_element_selectors_fail_closed():
    module._current_net = _network()
    result = module.run_contingency_analysis.invoke({"elements": ["line", "bus"]})
    assert result["status"] == "error"
    assert "Unsupported contingency element types" in result["message"]
def test_empty_element_list_returns_explicit_zero_work_summary(monkeypatch):
    module._current_net = _network()

    def unexpected_powerflow(*args, **kwargs):
        pytest.fail("an empty contingency scope must not run a power flow")

    monkeypatch.setattr(module.pp, "runpp", unexpected_powerflow)
    result = module.run_contingency_analysis.invoke({"elements": []})
    assert result["status"] == "success"
    assert result["summary"] == {
        "tested": 0,
        "converged": 0,
        "failed": 0,
        "converged_with_violations": 0,
        "skipped": 0,
    }
    assert result["results"] == []
    assert result["skipped_contingencies"] == []



def test_nonconverged_base_case_aborts_n_minus_one_sweep(monkeypatch):
    module._current_net = _network()
    calls = {"count": 0}

    def nonconverged_base(contingency_net, *args, **kwargs):
        calls["count"] += 1
        contingency_net.converged = False

    monkeypatch.setattr(module.pp, "runpp", nonconverged_base)
    result = module.run_contingency_analysis.invoke({})

    assert result["status"] == "error"
    assert result["base_case"]["converged"] is False
    assert result["results"] == []
    assert calls["count"] == 1



def test_contingency_analysis_does_not_mutate_callers_network():
    net = _network()
    original_in_service = net.line["in_service"].tolist()
    module._current_net = net

    result = module.run_contingency_analysis.invoke({})

    assert result["status"] == "success"
    assert net.line["in_service"].tolist() == original_in_service
    assert net.res_bus.empty
    assert net.res_line.empty



def test_out_of_service_elements_are_reported_as_skipped():
    net = _network()
    module.pp.create_line_from_parameters(
        net, 0, 2, length_km=12, r_ohm_per_km=0.05,
        x_ohm_per_km=0.15, c_nf_per_km=100, max_i_ka=1.0
    )
    net.line.at[0, "in_service"] = False
    module._current_net = net

    result = module.run_contingency_analysis.invoke({})

    assert result["status"] == "success"
    assert result["summary"]["skipped"] == 1
    assert result["skipped_contingencies"] == ["line_0"]
    assert all(row["contingency"] != "line_0" for row in result["results"])
