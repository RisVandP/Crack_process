from __future__ import annotations

import json

from src.experiments.experiment import _scenario_data, load_experiment_config, run_experiment
from src.io.data_loader import parse_problem
from src.model.solution_utils import empty_solution


def test_example_json_has_required_case_and_scenario_counts():
    cfg = load_experiment_config("configs/example.json")
    assert len(cfg["cases"]) == 12
    assert cfg["algorithms"] == ["VF", "VDF", "MG", "MG-LS"]
    for case in cfg["cases"]:
        assert len(case["uncertainty_scenarios"]) == 8


def test_example_json_uses_only_formal_geometry_inputs():
    cfg = load_experiment_config("configs/example.json")
    expected = {
        "C01": (6, 5, 3, 2, 5.0, 2),
        "C02": (4, 3, 2, 2, 6.0, 1),
        "C03": (8, 6, 5, 3, 5.2, 3),
        "C04": (8, 6, 4, 2, 4.1, 3),
        "C05": (6, 5, 3, 2, 5.4, 4),
        "C06": (6, 5, 3, 2, 5.1, 4),
        "C07": (7, 5, 4, 2, 5.0, 3),
        "C08": (6, 5, 5, 1, 5.0, 2),
        "C09": (6, 5, 4, 2, 5.2, 2),
        "C10": (6, 4, 2, 1, 3.4, 2),
        "C11": (6, 5, 3, 3, 5.1, 3),
        "C12": (8, 6, 4, 1, 3.8, 5),
    }
    for case in cfg["cases"]:
        raw = case["deterministic_input"]
        m, n, ordinary, precision, deadline, cracks = expected[case["id"]]
        assert "random_seed" not in raw
        assert raw["board"]["m"] == m
        assert raw["board"]["n"] == n
        assert raw["deadline"] == deadline
        assert sum(1 for device in raw["devices"] if device["type"] == "ordinary") == ordinary
        assert sum(1 for device in raw["devices"] if device["type"] == "precision") == precision
        assert raw["cracks"]["mode"] == "geometry"
        assert len(raw["cracks"]["items"]) == cracks
        assert "blocks" not in raw["cracks"]
        for item in raw["cracks"]["items"]:
            assert len(item["polyline"]) >= 5
            assert item["width"] > 0
        for scenario in case["uncertainty_scenarios"]:
            assert "hidden_cracks" in scenario
            assert "crack_updates" not in scenario
            assert "edge_loss_updates" not in scenario


def test_all_example_cases_parse_and_have_valid_inputs():
    cfg = load_experiment_config("configs/example.json")
    for case in cfg["cases"]:
        data = parse_problem(case["deterministic_input"])
        assert data.blocks
        assert data.edges
        assert data.deadline > 0
        assert all(device.speed > 0 for device in data.devices)
        assert len(data.blocks) == data.board.m * data.board.n
        assert len(data.edges) == (data.board.m - 1) * data.board.n + data.board.m * (data.board.n - 1)


def test_hidden_crack_scenarios_are_derived_from_geometry():
    cfg = load_experiment_config("configs/example.json")
    case = cfg["cases"][0]
    data = parse_problem(case["deterministic_input"])
    scenario = next(item for item in case["uncertainty_scenarios"] if item["id"] == "U6")
    scenario_data = _scenario_data(data, scenario)
    assert len(scenario_data.cracks) > len(data.cracks)
    base_cs = {block.id: block.crack_severity for block in data.blocks}
    scenario_cs = {block.id: block.crack_severity for block in scenario_data.blocks}
    assert any(abs(scenario_cs[block_id] - base_cs[block_id]) > 1e-9 for block_id in base_cs)


def test_experiment_two_stage_counts_and_hash_stability(tmp_path, monkeypatch):
    import src.experiments.experiment as experiment

    calls = {method: 0 for method in ["VF", "VDF", "MG", "MG-LS"]}

    def fake_solver(method):
        def solve(data):
            calls[method] += 1
            return empty_solution(data, method)

        return solve

    monkeypatch.setattr(experiment, "METHODS", {method: fake_solver(method) for method in calls})
    result = run_experiment("configs/example.json", tmp_path)
    assert len(result["stage1_rows"]) == 48
    assert len(result["stage2_rows"]) == 384
    assert calls == {"VF": 12, "VDF": 12, "MG": 12, "MG-LS": 12}

    for case_id in {row["case_id"] for row in result["stage1_rows"]}:
        for method in calls:
            hashes = {
                row["solution_hash"]
                for row in result["stage2_rows"]
                if row["case_id"] == case_id and row["method"] == method
            }
            assert len(hashes) == 1


def test_experiment_scores_and_ranking_are_valid(tmp_path, monkeypatch):
    import src.experiments.experiment as experiment

    def fake_solver(method):
        return lambda data: empty_solution(data, method)

    monkeypatch.setattr(experiment, "METHODS", {method: fake_solver(method) for method in ["VF", "VDF", "MG", "MG-LS"]})
    result = run_experiment("configs/example.json", tmp_path)
    ranking = result["final_ranking"]
    assert {row["method"] for row in ranking} == {"VF", "VDF", "MG", "MG-LS"}
    assert len(ranking) == 4
    assert sorted(row["rank"] for row in ranking) == [1, 2, 3, 4]
    for row in ranking:
        assert 0.0 <= row["final_score"] <= 1.0


def test_experiment_repeated_fake_runs_are_stable_except_output_files(tmp_path, monkeypatch):
    import src.experiments.experiment as experiment

    def fake_solver(method):
        return lambda data: empty_solution(data, method)

    monkeypatch.setattr(experiment, "METHODS", {method: fake_solver(method) for method in ["VF", "VDF", "MG", "MG-LS"]})
    first = run_experiment("configs/example.json", tmp_path / "one")
    second = run_experiment("configs/example.json", tmp_path / "two")
    first_rows = [{k: v for k, v in row.items() if k != "run_seconds"} for row in first["stage1_rows"]]
    second_rows = [{k: v for k, v in row.items() if k != "run_seconds"} for row in second["stage1_rows"]]
    assert json.dumps(first_rows, sort_keys=True) == json.dumps(second_rows, sort_keys=True)
    assert json.dumps(first["stage2_rows"], sort_keys=True) == json.dumps(second["stage2_rows"], sort_keys=True)
