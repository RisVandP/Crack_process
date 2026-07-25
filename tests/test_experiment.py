from __future__ import annotations

import json

from src.experiment import load_experiment_config, run_experiment
from src.data_loader import parse_problem
from src.solution_utils import empty_solution


def test_example_json_has_required_case_and_scenario_counts():
    cfg = load_experiment_config("configs/example.json")
    assert len(cfg["cases"]) == 12
    assert cfg["algorithms"] == ["VF", "VDF", "MG", "CNAG-LS"]
    for case in cfg["cases"]:
        assert len(case["uncertainty_scenarios"]) == 8


def test_all_example_cases_parse_and_have_valid_inputs():
    cfg = load_experiment_config("configs/example.json")
    for case in cfg["cases"]:
        data = parse_problem(case["deterministic_input"])
        assert data.blocks
        assert data.edges
        assert data.deadline > 0
        assert all(device.speed > 0 for device in data.devices)


def test_experiment_two_stage_counts_and_hash_stability(tmp_path, monkeypatch):
    import src.experiment as experiment

    calls = {method: 0 for method in ["VF", "VDF", "MG", "CNAG-LS"]}

    def fake_solver(method):
        def solve(data):
            calls[method] += 1
            return empty_solution(data, method)

        return solve

    monkeypatch.setattr(experiment, "METHODS", {method: fake_solver(method) for method in calls})
    result = run_experiment("configs/example.json", tmp_path)
    assert len(result["stage1_rows"]) == 48
    assert len(result["stage2_rows"]) == 384
    assert calls == {"VF": 12, "VDF": 12, "MG": 12, "CNAG-LS": 12}

    for case_id in {row["case_id"] for row in result["stage1_rows"]}:
        for method in calls:
            hashes = {
                row["solution_hash"]
                for row in result["stage2_rows"]
                if row["case_id"] == case_id and row["method"] == method
            }
            assert len(hashes) == 1


def test_experiment_scores_and_ranking_are_valid(tmp_path, monkeypatch):
    import src.experiment as experiment

    def fake_solver(method):
        return lambda data: empty_solution(data, method)

    monkeypatch.setattr(experiment, "METHODS", {method: fake_solver(method) for method in ["VF", "VDF", "MG", "CNAG-LS"]})
    result = run_experiment("configs/example.json", tmp_path)
    ranking = result["final_ranking"]
    assert {row["method"] for row in ranking} == {"VF", "VDF", "MG", "CNAG-LS"}
    assert len(ranking) == 4
    assert sorted(row["rank"] for row in ranking) == [1, 2, 3, 4]
    for row in ranking:
        assert 0.0 <= row["final_score"] <= 1.0


def test_experiment_repeated_fake_runs_are_stable_except_output_files(tmp_path, monkeypatch):
    import src.experiment as experiment

    def fake_solver(method):
        return lambda data: empty_solution(data, method)

    monkeypatch.setattr(experiment, "METHODS", {method: fake_solver(method) for method in ["VF", "VDF", "MG", "CNAG-LS"]})
    first = run_experiment("configs/example.json", tmp_path / "one")
    second = run_experiment("configs/example.json", tmp_path / "two")
    first_rows = [{k: v for k, v in row.items() if k != "run_seconds"} for row in first["stage1_rows"]]
    second_rows = [{k: v for k, v in row.items() if k != "run_seconds"} for row in second["stage1_rows"]]
    assert json.dumps(first_rows, sort_keys=True) == json.dumps(second_rows, sort_keys=True)
    assert json.dumps(first["stage2_rows"], sort_keys=True) == json.dumps(second["stage2_rows"], sort_keys=True)
