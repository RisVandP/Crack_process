from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.Algorithm.CNAG_LS import solve_cnag_ls
from src.data_loader import parse_problem
from src.deterministic_model import processing_time
from src.feasibility_checker import check_solution
from src.Algorithm.MG import _incident_edges, _single_move_delta, solve_marginal_greedy
from src.Algorithm.VDF import solve_value_density_first
from src.Algorithm.VF import solve_value_first
from src.method_comparison import run_method_comparison
from src.solution_evaluator import objective_breakdown
from src.solution_utils import apply_task_copy, empty_solution


def comparison_raw_instance():
    return {
        "board": {"width": 6.0, "height": 4.0, "m": 3, "n": 1, "base_value": 80.0},
        "deadline": 2.7,
        "devices": [
            {"id": "O_fast", "type": "ordinary", "speed": 8.0},
            {"id": "P_slow", "type": "precision", "speed": 3.0},
        ],
        "values": {
            "r_ordinary": 1.0,
            "r_precision": 2.0,
            "default_same_precision_reward": 2.0,
            "default_precision_mismatch_penalty": 1.0,
            "default_cross_crack_loss": 0.0,
            "alpha": 0.5,
        },
        "cracks": {
            "mode": "direct",
            "blocks": {
                "B_1_1": {"C": 1, "CS": 0.6},
                "B_2_1": {"C": 0, "CS": 0.0},
                "B_3_1": {"C": 0, "CS": 0.0},
            },
        },
    }


def test_four_heuristics_are_feasible():
    data = parse_problem(comparison_raw_instance())
    for solve in [solve_value_first, solve_value_density_first, solve_marginal_greedy, solve_cnag_ls]:
        solution = solve(data)
        check = check_solution(data, solution)
        assert check.feasible, check.messages


def test_greedy_repeated_runs_are_deterministic():
    data = parse_problem(comparison_raw_instance())
    first = solve_value_density_first(data)
    second = solve_value_density_first(data)
    assert first.x == second.x
    assert first.y_ordinary == second.y_ordinary
    assert first.y_precision == second.y_precision


def test_missing_precision_device_keeps_precision_assignments_zero():
    raw = comparison_raw_instance()
    raw["devices"] = [{"id": "O_only", "type": "ordinary", "speed": 8.0}]
    data = parse_problem(raw)
    solution = solve_value_first(data)
    assert all(value == 0 for value in solution.y_precision.values())
    assert check_solution(data, solution).feasible


def test_crack_severity_increases_processing_time():
    data = parse_problem(comparison_raw_instance())
    cracked_time = processing_time(data, "B_1_1", "O_fast")
    normal_time = processing_time(data, "B_2_1", "O_fast")
    assert cracked_time > normal_time


def test_cnag_ls_not_below_construction_value():
    data = parse_problem(comparison_raw_instance())
    solution = solve_cnag_ls(data)
    assert solution.objective_value + 1e-6 >= float(solution.metadata["construction_objective"])


def test_cnag_ls_not_below_marginal_greedy():
    data = parse_problem(comparison_raw_instance())
    cnag = solve_cnag_ls(data)
    marginal = solve_marginal_greedy(data)
    assert cnag.objective_value + 1e-6 >= marginal.objective_value


def test_all_methods_use_unified_objective_recalculation():
    data = parse_problem(comparison_raw_instance())
    for solution in [solve_value_first(data), solve_value_density_first(data), solve_marginal_greedy(data), solve_cnag_ls(data)]:
        breakdown = objective_breakdown(data, solution)
        assert abs(breakdown["total_objective"] - solution.objective_value) < 1e-5


def test_marginal_greedy_single_move_delta_matches_full_recalculation():
    data = parse_problem(comparison_raw_instance())
    solution = empty_solution(data, "MG")
    incident = _incident_edges(data)
    old_value = objective_breakdown(data, solution)["total_objective"]
    for block_id, process_type, device_id in [
        ("B_1_1", "ordinary", "O_fast"),
        ("B_1_1", "precision", "P_slow"),
    ]:
        candidate = apply_task_copy(data, solution, block_id, process_type, device_id)
        full_delta = objective_breakdown(data, candidate)["total_objective"] - old_value
        analytic_delta = _single_move_delta(data, solution, block_id, process_type, incident[block_id])
        assert abs(full_delta - analytic_delta) < 1e-6


def test_method_comparison_outputs_complete_files(tmp_path):
    data = parse_problem(comparison_raw_instance())
    rows = run_method_comparison(data, tmp_path)
    assert {row["method"] for row in rows} == {"VF", "VDF", "MG", "CNAG-LS"}
    for name in ["method_comparison.csv", "method_comparison.json", "method_comparison.png"]:
        assert (tmp_path / name).exists()
    with (tmp_path / "method_comparison.json").open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    required = {
        "method",
        "feasible",
        "status",
        "objective_value",
        "difference_to_cnag_ls_percent",
        "processed_count",
        "average_utilization",
        "max_utilization",
        "run_seconds",
        "intrinsic_block_value",
        "cross_crack_loss",
    }
    assert required.issubset(loaded[0].keys())


def test_invalid_speed_and_deadline_are_rejected():
    raw = comparison_raw_instance()
    raw["devices"][0]["speed"] = 0
    with pytest.raises(ValueError):
        parse_problem(raw)
    raw = comparison_raw_instance()
    raw["deadline"] = 0
    with pytest.raises(ValueError):
        parse_problem(raw)


def test_cli_all_generates_expected_files(tmp_path):
    output_dir = tmp_path / "cli_outputs"
    config_path = tmp_path / "small_cli_config.json"
    config_path.write_text(json.dumps(comparison_raw_instance(), ensure_ascii=False), encoding="utf-8")
    cmd = [
        sys.executable,
        "-m",
        "src.main",
        "--config",
        str(config_path),
        "--output",
        str(output_dir),
        "--method",
        "all",
    ]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert "CNAG-LS" in completed.stdout
    assert (output_dir / "method_comparison.csv").exists()
    assert (output_dir / "cnag_ls" / "assignments.csv").exists()


def test_source_and_requirements_do_not_reference_package_solvers():
    banned = ["pulp", "cbc", "gurobi", "cplex", "scip", "glpk", "ortools", "cvxpy"]
    paths = list(Path("src").glob("*.py")) + [Path("requirements.txt")]
    text = "\n".join(path.read_text(encoding="utf-8").lower() for path in paths)
    for token in banned:
        assert token not in text
