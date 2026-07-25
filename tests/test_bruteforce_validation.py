from __future__ import annotations

from src.data_loader import parse_problem
from src.exact_backtracking import solve_exact_backtracking, ExactLimits
from src.greedy_baselines import solve_value_first


def small_raw_instance():
    return {
        "board": {"width": 4.0, "height": 4.0, "m": 2, "n": 2, "base_value": 50.0},
        "deadline": 3.0,
        "devices": [
            {"id": "O1", "type": "ordinary", "speed": 5.0},
            {"id": "P1", "type": "precision", "speed": 4.0},
        ],
        "values": {
            "r_ordinary": 1.0,
            "r_precision": 1.8,
            "default_same_precision_reward": 3.0,
            "default_precision_mismatch_penalty": 1.0,
            "default_cross_crack_loss": 0.0,
            "alpha": 0.2,
            "cross_crack_loss": {"B_1_1|B_1_2": 2.0},
        },
        "cracks": {
            "blocks": {
                "B_1_1": {"C": 1, "CS": 0.2},
                "B_2_2": {"C": 1, "CS": 0.1},
            }
        },
    }


def test_exact_backtracking_runs_on_small_instance():
    data = parse_problem(small_raw_instance())
    solution = solve_exact_backtracking(data, ExactLimits(max_blocks=4, max_nodes=100000))
    assert solution.status == "Optimal"


def test_heuristic_not_above_exact_optimum_on_small_instance():
    data = parse_problem(small_raw_instance())
    exact = solve_exact_backtracking(data, ExactLimits(max_blocks=4, max_nodes=100000))
    heuristic = solve_value_first(data)
    assert exact.status == "Optimal"
    assert heuristic.objective_value <= exact.objective_value + 1e-6


def test_exact_backtracking_limit_is_reported():
    data = parse_problem(small_raw_instance())
    solution = solve_exact_backtracking(data, ExactLimits(max_blocks=1))
    assert solution.status == "not_run"
