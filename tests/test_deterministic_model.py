from __future__ import annotations

from src.data_loader import load_problem
from src.deterministic_model import solve_deterministic
from src.feasibility_checker import check_solution
from src.solution_evaluator import objective_breakdown


def test_example_solution_is_feasible():
    data = load_problem("configs/deterministic_example.json")
    solution = solve_deterministic(data, msg=False)
    check = check_solution(data, solution)
    assert solution.status == "Optimal"
    assert check.feasible, check.messages


def test_edges_have_no_duplicates_or_diagonals():
    data = load_problem("configs/deterministic_example.json")
    assert len(data.edges) == len(set(data.edges))
    pos = {block.id: (block.i, block.j) for block in data.blocks}
    for u, v in data.edges:
        assert abs(pos[u][0] - pos[v][0]) + abs(pos[u][1] - pos[v][1]) == 1


def test_objective_recalculation_matches_solver():
    data = load_problem("configs/deterministic_example.json")
    solution = solve_deterministic(data, msg=False)
    breakdown = objective_breakdown(data, solution)
    assert abs(breakdown["total_objective"] - solution.objective_value) < 1e-5


def test_geometry_cracks_generate_block_and_edge_effects():
    data = load_problem("configs/deterministic_example.json")
    cracked_blocks = [block for block in data.blocks if block.crack_present == 1]
    assert cracked_blocks
    assert all(0.0 <= block.crack_severity <= 1.0 for block in data.blocks)
    assert any(value > 0 for value in data.cross_crack_loss.values())
