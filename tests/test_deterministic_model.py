from __future__ import annotations

from src.data_loader import load_problem
from src.feasibility_checker import check_solution
from src.greedy_baselines import solve_value_first
from src.solution_evaluator import objective_breakdown


def test_example_solution_is_feasible():
    data = load_problem("configs/deterministic_example.json")
    solution = solve_value_first(data)
    check = check_solution(data, solution)
    assert solution.status == "Feasible"
    assert check.feasible, check.messages


def test_example_config_matches_medium_benchmark_shape():
    data = load_problem("configs/deterministic_example.json")
    cracked_blocks = [block for block in data.blocks if block.crack_present == 1]
    cross_edges = [edge for edge, value in data.cross_crack_loss.items() if value > 0]
    assert data.board.width == 32.0
    assert data.board.height == 18.0
    assert data.board.m == 8
    assert data.board.n == 6
    assert data.board.area_per_block == 12.0
    assert len(data.blocks) == 48
    assert len(data.edges) == 82
    assert 16 <= len(cracked_blocks) <= 20
    assert 10 <= len(cross_edges) <= 16


def test_edges_have_no_duplicates_or_diagonals():
    data = load_problem("configs/deterministic_example.json")
    assert len(data.edges) == len(set(data.edges))
    pos = {block.id: (block.i, block.j) for block in data.blocks}
    for u, v in data.edges:
        assert abs(pos[u][0] - pos[v][0]) + abs(pos[u][1] - pos[v][1]) == 1


def test_objective_recalculation_matches_solver():
    data = load_problem("configs/deterministic_example.json")
    solution = solve_value_first(data)
    breakdown = objective_breakdown(data, solution)
    assert abs(breakdown["total_objective"] - solution.objective_value) < 1e-5


def test_geometry_cracks_generate_block_and_edge_effects():
    data = load_problem("configs/deterministic_example.json")
    cracked_blocks = [block for block in data.blocks if block.crack_present == 1]
    assert cracked_blocks
    assert all(0.0 <= block.crack_severity <= 1.0 for block in data.blocks)
    assert any(value > 0 for value in data.cross_crack_loss.values())
