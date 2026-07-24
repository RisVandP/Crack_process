from __future__ import annotations

import itertools

from src.data_loader import parse_problem
from src.deterministic_model import processing_time, solve_deterministic


def small_raw_instance():
    return {
        "random_seed": 20260724,
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


def brute_force_best_value(data):
    """枚举每个木块的状态和设备分配，独立寻找全局最优值。"""

    block_ids = data.block_ids
    ordinary_devices = data.ordinary_device_ids
    precision_devices = data.precision_device_ids
    options = []
    for _ in block_ids:
        block_options = [("none", None)]
        block_options += [("ordinary", k) for k in ordinary_devices]
        block_options += [("precision", k) for k in precision_devices]
        options.append(block_options)

    best = None
    area = data.board.area_per_block
    block_map = data.block_by_id()
    for assignment in itertools.product(*options):
        device_time = {k: 0.0 for k in data.device_ids}
        yP = {}
        value = sum(data.board.base_value * block.intrinsic_value_factor for block in data.blocks)
        for block_id, (status, device_id) in zip(block_ids, assignment):
            yP[block_id] = int(status == "precision")
            if device_id is not None:
                device_time[device_id] += processing_time(data, block_id, device_id)
            if status == "ordinary":
                value += area * data.values.r_ordinary
            elif status == "precision":
                value += area * data.values.r_precision
        if any(total > data.deadline + 1e-9 for total in device_time.values()):
            continue
        for edge in data.edges:
            u, v = edge
            h = yP[u] * yP[v]
            m = int(yP[u] != yP[v])
            value += data.same_precision_reward[edge] * h
            value -= data.precision_mismatch_penalty[edge] * m
            value -= data.cross_crack_loss[edge]
        if best is None or value > best:
            best = value
    assert best is not None
    return best


def test_bilp_matches_bruteforce_on_small_instance():
    data = parse_problem(small_raw_instance())
    solution = solve_deterministic(data, msg=False)
    brute_best = brute_force_best_value(data)
    assert solution.status == "Optimal"
    assert abs(solution.objective_value - brute_best) < 1e-5
