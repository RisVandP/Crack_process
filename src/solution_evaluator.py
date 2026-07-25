from __future__ import annotations

from typing import Dict, Optional

from .data_models import BlockId, DeviceId, ProblemData, Solution
from .deterministic_model import processing_time


def assigned_device(solution: Solution, block_id: BlockId, device_ids: list[DeviceId]) -> Optional[DeviceId]:
    for device_id in device_ids:
        if solution.x[(block_id, device_id)] == 1:
            return device_id
    return None


def processing_increment(data: ProblemData, solution: Solution, block_id: BlockId) -> float:
    """计算单个木块的加工增值。"""

    area = data.board.area_per_block
    return area * data.values.r_ordinary * solution.y_ordinary[block_id] + area * data.values.r_precision * solution.y_precision[block_id]


def objective_breakdown(data: ProblemData, solution: Solution) -> Dict[str, float]:
    """独立重算目标函数各组成部分。"""

    intrinsic_value = sum(data.board.base_value * block.intrinsic_value_factor for block in data.blocks)
    ordinary_increment = sum(data.board.area_per_block * data.values.r_ordinary * solution.y_ordinary[u] for u in data.block_ids)
    precision_increment = sum(data.board.area_per_block * data.values.r_precision * solution.y_precision[u] for u in data.block_ids)
    same_precision_reward = sum(data.same_precision_reward[e] * solution.h[e] for e in data.edges)
    precision_mismatch_penalty = sum(data.precision_mismatch_penalty[e] * solution.m[e] for e in data.edges)
    cross_crack_loss = sum(data.cross_crack_loss[e] for e in data.edges)
    total = (
        intrinsic_value
        + ordinary_increment
        + precision_increment
        + same_precision_reward
        - precision_mismatch_penalty
        - cross_crack_loss
    )
    return {
        "intrinsic_block_value": intrinsic_value,
        "ordinary_processing_increment": ordinary_increment,
        "precision_processing_increment": precision_increment,
        "same_precision_reward": same_precision_reward,
        "precision_mismatch_penalty": precision_mismatch_penalty,
        "cross_crack_loss": cross_crack_loss,
        "total_objective": total,
    }


def device_usage(data: ProblemData, solution: Solution) -> Dict[DeviceId, Dict[str, float]]:
    usage: Dict[DeviceId, Dict[str, float]] = {}
    for device in data.devices:
        assigned_blocks = [u for u in data.block_ids if solution.x[(u, device.id)] == 1]
        total_time = sum(processing_time(data, u, device.id) for u in assigned_blocks)
        usage[device.id] = {
            "assigned_count": float(len(assigned_blocks)),
            "total_processing_time": total_time,
            "deadline": data.deadline,
            "utilization": total_time / data.deadline if data.deadline > 0 else 0.0,
        }
    return usage
