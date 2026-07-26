from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..model.data_models import BlockId, DeviceId, ProblemData, Solution, processing_time


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

    intrinsic_value = data.intrinsic_block_value
    ordinary_increment = sum(data.board.area_per_block * data.values.r_ordinary * solution.y_ordinary[u] for u in data.block_ids)
    precision_increment = sum(data.board.area_per_block * data.values.r_precision * solution.y_precision[u] for u in data.block_ids)
    same_precision_reward = sum(data.same_precision_reward[e] * solution.h[e] for e in data.edges)
    precision_mismatch_penalty = sum(data.precision_mismatch_penalty[e] * solution.m[e] for e in data.edges)
    cross_crack_loss = data.cross_crack_loss_total
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


@dataclass
class CheckResult:
    feasible: bool
    messages: List[str]


def check_solution(data: ProblemData, solution: Solution, tolerance: float = 1e-6) -> CheckResult:
    """检查方案是否满足唯一分配、设备兼容、工期和目标值一致性。"""

    messages: List[str] = []
    K = data.device_ids
    KO = set(data.ordinary_device_ids)
    KP = set(data.precision_device_ids)

    if len(set(data.edges)) != len(data.edges):
        messages.append("相邻边集合存在重复边。")
    block_pos = {b.id: (b.i, b.j) for b in data.blocks}
    for u, v in data.edges:
        dist = abs(block_pos[u][0] - block_pos[v][0]) + abs(block_pos[u][1] - block_pos[v][1])
        if dist != 1:
            messages.append(f"边 {(u, v)} 不是上下左右相邻。")

    for u in data.block_ids:
        assigned = sum(solution.x[(u, k)] for k in K)
        if assigned > 1:
            messages.append(f"{u} 被分配给超过一台设备。")
        if solution.y0[u] + solution.y_ordinary[u] + solution.y_precision[u] != 1:
            messages.append(f"{u} 未处于唯一加工状态。")
        if sum(solution.x[(u, k)] for k in KO) != solution.y_ordinary[u]:
            messages.append(f"{u} 普通加工状态与普通设备分配不一致。")
        if sum(solution.x[(u, k)] for k in KP) != solution.y_precision[u]:
            messages.append(f"{u} 精密加工状态与精密设备分配不一致。")
        if solution.y0[u] == 1 and assigned != 0:
            messages.append(f"{u} 不加工但仍有设备分配。")

    for device_id in K:
        total_time = sum(processing_time(data, u, device_id) * solution.x[(u, device_id)] for u in data.block_ids)
        if total_time > data.deadline + tolerance:
            messages.append(f"设备 {device_id} 超过共同工期：{total_time} > {data.deadline}。")

    for edge in data.edges:
        u, v = edge
        expected_h = solution.y_precision[u] * solution.y_precision[v]
        expected_m = int(solution.y_precision[u] != solution.y_precision[v])
        if solution.h[edge] != expected_h:
            messages.append(f"h{edge} 不符合AND逻辑。")
        if solution.m[edge] != expected_m:
            messages.append(f"m{edge} 不符合XOR逻辑。")

    recomputed = objective_breakdown(data, solution)["total_objective"]
    if abs(recomputed - solution.objective_value) > 1e-4:
        messages.append(f"目标函数重算不一致：solver={solution.objective_value}, recomputed={recomputed}。")

    return CheckResult(feasible=not messages, messages=messages)
