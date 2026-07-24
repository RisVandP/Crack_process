from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .data_models import ProblemData, Solution
from .deterministic_model import processing_time
from .solution_evaluator import objective_breakdown


@dataclass
class CheckResult:
    feasible: bool
    messages: List[str]


def check_solution(data: ProblemData, solution: Solution, tolerance: float = 1e-6) -> CheckResult:
    """独立检查方案是否满足确定性BILP的主要约束。"""

    messages: List[str] = []
    K = data.device_ids
    KO = set(data.ordinary_device_ids)
    KP = set(data.precision_device_ids)

    # 检查相邻边：无重复、无斜对角。
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
            messages.append(f"{u} 未恰好处于一种加工状态。")
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
            messages.append(f"h{edge} 不符合逻辑AND。")
        if solution.m[edge] != expected_m:
            messages.append(f"m{edge} 不符合逻辑XOR。")

    recomputed = objective_breakdown(data, solution)["total_objective"]
    if abs(recomputed - solution.objective_value) > 1e-4:
        messages.append(f"目标函数重算不一致：solver={solution.objective_value}, recomputed={recomputed}。")

    return CheckResult(feasible=not messages, messages=messages)
