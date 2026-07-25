from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from .data_models import BlockId, DeviceId, ProblemData, Solution
from .deterministic_model import processing_time
from .solution_utils import apply_task_copy, device_loads, empty_solution, finalize_solution, is_block_processed


TOL = 1e-9


@dataclass(frozen=True)
class MarginalMove:
    block_id: BlockId
    process_type: str
    device_id: DeviceId
    delta: float
    duration: float
    score: float


def solve_marginal_greedy(data: ProblemData, method_name: str = "MG") -> Solution:
    """动态边际收益贪心。

    每轮重新枚举可行候选，按 DeltaF / 加工时间选择正收益候选。
    """

    start = time.perf_counter()
    solution = empty_solution(data, method_name)
    iterations = 0
    while True:
        move = best_marginal_single_move(data, solution)
        if move is None or move.delta <= TOL:
            break
        solution = apply_task_copy(data, solution, move.block_id, move.process_type, move.device_id)
        iterations += 1
    finalize_solution(data, solution)
    solution.solve_seconds = time.perf_counter() - start
    solution.status = "Feasible"
    solution.metadata = {"construction_iterations": iterations, "stop_reason": "no_positive_marginal"}  # type: ignore[attr-defined]
    return solution


def best_marginal_single_move(data: ProblemData, solution: Solution) -> Optional[MarginalMove]:
    loads = device_loads(data, solution)
    incident_edges = _incident_edges(data)
    best: Optional[MarginalMove] = None
    for block_id in data.block_ids:
        if is_block_processed(solution, block_id):
            continue
        for process_type, device_ids in [("ordinary", data.ordinary_device_ids), ("precision", data.precision_device_ids)]:
            for device_id in device_ids:
                duration = processing_time(data, block_id, device_id)
                if duration <= 0 or duration > data.deadline + TOL:
                    continue
                if loads[device_id] + duration > data.deadline + TOL:
                    continue
                delta = _single_move_delta(data, solution, block_id, process_type, incident_edges[block_id])
                score = delta / duration
                move = MarginalMove(block_id, process_type, device_id, delta, duration, score)
                if _is_better_move(move, best):
                    best = move
    return best


def _incident_edges(data: ProblemData) -> dict[BlockId, list[tuple[BlockId, tuple[BlockId, BlockId]]]]:
    incident = {u: [] for u in data.block_ids}
    for edge in data.edges:
        u, v = edge
        incident[u].append((v, edge))
        incident[v].append((u, edge))
    return incident


def _single_move_delta(data: ProblemData, solution: Solution, block_id: BlockId, process_type: str, incident_edges) -> float:
    area = data.board.area_per_block
    if process_type == "ordinary":
        return area * data.values.r_ordinary
    if process_type != "precision":
        raise ValueError(f"未知加工方式：{process_type}")

    delta = area * data.values.r_precision
    for neighbor, edge in incident_edges:
        old_h = solution.h[edge]
        old_m = solution.m[edge]
        new_h = solution.y_precision[neighbor]
        new_m = int(solution.y_precision[neighbor] == 0)
        delta += data.same_precision_reward[edge] * (new_h - old_h)
        delta -= data.precision_mismatch_penalty[edge] * (new_m - old_m)
    return delta


def _is_better_move(move: MarginalMove, best: Optional[MarginalMove]) -> bool:
    if best is None:
        return True
    key = (-move.score, -move.delta, move.duration, move.block_id, move.device_id, 0 if move.process_type == "ordinary" else 1)
    best_key = (-best.score, -best.delta, best.duration, best.block_id, best.device_id, 0 if best.process_type == "ordinary" else 1)
    return key < best_key
