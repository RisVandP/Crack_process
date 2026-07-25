from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List

from .data_models import BlockId, DeviceId, ProblemData, Solution
from .deterministic_model import processing_time
from .solution_utils import assign_block, empty_solution, finalize_solution, is_block_processed, device_loads


TOL = 1e-9


@dataclass(frozen=True)
class Candidate:
    block_id: BlockId
    process_type: str
    device_id: DeviceId
    score: float
    single_value: float
    duration: float


def solve_value_first(data: ProblemData) -> Solution:
    """单块价值优先基线：按 v_u + A*r_q 排序后依次安排。"""

    return _solve_static_greedy(data, "VF", _value_score)


def solve_value_density_first(data: ProblemData) -> Solution:
    """单位加工时间价值优先基线：按 (v_u + A*r_q)/hat_t[u,k] 排序。"""

    return _solve_static_greedy(data, "VDF", _density_score)


def _solve_static_greedy(data: ProblemData, method: str, score_fn: Callable[[ProblemData, BlockId, str, DeviceId], float]) -> Solution:
    start = time.perf_counter()
    solution = empty_solution(data, method)
    loads = device_loads(data, solution)
    candidates = _build_candidates(data, score_fn)
    candidates.sort(
        key=lambda item: (
            -item.score,
            -item.single_value,
            item.duration,
            item.block_id,
            item.device_id,
            0 if item.process_type == "ordinary" else 1,
        )
    )
    for cand in candidates:
        if is_block_processed(solution, cand.block_id):
            continue
        if loads[cand.device_id] + cand.duration > data.deadline + TOL:
            continue
        assign_block(data, solution, cand.block_id, cand.process_type, cand.device_id)
        loads[cand.device_id] += cand.duration
    finalize_solution(data, solution)
    solution.solve_seconds = time.perf_counter() - start
    solution.status = "Feasible"
    return solution


def _build_candidates(data: ProblemData, score_fn: Callable[[ProblemData, BlockId, str, DeviceId], float]) -> List[Candidate]:
    candidates: List[Candidate] = []
    for block_id in data.block_ids:
        for device_id in data.ordinary_device_ids:
            candidates.append(_make_candidate(data, block_id, "ordinary", device_id, score_fn))
        for device_id in data.precision_device_ids:
            candidates.append(_make_candidate(data, block_id, "precision", device_id, score_fn))
    return [item for item in candidates if item.duration <= data.deadline + TOL]


def _make_candidate(data: ProblemData, block_id: BlockId, process_type: str, device_id: DeviceId, score_fn) -> Candidate:
    duration = processing_time(data, block_id, device_id)
    if duration <= 0:
        raise ValueError(f"候选项 {(block_id, process_type, device_id)} 的加工时间非法：{duration}")
    single_value = single_processed_value(data, block_id, process_type)
    return Candidate(block_id, process_type, device_id, score_fn(data, block_id, process_type, device_id), single_value, duration)


def single_processed_value(data: ProblemData, block_id: BlockId, process_type: str) -> float:
    block = data.block_by_id()[block_id]
    value = data.board.base_value * block.intrinsic_value_factor
    if process_type == "ordinary":
        return value + data.board.area_per_block * data.values.r_ordinary
    if process_type == "precision":
        return value + data.board.area_per_block * data.values.r_precision
    raise ValueError(f"未知加工方式：{process_type}")


def _value_score(data: ProblemData, block_id: BlockId, process_type: str, device_id: DeviceId) -> float:
    return single_processed_value(data, block_id, process_type)


def _density_score(data: ProblemData, block_id: BlockId, process_type: str, device_id: DeviceId) -> float:
    duration = processing_time(data, block_id, device_id)
    if duration <= 0:
        raise ValueError(f"{block_id} 在设备 {device_id} 上的加工时间非法。")
    return single_processed_value(data, block_id, process_type) / duration
