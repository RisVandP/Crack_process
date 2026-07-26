from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List

from ..model.data_models import BlockId, DeviceId, ProblemData, Solution
from ..model.data_models import processing_time
from .solution import assign_block, empty_solution, finalize_solution, is_block_processed, device_loads


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
    # 按单块加工后的绝对价值排序，构造VF基线方案。
    """单块价值优先基线：按 v_u + A*r_q 排序后依次安排。"""

    return _solve_static_greedy(data, "VF", _value_score)


def _solve_static_greedy(data: ProblemData, method: str, score_fn: Callable[[ProblemData, BlockId, str, DeviceId], float]) -> Solution:
    # 使用给定评分函数对候选任务排序，并在工期约束内贪心分配。
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
    # 枚举所有木块、加工方式和设备组成的可选任务。
    candidates: List[Candidate] = []
    for block_id in data.block_ids:
        for device_id in data.ordinary_device_ids:
            candidates.append(_make_candidate(data, block_id, "ordinary", device_id, score_fn))
        for device_id in data.precision_device_ids:
            candidates.append(_make_candidate(data, block_id, "precision", device_id, score_fn))
    return [item for item in candidates if item.duration <= data.deadline + TOL]


def _make_candidate(data: ProblemData, block_id: BlockId, process_type: str, device_id: DeviceId, score_fn) -> Candidate:
    # 计算单个候选任务的耗时、单块价值和排序分数。
    duration = processing_time(data, block_id, device_id)
    if duration <= 0:
        raise ValueError(f"候选项 {(block_id, process_type, device_id)} 的加工时间非法：{duration}")
    single_value = single_processed_value(data, block_id, process_type)
    return Candidate(block_id, process_type, device_id, score_fn(data, block_id, process_type, device_id), single_value, duration)


def single_processed_value(data: ProblemData, block_id: BlockId, process_type: str) -> float:
    # 计算木块在指定加工方式下的单块基础收益。
    value = data.block_intrinsic_value[block_id]
    if process_type == "ordinary":
        return value + data.board.area_per_block * data.values.r_ordinary
    if process_type == "precision":
        return value + data.board.area_per_block * data.values.r_precision
    raise ValueError(f"未知加工方式：{process_type}")


def _value_score(data: ProblemData, block_id: BlockId, process_type: str, device_id: DeviceId) -> float:
    # 返回VF使用的绝对价值评分。
    return single_processed_value(data, block_id, process_type)


def _density_score(data: ProblemData, block_id: BlockId, process_type: str, device_id: DeviceId) -> float:
    # 返回VDF使用的单位时间价值评分。
    duration = processing_time(data, block_id, device_id)
    if duration <= 0:
        raise ValueError(f"{block_id} 在设备 {device_id} 上的加工时间非法。")
    return single_processed_value(data, block_id, process_type) / duration
