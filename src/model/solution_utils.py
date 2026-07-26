from __future__ import annotations

import copy

from .data_models import BlockId, DeviceId, ProblemData, Solution
from .data_models import processing_time
from ..evaluation.solution_evaluator import objective_breakdown


def empty_solution(data: ProblemData, method: str, status: str = "Feasible") -> Solution:
    """创建空加工方案：所有木块均不加工。"""

    solution = Solution(
        status=status,
        objective_value=0.0,
        solve_seconds=0.0,
        x={(u, k): 0 for u in data.block_ids for k in data.device_ids},
        y0={u: 1 for u in data.block_ids},
        y_ordinary={u: 0 for u in data.block_ids},
        y_precision={u: 0 for u in data.block_ids},
        h={edge: 0 for edge in data.edges},
        m={edge: 0 for edge in data.edges},
        solver_name=method,
    )
    finalize_solution(data, solution)
    return solution


def clone_solution(solution: Solution) -> Solution:
    return copy.deepcopy(solution)


def device_loads(data: ProblemData, solution: Solution) -> dict[DeviceId, float]:
    return {
        device_id: sum(processing_time(data, u, device_id) * solution.x[(u, device_id)] for u in data.block_ids)
        for device_id in data.device_ids
    }


def is_block_processed(solution: Solution, block_id: BlockId) -> bool:
    return solution.y_ordinary[block_id] == 1 or solution.y_precision[block_id] == 1


def set_unprocessed(data: ProblemData, solution: Solution, block_id: BlockId) -> None:
    for device_id in data.device_ids:
        solution.x[(block_id, device_id)] = 0
    solution.y0[block_id] = 1
    solution.y_ordinary[block_id] = 0
    solution.y_precision[block_id] = 0


def assign_block(data: ProblemData, solution: Solution, block_id: BlockId, process_type: str, device_id: DeviceId) -> None:
    """将木块设置为指定加工方式和设备，不在此处做可行性判断。"""

    set_unprocessed(data, solution, block_id)
    solution.x[(block_id, device_id)] = 1
    solution.y0[block_id] = 0
    if process_type == "ordinary":
        solution.y_ordinary[block_id] = 1
    elif process_type == "precision":
        solution.y_precision[block_id] = 1
    else:
        raise ValueError(f"未知加工方式：{process_type}")


def finalize_solution(data: ProblemData, solution: Solution) -> Solution:
    """根据加工状态更新h、m和完整目标函数值。"""

    for edge in data.edges:
        u, v = edge
        solution.h[edge] = solution.y_precision[u] * solution.y_precision[v]
        solution.m[edge] = int(solution.y_precision[u] != solution.y_precision[v])
    solution.objective_value = objective_breakdown(data, solution)["total_objective"]
    return solution


def apply_task_copy(data: ProblemData, solution: Solution, block_id: BlockId, process_type: str, device_id: DeviceId) -> Solution:
    new_solution = clone_solution(solution)
    assign_block(data, new_solution, block_id, process_type, device_id)
    return finalize_solution(data, new_solution)
