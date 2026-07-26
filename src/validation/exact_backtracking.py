from __future__ import annotations

import time
from dataclasses import dataclass

from ..model.data_models import ProblemData, Solution
from ..model.data_models import processing_time
from ..model.solution_utils import assign_block, clone_solution, empty_solution, finalize_solution


@dataclass(frozen=True)
class ExactLimits:
    max_blocks: int = 10
    max_nodes: int = 500_000
    time_limit_seconds: float = 10.0


def solve_exact_backtracking(data: ProblemData, limits: ExactLimits | None = None) -> Solution:
    """极小规模精确回溯验证器。

    自行DFS枚举所有木块状态和设备分配，不调用任何包求解器。
    """

    lim = limits or ExactLimits()
    start = time.perf_counter()
    if len(data.blocks) > lim.max_blocks:
        sol = empty_solution(data, "ExactBacktracking", status="not_run")
        sol.metadata["reason"] = f"block_count {len(data.blocks)} exceeds max_blocks {lim.max_blocks}"
        return sol

    options = _options_by_block(data)
    best = empty_solution(data, "ExactBacktracking", status="Optimal")
    current = empty_solution(data, "ExactBacktracking", status="searching")
    nodes = 0
    completed = True

    def dfs(index: int, solution: Solution, loads: dict[str, float]) -> None:
        nonlocal best, nodes, completed
        if nodes >= lim.max_nodes or time.perf_counter() - start > lim.time_limit_seconds:
            completed = False
            return
        nodes += 1
        if index == len(data.block_ids):
            finalize_solution(data, solution)
            if solution.objective_value > best.objective_value:
                best = clone_solution(solution)
                best.status = "Optimal"
            return
        u = data.block_ids[index]
        for process_type, device_id in options[u]:
            if device_id is None:
                nxt = clone_solution(solution)
                dfs(index + 1, nxt, loads.copy())
                if not completed:
                    return
                continue
            duration = processing_time(data, u, device_id)
            if loads[device_id] + duration > data.deadline + 1e-9:
                continue
            nxt = clone_solution(solution)
            assign_block(data, nxt, u, process_type, device_id)
            new_loads = loads.copy()
            new_loads[device_id] += duration
            dfs(index + 1, nxt, new_loads)
            if not completed:
                return

    dfs(0, current, {k: 0.0 for k in data.device_ids})
    finalize_solution(data, best)
    best.solver_name = "ExactBacktracking"
    best.solve_seconds = time.perf_counter() - start
    if not completed:
        best.status = "limit_reached"
    best.metadata.update({"nodes": nodes, "completed": completed})
    return best


def _options_by_block(data: ProblemData):
    options = {}
    for u in data.block_ids:
        opts = [("none", None)]
        opts.extend(("ordinary", k) for k in data.ordinary_device_ids)
        opts.extend(("precision", k) for k in data.precision_device_ids)
        options[u] = opts
    return options
