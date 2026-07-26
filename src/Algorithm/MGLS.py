from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import combinations, product
from typing import Iterable, Optional

from ..model.data_models import BlockId, DeviceId, ProblemData, Solution
from ..evaluation.solution_evaluator import check_solution
from .solution import (
    apply_task_copy,
    assign_block,
    clone_solution,
    finalize_solution,
    is_block_processed,
    set_unprocessed,
)
from .MG import solve_marginal_greedy


TOL = 1e-9


@dataclass
class MGLSConfig:
    max_local_iterations: int = 80
    max_edge_release: int = 1
    edge_rebuild_block_limit: int = 24


def solve_mgls(data: ProblemData, config: MGLSConfig | None = None) -> Solution:
    # 先运行MG得到初始解，再用局部搜索修正早期贪心决策。
    cfg = config or MGLSConfig()
    start = time.perf_counter()
    solution = solve_marginal_greedy(data)
    solution.solver_name = "MG-LS"
    initial_value = solution.objective_value
    solution, local_iters, success_counts = _local_search(data, solution, cfg)
    finalize_solution(data, solution)
    solution.solve_seconds = time.perf_counter() - start
    solution.solver_name = "MG-LS"
    solution.status = "Feasible"
    solution.metadata.update(
        {
            "construction_method": "MG",
            "construction_objective": initial_value,
            "final_objective": solution.objective_value,
            "local_search_improvement": solution.objective_value - initial_value,
            "local_search_improvement_percent": (solution.objective_value - initial_value) / max(abs(initial_value), 1e-9) * 100.0,
            "local_search_iterations": local_iters,
            "neighborhood_success_counts": success_counts,
            "stop_reason": "no_improving_neighbor" if local_iters < cfg.max_local_iterations else "max_local_iterations",
        }
    )
    return solution


def _local_search(data: ProblemData, solution: Solution, cfg: MGLSConfig) -> tuple[Solution, int, dict[str, int]]:
    # 每轮选择目标值增量最大的可行邻域方案，直到没有正改进。
    success_counts = {name: 0 for name in ["SingleAdjust", "PrecisionExchange", "EdgeRebuild"]}
    current = finalize_solution(data, solution)
    for iteration in range(cfg.max_local_iterations):
        best = _best_neighbor(data, current, cfg)
        if best is None or best.objective_value <= current.objective_value + TOL:
            return current, iteration, success_counts
        move_type = str(best.metadata.get("last_move_type", ""))
        if move_type in success_counts:
            success_counts[move_type] += 1
        current = best
    return current, cfg.max_local_iterations, success_counts


def _best_neighbor(data: ProblemData, solution: Solution, cfg: MGLSConfig) -> Optional[Solution]:
    # 汇总三类邻域候选，并返回完整目标值最高的可行改进。
    best: Optional[Solution] = None
    for neighbors in [
        _neighbors_single_adjust(data, solution),
        _neighbors_precision_exchange(data, solution),
        _neighbors_edge_rebuild(data, solution, cfg),
    ]:
        for cand in neighbors:
            if cand.objective_value <= solution.objective_value + TOL:
                continue
            if not check_solution(data, cand).feasible:
                continue
            if best is None or _is_better_solution(cand, best):
                best = cand
    return best


def _neighbors_single_adjust(data: ProblemData, solution: Solution) -> Iterable[Solution]:
    # 枚举单个木块在不加工、普通加工、精密加工之间的调整。
    for u in data.block_ids:
        if is_block_processed(solution, u):
            yield _tag(_set_block_unprocessed(data, solution, u), "SingleAdjust", f"{u}-0")
        for k in data.ordinary_device_ids:
            if solution.y_ordinary[u] == 1 and solution.x[(u, k)] == 1:
                continue
            yield _tag(apply_task_copy(data, solution, u, "ordinary", k), "SingleAdjust", f"{u}-O-{k}")
        for k in data.precision_device_ids:
            if solution.y_precision[u] == 1 and solution.x[(u, k)] == 1:
                continue
            yield _tag(apply_task_copy(data, solution, u, "precision", k), "SingleAdjust", f"{u}-P-{k}")


def _neighbors_precision_exchange(data: ProblemData, solution: Solution) -> Iterable[Solution]:
    # 枚举“释放一个精加工机会，再交给另一个木块精加工”的交换。
    precision_blocks = [u for u in data.block_ids if solution.y_precision[u] == 1]
    non_precision_blocks = [u for u in data.block_ids if solution.y_precision[u] == 0]
    for old_u in precision_blocks:
        for new_u in non_precision_blocks:
            for old_state, old_device in _released_precision_states(data):
                for new_device in data.precision_device_ids:
                    cand = clone_solution(solution)
                    _apply_state(data, cand, old_u, old_state, old_device)
                    assign_block(data, cand, new_u, "precision", new_device)
                    yield _tag(finalize_solution(data, cand), "PrecisionExchange", f"{old_u}-{old_state}-{new_u}-P-{new_device}")


def _neighbors_edge_rebuild(data: ProblemData, solution: Solution, cfg: MGLSConfig) -> Iterable[Solution]:
    # 枚举相邻边两端同时精加工，并释放少量已有精加工块腾出容量。
    if len(data.blocks) > cfg.edge_rebuild_block_limit:
        return
    precision_blocks = [u for u in data.block_ids if solution.y_precision[u] == 1]
    for edge in data.edges:
        u, v = edge
        releasable = [w for w in precision_blocks if w not in edge]
        for releases in _release_sets(releasable, cfg.max_edge_release):
            for release_states in product(_released_precision_states(data), repeat=len(releases)):
                for ku in data.precision_device_ids:
                    for kv in data.precision_device_ids:
                        cand = clone_solution(solution)
                        for block_id, (state, device_id) in zip(releases, release_states):
                            _apply_state(data, cand, block_id, state, device_id)
                        assign_block(data, cand, u, "precision", ku)
                        assign_block(data, cand, v, "precision", kv)
                        yield _tag(finalize_solution(data, cand), "EdgeRebuild", f"{releases}-{edge}-{ku}-{kv}")


def _release_sets(blocks: list[BlockId], max_size: int) -> Iterable[tuple[BlockId, ...]]:
    # 生成最多释放max_size个精加工木块的组合。
    yield ()
    for size in range(1, max(0, max_size) + 1):
        yield from combinations(blocks, size)


def _released_precision_states(data: ProblemData) -> list[tuple[str, DeviceId | None]]:
    # 生成释放精加工块后的可选状态：不加工或改为普通加工。
    states: list[tuple[str, DeviceId | None]] = [("unprocessed", None)]
    states.extend(("ordinary", k) for k in data.ordinary_device_ids)
    return states


def _set_block_unprocessed(data: ProblemData, solution: Solution, block_id: BlockId) -> Solution:
    # 返回将指定木块改为不加工后的方案副本。
    cand = clone_solution(solution)
    set_unprocessed(data, cand, block_id)
    return finalize_solution(data, cand)


def _apply_state(data: ProblemData, solution: Solution, block_id: BlockId, state: str, device_id: DeviceId | None) -> None:
    # 在候选方案上应用指定木块的新加工状态。
    if state == "unprocessed":
        set_unprocessed(data, solution, block_id)
    elif state == "ordinary" and device_id is not None:
        assign_block(data, solution, block_id, "ordinary", device_id)
    elif state == "precision" and device_id is not None:
        assign_block(data, solution, block_id, "precision", device_id)
    else:
        raise ValueError(f"非法加工状态：{state}, {device_id}")


def _tag(solution: Solution, move_type: str, tie_key: str) -> Solution:
    # 记录邻域类型和稳定排序键，便于分析局部搜索过程。
    solution.metadata["last_move_type"] = move_type
    solution.metadata["tie_key"] = tie_key
    return solution


def _is_better_solution(candidate: Solution, best: Solution) -> bool:
    # 目标值相同时使用稳定字典序，保证重复运行结果一致。
    return (-candidate.objective_value, str(candidate.metadata.get("tie_key", ""))) < (
        -best.objective_value,
        str(best.metadata.get("tie_key", "")),
    )
