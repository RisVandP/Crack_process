from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from ..data_models import ProblemData, Solution
from ..deterministic_model import processing_time
from ..feasibility_checker import check_solution
from .MG import best_marginal_single_move, solve_marginal_greedy
from ..solution_utils import (
    apply_task_copy,
    assign_block,
    clone_solution,
    empty_solution,
    finalize_solution,
    is_block_processed,
    set_unprocessed,
)


TOL = 1e-9


@dataclass
class CNAGConfig:
    max_construction_iterations: int = 10_000
    max_local_iterations: int = 80
    max_pair_release: int = 1
    pair_construction_block_limit: int = 30
    pair_release_block_limit: int = 24


@dataclass(frozen=True)
class PairMove:
    edge: tuple[str, str]
    device_u: str
    device_v: str
    delta: float
    duration: float
    score: float


def solve_cnag_ls(data: ProblemData, config: CNAGConfig | None = None) -> Solution:
    # 先构造初始方案，再用多邻域局部搜索改进方案。
    """Crack-and-Neighborhood-Aware Greedy Local Search (CNAG-LS)."""

    cfg = config or CNAGConfig()
    start = time.perf_counter()
    solution, construction_iters, pair_accepts = _construct_with_single_and_pair(data, cfg)
    fallback_used = False
    marginal_solution = solve_marginal_greedy(data)
    if marginal_solution.objective_value > solution.objective_value + TOL:
        solution = marginal_solution
        solution.solver_name = "CNAG-LS"
        fallback_used = True
    initial_value = solution.objective_value
    solution, local_iters, success_counts = _local_search(data, solution, cfg)
    finalize_solution(data, solution)
    solution.solve_seconds = time.perf_counter() - start
    solution.solver_name = "CNAG-LS"
    solution.status = "Feasible"
    solution.metadata.update(
        {
            "construction_objective": initial_value,
            "final_objective": solution.objective_value,
            "local_search_improvement": solution.objective_value - initial_value,
            "local_search_improvement_percent": (solution.objective_value - initial_value) / max(abs(initial_value), 1e-9) * 100.0,
            "construction_iterations": construction_iters,
            "pair_accepts": pair_accepts,
            "marginal_fallback_used": fallback_used,
            "local_search_iterations": local_iters,
            "neighborhood_success_counts": success_counts,
            "stop_reason": "no_improving_neighbor" if local_iters < cfg.max_local_iterations else "max_local_iterations",
        }
    )
    return solution


def _construct_with_single_and_pair(data: ProblemData, cfg: CNAGConfig) -> tuple[Solution, int, int]:
    # 混合使用单块边际插入和相邻成对精加工来构造初始解。
    solution = empty_solution(data, "CNAG-LS")
    pair_accepts = 0
    iterations = 0
    while iterations < cfg.max_construction_iterations:
        single = best_marginal_single_move(data, solution)
        pair = _best_pair_move(data, solution, cfg)
        best_kind = None
        if single is not None and single.delta > TOL:
            best_kind = "single"
        if pair is not None and pair.delta > TOL:
            if best_kind is None or (-pair.score, -pair.delta, pair.edge, pair.device_u, pair.device_v) < (
                -single.score,
                -single.delta,
                (single.block_id, single.block_id),
                single.device_id,
                single.device_id,
            ):
                best_kind = "pair"
        if best_kind is None:
            break
        if best_kind == "single":
            solution = apply_task_copy(data, solution, single.block_id, single.process_type, single.device_id)  # type: ignore[union-attr]
        else:
            solution = _apply_pair_move(data, solution, pair)  # type: ignore[arg-type]
            pair_accepts += 1
        iterations += 1
    return finalize_solution(data, solution), iterations, pair_accepts


def _best_pair_move(data: ProblemData, solution: Solution, cfg: CNAGConfig) -> Optional[PairMove]:
    # 在规模允许时寻找收益密度最高的相邻双精加工动作。
    if len(data.blocks) > cfg.pair_construction_block_limit:
        return None
    current = solution.objective_value
    best: Optional[PairMove] = None
    for edge in data.edges:
        u, v = edge
        if is_block_processed(solution, u) or is_block_processed(solution, v):
            continue
        for ku in data.precision_device_ids:
            for kv in data.precision_device_ids:
                duration_u = processing_time(data, u, ku)
                duration_v = processing_time(data, v, kv)
                if duration_u <= 0 or duration_v <= 0:
                    continue
                candidate = _apply_pair_move(data, solution, PairMove(edge, ku, kv, 0.0, duration_u + duration_v, 0.0))
                check = check_solution(data, candidate)
                if not check.feasible:
                    continue
                delta = candidate.objective_value - current
                duration = duration_u + duration_v
                move = PairMove(edge, ku, kv, delta, duration, delta / duration)
                if _is_better_pair(move, best):
                    best = move
    return best


def _apply_pair_move(data: ProblemData, solution: Solution, move: PairMove) -> Solution:
    # 将一条相邻边两侧木块同时分配为精密加工。
    new_solution = clone_solution(solution)
    u, v = move.edge
    assign_block(data, new_solution, u, "precision", move.device_u)
    assign_block(data, new_solution, v, "precision", move.device_v)
    return finalize_solution(data, new_solution)


def _is_better_pair(move: PairMove, best: Optional[PairMove]) -> bool:
    # 按收益密度、增量和稳定字典序比较成对动作。
    if best is None:
        return True
    return (-move.score, -move.delta, move.duration, move.edge, move.device_u, move.device_v) < (
        -best.score,
        -best.delta,
        best.duration,
        best.edge,
        best.device_u,
        best.device_v,
    )


def _local_search(data: ProblemData, solution: Solution, cfg: CNAGConfig) -> tuple[Solution, int, dict[str, int]]:
    # 反复选择最优邻域改进，直到没有更优方案或达到迭代上限。
    success_counts = {name: 0 for name in ["Add", "Drop", "ModeChange", "Relocate", "Swap", "PairInsert"]}
    current = finalize_solution(data, solution)
    for iteration in range(cfg.max_local_iterations):
        best = _best_neighbor(data, current, cfg)
        if best is None or best.objective_value <= current.objective_value + TOL:
            return current, iteration, success_counts
        move_type = best.metadata.get("last_move_type", "Unknown")
        if move_type in success_counts:
            success_counts[move_type] += 1
        current = best
    return current, cfg.max_local_iterations, success_counts


def _best_neighbor(data: ProblemData, solution: Solution, cfg: CNAGConfig) -> Optional[Solution]:
    # 流式枚举所有邻域候选，并返回目标值最高的可行改进。
    best: Optional[Solution] = None
    for neighbors in [
        _neighbors_add(data, solution),
        _neighbors_drop(data, solution),
        _neighbors_mode_change(data, solution),
        _neighbors_relocate(data, solution),
        _neighbors_swap(data, solution),
        _neighbors_pair_insert(data, solution, cfg),
    ]:
        for cand in neighbors:
            if cand.objective_value <= solution.objective_value + TOL:
                continue
            if not check_solution(data, cand).feasible:
                continue
            if best is None or (-cand.objective_value, str(cand.metadata.get("tie_key", ""))) < (
                -best.objective_value,
                str(best.metadata.get("tie_key", "")),
            ):
                best = cand
    return best


def _tag(solution: Solution, move_type: str, tie_key: str) -> Solution:
    # 给候选方案记录邻域类型和稳定排序键。
    solution.metadata["last_move_type"] = move_type
    solution.metadata["tie_key"] = tie_key
    return solution


def _neighbors_add(data: ProblemData, solution: Solution):
    # 枚举给未加工木块新增一次加工的邻域。
    for u in data.block_ids:
        if is_block_processed(solution, u):
            continue
        for q, device_ids in [("ordinary", data.ordinary_device_ids), ("precision", data.precision_device_ids)]:
            for k in device_ids:
                yield _tag(apply_task_copy(data, solution, u, q, k), "Add", f"{u}-{q}-{k}")


def _neighbors_drop(data: ProblemData, solution: Solution):
    # 枚举将已加工木块改为不加工的邻域。
    for u in data.block_ids:
        if not is_block_processed(solution, u):
            continue
        cand = clone_solution(solution)
        set_unprocessed(data, cand, u)
        yield _tag(finalize_solution(data, cand), "Drop", u)


def _neighbors_mode_change(data: ProblemData, solution: Solution):
    # 枚举普通加工与精密加工之间切换的邻域。
    for u in data.block_ids:
        if solution.y_ordinary[u] == 1:
            for k in data.precision_device_ids:
                yield _tag(apply_task_copy(data, solution, u, "precision", k), "ModeChange", f"{u}-P-{k}")
        elif solution.y_precision[u] == 1:
            for k in data.ordinary_device_ids:
                yield _tag(apply_task_copy(data, solution, u, "ordinary", k), "ModeChange", f"{u}-O-{k}")


def _neighbors_relocate(data: ProblemData, solution: Solution):
    # 枚举在同类设备之间迁移木块的邻域。
    for u in data.block_ids:
        if solution.y_ordinary[u] == 1:
            for k in data.ordinary_device_ids:
                if solution.x[(u, k)] == 0:
                    yield _tag(apply_task_copy(data, solution, u, "ordinary", k), "Relocate", f"{u}-{k}")
        elif solution.y_precision[u] == 1:
            for k in data.precision_device_ids:
                if solution.x[(u, k)] == 0:
                    yield _tag(apply_task_copy(data, solution, u, "precision", k), "Relocate", f"{u}-{k}")


def _neighbors_swap(data: ProblemData, solution: Solution):
    # 枚举释放一个已加工木块并加入一个未加工木块的交换邻域。
    processed = [u for u in data.block_ids if is_block_processed(solution, u)]
    unprocessed = [u for u in data.block_ids if not is_block_processed(solution, u)]
    for drop_u in processed:
        for add_u in unprocessed:
            base = clone_solution(solution)
            set_unprocessed(data, base, drop_u)
            for q, device_ids in [("ordinary", data.ordinary_device_ids), ("precision", data.precision_device_ids)]:
                for k in device_ids:
                    yield _tag(apply_task_copy(data, base, add_u, q, k), "Swap", f"{drop_u}-{add_u}-{q}-{k}")


def _neighbors_pair_insert(data: ProblemData, solution: Solution, cfg: CNAGConfig):
    # 枚举相邻双精加工插入，并可少量释放已有木块腾出工期。
    processed = [u for u in data.block_ids if is_block_processed(solution, u)]
    release_sets = [()]
    if len(data.blocks) <= cfg.pair_release_block_limit:
        release_sets.extend((u,) for u in processed)
        if cfg.max_pair_release >= 2:
            for i, u in enumerate(processed):
                for v in processed[i + 1 :]:
                    release_sets.append((u, v))
    for releases in release_sets:
        base = clone_solution(solution)
        for u in releases:
            set_unprocessed(data, base, u)
        finalize_solution(data, base)
        for edge in data.edges:
            u, v = edge
            if is_block_processed(base, u) or is_block_processed(base, v):
                continue
            for ku in data.precision_device_ids:
                for kv in data.precision_device_ids:
                    cand = _apply_pair_move(data, base, PairMove(edge, ku, kv, 0.0, 0.0, 0.0))
                    yield _tag(cand, "PairInsert", f"{releases}-{edge}-{ku}-{kv}")
