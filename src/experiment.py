from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

from .cnag_ls import solve_cnag_ls
from .crack_geometry import compute_crack_effects
from .data_loader import parse_problem, parse_crack_items
from .data_models import Block, ProblemData, Solution
from .deterministic_model import processing_time
from .feasibility_checker import check_solution
from .greedy_baselines import solve_value_density_first, solve_value_first
from .marginal_greedy import solve_marginal_greedy
from .solution_evaluator import device_usage, objective_breakdown


METHODS: Dict[str, Callable[[ProblemData], Solution]] = {
    "VF": solve_value_first,
    "VDF": solve_value_density_first,
    "MG": solve_marginal_greedy,
    "CNAG-LS": solve_cnag_ls,
}

METHOD_FILE = {
    "VF": "vf",
    "VDF": "vdf",
    "MG": "mg",
    "CNAG-LS": "cnag_ls",
}


def load_experiment_config(path: str | Path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    _validate_experiment_config(cfg)
    return cfg


def run_experiment(config_path: str | Path, output_dir: str | Path) -> dict:
    cfg = load_experiment_config(config_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "solutions").mkdir(parents=True, exist_ok=True)

    stage1_rows: list[dict] = []
    stage2_rows: list[dict] = []
    solutions: dict[tuple[str, str], Solution] = {}

    algorithms = cfg["algorithms"]
    for case in cfg["cases"]:
        case_id = case["id"]
        data = parse_problem(case["deterministic_input"])
        case_solution_dir = out / "solutions" / case_id
        case_solution_dir.mkdir(parents=True, exist_ok=True)

        for method in algorithms:
            solution = METHODS[method](data)
            solutions[(case_id, method)] = solution
            solution_hash = hash_solution(solution)
            check = check_solution(data, solution)
            breakdown = objective_breakdown(data, solution)
            usage = device_usage(data, solution)
            stage1_row = _stage1_row(case, method, data, solution, solution_hash, check.feasible, breakdown, usage)
            stage1_rows.append(stage1_row)
            _write_json(_solution_payload(case, method, solution, solution_hash, breakdown, usage, check.feasible), case_solution_dir / f"{METHOD_FILE[method]}.json")

            for scenario in case["uncertainty_scenarios"]:
                stage2_rows.append(eval_scenario(case, data, solution, solution_hash, stage1_row, scenario))

    method_summary, ranking = summarize_methods(cfg["evaluation_weights"], stage1_rows, stage2_rows)
    _write_rows(stage1_rows, out / "stage1_deterministic_rows.csv")
    _write_json(stage1_rows, out / "stage1_deterministic_rows.json")
    _write_rows(stage2_rows, out / "stage2_scenario_rows.csv")
    _write_json(stage2_rows, out / "stage2_scenario_rows.json")
    _write_rows(method_summary, out / "method_summary.csv")
    _write_json(method_summary, out / "method_summary.json")
    _write_rows(ranking, out / "final_ranking.csv")
    _write_json(ranking, out / "final_ranking.json")
    _plot_outputs(stage1_rows, stage2_rows, ranking, out)
    _write_analysis(cfg, stage1_rows, stage2_rows, ranking, out / "EXPERIMENT_ANALYSIS.md")
    return {"stage1_rows": stage1_rows, "stage2_rows": stage2_rows, "method_summary": method_summary, "final_ranking": ranking}


def eval_scenario(case: dict, data: ProblemData, solution: Solution, solution_hash: str, det_row: dict, scenario: dict) -> dict:
    scenario_data = _scenario_data(data, scenario)
    breakdown = objective_breakdown(scenario_data, solution)
    usage = _scenario_usage(data, scenario_data, solution, scenario)
    loads = [item["load"] for item in usage.values()]
    utils = [item["utilization"] for item in usage.values()]
    unavailable_count = sum(item["unavailable_assignment_count"] for item in usage.values())
    overtime_values = [item["overtime"] for item in usage.values()]
    overtime_device_count = sum(1 for item in usage.values() if item["overtime"] > 1e-9)
    total_overtime = sum(overtime_values)
    feasible = unavailable_count == 0 and overtime_device_count == 0
    deterministic_objective = float(det_row["objective_value"])
    scenario_objective = breakdown["total_objective"]
    objective_change = scenario_objective - deterministic_objective
    value_drop_ratio = max(0.0, deterministic_objective - scenario_objective) / max(abs(deterministic_objective), 1e-9)
    return {
        "case_id": case["id"],
        "case_name": case["name"],
        "scenario_id": scenario["id"],
        "scenario_name": scenario["name"],
        "method": solution.solver_name,
        "solution_hash": solution_hash,
        "scenario_objective_if_completed": scenario_objective,
        "deterministic_objective": deterministic_objective,
        "objective_change": objective_change,
        "value_drop_ratio": value_drop_ratio,
        "feasible_under_scenario": feasible,
        "unavailable_assignment_count": unavailable_count,
        "overtime_device_count": overtime_device_count,
        "total_overtime": total_overtime,
        "max_overtime": max(overtime_values, default=0.0),
        "mean_utilization": sum(utils) / len(utils) if utils else 0.0,
        "max_utilization": max(utils, default=0.0),
        "intrinsic_block_value": breakdown["intrinsic_block_value"],
        "ordinary_processing_increment": breakdown["ordinary_processing_increment"],
        "precision_processing_increment": breakdown["precision_processing_increment"],
        "same_precision_reward": breakdown["same_precision_reward"],
        "precision_mismatch_penalty": breakdown["precision_mismatch_penalty"],
        "cross_crack_loss": breakdown["cross_crack_loss"],
    }


def summarize_methods(weights: dict, stage1_rows: list[dict], stage2_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    methods = sorted({row["method"] for row in stage1_rows})
    det_scores = _deterministic_quality(stage1_rows)
    avg_scores, worst_scores = _scenario_scores(stage2_rows)
    feasibility_scores = {
        method: _mean(1.0 if row["feasible_under_scenario"] else 0.0 for row in stage2_rows if row["method"] == method)
        for method in methods
    }
    runtime_scores = _runtime_scores(stage1_rows)

    summary = []
    for method in methods:
        final = (
            weights["deterministic_quality"] * det_scores[method]
            + weights["average_robustness"] * avg_scores[method]
            + weights["worst_case"] * worst_scores[method]
            + weights["scenario_feasibility"] * feasibility_scores[method]
            + weights["runtime_efficiency"] * runtime_scores[method]
        )
        summary.append(
            {
                "method": method,
                "deterministic_quality_score": det_scores[method],
                "average_robustness_score": avg_scores[method],
                "worst_case_score": worst_scores[method],
                "scenario_feasibility_score": feasibility_scores[method],
                "runtime_efficiency_score": runtime_scores[method],
                "final_score": final,
            }
        )
    ranking = sorted(summary, key=lambda row: (-row["final_score"], row["method"]))
    for idx, row in enumerate(ranking, start=1):
        row["rank"] = idx
    return summary, ranking


def hash_solution(solution: Solution) -> str:
    payload = {
        "x": {f"{u}|{k}": v for (u, k), v in sorted(solution.x.items())},
        "y0": dict(sorted(solution.y0.items())),
        "y_ordinary": dict(sorted(solution.y_ordinary.items())),
        "y_precision": dict(sorted(solution.y_precision.items())),
        "h": {f"{u}|{v}": value for (u, v), value in sorted(solution.h.items())},
        "m": {f"{u}|{v}": value for (u, v), value in sorted(solution.m.items())},
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _stage1_row(case: dict, method: str, data: ProblemData, solution: Solution, solution_hash: str, feasible: bool, breakdown: dict, usage: dict) -> dict:
    utils = [item["utilization"] for item in usage.values()]
    return {
        "case_id": case["id"],
        "case_name": case["name"],
        "difficulty": case["difficulty"],
        "method": method,
        "solution_hash": solution_hash,
        "feasible": feasible,
        "status": solution.status,
        "objective_value": breakdown["total_objective"],
        "objective_decision": breakdown["ordinary_processing_increment"] + breakdown["precision_processing_increment"] + breakdown["same_precision_reward"] - breakdown["precision_mismatch_penalty"],
        "processed_count": sum(solution.y_ordinary[u] + solution.y_precision[u] for u in data.block_ids),
        "ordinary_count": sum(solution.y_ordinary.values()),
        "precision_count": sum(solution.y_precision.values()),
        "unprocessed_count": sum(solution.y0.values()),
        "mean_utilization": sum(utils) / len(utils) if utils else 0.0,
        "max_utilization": max(utils, default=0.0),
        "run_seconds": solution.solve_seconds,
        "intrinsic_block_value": breakdown["intrinsic_block_value"],
        "ordinary_processing_increment": breakdown["ordinary_processing_increment"],
        "precision_processing_increment": breakdown["precision_processing_increment"],
        "same_precision_reward": breakdown["same_precision_reward"],
        "precision_mismatch_penalty": breakdown["precision_mismatch_penalty"],
        "cross_crack_loss": breakdown["cross_crack_loss"],
    }


def _solution_payload(case: dict, method: str, solution: Solution, solution_hash: str, breakdown: dict, usage: dict, feasible: bool) -> dict:
    assignments = []
    for block_id in sorted(solution.y0):
        device_id = ""
        for u, k in sorted(solution.x):
            if u == block_id and solution.x[(u, k)] == 1:
                device_id = k
                break
        state = "unprocessed"
        if solution.y_ordinary[block_id]:
            state = "ordinary"
        elif solution.y_precision[block_id]:
            state = "precision"
        assignments.append({"block_id": block_id, "state": state, "device_id": device_id})
    return {
        "case_id": case["id"],
        "method": method,
        "solution_hash": solution_hash,
        "feasible": feasible,
        "objective_value": solution.objective_value,
        "objective_breakdown": breakdown,
        "device_usage": usage,
        "assignments": assignments,
    }


def _scenario_data(data: ProblemData, scenario: dict) -> ProblemData:
    speed_map = {item["id"]: float(item["speed"]) for item in scenario.get("device_states", [])}
    devices = [replace(device, speed=speed_map.get(device.id, device.speed)) for device in data.devices]
    hidden = parse_crack_items(scenario.get("hidden_cracks", {}).get("items", []), data.board)
    all_cracks = tuple(data.cracks) + tuple(hidden)
    crack_by_block, edge_loss = compute_crack_effects(
        data.board,
        data.edges,
        all_cracks,
        data.crack_epsilon,
        data.crack_r_max,
        data.crack_lambda_b,
    )
    blocks = []
    for block in data.blocks:
        update = crack_by_block.get(block.id, {})
        blocks.append(
            Block(
                id=block.id,
                i=block.i,
                j=block.j,
                crack_present=int(update.get("C", block.crack_present)),
                crack_severity=float(update.get("CS", block.crack_severity)),
            )
        )
    return replace(
        data,
        devices=devices,
        blocks=blocks,
        cross_crack_loss=edge_loss,
        cracks=all_cracks,
    )


def _scenario_usage(original_data: ProblemData, scenario_data: ProblemData, solution: Solution, scenario: dict) -> dict:
    removed = set(scenario.get("removed_devices", []))
    usage = {}
    for device in scenario_data.devices:
        assigned = [u for u in scenario_data.block_ids if solution.x[(u, device.id)] == 1]
        unavailable = len(assigned) if device.id in removed else 0
        if device.id in removed:
            load = 0.0
        else:
            load = sum(processing_time(scenario_data, u, device.id) for u in assigned)
        overtime = max(0.0, load - original_data.deadline)
        usage[device.id] = {
            "assigned_count": len(assigned),
            "load": load,
            "deadline": original_data.deadline,
            "utilization": load / original_data.deadline if original_data.deadline > 0 else 0.0,
            "overtime": overtime,
            "unavailable_assignment_count": unavailable,
        }
    return usage


def _deterministic_quality(rows: list[dict]) -> dict[str, float]:
    scores: dict[str, list[float]] = {}
    for case_id in sorted({row["case_id"] for row in rows}):
        case_rows = [row for row in rows if row["case_id"] == case_id]
        values = [float(row["objective_value"]) for row in case_rows]
        norm = _normalize_higher(values)
        for row, score in zip(case_rows, norm):
            scores.setdefault(row["method"], []).append(score)
    return {method: _mean(vals) for method, vals in scores.items()}


def _scenario_scores(rows: list[dict]) -> tuple[dict[str, float], dict[str, float]]:
    scores: dict[str, list[float]] = {}
    for key in sorted({(row["case_id"], row["scenario_id"]) for row in rows}):
        group = [row for row in rows if (row["case_id"], row["scenario_id"]) == key]
        values = [float(row["scenario_objective_if_completed"]) for row in group]
        norm = _normalize_higher(values)
        for row, score in zip(group, norm):
            scores.setdefault(row["method"], []).append(score)
    avg = {method: _mean(vals) for method, vals in scores.items()}
    worst = {method: min(vals) if vals else 0.0 for method, vals in scores.items()}
    return avg, worst


def _runtime_scores(rows: list[dict]) -> dict[str, float]:
    scores: dict[str, list[float]] = {}
    for case_id in sorted({row["case_id"] for row in rows}):
        case_rows = [row for row in rows if row["case_id"] == case_id]
        values = [float(row["run_seconds"]) for row in case_rows]
        norm = _normalize_lower(values)
        for row, score in zip(case_rows, norm):
            scores.setdefault(row["method"], []).append(score)
    return {method: _mean(vals) for method, vals in scores.items()}


def _normalize_higher(values: list[float]) -> list[float]:
    low, high = min(values), max(values)
    if abs(high - low) < 1e-12:
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _normalize_lower(values: list[float]) -> list[float]:
    low, high = min(values), max(values)
    if abs(high - low) < 1e-12:
        return [1.0 for _ in values]
    return [(high - value) / (high - low) for value in values]


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def _validate_experiment_config(cfg: dict) -> None:
    if cfg.get("schema_version") != "2.0":
        raise ValueError("schema_version必须为2.0。")
    if set(cfg["algorithms"]) != set(METHODS):
        raise ValueError("algorithms必须包含VF、VDF、MG、CNAG-LS。")
    if len(cfg["cases"]) != 12:
        raise ValueError("正式实验必须包含12个确定性案例。")
    weight_sum = sum(float(v) for v in cfg["evaluation_weights"].values())
    if abs(weight_sum - 1.0) > 1e-9:
        raise ValueError("evaluation_weights之和必须为1。")
    case_ids = [case["id"] for case in cfg["cases"]]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("正式实验案例ID不能重复。")
    for case in cfg["cases"]:
        if "random_seed" in case["deterministic_input"]:
            raise ValueError(f"{case['id']} 的正式实验输入不允许使用random_seed。")
        data = parse_problem(case["deterministic_input"])
        scenarios = case.get("uncertainty_scenarios", [])
        if len(scenarios) != 8:
            raise ValueError(f"{case['id']} 必须包含8个不确定情景。")
        scenario_ids = [scenario["id"] for scenario in scenarios]
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError(f"{case['id']} 的情景ID不能重复。")
        device_ids = set(data.device_ids)
        for scenario in scenarios:
            for device_id in scenario.get("removed_devices", []):
                if device_id not in device_ids:
                    raise ValueError(f"{case['id']} {scenario['id']} 的设备 {device_id} 不存在。")
            state_ids = [state["id"] for state in scenario.get("device_states", [])]
            if len(state_ids) != len(set(state_ids)):
                raise ValueError(f"{case['id']} {scenario['id']} 重复描述同一设备。")
            for state in scenario.get("device_states", []):
                if state["id"] not in device_ids:
                    raise ValueError(f"{case['id']} {scenario['id']} 的设备 {state['id']} 不存在。")
                if float(state["speed"]) <= 0:
                    raise ValueError(f"{case['id']} {scenario['id']} 的设备速度必须为正。")
            if "crack_updates" in scenario or "edge_loss_updates" in scenario:
                raise ValueError(f"{case['id']} {scenario['id']} 不允许使用crack_updates或edge_loss_updates。")
            if "hidden_cracks" not in scenario:
                raise ValueError(f"{case['id']} {scenario['id']} 必须包含hidden_cracks。")
            hidden = parse_crack_items(scenario["hidden_cracks"].get("items", []), data.board)
            observed_ids = {crack.id for crack in data.cracks}
            for crack in hidden:
                if crack.id in observed_ids:
                    raise ValueError(f"{case['id']} {scenario['id']} 的隐藏裂纹ID与观测裂纹重复：{crack.id}")
        cracks = case["deterministic_input"].get("cracks", {})
        if cracks.get("mode") != "geometry":
            raise ValueError(f"{case['id']} 正式实验必须使用geometry裂纹模式。")
        if not cracks.get("items"):
            raise ValueError(f"{case['id']} 至少需要一条观测裂纹。")
        for item in cracks.get("items", []):
            for x, y in item["polyline"]:
                if not 0.0 <= float(x) <= data.board.width or not 0.0 <= float(y) <= data.board.height:
                    raise ValueError(f"{case['id']} 裂纹坐标超出板材范围。")


def _write_rows(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(payload: Any, path: Path) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _plot_outputs(stage1_rows: list[dict], stage2_rows: list[dict], ranking: list[dict], out: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "crack_process_mpl"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    methods = sorted({row["method"] for row in stage1_rows})
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(methods, [_mean(float(row["objective_value"]) for row in stage1_rows if row["method"] == method) for method in methods])
    ax.set_title("Mean deterministic objective")
    fig.tight_layout()
    fig.savefig(out / "deterministic_by_case.png", dpi=180)
    plt.close(fig)

    cases = sorted({row["case_id"] for row in stage2_rows})
    heat = [[_mean(float(row["scenario_objective_if_completed"]) for row in stage2_rows if row["case_id"] == case and row["method"] == method) for method in methods] for case in cases]
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(heat, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(methods)), methods)
    ax.set_yticks(range(len(cases)), cases)
    ax.set_title("Mean scenario objective by case")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "scenario_heatmap.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(methods, [_mean(1.0 if row["feasible_under_scenario"] else 0.0 for row in stage2_rows if row["method"] == method) for method in methods])
    ax.set_ylim(0, 1.05)
    ax.set_title("Scenario feasibility rate")
    fig.tight_layout()
    fig.savefig(out / "feasibility_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([row["method"] for row in ranking], [row["final_score"] for row in ranking])
    ax.set_ylim(0, 1.05)
    ax.set_title("Final score comparison")
    fig.tight_layout()
    fig.savefig(out / "final_score_comparison.png", dpi=180)
    plt.close(fig)


def _write_analysis(cfg: dict, stage1_rows: list[dict], stage2_rows: list[dict], ranking: list[dict], path: Path) -> None:
    lines = [
        "# 两阶段实验分析",
        "",
        "本报告由 `src.experiment` 根据实际输出自动生成。",
        "",
        "## 实验设计",
        "",
        "阶段一对12个确定性现实组合分别运行四种算法并保存固定方案；阶段二将同一固定方案放入8个不确定压力情景中评价，不重新求解。",
        "",
        "## 确定性组合",
    ]
    for case in cfg["cases"]:
        lines.append(f"- {case['id']} {case['name']}：{case['description']}")
    lines.extend(["", "## 不确定情景"])
    for scenario in cfg["cases"][0]["uncertainty_scenarios"]:
        lines.append(f"- {scenario['id']} {scenario['name']}：{scenario['description']}")
    lines.extend(["", "## 综合排名"])
    for row in ranking:
        lines.append(f"- 第{row['rank']}名：{row['method']}，final_score={row['final_score']:.4f}，可行率={row['scenario_feasibility_score']:.4f}")
    lines.extend(
        [
            "",
            "## 方法综合表现",
            "",
            "| method | deterministic_quality | average_robustness | worst_case | scenario_feasibility | runtime_efficiency | mean_runtime_s |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in ranking:
        method = row["method"]
        mean_runtime = _mean(float(item["run_seconds"]) for item in stage1_rows if item["method"] == method)
        lines.append(
            f"| {method} | {row['deterministic_quality_score']:.4f} | {row['average_robustness_score']:.4f} | "
            f"{row['worst_case_score']:.4f} | {row['scenario_feasibility_score']:.4f} | "
            f"{row['runtime_efficiency_score']:.4f} | {mean_runtime:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 输出规模",
            "",
            f"- 确定性方案数：{len(stage1_rows)}",
            f"- 情景评价记录数：{len(stage2_rows)}",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="运行裂纹板材两阶段综合实验")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_experiment(args.config, args.output)
    print(
        f"完成实验：{len(result['stage1_rows'])} 个确定性方案，"
        f"{len(result['stage2_rows'])} 条情景评价记录。输出目录：{args.output}"
    )


if __name__ == "__main__":
    main()
