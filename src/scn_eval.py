from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from .cnag_ls import solve_cnag_ls
from .data_loader import load_problem, parse_edge_key
from .data_models import EdgeKey, ProblemData, Solution
from .greedy_baselines import solve_value_density_first, solve_value_first
from .marginal_greedy import solve_marginal_greedy
from .solution_evaluator import objective_breakdown


ALL_METHODS = {
    "vf": solve_value_first,
    "vdf": solve_value_density_first,
    "mg": solve_marginal_greedy,
    "cnag": solve_cnag_ls,
}

METHODS = {
    **ALL_METHODS,
    "value_first": solve_value_first,
    "value_density_first": solve_value_density_first,
    "marginal_greedy": solve_marginal_greedy,
    "cnag_ls": solve_cnag_ls,
}


@dataclass(frozen=True)
class Scenario:
    id: str
    level: str
    description: str
    removed_devices: List[str]
    speed_multiplier: Dict[str, float]
    crack_updates: Dict[str, Dict[str, float]]
    edge_loss_updates: Dict[EdgeKey, float]


def load_scn_cfg(path: str | Path) -> tuple[ProblemData, List[Scenario]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    data = load_problem(raw["base_config"])
    scenarios = [_parse_scenario(item) for item in raw["scenarios"]]
    if not scenarios:
        raise ValueError("至少需要设置一个情景。")
    return data, scenarios


def eval_solution(data: ProblemData, solution: Solution, scenarios: List[Scenario]) -> dict:
    rows = [eval_one_scn(data, solution, scenario) for scenario in scenarios]
    scenario_values = [row["scenario_objective"] for row in rows]
    best_value = max(scenario_values)
    worst_value = min(row["scenario_objective"] for row in rows)
    average_value = sum(scenario_values) / len(scenario_values)
    return {
        "method": solution.solver_name,
        "deterministic_objective": solution.objective_value,
        "scenario_count": len(rows),
        "average_scenario_value": average_value,
        "best_scenario_value": best_value,
        "worst_value": worst_value,
        "value_range": best_value - worst_value,
        "scenario_rows": rows,
    }


def eval_one_scn(data: ProblemData, solution: Solution, scenario: Scenario) -> dict:
    block_map = data.block_by_id()
    intrinsic_value = 0.0
    for block in data.blocks:
        cs = scenario.crack_updates.get(block.id, {}).get("CS", block.crack_severity)
        intrinsic_value += data.board.base_value * (1.0 - cs)

    base_breakdown = objective_breakdown(data, solution)
    same_reward = base_breakdown["same_precision_reward"]
    mismatch_loss = base_breakdown["precision_mismatch_penalty"]
    ordinary_increment = base_breakdown["ordinary_processing_increment"]
    precision_increment = base_breakdown["precision_processing_increment"]

    cross_loss = 0.0
    for edge in data.edges:
        cross_loss += scenario.edge_loss_updates.get(edge, data.cross_crack_loss[edge])

    capacity_penalty = 0.0
    unavailable_assignments = 0
    overtime_devices = 0
    device_loads = {}
    for device in data.devices:
        removed = device.id in scenario.removed_devices
        eta = float(scenario.speed_multiplier.get(device.id, 1.0))
        if eta <= 0:
            raise ValueError(f"情景 {scenario.id} 中设备 {device.id} 的性能保持系数必须为正。")
        load = 0.0
        for block_id in data.block_ids:
            if solution.x[(block_id, device.id)] != 1:
                continue
            if removed:
                unavailable_assignments += 1
                continue
            load += _scenario_processing_time(data, block_id, device.id, scenario)
        device_loads[device.id] = load
        if removed:
            continue
        if load > data.deadline + 1e-9:
            overtime_devices += 1
            capacity_penalty += (load - data.deadline) * data.board.base_value

    scenario_objective_before_penalty = (
        intrinsic_value
        + ordinary_increment
        + precision_increment
        + same_reward
        - mismatch_loss
        - cross_loss
    )
    # 不可用设备上的任务视为无法完成，给出清晰惩罚；这样同一方案在坏情景下会被扣分。
    unavailable_penalty = unavailable_assignments * data.board.base_value
    scenario_objective = scenario_objective_before_penalty - capacity_penalty - unavailable_penalty
    return {
        "scenario_id": scenario.id,
        "scenario_level": scenario.level,
        "description": scenario.description,
        "intrinsic_block_value": intrinsic_value,
        "ordinary_processing_increment": ordinary_increment,
        "precision_processing_increment": precision_increment,
        "same_precision_reward": same_reward,
        "precision_mismatch_penalty": mismatch_loss,
        "cross_crack_loss": cross_loss,
        "capacity_penalty": capacity_penalty,
        "unavailable_penalty": unavailable_penalty,
        "scenario_objective": scenario_objective,
        "device_loads": json.dumps(device_loads, ensure_ascii=False),
        "overtime_devices": overtime_devices,
        "unavailable_assignments": unavailable_assignments,
    }


def run_scn_eval(config_path: str | Path, output_dir: str | Path, method: str = "all") -> list[dict]:
    data, scenarios = load_scn_cfg(config_path)
    methods = ALL_METHODS if method == "all" else {method: METHODS[method]}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    all_rows = []
    for method_name, solve_fn in methods.items():
        solution = solve_fn(data)
        result = eval_solution(data, solution, scenarios)
        result["method"] = solution.solver_name
        summaries.append({k: v for k, v in result.items() if k != "scenario_rows"})
        for row in result["scenario_rows"]:
            row = {"method": solution.solver_name, **row}
            all_rows.append(row)
    _write_csv(all_rows, output / "scn_rows.csv")
    _write_csv(summaries, output / "scn_sum.csv")
    (output / "scn_sum.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_summary(summaries, output / "scn_plot.png")
    return summaries


def _scenario_processing_time(data: ProblemData, block_id: str, device_id: str, scenario: Scenario) -> float:
    block = data.block_by_id()[block_id]
    device = data.device_by_id()[device_id]
    crack_info = scenario.crack_updates.get(block_id, {})
    c = int(crack_info.get("C", block.crack_present))
    cs = float(crack_info.get("CS", block.crack_severity))
    eta = float(scenario.speed_multiplier.get(device_id, 1.0))
    return data.board.area_per_block / (eta * device.speed) * (1.0 + data.values.alpha * c * cs)


def _parse_scenario(raw: Dict[str, Any]) -> Scenario:
    return Scenario(
        id=str(raw["id"]),
        level=str(raw.get("level", raw["id"])),
        description=str(raw.get("description", "")),
        removed_devices=[str(item) for item in raw.get("removed_devices", [])],
        speed_multiplier={str(k): float(v) for k, v in raw.get("speed_multiplier", {}).items()},
        crack_updates={str(k): {"C": int(v.get("C", 0)), "CS": float(v.get("CS", 0.0))} for k, v in raw.get("crack_updates", {}).items()},
        edge_loss_updates={parse_edge_key(k): float(v) for k, v in raw.get("edge_loss_updates", {}).items()},
    )


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_summary(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    methods = [row["method"] for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, key, title in [
        (axes[0], "average_scenario_value", "各场景平均表现"),
        (axes[1], "worst_value", "最坏场景表现"),
        (axes[2], "value_range", "场景波动范围"),
    ]:
        ax.bar(methods, [float(row[key]) for row in rows], color="#457b9d")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="评估确定性方案在多个梯度场景下的适用表现")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", choices=["vf", "vdf", "mg", "cnag", "value_first", "value_density_first", "marginal_greedy", "cnag_ls", "all"], default="all")
    args = parser.parse_args()
    summaries = run_scn_eval(args.config, args.output, args.method)
    for row in summaries:
        print(
            f"{row['method']}: deterministic={row['deterministic_objective']:.6f}, "
            f"average={row['average_scenario_value']:.6f}, worst={row['worst_value']:.6f}, "
            f"range={row['value_range']:.6f}"
        )


if __name__ == "__main__":
    main()
