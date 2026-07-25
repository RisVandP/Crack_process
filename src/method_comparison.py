from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from .cnag_ls import solve_cnag_ls
from .data_loader import load_problem
from .exact_backtracking import ExactLimits, solve_exact_backtracking
from .feasibility_checker import check_solution
from .greedy_baselines import solve_value_density_first, solve_value_first
from .marginal_greedy import solve_marginal_greedy
from .reporting import write_solution_outputs
from .solution_evaluator import device_usage, objective_breakdown


METHOD_DIRS = {
    "VF": "vf",
    "VDF": "vdf",
    "MG": "mg",
    "CNAG-LS": "cnag_ls",
    "Exact": "exact",
}


def run_method_comparison(data, output_dir: str | Path, include_exact: bool = False) -> List[Dict[str, float | str | bool]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    methods = [
        ("VF", solve_value_first),
        ("VDF", solve_value_density_first),
        ("MG", solve_marginal_greedy),
        ("CNAG-LS", solve_cnag_ls),
    ]
    if include_exact:
        methods.append(("Exact", lambda instance: solve_exact_backtracking(instance, ExactLimits(max_blocks=8))))

    results = []
    for method_name, solve_fn in methods:
        method_dir = output_path / METHOD_DIRS[method_name]
        method_dir.mkdir(parents=True, exist_ok=True)
        solution = solve_fn(data)
        check = check_solution(data, solution)
        breakdown = write_solution_outputs(data, solution, method_dir)
        results.append(_comparison_row(method_name, data, solution, breakdown, check.feasible, check.messages))

    cnag_value = next(row["objective_value"] for row in results if row["method"] == "CNAG-LS")
    exact_rows = [row for row in results if row["method"] == "Exact" and row["status"] == "Optimal"]
    exact_value = exact_rows[0]["objective_value"] if exact_rows else None
    for row in results:
        row["difference_to_cnag_ls_percent"] = _gap(float(cnag_value), float(row["objective_value"]))
        row["gap_to_exact_percent"] = "" if exact_value is None else _gap(float(exact_value), float(row["objective_value"]))

    _write_comparison_csv(results, output_path / "method_comparison.csv")
    _write_comparison_json(results, output_path / "method_comparison.json")
    _plot_method_comparison(results, output_path / "method_comparison.png")
    return results


def _comparison_row(method_name: str, data, solution, breakdown, feasible: bool, messages: List[str]) -> Dict[str, float | str | bool]:
    usage = device_usage(data, solution)
    utilizations = [item["utilization"] for item in usage.values()]
    return {
        "method": method_name,
        "feasible": feasible,
        "status": solution.status,
        "objective_value": breakdown["total_objective"],
        "difference_to_cnag_ls_percent": 0.0,
        "gap_to_exact_percent": "",
        "processed_count": sum(solution.y_ordinary[u] + solution.y_precision[u] for u in data.block_ids),
        "ordinary_count": sum(solution.y_ordinary.values()),
        "precision_count": sum(solution.y_precision.values()),
        "unprocessed_count": sum(solution.y0.values()),
        "average_utilization": sum(utilizations) / len(utilizations) if utilizations else 0.0,
        "max_utilization": max(utilizations) if utilizations else 0.0,
        "device_loads": json.dumps({k: v["total_processing_time"] for k, v in usage.items()}, ensure_ascii=False),
        "run_seconds": solution.solve_seconds,
        "construction_objective": solution.metadata.get("construction_objective", ""),
        "local_search_improvement": solution.metadata.get("local_search_improvement", ""),
        "local_search_iterations": solution.metadata.get("local_search_iterations", ""),
        "intrinsic_block_value": breakdown["intrinsic_block_value"],
        "ordinary_processing_increment": breakdown["ordinary_processing_increment"],
        "precision_processing_increment": breakdown["precision_processing_increment"],
        "same_precision_reward": breakdown["same_precision_reward"],
        "precision_mismatch_penalty": breakdown["precision_mismatch_penalty"],
        "cross_crack_loss": breakdown["cross_crack_loss"],
        "feasibility_messages": "; ".join(messages),
    }


def _gap(reference_value: float, method_value: float) -> float:
    return (reference_value - method_value) / max(abs(reference_value), 1e-9) * 100.0


def _write_comparison_csv(rows: List[Dict[str, float | str | bool]], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_comparison_json(rows: List[Dict[str, float | str | bool]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def _plot_method_comparison(rows: List[Dict[str, float | str | bool]], path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    names = [str(row["method"]) for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    metrics = [
        ("objective_value", "完整目标函数值"),
        ("run_seconds", "运行时间"),
        ("processed_count", "完成加工数量"),
        ("average_utilization", "平均设备利用率"),
    ]
    for ax, (key, title) in zip(axes.ravel(), metrics):
        ax.bar(names, [float(row[key]) for row in rows], color="#457b9d")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="运行四种自研启发式算法的统一比较")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-exact", action="store_true")
    args = parser.parse_args()
    data = load_problem(args.config)
    rows = run_method_comparison(data, args.output, include_exact=args.include_exact)
    for row in rows:
        print(
            f"{row['method']}: status={row['status']}, feasible={row['feasible']}, "
            f"objective={float(row['objective_value']):.6f}, "
            f"diff_to_cnag={float(row['difference_to_cnag_ls_percent']):.3f}%"
        )


if __name__ == "__main__":
    main()
