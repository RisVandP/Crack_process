from __future__ import annotations

import csv
import json
from pathlib import Path

from .data_loader import edge_key_to_str
from .deterministic_model import processing_time
from .feasibility_checker import CheckResult
from .solution_evaluator import assigned_device, device_usage, objective_breakdown, processing_increment
from .visualization import plot_device_assignment, plot_device_utilization, plot_processing_status


def write_solution_outputs(data, solution, output_dir: str | Path, filename_prefix: str = "") -> dict:
    """统一输出一个方案的CSV、JSON和图片。

    filename_prefix为空时输出通用文件名。
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    breakdown = objective_breakdown(data, solution)
    write_assignments(data, solution, output_path / f"{filename_prefix}assignments.csv")
    write_device_utilization(data, solution, output_path / "device_utilization.csv")
    write_objective_breakdown(breakdown, output_path / "objective_breakdown.csv")
    write_summary(data, solution, breakdown, [], output_path / "summary.json")
    plot_processing_status(data, solution, output_path / "processing_status_grid.png")
    plot_device_assignment(data, solution, output_path / "device_assignment_grid.png")
    plot_device_utilization(data, solution, output_path / "device_utilization.png")
    return breakdown


def write_assignments(data, solution, path: Path) -> None:
    block_map = data.block_by_id()
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "block_id",
                "i",
                "j",
                "crack_present",
                "crack_severity",
                "processing_status",
                "assigned_device",
                "estimated_processing_time",
                "processing_increment",
            ],
        )
        writer.writeheader()
        for block_id in data.block_ids:
            block = block_map[block_id]
            device_id = assigned_device(solution, block_id, data.device_ids)
            status = "none"
            if solution.y_ordinary[block_id] == 1:
                status = "ordinary"
            elif solution.y_precision[block_id] == 1:
                status = "precision"
            writer.writerow(
                {
                    "block_id": block_id,
                    "i": block.i,
                    "j": block.j,
                    "crack_present": block.crack_present,
                    "crack_severity": block.crack_severity,
                    "processing_status": status,
                    "assigned_device": device_id or "",
                    "estimated_processing_time": processing_time(data, block_id, device_id) if device_id else 0.0,
                    "processing_increment": processing_increment(data, solution, block_id),
                }
            )


def write_device_utilization(data, solution, path: Path) -> None:
    usage = device_usage(data, solution)
    device_map = data.device_by_id()
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["device_id", "device_type", "speed", "assigned_block_count", "total_processing_time", "deadline", "utilization"],
        )
        writer.writeheader()
        for device_id in data.device_ids:
            item = usage[device_id]
            writer.writerow(
                {
                    "device_id": device_id,
                    "device_type": device_map[device_id].device_type,
                    "speed": device_map[device_id].speed,
                    "assigned_block_count": int(item["assigned_count"]),
                    "total_processing_time": item["total_processing_time"],
                    "deadline": item["deadline"],
                    "utilization": item["utilization"],
                }
            )


def write_objective_breakdown(breakdown, path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["component", "value"])
        writer.writeheader()
        for key, value in breakdown.items():
            if key.startswith("_"):
                continue
            writer.writerow({"component": key, "value": value})


def write_summary(data, solution, breakdown, check_messages, path: Path) -> None:
    summary = {
        "model_name": "裂纹板材自研启发式求解方法",
        "method": solution.solver_name,
        "status": solution.status,
        "objective_value": solution.objective_value,
        "solve_seconds": solution.solve_seconds,
        "block_count": len(data.blocks),
        "edge_count": len(data.edges),
        "device_count": len(data.devices),
        "processed_count": sum(solution.y_ordinary[u] + solution.y_precision[u] for u in data.block_ids),
        "ordinary_count": sum(solution.y_ordinary.values()),
        "precision_count": sum(solution.y_precision.values()),
        "unprocessed_count": sum(solution.y0.values()),
        "objective_breakdown": breakdown,
        "feasibility_messages": check_messages,
        "edge_keys": [edge_key_to_str(edge) for edge in data.edges],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def print_run_summary(data, solution, breakdown, output_dir: Path) -> None:
    print("模型名称：裂纹板材自研启发式求解方法")
    print(f"方法：{solution.solver_name}")
    print(f"状态：{solution.status}")
    print(f"目标函数值：{solution.objective_value:.6f}")
    print(f"独立重算总价值：{breakdown['total_objective']:.6f}")
    print(f"运行时间：{solution.solve_seconds:.3f} 秒")
    print(f"完成加工木块数：{sum(solution.y_ordinary[u] + solution.y_precision[u] for u in data.block_ids)}")
    print(f"普通加工木块数：{sum(solution.y_ordinary.values())}")
    print(f"精密加工木块数：{sum(solution.y_precision.values())}")
    print(f"未加工木块数：{sum(solution.y0.values())}")
    print(f"输出目录：{output_dir}")
