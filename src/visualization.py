from __future__ import annotations

from pathlib import Path

from .data_models import ProblemData, Solution
from .solution_evaluator import assigned_device, device_usage


def _setup_matplotlib():
    """延迟导入matplotlib，避免测试和无图形环境启动时变慢。"""

    import matplotlib.pyplot as plt

    # 常见中文字体优先；如果系统没有这些字体，matplotlib会自动回退。
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def plot_processing_status(data: ProblemData, solution: Solution, output_path: str | Path) -> None:
    """生成木块加工状态网格图。"""

    plt = _setup_matplotlib()
    colors = {0: "#e8e8e8", 1: "#8ecae6", 2: "#ffb703"}
    labels = {0: "不加工", 1: "普通", 2: "精密"}
    fig, ax = plt.subplots(figsize=(1.5 * data.board.m, 1.2 * data.board.n))
    for block in data.blocks:
        state = solution.y_ordinary[block.id] + 2 * solution.y_precision[block.id]
        rect = plt.Rectangle((block.i - 1, data.board.n - block.j), 1, 1, facecolor=colors[state], edgecolor="black")
        ax.add_patch(rect)
        ax.text(block.i - 0.5, data.board.n - block.j + 0.56, block.id, ha="center", va="center", fontsize=9)
        ax.text(block.i - 0.5, data.board.n - block.j + 0.28, labels[state], ha="center", va="center", fontsize=9)
    ax.set_xlim(0, data.board.m)
    ax.set_ylim(0, data.board.n)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("木块加工状态网格图")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_device_assignment(data: ProblemData, solution: Solution, output_path: str | Path) -> None:
    """生成木块-设备分配网格图。"""

    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(1.5 * data.board.m, 1.2 * data.board.n))
    for block in data.blocks:
        device_id = assigned_device(solution, block.id, data.device_ids) or "None"
        rect = plt.Rectangle((block.i - 1, data.board.n - block.j), 1, 1, facecolor="#d8f3dc", edgecolor="black")
        ax.add_patch(rect)
        ax.text(block.i - 0.5, data.board.n - block.j + 0.56, block.id, ha="center", va="center", fontsize=9)
        ax.text(block.i - 0.5, data.board.n - block.j + 0.28, device_id, ha="center", va="center", fontsize=9)
    ax.set_xlim(0, data.board.m)
    ax.set_ylim(0, data.board.n)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("木块-设备分配网格图")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_device_utilization(data: ProblemData, solution: Solution, output_path: str | Path) -> None:
    """生成设备利用率柱状图。"""

    plt = _setup_matplotlib()
    usage = device_usage(data, solution)
    device_ids = data.device_ids
    values = [usage[k]["utilization"] for k in device_ids]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(device_ids, values, color="#457b9d")
    ax.axhline(1.0, color="#d00000", linestyle="--", linewidth=1)
    ax.set_ylim(0, max(1.05, max(values, default=0) * 1.15))
    ax.set_ylabel("利用率")
    ax.set_title("各设备利用率")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
