from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from itertools import product
from pathlib import Path

from .cnag_ls import CNAGConfig, solve_cnag_ls
from .data_loader import parse_problem
from .greedy_baselines import solve_value_density_first, solve_value_first
from .marginal_greedy import solve_marginal_greedy
from .solution_evaluator import device_usage, objective_breakdown


METHODS = [
    ("VF", solve_value_first),
    ("VDF", solve_value_density_first),
    ("MG", solve_marginal_greedy),
    ("CNAG-LS", lambda data: solve_cnag_ls(data, CNAGConfig(max_local_iterations=20))),
]


def run_grid_exp(config_path: str | Path, output_dir: str | Path, limit: int | None = None):
    """运行多维参数网格实验，用于分析算法适用范围。"""

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    base_seed = int(config.get("random_seed", 20260724))
    reps = int(config.get("reps", 1))
    combos = list(
        product(
            config["boards"].items(),
            config["time"].items(),
            config["cracks"].items(),
            config["adj"].items(),
            config["devs"].items(),
            range(reps),
        )
    )
    max_cases = limit if limit is not None else config.get("max_cases")
    if max_cases is not None:
        combos = _pick_cases(combos, int(max_cases))

    for case_id, combo in enumerate(combos, start=1):
        (board_name, board), (time_name, time_factor), (crack_name, crack_cfg), (adj_name, adj), (dev_name, dev_cfg), rep = combo
        seed = base_seed + case_id * 17 + rep
        data = parse_problem(_make_instance(seed, board, time_factor, crack_cfg, adj, dev_cfg))
        cnag_value = None
        method_rows = []
        for method_name, solve in METHODS:
            solution = solve(data)
            breakdown = objective_breakdown(data, solution)
            usage = device_usage(data, solution)
            utils = [item["utilization"] for item in usage.values()]
            row = {
                "case_id": case_id,
                "seed": seed,
                "board": board_name,
                "time": time_name,
                "crack": crack_name,
                "adj": adj_name,
                "dev": dev_name,
                "method": method_name,
                "objective": breakdown["total_objective"],
                "runtime": solution.solve_seconds,
                "processed_count": sum(solution.y_ordinary[u] + solution.y_precision[u] for u in data.block_ids),
                "average_utilization": sum(utils) / len(utils),
                "same_precision_reward": breakdown["same_precision_reward"],
                "precision_mismatch_penalty": breakdown["precision_mismatch_penalty"],
            }
            method_rows.append(row)
            if method_name == "CNAG-LS":
                cnag_value = row["objective"]
        for row in method_rows:
            row["cnag_gap_pct"] = (cnag_value - row["objective"]) / max(abs(row["objective"]), 1e-9) * 100.0
            rows.append(row)

    summary = _summarize(rows)
    _write_csv(rows, output / "grid_rows.csv")
    _write_csv(summary, output / "grid_sum.csv")
    (output / "grid_sum.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_grid(rows, output)
    _write_markdown(summary, output / "GRID_ANALYSIS.md")
    return rows, summary


def _pick_cases(combos, max_cases):
    """从完整组合中均匀选取案例，避免只覆盖配置开头的少数梯度。"""

    if max_cases <= 0 or max_cases >= len(combos):
        return combos
    if max_cases == 1:
        return [combos[0]]
    step = (len(combos) - 1) / (max_cases - 1)
    indexes = sorted({round(i * step) for i in range(max_cases)})
    return [combos[i] for i in indexes]


def _make_instance(seed, board_cfg, time_factor, crack_cfg, adj, dev_cfg):
    rng = random.Random(seed)
    m, n = int(board_cfg["m"]), int(board_cfg["n"])
    blocks = _make_cracks(rng, m, n, crack_cfg)
    width, height = float(board_cfg["width"]), float(board_cfg["height"])
    area = width * height / (m * n)
    devices = _make_devices(dev_cfg)
    avg_speed = sum(item["speed"] for item in devices) / len(devices)
    deadline = max(area / avg_speed * m * n / len(devices) * float(time_factor), area / max(item["speed"] for item in devices))
    return {
        "random_seed": seed,
        "board": {
            "width": width,
            "height": height,
            "m": m,
            "n": n,
            "base_value": float(board_cfg["base_value"]),
        },
        "deadline": deadline,
        "devices": devices,
        "values": {
            "r_ordinary": 1.2,
            "r_precision": 2.4,
            "default_same_precision_reward": float(adj["same"]),
            "default_precision_mismatch_penalty": float(adj["diff"]),
            "default_cross_crack_loss": float(adj["cross"]),
            "alpha": 0.55,
        },
        "cracks": {"mode": "direct", "blocks": blocks},
    }


def _make_cracks(rng, m, n, crack_cfg):
    """按不同裂缝分布生成块内裂缝，体现随机、聚集、带状三类差异。"""

    blocks = {}
    rate = float(crack_cfg["rate"])
    max_cs = float(crack_cfg["max_cs"])
    pattern = crack_cfg.get("pattern", "random")
    center_i = rng.randint(max(1, m // 3), max(1, m - m // 3))
    center_j = rng.randint(max(1, n // 3), max(1, n - n // 3))
    band_col = rng.randint(1, m)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            p = rate
            if pattern == "cluster":
                dist = abs(i - center_i) + abs(j - center_j)
                p = min(0.90, rate * (2.5 if dist <= 1 else 0.7))
            elif pattern == "band":
                p = min(0.95, rate * (2.2 if abs(i - band_col) <= 1 else 0.45))
            if rng.random() < p:
                blocks[f"B_{i}_{j}"] = {"C": 1, "CS": round(rng.uniform(0.08, max_cs), 3)}
    return blocks


def _make_devices(dev_cfg):
    devices = []
    for idx, speed in enumerate(dev_cfg["ordinary"], start=1):
        devices.append({"id": f"O{idx}", "type": "ordinary", "speed": float(speed)})
    for idx, speed in enumerate(dev_cfg["precision"], start=1):
        devices.append({"id": f"P{idx}", "type": "precision", "speed": float(speed)})
    return devices


def _summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (row["board"], row["time"], row["crack"], row["adj"], row["dev"], row["method"])
        groups[key].append(row)
    summary = []
    for key, items in groups.items():
        summary.append(
            {
                "board": key[0],
                "time": key[1],
                "crack": key[2],
                "adj": key[3],
                "dev": key[4],
                "method": key[5],
                "mean_objective": sum(x["objective"] for x in items) / len(items),
                "mean_runtime": sum(x["runtime"] for x in items) / len(items),
                "mean_processed_count": sum(x["processed_count"] for x in items) / len(items),
                "mean_average_utilization": sum(x["average_utilization"] for x in items) / len(items),
                "mean_cnag_gap_pct": sum(x["cnag_gap_pct"] for x in items) / len(items),
            }
        )
    return summary


def _write_csv(rows, path):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_grid(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    methods = [name for name, _ in METHODS]
    avg_obj = [sum(r["objective"] for r in rows if r["method"] == m) / max(1, sum(1 for r in rows if r["method"] == m)) for m in methods]
    avg_rt = [sum(r["runtime"] for r in rows if r["method"] == m) / max(1, sum(1 for r in rows if r["method"] == m)) for m in methods]
    avg_gap = [sum(r["cnag_gap_pct"] for r in rows if r["method"] == m) / max(1, sum(1 for r in rows if r["method"] == m)) for m in methods]
    for values, filename, title in [
        (avg_obj, "obj_by_alg.png", "平均目标值"),
        (avg_rt, "rt_by_alg.png", "平均运行时间"),
        (avg_gap, "gap_by_alg.png", "相对CNAG-LS差距"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(methods, values, color="#457b9d")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)


def _write_markdown(summary, path):
    method_rows = defaultdict(list)
    for row in summary:
        method_rows[row["method"]].append(row)

    def avg(method, key):
        rows = method_rows.get(method, [])
        return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0

    cnag_obj = avg("CNAG-LS", "mean_objective")
    marginal_obj = avg("MG", "mean_objective")
    vf_obj = avg("VF", "mean_objective")
    vdf_obj = avg("VDF", "mean_objective")
    cnag_rt = avg("CNAG-LS", "mean_runtime")
    marginal_rt = avg("MG", "mean_runtime")
    vf_rt = avg("VF", "mean_runtime")
    vdf_rt = avg("VDF", "mean_runtime")
    text = [
        "# 参数网格实验分析",
        "",
        "本文件由 `src.exp_grid` 根据实际运行结果生成。",
        "",
        "## 主要观察",
        "",
        f"- 平均目标值：CNAG-LS={cnag_obj:.2f}，MG={marginal_obj:.2f}，VF={vf_obj:.2f}，VDF={vdf_obj:.2f}。",
        f"- 平均运行时间：CNAG-LS={cnag_rt:.4f}s，MG={marginal_rt:.4f}s，VF={vf_rt:.4f}s，VDF={vdf_rt:.4f}s。",
        "- 简单基线排序阶段不考虑相邻组合项，因此在强相邻奖励/惩罚场景下可能损失更多。",
        "- CNAG-LS 运行时间高于简单基线，换取动态边际收益、成对候选和局部搜索带来的改进机会。",
    ]
    text.extend(
        [
            "",
            "## 当前限制",
            "",
            "- 场景生成使用规则网格和直接裂纹严重程度输入，未涉及图像识别。",
            "- CNAG-LS仍是确定性局部搜索，可能陷入局部最优。",
            "- 大规模实例下可进一步加入候选剪枝、增量缓存或并行边际收益计算。",
        ]
    )
    path.write_text("\n".join(text), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="运行裂纹板材多维参数网格实验")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None, help="最多运行多少组参数案例；不填则使用配置文件中的max_cases")
    args = parser.parse_args()
    rows, summary = run_grid_exp(args.config, args.output, limit=args.limit)
    print(f"完成网格实验：{len(rows)} 条方法结果，{len(summary)} 条汇总结果。输出目录：{args.output}")


if __name__ == "__main__":
    main()
