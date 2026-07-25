from __future__ import annotations

import argparse
from pathlib import Path

from .Algorithm.CNAG_LS import solve_cnag_ls
from .data_loader import load_problem
from .exact_backtracking import solve_exact_backtracking
from .feasibility_checker import check_solution
from .Algorithm.MG import solve_marginal_greedy
from .Algorithm.VDF import solve_value_density_first
from .Algorithm.VF import solve_value_first
from .method_comparison import run_method_comparison
from .reporting import print_run_summary, write_solution_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="裂纹板材自研启发式算法求解")
    parser.add_argument("--config", required=True, help="输入JSON配置文件路径")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument(
        "--method",
        choices=["vf", "vdf", "mg", "cnag", "exact", "all", "value_first", "value_density_first", "marginal_greedy", "cnag_ls"],
        default="all",
        help="运行单个方法或四种启发式比较；vf/vdf/mg/cnag为四种启发式，exact为小规模自编回溯验证",
    )
    args = parser.parse_args()

    data = load_problem(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.method == "all":
        rows = run_method_comparison(data, output_dir)
        for row in rows:
            print(
                f"{row['method']}: status={row['status']}, feasible={row['feasible']}, "
                f"objective={float(row['objective_value']):.6f}, "
                f"diff_to_cnag={float(row['difference_to_cnag_ls_percent']):.3f}%"
            )
        return

    solution = _solve_single_method(data, args.method)
    check = check_solution(data, solution)
    if not check.feasible and solution.status not in {"not_run", "limit_reached"}:
        raise RuntimeError("方案可行性检查失败：\n" + "\n".join(check.messages))
    breakdown = write_solution_outputs(data, solution, output_dir)
    print_run_summary(data, solution, breakdown, output_dir)


def _solve_single_method(data, method: str):
    if method in {"vf", "value_first"}:
        return solve_value_first(data)
    if method in {"vdf", "value_density_first"}:
        return solve_value_density_first(data)
    if method in {"mg", "marginal_greedy"}:
        return solve_marginal_greedy(data)
    if method in {"cnag", "cnag_ls"}:
        return solve_cnag_ls(data)
    if method == "exact":
        return solve_exact_backtracking(data)
    raise ValueError(f"未知方法：{method}")


if __name__ == "__main__":
    main()
