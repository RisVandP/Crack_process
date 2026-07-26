from __future__ import annotations

import argparse

from .experiments.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch experiments for cracked board processing.")
    parser.add_argument("--config", required=True, help="Experiment JSON config path.")
    parser.add_argument("--output", required=True, help="Experiment output directory.")
    args = parser.parse_args()

    result = run_experiment(args.config, args.output)
    print(
        f"Completed experiment: {len(result['stage1_rows'])} deterministic solutions, "
        f"{len(result['stage2_rows'])} scenario evaluations. Output: {args.output}"
    )


if __name__ == "__main__":
    main()
