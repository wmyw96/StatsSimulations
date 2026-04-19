"""Visualize partial linear model simulation results."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiment_defs import build_evaluator_from_exp_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize PLM simulation results.")
    parser.add_argument("--exp_id", required=True, help="Experiment identifier, for example 1.1_1.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluator = build_evaluator_from_exp_id(exp_id=args.exp_id, n_trials=1)
    metric_to_label = {
        "beta_hat_mse": "MSE of AIPW beta estimate",
        "beta_init_mse": "MSE of initial beta estimate",
        "mu_mse": "MSE of mu estimate",
        "pi_mse": "MSE of pi estimate",
    }

    try:
        mpl_config_dir = Path(__file__).resolve().parent / ".mplconfig"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit("matplotlib is required to visualize results.") from error

    fig_dir = Path(__file__).resolve().parent / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    n_values = list(evaluator.dgp_param_grid["n"])

    summaries = {
        n: evaluator.query_results(
            {
                "d": 1,
                "func_mu_name": "sin_2pi_first_coordinate",
                "func_pi_name": "sin_2pi_first_coordinate",
                "beta": 0.0,
                "sigma_u": 0.5,
                "sigma_eps": 0.5,
                "n_test": 10000,
                "n": n,
            },
            mode="summary",
        )
        for n in n_values
    }

    method_names = sorted(
        {
            method_name
            for summary in summaries.values()
            for method_name in summary
        }
    )

    for metric_key, metric_label in metric_to_label.items():
        plt.figure(figsize=(6, 4))
        for method_name in method_names:
            x_values = []
            y_values = []
            for n in n_values:
                summary = summaries[n]
                if method_name not in summary:
                    continue
                x_values.append(n)
                y_values.append(summary[method_name][metric_key])
            if x_values:
                plt.plot(x_values, y_values, marker="o", label=method_name)

        plt.xscale("log", base=2)
        plt.xlabel("n")
        plt.ylabel(metric_label)
        plt.title(f"{args.exp_id}: {metric_label}")
        plt.legend()
        plt.tight_layout()
        output_path = fig_dir / f"{args.exp_id}_{metric_key}.png"
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
