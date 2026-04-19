"""Visualize partial linear model simulation results."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiment_defs import build_evaluator_from_exp_id, normalize_exp_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize PLM simulation results.")
    parser.add_argument("--exp_id", required=True, help="Experiment identifier, for example 1.1.2.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, display_exp_id = normalize_exp_id(args.exp_id)
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
    result_path = evaluator.result_path
    results = json.loads(result_path.read_text())

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
        plt.title(f"{display_exp_id}: {metric_label}")
        plt.legend()
        plt.tight_layout()
        output_path = fig_dir / f"{display_exp_id}_{metric_key}.png"
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"Saved {output_path}")

    plt.figure(figsize=(6, 4))
    curve_specs = [
        ("dml_nn", "beta_hat_mse", "DML AIPW estimate"),
        ("dml_nn", "beta_init_mse", "NN initial beta estimate"),
        ("oracle_aipw", "beta_hat_mse", "Oracle AIPW estimate"),
    ]
    for method_name, metric_key, label in curve_specs:
        x_values = []
        y_values = []
        for n in n_values:
            summary = summaries[n]
            if method_name not in summary:
                continue
            x_values.append(n)
            y_values.append(summary[method_name][metric_key])
        if x_values:
            plt.plot(x_values, y_values, marker="o", label=label)

    plt.xscale("log", base=2)
    plt.xlabel("n")
    plt.ylabel("Mean squared error of beta estimate")
    plt.title(f"{display_exp_id}: beta-error comparison")
    plt.legend()
    plt.tight_layout()
    beta_compare_path = fig_dir / f"{display_exp_id}_beta_error_comparison.png"
    plt.savefig(beta_compare_path, dpi=200)
    plt.close()
    print(f"Saved {beta_compare_path}")

    trial_points = defaultdict(list)
    for trial in results["trial_results"]:
        n = int(trial["dgp_config"]["n"])
        for estimator_record in trial["estimator_results"]:
            if estimator_record["estimator_name"] != "dml_nn":
                continue
            trial_points[n].append(
                {
                    "trial_id": int(trial["trial_id"]),
                    "mu_pi_product_mean": float(estimator_record["mu_pi_product_mean"]),
                    "mu_pi_product_true_mean": float(estimator_record["mu_pi_product_true_mean"]),
                    "beta_sq_error": float(estimator_record["beta_sq_error"]),
                }
            )

    plt.figure(figsize=(6, 4))
    for n in sorted(trial_points):
        x_values = [max(point["mu_pi_product_mean"], 1e-16) for point in trial_points[n]]
        y_values = [max(point["beta_sq_error"], 1e-16) for point in trial_points[n]]
        plt.scatter(x_values, y_values, label=f"n={n}", alpha=0.8)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Trial-level mean(mu_hat * pi_hat)")
    plt.ylabel("Trial-level beta_hat squared error")
    plt.title(f"{display_exp_id}: nuisance-product vs beta error (DML)")
    plt.legend()
    plt.tight_layout()
    combined_scatter_path = fig_dir / f"{display_exp_id}_mu_pi_product_mean_vs_beta_hat_scatter.png"
    plt.savefig(combined_scatter_path, dpi=200)
    plt.close()
    print(f"Saved {combined_scatter_path}")

    for n in sorted(trial_points):
        plt.figure(figsize=(6, 4))
        x_values = [max(point["mu_pi_product_mean"], 1e-16) for point in trial_points[n]]
        y_values = [max(point["beta_sq_error"], 1e-16) for point in trial_points[n]]
        plt.scatter(x_values, y_values, alpha=0.85)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Trial-level mean(mu_hat * pi_hat)")
        plt.ylabel("Trial-level beta_hat squared error")
        plt.title(f"{display_exp_id}: nuisance-product vs beta error (n={n})")
        if trial_points[n]:
            reference_level = max(trial_points[n][0]["mu_pi_product_true_mean"], 1e-16)
            plt.axvline(reference_level, color="black", linestyle="--", linewidth=1.0, label="oracle mean(mu*pi)")
            plt.legend()
        plt.tight_layout()
        per_n_scatter_path = fig_dir / f"{display_exp_id}_n{n}_mu_pi_product_mean_vs_beta_hat_scatter.png"
        plt.savefig(per_n_scatter_path, dpi=200)
        plt.close()
        print(f"Saved {per_n_scatter_path}")


if __name__ == "__main__":
    main()
