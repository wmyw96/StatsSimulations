"""Visualize partial linear model simulation results."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiment_defs import build_evaluator_from_exp_id, normalize_exp_id

COLOR_BANK = {
    "myred": "#ae1908",
    "myblue": "#05348b",
    "myorange": "#ec813b",
    "mylightblue": "#9acdc4",
    "mypurple": "#743096",
    "myyellow": "#e5a84b",
    "mygreen": "#6bb392",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize PLM simulation results.")
    parser.add_argument("--exp_id", required=True, help="Experiment identifier, for example 1.1.2.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, display_exp_id = normalize_exp_id(args.exp_id)
    family_display_id = display_exp_id.rsplit(".", 1)[0]
    evaluator = build_evaluator_from_exp_id(exp_id=args.exp_id, n_trials=1)
    metric_specs = {
        "beta_hat_mse": {
            "label": "MSE of AIPW beta estimate",
            "filename_stem": "beta_hat_mse",
        },
        "beta_init_mse": {
            "label": "MSE of joint least-squares beta estimate",
            "filename_stem": "beta_joint_lse_mse",
        },
        "mu_mse": {
            "label": "MSE of mu estimate",
            "filename_stem": "mu_mse",
        },
        "pi_mse": {
            "label": "MSE of pi estimate",
            "filename_stem": "pi_mse",
        },
    }

    try:
        mpl_config_dir = Path(__file__).resolve().parent / ".mplconfig"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit("matplotlib is required to visualize results.") from error

    fig_root = Path(__file__).resolve().parent / "figs"
    fig_dir = fig_root / family_display_id
    fig_dir.mkdir(parents=True, exist_ok=True)
    n_values = list(evaluator.dgp_param_grid["n"])
    result_path = evaluator.result_path
    results = json.loads(result_path.read_text())
    fixed_dgp_config = {
        key: value
        for key, value in evaluator.dgp_param_grid.items()
        if key != "n"
    }

    summaries = {
        n: evaluator.query_results(
            {
                **fixed_dgp_config,
                "n": n,
            },
            mode="summary",
        )
        for n in n_values
    }

    if family_display_id == "1.3":
        _plot_family_13_unified(
            display_exp_id=display_exp_id,
            fig_dir=fig_dir,
            n_values=n_values,
            summaries=summaries,
        )
        return

    method_names = sorted(
        {
            method_name
            for summary in summaries.values()
            for method_name in summary
        }
    )

    for metric_key, metric_spec in metric_specs.items():
        metric_label = metric_spec["label"]
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
        output_path = fig_dir / f"{display_exp_id}_{metric_spec['filename_stem']}.png"
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"Saved {output_path}")

    plt.figure(figsize=(6, 4))
    curve_specs = [
        ("dml_nn", "beta_hat_mse", "DML AIPW estimate"),
        ("dml_nn", "beta_init_mse", "NN joint least-squares beta estimate"),
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


def _plot_family_13_unified(
    display_exp_id: str,
    fig_dir: Path,
    n_values: list[int],
    summaries: dict[int, dict[str, dict[str, float | int]]],
) -> None:
    import matplotlib.pyplot as plt

    line_specs = [
        ("oracle_aipw", "beta_hat_mse", "Oracle AIPW", COLOR_BANK["myred"], "-"),
        ("dml_nn", "beta_hat_mse", "DML AIPW", COLOR_BANK["myorange"], "-"),
        ("dml_nn", "beta_init_mse", "NN joint LSE beta", COLOR_BANK["mygreen"], "-"),
        ("dml_nn", "mu_mse", "DML mu", COLOR_BANK["myblue"], "--"),
        ("dml_nn", "pi_mse", "DML pi", COLOR_BANK["mylightblue"], "--"),
    ]

    plt.figure(figsize=(7, 4.5))
    x_values = np.log2(np.asarray(n_values, dtype=float))
    for method_name, metric_key, label, color, linestyle in line_specs:
        y_values = []
        valid_x_values = []
        for n in n_values:
            summary = summaries[n]
            if method_name not in summary:
                continue
            metric_value = float(summary[method_name][metric_key])
            if metric_value <= 0.0:
                continue
            valid_x_values.append(np.log2(float(n)))
            y_values.append(np.log2(metric_value))
        if valid_x_values:
            plt.plot(
                valid_x_values,
                y_values,
                color=color,
                linestyle=linestyle,
                linewidth=2.3,
                label=label,
            )

    plt.xlabel(r"$\log_2(n)$")
    plt.ylabel(r"$\log_2(\mathrm{MSE})$")
    plt.title(f"{display_exp_id}: beta and nuisance scaling")
    plt.xticks(x_values, [str(int(x)) for x in x_values])
    plt.legend()
    plt.tight_layout()
    output_path = fig_dir / f"{display_exp_id}_unified_mse_scaling.png"
    plt.savefig(output_path, dpi=220)
    plt.close()
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
