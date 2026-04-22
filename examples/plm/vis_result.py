"""Visualize partial linear model simulation results."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiment_defs import FUNCTION_LABELS, build_evaluator_from_exp_id, normalize_exp_id

COLOR_BANK = {
    "myred": "#ae1908",
    "myblue": "#05348b",
    "myorange": "#ec813b",
    "mylightblue": "#9acdc4",
    "mypurple": "#743096",
    "myyellow": "#e5a84b",
    "mygreen": "#6bb392",
}

LAMBDA_SWEEP_COLORS = [
    COLOR_BANK["myred"],
    COLOR_BANK["myblue"],
    COLOR_BANK["myorange"],
    COLOR_BANK["mypurple"],
    COLOR_BANK["myyellow"],
    COLOR_BANK["mygreen"],
    COLOR_BANK["mylightblue"],
]


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
    if family_display_id == "1.4":
        grouped_records = _collect_tracking_records_by_lambda(results)
        if len(grouped_records) > 1:
            if display_exp_id == "1.4.4":
                _plot_family_14_dual_source_validation_sweep(
                    display_exp_id=display_exp_id,
                    fig_dir=fig_dir,
                    grouped_records=grouped_records,
                )
                return
            _plot_family_14_lambda_sweep(
                display_exp_id=display_exp_id,
                fig_dir=fig_dir,
                grouped_records=grouped_records,
            )
            return
        _plot_family_14_nuisance_paths(
            display_exp_id=display_exp_id,
            fig_dir=fig_dir,
            results=results,
        )
        return
    if family_display_id in {"1.5", "1.6"}:
        _plot_family_15_pi_complexity(
            display_exp_id=display_exp_id,
            fig_dir=fig_dir,
            evaluator=evaluator,
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
        ("plm_minimax_debias", "beta_hat_mse", "Minimax debias estimate"),
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
        ("plm_minimax_debias", "beta_hat_mse", "Minimax debias beta", COLOR_BANK["mypurple"], "-"),
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


def _plot_family_14_nuisance_paths(
    display_exp_id: str,
    fig_dir: Path,
    results: dict[str, object],
) -> None:
    import matplotlib.pyplot as plt

    tracking_split = _infer_tracking_split(results)
    plt.figure(figsize=(7.5, 4.8))
    mu_labeled = False
    pi_labeled = False
    for trial in results["trial_results"]:
        estimator_records = [
            record
            for record in trial["estimator_results"]
            if "mu_mse_path" in record and "pi_mse_path" in record
        ]
        if not estimator_records:
            continue
        record = estimator_records[0]
        epoch_grid = record.get("epoch_grid", [])
        mu_path = record.get("mu_mse_path", [])
        pi_path = record.get("pi_mse_path", [])
        if not epoch_grid or not mu_path or not pi_path:
            continue
        plt.plot(
            epoch_grid,
            mu_path,
            color=COLOR_BANK["myred"],
            linewidth=1.2,
            alpha=0.5,
            label="mu path" if not mu_labeled else None,
        )
        mu_labeled = True
        plt.plot(
            epoch_grid,
            pi_path,
            color=COLOR_BANK["myblue"],
            linewidth=1.2,
            alpha=0.5,
            label="pi path" if not pi_labeled else None,
        )
        pi_labeled = True

    plt.xlabel("Epoch")
    plt.ylabel(f"Oracle nuisance MSE on {tracking_split}")
    plt.yscale("log")
    plt.title(f"{display_exp_id}: nuisance-learning trajectories")
    plt.legend()
    plt.tight_layout()
    output_path = fig_dir / f"{display_exp_id}_nuisance_mse_paths.png"
    plt.savefig(output_path, dpi=220)
    plt.close()
    print(f"Saved {output_path}")


def _infer_tracking_split(results: dict[str, object]) -> str:
    """Infer the tracking source label from the saved trial records."""
    for trial in results["trial_results"]:
        for record in trial["estimator_results"]:
            if "tracking_split" in record:
                return str(record["tracking_split"])
    return "tracked sample"


def _plot_family_14_validation_paths(
    display_exp_id: str,
    fig_dir: Path,
    results: dict[str, object],
) -> None:
    import matplotlib.pyplot as plt

    epoch_grid = None
    mu_paths = []
    pi_paths = []
    for trial in results["trial_results"]:
        estimator_records = [
            record
            for record in trial["estimator_results"]
            if "mu_mse_path" in record and "pi_mse_path" in record
        ]
        if not estimator_records:
            continue
        record = estimator_records[0]
        if epoch_grid is None:
            epoch_grid = record.get("epoch_grid", [])
        mu_paths.append(np.asarray(record.get("mu_mse_path", []), dtype=float))
        pi_paths.append(np.asarray(record.get("pi_mse_path", []), dtype=float))

    if epoch_grid is None or not mu_paths or not pi_paths:
        raise SystemExit(f"No validation tracking records were found in the results for {display_exp_id}.")

    epoch_grid = np.asarray(epoch_grid, dtype=float)
    mu_array = np.asarray(mu_paths, dtype=float)
    pi_array = np.asarray(pi_paths, dtype=float)
    tracking_split = _infer_tracking_split(results)

    for metric_name, metric_array, color, filename_stem in (
        ("mu", mu_array, COLOR_BANK["myred"], "mu_validation_paths"),
        ("pi", pi_array, COLOR_BANK["myblue"], "pi_validation_paths"),
    ):
        plt.figure(figsize=(7.4, 4.8))
        for path in metric_array:
            plt.plot(epoch_grid, path, color=color, linewidth=1.2, alpha=0.35)
        plt.plot(
            epoch_grid,
            metric_array.mean(axis=0),
            color="black",
            linewidth=2.2,
            label="trial average",
        )
        plt.xlabel("Epoch")
        plt.ylabel(f"Oracle {metric_name} MSE on {tracking_split}")
        plt.yscale("log")
        plt.title(f"{display_exp_id}: validation {metric_name}-learning trajectories")
        plt.legend()
        plt.tight_layout()
        output_path = fig_dir / f"{display_exp_id}_{filename_stem}.png"
        plt.savefig(output_path, dpi=220)
        plt.close()
        print(f"Saved {output_path}")


def _collect_tracking_records_by_lambda(
    results: dict[str, object],
) -> list[tuple[float, str, list[dict[str, object]]]]:
    """Group tracking records by the shared lambda value stored in method_config."""
    grouped: dict[float, list[dict[str, object]]] = defaultdict(list)
    labels: dict[float, str] = {}
    for trial in results["trial_results"]:
        for record in trial["estimator_results"]:
            if (
                ("mu_mse_path" not in record or "pi_mse_path" not in record)
                and "tracking_paths" not in record
            ):
                continue
            method_config = record.get("method_config", {})
            lambda_value = float(method_config.get("lambda_mu"))
            lambda_label = method_config.get("lambda_label", f"{lambda_value:.0e}")
            grouped[lambda_value].append(record)
            labels[lambda_value] = str(lambda_label)
    return [
        (lambda_value, labels[lambda_value], grouped[lambda_value])
        for lambda_value in sorted(grouped)
    ]


def _plot_family_14_dual_source_validation_sweep(
    display_exp_id: str,
    fig_dir: Path,
    grouped_records: list[tuple[float, str, list[dict[str, object]]]],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if not grouped_records:
        raise SystemExit(f"No tracking records were found in the results for {display_exp_id}.")

    source_specs = [
        ("D2", "d2"),
        ("validation", "validation"),
    ]
    for source_label, filename_stem in source_specs:
        fig, axis = plt.subplots(figsize=(8.8, 5.3))
        lambda_handles = []
        positive_values = []
        for idx, (_, lambda_label, records) in enumerate(grouped_records):
            color = LAMBDA_SWEEP_COLORS[idx % len(LAMBDA_SWEEP_COLORS)]
            tracking_paths = [
                record.get("tracking_paths", {}).get(source_label)
                for record in records
                if source_label in record.get("tracking_paths", {})
            ]
            if not tracking_paths:
                continue
            epoch_grid = records[0]["epoch_grid"]
            mu_paths = np.asarray([path["mu_mse_path"] for path in tracking_paths], dtype=float)
            pi_paths = np.asarray([path["pi_mse_path"] for path in tracking_paths], dtype=float)
            mu_mean = mu_paths.mean(axis=0)
            pi_mean = pi_paths.mean(axis=0)
            positive_values.extend(value for value in mu_mean if value > 0.0)
            positive_values.extend(value for value in pi_mean if value > 0.0)
            axis.plot(epoch_grid, mu_mean, color=color, linewidth=2.0)
            axis.plot(epoch_grid, pi_mean, color=color, linewidth=2.0, linestyle="--")
            lambda_handles.append(Line2D([0], [0], color=color, linewidth=2.4, label=rf"$\lambda={lambda_label}$"))

        axis.set_xlabel("Epoch")
        axis.set_ylabel(f"Average oracle nuisance MSE on {source_label}")
        axis.set_yscale("log")
        if positive_values:
            axis.set_ylim(max(min(positive_values) * 0.8, 1e-8), max(positive_values) * 1.1)
        axis.set_title(f"{display_exp_id}: average nuisance-learning paths on {source_label}")
        lambda_legend = axis.legend(handles=lambda_handles, loc="upper right", title=r"$\lambda$")
        axis.add_artist(lambda_legend)
        style_handles = [
            Line2D([0], [0], color="black", linewidth=2.0, linestyle="-", label="mu average"),
            Line2D([0], [0], color="black", linewidth=2.0, linestyle="--", label="pi average"),
        ]
        axis.legend(handles=style_handles, loc="lower left", frameon=False)
        fig.tight_layout()
        output_path = fig_dir / f"{display_exp_id}_{filename_stem}_average_paths.png"
        fig.savefig(output_path, dpi=220)
        plt.close(fig)
        print(f"Saved {output_path}")


def _plot_family_14_lambda_sweep(
    display_exp_id: str,
    fig_dir: Path,
    grouped_records: list[tuple[float, str, list[dict[str, object]]]],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if not grouped_records:
        raise SystemExit(f"No tracking records were found in the results for {display_exp_id}.")

    positive_values = []
    for _, _, records in grouped_records:
        for record in records:
            positive_values.extend(value for value in record.get("mu_mse_path", []) if value > 0.0)
            positive_values.extend(value for value in record.get("pi_mse_path", []) if value > 0.0)
    y_min = max(min(positive_values) * 0.8, 1e-8)
    y_max = max(positive_values) * 1.1

    n_panels = len(grouped_records)
    n_cols = 3
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.1 * n_cols, 3.8 * n_rows), sharex=True, sharey=True)
    flat_axes = np.atleast_1d(axes).flatten()
    max_epoch = max(
        record.get("epoch_grid", [200])[-1]
        for _, _, records in grouped_records
        for record in records
        if record.get("epoch_grid")
    )
    for axis, (_, lambda_label, records) in zip(flat_axes, grouped_records):
        mu_labeled = False
        pi_labeled = False
        for record in records:
            epoch_grid = record.get("epoch_grid", [])
            mu_path = record.get("mu_mse_path", [])
            pi_path = record.get("pi_mse_path", [])
            if not epoch_grid or not mu_path or not pi_path:
                continue
            axis.plot(
                epoch_grid,
                mu_path,
                color=COLOR_BANK["myred"],
                linewidth=1.0,
                alpha=0.45,
                label="mu path" if not mu_labeled else None,
            )
            mu_labeled = True
            axis.plot(
                epoch_grid,
                pi_path,
                color=COLOR_BANK["myblue"],
                linewidth=1.0,
                alpha=0.45,
                label="pi path" if not pi_labeled else None,
            )
            pi_labeled = True
        axis.set_title(rf"$\lambda = {lambda_label}$")
        axis.set_yscale("log")
        axis.set_ylim(y_min, y_max)
        axis.set_xlim(0, max_epoch)
        axis.label_outer()
    for axis in flat_axes[n_panels:]:
        axis.set_visible(False)

    fig.supxlabel("Epoch")
    fig.supylabel("Oracle nuisance MSE on D2")
    legend_handles = [
        Line2D([0], [0], color=COLOR_BANK["myred"], linewidth=1.2, label="mu path"),
        Line2D([0], [0], color=COLOR_BANK["myblue"], linewidth=1.2, label="pi path"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(f"{display_exp_id}: nuisance-learning paths across lambda", y=0.98)
    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.94))
    panel_path = fig_dir / f"{display_exp_id}_lambda_path_panels.png"
    fig.savefig(panel_path, dpi=220)
    plt.close(fig)
    print(f"Saved {panel_path}")

    fig, axis = plt.subplots(figsize=(8.6, 5.2))
    lambda_handles = []
    for idx, (_, lambda_label, records) in enumerate(grouped_records):
        color = LAMBDA_SWEEP_COLORS[idx % len(LAMBDA_SWEEP_COLORS)]
        epoch_grid = records[0]["epoch_grid"]
        mu_paths = np.asarray([record["mu_mse_path"] for record in records], dtype=float)
        pi_paths = np.asarray([record["pi_mse_path"] for record in records], dtype=float)
        mu_mean = mu_paths.mean(axis=0)
        pi_mean = pi_paths.mean(axis=0)
        axis.plot(epoch_grid, mu_mean, color=color, linewidth=2.0)
        axis.plot(epoch_grid, pi_mean, color=color, linewidth=2.0, linestyle="--")
        lambda_handles.append(Line2D([0], [0], color=color, linewidth=2.4, label=rf"$\lambda={lambda_label}$"))

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Average oracle nuisance MSE on D2")
    axis.set_yscale("log")
    axis.set_title(f"{display_exp_id}: average nuisance-learning paths across lambda")
    lambda_legend = axis.legend(handles=lambda_handles, loc="upper right", title=r"$\lambda$")
    axis.add_artist(lambda_legend)
    style_handles = [
        Line2D([0], [0], color="black", linewidth=2.0, linestyle="-", label="mu average"),
        Line2D([0], [0], color="black", linewidth=2.0, linestyle="--", label="pi average"),
    ]
    axis.legend(handles=style_handles, loc="lower left", frameon=False)
    fig.tight_layout()
    average_path = fig_dir / f"{display_exp_id}_lambda_average_paths.png"
    fig.savefig(average_path, dpi=220)
    plt.close(fig)
    print(f"Saved {average_path}")


def _plot_family_15_pi_complexity(
    display_exp_id: str,
    fig_dir: Path,
    evaluator,
) -> None:
    import matplotlib.pyplot as plt

    fixed_dgp_config = {
        key: value
        for key, value in evaluator.dgp_param_grid.items()
        if key not in {"func_pi_name", "n"}
    }
    fixed_dgp_config["n"] = int(evaluator.dgp_param_grid["n"][0])
    pi_specs = [
        (func_pi_name, FUNCTION_LABELS.get(func_pi_name, func_pi_name))
        for func_pi_name in evaluator.dgp_param_grid["func_pi_name"]
    ]
    if display_exp_id in {"1.6.9", "1.6.10", "1.6.11"}:
        _plot_family_169_mse_and_grouped_bias_variance(
            display_exp_id=display_exp_id,
            fig_dir=fig_dir,
            evaluator=evaluator,
            fixed_dgp_config=fixed_dgp_config,
            pi_specs=pi_specs,
        )
        return

    x_values = np.arange(len(pi_specs), dtype=float)
    method_specs = [
        ("dml_nn", "DML AIPW beta", COLOR_BANK["myorange"]),
        ("plm_minimax_debias", "Minimax debias beta", COLOR_BANK["mypurple"]),
    ]
    results = json.loads(evaluator.result_path.read_text())

    def build_decompositions() -> list[tuple[str, dict[str, dict[str, float | int]]]]:
        decompositions = []
        for func_pi_name, label in pi_specs:
            param_config = {
                **fixed_dgp_config,
                "func_pi_name": func_pi_name,
            }
            config_signature = evaluator._config_signature(param_config)
            method_errors = {method_name: [] for method_name, _, _ in method_specs}
            for trial in results["trial_results"]:
                if evaluator._config_signature(trial["dgp_config"]) != config_signature:
                    continue
                beta_true = float(trial["beta_true"])
                for estimator_record in trial["estimator_results"]:
                    method_name = estimator_record["estimator_name"]
                    if method_name not in method_errors:
                        continue
                    method_errors[method_name].append(float(estimator_record["beta_hat"]) - beta_true)

            method_decomposition: dict[str, dict[str, float | int]] = {}
            for method_name, errors in method_errors.items():
                if not errors:
                    continue
                error_values = np.asarray(errors, dtype=float)
                error_mean = float(np.mean(error_values))
                method_decomposition[method_name] = {
                    "beta_hat_mse": float(np.mean(error_values**2)),
                    "beta_hat_bias_sq": error_mean**2,
                    "beta_hat_variance": float(np.mean((error_values - error_mean) ** 2)),
                    "num_trials": int(error_values.size),
                }
            decompositions.append((label, method_decomposition))
        return decompositions

    def plot_decomposition_metric(
        decompositions: list[tuple[str, dict[str, dict[str, float | int]]]],
        *,
        metric_key: str,
        y_label: str,
        title_suffix: str,
        filename_stem: str,
        legacy_alias: bool = False,
    ) -> None:
        plt.figure(figsize=(7.4, 4.8))
        for method_name, label, color in method_specs:
            y_values = []
            valid_x = []
            for idx, (_, method_decomposition) in enumerate(decompositions):
                if method_name not in method_decomposition:
                    continue
                metric_value = max(float(method_decomposition[method_name][metric_key]), 1e-16)
                valid_x.append(x_values[idx])
                y_values.append(metric_value)
            if valid_x:
                plt.plot(
                    valid_x,
                    y_values,
                    color=color,
                    linestyle="-",
                    linewidth=2.4,
                    marker="o",
                    label=label,
                )

        plt.yscale("log")
        plt.xticks(x_values, [label for label, _ in decompositions])
        plt.xlabel(r"Treatment regression $\pi(x)$")
        plt.ylabel(y_label)
        plt.title(f"{display_exp_id}: {title_suffix}")
        plt.legend()
        plt.tight_layout()
        output_path = fig_dir / f"{display_exp_id}_{filename_stem}.png"
        plt.savefig(output_path, dpi=220)
        if legacy_alias:
            legacy_path = fig_dir / f"{display_exp_id}_pi_complexity_mse_comparison.png"
            plt.savefig(legacy_path, dpi=220)
            print(f"Saved {legacy_path}")
        plt.close()
        print(f"Saved {output_path}")

    decompositions = build_decompositions()
    plot_decomposition_metric(
        decompositions,
        metric_key="beta_hat_mse",
        y_label="Mean squared error of beta estimate",
        title_suffix="mean beta MSE across trials",
        filename_stem="pi_complexity_mean_mse_comparison",
        legacy_alias=True,
    )
    plot_decomposition_metric(
        decompositions,
        metric_key="beta_hat_bias_sq",
        y_label=r"Squared bias of beta estimate",
        title_suffix="squared bias of beta estimation error",
        filename_stem="pi_complexity_beta_bias_sq",
    )
    plot_decomposition_metric(
        decompositions,
        metric_key="beta_hat_variance",
        y_label=r"Variance of beta estimation error",
        title_suffix="variance of beta estimation error",
        filename_stem="pi_complexity_beta_variance",
    )
    if display_exp_id in {"1.6.12", "1.6.13"}:
        _plot_family_1612_unified_mean_curves(
            display_exp_id=display_exp_id,
            fig_dir=fig_dir,
            evaluator=evaluator,
            fixed_dgp_config=fixed_dgp_config,
            pi_specs=pi_specs,
        )


def _plot_family_1612_unified_mean_curves(
    *,
    display_exp_id: str,
    fig_dir: Path,
    evaluator,
    fixed_dgp_config: dict,
    pi_specs: list[tuple[str, str]],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    results = json.loads(evaluator.result_path.read_text())
    x_values = np.arange(len(pi_specs), dtype=float)

    dml_beta_mse_means: list[float] = []
    minimax_beta_mse_means: list[float] = []
    mu_mse_means: list[float] = []
    pi_mse_means: list[float] = []
    oracle_beta_mse_values: list[float] = []

    for func_pi_name, _label in pi_specs:
        param_config = {
            **fixed_dgp_config,
            "func_pi_name": func_pi_name,
        }
        config_signature = evaluator._config_signature(param_config)
        matching_trials = [
            trial
            for trial in results["trial_results"]
            if evaluator._config_signature(trial["dgp_config"]) == config_signature
        ]

        dml_values = []
        minimax_values = []
        mu_values = []
        pi_values = []
        for trial in matching_trials:
            for estimator_record in trial["estimator_results"]:
                method_name = estimator_record["estimator_name"]
                if method_name == "oracle_aipw":
                    oracle_beta_mse_values.append(float(estimator_record["beta_sq_error"]))
                elif method_name == "dml_nn":
                    dml_values.append(float(estimator_record["beta_sq_error"]))
                    mu_values.append(float(estimator_record["mu_mse"]))
                    pi_values.append(float(estimator_record["pi_mse"]))
                elif method_name == "plm_minimax_debias":
                    minimax_values.append(float(estimator_record["beta_sq_error"]))

        dml_beta_mse_means.append(float(np.mean(dml_values)))
        minimax_beta_mse_means.append(float(np.mean(minimax_values)))
        mu_mse_means.append(float(np.mean(mu_values)))
        pi_mse_means.append(float(np.mean(pi_values)))

    fig, axis = plt.subplots(figsize=(8.4, 5.1))
    oracle_level = float(np.mean(oracle_beta_mse_values))
    axis.axhline(
        oracle_level,
        color=COLOR_BANK["myred"],
        linestyle="--",
        linewidth=2.2,
        label="Oracle AIPW beta MSE",
    )
    axis.plot(
        x_values,
        dml_beta_mse_means,
        color=COLOR_BANK["mygreen"],
        linestyle="-",
        linewidth=2.6,
        marker="o",
        label="DML beta MSE",
    )
    axis.plot(
        x_values,
        minimax_beta_mse_means,
        color=COLOR_BANK["myorange"],
        linestyle="-",
        linewidth=2.6,
        marker="o",
        label="Minimax debias beta MSE",
    )
    axis.plot(
        x_values,
        mu_mse_means,
        color=COLOR_BANK["myblue"],
        linestyle=":",
        linewidth=2.6,
        marker="o",
        label="DML mu MSE",
    )
    axis.plot(
        x_values,
        pi_mse_means,
        color=COLOR_BANK["mylightblue"],
        linestyle=":",
        linewidth=2.6,
        marker="o",
        label="DML pi MSE",
    )

    axis.set_yscale("log")
    axis.set_xticks(x_values)
    axis.set_xticklabels([label for _, label in pi_specs], rotation=15, ha="right")
    axis.set_xlabel(r"Treatment regression $\pi(x)$")
    axis.set_ylabel("Mean squared error")
    axis.set_title(f"{display_exp_id}: unified nuisance and beta MSE comparison")
    legend_handles = [
        Line2D([0], [0], color=COLOR_BANK["myred"], linestyle="--", linewidth=2.2, label="Oracle AIPW beta MSE"),
        Line2D([0], [0], color=COLOR_BANK["mygreen"], linestyle="-", marker="o", linewidth=2.6, label="DML beta MSE"),
        Line2D([0], [0], color=COLOR_BANK["myorange"], linestyle="-", marker="o", linewidth=2.6, label="Minimax debias beta MSE"),
        Line2D([0], [0], color=COLOR_BANK["myblue"], linestyle=":", marker="o", linewidth=2.6, label="DML mu MSE"),
        Line2D([0], [0], color=COLOR_BANK["mylightblue"], linestyle=":", marker="o", linewidth=2.6, label="DML pi MSE"),
    ]
    axis.legend(handles=legend_handles, loc="upper left")
    fig.tight_layout()
    output_path = fig_dir / f"{display_exp_id}_unified_mse_mean_curve.png"
    fig.savefig(output_path, dpi=220)
    legacy_path = fig_dir / f"{display_exp_id}_unified_mse_boxplot.png"
    fig.savefig(legacy_path, dpi=220)
    plt.close(fig)
    print(f"Saved {output_path}")
    print(f"Saved {legacy_path}")


def _plot_family_169_mse_and_grouped_bias_variance(
    *,
    display_exp_id: str,
    fig_dir: Path,
    evaluator,
    fixed_dgp_config: dict,
    pi_specs: list[tuple[str, str]],
) -> None:
    import matplotlib.pyplot as plt

    x_values = np.arange(len(pi_specs), dtype=float)
    results = json.loads(evaluator.result_path.read_text())
    raw_beta_values = evaluator.dgp_param_grid["beta_values"]
    if isinstance(raw_beta_values, str):
        beta_values = [float(value) for value in raw_beta_values.split(",")]
    else:
        beta_values = [float(value) for value in raw_beta_values]
    beta_method_specs = [
        ("dml_nn", "DML AIPW beta", COLOR_BANK["myorange"]),
        ("plm_minimax_debias", "Minimax debias beta", COLOR_BANK["mypurple"]),
    ]
    mse_line_specs = [
        ("oracle_aipw", "beta_hat_mse", "Oracle AIPW beta", COLOR_BANK["myred"], "-"),
        ("dml_nn", "beta_hat_mse", "DML AIPW beta", COLOR_BANK["myorange"], "-"),
        ("plm_minimax_debias", "beta_hat_mse", "Minimax debias beta", COLOR_BANK["mypurple"], "-"),
        ("dml_nn", "mu_mse", "DML mu", COLOR_BANK["myblue"], "--"),
        ("dml_nn", "pi_mse", "DML pi", COLOR_BANK["mylightblue"], "--"),
    ]

    per_pi_records = []
    for func_pi_name, label in pi_specs:
        param_config = {
            **fixed_dgp_config,
            "func_pi_name": func_pi_name,
        }
        config_signature = evaluator._config_signature(param_config)
        matching_trials = [
            trial
            for trial in results["trial_results"]
            if evaluator._config_signature(trial["dgp_config"]) == config_signature
        ]
        per_pi_records.append((label, matching_trials))

    mse_summary: dict[str, dict[str, list[float]]] = {
        method_name: defaultdict(list) for method_name, _, _, _, _ in mse_line_specs
    }
    grouped_decomposition: dict[str, dict[str, list[float]]] = {
        method_name: {"bias_sq": [], "variance": []}
        for method_name, _, _ in beta_method_specs
    }

    for label, matching_trials in per_pi_records:
        del label
        estimator_records_by_method: dict[str, list[dict]] = defaultdict(list)
        beta_hat_by_method_and_beta = {
            method_name: {beta_value: [] for beta_value in beta_values}
            for method_name, _, _ in beta_method_specs
        }
        for trial in matching_trials:
            beta_true = float(trial["beta_true"])
            for estimator_record in trial["estimator_results"]:
                method_name = estimator_record["estimator_name"]
                estimator_records_by_method[method_name].append(estimator_record)
                if method_name in beta_hat_by_method_and_beta:
                    beta_hat_by_method_and_beta[method_name][beta_true].append(float(estimator_record["beta_hat"]))

        for method_name, metric_key, _, _color, _linestyle in mse_line_specs:
            records = estimator_records_by_method.get(method_name, [])
            if not records:
                continue
            if metric_key == "beta_hat_mse":
                metric_values = [float(record["beta_sq_error"]) for record in records]
            else:
                metric_values = [float(record[metric_key]) for record in records]
            mse_summary[method_name][metric_key].append(float(np.mean(metric_values)))

        for method_name, _, _color in beta_method_specs:
            bias_terms = []
            variance_terms = []
            for beta_value in beta_values:
                beta_hats = np.asarray(beta_hat_by_method_and_beta[method_name][beta_value], dtype=float)
                if beta_hats.size == 0:
                    continue
                bias_terms.append((float(np.mean(beta_hats)) - beta_value) ** 2)
                variance_terms.append(float(np.var(beta_hats)))
            grouped_decomposition[method_name]["bias_sq"].append(float(np.mean(bias_terms)))
            grouped_decomposition[method_name]["variance"].append(float(np.mean(variance_terms)))

    plt.figure(figsize=(7.6, 4.9))
    for method_name, metric_key, label, color, linestyle in mse_line_specs:
        y_values = mse_summary.get(method_name, {}).get(metric_key, [])
        if not y_values:
            continue
        plt.plot(
            x_values[: len(y_values)],
            [max(float(value), 1e-16) for value in y_values],
            color=color,
            linestyle=linestyle,
            linewidth=2.4,
            marker="o",
            label=label,
        )
    plt.yscale("log")
    plt.xticks(x_values, [label for _, label in pi_specs])
    plt.xlabel(r"Treatment regression $\pi(x)$")
    plt.ylabel("Mean MSE across trials")
    plt.title(f"{display_exp_id}: nuisance and beta MSE")
    plt.legend()
    plt.tight_layout()
    mse_path = fig_dir / f"{display_exp_id}_pi_complexity_requested_mse.png"
    plt.savefig(mse_path, dpi=220)
    plt.close()
    print(f"Saved {mse_path}")

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), sharex=True)
    bar_width = 0.36
    offsets = [-bar_width / 2.0, bar_width / 2.0]
    for axis, metric_key, title, ylabel in [
        (axes[0], "bias_sq", "Mean squared bias", "Average squared bias"),
        (axes[1], "variance", "Mean variance", "Average variance"),
    ]:
        for offset, (method_name, label, color) in zip(offsets, beta_method_specs):
            values = grouped_decomposition[method_name][metric_key]
            axis.bar(x_values + offset, values, width=bar_width, color=color, label=label)
        axis.set_yscale("log")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x_values)
        axis.set_xticklabels([label for _, label in pi_specs], rotation=15, ha="right")
        axis.set_xlabel(r"Treatment regression $\pi(x)$")
    axes[0].legend()
    fig.suptitle(f"{display_exp_id}: beta bias-variance decomposition by beta group")
    fig.tight_layout()
    bias_variance_path = fig_dir / f"{display_exp_id}_beta_grouped_bias_variance_hist.png"
    fig.savefig(bias_variance_path, dpi=220)
    plt.close(fig)
    print(f"Saved {bias_variance_path}")


if __name__ == "__main__":
    main()
