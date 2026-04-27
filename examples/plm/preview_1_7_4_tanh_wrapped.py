"""Preview the tanh-wrapped 1.7.4 Fourier family f_r(x) = g_r(tanh(x))."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

R_VALUES = (1, 2, 4, 8)
NUM_TERMS = 10
COLOR_BANK = {
    "myred": "#ae1908",
    "myblue": "#05348b",
    "myorange": "#ec813b",
    "mypurple": "#743096",
}
LINE_COLORS = [
    COLOR_BANK["myred"],
    COLOR_BANK["myblue"],
    COLOR_BANK["myorange"],
    COLOR_BANK["mypurple"],
]
DOMAIN_RADIUS = math.sqrt(5.0)


def evaluate_g_r(z: np.ndarray, *, r: float, num_terms: int = NUM_TERMS) -> np.ndarray:
    """Evaluate the normalized Fourier family with a_k = 1 and b_k = (-1)^k."""
    z_grid = np.asarray(z, dtype=float).reshape(-1)
    if r <= 0:
        raise ValueError("r must be positive.")
    if num_terms <= 0:
        raise ValueError("num_terms must be positive.")

    k = np.arange(1, num_terms + 1, dtype=float)
    weights = k ** (-1.0 / float(r))
    a_coeffs = np.ones_like(k)
    b_coeffs = (-1.0) ** k
    denominator = np.sqrt(np.sum((a_coeffs**2 + b_coeffs**2) * (weights**2)))
    if denominator <= 0:
        raise ZeroDivisionError("The g_r normalization denominator must be positive.")

    phases = np.pi * np.outer(z_grid, k)
    numerator = (
        np.sin(phases) * (a_coeffs * weights)[None, :]
        + np.cos(phases) * (b_coeffs * weights)[None, :]
    )
    return np.sum(numerator, axis=1) / denominator


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    fig_dir = repo_root / "examples" / "plm" / "figs" / "1.7"
    fig_dir.mkdir(parents=True, exist_ok=True)
    mpl_config_dir = repo_root / "examples" / "plm" / ".mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib.pyplot as plt

    x_grid = np.linspace(-DOMAIN_RADIUS, DOMAIN_RADIUS, 2401)
    z_grid = np.tanh(x_grid)

    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    for r_value, color in zip(R_VALUES, LINE_COLORS, strict=True):
        y_grid = evaluate_g_r(z_grid, r=float(r_value))
        axis.plot(
            x_grid,
            y_grid,
            color=color,
            linewidth=2.2,
            label=rf"$f_{{{r_value}}}(x)=g_{{{r_value}}}(\tanh(x))$",
        )

    axis.set_xlabel(r"$x$")
    axis.set_ylabel(r"$f_r(x)$")
    axis.set_title(r"Preview of $f_r(x)=g_r(\tanh(x))$ for the 1.7.4 family")
    axis.legend(loc="upper right")
    axis.grid(alpha=0.18, linewidth=0.6)
    figure.tight_layout()

    output_path = fig_dir / "1.7.4_tanh_wrapped_preview.png"
    figure.savefig(output_path, dpi=220)
    plt.close(figure)

    metadata_path = fig_dir / "1.7.4_tanh_wrapped_preview_metadata.json"
    metadata = {
        "r_values": list(R_VALUES),
        "num_terms": NUM_TERMS,
        "coefficients": {"a_k": "1", "b_k": "(-1)^k"},
        "domain": [-DOMAIN_RADIUS, DOMAIN_RADIUS],
        "wrapped_function": "f_r(x) = g_r(tanh(x))",
        "normalization": "sqrt(sum_{k=1}^{10} (a_k^2 + b_k^2) k^(-2/r))",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved preview figure to {output_path}")
    print(f"Saved preview metadata to {metadata_path}")


if __name__ == "__main__":
    main()
