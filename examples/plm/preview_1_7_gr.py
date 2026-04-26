"""Preview the shared random Fourier family g_r for the planned PLM 1.7.1 study."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

R_VALUES = (1, 2, 4, 8)
NUM_TERMS = 34
DIMENSION = 5
PREVIEW_SEED = 20260426
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


def build_shared_random_family(
    *,
    seed: int = PREVIEW_SEED,
    num_terms: int = NUM_TERMS,
    d: int = DIMENSION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return shared random coefficients and a normalized projection vector q."""
    if num_terms <= 0:
        raise ValueError("num_terms must be positive.")
    if d <= 0:
        raise ValueError("d must be positive.")

    rng = np.random.default_rng(seed)
    a_coeffs = rng.uniform(0.0, 2.0, size=num_terms)
    b_coeffs = rng.uniform(0.0, 2.0, size=num_terms)
    q = rng.normal(size=d)
    q /= np.linalg.norm(q)
    return a_coeffs, b_coeffs, q


def evaluate_g_r(
    x: np.ndarray,
    *,
    r: float,
    a_coeffs: np.ndarray,
    b_coeffs: np.ndarray,
) -> np.ndarray:
    """Evaluate the normalized shared-coefficient Fourier family g_r on a grid."""
    x_grid = np.asarray(x, dtype=float).reshape(-1)
    if r <= 0:
        raise ValueError("r must be positive.")
    if a_coeffs.shape != b_coeffs.shape:
        raise ValueError("a_coeffs and b_coeffs must have the same shape.")

    k = np.arange(1, len(a_coeffs) + 1, dtype=float)
    weights = k ** (-1.0 / float(r))
    denominator = np.sqrt(np.sum((a_coeffs**2 + b_coeffs**2) * (weights**2)))
    if denominator <= 0:
        raise ZeroDivisionError("The g_r normalization denominator must be positive.")

    phases = np.pi * np.outer(x_grid, k)
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

    a_coeffs, b_coeffs, q = build_shared_random_family()
    x_grid = np.linspace(-1.0, 1.0, 2001)

    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    for r_value, color in zip(R_VALUES, LINE_COLORS, strict=True):
        y_grid = evaluate_g_r(
            x_grid,
            r=float(r_value),
            a_coeffs=a_coeffs,
            b_coeffs=b_coeffs,
        )
        axis.plot(
            x_grid,
            y_grid,
            color=color,
            linewidth=2.2,
            label=rf"$g_{{{r_value}}}(x)$",
        )

    axis.set_xlabel(r"$x$")
    axis.set_ylabel(r"$g_r(x)$")
    axis.set_title(r"Preview of the shared-random Fourier family $g_r(x)$")
    axis.legend(loc="upper right")
    axis.grid(alpha=0.18, linewidth=0.6)
    figure.tight_layout()

    output_path = fig_dir / "1.7.1_g_r_preview.png"
    figure.savefig(output_path, dpi=220)
    plt.close(figure)

    metadata_path = fig_dir / "1.7.1_g_r_preview_metadata.json"
    metadata = {
        "preview_seed": PREVIEW_SEED,
        "r_values": list(R_VALUES),
        "num_terms": NUM_TERMS,
        "dimension": DIMENSION,
        "normalized_q": q.tolist(),
        "normalization": "sqrt(sum_k (a_k^2 + b_k^2) k^(-2/r))",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved preview figure to {output_path}")
    print(f"Saved preview metadata to {metadata_path}")
    print("Normalized q =", np.array2string(q, precision=6, separator=", "))


if __name__ == "__main__":
    main()
