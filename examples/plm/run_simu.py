"""Run partial linear model simulation experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiment_defs import build_evaluator_from_exp_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PLM simulation experiments.")
    parser.add_argument("--exp_id", required=True, help="Experiment identifier, for example 1.1.2.")
    parser.add_argument("--ntrials", required=True, type=int, help="Number of trials to run.")
    parser.add_argument("--seed_offset", default=0, type=int, help="Seed offset for reproducibility.")
    parser.add_argument("--device", default="cpu", help="PyTorch device string.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluator = build_evaluator_from_exp_id(
        exp_id=args.exp_id,
        n_trials=args.ntrials,
        seed_offset=args.seed_offset,
        device=args.device,
    )
    evaluator.run()
    print(f"Saved simulation results to {evaluator.result_path}")


if __name__ == "__main__":
    main()
