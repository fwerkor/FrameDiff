#!/usr/bin/env python3
"""RQ1 analysis: compute diff metrics from saved PTA/MSA output tensors."""
import argparse
import json
from pathlib import Path

from experiments.common.config_loader import get_config
from experiments.common.metrics import compute_diff_metrics
from experiments.common.tensor_io import load_tensor


def analyze_rq1(output_dir: Path):
    """Load saved tensors and compute differential metrics."""
    results = []

    for comp_dir in sorted(output_dir.iterdir()):
        if not comp_dir.is_dir():
            continue
        comp_name = comp_dir.name
        print(f"\n=== Component: {comp_name} ===")

        # Find all iterations that have both PTA and MSA outputs
        pta_files = sorted(comp_dir.glob("iter_*_pta_output.pt"))
        for pta_path in pta_files:
            msa_path = pta_path.with_name(pta_path.name.replace("_pta_", "_msa_"))
            if not msa_path.exists():
                continue

            iter_str = pta_path.stem.split("_")[1]
            try:
                pta_tensor = load_tensor(pta_path)
                msa_tensor = load_tensor(msa_path)
            except Exception as e:
                print(f"  iter {iter_str} load error: {e}")
                continue

            metrics = compute_diff_metrics(pta_tensor, msa_tensor)
            metrics["component"] = comp_name
            metrics["iteration"] = int(iter_str)
            results.append(metrics)
            print(f"  iter {iter_str}: max_diff={metrics['max_abs_diff']:.6e}, mean_diff={metrics['mean_abs_diff']:.6e}")

    summary_path = output_dir / "rq1_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {summary_path} ({len(results)} entries)")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Path to rq1_diff output directory. Defaults to config value.")
    args = parser.parse_args()

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        cfg = get_config("component")
        out_dir = Path(cfg["experiment"]["output_dir"]) / "rq1_diff"

    analyze_rq1(out_dir)
