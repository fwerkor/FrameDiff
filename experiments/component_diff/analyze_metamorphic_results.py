#!/usr/bin/env python3
"""RQ2 analysis: compute metamorphic metrics from saved baseline/perturbed tensors."""
import argparse
import json
from pathlib import Path

from experiments.frame_diff_common.config_loader import get_config
from experiments.frame_diff_common.metrics import compute_metamorphic_metrics
from experiments.frame_diff_common.tensor_io import load_tensor


def analyze_rq2(output_dir: Path):
    """Load saved tensors and compute metamorphic metrics."""
    results = []

    for comp_dir in sorted(output_dir.iterdir()):
        if not comp_dir.is_dir():
            continue
        comp_name = comp_dir.name
        print(f"\n=== Component: {comp_name} ===")

        for sigma_dir in sorted(comp_dir.iterdir()):
            if not sigma_dir.is_dir():
                continue
            sigma_str = sigma_dir.name.replace("sigma_", "")
            try:
                sigma_val = float(sigma_str)
            except ValueError:
                continue
            print(f"  sigma={sigma_val}")

            baseline_files = sorted(sigma_dir.glob("iter_*_baseline.pt"))
            for baseline_path in baseline_files:
                perturbed_path = baseline_path.with_name(baseline_path.name.replace("_baseline", "_perturbed"))
                if not perturbed_path.exists():
                    continue

                parts = baseline_path.stem.split("_")
                iter_str = parts[1]
                backend = parts[2]

                try:
                    baseline_tensor = load_tensor(baseline_path)
                    perturbed_tensor = load_tensor(perturbed_path)
                except Exception as e:
                    print(f"    iter {iter_str} {backend} load error: {e}")
                    continue

                metrics = compute_metamorphic_metrics(baseline_tensor, perturbed_tensor)
                metrics["component"] = comp_name
                metrics["sigma"] = sigma_val
                metrics["iteration"] = int(iter_str)
                metrics["backend"] = backend
                results.append(metrics)
                print(f"    {backend} iter {iter_str}: max_delta={metrics['max_abs_delta']:.6e}")

    summary_path = output_dir / "rq2_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {summary_path} ({len(results)} entries)")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Path to rq2_meta output directory. Defaults to config value.")
    args = parser.parse_args()

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        cfg = get_config("component")
        out_dir = Path(cfg["experiment"]["output_dir"]) / "rq2_meta"

    analyze_rq2(out_dir)
