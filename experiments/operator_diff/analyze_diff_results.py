import json
import argparse
from pathlib import Path

from experiments.frame_diff_common.config_loader import get_config, reset_config
from experiments.frame_diff_common.tensor_io import load_tensor, to_torch
from experiments.frame_diff_common.metrics import compute_diff_metrics
from experiments.operator_diff import operator_registry

OPERATOR_REGISTRY = operator_registry.OPERATOR_REGISTRY


def run_analysis():
    reset_config()
    cfg = get_config("operator")
    out_dir = Path(cfg["experiment"]["output_dir"]) / "rq1_diff"
    num_iter = cfg["experiment"]["num_iterations"]
    eps = cfg["metrics"]["eps"]

    all_results = []

    for op_name, entry in OPERATOR_REGISTRY.items():
        if entry.get("skip"):
            continue
        if entry.get("skip_msa"):
            print(f"\nOperator: {op_name} (MSA skipped, no diff analysis)")
            continue

        print(f"\nOperator: {op_name}")
        op_results = {"operator": op_name, "iterations": []}

        for i in range(num_iter):
            pta_path = out_dir / op_name / f"iter_{i:03d}_pta_output.pt"
            msa_path = out_dir / op_name / f"iter_{i:03d}_msa_output.pt"

            if not pta_path.exists() or not msa_path.exists():
                print(f"  iter {i}: missing output files, skipped")
                continue

            try:
                pta_out = to_torch(load_tensor(pta_path))
                msa_out = to_torch(load_tensor(msa_path))

                # Handle tuple outputs
                if isinstance(pta_out, (list, tuple)):
                    pta_out = pta_out[0]
                if isinstance(msa_out, (list, tuple)):
                    msa_out = msa_out[0]

                if pta_out.shape != msa_out.shape:
                    print(f"  iter {i}: shape mismatch PTA {pta_out.shape} vs MSA {msa_out.shape}")
                    continue

                metrics = compute_diff_metrics(pta_out, msa_out, eps=eps)
                iter_result = {"iteration": i, **metrics}
                op_results["iterations"].append(iter_result)
                print(f"  iter {i}: max={metrics['max_abs_diff']:.6e}, mean={metrics['mean_abs_diff']:.6e}")
            except Exception as e:
                print(f"  iter {i} analysis failed: {e}")

        all_results.append(op_results)

    metrics_path = out_dir / "rq1_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze RQ1 differential test results")
    args = parser.parse_args()
    run_analysis()
