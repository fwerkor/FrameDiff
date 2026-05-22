import json
import argparse
from pathlib import Path

from experiments.frame_diff_common.config_loader import get_config, reset_config
from experiments.frame_diff_common.tensor_io import load_tensor, to_torch
from experiments.frame_diff_common.metrics import compute_metamorphic_metrics
from experiments.operator_diff import operator_registry

OPERATOR_REGISTRY = operator_registry.OPERATOR_REGISTRY


def run_analysis():
    reset_config()
    cfg = get_config("operator")
    out_dir = Path(cfg["experiment"]["output_dir"]) / "rq2_meta"
    num_iter = cfg["experiment"]["num_iterations"]
    sigmas = [float(s) for s in cfg["perturbation"]["sigmas"]]
    eps = cfg["metrics"]["eps"]

    all_results = []

    for op_name, entry in OPERATOR_REGISTRY.items():
        if entry.get("skip"):
            continue
        if entry.get("input_type") == "int":
            print(f"\nOperator: {op_name} (skipped: integer input)")
            continue

        print(f"\nOperator: {op_name}")
        op_results = {"operator": op_name, "sigmas": {}}

        for sigma in sigmas:
            print(f"  Sigma: {sigma}")
            sigma_results = []

            for i in range(num_iter):
                for backend in ["pta", "msa"]:
                    if entry.get("skip_msa") and backend == "msa":
                        continue

                    bl_path = out_dir / op_name / f"sigma_{sigma}" / f"iter_{i:03d}_{backend}_baseline.pt"
                    pt_path = out_dir / op_name / f"sigma_{sigma}" / f"iter_{i:03d}_{backend}_perturbed.pt"

                    if not bl_path.exists() or not pt_path.exists():
                        continue

                    try:
                        bl_torch = to_torch(load_tensor(bl_path))
                        pt_torch = to_torch(load_tensor(pt_path))

                        # Handle tuple outputs
                        if isinstance(bl_torch, (list, tuple)):
                            bl_torch = bl_torch[0]
                        if isinstance(pt_torch, (list, tuple)):
                            pt_torch = pt_torch[0]

                        metrics = compute_metamorphic_metrics(bl_torch, pt_torch, eps=eps)
                        sigma_results.append({
                            "iteration": i,
                            "backend": backend,
                            **metrics,
                        })
                        print(f"    {backend.upper()} iter {i}: delta_max={metrics['max_abs_delta']:.6e}")
                    except Exception as e:
                        print(f"    {backend.upper()} iter {i} analysis failed: {e}")

            op_results["sigmas"][str(sigma)] = sigma_results

        all_results.append(op_results)

    metrics_path = out_dir / "rq2_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze RQ2 metamorphic test results")
    args = parser.parse_args()
    run_analysis()
