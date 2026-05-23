import json
import argparse
from pathlib import Path

import torch
from experiments.common.config_loader import get_config, reset_config
from experiments.common.tensor_manager import TensorManager
from experiments.common.tensor_io import save_tensor
from experiments.operator import operator_registry

OPERATOR_REGISTRY = operator_registry.OPERATOR_REGISTRY
get_operator_factory = operator_registry.get_operator_factory


def add_uniform_perturbation(tensor, sigma: float):
    """每个数值加同一个固定的单向扰动 sigma."""
    if isinstance(tensor, (list, tuple)):
        return [add_uniform_perturbation(t, sigma) for t in tensor]
    return tensor + sigma


def generate_input(tm: TensorManager, op_name: str, entry: dict, iteration: int):
    shape = entry.get("input_shape", (32, 2, 1024))
    input_type = entry.get("input_type", "float")
    input_types = entry.get("input_types", None)
    input_ranges = entry.get("input_ranges", None)
    if entry.get("multi_input"):
        inputs = []
        for i, s in enumerate(shape):
            itype = input_types[i] if input_types else input_type
            if itype == "int":
                irange = input_ranges[i] if input_ranges else (0, 40000)
                inputs.append(tm.generate(f"{op_name}_input_{i}", iteration, s, dtype=torch.int64, low=irange[0], high=irange[1]))
            elif itype == "bool":
                inputs.append(tm.generate(f"{op_name}_input_{i}", iteration, s, dtype=torch.bool))
            else:
                inputs.append(tm.generate(f"{op_name}_input_{i}", iteration, s))
        return inputs
    else:
        if input_type == "int":
            input_range = entry.get("input_range", (0, 50000))
            return tm.generate(op_name + "_input", iteration, shape, dtype=torch.int64, low=input_range[0], high=input_range[1])
        return tm.generate(op_name + "_input", iteration, shape)


def _to_ms(tensor):
    import mindspore as ms
    if isinstance(tensor, (list, tuple)):
        return [_to_ms(t) for t in tensor]
    if hasattr(tensor, "asnumpy"):
        return tensor
    return ms.Tensor(tensor.numpy())


def run_operator(op, inputs, backend: str):
    if backend == "pta":
        if isinstance(inputs, (list, tuple)):
            return op(*inputs)
        return op(inputs)
    else:
        try:
            if isinstance(inputs, (list, tuple)):
                ms_inputs = _to_ms(inputs)
                return op(*ms_inputs)
            return op(_to_ms(inputs))
        except Exception as e:
            print(f"  MSA execution error: {e}")
            return None


def run_rq2(backend_filter: str = None):
    reset_config()
    cfg = get_config("operator")
    out_dir = Path(cfg["experiment"]["output_dir"]) / "rq2_meta"
    num_iter = cfg["experiment"]["num_iterations"]
    device_str = cfg["experiment"].get("device", "cpu")
    sigmas = [float(s) for s in cfg["perturbation"]["sigmas"]]

    tm = TensorManager(seed=cfg["experiment"]["seed"], device=device_str)

    backends = ["pta", "msa"] if backend_filter is None else [backend_filter]

    for op_name, entry in OPERATOR_REGISTRY.items():
        if entry.get("skip"):
            continue
        # Skip integer-input operators for metamorphic test (noise on discrete IDs is meaningless)
        if entry.get("input_type") == "int":
            print(f"\nOperator: {op_name} (skipped: integer input)")
            continue
        if "msa" in backends and entry.get("skip_msa"):
            print(f"\nOperator: {op_name} (skipped: MSA not supported)")
            backends_to_run = [b for b in backends if b != "msa"]
        else:
            backends_to_run = backends

        print(f"\nOperator: {op_name}")

        for sigma in sigmas:
            print(f"  Sigma: {sigma}")

            for i in range(num_iter):
                inputs = generate_input(tm, op_name, entry, i)
                inputs_perturbed = add_uniform_perturbation(inputs, sigma)

                for backend in backends_to_run:
                    try:
                        factory, _ = get_operator_factory(op_name, backend)
                        op = factory()
                        if op is None:
                            continue
                        if hasattr(op, "to"):
                            op = op.to(device_str)

                        baseline = run_operator(op, inputs, backend)
                        # Need a fresh op instance for perturbed run
                        op2 = factory()
                        if op2 is not None and hasattr(op2, "to"):
                            op2 = op2.to(device_str)
                        # Sync weights: perturbed op must have identical weights to baseline
                        if op2 is not None:
                            if backend == "pta" and hasattr(op, "named_parameters") and hasattr(op2, "named_parameters"):
                                for (_, p1), (_, p2) in zip(op.named_parameters(), op2.named_parameters()):
                                    p2.data.copy_(p1.data)
                            elif backend == "msa" and hasattr(op, "get_parameters") and hasattr(op2, "get_parameters"):
                                for p1, p2 in zip(op.get_parameters(), op2.get_parameters()):
                                    p2.set_data(p1.data)
                        perturbed = run_operator(op2, inputs_perturbed, backend)

                        if baseline is not None and perturbed is not None:
                            save_tensor(baseline, out_dir / op_name / f"sigma_{sigma}" / f"iter_{i:03d}_{backend}_baseline.pt")
                            save_tensor(perturbed, out_dir / op_name / f"sigma_{sigma}" / f"iter_{i:03d}_{backend}_perturbed.pt")
                            print(f"    {backend.upper()} iter {i}: baseline + perturbed saved")
                    except Exception as e:
                        import traceback
                        print(f"    {backend.upper()} iter {i} failed: {e}")
                        traceback.print_exc()

    print(f"\nAll raw outputs saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["pta", "msa"], default=None)
    args = parser.parse_args()
    run_rq2(backend_filter=args.backend)
