import argparse
from pathlib import Path

import torch
from experiments.frame_diff_common.config_loader import get_config, reset_config
from experiments.frame_diff_common.tensor_manager import TensorManager
from experiments.frame_diff_common.tensor_io import save_tensor
from experiments.frame_diff_common.weight_sync import sync_weights, save_pta_weights, set_msa_weights_from_pta
from experiments.operator_diff import operator_registry

OPERATOR_REGISTRY = operator_registry.OPERATOR_REGISTRY
get_operator_factory = operator_registry.get_operator_factory


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


def run_rq1(backend_filter: str = None):
    reset_config()
    cfg = get_config("operator")
    out_dir = Path(cfg["experiment"]["output_dir"]) / "rq1_diff"
    num_iter = cfg["experiment"]["num_iterations"]
    device_str = cfg["experiment"].get("device", "cpu")

    tm = TensorManager(seed=cfg["experiment"]["seed"], device=device_str)

    backends = ["pta", "msa"] if backend_filter is None else [backend_filter]

    for op_name, entry in OPERATOR_REGISTRY.items():
        if entry.get("skip"):
            print(f"Skipping {op_name} (marked as skip)")
            continue
        if "msa" in backends and entry.get("skip_msa"):
            print(f"Skipping {op_name} MSA (not supported)")
            backends_to_run = [b for b in backends if b != "msa"]
        else:
            backends_to_run = backends

        print(f"\nOperator: {op_name}")

        for i in range(num_iter):
            inputs = generate_input(tm, op_name, entry, i)

            # Create operator instances for each backend first
            ops = {}
            for backend in backends_to_run:
                try:
                    factory, _ = get_operator_factory(op_name, backend)
                    op = factory()
                    if op is None:
                        continue
                    if hasattr(op, "to"):
                        op = op.to(device_str)
                    ops[backend] = op
                except Exception as e:
                    print(f"  {backend.upper()} create op iter {i} failed: {e}")

            # Sync weights if both PTA and MSA operators exist (same process)
            if "pta" in ops and "msa" in ops:
                try:
                    sync_weights(ops["pta"], ops["msa"], op_name, i)
                except Exception as e:
                    print(f"  Weight sync iter {i} failed: {e}")

            # Save PTA weights for cross-environment MSA loading
            if "pta" in ops:
                try:
                    save_pta_weights(ops["pta"], op_name, i, out_dir)
                except Exception as e:
                    print(f"  PTA save weights iter {i} failed: {e}")

            # Load PTA weights into MSA operator
            if "msa" in ops:
                try:
                    loaded = set_msa_weights_from_pta(ops["msa"], op_name, i, out_dir)
                    if not loaded:
                        print(f"  MSA iter {i}: PTA weights not found, using default init")
                except Exception as e:
                    print(f"  MSA load weights iter {i} failed: {e}")

            # Run operators
            for backend in backends_to_run:
                if backend not in ops:
                    continue
                try:
                    output = run_operator(ops[backend], inputs, backend)
                    if output is not None:
                        save_tensor(output, out_dir / op_name / f"iter_{i:03d}_{backend}_output.pt")
                        print(f"  {backend.upper()} iter {i}: output saved")
                except Exception as e:
                    print(f"  {backend.upper()} iter {i} failed: {e}")

    print(f"\nAll raw outputs saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["pta", "msa"], default=None, help="Run only one backend")
    args = parser.parse_args()
    run_rq1(backend_filter=args.backend)
