"""Full-network tensor/weight export for FrameDiff language-model experiments."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any


COMPONENT_CATALOG: list[dict[str, Any]] = [
    {"id": 1, "name": "embedding_operator", "level": "operator"},
    {"id": 2, "name": "normalization_operator", "level": "operator"},
    {"id": 3, "name": "linear_operator", "level": "operator"},
    {"id": 4, "name": "matmul_attention_score_operator", "level": "operator"},
    {"id": 5, "name": "attention_core_operator", "level": "operator"},
    {"id": 6, "name": "flash_attention_operator", "level": "operator"},
    {"id": 7, "name": "softmax_operator", "level": "operator"},
    {"id": 8, "name": "gelu_activation_operator", "level": "operator"},
    {"id": 9, "name": "silu_swiglu_activation_operator", "level": "operator"},
    {"id": 10, "name": "residual_elementwise_operator", "level": "operator"},
    {"id": 11, "name": "embedding_layer", "level": "component"},
    {"id": 12, "name": "self_attention_block", "level": "component"},
    {"id": 13, "name": "ffn_block", "level": "component"},
    {"id": 14, "name": "decoder_block", "level": "component"},
    {"id": 15, "name": "output_layer", "level": "component"},
    {"id": 16, "name": "moe_ffn_block", "level": "component"},
    {"id": 17, "name": "mla_self_attention_block", "level": "component"},
]

FULL_NETWORK_COMPONENT = {"id": 0, "name": "full_network", "level": "network"}

_STATE = {
    "step": -1,
    "counter": 0,
    "manifest_written": set(),
    "emitted_records": set(),
}
_LOCK = threading.Lock()


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def trace_enabled() -> bool:
    return _env_flag("LMSV_FULLNET_TRACE") or _env_flag("LMSV_DEBUG_COMPARE")


def is_rank0() -> bool:
    try:
        import torch

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
    except Exception:
        pass

    try:
        from mindspore.communication.management import get_rank

        return get_rank() == 0
    except Exception:
        return True


def set_trace_step(step: int) -> None:
    _STATE["step"] = int(step)
    os.environ["LMSV_FULLNET_TRACE_STEP"] = str(int(step))


def get_trace_step() -> int:
    raw = os.getenv("LMSV_FULLNET_TRACE_STEP")
    if raw is not None:
        try:
            return int(raw)
        except ValueError:
            pass
    return int(_STATE["step"])


def _trace_root() -> Path:
    raw = os.getenv("LMSV_FULLNET_TRACE_DIR") or "res/fullnet_trace"
    return Path(raw).expanduser().resolve()


def _context() -> dict[str, Any]:
    return {
        "backend": os.getenv("LMSV_FULLNET_TRACE_BACKEND", "unknown"),
        "run": os.getenv("LMSV_FULLNET_TRACE_RUN", "baseline"),
        "iteration": os.getenv("LMSV_FULLNET_TRACE_ITER", os.getenv("MUTATE_ROUND", "0")),
        "step": get_trace_step(),
        "pid": os.getpid(),
    }


def _safe_part(value: Any, *, max_len: int = 80) -> str:
    text = str(value if value is not None else "none")
    text = re.sub(r"[^A-Za-z0-9_.=-]+", "_", text).strip("_")
    if not text:
        text = "none"
    return text[:max_len]


def _trace_mode() -> str:
    return os.getenv("LMSV_FULLNET_TRACE_MODE", "output_only").strip().lower()


def _full_trace_enabled() -> bool:
    mode = _trace_mode()
    granularity = os.getenv("LMSV_FULLNET_TRACE_GRANULARITY", "").strip().lower()
    return mode in {"full", "all", "debug", "layer"} or granularity in {"layer", "node", "full", "all"}


def _is_output_tensor(stage: Any, name: Any) -> bool:
    text = f"{stage}.{name}".lower()
    return "output" in text or "logits" in text


def _component_instance_key(component_id: int, stage: Any, name: Any, node_id: Any | None) -> str:
    if node_id is not None:
        return f"node:{node_id}"
    text = f"{stage}.{name}"
    match = re.search(r"(decoder|block|layer)[_-]?(\d+)", text, flags=re.IGNORECASE)
    if match:
        return f"{match.group(1).lower()}:{match.group(2)}"
    return f"component:{component_id}"


def _should_emit_tensor(component_id: int, stage: str, name: Any, node_id: Any | None) -> bool:
    if _full_trace_enabled():
        return True
    if not _is_output_tensor(stage, name):
        return False
    ctx = _context()
    key = (
        "tensor",
        ctx["backend"],
        ctx["run"],
        ctx["iteration"],
        ctx["step"],
        str(_trace_root()),
        component_id,
        _component_instance_key(component_id, stage, name, node_id),
    )
    with _LOCK:
        if key in _STATE["emitted_records"]:
            return False
        _STATE["emitted_records"].add(key)
    return True


def _next_counter() -> int:
    with _LOCK:
        _STATE["counter"] += 1
        return int(_STATE["counter"])


def _component_lookup(component_id: int | None, component_name: str | None) -> tuple[int, str]:
    if component_id is None:
        if component_name == FULL_NETWORK_COMPONENT["name"]:
            return 0, str(component_name)
        for component in COMPONENT_CATALOG:
            if component["name"] == component_name:
                return int(component["id"]), str(component["name"])
        return -1, str(component_name or "unknown")
    if component_id == 0:
        return 0, str(component_name or FULL_NETWORK_COMPONENT["name"])
    for component in COMPONENT_CATALOG:
        if int(component["id"]) == int(component_id):
            return int(component_id), str(component_name or component["name"])
    return int(component_id), str(component_name or "unknown")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_index(root: Path, payload: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        with (root / "trace_index.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def trace_components_manifest(extra: dict[str, Any] | None = None) -> None:
    if not trace_enabled() or not is_rank0():
        return
    root = _trace_root()
    root_key = str(root)
    with _LOCK:
        if root_key in _STATE["manifest_written"]:
            return
        _STATE["manifest_written"].add(root_key)
    payload = {
        "task": "fullnet",
        "full_network_component": FULL_NETWORK_COMPONENT,
        "components": COMPONENT_CATALOG,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "context": _context(),
    }
    if extra:
        payload["extra"] = extra
    _write_json(root / "components.json", payload)
    trace_event("trace_manifest", {"component_count": len(COMPONENT_CATALOG)})


def _is_torch_tensor(value: Any) -> bool:
    try:
        import torch

        return isinstance(value, torch.Tensor)
    except Exception:
        return False


def _is_mindspore_tensor(value: Any) -> bool:
    try:
        import mindspore as ms

        return isinstance(value, (ms.Tensor, ms.Parameter))
    except Exception:
        return False


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(item) for item in tuple(shape)]
    except Exception:
        return list(shape) if isinstance(shape, (list, tuple)) else None


def _numel(value: Any) -> int | None:
    try:
        if _is_torch_tensor(value):
            return int(value.numel())
        if _is_mindspore_tensor(value):
            return int(value.size)
    except Exception:
        return None
    return None


def _stats(value: Any) -> dict[str, float] | None:
    try:
        if _is_torch_tensor(value):
            tensor = value.detach()
            if tensor.numel() == 0:
                return None
            if tensor.dtype == getattr(__import__("torch"), "bool"):
                tensor = tensor.to(dtype=__import__("torch").float32)
            elif not tensor.dtype.is_floating_point:
                tensor = tensor.float()
            else:
                tensor = tensor.float()
            tensor = tensor.cpu()
            return {
                "mean": float(tensor.mean().item()),
                "std": float(tensor.std(unbiased=False).item()) if tensor.numel() > 1 else 0.0,
                "min": float(tensor.min().item()),
                "max": float(tensor.max().item()),
                "sum": float(tensor.sum().item()),
            }
        if _is_mindspore_tensor(value):
            import numpy as np

            array = value.asnumpy().astype(np.float32, copy=False)
            if array.size == 0:
                return None
            return {
                "mean": float(array.mean()),
                "std": float(array.std()),
                "min": float(array.min()),
                "max": float(array.max()),
                "sum": float(array.sum()),
            }
    except Exception:
        return None
    return None


def _context_dir(root: Path, kind: str) -> Path:
    ctx = _context()
    return (
        root
        / _safe_part(ctx["backend"])
        / _safe_part(ctx["run"])
        / f"iter_{_safe_part(ctx['iteration'])}"
        / f"step_{_safe_part(ctx['step'])}"
        / kind
    )


def _save_tensor_file(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if _is_torch_tensor(value):
        import torch

        tensor = value.detach().cpu()
        torch.save(tensor, path.with_suffix(".pt"))
        return str(path.with_suffix(".pt"))
    if _is_mindspore_tensor(value):
        import numpy as np

        np.save(path.with_suffix(".npy"), value.asnumpy())
        return str(path.with_suffix(".npy"))
    _write_json(path.with_suffix(".json"), {"value": repr(value)})
    return str(path.with_suffix(".json"))


def trace_tensor(
    component_id: int | None,
    component_name: str | None,
    tensor_name: str,
    tensor: Any,
    *,
    stage: str = "forward",
    node_id: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> str | None:
    if not trace_enabled() or not is_rank0() or tensor is None:
        return None
    if not (_is_torch_tensor(tensor) or _is_mindspore_tensor(tensor)):
        return None
    trace_components_manifest()
    root = _trace_root()
    component_id, component_name = _component_lookup(component_id, component_name)
    if not _should_emit_tensor(component_id, stage, tensor_name, node_id):
        return None
    counter = _next_counter()
    base = (
        _context_dir(root, "tensors")
        / f"{counter:06d}_c{component_id}_{_safe_part(component_name)}"
        f"_{_safe_part(stage)}_{_safe_part(tensor_name)}"
    )
    try:
        file_path = _save_tensor_file(base, tensor)
    except Exception as exc:
        trace_event(
            "trace_tensor_error",
            {
                "component_id": component_id,
                "component_name": component_name,
                "tensor_name": tensor_name,
                "stage": stage,
                "error": str(exc),
            },
        )
        return None

    payload = {
        "kind": "tensor",
        "component_id": component_id,
        "component_name": component_name,
        "tensor_name": tensor_name,
        "stage": stage,
        "node_id": node_id,
        "path": file_path,
        "shape": _shape(tensor),
        "dtype": str(getattr(tensor, "dtype", None)),
        "numel": _numel(tensor),
        "stats": _stats(tensor),
        "timestamp": time.time(),
        **_context(),
    }
    if extra:
        payload["extra"] = extra
    _append_index(root, payload)
    return file_path


def _state_dict_cpu(module: Any) -> dict[str, Any]:
    state = {}
    if module is None or not hasattr(module, "state_dict"):
        return state
    try:
        raw_state = module.state_dict()
    except Exception:
        return state
    for name, value in raw_state.items():
        try:
            if _is_torch_tensor(value):
                state[str(name)] = value.detach().cpu()
            elif _is_mindspore_tensor(value):
                state[str(name)] = value.asnumpy()
            else:
                state[str(name)] = value
        except Exception:
            continue
    return state


def trace_module_weights(
    component_id: int | None,
    component_name: str | None,
    module: Any,
    *,
    stage: str = "forward",
    node_id: Any | None = None,
    module_name: str | None = None,
    extra: dict[str, Any] | None = None,
) -> str | None:
    if not trace_enabled() or not is_rank0() or module is None:
        return None
    if not _env_flag("LMSV_FULLNET_TRACE_FULL_WEIGHTS", True):
        return None
    if not _full_trace_enabled():
        return None
    state = _state_dict_cpu(module)
    if not state:
        return None
    trace_components_manifest()
    root = _trace_root()
    component_id, component_name = _component_lookup(component_id, component_name)
    counter = _next_counter()
    base = (
        _context_dir(root, "weights")
        / f"{counter:06d}_c{component_id}_{_safe_part(component_name)}"
        f"_{_safe_part(stage)}_{_safe_part(module_name or type(module).__name__)}"
    )
    try:
        if any(_is_torch_tensor(value) for value in state.values()):
            import torch

            path = base.with_suffix(".pt")
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state, path)
        else:
            path = base.with_suffix(".json")
            _write_json(path, {key: repr(value) for key, value in state.items()})
    except Exception as exc:
        trace_event(
            "trace_weight_error",
            {
                "component_id": component_id,
                "component_name": component_name,
                "module_name": module_name,
                "stage": stage,
                "error": str(exc),
            },
        )
        return None

    payload = {
        "kind": "weights",
        "component_id": component_id,
        "component_name": component_name,
        "module_name": module_name or type(module).__name__,
        "stage": stage,
        "node_id": node_id,
        "path": str(path),
        "parameter_count": sum(_numel(value) or 0 for value in state.values()),
        "state_keys": list(state.keys()),
        "timestamp": time.time(),
        **_context(),
    }
    if extra:
        payload["extra"] = extra
    _append_index(root, payload)
    return str(path)


def _flatten_tensor_leaves(value: Any, prefix: str = "value", limit: int | None = None):
    seen = 0

    def _walk(item: Any, name: str):
        nonlocal seen
        if limit is not None and seen >= limit:
            return
        if _is_torch_tensor(item) or _is_mindspore_tensor(item):
            seen += 1
            yield name, item
            return
        if isinstance(item, dict):
            for key, child in item.items():
                yield from _walk(child, f"{name}.{_safe_part(key)}")
            return
        if isinstance(item, (tuple, list)):
            for idx, child in enumerate(item):
                yield from _walk(child, f"{name}.{idx}")

    yield from _walk(value, prefix)


def trace_nested_tensors(
    component_id: int | None,
    component_name: str | None,
    base_name: str,
    value: Any,
    *,
    stage: str = "forward",
    node_id: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> list[str]:
    if not trace_enabled() or not is_rank0():
        return []
    try:
        limit = int(os.getenv("LMSV_FULLNET_TRACE_MAX_LEAVES", "64"))
        if limit <= 0:
            limit = None
    except ValueError:
        limit = 64
    paths: list[str] = []
    for tensor_name, tensor in _flatten_tensor_leaves(value, base_name, limit):
        path = trace_tensor(
            component_id,
            component_name,
            tensor_name,
            tensor,
            stage=stage,
            node_id=node_id,
            extra=extra,
        )
        if path:
            paths.append(path)
    return paths


def trace_event(event: str, payload: dict[str, Any] | None = None) -> None:
    if not trace_enabled() or not is_rank0():
        return
    root = _trace_root()
    item = {
        "kind": "event",
        "event": event,
        "payload": payload or {},
        "timestamp": time.time(),
        **_context(),
    }
    _append_index(root, item)


def _to_scalar(value: Any) -> float | int | str | None:
    try:
        if _is_torch_tensor(value):
            if value.numel() == 1:
                return float(value.detach().cpu().item())
            return float(value.detach().float().cpu().mean().item())
        if _is_mindspore_tensor(value):
            array = value.asnumpy()
            if array.size == 1:
                return float(array.reshape(-1)[0])
            return float(array.astype("float32").mean())
        if isinstance(value, (int, float)):
            return value
    except Exception:
        pass
    return repr(value) if value is not None else None


def trace_loss(name: str, value: Any, *, extra: dict[str, Any] | None = None) -> None:
    if not trace_enabled() or not is_rank0():
        return
    payload = {
        "name": name,
        "value": _to_scalar(value),
        "shape": _shape(value),
        "dtype": str(getattr(value, "dtype", None)),
    }
    if extra:
        payload["extra"] = extra
    trace_event("loss", payload)


def maybe_perturb_tensor(
    tensor: Any,
    *,
    tensor_name: str,
    component_id: int | None = 0,
    component_name: str | None = "full_network",
    stage: str = "input_perturbation",
    node_id: Any | None = None,
) -> Any:
    if tensor is None or not _env_flag("LMSV_FULLNET_PERTURB"):
        return tensor
    if not _is_torch_tensor(tensor):
        trace_event("input_perturbation_skipped", {"tensor_name": tensor_name, "reason": "not_torch_tensor"})
        return tensor
    try:
        import torch

        if not tensor.dtype.is_floating_point:
            trace_event(
                "input_perturbation_skipped",
                {"tensor_name": tensor_name, "reason": f"non_float_dtype:{tensor.dtype}"},
            )
            return tensor
        eps = float(os.getenv("LMSV_FULLNET_PERTURB_EPS", "1e-5"))
        delta = torch.full_like(tensor, eps)
        perturbed = tensor + delta
        trace_tensor(component_id, component_name, f"{tensor_name}.baseline", tensor, stage=stage, node_id=node_id)
        trace_tensor(component_id, component_name, f"{tensor_name}.delta", delta, stage=stage, node_id=node_id)
        trace_tensor(component_id, component_name, f"{tensor_name}.perturbed", perturbed, stage=stage, node_id=node_id)
        trace_event(
            "input_perturbation",
            {"tensor_name": tensor_name, "eps": eps, "direction": "positive", "stage": stage, "node_id": node_id},
        )
        return perturbed
    except Exception as exc:
        trace_event("input_perturbation_error", {"tensor_name": tensor_name, "error": str(exc)})
        return tensor
