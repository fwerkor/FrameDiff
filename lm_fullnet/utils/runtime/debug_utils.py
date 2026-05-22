import os
from typing import Any, Optional


DEBUG_PREFIX = "[LMSV_DEBUG]"
_STATE = {
    "step": -1,
    "weights_logged": False,
}


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return value.lower() in ("1", "true", "yes", "on")


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


def begin_debug_step() -> int:
    _STATE["step"] += 1
    return _STATE["step"]


def set_debug_step(step: int) -> None:
    _STATE["step"] = int(step)


def get_debug_step() -> int:
    return _STATE["step"]


def should_log_full(step: Optional[int] = None) -> bool:
    if step is None:
        step = get_debug_step()
    return step <= 0


def should_log_heavy() -> bool:
    return _env_flag("LMSV_DEBUG_COMPARE")


def should_log_weights_once() -> bool:
    return is_rank0() and not _STATE["weights_logged"] and should_log_full()


def mark_weights_logged() -> None:
    _STATE["weights_logged"] = True


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


def _shape_of(value: Any) -> Optional[tuple]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(shape)
    except Exception:
        return shape


def _dtype_of(value: Any) -> Any:
    return getattr(value, "dtype", None)


def _numel_of(value: Any) -> Optional[int]:
    try:
        if _is_torch_tensor(value):
            return int(value.numel())
        if _is_mindspore_tensor(value):
            return int(value.size)
    except Exception:
        return None
    return None


def _to_numpy_like(value: Any, to_float: bool = False):
    try:
        if _is_torch_tensor(value):
            tensor = value.detach()
            if to_float:
                tensor = tensor.float()
            return tensor.cpu().reshape(-1).tolist()
        if _is_mindspore_tensor(value):
            tensor = value.detach() if hasattr(value, "detach") else value
            if to_float:
                import mindspore as ms

                tensor = tensor.astype(ms.float32)
            return tensor.asnumpy().reshape(-1).tolist()
    except Exception:
        return None
    return None


def _stats(value: Any) -> Optional[dict]:
    try:
        if _is_torch_tensor(value):
            tensor = value.detach().float().cpu()
            if tensor.numel() == 0:
                return None
            return {
                "mean": float(tensor.mean().item()),
                "std": float(tensor.std(unbiased=False).item()) if tensor.numel() > 1 else 0.0,
                "min": float(tensor.min().item()),
                "max": float(tensor.max().item()),
                "sum": float(tensor.sum().item()),
            }
        if _is_mindspore_tensor(value):
            import numpy as np
            import mindspore as ms

            array = value.astype(ms.float32).asnumpy()
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


def _format_value(value: Any) -> str:
    try:
        if value is None:
            return "None"
        if isinstance(value, float):
            return f"{value:.6f}"
        if isinstance(value, (int, bool)):
            return str(int(value)) if isinstance(value, bool) else str(value)
        return str(value)
    except Exception:
        return "<unprintable>"


def _format_sample(value: Any, max_items: int) -> str:
    flat = _to_numpy_like(value, to_float=False)
    if flat is None:
        return "[]"
    return "[" + ",".join(_format_value(item) for item in flat[:max_items]) + "]"


def debug_message(message: str) -> None:
    if not is_rank0():
        return
    print(f"{DEBUG_PREFIX} step={get_debug_step()} {message}")


def debug_scalar(name: str, value: Any, extra: Optional[str] = None) -> None:
    if not is_rank0():
        return
    line = f"{DEBUG_PREFIX} step={get_debug_step()} {name}={_format_value(value)}"
    if extra:
        line = f"{line} {extra}"
    print(line)


def debug_tensor_summary(
    name: str,
    tensor: Any,
    max_items: int = 8,
    include_stats: bool = True,
    include_sum: bool = False,
    extra: Optional[str] = None,
) -> None:
    if not is_rank0():
        return
    if tensor is None:
        print(f"{DEBUG_PREFIX} step={get_debug_step()} {name}=None")
        return

    parts = [
        f"{DEBUG_PREFIX} step={get_debug_step()}",
        name,
        f"shape={_shape_of(tensor)}",
        f"dtype={_dtype_of(tensor)}",
    ]

    stats = _stats(tensor) if include_stats or include_sum else None
    if include_stats and stats is not None:
        parts.extend(
            [
                f"mean={_format_value(stats['mean'])}",
                f"std={_format_value(stats['std'])}",
                f"min={_format_value(stats['min'])}",
                f"max={_format_value(stats['max'])}",
            ]
        )
    if include_sum and stats is not None:
        parts.append(f"sum={_format_value(stats['sum'])}")

    parts.append(f"sample={_format_sample(tensor, max_items)}")
    if extra:
        parts.append(extra)
    print(" ".join(parts))


def debug_parameter_summary(name: str, tensor: Any, max_items: int = 8) -> None:
    debug_tensor_summary(name=name, tensor=tensor, max_items=max_items, include_stats=True, include_sum=False)
