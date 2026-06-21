"""Runtime patch for MSA tensor-parallel linear mixed-dtype matmul.

MSAdapter/MindSpore MatMulExt is stricter than PyTorch autocast and rejects
fp32 activations with bf16 weights. This happens for Megatron's
``--fp32-residual-connection --bf16`` configuration: residual streams can stay
fp32 while parameters are bf16. Patch only the MSA Megatron linear autograd
functions so the matmul operands use a common dtype before calling MatMulExt.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any

_PATCH_LOCK = threading.Lock()
_PATCH_APPLIED = False
_IMPORT_HOOK_INSTALLED = False
_ORIGINAL_IMPORT = None
_PATCHED_MODULES: set[str] = set()
_TARGET_MODULE = "megatron.core.tensor_parallel.layers"
_TRUE_VALUES = {"1", "true", "yes", "on"}
_LOG_TRUE_VALUES = _TRUE_VALUES


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _rank_hint() -> str:
    return os.getenv("RANK_ID") or os.getenv("RANK") or "?"


def _log_enabled() -> bool:
    return os.getenv("LMSV_PATCH_LOG", "1").strip().lower() in _LOG_TRUE_VALUES


def _emit(message: str) -> None:
    if not _log_enabled():
        return
    try:
        sys.stderr.write(f"[{_timestamp()}] [LMSV_PATCH] [RANK={_rank_hint()}] {message}\n")
        sys.stderr.flush()
    except Exception:
        pass


def _is_enabled() -> bool:
    # Enabled by submodule_entry only for the MSA runtime path.
    return os.getenv("LMSV_ENABLE_MSA_LINEAR_DTYPE_PATCH", "0").strip().lower() in _TRUE_VALUES


def _dtype_name(dtype: Any) -> str:
    try:
        return str(dtype).lower()
    except Exception:
        return ""


def _is_low_precision(dtype: Any) -> bool:
    name = _dtype_name(dtype)
    return "bfloat16" in name or "float16" in name or name.endswith("half")


def _is_fp32(dtype: Any) -> bool:
    name = _dtype_name(dtype)
    return "float32" in name or name.endswith("float")


def _cast_tensor(tensor: Any, dtype: Any) -> Any:
    if dtype is None:
        return tensor
    try:
        return tensor.to(dtype=dtype)
    except TypeError:
        return tensor.to(dtype)


def _align_matmul_lhs_rhs(lhs: Any, rhs: Any) -> tuple[Any, Any]:
    """Align matmul operands for MindSpore MatMulExt.

    Prefer the low-precision operand's dtype when one side is fp32 and the other
    is fp16/bf16. This mirrors the effective autocast behavior used by the
    PyTorch/NPU path for bf16 models while avoiding a MatMulExt type error.
    """

    lhs_dtype = getattr(lhs, "dtype", None)
    rhs_dtype = getattr(rhs, "dtype", None)
    if lhs_dtype is None or rhs_dtype is None or lhs_dtype == rhs_dtype:
        return lhs, rhs

    if _is_low_precision(rhs_dtype) and _is_fp32(lhs_dtype):
        return _cast_tensor(lhs, rhs_dtype), rhs
    if _is_low_precision(lhs_dtype) and _is_fp32(rhs_dtype):
        return lhs, _cast_tensor(rhs, lhs_dtype)
    if _is_low_precision(rhs_dtype):
        return _cast_tensor(lhs, rhs_dtype), rhs
    if _is_low_precision(lhs_dtype):
        return lhs, _cast_tensor(rhs, lhs_dtype)
    return lhs, _cast_tensor(rhs, lhs_dtype)


def _matmul(lhs: Any, rhs: Any, torch_module: Any) -> Any:
    lhs, rhs = _align_matmul_lhs_rhs(lhs, rhs)
    return torch_module.matmul(lhs, rhs)


def _add_bias(output: Any, bias: Any) -> Any:
    if bias is None:
        return output
    try:
        if getattr(output, "dtype", None) is not None and getattr(bias, "dtype", None) != getattr(output, "dtype", None):
            bias = _cast_tensor(bias, getattr(output, "dtype", None))
    except Exception:
        pass
    return output + bias


def _decorate(module: Any, name: str, func: Any) -> Any:
    decorator = getattr(module, name, None)
    if decorator is None:
        return func
    try:
        return decorator(func)
    except Exception:
        return func


def _patch_module_obj(module: Any) -> bool:
    if module is None:
        return False
    module_name = getattr(module, "__name__", _TARGET_MODULE)
    if module_name in _PATCHED_MODULES:
        return False

    torch = getattr(module, "torch", None)
    frozen_cls = getattr(module, "LinearWithFrozenWeight", None)
    grad_cls = getattr(module, "LinearWithGradAccumulationAndAsyncCommunication", None)
    if torch is None or frozen_cls is None or grad_cls is None:
        return False

    custom_fwd = lambda f: _decorate(module, "custom_fwd", f)
    custom_bwd = lambda f: _decorate(module, "custom_bwd", f)

    @custom_fwd
    def _frozen_forward(ctx, input, weight, bias, allreduce_dgrad):
        ctx.save_for_backward(weight)
        ctx.allreduce_dgrad = allreduce_dgrad
        output = _matmul(input, weight.t(), torch)
        return _add_bias(output, bias)

    @custom_bwd
    def _frozen_backward(ctx, grad_output):
        (weight,) = ctx.saved_tensors
        grad_input = _matmul(grad_output, weight, torch)
        if ctx.allreduce_dgrad:
            torch.distributed.all_reduce(grad_input, group=module.get_tensor_model_parallel_group())
        return grad_input, None, None, None

    @custom_fwd
    def _grad_forward(
        ctx,
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer,
        wgrad_deferral_limit,
    ):
        ctx.save_for_backward(input, weight)
        ctx.use_bias = bias is not None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.allreduce_dgrad = allreduce_dgrad
        ctx.sequence_parallel = sequence_parallel
        ctx.wgrad_deferral_limit = wgrad_deferral_limit
        ctx.grad_output_buffer = grad_output_buffer

        if sequence_parallel:
            world_size = module.get_tensor_model_parallel_world_size()
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size
            all_gather_buffer = module.get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            module.dist_all_gather_func(all_gather_buffer, input, group=module.get_tensor_model_parallel_group())
            total_input = all_gather_buffer
        else:
            total_input = input

        output = _matmul(total_input, weight.t(), torch)
        return _add_bias(output, bias)

    @custom_bwd
    def _grad_backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias
        grad_output_buffer = ctx.grad_output_buffer
        wgrad_deferral_limit = ctx.wgrad_deferral_limit

        wgrad_compute = True
        if grad_output_buffer is not None:
            if wgrad_deferral_limit == 0 or len(grad_output_buffer) < wgrad_deferral_limit:
                grad_output_buffer.append(grad_output)
                wgrad_compute = False

        if wgrad_compute:
            if ctx.sequence_parallel:
                world_size = module.get_tensor_model_parallel_world_size()
                dim_size = list(input.size())
                dim_size[0] = dim_size[0] * world_size
                all_gather_buffer = module.get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
                handle = module.dist_all_gather_func(
                    all_gather_buffer,
                    input,
                    group=module.get_tensor_model_parallel_group(),
                    async_op=True,
                )
                total_input = all_gather_buffer
            else:
                total_input = input

        grad_input = _matmul(grad_output, weight, torch)

        if ctx.sequence_parallel and wgrad_compute:
            handle.wait()

        if wgrad_compute:
            grad_output, total_input = module.prepare_input_tensors_for_wgrad_compute(
                grad_output, total_input
            )

        if ctx.allreduce_dgrad:
            handle = torch.distributed.all_reduce(
                grad_input,
                group=module.get_tensor_model_parallel_group(),
                async_op=True,
            )

        if ctx.sequence_parallel:
            assert not ctx.allreduce_dgrad
            dim_size = list(input.size())
            sub_grad_input = torch.empty(
                dim_size,
                dtype=input.dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )
            handle = module.dist_reduce_scatter_func(
                sub_grad_input,
                grad_input,
                group=module.get_tensor_model_parallel_group(),
                async_op=True,
            )

        if ctx.gradient_accumulation_fusion:
            if wgrad_compute:
                if weight.main_grad.dtype == torch.float32:
                    module.fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                        total_input, grad_output, weight.main_grad
                    )
                elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
                    module.fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                        total_input, grad_output, weight.main_grad
                    )
                else:
                    raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")

            if hasattr(weight, 'grad_added_to_main_grad'):
                if getattr(weight, 'zero_out_wgrad', False):
                    grad_weight = torch.zeros(
                        weight.main_grad.shape,
                        dtype=input.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                else:
                    grad_weight = torch.empty(
                        weight.main_grad.shape,
                        dtype=input.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                weight.grad_added_to_main_grad = True
            else:
                grad_weight = None
        else:
            grad_weight = _matmul(grad_output.t(), total_input, torch) if wgrad_compute else None

        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if ctx.sequence_parallel:
            handle.wait()
            return sub_grad_input, grad_weight, grad_bias, None, None, None, None, None

        if ctx.allreduce_dgrad:
            handle.wait()

        return grad_input, grad_weight, grad_bias, None, None, None, None, None

    frozen_cls.forward = staticmethod(_frozen_forward)
    frozen_cls.backward = staticmethod(_frozen_backward)
    grad_cls.forward = staticmethod(_grad_forward)
    grad_cls.backward = staticmethod(_grad_backward)

    _PATCHED_MODULES.add(module_name)
    _emit(f"patched MSA tensor_parallel linear dtype alignment in module: {module_name}")
    return True


def _try_patch_loaded_module(trigger: str) -> bool:
    module = sys.modules.get(_TARGET_MODULE)
    patched = _patch_module_obj(module)
    if patched:
        _emit(f"MSA linear dtype patch applied on trigger: {trigger}")
    return patched


def _install_import_hook() -> bool:
    global _IMPORT_HOOK_INSTALLED, _ORIGINAL_IMPORT
    if _IMPORT_HOOK_INSTALLED:
        return True
    try:
        import builtins
    except Exception:
        return False

    _ORIGINAL_IMPORT = builtins.__import__

    def _lmsv_import(name, globals=None, locals=None, fromlist=(), level=0):
        module = _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)
        try:
            if isinstance(name, str) and name.startswith("megatron."):
                _try_patch_loaded_module(trigger=name)
        except Exception:
            pass
        return module

    builtins.__import__ = _lmsv_import
    _IMPORT_HOOK_INSTALLED = True
    _emit("MSA linear dtype lazy import hook installed")
    return True


def apply_msa_linear_dtype_patch() -> bool:
    """Install lazy import patch for MSA Megatron tensor-parallel linear."""
    global _PATCH_APPLIED
    if not _is_enabled():
        return False
    with _PATCH_LOCK:
        if _PATCH_APPLIED:
            _try_patch_loaded_module(trigger="re-apply")
            _emit("apply_msa_linear_dtype_patch called again; hook already active")
            return True
        if not _install_import_hook():
            _emit("failed to install MSA linear dtype import hook")
            return False
        _PATCH_APPLIED = True
        patched_now = _try_patch_loaded_module(trigger="initial")
        if patched_now:
            _emit("MSA linear dtype patch enabled (immediate)")
        else:
            _emit("MSA linear dtype patch armed (waiting for module import)")
        return True
