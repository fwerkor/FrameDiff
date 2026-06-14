#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime patch for shared-weight save/load without modifying source entries."""

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

_DEFAULT_TARGET_MODULES = (
    "core.subgraph",
    "core.graph",
    "utils.runtime.core.subgraph",
    "utils.runtime.core.graph",
)
_PATCHED_MODULES = set()
_SEEN_MISSING_TARGET = set()
_STRUCTURE_REPORTED = set()

_TRUE_VALUES = {"1", "true", "yes", "on"}
_LOG_TRUE_VALUES = {"1", "true", "yes", "on"}
_DISCOVERY_KEYWORDS = [
    "embed",
    "embedding",
    "word",
    "tok",
    "token",
    "vocab",
    "attention",
    "attn",
    "qkv",
    "mlp",
    "proj",
    "norm",
    "output",
    "lm_head",
]


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _is_log_enabled() -> bool:
    return os.getenv("LMSV_PATCH_LOG", "1").strip().lower() in _LOG_TRUE_VALUES


def _rank_hint() -> str:
    for key in ("RANK", "LOCAL_RANK"):
        value = os.getenv(key)
        if value is not None and value != "":
            return f"{key}={value}"
    return "RANK=?"


def _emit(message: str) -> None:
    if not _is_log_enabled():
        return
    try:
        sys.stderr.write(f"[{_timestamp()}] [LMSV_PATCH] [{_rank_hint()}] {message}\n")
        sys.stderr.flush()
    except Exception:
        pass


def _is_rank0() -> bool:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return _rank() == 0


def _debug(message: str) -> None:
    if not _is_rank0():
        return
    try:
        print(f"[LMSV_DEBUG] {message}")
    except Exception:
        pass


def _to_flat_list(value: Any, max_items: int = 16):
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().reshape(-1).tolist()[:max_items]
    except Exception:
        return []
    return []


def _tensor_stats(value: Any) -> dict[str, Any]:
    stats = {"shape": None, "dtype": None, "mean": None, "std": None, "min": None, "max": None, "sample": []}
    try:
        import torch

        if not isinstance(value, torch.Tensor):
            return stats
        tensor = value.detach().float().cpu()
        stats["shape"] = tuple(value.shape)
        stats["dtype"] = str(value.dtype)
        if tensor.numel() > 0:
            stats["mean"] = float(tensor.mean().item())
            stats["std"] = float(tensor.std(unbiased=False).item()) if tensor.numel() > 1 else 0.0
            stats["min"] = float(tensor.min().item())
            stats["max"] = float(tensor.max().item())
            stats["sample"] = _to_flat_list(value, 16)
    except Exception:
        return stats
    return stats


def _fmt_list(items: list[Any]) -> str:
    return "[" + ",".join(str(item) for item in items) + "]"


def _fmt_float(value: Any) -> str:
    try:
        if value is None:
            return "na"
        if isinstance(value, (int, float)):
            return f"{float(value):.6f}"
    except Exception:
        return "na"
    return "na"


def _safe_repr(value: Any) -> str:
    try:
        return str(value)
    except Exception:
        return "<unprintable>"


def _preview(items: list[Any], limit: int = 200) -> list[Any]:
    return list(items[:limit])


def _keyword_matches(items: list[str]) -> dict[str, list[str]]:
    result = {}
    for keyword in _DISCOVERY_KEYWORDS:
        hits = [item for item in items if keyword in item.lower()]
        if hits:
            result[keyword] = hits
    return result


def _debug_list(stage: str, name: str, items: list[Any], limit: int = 200) -> None:
    preview = _preview(items, limit)
    _debug(f"stage={stage} {name}_count={len(items)} {name}={_fmt_list(preview)}")


def _debug_keyword_groups(stage: str, name: str, items: list[str]) -> None:
    groups = _keyword_matches(items)
    if not groups:
        _debug(f"stage={stage} {name}_keyword_groups=[]")
        return
    for keyword, hits in groups.items():
        _debug(f"stage={stage} {name}_keyword={keyword} hits={_fmt_list(_preview(hits, 200))}")


def _fmt_stats(name: str, key: str | None, value: Any, stage: str) -> None:
    if value is None:
        _debug(f"stage={stage} {name}=None key={key}")
        return
    try:
        stats = _tensor_stats(value)
        _debug(
            f"stage={_safe_repr(stage)} {name} key={_safe_repr(key)} "
            f"shape={_safe_repr(stats.get('shape'))} dtype={_safe_repr(stats.get('dtype'))} "
            f"mean={_fmt_float(stats.get('mean'))} std={_fmt_float(stats.get('std'))} "
            f"min={_fmt_float(stats.get('min'))} max={_fmt_float(stats.get('max'))} "
            f"sample={_fmt_list(list(stats.get('sample') or []))}"
        )
    except Exception as exc:
        _debug(
            f"stage={_safe_repr(stage)} {name} key={_safe_repr(key)} "
            f"stats_unavailable error={_safe_repr(exc)}"
        )


def _candidate_named_params(graph: Any):
    try:
        return list(graph.named_parameters())
    except Exception:
        return []


def _candidate_named_modules(graph: Any):
    try:
        return list(graph.named_modules())
    except Exception:
        return []


def _block_num_name(block_num: Any) -> str:
    mapping = {
        0: "input_layernorm",
        1: "self_attention",
        2: "core_attention",
        3: "scale_mask_softmax",
        4: "attention_dropout",
        5: "linear_proj",
        6: "linear_qkv",
        7: "pre_mlp_layernorm",
        8: "mlp",
        9: "mlp.linear_fc1",
        10: "mlp.linear_fc2",
    }
    return mapping.get(block_num, str(block_num))


def _report_wrapper_structure(graph: Any, stage: str) -> None:
    graph_id = id(graph)
    if graph_id in _STRUCTURE_REPORTED:
        return
    _STRUCTURE_REPORTED.add(graph_id)

    graph_type = f"{type(graph).__module__}.{type(graph).__name__}"
    _debug(f"stage={stage} graph_object_type={graph_type}")
    _debug(f"stage={stage} shared_weight_target_modules={_fmt_list(_target_modules())}")

    param_names = [name for name, _ in _candidate_named_params(graph)]
    module_items = _candidate_named_modules(graph)
    module_names = [name for name, _ in module_items]
    _debug(f"stage={stage} graph_named_parameters_all={_fmt_list(param_names)}")
    _debug(f"stage={stage} graph_named_modules_all={_fmt_list(module_names)}")
    _debug_keyword_groups(stage, "graph_named_parameters", param_names)
    _debug_keyword_groups(stage, "graph_named_modules", module_names)

    registered_module_ids = {id(module): name for name, module in module_items}
    if hasattr(graph, "nodes") and isinstance(getattr(graph, "nodes"), dict):
        node_summaries = []
        for node_id in sorted(graph.nodes.keys()):
            node = graph.nodes[node_id]
            block = getattr(node, "block", None)
            block_type = None if block is None else f"{type(block).__module__}.{type(block).__name__}"
            registered_name = registered_module_ids.get(id(block))
            node_summaries.append(
                "id={id} str_op={str_op} block_num={block_num} block_slot={block_slot} "
                "block_type={block_type} registered={registered} registered_name={registered_name}".format(
                    id=node_id,
                    str_op=getattr(node, "str_op", None),
                    block_num=getattr(node, "block_num", None),
                    block_slot=_block_num_name(getattr(node, "block_num", None)),
                    block_type=block_type,
                    registered=registered_name is not None,
                    registered_name=registered_name,
                )
            )
        _debug(f"stage={stage} graph_node_blocks={_fmt_list(node_summaries)}")
        has_embedding_node = any("embed" in str(getattr(node, "str_op", "")).lower() for node in graph.nodes.values())
        has_decoder_node = any("decoder" in str(getattr(node, "str_op", "")).lower() for node in graph.nodes.values())
        _debug(
            f"stage={stage} graph_has_embedding_node={has_embedding_node} "
            f"graph_has_decoder_node={has_decoder_node} graph_node_count={len(graph.nodes)}"
        )


def _find_embedding_candidates(graph: Any) -> list[tuple[str, Any]]:
    names = []
    for name, param in _candidate_named_params(graph):
        lower = name.lower()
        if "embed" in lower or "word_embeddings" in lower:
            names.append((name, param))
    return names


def _find_lm_head_candidate(graph: Any) -> tuple[str | None, Any]:
    for name, param in _candidate_named_params(graph):
        lower = name.lower()
        if "lm_head" in lower or "output_layer" in lower:
            return name, param
    return None, None


def _storage_id(tensor: Any) -> Any:
    try:
        if hasattr(tensor, "untyped_storage"):
            return tensor.untyped_storage().data_ptr()
        if hasattr(tensor, "storage"):
            return tensor.storage().data_ptr()
    except Exception:
        return None
    return None


def _same_tensor_data(a: Any, b: Any) -> bool | None:
    try:
        import torch

        if a is None or b is None:
            return None
        return bool(torch.equal(a.detach().cpu(), b.detach().cpu()))
    except Exception:
        return None


def _snapshot_embedding(graph: Any, stage: str) -> None:
    candidates = _find_embedding_candidates(graph)
    _debug(f"stage={stage} embedding_param_names={_fmt_list([name for name, _ in candidates])}")
    if not candidates:
        _debug(f"stage={stage} embedding_effective_param=None")
        return
    name, param = candidates[0]
    setattr(graph, "_lmsv_debug_embedding_name", name)
    setattr(graph, f"_lmsv_debug_snapshot_{stage}", param.detach().cpu().clone())
    _fmt_stats("embedding_effective_param", name, param, stage)


def _report_embedding_change(graph: Any, from_stage: str, to_stage: str) -> None:
    before = getattr(graph, f"_lmsv_debug_snapshot_{from_stage}", None)
    after = getattr(graph, f"_lmsv_debug_snapshot_{to_stage}", None)
    changed = None if before is None or after is None else (not _same_tensor_data(before, after))
    _debug(f"stage={to_stage} embedding_changed_from_{from_stage}={changed}")


def _report_tie_state(graph: Any, stage: str) -> None:
    emb_candidates = _find_embedding_candidates(graph)
    emb_name, emb_param = emb_candidates[0] if emb_candidates else (None, None)
    lm_name, lm_param = _find_lm_head_candidate(graph)
    same_obj = emb_param is not None and lm_param is not None and emb_param is lm_param
    same_storage = None
    if emb_param is not None and lm_param is not None:
        emb_storage = _storage_id(emb_param)
        lm_storage = _storage_id(lm_param)
        same_storage = emb_storage is not None and emb_storage == lm_storage
    _debug(
        f"stage={stage} tie_check embedding_name={emb_name} lm_head_name={lm_name} "
        f"same_object={same_obj} shared_storage={same_storage}"
    )


def _report_state_dict_keys(graph: Any, state_dict: dict[str, Any], load_result: Any, stage: str) -> None:
    model_keys = [name for name, _ in _candidate_named_params(graph)]
    embedding_param_names = [name for name, _ in _find_embedding_candidates(graph)]
    embedding_sd_keys = sorted([key for key in state_dict.keys() if "embed" in key.lower() or "word_embeddings" in key.lower()])
    matched = sorted(set(model_keys) & set(state_dict.keys()))
    missing = list(getattr(load_result, "missing_keys", []))
    unexpected = list(getattr(load_result, "unexpected_keys", []))
    _debug(f"stage={stage} shared_weight_embedding_keys={_fmt_list(embedding_sd_keys)}")
    _debug(f"stage={stage} model_embedding_param_names={_fmt_list(embedding_param_names)}")
    _debug(f"stage={stage} load_matched_keys={_fmt_list(matched)}")
    _debug(f"stage={stage} load_missing_keys={_fmt_list(missing)}")
    _debug(f"stage={stage} load_unexpected_keys={_fmt_list(unexpected)}")
    effective_name = getattr(graph, "_lmsv_debug_embedding_name", None)
    matched_key = effective_name if effective_name in state_dict else None
    if matched_key is None and effective_name is not None:
        suffix_hits = [key for key in state_dict.keys() if key.endswith(effective_name)]
        matched_key = suffix_hits[0] if suffix_hits else None
    _debug(f"stage={stage} effective_embedding_match model_param={effective_name} shared_weight_key={matched_key}")
    if matched == ["output_layer.weight"]:
        _debug("stage=after_load load_match_reason=only_output_layer_weight_matched")


def _report_model_param_discovery(graph: Any, stage: str) -> None:
    model_param_names = [name for name, _ in _candidate_named_params(graph)]
    _debug_list(stage, "model_named_parameters", model_param_names, limit=200)
    _debug_keyword_groups(stage, "model_named_parameters", model_param_names)
    state_dict_owner = getattr(getattr(graph, "state_dict", None), "__qualname__", None)
    load_state_dict_owner = getattr(getattr(graph, "load_state_dict", None), "__qualname__", None)
    _debug(f"stage={stage} graph_state_dict_impl={state_dict_owner}")
    _debug(f"stage={stage} graph_load_state_dict_impl={load_state_dict_owner}")
    has_embedding = any(
        keyword in name.lower()
        for name in model_param_names
        for keyword in ("embed", "embedding", "word", "tok", "token", "vocab")
    )
    _debug(f"stage={stage} model_has_suspected_embedding_param={has_embedding}")


def _report_state_dict_discovery(state_dict: dict[str, Any], stage: str) -> None:
    keys = list(state_dict.keys())
    _debug_list(stage, "shared_weight_keys", keys, limit=200)
    _debug_keyword_groups(stage, "shared_weight_keys", keys)
    coverage = {
        "embedding": any(keyword in key.lower() for key in keys for keyword in ("embed", "embedding", "word", "tok", "token", "vocab")),
        "attention": any(keyword in key.lower() for key in keys for keyword in ("attention", "attn", "qkv", "proj")),
        "mlp": any(keyword in key.lower() for key in keys for keyword in ("mlp", "fc1", "fc2", "gate", "up_proj", "down_proj")),
        "output_layer": any(keyword in key.lower() for key in keys for keyword in ("output_layer", "lm_head")),
    }
    _debug(
        f"stage={stage} shared_weight_coverage "
        f"embedding={coverage['embedding']} attention={coverage['attention']} "
        f"mlp={coverage['mlp']} output_layer={coverage['output_layer']}"
    )
    has_embedding = any(
        keyword in key.lower()
        for key in keys
        for keyword in ("embed", "embedding", "word", "tok", "token", "vocab")
    )
    _debug(f"stage={stage} shared_weight_has_suspected_embedding_key={has_embedding}")
    if len(keys) <= 8:
        _debug(f"stage={stage} shared_weight_small_keyset={_fmt_list(keys)}")
    if keys == ["output_layer.weight"]:
        _debug(f"stage={stage} shared_weight_reason=only_output_layer_weight_was_saved")


def _is_enabled() -> bool:
    return os.getenv("LMSV_ENABLE_SUBMODULE_SHARED_WEIGHT_PATCH", "0").strip().lower() in _TRUE_VALUES


def _target_modules() -> list[str]:
    raw = os.getenv("LMSV_SHARED_WEIGHT_TARGET_MODULES", "").strip()
    if not raw:
        return list(_DEFAULT_TARGET_MODULES)

    modules = []
    for part in raw.replace(";", ",").split(","):
        item = part.strip()
        if item:
            modules.append(item)
    return modules or list(_DEFAULT_TARGET_MODULES)


def _matches_target_module(name: str) -> bool:
    if not isinstance(name, str) or not name:
        return False

    targets = _target_modules()
    if name in targets:
        return True

    for target in targets:
        if not target:
            continue
        if name.endswith(f".{target}") or target.endswith(f".{name}"):
            return True
    return False


def _shared_path() -> str:
    return os.getenv("LMSV_SHARED_WEIGHT_PATH", "").strip()


def _shared_mode() -> str:
    return os.getenv("LMSV_SHARED_WEIGHT_MODE", "").strip().lower()


def _shared_timeout_s() -> int:
    raw = os.getenv("LMSV_SHARED_WEIGHT_WAIT_TIMEOUT", "300").strip() or "300"
    try:
        return max(1, int(raw))
    except Exception:
        return 300


def _barrier() -> None:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        pass


def _rank() -> int:
    value = os.getenv("RANK", "").strip()
    if value == "":
        return 0
    try:
        return int(value)
    except Exception:
        return 0


def _sync_shared_weights(graph: Any) -> None:
    mode = _shared_mode()
    path = _shared_path()
    if mode not in ("save", "load") or not path:
        return

    import torch

    if mode == "save":
        if _rank() == 0:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            state_dict = graph.state_dict()
            _report_wrapper_structure(graph, "save")
            _report_model_param_discovery(graph, "save")
            _report_state_dict_discovery(state_dict, "save")
            cpu_state = {}
            for key, value in state_dict.items():
                item = value.detach() if hasattr(value, "detach") else value
                item = item.cpu() if hasattr(item, "cpu") else item
                cpu_state[key] = item
            torch.save(cpu_state, path)
            _emit(f"shared weight saved: {path} (tensors={len(cpu_state)})")
        _barrier()
        return

    # load mode
    if _rank() == 0:
        deadline = time.time() + _shared_timeout_s()
        while time.time() < deadline:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                break
            time.sleep(1)
        if not os.path.exists(path):
            raise FileNotFoundError(f"shared weight not found: {path}")
    _barrier()

    _snapshot_embedding(graph, "init_before_load")
    _report_tie_state(graph, "init_before_load")
    _report_wrapper_structure(graph, "init_before_load")
    _report_model_param_discovery(graph, "init_before_load")
    state_dict = torch.load(path, map_location="cpu")
    _report_state_dict_discovery(state_dict, "load_file")
    target_state = graph.state_dict()
    compatible_state = {}
    skipped_mismatched = []
    for key, value in state_dict.items():
        target_value = target_state.get(key)
        if (
            target_value is not None
            and hasattr(value, "shape")
            and hasattr(target_value, "shape")
            and tuple(value.shape) != tuple(target_value.shape)
        ):
            skipped_mismatched.append((key, tuple(value.shape), tuple(target_value.shape)))
            continue
        compatible_state[key] = value
    if skipped_mismatched:
        preview = ", ".join(
            f"{key}: {src}->{dst}"
            for key, src, dst in skipped_mismatched[:8]
        )
        _emit(
            f"weight load skipped shape-mismatched tensors: "
            f"count={len(skipped_mismatched)} preview=[{preview}]"
        )
    load_result = graph.load_state_dict(compatible_state, strict=False)
    _report_state_dict_keys(graph, compatible_state, load_result, "after_load")
    _snapshot_embedding(graph, "after_load")
    _report_embedding_change(graph, "init_before_load", "after_load")
    _report_tie_state(graph, "after_load")
    missing = list(getattr(load_result, "missing_keys", []))
    unexpected = list(getattr(load_result, "unexpected_keys", []))
    _emit(f"shared weight loaded: {path} (missing={len(missing)}, unexpected={len(unexpected)})")
    _barrier()


def _patch_module_obj(module_name: str, module: Any) -> bool:
    if module_name in _PATCHED_MODULES:
        return False
    if module is None:
        return False

    graph_cls = getattr(module, "Graph", None)
    if graph_cls is None:
        if module_name not in _SEEN_MISSING_TARGET:
            _SEEN_MISSING_TARGET.add(module_name)
            _emit(f"module loaded but Graph missing: {module_name}")
        return False

    original = getattr(graph_cls, "load", None)
    if original is None:
        if module_name not in _SEEN_MISSING_TARGET:
            _SEEN_MISSING_TARGET.add(module_name)
            _emit(f"module loaded but Graph.load missing: {module_name}")
        return False

    if getattr(original, "_lmsv_shared_wrapped", False):
        _PATCHED_MODULES.add(module_name)
        return True

    def wrapped(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        try:
            _sync_shared_weights(self)
        except Exception as exc:
            _emit(f"shared weight sync failed: {exc}")
        return result

    wrapped._lmsv_shared_wrapped = True
    wrapped._lmsv_shared_original = original
    graph_cls.load = wrapped

    original_forward = getattr(graph_cls, "forward", None)
    if original_forward is not None and not getattr(original_forward, "_lmsv_shared_forward_wrapped", False):
        def wrapped_forward(self, *args, **kwargs):
            if not getattr(self, "_lmsv_debug_forward_snapshot_done", False):
                _snapshot_embedding(self, "before_first_forward")
                _report_embedding_change(self, "after_load", "before_first_forward")
                _report_tie_state(self, "before_first_forward")
                self._lmsv_debug_forward_snapshot_done = True
            return original_forward(self, *args, **kwargs)

        wrapped_forward._lmsv_shared_forward_wrapped = True
        wrapped_forward._lmsv_shared_forward_original = original_forward
        graph_cls.forward = wrapped_forward
    _PATCHED_MODULES.add(module_name)
    _emit(f"patched Graph.load in module: {module_name}")
    return True


def _try_patch_loaded_module(trigger: str) -> bool:
    patched_any = False
    for module_name in _target_modules():
        module = sys.modules.get(module_name)
        if _patch_module_obj(module_name, module):
            patched_any = True
    if patched_any:
        _emit(f"shared weight patch applied on trigger: {trigger}")
    return patched_any


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
            if _matches_target_module(name):
                _try_patch_loaded_module(trigger=name)
        except Exception:
            pass
        return module

    builtins.__import__ = _lmsv_import
    _IMPORT_HOOK_INSTALLED = True
    _emit("shared weight lazy import hook installed")
    return True


def apply_shared_weight_patch() -> bool:
    """Install lazy patch hook and patch Graph.load on configured target modules."""
    global _PATCH_APPLIED
    if not _is_enabled():
        return False

    with _PATCH_LOCK:
        if _PATCH_APPLIED:
            _try_patch_loaded_module(trigger="re-apply")
            _emit("apply_shared_weight_patch called again; hook already active")
            return True

        if not _install_import_hook():
            _emit("failed to install shared weight import hook")
            return False

        _PATCH_APPLIED = True
        patched_now = _try_patch_loaded_module(trigger="initial")
        if patched_now:
            _emit("shared weight patch enabled (immediate)")
        else:
            _emit("shared weight patch armed (waiting for module import)")
        return True
