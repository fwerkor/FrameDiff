import os 
os.environ["PYTORCH_NPU_FORCE_FALLBACK"] = "1"

from torch import Tensor
import copy
import sys
import torch
import torch.nn.functional as F
import json
import torch_npu
from ruamel.yaml import YAML
from megatron.core import tensor_parallel
from megatron.core import parallel_state
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_layer import TransformerConfig
from megatron.core.models.common.language_module.language_module import LanguageModule
from typing import Optional

from megatron.core.models.common.embeddings.rotary_pos_embedding import (
    # MultimodalRotaryEmbedding,
    RotaryEmbedding,
)
# from megatron.core.inference.contexts import StaticInferenceContext

# distributed settings
# random_seed = 42
# torch.distributed.init_process_group(
#     backend="nccl",
#     # init_method="tcp://127.0.0.1:12340",
#     rank=0,
#     world_size=1,
# )
# parallel_state.initialize_model_parallel(1, 1)
# model_parallel_cuda_manual_seed(random_seed)

sys.path.append(".")
sys.path.append("..")

from utils.runtime import model_helpers
from utils.runtime.common_utils import *
from utils.runtime.OperatorSet import insert_operators
from utils.runtime.debug_utils import (
    debug_message,
    debug_parameter_summary,
    debug_scalar,
    debug_tensor_summary,
    is_rank0,
    mark_weights_logged,
    should_log_full,
    should_log_heavy,
    should_log_weights_once,
)
from utils.runtime.fullnet_trace import (
    COMPONENT_CATALOG,
    classify_fullnet_component,
    maybe_perturb_tensor,
    trace_components_manifest,
    trace_event,
    trace_loss,
    trace_module_weights,
    trace_nested_tensors,
    trace_tensor,
)
import random
import math


def _extract_node_transformer_config(config_data):
    """Return a TransformerConfig-compatible dict from full or flat node config."""
    if not isinstance(config_data, dict):
        return {}
    if "after" in config_data and isinstance(config_data["after"], dict):
        config_data = config_data["after"]

    cfg = model_helpers.extract_graph_transformer_config_from_yaml(config_data)
    valid_fields = set(TransformerConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in cfg.items() if k in valid_fields}
    _prepare_transformer_config_dict(filtered)
    return filtered


def _lmsv_exact_gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _normalize_init_method(config: dict) -> None:
    init_method = config.get("init_method")
    if init_method is None:
        config["init_method"] = torch.nn.init.xavier_uniform_
    elif init_method == "torch.nn.init.xavier_uniform_":
        config["init_method"] = torch.nn.init.xavier_uniform_


def _normalize_torch_dtype_fields(config: dict) -> None:
    dtype_map = {
        "torch.float16": torch.float16,
        "torch.float32": torch.float32,
        "torch.float64": torch.float64,
        "torch.bfloat16": torch.bfloat16,
        "torch.half": torch.half,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "half": torch.half,
    }
    for key in ("params_dtype", "autocast_dtype", "pipeline_dtype"):
        value = config.get(key)
        if isinstance(value, str) and value in dtype_map:
            config[key] = dtype_map[value]


def _safe_int(value, default=0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _normalize_context_parallel_fields(config: dict) -> None:
    if not isinstance(config, dict):
        return
    cp = _safe_int(config.get("context_parallel_size", 1), 1)
    num_layers = _safe_int(config.get("num_layers", 0), 0)
    if cp <= 1 or num_layers <= 0:
        return
    value = config.get("cp_comm_type")
    if value is None or value == "":
        config["cp_comm_type"] = ["p2p"] * num_layers
    elif isinstance(value, str):
        config["cp_comm_type"] = [value] * num_layers
    elif isinstance(value, (list, tuple)) and len(value) == 1 and num_layers > 1:
        config["cp_comm_type"] = list(value) * num_layers


def _pad_context_parallel_config_for_layer_numbering(config) -> None:
    if config is None:
        return
    cp = _safe_int(getattr(config, "context_parallel_size", 1), 1)
    if cp <= 1:
        return
    value = getattr(config, "cp_comm_type", None)
    num_layers = _safe_int(getattr(config, "num_layers", 0), 0)
    if num_layers <= 0:
        return
    if value is None or value == "":
        value = ["p2p"] * num_layers
    elif isinstance(value, str):
        value = [value] * num_layers
    elif isinstance(value, tuple):
        value = list(value)
    elif isinstance(value, list):
        value = list(value)
    else:
        return
    if len(value) == 0:
        value = ["p2p"] * num_layers
    while len(value) <= num_layers:
        value.append(value[-1])
    config.cp_comm_type = value


def _maybe_get_parallel_world_size(func_name: str) -> int | None:
    try:
        func = getattr(parallel_state, func_name, None)
        if func is None:
            return None
        value = int(func())
        return value if value > 0 else None
    except Exception:
        return None


def _sync_runtime_parallel_fields(config: dict) -> None:
    if not isinstance(config, dict):
        return
    mapping = {
        "tensor_model_parallel_size": "get_tensor_model_parallel_world_size",
        "pipeline_model_parallel_size": "get_pipeline_model_parallel_world_size",
        "context_parallel_size": "get_context_parallel_world_size",
        "expert_model_parallel_size": "get_expert_model_parallel_world_size",
    }
    for key, func_name in mapping.items():
        value = _maybe_get_parallel_world_size(func_name)
        if value is not None:
            config[key] = value


def _stabilize_unsupported_moe_bias(config: dict) -> None:
    if not isinstance(config, dict):
        return
    has_moe = any(
        _safe_int(config.get(key, 0), 0) > 0
        for key in ("num_moe_experts", "num_experts", "moe_ffn_hidden_size")
    ) or bool(config.get("moe_grouped_gemm"))
    if has_moe and bool(config.get("add_bias_linear", False)):
        # MindSpeed/Megatron rejects add_bias_linear=True for MoE layers.
        # Keep the run executable and make the linear-bias variant a no-op for MoE models.
        config["add_bias_linear"] = False


def _config_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'on'}
    return bool(value)


def _sync_runtime_precision_fields(config: dict) -> None:
    if not isinstance(config, dict):
        return
    try:
        from megatron.training import get_args
        args = get_args()
    except Exception:
        return
    for key in ('bf16', 'fp16', 'reuse_fp32_param', 'fp32_residual_connection'):
        if hasattr(args, key) and _config_bool(getattr(args, key)):
            config[key] = True


def _sync_runtime_moe_fields(config: dict) -> None:
    if not isinstance(config, dict):
        return
    try:
        from megatron.training import get_args
        args = get_args()
    except Exception:
        return
    for source_key in ('num_moe_experts', 'num_experts'):
        if hasattr(args, source_key):
            value = getattr(args, source_key)
            if value not in (None, '', 0):
                config['num_moe_experts'] = value
                config.setdefault('num_experts', value)
                break


def _normalize_feature_dependencies(config: dict) -> None:
    if not isinstance(config, dict):
        return
    _sync_runtime_precision_fields(config)
    _sync_runtime_moe_fields(config)
    has_low_precision = _config_bool(config.get('bf16')) or _config_bool(config.get('fp16'))
    needs_low_precision = _config_bool(config.get('reuse_fp32_param')) or _config_bool(config.get('fp32_residual_connection'))
    if needs_low_precision and not has_low_precision:
        config['bf16'] = True
    if config.get('recompute_num_layers') not in (None, '') and not config.get('recompute_granularity'):
        config['recompute_granularity'] = 'full'
    if config.get('recompute_granularity') == 'full':
        if not config.get('recompute_method'):
            config['recompute_method'] = 'uniform'
        if config.get('recompute_num_layers') in (None, '', 0):
            config['recompute_num_layers'] = 1

def _prepare_transformer_config_dict(config: dict, *, sync_parallel: bool = True) -> dict:
    if not isinstance(config, dict):
        return config
    if sync_parallel:
        _sync_runtime_parallel_fields(config)
    _normalize_init_method(config)
    _normalize_torch_dtype_fields(config)
    _normalize_feature_dependencies(config)
    _stabilize_unsupported_moe_bias(config)
    _normalize_context_parallel_fields(config)
    return config


def reshape_tensor_nd(
        input_tensor: torch.Tensor,
        target_shape: tuple,
        fill_value: float = 0
) -> torch.Tensor:

    input_shape = input_tensor.shape
    input_numel = input_tensor.numel()
    target_numel = torch.prod(torch.tensor(target_shape)).item()

    # Step 1: 将输入展平为1D向量
    flattened = input_tensor.flatten()

    # Step 2: 处理元素数量差异
    if target_numel > input_numel:
        # 填充不足部分
        padded = torch.cat([
            flattened,
            torch.full((target_numel - input_numel,), fill_value, dtype=input_tensor.dtype, device=input_tensor.device)
        ])
        output = padded
    elif target_numel < input_numel:
        # 裁剪多余部分
        output = flattened[:target_numel]
    else:
        output = flattened

    # Step 3: 调整为目标形状
    return output.reshape(target_shape)

class Node:
    def __init__(self, config, index=-1):
        super().__init__()
        self.from_nodes = []
        self.to_nodes = []
        self.layer_limits = []
        self.op = None
        self.id = index
        self.origin_id = -1
        self.state = 'none'
        self.is_des = False
        self.is_src = False
        self.visit_count = 0
        self.succ_count = 0
        self.str_op = 'empty'
        self.params = {}
        self.input_shape = []
        self.output_shape = []
        self.in_degree = len(self.from_nodes)
        self.out_degree = len(self.to_nodes)
        self.config = config
        self.block = None
        
class Graph(LanguageModule):

    def __init__(
            self,
            config_path: str = None,
            config_dict: dict = None,
            nums: list = None,
            mutated_nodes: dict = None,
    ):
        # 支持两种初始化方式：配置文件路径或配置字典
        if config_dict is not None:
            # 使用配置字典初始化
            model_config = config_dict
            if 'config' in model_config:
                _prepare_transformer_config_dict(model_config['config'])
        elif config_path is not None:
            # 使用配置文件路径初始化
            config_path = model_helpers.resolve_repo_path(config_path)
            yaml = YAML()
            with open(config_path, 'r', encoding='utf-8') as file:
                model_config = yaml.load(file)
                _prepare_transformer_config_dict(model_config['config'])
        else:
            raise ValueError("必须提供 config_path 或 config_dict 中的一个")

        valid_fields = set(TransformerConfig.__dataclass_fields__.keys())
        cfg_dict = {k: v for k, v in model_config["config"].items() if k in valid_fields}
        _prepare_transformer_config_dict(cfg_dict)
        transformerblock_config = TransformerConfig(**cfg_dict)
        _pad_context_parallel_config_for_layer_numbering(transformerblock_config)
        model_config["config"] = transformerblock_config
        self.total_config = model_config
        config = dict()
        for key, value in model_config.items():
            if key != "config":
                config[key] = value
        super().__init__(config=transformerblock_config)
        # Use the already-normalized model TransformerConfig for placeholder nodes.
        # Creating a second hard-coded TransformerConfig here allows MindSpeed's
        # global argument wrapper to inject CP args (for example cp_comm_type)
        # into a mismatched 24-layer config before Graph.load runs.
        init_config = transformerblock_config
        # nums.append(nums[-1]+1)
        self.nodes = dict(zip([id for id in nums], [Node(config=init_config, index=id) for id in nums]))  # ǰ���һ����

        # 修改：保存变异节点信息作为实例属性
        self.mutated_nodes = mutated_nodes if mutated_nodes is not None else {}

        self.embedding = None
        if self.total_config['position_embedding_type'] == 'rope' and not self.config.multi_latent_attention:
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=self.total_config['rotary_percent'],
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=self.total_config['seq_len_interpolation_factor'],
                rotary_base=self.total_config['rotary_base'],
                rope_scaling=self.total_config['rope_scaling'],
                rope_scaling_factor=self.total_config['rope_scaling_factor'],
                use_cpu_initialization=self.config.use_cpu_initialization,
            )
        # elif self.total_config['position_embedding_type'] == 'mrope' and not self.config.multi_latent_attention:
        #     self.rotary_pos_emb = MultimodalRotaryEmbedding(
        #         kv_channels=self.config.kv_channels,
        #         rotary_percent=self.total_config['rotary_percent'],
        #         rotary_interleaved=self.config.rotary_interleaved,
        #         seq_len_interpolation_factor=self.total_config['seq_len_interpolation_factor'],
        #         rotary_base=self.total_config['rotary_base'],
        #     )
        #     self.mrope_section = self.config.mrope_section
        #     assert (
        #             self.mrope_section is not None
        #     ), "mrope require mrope_section setting, but we got None from TransformerConfig"
        #     self.rotary_pos_emb_cache = {}
        self.pre_process = True
        self.post_process = True
        if self.post_process:

            if self.config.defer_embedding_wgrad_compute:
                # The embedding activation buffer preserves a reference to the input activations
                # of the final embedding projection layer GEMM. It will hold the activations for
                # all the micro-batches of a global batch for the last pipeline stage. Once we are
                # done with all the back props for all the microbatches for the last pipeline stage,
                # it will be in the pipeline flush stage. During this pipeline flush we use the
                # input activations stored in embedding activation buffer and gradient outputs
                # stored in gradient buffer to calculate the weight gradients for the embedding
                # final linear layer.
                self.embedding_activation_buffer = []
                self.grad_output_buffer = []
            else:
                self.embedding_activation_buffer = None
                self.grad_output_buffer = None

            self.share_embeddings_and_output_weights = False

            if self.config._cpu_offloading_context == 'None':
                self.config._cpu_offloading_context = None
            self.output_layer = tensor_parallel.ColumnParallelLinear(
                self.config.hidden_size,
                self.total_config['vocab_size'],
                config=self.config,
                init_method=self.config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=False,
                skip_weight_param_allocation=self.pre_process
                                             and self.share_embeddings_and_output_weights,
                embedding_activation_buffer=self.embedding_activation_buffer,
                grad_output_buffer=self.grad_output_buffer,
            )

    @staticmethod
    def _debug_preview(items, limit=200):
        preview = list(items[:limit])
        return "[" + ",".join(str(item) for item in preview) + "]"

    def _emit_load_structure_summary_once(self):
        if getattr(self, "_lmsv_load_structure_logged", False):
            return
        if not is_rank0() or not (should_log_full() or should_log_heavy()):
            return

        node_summaries = []
        for node_id in sorted(self.nodes.keys()):
            node = self.nodes[node_id]
            block = getattr(node, "block", None)
            block_type = None if block is None else f"{type(block).__module__}.{type(block).__name__}"
            node_summaries.append(
                "id={id},str_op={str_op},from={from_nodes},to={to_nodes},block_type={block_type},"
                "hidden_size={hidden_size},heads={heads}".format(
                    id=node_id,
                    str_op=getattr(node, "str_op", None),
                    from_nodes=getattr(node, "from_nodes", None),
                    to_nodes=getattr(node, "to_nodes", None),
                    block_type=block_type,
                    hidden_size=getattr(getattr(node, "config", None), "hidden_size", None),
                    heads=getattr(getattr(node, "config", None), "num_attention_heads", None),
                )
            )

        named_param_names = [name for name, _ in self.named_parameters()]
        state_dict_keys = list(self.state_dict().keys())
        has_embedding = any("embedding" in str(getattr(node, "str_op", "")).lower() for node in self.nodes.values())
        has_decoder = any("decoder" in str(getattr(node, "str_op", "")).lower() for node in self.nodes.values())
        has_output = hasattr(self, "output_layer") and self.output_layer is not None

        print(f"[LMSV_DEBUG] step=-1 graph.load.type={type(self).__module__}.{type(self).__name__}")
        print(
            f"[LMSV_DEBUG] step=-1 graph.load.node_count={len(self.nodes)} "
            f"graph.load.nodes={self._debug_preview(node_summaries, limit=200)}"
        )
        print(
            f"[LMSV_DEBUG] step=-1 graph.load.named_parameters_count={len(named_param_names)} "
            f"graph.load.named_parameters={self._debug_preview(named_param_names, limit=200)}"
        )
        print(
            f"[LMSV_DEBUG] step=-1 graph.load.state_dict_keys_count={len(state_dict_keys)} "
            f"graph.load.state_dict_keys={self._debug_preview(state_dict_keys, limit=200)}"
        )
        print(
            f"[LMSV_DEBUG] step=-1 graph.load.coverage "
            f"embedding={has_embedding} decoder={has_decoder} output_layer={has_output}"
        )
        self._lmsv_load_structure_logged = True

    @staticmethod
    def _stabilize_decoder_mlp_config(config):
        if config is None:
            return
        if hasattr(config, "bias_activation_fusion"):
            config.bias_activation_fusion = False
        if hasattr(config, "use_fused_swiglu"):
            config.use_fused_swiglu = False
        if hasattr(config, "bias_swiglu_fusion"):
            config.bias_swiglu_fusion = False
        if hasattr(config, "use_fused_mlp"):
            config.use_fused_mlp = False

    def _stabilize_graph_mlp_configs(self):
        self._stabilize_decoder_mlp_config(getattr(self, "config", None))
        for node in getattr(self, "nodes", {}).values():
            if "decoder" in str(getattr(node, "str_op", "")).lower():
                self._stabilize_decoder_mlp_config(getattr(node, "config", None))

    @staticmethod
    def _stabilize_decoder_mlp_module(decoder_block):
        if decoder_block is None or not hasattr(decoder_block, "layers"):
            return
        for layer in getattr(decoder_block, "layers", []):
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue
            activation = getattr(mlp, "activation_func", None)
            config = getattr(mlp, "config", None)
            config_activation = getattr(config, "activation_func", None) if config is not None else None
            activation_name = getattr(activation, "__name__", str(activation)).lower()
            config_activation_name = getattr(config_activation, "__name__", str(config_activation)).lower()
            if "gelu" in activation_name or "gelu" in config_activation_name:
                mlp.activation_func = _lmsv_exact_gelu
                if config is not None:
                    config.activation_func = _lmsv_exact_gelu

    def set_input_tensor(self, input_tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for gpt/bert'
        self.nodes[1].block.set_input_tensor(input_tensor[0])

    def forward(self,
            input_ids: Tensor = None,
            input_data: Tensor = None,
            position_ids: Tensor = None,
            attention_mask: Tensor = None,
            labels: Tensor = None,
            inference_context = None,
            loss_mask: Optional[Tensor] = None,
            debug=True,
            log_file: str = "logs/graph_forward_debug.log"):
        """
        前向传播方法
        """
        def _find_first_node_block(*keywords):
            for node_id in sorted(self.nodes.keys()):
                node = self.nodes[node_id]
                op_name = getattr(node, "str_op", "").lower()
                if all(keyword in op_name for keyword in keywords) and getattr(node, "block", None) is not None:
                    return node.block
            return None

        def _find_first_param(module, preferred_keywords):
            if module is None or not hasattr(module, "named_parameters"):
                return None, None
            lowered = [keyword.lower() for keyword in preferred_keywords]
            fallback = None
            for name, param in module.named_parameters():
                lower_name = name.lower()
                if fallback is None and "weight" in lower_name:
                    fallback = (name, param)
                if "weight" in lower_name and any(keyword in lower_name for keyword in lowered):
                    return name, param
            if fallback is not None:
                return fallback
            return None, None

        def _emit_weight_debug_once():
            if not should_log_weights_once():
                return

            embedding_block = self.embedding or _find_first_node_block("embedding")
            _, embedding_weight = _find_first_param(embedding_block, ["word", "embedding"])
            debug_parameter_summary("weight.embedding", embedding_weight, max_items=8)

            first_decoder_block = _find_first_node_block("decoder")
            attn_name, attn_weight = _find_first_param(
                first_decoder_block,
                ["self_attention", "attention", "attn", "query_key_value", "qkv"],
            )
            debug_parameter_summary("weight.first_attention", attn_weight, max_items=8)
            if attn_name is not None:
                debug_scalar("weight.first_attention_name", attn_name)

            mlp_name, mlp_weight = _find_first_param(
                first_decoder_block,
                ["mlp", "dense_h_to_4h", "up_proj", "gate_proj", "fc1", "w1", "w3"],
            )
            debug_parameter_summary("weight.first_mlp", mlp_weight, max_items=8)
            if mlp_name is not None:
                debug_scalar("weight.first_mlp_name", mlp_name)

            norm_name, norm_weight = _find_first_param(self, ["final_norm", "final_layernorm", "norm"])
            debug_parameter_summary("weight.final_norm", norm_weight, max_items=8)
            if norm_name is not None:
                debug_scalar("weight.final_norm_name", norm_name)

            lm_head_name, lm_head_weight = _find_first_param(self.output_layer if hasattr(self, "output_layer") else self, ["output_layer", "lm_head"])
            debug_parameter_summary("weight.lm_head", lm_head_weight, max_items=8)
            if lm_head_name is not None:
                debug_scalar("weight.lm_head_name", lm_head_name)

            mark_weights_logged()

        decoder_count = 0

        def _emit_layer_summary(name, tensor):
            if should_log_full() or should_log_heavy():
                debug_tensor_summary(name, tensor, max_items=8, include_stats=True)

        def _extract_first_tensor(value):
            if isinstance(value, torch.Tensor):
                return value
            if isinstance(value, (tuple, list)):
                for item in value:
                    tensor = _extract_first_tensor(item)
                    if tensor is not None:
                        return tensor
            return None

        def _first_floating_parameter_dtype(module):
            if module is None or not hasattr(module, "parameters"):
                return None
            for param in module.parameters():
                dtype = getattr(param, "dtype", None)
                if dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
                    return dtype
                dtype_text = str(dtype)
                if dtype_text in {"Float16", "Float32", "Float64", "BFloat16", "torch.float16", "torch.float32", "torch.float64", "torch.bfloat16"}:
                    return dtype
            return None

        def _cast_floating_tensor_to_dtype(tensor, target_dtype):
            if tensor is None or target_dtype is None:
                return tensor
            try:
                is_floating = bool(getattr(tensor, "is_floating_point", lambda: False)())
            except Exception:
                is_floating = str(getattr(tensor, "dtype", "")).lower() in {
                    "float16", "float32", "float64", "bfloat16",
                    "torch.float16", "torch.float32", "torch.float64", "torch.bfloat16",
                    "float16", "float32", "float64", "bfloat16",
                }
            if not is_floating:
                return tensor
            if getattr(tensor, "dtype", None) == target_dtype:
                return tensor
            try:
                return tensor.to(dtype=target_dtype)
            except TypeError:
                return tensor.to(target_dtype)
            except Exception:
                return tensor

        def _component_for_module(module_name, module):
            return classify_fullnet_component(module_name, module)

        def _component_for_name(component_name):
            for component in COMPONENT_CATALOG:
                if component["name"] == component_name:
                    return int(component["id"]), str(component["name"])
            return -1, str(component_name)

        def _trace_io(component_id, component_name, base_name, value, stage, node_id=None, extra=None):
            trace_nested_tensors(
                component_id,
                component_name,
                base_name,
                value,
                stage=stage,
                node_id=node_id,
                extra=extra,
            )

        def _trace_weight(component_id, component_name, module, stage, node_id=None, module_name=None, extra=None):
            trace_module_weights(
                component_id,
                component_name,
                module,
                stage=stage,
                node_id=node_id,
                module_name=module_name,
                extra=extra,
            )

        def _register_block0_compare_hooks(decoder_block):
            if not (should_log_full() or should_log_heavy()):
                return []
            try:
                layer0 = decoder_block.layers[0]
            except Exception:
                return []

            mlp_module = getattr(layer0, "mlp", None)
            shadow_cache = {"fc1_output": None, "fc1_bias": None}

            def _resolve_activation_func(module):
                activation = getattr(module, "activation_func", None)
                if callable(activation):
                    return activation
                config = getattr(module, "config", None)
                activation = getattr(config, "activation_func", None)
                if callable(activation):
                    return activation
                if isinstance(activation, str):
                    lowered = activation.lower()
                    if lowered == "silu":
                        return F.silu
                    if lowered == "gelu":
                        return F.gelu
                return F.gelu

            def _emit_mlp_config_once():
                if mlp_module is None or getattr(mlp_module, "_lmsv_debug_config_logged", False):
                    return
                config = getattr(mlp_module, "config", None)
                activation = getattr(mlp_module, "activation_func", None)
                if activation is None and config is not None:
                    activation = getattr(config, "activation_func", None)
                activation_name = getattr(activation, "__name__", str(activation))
                debug_scalar(
                    "block0.mlp.config",
                    {
                        "activation_func": activation_name,
                        "gated_linear_unit": getattr(config, "gated_linear_unit", None),
                        "bias_activation_fusion": getattr(config, "bias_activation_fusion", None),
                        "bias_swiglu_fusion": getattr(config, "bias_swiglu_fusion", None),
                        "use_fused_swiglu": getattr(config, "use_fused_swiglu", None),
                        "use_fused_mlp": getattr(config, "use_fused_mlp", None),
                    },
                )
                mlp_module._lmsv_debug_config_logged = True

            def _compute_shadow_activation(actual_fc2_input):
                fc1_output = shadow_cache.get("fc1_output")
                fc1_bias = shadow_cache.get("fc1_bias")
                if fc1_output is None:
                    debug_tensor_summary("block0.mlp.linear_fc1.bias", None, max_items=8, include_stats=True)
                    debug_tensor_summary("block0.mlp.shadow.after_bias", None, max_items=8, include_stats=True)
                    debug_tensor_summary("block0.mlp.shadow.after_activation", None, max_items=8, include_stats=True)
                    return

                debug_tensor_summary("block0.mlp.linear_fc1.bias", fc1_bias, max_items=8, include_stats=True)
                trace_tensor(13, "ffn_block", "block0.mlp.linear_fc1.bias", fc1_bias, stage="shadow_activation")

                shadow = fc1_output
                if fc1_bias is not None:
                    shadow = shadow + fc1_bias
                debug_tensor_summary("block0.mlp.shadow.after_bias", shadow, max_items=8, include_stats=True)
                trace_tensor(13, "ffn_block", "block0.mlp.shadow.after_bias", shadow, stage="shadow_activation")

                activation_func = _resolve_activation_func(mlp_module)
                config = getattr(mlp_module, "config", None)
                if getattr(config, "gated_linear_unit", False):
                    gate, value = torch.chunk(shadow, 2, dim=-1)
                    debug_tensor_summary("block0.mlp.shadow.gate", gate, max_items=8, include_stats=True)
                    debug_tensor_summary("block0.mlp.shadow.value", value, max_items=8, include_stats=True)
                    trace_tensor(9, "silu_swiglu_activation_operator", "block0.mlp.shadow.gate", gate, stage="shadow_activation")
                    trace_tensor(9, "silu_swiglu_activation_operator", "block0.mlp.shadow.value", value, stage="shadow_activation")
                    shadow = activation_func(gate) * value
                    activation_component = (9, "silu_swiglu_activation_operator")
                else:
                    shadow = activation_func(shadow)
                    activation_component = (8, "gelu_activation_operator")
                debug_tensor_summary("block0.mlp.shadow.after_activation", shadow, max_items=8, include_stats=True)
                trace_tensor(
                    activation_component[0],
                    activation_component[1],
                    "block0.mlp.shadow.after_activation",
                    shadow,
                    stage="shadow_activation",
                )

                if actual_fc2_input is not None:
                    try:
                        diff = (shadow.detach() - actual_fc2_input.detach()).float().abs()
                        debug_scalar("block0.mlp.shadow_vs_fc2_input.max_abs_diff", float(diff.max().cpu().item()))
                        debug_scalar("block0.mlp.shadow_vs_fc2_input.mean_abs_diff", float(diff.mean().cpu().item()))
                    except Exception:
                        debug_scalar("block0.mlp.shadow_vs_fc2_input.max_abs_diff", "unavailable")

            def _trace_attention_score_proxy(qkv_tensor):
                if qkv_tensor is None or not isinstance(qkv_tensor, torch.Tensor):
                    return
                attention_module = getattr(layer0, "self_attention", None)
                config = getattr(attention_module, "config", None)
                if config is None:
                    return
                try:
                    heads = int(getattr(config, "num_attention_heads", 0) or 0)
                    groups = int(getattr(config, "num_query_groups", heads) or heads)
                    kv_channels = int(
                        getattr(config, "kv_channels", 0)
                        or (int(getattr(config, "hidden_size", qkv_tensor.shape[-1])) // max(1, heads))
                    )
                    q_size = heads * kv_channels
                    kv_size = groups * kv_channels
                    if heads <= 0 or groups <= 0 or qkv_tensor.shape[-1] < q_size + 2 * kv_size:
                        return
                    q_tensor, k_tensor, v_tensor = torch.split(qkv_tensor, [q_size, kv_size, kv_size], dim=-1)
                    trace_tensor(4, "matmul_attention_score_operator", "block0.attention_score.q", q_tensor, stage="attention_score_proxy")
                    trace_tensor(4, "matmul_attention_score_operator", "block0.attention_score.k", k_tensor, stage="attention_score_proxy")
                    trace_tensor(5, "attention_core_operator", "block0.attention_core.v", v_tensor, stage="attention_score_proxy")
                    seq_len, batch_size = q_tensor.shape[0], q_tensor.shape[1]
                    q_heads = q_tensor.reshape(seq_len, batch_size, heads, kv_channels).permute(1, 2, 0, 3).float()
                    k_heads = k_tensor.reshape(seq_len, batch_size, groups, kv_channels).permute(1, 2, 0, 3).float()
                    if heads % groups == 0 and heads != groups:
                        k_heads = k_heads.repeat_interleave(heads // groups, dim=1)
                    if k_heads.shape[1] != q_heads.shape[1]:
                        return
                    scores = torch.matmul(q_heads, k_heads.transpose(-1, -2)) / math.sqrt(max(1, kv_channels))
                    trace_tensor(4, "matmul_attention_score_operator", "block0.attention_score.proxy_output", scores, stage="attention_score_proxy")
                    probs = torch.softmax(scores, dim=-1)
                    trace_tensor(7, "softmax_operator", "block0.softmax.proxy_output", probs, stage="attention_softmax_proxy")
                    if bool(getattr(config, "use_flash_attention", False)):
                        trace_tensor(6, "flash_attention_operator", "block0.flash_attention.proxy_scores", scores, stage="flash_attention_proxy")
                except Exception as exc:
                    trace_event("attention_score_proxy_error", {"error": str(exc)})

            hook_specs = [
                ("block0.input_layernorm.output", getattr(layer0, "input_layernorm", None)),
                ("block0.self_attention.linear_qkv.output", getattr(getattr(layer0, "self_attention", None), "linear_qkv", None)),
                ("block0.self_attention.linear_proj.output", getattr(getattr(layer0, "self_attention", None), "linear_proj", None)),
                ("block0.pre_mlp_layernorm.output", getattr(layer0, "pre_mlp_layernorm", None)),
                ("block0.mlp.linear_fc1.output", getattr(getattr(layer0, "mlp", None), "linear_fc1", None)),
                ("block0.mlp.linear_fc2.output", getattr(getattr(layer0, "mlp", None), "linear_fc2", None)),
                ("block0.mlp.output", getattr(layer0, "mlp", None)),
                ("block0.layer0.output", layer0),
            ]

            hooks = []
            pre_hook_specs = [
                ("block0.mlp.linear_fc2.input", getattr(getattr(layer0, "mlp", None), "linear_fc2", None)),
            ]

            for summary_name, module in pre_hook_specs:
                if module is None or not hasattr(module, "register_forward_pre_hook"):
                    debug_tensor_summary(summary_name, None, max_items=8, include_stats=True)
                    continue

                def _pre_hook(_module, _inputs, summary_name=summary_name):
                    tensor = _extract_first_tensor(_inputs)
                    debug_tensor_summary(summary_name, tensor, max_items=8, include_stats=True)
                    component_id, component_name = _component_for_module(summary_name, _module)
                    trace_nested_tensors(
                        component_id,
                        component_name,
                        summary_name,
                        _inputs,
                        stage="block0_pre_hook_input",
                    )
                    trace_module_weights(
                        component_id,
                        component_name,
                        _module,
                        stage="block0_pre_hook_weights",
                        module_name=summary_name,
                    )
                    _emit_mlp_config_once()
                    _compute_shadow_activation(tensor)

                hooks.append(module.register_forward_pre_hook(_pre_hook))

            for summary_name, module in hook_specs:
                if module is None or not hasattr(module, "register_forward_hook"):
                    debug_tensor_summary(summary_name, None, max_items=8, include_stats=True)
                    continue

                def _hook(_module, _inputs, _output, summary_name=summary_name):
                    tensor = _extract_first_tensor(_output)
                    debug_tensor_summary(summary_name, tensor, max_items=8, include_stats=True)
                    component_id, component_name = _component_for_module(summary_name, _module)
                    trace_nested_tensors(
                        component_id,
                        component_name,
                        f"{summary_name}.input",
                        _inputs,
                        stage="block0_hook_input",
                    )
                    trace_nested_tensors(
                        component_id,
                        component_name,
                        f"{summary_name}.output",
                        _output,
                        stage="block0_hook_output",
                    )
                    if "linear_qkv" in summary_name:
                        _trace_attention_score_proxy(tensor)
                    trace_module_weights(
                        component_id,
                        component_name,
                        _module,
                        stage="block0_hook_weights",
                        module_name=summary_name,
                    )

                hooks.append(module.register_forward_hook(_hook))

            linear_fc1 = getattr(mlp_module, "linear_fc1", None)
            if linear_fc1 is not None and hasattr(linear_fc1, "register_forward_hook"):
                def _fc1_hook(_module, _inputs, _output):
                    if isinstance(_output, (tuple, list)):
                        shadow_cache["fc1_output"] = _extract_first_tensor(_output)
                        shadow_cache["fc1_bias"] = _extract_first_tensor(_output[1:]) if len(_output) > 1 else None
                    else:
                        shadow_cache["fc1_output"] = _extract_first_tensor(_output)
                        shadow_cache["fc1_bias"] = None

                hooks.append(linear_fc1.register_forward_hook(_fc1_hook))
            return hooks

        def _emit_final_norm_summary(hidden_states):
            final_norm_module = None
            for attr_name in ("final_norm", "final_layernorm"):
                candidate = getattr(self, attr_name, None)
                if candidate is not None:
                    final_norm_module = candidate
                    break
            if final_norm_module is None:
                debug_tensor_summary("final_norm.output", None, max_items=8, include_stats=True)
                return hidden_states
            trace_tensor(2, "normalization_operator", "final_norm.input", hidden_states, stage="final_norm_input")
            trace_module_weights(2, "normalization_operator", final_norm_module, stage="final_norm_weights", module_name="final_norm")
            norm_output = final_norm_module(hidden_states)
            _emit_layer_summary("final_norm.output", norm_output)
            trace_tensor(2, "normalization_operator", "final_norm.output", norm_output, stage="final_norm_output")
            return norm_output

        # -------------------------
        # 日志与确定性工具函数
        # -------------------------
        def _analyze_decoder_internals(decoder_block, input_data, attention_mask, node_id, log_file):
            """详细分析decoder内部结构"""
            def _write(msg):
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(msg)

            def _safe_stat_str(value):
                try:
                    if value is None:
                        return "na"
                    if hasattr(value, "detach"):
                        value = value.detach()
                    if hasattr(value, "float"):
                        value = value.float()
                    if hasattr(value, "mean") and not isinstance(value, (int, float)):
                        value = value.mean()
                    if hasattr(value, "item"):
                        value = value.item()
                    return f"{float(value):.6f}"
                except Exception:
                    return "na"
            
            def _log_tensor(name, t):
                if t is None:
                    _write(f"[{name}] None\n")
                    return
                try:
                    mean_value = t.mean() if hasattr(t, "mean") else None
                except Exception:
                    mean_value = None
                try:
                    std_value = t.std() if hasattr(t, "std") else None
                except Exception:
                    std_value = None
                _write(
                    f"[{name}] shape={getattr(t, 'shape', None)}, "
                    f"mean={_safe_stat_str(mean_value)}, std={_safe_stat_str(std_value)}\n"
                )
            
            _write(f"\n\n=== Detailed Analysis for Decoder {node_id} ===\n")
            
            # 分析每个TransformerLayer
            if hasattr(decoder_block, 'layers') and hasattr(decoder_block.layers, '__len__'):
                for i, layer in enumerate(decoder_block.layers):
                    _write(f"\n--- Layer {i} ---\n")
                    
                    # 分析每个子模块
                    for name, module in layer.named_children():
                        _write(f"  Submodule: {name} - {type(module).__name__}\n")
                        
                        # 记录参数统计
                        total_params = 0
                        for param_name, param in module.named_parameters():
                            if param is not None:
                                try:
                                    mean_value = param.mean() if hasattr(param, "mean") else None
                                except Exception:
                                    mean_value = None
                                try:
                                    std_value = param.std() if hasattr(param, "std") else None
                                except Exception:
                                    std_value = None
                                _write(
                                    f"    Param {param_name}: shape={getattr(param, 'shape', None)}, "
                                    f"mean={_safe_stat_str(mean_value)}, std={_safe_stat_str(std_value)}\n"
                                )
                                total_params += param.numel()
                        
                        _write(f"    Total parameters: {total_params}\n")
            
            # 分析最终layernorm
            if hasattr(decoder_block, 'final_layernorm'):
                _write(f"\n--- Final LayerNorm ---\n")
                module = decoder_block.final_layernorm
                for param_name, param in module.named_parameters():
                    if param is not None:
                        try:
                            mean_value = param.mean() if hasattr(param, "mean") else None
                        except Exception:
                            mean_value = None
                        try:
                            std_value = param.std() if hasattr(param, "std") else None
                        except Exception:
                            std_value = None
                        _write(
                            f"  Param {param_name}: shape={getattr(param, 'shape', None)}, "
                            f"mean={_safe_stat_str(mean_value)}, std={_safe_stat_str(std_value)}\n"
                        )
        
        def _ensure_dir(fp: str):
            d = os.path.dirname(fp)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)

        def _tensor_summary(t: torch.Tensor) -> str:
            try:
                info = f"dtype={t.dtype}, device={t.device}, shape={tuple(t.shape)}"
                if t.numel() > 0 and t.dtype.is_floating_point:
                    return info + \
                        f", min={float(t.min()):.6g}, max={float(t.max()):.6g}, mean={float(t.mean()):.6g}, std={float(t.std()):.6g}"
                return info
            except Exception as _:
                return f"(unable to summarize tensor: shape={getattr(t,'shape',None)})"

        def _format_tensor_content(t: torch.Tensor, max_elems: int = 100) -> str:
            try:
                if t.numel() <= max_elems:
                    return str(t.detach().cpu())
                flat = t.detach().flatten().cpu()
                head = flat[:max_elems]
                return f"{head} ... (truncated {t.numel()-max_elems} elems)"
            except Exception as _:
                return "(unable to print tensor content)"

        def _log_tensor(name: str, t: torch.Tensor):
            if t is None:
                _write(f"[{name}] None\n")
                return
            _write(f"[{name}] {_tensor_summary(t)}\n")
            _write(f"{_format_tensor_content(t)}\n\n")

        def _write(msg: str):
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(msg)

        def _save_module_io(module, input, output, module_name: str, node_id: str):
            """保存模块的输入输出和参数"""
            try:
                component_id, component_name = _component_for_module(module_name, module)
                module_extra = {
                    "module_name": module_name,
                    "module_type": type(module).__name__,
                    "resolved_component": component_name,
                }
                _write(f"\n--- Module IO: {module_name} ---\n")
                
                # 记录输入
                if isinstance(input, tuple):
                    for i, inp in enumerate(input):
                        if inp is not None:
                            _log_tensor(f"{module_name}.input[{i}]", inp)
                else:
                    _log_tensor(f"{module_name}.input", input)
                
                # 记录输出
                if isinstance(output, tuple):
                    for i, out in enumerate(output):
                        if out is not None:
                            _log_tensor(f"{module_name}.output[{i}]", out)
                else:
                    _log_tensor(f"{module_name}.output", output)
                
                # 记录参数
                for name, param in module.named_parameters():
                    if param is not None:
                        _log_tensor(f"{module_name}.param[{name}]", param)
                
                # 记录buffer
                for name, buffer in module.named_buffers():
                    if buffer is not None:
                        _log_tensor(f"{module_name}.buffer[{name}]", buffer)

                _trace_io(
                    component_id,
                    component_name,
                    f"{module_name}.input",
                    input,
                    stage="module_input",
                    node_id=node_id,
                    extra=module_extra,
                )
                _trace_io(
                    component_id,
                    component_name,
                    f"{module_name}.output",
                    output,
                    stage="module_output",
                    node_id=node_id,
                    extra=module_extra,
                )
                _trace_weight(
                    component_id,
                    component_name,
                    module,
                    stage="module_weights",
                    node_id=node_id,
                    module_name=module_name,
                    extra=module_extra,
                )
                        
            except Exception as e:
                _write(f"Error saving module {module_name} IO: {str(e)}\n")

        def _register_module_hooks(module, module_name: str, node_id: str):
            """为模块注册前向钩子"""
            hooks = []
            
            def hook_fn(module, input, output):
                _save_module_io(module, input, output, module_name, node_id)
            
            try:
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)
                
                # 递归为子模块注册钩子
                for name, submodule in module.named_children():
                    sub_hooks = _register_module_hooks(submodule, f"{module_name}.{name}", node_id)
                    hooks.extend(sub_hooks)
                    
            except Exception as e:
                _write(f"Error registering hooks for {module_name}: {str(e)}\n")
                
            return hooks

        def _ensure_tensor_ready(tensor, name="tensor"):
            """Ensure tensor is ready for device transfer"""
            if tensor is None:
                return None
            try:
                # Ensure tensor is contiguous in memory
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                # Ensure tensor is on the correct device
                return tensor
            except Exception as e:
                _write(f"Warning: Failed to prepare {name} for device transfer: {str(e)}\n")
                return tensor

        # 准备日志文件
        _ensure_dir(log_file)
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("=== Graph.forward debug log start ===\n")

        mutated_nodes = self.mutated_nodes
        trace_components_manifest(
            {
                "node_count": len(getattr(self, "nodes", {}) or {}),
                "mutated_node_count": len(mutated_nodes or {}),
            }
        )
        trace_event(
            "graph_forward_begin",
            {
                "node_ids": list(getattr(self, "nodes", {}).keys()),
                "mutated_nodes": sorted(str(key) for key in (mutated_nodes or {}).keys()),
            },
        )

        # 去随机化
        prev_training = self.training
        self.eval()
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        torch_infer_ctx = torch.no_grad()

        device = torch.device("npu" if torch.npu.is_available() else "cpu")
        cur_node = self.nodes[0]

        # 准备输入
        if input_ids is None:
            values = [[5032], [39706], [24761], [14473],
                    [35428], [1358], [20794], [6819]]
            input_ids = torch.tensor(values, dtype=torch.int64, device=device)
        if position_ids is None:
            position_ids = torch.arange(1, device=device).expand(8, 1)
        if attention_mask is None:
            seq_length = position_ids.size(1)
            attention_mask = torch.zeros(1, 1, seq_length, seq_length,
                                        dtype=torch.bool, device=device)

        if input_data is None:
            values = [[0.5032], [3.9706], [2.4761], [1.4473],
                    [0.35428], [0.1358], [0.20794], [0.6819]]
            input_data = torch.tensor(values, dtype=torch.float32, device=device)

        # 阶段：初始化
        _write("\n=== Stage[init] BEGIN ===\n")
        # Ensure tensors are ready before logging
        input_ids = _ensure_tensor_ready(input_ids, "input_ids")
        position_ids = _ensure_tensor_ready(position_ids, "position_ids")
        attention_mask = _ensure_tensor_ready(attention_mask, "attention_mask")
        input_data = _ensure_tensor_ready(input_data, "input_data")

        _log_tensor("input_ids", input_ids)
        _log_tensor("position_ids", position_ids)
        _log_tensor("attention_mask", attention_mask)
        _log_tensor("input_data(seed)", input_data)
        trace_tensor(0, "full_network", "input_ids", input_ids, stage="network_input")
        trace_tensor(0, "full_network", "position_ids", position_ids, stage="network_input")
        trace_tensor(0, "full_network", "attention_mask", attention_mask, stage="network_input")
        trace_tensor(0, "full_network", "input_data_seed", input_data, stage="network_input")
        _write("=== Stage[init] END ===\n")

        if debug:
            print(f"输入input_ids形状: {input_ids.shape}")
            print(f"输入position_ids形状: {position_ids.shape}")
            if mutated_nodes:
                print(f"检测到 {len(mutated_nodes)} 个变异节点: {list(mutated_nodes.keys())}")

        with torch_infer_ctx:
            output = input_data
            while True:
                if debug:
                    print(f"\n处理节点 {cur_node.id}: {cur_node.str_op}")
                if cur_node.block is not None:
                    try:
                        cur_block = cur_node.block.to(device) if hasattr(cur_node.block, 'to') else cur_node.block
                    except:
                        cur_block = cur_node.block
                    cur_block_prev_training = cur_block.training
                    cur_block.eval()

                    # 为decoder注册详细的调试钩子
                    decoder_hooks = []
                    if 'decoderlayer' in cur_node.str_op.lower() or 'mutated_decoder' in cur_node.str_op.lower():
                        if debug:
                            print(f"  注册decoder内部模块钩子...")
                        decoder_hooks = _register_module_hooks(cur_block, f"decoder_{cur_node.id}", cur_node.id)

                    if 'embedding' in cur_node.str_op.lower():
                        if debug:
                            print("  执行embedding...")
                        self.embedding = cur_node.block
                        if self.pre_process or self.post_process:
                            self.setup_embeddings_and_output_layer()
                        _emit_weight_debug_once()

                        # Stage: Embedding begin
                        _write("\n=== Stage[embedding] BEGIN ===\n")
                        # 结构与权重（embedding）
                        # _dump_block_structure_and_weights(cur_block, "embedding.block(before)")
                        _log_tensor("embedding.input_ids", input_ids)
                        _log_tensor("embedding.position_ids", position_ids)
                        trace_tensor(11, "embedding_layer", "input_ids", input_ids, stage="embedding_input", node_id=cur_node.id)
                        trace_tensor(11, "embedding_layer", "position_ids", position_ids, stage="embedding_input", node_id=cur_node.id)
                        trace_module_weights(
                            11,
                            "embedding_layer",
                            cur_block,
                            stage="embedding_weights_before",
                            node_id=cur_node.id,
                            module_name=f"embedding_{cur_node.id}",
                        )

                        # Ensure tensors are contiguous before moving to NPU
                        input_ids_ready = _ensure_tensor_ready(input_ids, "embedding_input_ids")
                        position_ids_ready = _ensure_tensor_ready(position_ids, "embedding_position_ids")
                        
                        input_ids_ready = input_ids_ready.to(device) if torch.npu.is_available() else input_ids_ready
                        position_ids_ready = position_ids_ready.to(device) if torch.npu.is_available() else position_ids_ready
                        output = cur_block(input_ids=input_ids_ready, position_ids=position_ids_ready)

                        # 你的后处理：转置 + repeat
                        input_data = output
                        input_data = input_data.transpose(0, 1)
                        input_data = input_data.repeat(4, 2, 1)
                        _emit_layer_summary("embedding.output", input_data)
                        trace_tensor(1, "embedding_operator", "embedding.output_raw", output, stage="embedding_output", node_id=cur_node.id)
                        trace_tensor(11, "embedding_layer", "embedding.output_processed.baseline", input_data, stage="embedding_output", node_id=cur_node.id)
                        input_data = maybe_perturb_tensor(
                            input_data,
                            tensor_name="embedding_output",
                            component_id=11,
                            component_name="embedding_layer",
                            stage="embedding_output_perturbation",
                            node_id=cur_node.id,
                        )

                        # Stage: Embedding end
                        _log_tensor("embedding.output_raw", output)
                        _log_tensor("embedding.output_processed", input_data)
                        trace_tensor(11, "embedding_layer", "embedding.output_processed", input_data, stage="embedding_output", node_id=cur_node.id)
                        trace_module_weights(
                            11,
                            "embedding_layer",
                            cur_block,
                            stage="embedding_weights_after",
                            node_id=cur_node.id,
                            module_name=f"embedding_{cur_node.id}",
                        )
                        # 前后再次转储（通常权重不变，这里用于确认确实无副作用）
                        # _dump_block_structure_and_weights(cur_block, "embedding.block(after)")
                        _write("=== Stage[embedding] END ===\n")

                        if debug:
                            print(f"  Embedding输出形状: {output.shape}")
                            print(f"  处理后形状: {input_data.shape}")

                    elif 'decoderlayer' in cur_node.str_op.lower() or 'mutated_decoder' in cur_node.str_op.lower():
                        decoder_index = decoder_count
                        decoder_count += 1
                        if cur_node.id in mutated_nodes and debug:
                            node_info = mutated_nodes[cur_node.id]
                            print(f"  执行变异decoder (基于 {node_info['source_file']}):")
                            print(f"    hidden_size: {node_info['config'].hidden_size}")
                            print(f"    num_layers: {node_info['config'].num_layers}")
                            print(f"    num_attention_heads: {node_info['config'].num_attention_heads}")
                        elif debug:
                            print("  执行标准decoder...")

                        # Stage: Decoder begin
                        _write(f"\n=== Stage[decoder_{cur_node.id}] BEGIN ===\n")
                        _log_tensor(f"decoder_{cur_node.id}.input(hidden_states)", input_data)
                        trace_tensor(
                            14,
                            "decoder_block",
                            f"decoder_{cur_node.id}.input_hidden_states",
                            input_data,
                            stage="decoder_input",
                            node_id=cur_node.id,
                            extra={"decoder_index": decoder_index},
                        )
                        trace_tensor(
                            14,
                            "decoder_block",
                            f"decoder_{cur_node.id}.attention_mask",
                            attention_mask,
                            stage="decoder_input",
                            node_id=cur_node.id,
                            extra={"decoder_index": decoder_index},
                        )
                        trace_module_weights(
                            14,
                            "decoder_block",
                            cur_block,
                            stage="decoder_weights_before",
                            node_id=cur_node.id,
                            module_name=f"decoder_{cur_node.id}",
                        )
                        if decoder_index == 0:
                            _emit_layer_summary("block0.input", input_data)

                        # 记录decoder block开始前的详细状态
                        _write(f"\n--- Decoder {cur_node.id} Initial State ---\n")
                        _log_tensor(f"decoder_{cur_node.id}.initial_input", input_data)
                        
                        # 维度检查和调整
                        expected_hidden = cur_node.config.hidden_size
                        if input_data.shape[-1] != expected_hidden:
                            if debug:
                                print(f"  调整输入维度: {input_data.shape[-1]} -> {expected_hidden}")
                            seq_len, batch_size, _ = input_data.shape
                            input_data = reshape_tensor_nd(input_data, (seq_len, batch_size, expected_hidden))
                            input_data = input_data.npu() if torch.npu.is_available() else input_data
                            _log_tensor(f"decoder_{cur_node.id}.input(adjusted)", input_data)
                            trace_tensor(
                                14,
                                "decoder_block",
                                f"decoder_{cur_node.id}.input_adjusted",
                                input_data,
                                stage="decoder_input_adjusted",
                                node_id=cur_node.id,
                                extra={"decoder_index": decoder_index, "expected_hidden": expected_hidden},
                            )

                        # RoPE处理
                        rotary_pos_emb = None
                        if (cur_node.id not in mutated_nodes and
                                hasattr(self, 'total_config') and
                                self.total_config.get('position_embedding_type') == 'rope' and
                                not self.config.multi_latent_attention and
                                hasattr(self, 'rotary_pos_emb')):

                            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                                inference_context, cur_block, input_data, self.config, None
                            )
                            rotary_pos_emb = self.rotary_pos_emb(
                                rotary_seq_len,
                                packed_seq=False,
                            )

                        elif (cur_node.id not in mutated_nodes and
                            hasattr(self, 'total_config') and
                            self.total_config.get('position_embedding_type') == 'mrope' and
                            not self.config.multi_latent_attention and
                            hasattr(self, 'rotary_pos_emb')):

                            if not self.config.flash_decode:
                                rotary_pos_emb = self.rotary_pos_emb(position_ids, self.mrope_section)
                            else:
                                raise NotImplementedError(
                                    "Flash decoding uses precomputed cos and sin for RoPE, not implmented in "
                                    "MultimodalRotaryEmbedding yet."
                                )
                        trace_nested_tensors(
                            12,
                            "self_attention_block",
                            f"decoder_{cur_node.id}.rotary_pos_emb",
                            rotary_pos_emb,
                            stage="decoder_rope",
                            node_id=cur_node.id,
                            extra={"decoder_index": decoder_index},
                        )

                        # 真正调用 decoder block
                        print(f"\ndecoder block {cur_node.id}: \n", cur_block)

                        decoder_param_dtype = _first_floating_parameter_dtype(cur_block)
                        input_data = _cast_floating_tensor_to_dtype(input_data, decoder_param_dtype)

                        _analyze_decoder_internals(cur_block, input_data, attention_mask, cur_node.id, "logs/decoder_info.log")

                        attention_mask_ready = _ensure_tensor_ready(attention_mask, "attention_mask")
                        attention_mask_ready = attention_mask_ready.npu() if torch.npu.is_available() else attention_mask_ready
                        compare_hooks = _register_block0_compare_hooks(cur_block) if decoder_index == 0 else []
                        decoder_input_for_residual = input_data
                        
                        output = cur_block(
                            hidden_states=input_data,
                            attention_mask=attention_mask_ready,
                            rotary_pos_emb=rotary_pos_emb,
                            packed_seq_params=None,
                            **(None or {}),
                        )

                        # 对齐 hidden_size
                        if output.shape[-1] < self.config.hidden_size:
                            padding_size = self.config.hidden_size - output.size(-1)
                            output = torch.nn.functional.pad(output, (0, padding_size), value=0)
                        elif output.shape[-1] > self.config.hidden_size:
                            output = output[..., :self.config.hidden_size]

                        input_data = output
                        if decoder_index == 0:
                            _emit_layer_summary("block0.output", output)
                        elif decoder_index == 1:
                            _emit_layer_summary("block1.output", output)
                        if len(cur_node.to_nodes) == 0:
                            _emit_layer_summary("last_block.output", output)

                        # 记录decoder block结束后的详细状态
                        _write(f"\n--- Decoder {cur_node.id} Final State ---\n")
                        _log_tensor(f"decoder_{cur_node.id}.final_output", output)
                        trace_tensor(
                            14,
                            "decoder_block",
                            f"decoder_{cur_node.id}.output",
                            output,
                            stage="decoder_output",
                            node_id=cur_node.id,
                            extra={"decoder_index": decoder_index},
                        )
                        if (
                            isinstance(output, torch.Tensor)
                            and isinstance(decoder_input_for_residual, torch.Tensor)
                            and tuple(output.shape) == tuple(decoder_input_for_residual.shape)
                        ):
                            trace_tensor(
                                10,
                                "residual_elementwise_operator",
                                f"decoder_{cur_node.id}.output_minus_input",
                                output - decoder_input_for_residual,
                                stage="decoder_residual_delta",
                                node_id=cur_node.id,
                                extra={"decoder_index": decoder_index},
                            )
                        trace_module_weights(
                            14,
                            "decoder_block",
                            cur_block,
                            stage="decoder_weights_after",
                            node_id=cur_node.id,
                            module_name=f"decoder_{cur_node.id}",
                        )

                        for hook in compare_hooks:
                            hook.remove()
                        
                        # Stage: Decoder end
                        _log_tensor(f"decoder_{cur_node.id}.output", output)
                        _write(f"=== Stage[decoder_{cur_node.id}] END ===\n")

                        if debug:
                            print(f"  Decoder输出形状: {output.shape}")

                    # 移除钩子
                    for hook in decoder_hooks:
                        hook.remove()

                    # 恢复 block 的训练状态
                    cur_block.train(cur_block_prev_training)

                if len(cur_node.to_nodes) == 0:
                    break
                cur_node = self.nodes[cur_node.to_nodes[0]]

            if debug:
                print(f"\n--- 最终输出处理 ---")
                print(f"最终输出形状: {input_data.shape}")
            _write("\n=== Stage[final_hidden] BEGIN ===\n")
            _log_tensor("final.hidden_states_before_output_layer", input_data)
            trace_tensor(0, "full_network", "hidden_states_before_output_layer", input_data, stage="network_hidden")
            _write("=== Stage[final_hidden] END ===\n")
            lm_head_input = _emit_final_norm_summary(input_data)
            trace_tensor(15, "output_layer", "lm_head.input", lm_head_input, stage="output_layer_input")

            if inference_context and not inference_context.is_static_batching():
                output = inference_context.last_token_logits(
                    output.squeeze(1).unsqueeze(0)
                ).unsqueeze(1)

            # logits and loss
            output_weight = None
            if self.share_embeddings_and_output_weights:
                output_weight = self.shared_embedding_or_output_weight()

            logits = None
            if hasattr(self, 'output_layer'):
                if debug:
                    print("执行output_layer...")
                _write("\n=== Stage[output_layer] BEGIN ===\n")
                _emit_layer_summary("lm_head.input", lm_head_input)
                trace_module_weights(15, "output_layer", self.output_layer, stage="output_layer_weights_before", module_name="output_layer")
                logits, _ = self.output_layer(
                    lm_head_input, weight=output_weight,
                )
                _log_tensor("output_layer.logits", logits)
                trace_tensor(15, "output_layer", "logits", logits, stage="output_layer_output")
                trace_module_weights(15, "output_layer", self.output_layer, stage="output_layer_weights_after", module_name="output_layer")
                _write("=== Stage[output_layer] END ===\n")
                debug_tensor_summary("logits", logits, max_items=16, include_stats=True)
                if getattr(logits, "dim", lambda: 0)() >= 3:
                    debug_tensor_summary("logits.raw_0_0_8", logits[0, 0, :8], max_items=8, include_stats=True)
                    if logits.shape[1] > 1:
                        debug_tensor_summary("logits.raw_0_1_8", logits[0, 1, :8], max_items=8, include_stats=True)
                if debug:
                    print(f"Logits形状: {logits.shape}")
            else:
                logits = input_data
                _write("\n=== Stage[no_output_layer] BEGIN ===\n")
                _log_tensor("no_output_layer.output", logits)
                trace_tensor(0, "full_network", "no_output_layer.output", logits, stage="network_output")
                _write("=== Stage[no_output_layer] END ===\n")
                debug_tensor_summary("logits", logits, max_items=16, include_stats=True)
                if debug:
                    print("没有output_layer，返回最后的hidden states")

            labels = None
            if labels is None:
                # [s b h] => [b s h]
                final_output = logits.transpose(0, 1).contiguous()
                if debug:
                    print(f"最终输出形状: {final_output.shape}")
                _write("\n=== Stage[final_output] BEGIN ===\n")
                _log_tensor("final_output(b,s,h or vocab)", final_output)
                trace_tensor(0, "full_network", "final_output", final_output, stage="network_output")
                trace_loss("overall_loss_norm", final_output.norm(), extra={"source": "graph.forward", "labels": False})
                trace_event("graph_forward_end", {"returned": "final_output", "labels": False})
                _write("=== Stage[final_output] END ===\n")
                return final_output

            loss = self.compute_language_model_loss(labels, logits)
            trace_loss("language_model_loss", loss, extra={"source": "graph.forward", "labels": True})
            trace_event("graph_forward_end", {"returned": "loss", "labels": True})
            debug_message("loss.compute_language_model_loss")
            if should_log_full() or should_log_heavy():
                debug_tensor_summary("loss.labels", labels, max_items=16, include_stats=False)
            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                debug_tensor_summary("loss.model_output", loss, max_items=16, include_stats=True, include_sum=True)
            else:
                debug_scalar("loss.model_output", loss.item() if hasattr(loss, "item") else loss)
            return loss
    
    def set_mutated_nodes(self, mutated_nodes: dict):
        """
        设置或更新变异节点信息

        Args:
            mutated_nodes: 变异节点信息字典
        """
        self.mutated_nodes = mutated_nodes if mutated_nodes is not None else {}

    def get_mutated_nodes(self):
        """
        获取变异节点信息

        Returns:
            dict: 变异节点信息字典
        """
        return self.mutated_nodes

    def load(self, config_yaml_path: str, config_json_path: str, debug: bool = True):
        """
        从之前生成的yaml配置文件加载图配置

        Args:
            config_yaml_path: 配置文件路径（例如：demo_graph_forward_n_nodes_configs/mutated_config_iter_001.yaml）
            debug: 是否打印调试信息

        Returns:
            bool: 加载是否成功
        """
        try:
            if debug:
                print(f"=== 从配置文件加载图: {config_yaml_path} ===")

            # 读取yaml配置文件
            yaml = YAML()
            with open(config_yaml_path, 'r', encoding='utf-8') as file:
                loaded_config = yaml.load(file)

            if debug:
                print(f"成功读取配置文件")
                if 'metadata' in loaded_config:
                    metadata = loaded_config['metadata']
                    print(f"  迭代轮数: {metadata.get('iteration', 'N/A')}")
                    print(f"  变异率: {metadata.get('mutation_rate', 'N/A')}")
                    print(f"  创建时间: {metadata.get('timestamp', 'N/A')}")

            # 从配置中提取基础配置和图结构
            if 'base_config' not in loaded_config or 'graph_structure' not in loaded_config:
                raise ValueError("配置文件格式错误：缺少base_config或graph_structure")

            base_config = loaded_config['base_config']
            graph_structure = loaded_config['graph_structure']

            # 重新初始化Graph的配置
            _prepare_transformer_config_dict(base_config['config'])

            # ------------------------------------------------------------------
            # 一些历史生成的 yaml 中会包含 TransformerConfig 当前版本并不支持的字段，
            # 直接传递会导致 "got an unexpected keyword argument" 的 TypeError。
            # 这里根据 dataclass 中定义的字段对配置进行一次过滤，忽略未知字段。
            # ------------------------------------------------------------------
            valid_fields = set(TransformerConfig.__dataclass_fields__.keys())
            filtered_cfg_dict = {
                k: v for k, v in base_config["config"].items() if k in valid_fields
            }
            unknown_keys = set(base_config["config"].keys()) - valid_fields
            if debug and unknown_keys:
                print(
                    f"  检测到未识别的 TransformerConfig 字段，已忽略: {sorted(list(unknown_keys))}"
                )

            _prepare_transformer_config_dict(filtered_cfg_dict)
            transformerblock_config = TransformerConfig(**filtered_cfg_dict)
            _pad_context_parallel_config_for_layer_numbering(transformerblock_config)
            self._stabilize_decoder_mlp_config(transformerblock_config)

            # 将过滤后的配置对象写回，保持 total_config 的完整性
            base_config["config"] = transformerblock_config
            self.total_config = base_config

            # 更新父类配置
            self.config = transformerblock_config

            # 重新创建nodes
            layer_configs = graph_structure.get('LayerConfig', {})
            node_ids = list(layer_configs.keys())
            
            with open(config_json_path, 'r', encoding='utf-8') as json_file:
                json_configs = json.load(json_file)  # 返回字典或列表
            
            if debug:
                print(f"  节点数量: {len(node_ids)}")
                print(f"  节点ID: {node_ids}")

            # 清空现有的节点
            self.nodes.clear()
            self.mutated_nodes.clear()

            # Reuse the already-normalized model TransformerConfig for node
            # initialization. A second hard-coded 24-layer config can receive
            # MindSpeed global CP args and fail cp_comm_type validation.
            init_config = transformerblock_config

            # 重建节点
            for node_id, layer_config in layer_configs.items():
                # 确保node_id是整数
                if isinstance(node_id, str):
                    node_id = int(node_id)

                node = Node(config=init_config, index=node_id)
                node.str_op = layer_config.get('name', 'unknown')
                node.from_nodes = layer_config.get('from', [])
                node.to_nodes = layer_config.get('to', [])
                node.params = layer_config.get('params', {})
                node.state = layer_config.get('state', 'none')

                # 设置node的度数
                node.in_degree = len(node.from_nodes)
                node.out_degree = len(node.to_nodes)
                if node.state == 'des':
                    node.out_degree = 1

                # 处理变异decoder - 检查params中是否有model_config路径
                if 'model_config' in node.params:
                    if debug:
                        print(f"  节点 {node_id} 是变异decoder，配置: {node.params['model_config']}")

                    # 从model_config路径加载变异配置
                    model_config_path = node.params['model_config']
                    try:
                        # 使用新的 YAML 实例读取，避免前面局部变量覆盖导致缺少 safe_load
                        yaml_mut = YAML()
                        mutated_yaml_config = yaml_mut.load(model_config_path)

                        # 提取 TransformerConfig/MLATransformerConfig 配置
                        mutated_transformer_config = _extract_node_transformer_config(mutated_yaml_config)
                        if mutated_transformer_config:
                            _prepare_transformer_config_dict(mutated_transformer_config)
                            mutated_config = TransformerConfig(**mutated_transformer_config)
                            self._stabilize_decoder_mlp_config(mutated_config)
                            node.config = mutated_config
                            node.str_op = 'mutated_decoder'

                            # 保存到mutated_nodes
                            self.mutated_nodes[node_id] = {
                                'config': mutated_config,
                                'source_file': model_config_path,
                                'original_config': mutated_transformer_config
                            }

                            if debug:
                                print(f"    ✓ 加载变异配置成功")
                                print(f"      hidden_size: {mutated_config.hidden_size}")
                                print(f"      num_layers: {mutated_config.num_layers}")
                                print(f"      num_attention_heads: {mutated_config.num_attention_heads}")

                    except Exception as e:
                        
                        if debug:
                            traceback.print_exc()
                            print(f"加载变异配置失败: {e}，使用默认配置")
                            
#                 with open(config_json_path, 'r', encoding='utf-8') as file:
#                     layer_configs = json.load(config_json_path)  # 返回字典或列表
                
#                 for node_idx,node_config in layer_configs.items():
#                     node_idx == "block_num_list":
#                         continue
#                     if isinstance(node_idx, str):
#                         node_idx = int(node_idx)
                if not node_id == 1:
                    node_entry = json_configs.get(str(node_id - 1), {})
                    node_transformer_config = _extract_node_transformer_config(node_entry)
                    if not node_transformer_config:
                        raise KeyError(
                            f"missing TransformerConfig/MLATransformerConfig for node {node_id - 1}"
                        )
                    _prepare_transformer_config_dict(node_transformer_config)
                    node.config = TransformerConfig(**node_transformer_config)
                    _pad_context_parallel_config_for_layer_numbering(node.config)
                    self._stabilize_decoder_mlp_config(node.config)
                
                if node.str_op.lower() == "embedding":
                    from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
                    node.block = LanguageModelEmbedding(
                        config=self.config,
                        vocab_size=self.total_config['vocab_size'],
                        max_sequence_length=self.total_config['max_sequence_length'],
                        position_embedding_type=self.total_config['position_embedding_type'],
                    )

                elif 'decoderlayer' in node.str_op.lower() or 'mutated_decoder' in node.str_op.lower():

                    from megatron.core.transformer.transformer_block import TransformerBlock
                    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

                    node.block = TransformerBlock(
                        config=node.config,
                        spec=get_gpt_layer_local_spec(
                            None,
                            False,
                            False,
                            # False,
                            # False,
                            # normalization="RMSNorm",
                        ),
                        pre_process=True,
                        post_process=True,
                        # vp_stage=None,
                    )
                    self._stabilize_decoder_mlp_module(node.block)

                self.nodes[node_id] = node

            # 重新初始化其他必要的组件
            self._stabilize_graph_mlp_configs()
            for node in self.nodes.values():
                if "decoder" in str(getattr(node, "str_op", "")).lower():
                    self._stabilize_decoder_mlp_module(getattr(node, "block", None))
            self._reinitialize_components()
            self._emit_load_structure_summary_once()

            if debug:
                print(f"✓ 图加载完成!")
                print(f"  总节点数: {len(self.nodes)}")
                print(f"  变异节点数: {len(self.mutated_nodes)}")
                if self.mutated_nodes:
                    print(f"  变异节点ID: {list(self.mutated_nodes.keys())}")

            return True

        except Exception as e:
            if debug:
                print(f"✗ 加载配置文件失败: {e}")
                import traceback
                traceback.print_exc()
            return False

    def _reinitialize_components(self):
        """重新初始化必要的组件"""
        # 重新设置embedding
        self.embedding = None

        # 重新初始化位置编码
        if self.total_config['position_embedding_type'] == 'rope' and not self.config.multi_latent_attention:
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=self.total_config['rotary_percent'],
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=self.total_config['seq_len_interpolation_factor'],
                rotary_base=self.total_config['rotary_base'],
                rope_scaling=self.total_config['rope_scaling'],
                rope_scaling_factor=self.total_config['rope_scaling_factor'],
                use_cpu_initialization=self.config.use_cpu_initialization,
            )
        # elif self.total_config['position_embedding_type'] == 'mrope' and not self.config.multi_latent_attention:
        #     self.rotary_pos_emb = MultimodalRotaryEmbedding(
        #         kv_channels=self.config.kv_channels,
        #         rotary_percent=self.total_config['rotary_percent'],
        #         rotary_interleaved=self.config.rotary_interleaved,
        #         seq_len_interpolation_factor=self.total_config['seq_len_interpolation_factor'],
        #         rotary_base=self.total_config['rotary_base'],
        #     )
        #     self.mrope_section = self.config.mrope_section
        #     self.rotary_pos_emb_cache = {}

        # 重新设置处理标志
        self.pre_process = True
        self.post_process = True

        # 重新初始化输出层
        if self.post_process:
            if self.config.defer_embedding_wgrad_compute:
                self.embedding_activation_buffer = []
                self.grad_output_buffer = []
            else:
                self.embedding_activation_buffer = None
                self.grad_output_buffer = None

            self.share_embeddings_and_output_weights = False

            if self.config._cpu_offloading_context == 'None':
                self.config._cpu_offloading_context = None

            self.output_layer = tensor_parallel.ColumnParallelLinear(
                self.config.hidden_size,
                self.total_config['vocab_size'],
                config=self.config,
                init_method=self.config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=False,
                skip_weight_param_allocation=self.pre_process
                                             and self.share_embeddings_and_output_weights,
                embedding_activation_buffer=self.embedding_activation_buffer,
                grad_output_buffer=self.grad_output_buffer,
            )

    def display(self):
        print('display graph:')
        for i in self.nodes.keys():
            print("id:" + str(self.nodes[i].id) + ", layer:" + str(self.nodes[i].str_op) + ", from:" + str(
                self.nodes[i].from_nodes) +
                  ", to:" + str(self.nodes[i].to_nodes))
        for i in self.nodes.keys():
            print("param" + str(self.nodes[i].id) + ":", self.nodes[i].params)

    def get_graph(self):
        g = []
        for i in self.nodes.keys():
            g.append(str(self.nodes[i]))
        return g

    def _node_role_aliases(self):
        aliases = {}
        embedding_idx = 0
        decoder_idx = 0
        for node_id in sorted(self.nodes.keys()):
            node = self.nodes[node_id]
            str_op = str(getattr(node, 'str_op', '')).lower()
            node_aliases = []
            if "embedding" in str_op:
                node_aliases.append(f"shared.embedding.{embedding_idx}.")
                embedding_idx += 1
            elif "decoder" in str_op:
                node_aliases.append(f"shared.decoder.{decoder_idx}.")
                decoder_idx += 1
            aliases[node_id] = node_aliases
        return aliases
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """重写state_dict以包含所有节点的参数"""
        if destination is None:
            destination = {}
        
        # 调用父类的state_dict来获取基础参数
        super().state_dict(destination, prefix, keep_vars)
        
        # 添加所有节点的block参数
        node_aliases = self._node_role_aliases()
        for node_id, node in self.nodes.items():
            if hasattr(node, 'block') and node.block is not None:
                node_state_dict = node.block.state_dict(prefix=f'{prefix}nodes.{node_id}.', keep_vars=keep_vars)
                destination.update(node_state_dict)
                for alias_prefix in node_aliases.get(node_id, []):
                    alias_state_dict = node.block.state_dict(prefix=f'{prefix}{alias_prefix}', keep_vars=keep_vars)
                    destination.update(alias_state_dict)
        
        return destination

    def load_state_dict(self, state_dict, strict=True):
        """重写load_state_dict以加载所有节点的参数"""
        # 先加载基础参数
        result = super().load_state_dict(state_dict, strict=False)
        
        # 加载节点参数
        node_aliases = self._node_role_aliases()
        for node_id, node in self.nodes.items():
            if hasattr(node, 'block') and node.block is not None:
                for alias_prefix in node_aliases.get(node_id, []):
                    alias_state_dict = {
                        k.replace(alias_prefix, ''): v
                        for k, v in state_dict.items()
                        if k.startswith(alias_prefix)
                    }
                    if alias_state_dict:
                        node.block.load_state_dict(alias_state_dict, strict=False)
                        break
                else:
                    node_prefix = f'nodes.{node_id}.'
                    node_state_dict = {
                        k.replace(node_prefix, ''): v
                        for k, v in state_dict.items()
                        if k.startswith(node_prefix)
                    }
                    if node_state_dict:
                        node.block.load_state_dict(node_state_dict, strict=False)
        
        return result

    def __len__(self):
        return len(self.nodes)
