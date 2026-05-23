#!/usr/bin/env python3
"""
外部变异系统模块
参考generate_graph.py的实现方式，支持配置变异和单decoder节点
支持合并配置文件并保存每次变异结果
"""
import os
import copy
import random
from ruamel.yaml import YAML
import torch
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from utils.runtime import model_helpers
from utils.runtime.core.graph import Graph, Node

yaml = YAML()

UPSTREAM_EFFECTIVE_MODEL_STRUCTURE_ARGS = [
    'num_layers', 'hidden_size', 'ffn_hidden_size', 'num_attention_heads', 'num_query_groups',
    'max_position_embeddings', 'q_lora_rank', 'kv_lora_rank', 'qk_rope_head_dim', 'qk_nope_head_dim',
    'v_head_dim', 'moe_intermediate_size', 'group_query_attention', 'multi_latent_attention',
    'rotary_percent', 'num_moe_experts', 'n_shared_experts', 'moe_router_topk', 'topk_group'
]

UPSTREAM_MUTATION_PARAM_OVERRIDES = {
    'num_moe_experts': {'enums': [1, 2]},
    'moe_router_topk': {'enums': [0, 1, 2, 6]},
    'moe_layer_freq': {'enums': [1, 0]},
    'n_shared_experts': {'enums': [2, 4]},
}

INERT_MUTATION_PARAMS = {
    'moe_ffn_hidden_size',
    'rope_scaling_type',
    'rope_scaling_factor',
    'rope_scaling_original_max_position_embeddings',
    'kv_channels',
    'untie_embeddings_and_output_weights',
    'disable_bias_linear',
    'swiglu',
    'no_gradient_accumulation_fusion',
}


def _effective_mutation_schema(model_structure_args: List[str], mutable_params: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    effective_params = copy.deepcopy(mutable_params or {})
    for key in INERT_MUTATION_PARAMS:
        effective_params.pop(key, None)
    for key, value in UPSTREAM_MUTATION_PARAM_OVERRIDES.items():
        if key in effective_params:
            effective_params[key] = copy.deepcopy(value)

    effective_structure_args = [
        key for key in UPSTREAM_EFFECTIVE_MODEL_STRUCTURE_ARGS if key in effective_params
    ]
    return effective_structure_args, effective_params


# ========== 变异规则装载（从YAML读取） ==========
def _load_mutation_schema(default_schema_path: str = None) -> Dict[str, Any]:
    """
    从YAML装载变异规则，包含：
      - model_structure_args: List[str]
      - mutable_params: Dict[str, constraints]
    若读取失败，返回空字典，由调用方使用内置默认值兜底。
    支持通过环境变量 MUTATION_SCHEMA_PATH 指定路径。
    """
    import os as _os
    schema_path = _os.environ.get('MUTATION_SCHEMA_PATH') or default_schema_path
    schema_path = model_helpers.resolve_repo_path(schema_path)
    if not schema_path:
        print("找不到变异参数yaml文件")
        return {}
    try:
        with open(schema_path, 'r', encoding='utf-8') as _f:
            data = yaml.load(_f)
            print("读取变异参数YAML成功")
            if not isinstance(data, dict):
                return {}
            return data
    except Exception as _e:
        print(f"✗ 读取变异参数YAML失败: {schema_path} -> {_e}")
        return {}


mlp_args = [
    'num_moe_experts',
    'moe_router_topk',
    'moe_router_load_balancing_type',
    'moe_aux_loss_coeff',
    'expert_model_parallel_size',
    'moe_grouped_gemm',
]

deepseekv3_mlp = {
    'num_moe_experts': 160,
    'moe_router_topk': 6,
    'moe_router_load_balancing_type': 'group_limited_greedy',
    'moe_aux_loss_coeff': 0.003,
    'expert_model_parallel_size': 2,
    'moe_grouped_gemm': True,
}

grok_mlp = {
    'num_moe_experts': 8,
    'moe_router_topk': 2,
    'moe_router_load_balancing_type': 'aux_loss',
    'moe_aux_loss_coeff': 1e-2,
    'expert_model_parallel_size': 2,
    'moe_grouped_gemm': True,
}

mixtral_mlp = {
    'num_moe_experts': 4,
    'moe_router_topk': 2,
    'moe_router_load_balancing_type': 'aux_loss',
    'moe_aux_loss_coeff': 0.02,
    'expert_model_parallel_size': 2,
    'moe_grouped_gemm': True,
}

qwen2_mlp = {
    'num_moe_experts': None,
    'moe_router_topk': 2,
    'moe_router_load_balancing_type': 'aux_loss',
    'moe_aux_loss_coeff': 0.0,
    'expert_model_parallel_size': 1,
    'moe_grouped_gemm': False,
}

self_attention_args = [
    'qk_layernorm',
    'attention_dropout',
    'num_attention_heads',
    'num_query_groups',
    'hidden_size',
    'ffn_hidden_size',
    'num_layers'
]

deepseekv3_self_attention = {
    'qk_layernorm': True,
    'attention_dropout': 0.0,
    'num_attention_heads': 16,
    'num_query_groups': None,
    'num_layers': 8,
    'hidden_size': 896,
    'ffn_hidden_size': 2304
}

grok_self_attention = {
    'qk_layernorm': False,
    'attention_dropout': 0.0,
    'num_attention_heads': 24,
    'num_query_groups': 4,
    'num_layers': 6,
    'hidden_size': 3072,
    'ffn_hidden_size': 16384
}

chatglm3_self_attention = {
    'qk_layernorm': False,
    'attention_dropout': 0.0,
    'num_attention_heads': 32,
    'num_query_groups': 2,
    'num_layers': 28,
    'hidden_size': 4096,
    'ffn_hidden_size': 13696
}

qwen2_self_attention = {
    'qk_layernorm': False,
    'attention_dropout': 0.0,
    'num_attention_heads': 16,
    'num_query_groups': 4,
    'num_layers': 16,
    'hidden_size': 1024,
    'ffn_hidden_size': 1024
}




class ConfigMutator:
    """配置变异器 - 负责对TransformerConfig进行变异和保存"""

    def __init__(self, structure_config_path: str = "assets/runtime/configs/structure_config.yaml",
                 template_config_path: str = "assets/runtime/configs/template_config.yaml",
                 output_dir: str = "mutated_configs",
                 config_dir: str = "model_config",
                 mutation_args_path: str = "assets/runtime/configs/mutation_schema.yaml"):
        """
        初始化配置变异器

        Args:
            structure_config_path: 结构配置文件路径
            template_config_path: 模板配置文件路径
            output_dir: 变异配置保存目录
        """
        _schema = _load_mutation_schema(default_schema_path=mutation_args_path)

        # 从YAML读取：model_structure_args
        self.model_structure_args = _schema.get('model_structure_args') if isinstance(_schema.get('model_structure_args'),
                                                                                 list) else None
        if not self.model_structure_args:
            # 兜底默认
            self.model_structure_args = [
                'num_layers', 'hidden_size', 'ffn_hidden_size', 'num_attention_heads', 'num_query_groups',
                'max_position_embeddings', 'q_lora_rank', 'kv_lora_rank', 'qk_rope_head_dim', 'qk_nope_head_dim',
                'v_head_dim', 'moe_intermediate_size', 'group_query_attention', 'multi_latent_attention',
                'rotary_percent', 'num_moe_experts', 'n_shared_experts', 'moe_router_topk', 'topk_group'
            ]

        # 从YAML读取：mutable_params
        self.mutable_params = _schema.get('mutable_params') if isinstance(_schema.get('mutable_params'), dict) else None
        if not self.mutable_params:
            self.mutable_params = {
                'num_layers': {'min_factor': 0.5, 'max_factor': 2.0, 'min_val': 4, 'max_val': 16},
                'hidden_size': {'min_factor': 0.5, 'max_factor': 2.0, 'min_val': 512, 'max_val': 4096},
                'ffn_hidden_size': {'min_factor': 0.5, 'max_factor': 2.0, 'min_val': 1024, 'max_val': 4096},
                'num_attention_heads': {'min_factor': 0.5, 'max_factor': 2.0, 'min_val': 4, 'max_val': 16},
                'num_query_groups': {'min_factor': 0.5, 'max_factor': 2.0, 'min_val': 4, 'max_val': 16},
                'max_position_embeddings': {'min_factor': 0.5, 'max_factor': 2.0, 'min_val': 1024, 'max_val': 40960},
                'moe_ffn_hidden_size': {'min_factor': 0.5, 'max_factor': 2.0, 'min_val': 512, 'max_val': 8192},
                'q_lora_rank': {'enums': [768, 1536]},
                'kv_lora_rank': {'enums': [256, 512]},
                'moe_router_load_balancing_type': {'enums': ['aux_loss', 'group_limited_greedy']},
                'rope_scaling_type': {'enums': ['yarn', 'llama3', 'longrope', 'plm']},
                'rope_scaling_factor': {'enums': [1, 2, 4, 8, 16]},
                'rope_scaling_original_max_position_embeddings': {'enums': [1024, 2048, 4096, 8192]},
                'sequence_parallel': {'enums': [True, False]},
                'normalization': {'enums': ['RMSNorm', 'LayerNorm']},
                'layernorm_epsilon': {'enums': [1e-5, 1e-6, 1e-7]},
                'gated_linear_unit': {'enums': [True, False]},
                'attention_dropout': {'enums': [0.0]},
                'hidden_dropout': {'enums': [0.0]},
                'weight_decay': {'enums': [1e-2, 1e-3, 1e-4, 0.0]},
                'add_qkv_bias': {'enums': [True, False]},
                'drop_path_rate': {'enums': [0.0]},
                'group_query_attention': {'enums': [True, False]},
                'position_embedding_type': {'enums': ['rope', 'learned_absolute']},
                'rotary_percent': {'enums': [0.0, 1.0]},
                'rotary_base': {'enums': [1e5, 5e6]},
                'multi_latent_attention': {'enums': [True, False]},
                'qk_rope_head_dim': {'enums': [64, 128]},
                'qk_nope_head_dim': {'enums': [64, 128]},
                'v_head_dim': {'enums': [64, 128]},
                'kv_channels': {'enums': [64, 128]},
                'qk_layernorm': {'enums': [True, False]},
                'use_flash_attn': {'enums': [True, False]},
                'attention_softmax_in_fp32': {'enums': [True]},
                'no_masked_softmax_fusion': {'enums': [True, False]},
                'moe_grouped_gemm': {'enums': [True, False]},
                'num_moe_experts': {'enums': [2, 4, 8, 16, 32, 64]},
                'moe_aux_loss_coeff': {'enums': [0, 1e-2]},
                'moe_router_topk': {'enums': [1, 2, 4, 8]},
                'moe_router_pre_softmax': {'enums': [True, False]},
                'moe_permutation_async_comm': {'enums': [True, False]},
                'use_fused_moe_token_permute_and_unpermute': {'enums': [True, False]},
                'moe_token_dispatcher_type': {'enums': ['alltoall']},
                'embedding_multiplier_scale': {'enums': [0, 78.38]},
                'output_multiplier_scale': {'enums': [0, 0.57]},
                'first_k_dense_replace': {'enums': [1, 0]},
                'moe_layer_freq': {'enums': [1, 2, 4]},
                'n_shared_experts': {'enums': [0, 1, 2, 4]},
                'untie_embeddings_and_output_weights': {'enums': [True, False]},
                'disable_bias_linear': {'enums': [True, False]},
                'swiglu': {'enums': [True, False]},
                'no_gradient_accumulation_fusion': {'enums': [True, False]},
                'moe_intermediate_size': {'enums': [768, 1536]},
                'topk_group': {'enums': [3, 6]},
                'moe_device_level_aux_loss_coeff': {'enums': [0.05, 0.03]},
                'moe_comm_aux_loss_coeff': {'enums': [0.02, 0.01]},
                'routed_scaling_factor': {'enums': [16.0, 8.0]},
                'seq_aux': {'enums': [True, False]},
                'input_jitter': {'enums': [True, False]},
                'use_cp_send_recv_overlap': {'enums': [True, False]}
            }

        self.model_structure_args, self.mutable_params = _effective_mutation_schema(
            self.model_structure_args,
            self.mutable_params,
        )
        self.other_args = [key for key in self.mutable_params.keys() if key not in self.model_structure_args]



        self.structure_config_path = model_helpers.resolve_repo_path(structure_config_path)
        self.template_config_path = model_helpers.resolve_repo_path(template_config_path)
        self.output_dir = output_dir
        self.config_dir = config_dir
        self.mutation_args_path = model_helpers.resolve_repo_path(mutation_args_path)
        self._base_seed = self._resolve_base_seed()
        self._mutation_invocation = 0

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 加载并合并配置文件
        self.structure_config = self._load_yaml(structure_config_path)
        self.template_config = self._load_yaml(template_config_path)
        self.combined_config = self._merge_configs()

        # 基础配置模板（参考generate_graph.py）
        self.base_config = {
            'tensor_model_parallel_size': 1,
            'pipeline_model_parallel_size': 1,
            'num_layers': 24,
            'hidden_size': 896,
            'ffn_hidden_size': 4864,
            'num_attention_heads': 14,
            'num_query_groups': 2,
            'attention_dropout': 0.0,
            'init_method_std': 0.01,
            'hidden_dropout': 0.0,
            'normalization': "RMSNorm",
            'layernorm_epsilon': 1e-6
        }

        print(f"✓ 成功加载并合并配置文件")
        print(f"  结构配置: {structure_config_path}")
        print(f"  模板配置: {template_config_path}")
        print(f"  输出目录: {output_dir}")

    def _resolve_base_seed(self) -> int:
        seed_text = os.environ.get("BASE_SEED", "").strip()
        if seed_text:
            try:
                return int(seed_text)
            except ValueError:
                pass
        return 42

    def _make_rng(self, *parts) -> random.Random:
        round_id = os.environ.get("MUTATE_ROUND", "").strip() or "0"
        return model_helpers.make_rng(self._base_seed, "ConfigMutator", round_id, *parts)

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        """宽松整型解析，兼容 None、'None'、空串等配置值。"""
        if value is None:
            return default
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized or normalized.lower() in {"none", "null"}:
                return default
            value = normalized
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _load_yaml(self, file_path: str) -> Dict[str, Any]:
        """加载yaml文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.load(f)
        except Exception as e:
            print(f"✗ 加载配置文件失败 {file_path}: {e}")
            # 如果文件不存在，返回空字典
            return {}

    def _merge_configs(self) -> Dict[str, Any]:
        """合并两个配置文件"""
        combined = {
            # 从template_config获取基础配置
            'base_config': copy.deepcopy(self.template_config),
            # 从structure_config获取图结构
            'graph_structure': copy.deepcopy(self.structure_config),
            # 添加元数据
            'metadata': {
                'created_from': {
                    'structure_config': self.structure_config_path,
                    'template_config': self.template_config_path
                },
                'creation_time': datetime.now().isoformat()
            }
        }
        return combined

    # def mutate_config(self, base_config: Dict[str, Any] = None, mutation_rate: float = 0.3,
    #                  is_last_decoder: bool = False, graph_hidden_size: int = None) -> TransformerConfig:
    #     """
    #     对TransformerConfig进行变异

    #     Args:
    #         base_config: 基础配置字典，如果为None则使用默认配置
    #         mutation_rate: 变异概率
    #         is_last_decoder: 是否为最后一个decoder层
    #         graph_hidden_size: graph配置中的hidden_size，用于最后一个decoder层

    #     Returns:
    #         TransformerConfig: 变异后的配置
    #     """
    #     if base_config is None:
    #         base_config = self.base_config.copy()
    #     else:
    #         base_config = base_config.copy()

    #     # 定义可变异的参数及其变异范围
    #     mutable_params = {
    #         'num_layers': {'min_factor': 0.5, 'max_factor': 1.5, 'min_val': 1, 'max_val': 8},
    #         'hidden_size': {'min_factor': 0.7, 'max_factor': 1.3, 'min_val': 512, 'max_val': 2048},
    #         'ffn_hidden_size': {'min_factor': 0.7, 'max_factor': 1.3, 'min_val': 1024, 'max_val': 4096},
    #         'num_attention_heads': {'min_factor': 0.5, 'max_factor': 1.5, 'min_val': 4, 'max_val': 16},
    #         'num_query_groups': {'min_factor': 0.5, 'max_factor': 2.0, 'min_val': 1, 'max_val': 8},
    #         'max_position_embeddings': {'min_factor': 0.5, 'max_factor': 2.0, 'min_val': 1024, 'max_val': 40960},
    #         'tensor_model_parallel_size': {'min_factor': 0.5, 'max_factor': 2.0, 'min_val': 1, 'max_val': 8},
    #         'pipeline_model_parallel_size': {'min_factor': 0.5, 'max_factor': 2.0, 'min_val': 1, 'max_val': 8}
    #     }

    #     print(f"变异率: {mutation_rate}")

    #     for param_name, constraints in mutable_params.items():
    #         if param_name in base_config:
    #             # 特殊处理：如果是最后一个decoder层，hidden_size必须与graph保持一致
    #             if param_name == 'hidden_size' and is_last_decoder and graph_hidden_size is not None:
    #                 if base_config[param_name] != graph_hidden_size:
    #                     print(f"    强制设置最后decoder层的 hidden_size: {base_config[param_name]} -> {graph_hidden_size}")
    #                     base_config[param_name] = graph_hidden_size
    #                     # 相应调整ffn_hidden_size
    #                     if 'ffn_hidden_size' in base_config:
    #                         ratio = base_config['ffn_hidden_size'] / base_config[param_name] if base_config[param_name] != 0 else 4
    #                         base_config['ffn_hidden_size'] = int(graph_hidden_size * ratio)
    #                 continue

    #             # 正常变异逻辑
    #             if random.random() < mutation_rate:
    #                 print(f"mutating {param_name}")
    #                 original_value = base_config[param_name]

    #                 # 计算变异范围
    #                 min_val = max(constraints['min_val'],
    #                             int(original_value * constraints['min_factor']))
    #                 max_val = min(constraints['max_val'],
    #                             int(original_value * constraints['max_factor']))

    #                 # 确保min_val <= max_val
    #                 if min_val > max_val:
    #                     min_val, max_val = max_val, min_val

    #                 # 生成新值
    #                 if param_name in ['num_attention_heads', 'num_query_groups']:
    #                     # 注意力头数需要是合理的值
    #                     possible_values = [2**i for i in range(1, 5) if 2**i >= min_val and 2**i <= max_val]
    #                     if possible_values:
    #                         new_value = random.choice(possible_values)
    #                     else:
    #                         new_value = original_value
    #                 else:
    #                     new_value = random.randint(min_val, max_val)

    #                 base_config[param_name] = new_value

    #     # 创建TransformerConfig对象
    #     return TransformerConfig(**base_config)

    def mutate_config_dict(self, base_config: Dict[str, Any] = None, mutation_num: int = 3,
                           is_last_decoder: bool = False, graph_hidden_size: int = None) -> Dict[str, Any]:
        """
        对TransformerConfig进行变异

        Args:
            base_config: 基础配置字典，如果为None则使用默认配置
            mutation_num: 要变异的参数数量（仅用于基于参数数量的变异）
            is_last_decoder: 是否为最后一个decoder层
            graph_hidden_size: graph配置中的hidden_size，用于最后一个decoder层

        Returns:
            Dict[str, Any]: 变异后的配置字典
        """
        if base_config is None:
            base_config = self.base_config.copy()
        else:
            base_config = base_config.copy()

        print("\n=== 调用 mutate_config_dict ===")
        self._mutation_invocation += 1
        rng = self._make_rng(
            "mutate_config_dict",
            self._mutation_invocation,
            mutation_num,
            int(is_last_decoder),
            graph_hidden_size if graph_hidden_size is not None else "none"
        )

        # 随机选择变异方式：0.2概率选择整体变异，0.8概率选择参数数量变异

        rand_val = rng.random()
        if rand_val < 0.2:
            mutation_type = 'config_based'
        else:
            mutation_type = 'param_count_based'
         
        print(f"选择的变异方式: {mutation_type}")

        if mutation_type == 'config_based':
            return self._mutate_config_based(base_config, rng, is_last_decoder, graph_hidden_size)
        return self._mutate_param_count_based(base_config, rng, mutation_num, is_last_decoder, graph_hidden_size)

    def _mutate_config_based(self, base_config: Dict[str, Any], rng: random.Random,
                             is_last_decoder: bool = False,
                             graph_hidden_size: int = None) -> Dict[str, Any]:
        """
        基于配置的整体变异方法

        Args:
            base_config: 基础配置字典
            is_last_decoder: 是否为最后一个decoder层
            graph_hidden_size: graph配置中的hidden_size

        Returns:
            Dict[str, Any]: 变异后的配置字典
        """
        print("执行基于配置的整体变异")

        # 随机选择变异类型：mlp_args 或 self_attention_args
        rand_val = rng.random()
        if rand_val < 0.5:
            mutation_subtype = 'mlp_args'
        else:
            mutation_subtype = 'self_attention_args'

        print(f"选择的变异子类型: {mutation_subtype}")

        def _apply_preset_config(target: Dict[str, Any], preset: Dict[str, Any], preset_name: str, tag: str) -> bool:
            """将预设配置应用到 target，返回是否产生变化"""
            any_change = False
            for param_name, param_value in preset.items():
                if param_name in target:
                    original_value = target[param_name]
                    if original_value != param_value:
                        any_change = True
                    target[param_name] = param_value
                    print(f"{tag}变异 {param_name}: {original_value} -> {param_value}")
                else:
                    target[param_name] = param_value
                    any_change = True
                    print(f"{tag}新增 {param_name}: {param_value}")
            print(f"应用{tag}配置: {preset_name}，是否发生变化: {any_change}")
            return any_change

        def _force_minimal_tweak(target: Dict[str, Any]) -> None:
            """若所有预设均不产生变化，进行一次最小可行改动以保证配置改变"""
            print("未产生变化，进行强制微调以确保配置改变")
            # 优先尝试布尔型参数翻转
            for key in ['sequence_parallel', 'gated_linear_unit', 'qk_layernorm']:
                if key in target and isinstance(target[key], bool):
                    target[key] = not target[key]
                    print(f"强制微调: 翻转 {key} -> {target[key]}")
                    return
            # 尝试对整数参数做 ±1 微调
            for key in ['num_layers', 'hidden_size', 'ffn_hidden_size', 'num_attention_heads', 'num_query_groups']:
                if key in target and isinstance(target[key], int):
                    new_val = max(1, target[key] + 1)
                    if new_val != target[key]:
                        print(f"强制微调: {key} {target[key]} -> {new_val}")
                        target[key] = new_val
                        return

        changed = False
        if mutation_subtype == 'mlp_args':
            mlp_configs = {
                'deepseekv3_mlp': deepseekv3_mlp,
                'grok_mlp': grok_mlp,
                'mixtral_mlp': mixtral_mlp,
                'qwen2_mlp': qwen2_mlp
            }
            candidate_names = list(mlp_configs.keys())
            rng.shuffle(candidate_names)
            for name in candidate_names:
                if _apply_preset_config(base_config, mlp_configs[name], name, tag="MLP"):
                    changed = True
                    break
            if not changed:
                # 尝试自注意力预设
                attention_configs = {
                    'deepseekv3_self_attention': deepseekv3_self_attention,
                    'grok_self_attention': grok_self_attention,
                    'chatglm3_self_attention': chatglm3_self_attention,
                    'qwen2_self_attention': qwen2_self_attention
                }
                candidate_names = list(attention_configs.keys())
                rng.shuffle(candidate_names)
                for name in candidate_names:
                    if _apply_preset_config(base_config, attention_configs[name], name, tag="注意力"):
                        changed = True
                        break
            if not changed:
                _force_minimal_tweak(base_config)
        else:
            attention_configs = {
                'deepseekv3_self_attention': deepseekv3_self_attention,
                'grok_self_attention': grok_self_attention,
                'chatglm3_self_attention': chatglm3_self_attention,
                'qwen2_self_attention': qwen2_self_attention
            }
            candidate_names = list(attention_configs.keys())
            rng.shuffle(candidate_names)
            for name in candidate_names:
                if _apply_preset_config(base_config, attention_configs[name], name, tag="注意力"):
                    changed = True
                    break
            if not changed:
                # 尝试 MLP 预设
                mlp_configs = {
                    'deepseekv3_mlp': deepseekv3_mlp,
                    'grok_mlp': grok_mlp,
                    'mixtral_mlp': mixtral_mlp,
                    'qwen2_mlp': qwen2_mlp
                }
                candidate_names = list(mlp_configs.keys())
                rng.shuffle(candidate_names)
                for name in candidate_names:
                    if _apply_preset_config(base_config, mlp_configs[name], name, tag="MLP"):
                        changed = True
                        break
            if not changed:
                _force_minimal_tweak(base_config)

        # 处理最后一个decoder层的hidden_size约束
        if is_last_decoder and graph_hidden_size is not None:
            if 'hidden_size' in base_config and base_config['hidden_size'] != graph_hidden_size:
                print(f"调整最后一个decoder层的hidden_size: {base_config['hidden_size']} -> {graph_hidden_size}")
                base_config['hidden_size'] = graph_hidden_size
                if 'ffn_hidden_size' in base_config:
                    ratio = base_config['ffn_hidden_size'] / base_config['hidden_size'] if base_config[
                                                                                               'hidden_size'] != 0 else 4
                    base_config['ffn_hidden_size'] = int(graph_hidden_size * ratio)
        # 全局约束（顺序重要）：
        # 1) 保证 num_attention_heads 是 num_query_groups 的倍数
        # 2) 再保证 hidden_size 是 num_attention_heads 的倍数
        self._enforce_heads_groups_constraint(base_config)
        self._enforce_hidden_size_heads_constraint(base_config)
        self._enforce_moe_router_constraints(base_config)

        return base_config

    def _mutate_param_count_based(self, base_config: Dict[str, Any], rng: random.Random,
                                  mutation_num: int = 3,
                                  is_last_decoder: bool = False, graph_hidden_size: int = None) -> Dict[str, Any]:
        """
        基于参数数量的变异方法（原有逻辑）

        Args:
            base_config: 基础配置字典
            mutation_num: 要变异的参数数量
            is_last_decoder: 是否为最后一个decoder层
            graph_hidden_size: graph配置中的hidden_size

        Returns:
            Dict[str, Any]: 变异后的配置字典
        """
        print("执行基于参数数量的变异")
        print(f"变异参数数量: {mutation_num}")

        # 选择参数来源集合：70% 概率使用结构相关参数，30% 概率使用其它参数
        rand_val = rng.random()
        if rand_val < 0.7:
            group_choice = 'model_shape_and_size'
        else:
            group_choice = 'other'
        chosen_pool = self.model_structure_args if group_choice == 'model_shape_and_size' else self.other_args
        print(f"参数数量变异分组选择: {group_choice}（候选池大小={len(chosen_pool)}）")

        # 从所选分组中筛选实际可用的参数（存在于 base_config 且在可变更集合中）
        available_params = [
            p for p in chosen_pool
            if p in base_config and p in self.mutable_params
        ]

        # 最后一层 decoder 的 hidden_size 受限，若存在则先处理并剔除该参数
        if 'hidden_size' in available_params and is_last_decoder and graph_hidden_size is not None:
            if base_config['hidden_size'] != graph_hidden_size:
                base_config['hidden_size'] = graph_hidden_size
                if 'ffn_hidden_size' in base_config:
                    ratio = base_config['ffn_hidden_size'] / base_config['hidden_size'] if base_config[
                                                                                               'hidden_size'] != 0 else 4
                    base_config['ffn_hidden_size'] = int(graph_hidden_size * ratio)
            available_params.remove('hidden_size')

        # 不再要求严格达到指定的变异数量
        if len(available_params) < mutation_num:
            print(f"可用于变异的参数数量不足（候选={len(available_params)}，期望={mutation_num}）。分组={group_choice}。已缩小期望！")
            mutation_num=len(available_params)

        desired_mutation_num = mutation_num
        selected_params = []
        mutated_params = []
        available_copy = available_params.copy()

        while len(mutated_params) < desired_mutation_num and available_copy:
            param_name = rng.choice(available_copy)
            available_copy.remove(param_name)

            original_value = base_config[param_name]
            constraints = self.mutable_params[param_name]
            new_value = original_value  # 初始化

            # enum 类型
            if 'enums' in constraints:
                enum_choices = [v for v in constraints['enums'] if v != original_value]
                if enum_choices:
                    new_value = rng.choice(enum_choices)
                else:
                    # 兜底：若只有一个枚举与原值相同，尝试对布尔类型取反
                    if isinstance(original_value, bool):
                        new_value = not original_value

            # 范围类型
            elif 'min_val' in constraints and 'max_val' in constraints:
                min_val = max(constraints['min_val'], int(original_value * constraints['min_factor']))
                max_val = min(constraints['max_val'], int(original_value * constraints['max_factor']))
                if min_val > max_val:
                    min_val, max_val = max_val, min_val
                if param_name in ['num_layers', 'num_attention_heads', 'num_query_groups', 'hidden_size',
                                  'ffn_hidden_size', 'max_position_embeddings']:
                    possible_values = [2 ** i for i in range(1, 13) if
                                       min_val <= 2 ** i <= max_val and 2 ** i != original_value]
                    if possible_values:
                        new_value = rng.choice(possible_values)
                else:
                    candidates = [v for v in range(min_val, max_val + 1) if v != original_value]
                    if candidates:
                        new_value = rng.choice(candidates)

                # 兜底：若仍未产生不同值，尽量在边界附近调整1个单位
                if new_value == original_value:
                    if original_value + 1 <= constraints['max_val']:
                        new_value = original_value + 1
                    elif original_value - 1 >= constraints['min_val']:
                        new_value = original_value - 1

            # 只有当产生不同值时才计入一次变异；否则尝试下一个可选参数
            if new_value != original_value:
                base_config[param_name] = new_value
                mutated_params.append(param_name)
                selected_params.append(param_name)
                print(f"变异 {param_name}: {original_value} -> {new_value}")

        if len(mutated_params) < desired_mutation_num:
            print(f"无法完成指定数量的参数变异（完成={len(mutated_params)}，期望={desired_mutation_num}）。")
            desired_mutation_num = len(mutated_params)

        # 添加约束：如果变异了num_attention_heads或num_query_groups，确保num_attention_heads是num_query_groups的倍数
        if 'num_attention_heads' in selected_params or 'num_query_groups' in selected_params:
            print("\n=== 应用注意力头约束 ===")
            num_attention_heads = base_config.get('num_attention_heads', 1)
            num_query_groups = base_config.get('num_query_groups', 1)

            print(f"当前 num_attention_heads: {num_attention_heads}")
            print(f"当前 num_query_groups: {num_query_groups}")

            # 确保num_attention_heads是num_query_groups的倍数
            if num_attention_heads % num_query_groups != 0:
                print(
                    f"约束冲突：num_attention_heads ({num_attention_heads}) 不是 num_query_groups ({num_query_groups}) 的倍数")

                # 调整num_attention_heads使其成为num_query_groups的倍数
                # 选择最接近原值的有效值
                quotient = num_attention_heads // num_query_groups
                if quotient == 0:
                    # 如果num_attention_heads太小，设置为num_query_groups
                    adjusted_heads = num_query_groups
                else:
                    # 选择最接近的倍数
                    lower_multiple = quotient * num_query_groups
                    upper_multiple = (quotient + 1) * num_query_groups

                    # 选择距离原值更近的倍数
                    if abs(num_attention_heads - lower_multiple) <= abs(num_attention_heads - upper_multiple):
                        adjusted_heads = lower_multiple
                    else:
                        adjusted_heads = upper_multiple

                # 确保调整后的值在合理范围内
                min_heads = self.mutable_params['num_attention_heads']['min_val']
                max_heads = self.mutable_params['num_attention_heads']['max_val']
                adjusted_heads = max(min_heads, min(max_heads, adjusted_heads))

                print(f"调整 num_attention_heads: {num_attention_heads} -> {adjusted_heads}")
                base_config['num_attention_heads'] = adjusted_heads

                # 如果调整后的值不在2的幂次中，尝试找到最接近的2的幂次
                if adjusted_heads not in [2 ** i for i in range(1, 13)]:
                    # 找到最接近的2的幂次
                    import math
                    log2 = math.log2(adjusted_heads)
                    lower_power = 2 ** int(log2)
                    upper_power = 2 ** (int(log2) + 1)

                    if abs(adjusted_heads - lower_power) <= abs(adjusted_heads - upper_power):
                        final_heads = lower_power
                    else:
                        final_heads = upper_power

                    # 确保最终值仍然是num_query_groups的倍数且在范围内
                    if final_heads % num_query_groups == 0 and min_heads <= final_heads <= max_heads:
                        print(f"进一步调整到2的幂次: {adjusted_heads} -> {final_heads}")
                        base_config['num_attention_heads'] = final_heads
                    else:
                        # 如果2的幂次不满足约束，保持之前的调整值
                        print(f"保持调整值 {adjusted_heads}（2的幂次不满足约束）")
            # 检查hidden_size是否存在
            hidden_size = base_config.get('hidden_size', None)
            if hidden_size is not None:
                print(f"当前 hidden_size: {hidden_size}")

                # 确保hidden_size // num_attention_heads是偶数
                quotient = hidden_size // num_attention_heads
                if quotient % 2 != 0:
                    print(f"约束冲突：hidden_size ({hidden_size}) // num_attention_heads ({num_attention_heads}) 不是偶数")

                    # 调整hidden_size使其满足约束
                    adjusted_hidden_size = hidden_size
                    if quotient == 1:
                        adjusted_hidden_size = num_attention_heads * 2
                    else:
                        adjusted_hidden_size = (quotient + 1) * num_attention_heads

                    # 确保调整后的hidden_size在合理范围内
                    min_hidden = self.mutable_params['hidden_size']['min_val']
                    max_hidden = self.mutable_params['hidden_size']['max_val']
                    adjusted_hidden_size = max(min_hidden, min(max_hidden, adjusted_hidden_size))

                    print(f"调整 hidden_size: {hidden_size} -> {adjusted_hidden_size}")
                    base_config['hidden_size'] = adjusted_hidden_size
                else:
                    print(f"约束满足：hidden_size ({hidden_size}) // num_attention_heads ({num_attention_heads}) 是偶数")

        # 全局约束（顺序重要）：先头-组，再隐藏维-头
        self._enforce_heads_groups_constraint(base_config)
        self._enforce_hidden_size_heads_constraint(base_config)
        self._enforce_moe_router_constraints(base_config)

        return base_config

    def _enforce_moe_router_constraints(self, base_config: Dict[str, Any]) -> None:
        """修复已知的 MoE 路由参数组合约束，避免生成运行期非法配置。"""
        model_helpers.enforce_moe_bias_constraint(base_config, log=True)

        if 'moe_router_topk' not in base_config:
            return

        try:
            moe_router_topk = int(base_config.get('moe_router_topk'))
        except (TypeError, ValueError):
            return

        try:
            num_experts = base_config.get('num_moe_experts', base_config.get('num_experts'))
            num_experts = int(num_experts) if num_experts is not None else None
        except (TypeError, ValueError):
            num_experts = None

        if moe_router_topk <= 0:
            print(f"修复 moe_router_topk: {moe_router_topk} -> 1")
            base_config['moe_router_topk'] = 1
            moe_router_topk = 1

        if num_experts is not None and num_experts > 0 and moe_router_topk > num_experts:
            print(f"修复 moe_router_topk: {moe_router_topk} -> {num_experts}（不能超过专家数）")
            base_config['moe_router_topk'] = num_experts
            moe_router_topk = num_experts

        if moe_router_topk == 1 and not bool(base_config.get('moe_router_pre_softmax', False)):
            print("修复 moe_router_pre_softmax: False -> True（topk=1 时必需）")
            base_config['moe_router_pre_softmax'] = True

    def _enforce_hidden_size_heads_constraint(self, base_config: Dict[str, Any]) -> None:
        """
        约束：hidden_size 必须为 num_attention_heads 的倍数；若不满足，则就近调整 hidden_size。
        同时保持 ffn_hidden_size 与 hidden_size 的比例不变（若存在）。
        """
        try:
            hidden_size = int(base_config.get('hidden_size')) if base_config.get('hidden_size') is not None else None
            num_heads = int(base_config.get('num_attention_heads')) if base_config.get(
                'num_attention_heads') is not None else None
        except (TypeError, ValueError):
            hidden_size, num_heads = None, None

        if hidden_size is None or num_heads is None or num_heads <= 0:
            return

        remainder = hidden_size % num_heads
        quotient = hidden_size // num_heads

        # 确保 hidden_size 是 num_heads 的偶数倍
        if remainder == 0 and quotient % 2 == 0:
            return

        # 记录原始比例用于调整 ffn_hidden_size
        ffn_hidden_size = base_config.get('ffn_hidden_size')
        ratio = None
        if isinstance(ffn_hidden_size, int) and hidden_size != 0:
            ratio = ffn_hidden_size / hidden_size

        # 优先向上取整到最近偶数倍数，避免过小
        ceil_multiple = ((hidden_size + num_heads - 1) // num_heads) * num_heads
        if (ceil_multiple // num_heads) % 2 != 0:
            ceil_multiple += num_heads

        floor_multiple = (hidden_size // num_heads) * num_heads
        if (floor_multiple // num_heads) % 2 != 0:
            floor_multiple -= num_heads

        # 选择更接近原值的偶数倍数，且不小于 num_heads
        if floor_multiple < num_heads:
            new_hidden_size = ceil_multiple
        else:
            if abs(ceil_multiple - hidden_size) < abs(hidden_size - floor_multiple):
                new_hidden_size = ceil_multiple
            else:
                new_hidden_size = floor_multiple if floor_multiple > 0 else num_heads

        if new_hidden_size != hidden_size:
            print(f"应用约束：调整 hidden_size 以整除注意力头数 ({num_heads})：{hidden_size} -> {new_hidden_size}")
            base_config['hidden_size'] = new_hidden_size
            # 同步调整 ffn_hidden_size，尽量保持比例
            if ratio is not None:
                adjusted_ffn = int(new_hidden_size * ratio)
                if adjusted_ffn != ffn_hidden_size:
                    print(f"同步调整 ffn_hidden_size 以保持比例：{ffn_hidden_size} -> {adjusted_ffn}")
                    base_config['ffn_hidden_size'] = adjusted_ffn

    def _enforce_heads_groups_constraint(self, base_config: Dict[str, Any]) -> None:
        """
        约束：num_attention_heads 必须为 num_query_groups 的倍数；
        若不满足，优先微调 num_attention_heads 到最近倍数（不小于 num_query_groups 且为正）。
        """
        try:
            heads = int(base_config.get('num_attention_heads')) if base_config.get(
                'num_attention_heads') is not None else None
            groups = int(base_config.get('num_query_groups')) if base_config.get(
                'num_query_groups') is not None else None
        except (TypeError, ValueError):
            heads, groups = None, None

        if heads is None or groups is None or groups <= 0:
            return

        if heads % groups == 0:
            return

        # 计算最近的上下倍数
        ceil_multiple = ((heads + groups - 1) // groups) * groups
        floor_multiple = (heads // groups) * groups
        if floor_multiple < groups:
            new_heads = max(groups, ceil_multiple)
        else:
            # 选择更接近原值的倍数
            if abs(ceil_multiple - heads) < abs(heads - floor_multiple):
                new_heads = ceil_multiple
            else:
                new_heads = floor_multiple if floor_multiple > 0 else groups

        if new_heads != heads:
            print(f"应用约束：调整 num_attention_heads 以整除 num_query_groups ({groups})：{heads} -> {new_heads}")
            base_config['num_attention_heads'] = new_heads

    def _enforce_extended_config_constraints(self, base_config: Dict[str, Any], *,
                                             is_last_decoder: bool = False,
                                             graph_hidden_size: Optional[int] = None) -> None:
        """约束额外脚本/运行参数，确保整网实际变异稳定可落地。"""
        tp = 1
        try:
            tp = max(1, int(base_config.get('tensor_model_parallel_size', 1)))
        except (TypeError, ValueError):
            tp = 1

        if is_last_decoder and graph_hidden_size is not None and 'hidden_size' in base_config:
            base_config['hidden_size'] = int(graph_hidden_size)

        self._enforce_heads_groups_constraint(base_config)
        self._enforce_hidden_size_heads_constraint(base_config)

        try:
            num_heads = int(base_config.get('num_attention_heads', 0))
        except (TypeError, ValueError):
            num_heads = 0
        try:
            hidden_size = int(base_config.get('hidden_size', 0))
        except (TypeError, ValueError):
            hidden_size = 0

        if num_heads > 0 and hidden_size > 0:
            if 'kv_channels' in base_config:
                try:
                    kv_channels = max(1, int(base_config.get('kv_channels', 0)))
                except (TypeError, ValueError):
                    kv_channels = max(1, hidden_size // num_heads)

                if is_last_decoder and graph_hidden_size is not None:
                    kv_channels = max(1, int(graph_hidden_size) // num_heads)
                else:
                    target_hidden = num_heads * kv_channels
                    if target_hidden != hidden_size:
                        print(f"应用约束：调整 hidden_size 以匹配 kv_channels：{hidden_size} -> {target_hidden}")
                        base_config['hidden_size'] = target_hidden
                        hidden_size = target_hidden
                base_config['kv_channels'] = kv_channels

        swiglu_enabled = bool(base_config.get('swiglu', False)) or bool(base_config.get('gated_linear_unit', False))
        if 'ffn_hidden_size' in base_config:
            try:
                ffn_hidden_size = int(base_config.get('ffn_hidden_size', 0))
            except (TypeError, ValueError):
                ffn_hidden_size = 0
            divisor = max(1, 2 * tp if swiglu_enabled else tp)
            if ffn_hidden_size > 0 and ffn_hidden_size % divisor != 0:
                adjusted = ((ffn_hidden_size + divisor - 1) // divisor) * divisor
                print(f"应用约束：调整 ffn_hidden_size 以满足切分：{ffn_hidden_size} -> {adjusted}")
                base_config['ffn_hidden_size'] = adjusted

        if 'moe_ffn_hidden_size' in base_config:
            try:
                moe_ffn_hidden_size = int(base_config.get('moe_ffn_hidden_size', 0))
            except (TypeError, ValueError):
                moe_ffn_hidden_size = 0
            divisor = max(1, 2 * tp if swiglu_enabled else tp)
            if moe_ffn_hidden_size > 0 and moe_ffn_hidden_size % divisor != 0:
                adjusted = ((moe_ffn_hidden_size + divisor - 1) // divisor) * divisor
                print(f"应用约束：调整 moe_ffn_hidden_size 以满足切分：{moe_ffn_hidden_size} -> {adjusted}")
                base_config['moe_ffn_hidden_size'] = adjusted

        num_experts = self._safe_int(
            base_config.get('num_moe_experts', base_config.get('num_experts', 0)),
            0,
        )
        if num_experts > 0:
            base_config['num_moe_experts'] = num_experts
            if not bool(base_config.get('disable_bias_linear', False)):
                print("应用约束：MoE 场景强制关闭 bias_linear")
                base_config['disable_bias_linear'] = True
            router_topk = self._safe_int(base_config.get('moe_router_topk', 1), 1)
            router_topk = max(1, min(router_topk, num_experts))
            if base_config.get('moe_router_topk') != router_topk:
                print(f"应用约束：调整 moe_router_topk：{base_config.get('moe_router_topk')} -> {router_topk}")
                base_config['moe_router_topk'] = router_topk

            if 'n_shared_experts' in base_config:
                n_shared = self._safe_int(base_config.get('n_shared_experts', 0), 0)
                n_shared = max(0, min(n_shared, max(0, num_experts - 1)))
                if base_config.get('n_shared_experts') != n_shared:
                    print(f"应用约束：调整 n_shared_experts：{base_config.get('n_shared_experts')} -> {n_shared}")
                    base_config['n_shared_experts'] = n_shared

        self._enforce_moe_router_constraints(base_config)

        raw_rope_type = base_config.get('rope_scaling_type')
        rope_type = str(raw_rope_type).strip().lower() if raw_rope_type is not None else ''
        if rope_type in {'none', 'null'}:
            rope_type = ''
        if rope_type not in {'', 'yarn', 'llama3', 'longrope', 'plm'}:
            rope_type = ''

        if not rope_type:
            base_config.pop('rope_scaling_type', None)
            base_config.pop('rope_scaling_factor', None)
            base_config.pop('rope_scaling_original_max_position_embeddings', None)
        else:
            base_config['rope_scaling_type'] = rope_type
            factor = self._safe_int(base_config.get('rope_scaling_factor', 1), 1)
            factor = max(1, factor)
            base_config['rope_scaling_factor'] = factor
            try:
                original_max = int(base_config.get(
                    'rope_scaling_original_max_position_embeddings',
                    base_config.get('max_position_embeddings', 4096)
                ) or base_config.get('max_position_embeddings', 4096))
            except (TypeError, ValueError):
                original_max = int(base_config.get('max_position_embeddings', 4096) or 4096)
            original_max = max(1, original_max)
            base_config['rope_scaling_original_max_position_embeddings'] = original_max

    def _find_transformer_section_name(self, model_config: Dict[str, Any]) -> str:
        if 'TransformerConfig' in model_config:
            return 'TransformerConfig'
        if 'MLATransformerConfig' in model_config:
            return 'MLATransformerConfig'
        model_config.setdefault('TransformerConfig', {})
        return 'TransformerConfig'

    def _build_full_mutation_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        section_name = self._find_transformer_section_name(model_config)
        transformer_cfg = copy.deepcopy(model_config.get(section_name, {}) or {})
        extra_cfg = copy.deepcopy(model_config.get('extra_config', {}) or {})
        spec_cfg = copy.deepcopy(model_config.get('get_gpt_layer_local_spec', {}) or {})

        full_config = copy.deepcopy(transformer_cfg)

        for key in (
            'rope_scaling_type',
            'rope_scaling_factor',
            'rope_scaling_original_max_position_embeddings',
            'untie_embeddings_and_output_weights',
            'no_gradient_accumulation_fusion',
            'kv_channels',
        ):
            if key in extra_cfg:
                full_config[key] = extra_cfg[key]

        add_bias_linear = transformer_cfg.get('add_bias_linear')
        if add_bias_linear is not None:
            full_config['disable_bias_linear'] = not bool(add_bias_linear)
        full_config.setdefault('untie_embeddings_and_output_weights', bool(extra_cfg.get('untie_embeddings_and_output_weights', False)))
        full_config.setdefault('no_gradient_accumulation_fusion', bool(extra_cfg.get('no_gradient_accumulation_fusion', False)))

        swiglu = extra_cfg.get('swiglu')
        if swiglu is None:
            swiglu = transformer_cfg.get('gated_linear_unit')
        if swiglu is None:
            swiglu = extra_cfg.get('use_fused_swiglu')
        if swiglu is not None:
            full_config['swiglu'] = bool(swiglu)

        if 'num_experts' in spec_cfg and 'num_moe_experts' not in full_config:
            full_config['num_moe_experts'] = spec_cfg.get('num_experts')
        if 'moe_grouped_gemm' in spec_cfg and 'moe_grouped_gemm' not in full_config:
            full_config['moe_grouped_gemm'] = spec_cfg.get('moe_grouped_gemm')

        if 'kv_channels' not in full_config:
            try:
                hidden = int(full_config.get('hidden_size', 0))
                heads = int(full_config.get('num_attention_heads', 0))
                if hidden > 0 and heads > 0 and hidden % heads == 0:
                    full_config['kv_channels'] = hidden // heads
            except (TypeError, ValueError):
                pass

        uses_rope = str(extra_cfg.get('position_embedding_type', transformer_cfg.get('position_embedding_type', ''))).strip().lower() == 'rope'
        if uses_rope:
            max_pos = int(full_config.get('max_position_embeddings', extra_cfg.get('seq_length', 4096)) or 4096)
            full_config.setdefault('rope_scaling_factor', 1)
            full_config.setdefault('rope_scaling_original_max_position_embeddings', max_pos)

        if self._safe_int(full_config.get('num_moe_experts', 0), 0) > 0:
            full_config.setdefault('n_shared_experts', 0)
            full_config.setdefault('moe_ffn_hidden_size', self._safe_int(full_config.get('ffn_hidden_size', 0), 0))
            full_config.setdefault('moe_layer_freq', 1)
            full_config['disable_bias_linear'] = True

        return full_config

    def _apply_full_mutation_config(self, original_model_config: Dict[str, Any],
                                    mutated_flat_config: Dict[str, Any]) -> Dict[str, Any]:
        updated = copy.deepcopy(original_model_config)
        section_name = self._find_transformer_section_name(updated)
        transformer_cfg = updated.setdefault(section_name, {})
        extra_cfg = updated.setdefault('extra_config', {})
        spec_cfg = updated.setdefault('get_gpt_layer_local_spec', {})

        transformer_keys = {
            'num_layers', 'hidden_size', 'ffn_hidden_size', 'num_attention_heads', 'num_query_groups',
            'max_position_embeddings', 'q_lora_rank', 'kv_lora_rank', 'qk_rope_head_dim', 'qk_nope_head_dim',
            'v_head_dim', 'moe_intermediate_size', 'group_query_attention', 'multi_latent_attention',
            'rotary_percent', 'num_moe_experts', 'n_shared_experts', 'moe_router_topk', 'topk_group',
            'moe_router_load_balancing_type', 'moe_aux_loss_coeff', 'moe_router_pre_softmax',
            'moe_permutation_async_comm', 'moe_token_dispatcher_type', 'moe_layer_freq',
            'normalization', 'layernorm_epsilon', 'attention_dropout', 'hidden_dropout',
            'add_qkv_bias', 'attention_softmax_in_fp32', 'qk_layernorm', 'use_flash_attn',
            'moe_grouped_gemm', 'tensor_model_parallel_size', 'pipeline_model_parallel_size',
            'sequence_parallel', 'moe_ffn_hidden_size'
        }
        extra_keys = {
            'rope_scaling_type', 'rope_scaling_factor', 'rope_scaling_original_max_position_embeddings',
            'untie_embeddings_and_output_weights', 'no_gradient_accumulation_fusion', 'kv_channels', 'swiglu'
        }

        for key, value in mutated_flat_config.items():
            if key == 'disable_bias_linear':
                transformer_cfg['add_bias_linear'] = not bool(value)
                continue
            if key in transformer_keys:
                transformer_cfg[key] = value
                continue
            if key in extra_keys:
                extra_cfg[key] = value

        normalized_transformer_cfg = model_helpers.normalize_transformer_scalar_types(transformer_cfg)
        transformer_cfg.clear()
        transformer_cfg.update(normalized_transformer_cfg)
        normalized_extra_cfg = model_helpers.normalize_transformer_scalar_types(extra_cfg)
        extra_cfg.clear()
        extra_cfg.update(normalized_extra_cfg)

        if 'swiglu' in extra_cfg:
            transformer_cfg['gated_linear_unit'] = bool(extra_cfg['swiglu'])

        if 'num_moe_experts' in transformer_cfg:
            spec_cfg['num_experts'] = model_helpers.coerce_optional_int(transformer_cfg['num_moe_experts'])
            if self._safe_int(transformer_cfg.get('num_moe_experts', 0), 0) > 0:
                transformer_cfg['add_bias_linear'] = False

        return updated

    def mutate_model_config(self, model_config: Dict[str, Any], mutation_num: int = 3,
                            is_last_decoder: bool = False,
                            graph_hidden_size: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """对完整模型 YAML 做变异，并回写到整网会消费的 after 配置。"""
        base_config = model_helpers.extract_graph_transformer_config_from_yaml(model_config)
        mutated_config_dict = self.mutate_config_dict(
            base_config=base_config,
            mutation_num=mutation_num,
            is_last_decoder=is_last_decoder,
            graph_hidden_size=graph_hidden_size,
        )
        mutated_model_config = copy.deepcopy(model_config)
        section_name = self._find_transformer_section_name(mutated_model_config)
        mutated_model_config[section_name] = mutated_config_dict
        diff = {'created': {}, 'deleted': [], 'modified': {}}
        for key, value in mutated_config_dict.items():
            if key not in base_config:
                diff['created'][key] = value
            elif base_config.get(key) != value:
                diff['modified'][key] = {'from': base_config.get(key), 'to': value}
        return mutated_config_dict, mutated_model_config, diff

    def create_and_save_mutated_config(self, iteration: int, mutation_num: int = 3) -> Tuple[Dict[str, Any], str]:
        """
        创建并保存变异的完整配置

        Args:
            iteration: 迭代轮数
            mutation_num: 要变异的参数数量

        Returns:
            Tuple: (变异配置, 保存路径)
        """
        print(f"\n=== 创建第 {iteration} 轮变异配置 ===")

        # 深拷贝基础合并配置
        mutated_config = copy.deepcopy(self.combined_config)

        # 更新元数据
        mutated_config['metadata'].update({
            'iteration': iteration,
            'mutation_num': mutation_num,
            'mutation_time': datetime.now().isoformat()
        })

        # 变异图结构中的每个layer
        if 'graph_structure' in mutated_config and 'LayerConfig' in mutated_config['graph_structure']:
            mutated_layers = {}

            for layer_id, layer_config in mutated_config['graph_structure']['LayerConfig'].items():
                print(f"处理层 {layer_id}: {layer_config['name']}")

                # 深拷贝层配置
                mutated_layer = copy.deepcopy(layer_config)

                # 如果是decoder层，应用变异
                if 'Decoder' in layer_config['name']:
                    mutated_layer = self._mutate_layer_config(mutated_layer, layer_id, mutation_num)

                mutated_layers[layer_id] = mutated_layer

            mutated_config['graph_structure']['LayerConfig'] = mutated_layers

        # 变异基础配置
        # if 'base_config' in mutated_config:
        #     mutated_config['base_config'] = self._mutate_base_config(
        #         mutated_config['base_config'], mutation_num
        #     )

        # 保存变异配置
        filepath = self._save_mutated_config(mutated_config, iteration)

        return mutated_config, filepath

    def _mutate_layer_config(self, layer_config: Dict[str, Any], layer_id: str,
                             mutation_num: int) -> Dict[str, Any]:
        """变异单个层配置"""
        print(f"  对层 {layer_id} 应用变异 (变异数量: {mutation_num})")

        # 删除layer_limits和layer_nums字段
        if 'layer_limits' in layer_config:
            removed_limits = layer_config.pop('layer_limits')
            print(f"    移除 layer_limits: {removed_limits}")

        if 'layer_nums' in layer_config:
            removed_nums = layer_config.pop('layer_nums')
            print(f"    移除 layer_nums: {removed_nums}")

        # 在params中添加model_config路径
        if 'params' not in layer_config:
            layer_config['params'] = {}

        # 根据层类型选择合适的model_config文件
        layer_name = layer_config.get('name', '').lower()
        config_file_path = self._select_model_config_path(layer_name, layer_id, mutation_num)

        if config_file_path:
            layer_config['params']['model_config'] = config_file_path
            print(f"    添加 model_config 路径: {config_file_path}")

        return layer_config

    def _select_model_config_path(self, layer_name: str, layer_id: str, mutation_num: int) -> str:
        """
        根据层名称和变异数量选择model_config文件路径

        Args:
            layer_name: 层名称
            layer_id: 层ID
            mutation_num: 变异数量

        Returns:
            str: model_config文件路径
        """
        # 获取model_config目录下的所有配置文件
        model_config_dir = self.config_dir
        available_configs = []

        if os.path.exists(model_config_dir):
            for file in os.listdir(model_config_dir):
                if file.endswith('.yaml'):
                    available_configs.append(file)

        # 如果没有找到配置文件，返回默认路径
        if not available_configs:
            print(f"    警告：未找到model_config目录下的配置文件，使用默认路径")
            return "../model_config/qwen2.yaml"

        # 为decoder层选择配置文件（基于变异数量）
        if 'decoder' in layer_name.lower():
            if mutation_num >= 3:  # 高变异数量时随机选择
                rng = self._make_rng("model_config_path", layer_name, layer_id, mutation_num)
                selected_config = rng.choice(available_configs)
                print(f"    基于变异数量 {mutation_num} 随机选择配置: {selected_config}")
            else:
                # 低变异数量时选择特定配置
                preferred_configs = [f for f in available_configs if 'qwen' in f.lower()]
                if preferred_configs:
                    selected_config = preferred_configs[0]
                else:
                    selected_config = available_configs[0]
                print(f"    使用首选配置: {selected_config}")
        else:
            # 非decoder层使用默认配置
            default_configs = [f for f in available_configs if 'baichuan' in f.lower()]
            if default_configs:
                selected_config = default_configs[0]
            else:
                selected_config = available_configs[0]
            print(f"    非decoder层使用默认配置: {selected_config}")

        return os.path.join(model_config_dir, selected_config)

    def _save_mutated_config(self, mutated_config: Dict[str, Any], iteration: int) -> str:
        """
        保存变异配置到文件

        Args:
            mutated_config: 变异后的配置
            iteration: 迭代轮数

        Returns:
            str: 保存的文件路径
        """
        filename = f"mutated_config_iter_{iteration:03d}.yaml"
        filepath = os.path.join(self.output_dir, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(mutated_config, f)

            print(f"✓ 成功保存变异配置: {filepath}")
            return filepath

        except Exception as e:
            print(f"✗ 保存变异配置失败: {e}")
            raise


class MutatedDecoderNode:
    """变异的Decoder节点 - 包含一个变异的decoder配置"""

    def __init__(self, node_id: int, mutated_config: TransformerConfig,
                 is_last_decoder: bool = False, graph_hidden_size: int = None):
        """
        初始化变异Decoder节点

        Args:
            node_id: 节点ID
            mutated_config: 变异后的TransformerConfig
            is_last_decoder: 是否为最后一个decoder层
            graph_hidden_size: graph配置中的hidden_size
        """
        self.id = node_id
        self.config = mutated_config
        self.is_last_decoder = is_last_decoder
        self.graph_hidden_size = graph_hidden_size
        self.decoder = None

        # Node基本属性
        self.from_nodes = []
        self.to_nodes = []
        self.state = 'none'
        self.str_op = 'mutated_decoder'
        self.params = {}
        self.input_shape = []
        self.output_shape = []

    def create_decoder(self) -> TransformerBlock:
        """创建decoder实例，参考generate_graph.py的实现"""
        try:
            print(f"为节点 {self.id} 创建变异decoder...")

            # 创建layer spec（参考generate_graph.py）
            layer_spec = get_gpt_layer_local_spec(
                None,  # num_experts
                False,  # moe_grouped_gemm
                False,  # qk_layernorm
                False,  # multi_latent_attention
                False,  # fp8
                normalization="RMSNorm",
            )

            # 创建TransformerBlock（参考generate_graph.py）
            self.decoder = TransformerBlock(
                config=self.config,
                spec=layer_spec,
                pre_process=True,
                post_process=True,
                vp_stage=None,
            )

            # 移动到NPU而不是CUDA
            if torch.npu.is_available():
                self.decoder = self.decoder.npu()
            elif torch.cuda.is_available():
                self.decoder = self.decoder.cuda()

            print(f"✓ 成功创建变异decoder，hidden_size: {self.config.hidden_size}")
            return self.decoder

        except Exception as e:
            print(f"✗ 创建变异decoder失败: {e}")
            raise

    def forward_decoder(self, input_data: torch.Tensor, attention_mask: torch.Tensor,
                        rotary_pos_emb=None, inference_context=None) -> torch.Tensor:
        """执行decoder前向传播"""
        if self.decoder is None:
            raise ValueError("请先调用 create_decoder() 方法")

        # 检查输入维度是否匹配
        expected_hidden_size = self.config.hidden_size
        if input_data.shape[-1] != expected_hidden_size:
            print(f"  调整输入维度: {input_data.shape} -> (..., {expected_hidden_size})")
            seq_len, batch_size, _ = input_data.shape
            input_data = torch.randn(seq_len, batch_size, expected_hidden_size,
                                     device=input_data.device, dtype=input_data.dtype)

        # 执行前向传播（参考graph.py的forward方法）
        output = self.decoder(
            hidden_states=input_data,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            packed_seq_params=None,
            sequence_len_offset=None,
            **(None or {}),
        )

        print(f"变异decoder前向传播完成: {input_data.shape} -> {output.shape}")
        return output


class MutatedGraph:
    """变异图类 - 管理包含变异decoder节点的图，支持自动保存变异配置"""

    def __init__(self, structure_config_path: str = "assets/runtime/configs/structure_config.yaml",
                 template_config_path: str = "assets/runtime/configs/template_config.yaml",
                 output_dir: str = "mutated_configs"):
        """
        初始化变异图

        Args:
            structure_config_path: 结构配置文件路径
            template_config_path: 模板配置文件路径
            output_dir: 变异配置保存目录
        """
        self.structure_config_path = structure_config_path
        self.template_config_path = template_config_path
        self.output_dir = output_dir
        self.config_path = template_config_path  # 保持向后兼容

        # 使用合并配置的变异器
        self.mutator = ConfigMutator(structure_config_path, template_config_path, output_dir)
        self.mutated_nodes = {}  # 存储变异decoder节点
        self.base_graph = None
        self.current_iteration = 0  # 当前迭代轮数

    def create_mutated_decoder_node(self, node_id: int, mutation_num: int = 3,
                                    is_last_decoder: bool = False, graph_hidden_size: int = None) -> MutatedDecoderNode:
        """
        创建变异decoder节点

        Args:
            node_id: 节点ID
            mutation_num: 要变异的参数数量
            is_last_decoder: 是否为最后一个decoder层
            graph_hidden_size: graph配置中的hidden_size

        Returns:
            MutatedDecoderNode: 变异decoder节点
        """
        print(f"为节点 {node_id} 创建变异decoder节点...")
        if is_last_decoder:
            print(f"  注意：这是最后一个decoder层，hidden_size将固定为 {graph_hidden_size}")

        # 变异配置
        mutated_config = self.mutator.mutate_config_dict(
            base_config=None,  # 使用默认配置
            mutation_num=mutation_num,
            is_last_decoder=is_last_decoder,
            graph_hidden_size=graph_hidden_size
        )

        # 创建变异decoder节点
        mutated_node = MutatedDecoderNode(node_id, mutated_config, is_last_decoder, graph_hidden_size)
        self.mutated_nodes[node_id] = mutated_node

        return mutated_node

    def create_graph_with_mutated_decoder(self, node_id: int, nums: List[int],
                                          mutation_num: int = 3, save_config: bool = True) -> Graph:
        """
        创建包含变异decoder节点的图（参考generate_graph.py的方式）

        Args:
            node_id: 要设置为变异decoder的节点ID
            nums: 所有节点ID列表
            mutation_num: 要变异的参数数量
            save_config: 是否保存变异配置

        Returns:
            Graph: 包含变异decoder的图
        """
        # 增加迭代轮数
        self.current_iteration += 1

        # 如果需要保存配置，先保存合并的变异配置
        if save_config:
            mutated_config, config_filepath = self.mutator.create_and_save_mutated_config(
                self.current_iteration, mutation_num
            )
            print(f"✓ 已保存第 {self.current_iteration} 轮变异配置到: {config_filepath}")

        # 创建基础图（参考generate_graph.py）
        graph = Graph(config_path=self.config_path, nums=nums)

        # 获取graph的hidden_size
        graph_hidden_size = graph.config.hidden_size

        # 判断是否为最后一个decoder层
        is_last_decoder = (node_id == max(nums))

        # 创建变异decoder节点
        mutated_node = self.create_mutated_decoder_node(
            node_id, mutation_num, is_last_decoder, graph_hidden_size
        )

        # 将变异decoder节点集成到图中（参考generate_graph.py的方式）
        if node_id in graph.nodes:
            graph_node = graph.nodes[node_id]
            graph_node.str_op = 'mutated_decoder'
            graph_node.config = mutated_node.config
            graph_node.params['mutated_decoder_node'] = mutated_node

            # 创建decoder实例
            graph_node.block = mutated_node.create_decoder()

        self.base_graph = graph
        return graph

    def create_multiple_mutated_graphs(self, node_ids: List[int], nums: List[int],
                                       mutation_nums: List[int] = None,
                                       iterations: int = 5) -> List[Tuple[Graph, str]]:
        """
        创建多个变异图，每个都保存配置文件

        Args:
            node_ids: 要变异的节点ID列表
            nums: 所有节点ID列表
            mutation_nums: 变异参数数量列表，如果为None则使用默认值
            iterations: 迭代次数

        Returns:
            List[Tuple[Graph, str]]: [(图对象, 配置文件路径), ...]
        """
        if mutation_nums is None:
            mutation_nums = [3] * iterations
        elif len(mutation_nums) != iterations:
            # 如果mutation_nums长度不够，循环使用
            mutation_nums = (mutation_nums * ((iterations // len(mutation_nums)) + 1))[:iterations]

        results = []

        for i in range(iterations):
            print(f"\n{'=' * 60}")
            print(f"创建第 {i + 1}/{iterations} 个变异图")
            print(f"{'=' * 60}")

            # 为每次迭代随机选择一个节点进行变异
            rng = self.mutator._make_rng("multiple_mutated_graphs", i, iterations, len(node_ids))
            node_id = rng.choice(node_ids)
            mutation_num = mutation_nums[i]

            graph = self.create_graph_with_mutated_decoder(
                node_id=node_id,
                nums=nums,
                mutation_num=mutation_num,
                save_config=True
            )

            config_filename = f"mutated_config_iter_{self.current_iteration:03d}.yaml"
            config_filepath = os.path.join(self.output_dir, config_filename)

            results.append((graph, config_filepath))

            print(f"✓ 第 {i + 1} 个变异图创建完成")
            print(f"  变异节点: {node_id}")
            print(f"  变异参数数量: {mutation_num}")
            print(f"  配置文件: {config_filepath}")

        return results

    def test_mutated_decoder_forward(self, node_id: int) -> torch.Tensor:
        """测试变异decoder前向传播"""
        if node_id not in self.mutated_nodes:
            raise ValueError(f"节点 {node_id} 不是变异decoder节点")

        mutated_node = self.mutated_nodes[node_id]

        # 创建测试输入
        batch_size = 2
        seq_length = 16
        hidden_size = mutated_node.config.hidden_size

        input_data = torch.randn(seq_length, batch_size, hidden_size, device="cuda")
        attention_mask = torch.triu(
            torch.ones(1, 1, seq_length, seq_length, device="cuda"),
            diagonal=1
        ).bool()
        attention_mask = ~attention_mask

        print(f"测试输入维度: {input_data.shape}")
        print(f"Decoder期望hidden_size: {hidden_size}")

        # 执行前向传播
        return mutated_node.forward_decoder(input_data, attention_mask)


def demo_mutation_system():
    """演示合并变异系统的使用"""
    print("=== 合并变异系统演示 ===")

    try:
        # 创建合并变异图 - 直接使用两个配置文件
        mutated_graph = MutatedGraph(
            structure_config_path="assets/runtime/configs/structure_config.yaml",
            template_config_path="assets/runtime/configs/template_config.yaml",
            output_dir="demo_mutated_configs"
        )

        print(f"\n--- 单次变异演示 ---")
        # 创建包含变异decoder的图，自动保存配置
        nums = [0, 1, 2]
        graph = mutated_graph.create_graph_with_mutated_decoder(
            node_id=2,  # 最后一个节点作为变异decoder
            nums=nums,
            mutation_num=3,
            save_config=True  # 自动保存变异配置
        )

        print(f"✓ 成功创建包含变异decoder的图，节点数: {len(graph.nodes)}")

        print(f"\n--- 批量变异演示 ---")
        # 创建多个变异图，每个都保存配置
        node_ids = [1, 2]  # 可以变异的节点
        mutation_nums = [2, 3, 4, 5]  # 不同的变异参数数量

        graphs_and_configs = mutated_graph.create_multiple_mutated_graphs(
            node_ids=node_ids,
            nums=nums,
            mutation_nums=mutation_nums,
            iterations=4
        )

        print(f"\n✓ 成功创建 {len(graphs_and_configs)} 个变异图")

        # 显示所有保存的配置文件
        print(f"\n--- 保存的配置文件 ---")
        for i, (graph, config_path) in enumerate(graphs_and_configs):
            print(f"  第{i + 1}个: {config_path}")

        # 测试变异decoder前向传播
        print("\n--- 测试变异decoder前向传播 ---")
        try:
            # 使用最后一个图进行测试
            last_graph = graphs_and_configs[-1][0]
            if 2 in mutated_graph.mutated_nodes:
                output = mutated_graph.test_mutated_decoder_forward(2)
                print("✓ 变异decoder前向传播测试成功")
        except Exception as e:
            print(f"变异decoder前向传播测试失败: {e}")

        # 显示配置信息
        print("\n--- 最后一次变异的配置信息 ---")
        if 2 in mutated_graph.mutated_nodes:
            mutated_node = mutated_graph.mutated_nodes[2]
            print(f"节点2是否为最后decoder层: {mutated_node.is_last_decoder}")
            print(f"Graph hidden_size: {mutated_node.graph_hidden_size}")
            print(f"变异后的hidden_size: {mutated_node.config.hidden_size}")
            print(f"变异后的num_layers: {mutated_node.config.num_layers}")
            print(f"变异后的num_attention_heads: {mutated_node.config.num_attention_heads}")

        # 显示保存统计
        print(f"\n--- 保存统计 ---")
        print(f"总迭代轮数: {mutated_graph.current_iteration}")
        print(f"配置保存目录: {mutated_graph.output_dir}")

        # 检查保存的文件
        if os.path.exists(mutated_graph.output_dir):
            saved_files = [f for f in os.listdir(mutated_graph.output_dir) if f.endswith('.yaml')]
            print(f"已保存的配置文件数量: {len(saved_files)}")
            for filename in sorted(saved_files):
                print(f"  - {filename}")

        return True

    except Exception as e:
        print(f"✗ 演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_config_only():
    """仅演示配置合并和保存功能"""
    print("=== 配置合并和保存演示 ===")

    try:
        # 创建配置变异器
        mutator = ConfigMutator(
            structure_config_path="assets/runtime/configs/structure_config.yaml",
            template_config_path="assets/runtime/configs/template_config.yaml",
            output_dir="config_only_demo"
        )

        # 创建多个变异配置
        iterations = [1, 2, 3, 4, 5]
        mutation_nums = [2, 3, 4, 5, 6]

        saved_configs = []

        for iteration, mutation_num in zip(iterations, mutation_nums):
            mutated_config, filepath = mutator.create_and_save_mutated_config(
                iteration=iteration,
                mutation_num=mutation_num
            )
            saved_configs.append((iteration, mutation_num, filepath))

            # 显示变异结果预览
            print(f"\n第 {iteration} 轮变异结果预览 (变异参数数量: {mutation_num}):")
            if 'base_config' in mutated_config and 'config' in mutated_config['base_config']:
                config_section = mutated_config['base_config']['config']
                print(f"  hidden_size: {config_section.get('hidden_size', 'N/A')}")
                print(f"  num_attention_heads: {config_section.get('num_attention_heads', 'N/A')}")

            print(f"  vocab_size: {mutated_config['base_config'].get('vocab_size', 'N/A')}")

        print(f"\n✓ 配置演示完成！共创建了 {len(saved_configs)} 个变异配置文件")

        return True

    except Exception as e:
        print(f"✗ 配置演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("选择演示模式:")
    print("1. 完整变异系统演示 (包含图创建和前向传播)")
    print("2. 仅配置合并和保存演示")
    print("3. 两个演示都运行")

    try:
        choice = input("请输入选择 (1/2/3，默认为1): ").strip()
        if not choice:
            choice = "1"

        success = True

        if choice in ["1", "3"]:
            print("\n" + "=" * 80)
            print("运行完整变异系统演示")
            print("=" * 80)
            success &= demo_mutation_system()

        if choice in ["2", "3"]:
            print("\n" + "=" * 80)
            print("运行配置合并和保存演示")
            print("=" * 80)
            success &= demo_config_only()

        if choice not in ["1", "2", "3"]:
            print("无效选择，运行默认演示...")
            success = demo_mutation_system()

        if success:
            print(f"\n{'=' * 80}")
            print("🎉 所有演示都成功完成!")
            print("📁 请查看生成的目录:")
            print("   - demo_mutated_configs/ (完整演示的配置文件)")
            print("   - config_only_demo/ (仅配置演示的文件)")
            print(f"{'=' * 80}")
        else:
            print(f"\n{'=' * 80}")
            print("❌ 部分演示失败，请检查错误信息")
            print(f"{'=' * 80}")

    except (KeyboardInterrupt, EOFError):
        print("\n\n用户中断，运行默认演示...")
        demo_mutation_system()
