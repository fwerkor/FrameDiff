#!/usr/bin/env python3
import mindspeed.megatron_adaptor as ma
import sys
from argparse import ArgumentParser
import torch.distributed as dist
from typing import Dict, Any
from mindspeed.arguments import process_args
import numpy as np
import torch
import torch_npu
from megatron.training.arguments import parse_args, validate_args
from megatron.training.global_vars import set_global_variables
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training import get_args
from megatron.training.initialize import _initialize_distributed,_init_autoresume,_set_random_seed,_compile_dependencies,_initialize_tp_communicators


def add_extra_args(parser):
    """Add custom arguments for mutation system"""
    # Add arguments that this script uses
    parser.add_argument("-c", "--configs", type=str, help="The path to the configs dir")
    parser.add_argument("-n", "--node-num", type=int, default = 1, help="nodes num")
    parser.add_argument("-r", "--rounds", type=int, default = 10, help="mutating rounds")
    parser.add_argument("--mutnm", type=int, default = 2, help="mutating num")
    parser.add_argument("-m", "--module", type=str, help="The targeted single module")
    parser.add_argument("--sub", type=str, help="The list of submodule num")
    parser.add_argument("--load-path", type=str, help="The path of the graph config to load")
    parser.add_argument("--args_path", type=str, help="The path of the mutation arguments yaml")
    return parser


def finish_mpu_init():
    args = get_args()
    _initialize_distributed(None,None)
    if args.rank == 0:
        print("> setting random seeds to {} ...".format(args.seed))
    _set_random_seed(args.seed, args.data_parallel_random_init)

args = parse_args(extra_args_provider=add_extra_args)
validate_args(args, {})


set_global_variables(args,False)
args = get_args()
finish_mpu_init()
_init_autoresume()
_compile_dependencies()

if args.tp_comm_overlap:
    _initialize_tp_communicators()

from megatron.training import get_args

import torch
import json
import random
import os
from ruamel.yaml import YAML
import copy
import time
import sys

from utils.runtime import common_utils, model_helpers
from utils.runtime.logger import Logger
from utils.runtime.core.withnum_mutation_system import ConfigMutator
from utils.runtime.core.graph import Graph, Node
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding


yaml = YAML()
TEMPLATE_CONFIG_PATH = model_helpers.resolve_repo_path("assets/runtime/configs/template_config.yaml")
STRUCTURE_CONFIG_PATH = model_helpers.resolve_repo_path("assets/runtime/configs/structure_config.yaml")

config_dir = ""
res_dir = ""
mutating_rate = 0.3

# params in one iteration
iteration = 0
# allow external mutate round to align numbering across reruns
_mutate_round = os.getenv("MUTATE_ROUND", "").strip()
if _mutate_round:
    try:
        iteration = int(_mutate_round) - 1
    except ValueError:
        raise ValueError(f"env MUTATE_ROUND must be int, got: {_mutate_round}")
# store this mutation iteration info
mutating_record = {}
record_file_name = ""
no_mutating = False


# 命令行参数（有默认值）
module = args.module if args.module else "../model_config/qwen2.yaml"
node_num = args.node_num if args.node_num else 1
config_dir = args.configs if args.configs else "model_config"
rounds = args.rounds if args.rounds else 100
mutating_num = args.mutnm if args.mutnm else 2

mutation_args_path = args.args_path

def seed_all(seed=43):
    """固定随机种子。"""
    model_helpers.seed_all(seed, np_module=np, torch_module=torch, torch_npu_module=torch_npu)
    
def load_random_model_configs(count: int = 2):
    return model_helpers.load_random_model_configs(yaml, module, config_dir, count=count)


def extract_transformer_config_from_yaml(yaml_config: dict) -> dict:
    return model_helpers.extract_graph_transformer_config_from_yaml(yaml_config)


def create_mutated_graph_with_random_configs(configs, selected_files, num_decoder_nodes: int = 2,
                                           mutated_node_ids: list = None,
                                           mutation_num: int = 3):
    """
    使用随机选择的模型配置创建包含变异decoder的图
    
    Args:
        num_decoder_nodes: decoder节点的数量
        mutated_node_ids: 要变异的节点ID列表，如果为None则变异所有decoder节点
        mutation_rate: 变异概率
        
    Returns:
        Tuple[Graph, Dict, List]: 图对象、变异节点配置、选择的配置文件
    """    
    # args = get_args()
    # config = core_transformer_config_from_args(args)


    # new configs after this iteration
    new_configs = []
    
    # store this mutation iteration info
    global mutating_record
    mutating_record = {}
    
    # 动态创建图配置
    network_nodes = []
    
    # add the first embedding node
    network_nodes.append({
        "id": 0,
        "name": "embedding",
        "from": [],
        "to": [1] if num_decoder_nodes > 0 else [],
        "params": {},
        "state": "src",
        "layer_limits": ["embedding"]
    })
    
    # add decoder nodes
    for i in range(num_decoder_nodes):
        node_id = i + 1
        from_nodes = [node_id - 1]  # 从前一个节点接收输入
        to_nodes = [node_id + 1] if i < num_decoder_nodes - 1 else []  # 连接到下一个节点
        state = "des" if i == num_decoder_nodes - 1 else "none"  # 最后一个节点标记为destination
        
        network_nodes.append({
            "id": node_id,
            "name": "Qwen2DecoderLayer",
            "from": from_nodes,
            "to": to_nodes,
            "params": {},
            "state": state,
            "layer_limits": ["Qwen2DecoderLayer"]
        })
    
    # 创建完整的图配置
    graph_config = {"network": network_nodes}
    
    print(f"\n创建包含 {num_decoder_nodes} 个decoder节点的图")
    print(f"节点连接: embedding(0) -> " + " -> ".join([f"decoder{i}({i+1})" for i in range(num_decoder_nodes)]))
    
    # 保存临时JSON文件
    with open("temp_graph.json", "w") as f:
        json.dump(graph_config, f)
    
    # 读取JSON配置
    with open("temp_graph.json", "r") as f:
        output_data = json.load(f)
        g = output_data['network']
    
    # 创建基础图
    graph = Graph(
        config_path=TEMPLATE_CONFIG_PATH,
        nums=[i for i in range(len(g))]
    )
    # print(graph.total_config)
    
    # 创建变异器 - 使用新的合并配置系统
    mutator = ConfigMutator(
        structure_config_path=STRUCTURE_CONFIG_PATH,
        template_config_path=TEMPLATE_CONFIG_PATH,
        output_dir=res_dir,
        config_dir=config_dir,
        mutation_args_path=mutation_args_path
    )
    mutated_nodes = {}
    
    # 为每次forward迭代保存变异配置
    iteration_id = getattr(create_mutated_graph_with_random_configs, '_iteration_counter', 0) + 1
    setattr(create_mutated_graph_with_random_configs, '_iteration_counter', iteration_id)
    
    # 如果没有指定要变异的节点，默认变异所有decoder节点
    if mutated_node_ids is None:
        mutated_node_ids = list(range(1, num_decoder_nodes + 1))
    
    print(f"将对节点 {mutated_node_ids} 进行变异")
    
    # 处理每个节点
    for node_ in g:
        node_id = node_['id']
        mutating_record[node_id] = {"mutated": False, "before": {}, "after": {}}
        config_index = (node_id - 1) % len(configs)
        mutating_record[node_id]["before"] = configs[config_index]
        config_after = configs[config_index]
        
        # extract TransformerConfig
        if node_id > 0:  # decoder节点
            # 循环使用配置文件
            base_config = extract_transformer_config_from_yaml(configs[config_index]) # before
            # print("***************************",base_config)
            source_file = selected_files[config_index]
        else:
            # embedding节点使用默认配置
            base_config = {
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
            source_file = "default"
        
        # create TransformerConfig object
        init_config = TransformerConfig(**base_config)
       
        # create Node
        node = Node(config=init_config, index=node_id)
        node.from_nodes = node_['from']
        node.to_nodes = node_['to']
        node.str_op = node_['name']
        node.params = node_['params']
        node.state = node_['state']
        
        # 处理layer_limits
        if 'layer_limits' in node_:
            node.layer_limits = node_['layer_limits']
            if node.layer_limits[0] != "none":
                import random
                random_num = random.randrange(len(node.layer_limits))
                node.str_op = node.layer_limits[random_num]
                
                # 检查是否需要变异此节点
                should_mutate = (node_id in mutated_node_ids) and (not no_mutating)
                is_last_decoder = (node_id == num_decoder_nodes)  # 最后一个decoder节点
                
                if "decoderlayer" in node.str_op.lower() and should_mutate:
                    mutating_record[node_id]["mutated"] = True
                    print(f"\n对节点 {node_id} 应用变异 (基于 {source_file})...")
                    
                    # 保存当前迭代的变异配置到文件（仅在第一个变异节点时保存）
                    if node_id == min(mutated_node_ids):
                        print(f"\n--- 保存第 {iteration_id} 轮变异配置 ---")
                        try:
                            saved_config, config_filepath = mutator.create_and_save_mutated_config(
                                iteration=iteration_id, 
                                mutation_num=mutation_num
                            )
                            print(f"已保存变异配置到: {config_filepath}")
                        except Exception as e:
                            print(f"保存变异配置失败: {e}")
                    
                    # mutate config
                    mutated_config_dict = mutator.mutate_config_dict(
                        base_config=base_config,
                        mutation_num=mutation_num,
                        is_last_decoder=is_last_decoder,
                        graph_hidden_size=graph.config.hidden_size
                    )
                    # record info: diff, after
                    diff = common_utils.compare_dicts(before=base_config, after=mutated_config_dict)
                    mutating_record[node_id]["diff"] = diff
                    common_utils.print_diff(diff)
                    mutated_config = TransformerConfig(**mutated_config_dict)
                    all_config_copy = copy.deepcopy(configs[config_index])
                    all_config_copy["TransformerConfig"] = mutated_config_dict
                    mutating_record[node_id]["after"] = all_config_copy
                    config_after = all_config_copy
                    
                    node.config = mutated_config
                    node.str_op = 'mutated_decoder'
                    mutated_nodes[node_id] = {
                        'config': mutated_config,
                        'source_file': source_file,
                        'original_config': base_config
                    }
                    
                elif node.str_op == "Qwen2DecoderLayer":
                    # 使用原始配置
                    print(f"\n节点 {node_id} 使用原始配置 (基于 {source_file})")
                    node.config = init_config
                
                # 创建TransformerBlock（参考generate_graph.py）
                if ('decoderlayer' in node.str_op.lower() or 'mutated_decoder' in node.str_op.lower()):
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
                
                elif  "embedding" in node.str_op.lower():
                    # 创建embedding层（参考generate_graph.py）
                    node.config = graph.total_config
                    node.block = LanguageModelEmbedding(
                        config=node.config['config'],
                        vocab_size=node.config['vocab_size'],
                        max_sequence_length=node.config['max_sequence_length'],
                        position_embedding_type=node.config['position_embedding_type'],
                    )
        
        # 设置节点属性
        node.in_degree = len(node.from_nodes)
        node.out_degree = len(node.to_nodes)
        if node.state == 'des':
            node.out_degree = 1
        
        graph.nodes[node.id] = node
        
        if node_id > 0:
            new_configs.append(config_after)
    
    # 修改：在所有节点处理完成后，设置完整的mutated_nodes到graph中
    graph.set_mutated_nodes(mutated_nodes)

    return graph, mutated_nodes, selected_files, new_configs


def demo_forward_with_random_configs(configs, selected_files, num_decoder_nodes: int = 2):
    global mutating_record
    """演示使用随机配置和变异的图的forward过程"""
    try:
        # 创建包含随机配置变异decoder的图
        all_decoder_ids = list(range(1, num_decoder_nodes + 1))
        graph, mutated_nodes, selected_files, new_configs = create_mutated_graph_with_random_configs(
            configs=configs,
            selected_files=selected_files,
            num_decoder_nodes=num_decoder_nodes,
            mutated_node_ids=all_decoder_ids,  # 变异所有decoder节点
            mutation_num=mutating_num
        )
        
        print(f"\n图创建完成，节点数: {len(graph.nodes)}")
        print(f"变异节点: {list(mutated_nodes.keys())}")
        print(f"使用的配置文件: {selected_files}")
        from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
        from megatron.core.distributed import DistributedDataParallelConfig
        from megatron.core.distributed import DistributedDataParallel as DDP
        
        # ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
        # model_graph = [graph]
        # model = [DDP(model_graph[0].total_config["config"], ddp_config, model_chunk, disable_bucketing=(model_chunk_idx > 0))
        #  for (model_chunk_idx, model_chunk) in enumerate(model_graph)]
        # optimizer_config = OptimizerConfig(
        #     optimizer='adam',
        #     lr=1e-4,
        #     weight_decay=0.01,
        # )
        # optimizer = get_megatron_optimizer(
        #     config=optimizer_config,
        #     model_chunks=model
        # )           
        # 执行forward过程 - 修改：直接调用graph.forward()方法
        print("\n--- 执行Forward过程 ---")
        
        # 直接调用graph的forward方法（mutated_nodes已在初始化时设置）
        # final_output = model_graph[0].forward(debug=True)
        final_output = graph.forward(debug=True)
        print(f"\nForward过程完成！最终输出形状: {final_output.shape}")
        mutating_record["success"]=True
        loss = final_output.norm()
        print("norm计算结果",loss)
        
        # clear temp files
        if os.path.exists("temp_graph.json"):
            os.remove("temp_graph.json")
        
        return selected_files, new_configs,loss
        
    except Exception as e:
        print(f"✗ Forward演示失败: {e}")
        
        # record err info to file and console
        import traceback
        traceback.print_exc()
        error_stack = traceback.format_exc()
        print(error_stack)
        mutating_record["success"]=False
        mutating_record["err_stack"]=error_stack
        
        return None

def print_npu_memory(tag=""):
    try:
        # 获取NPU当前分配的显存
        allocated = torch.npu.memory_allocated() / 1024 / 1024  # 单位 MB

        print(f"[NPU显存][{tag}] 当前分配: {allocated:.2f} MB")
    except Exception as e:
        print(f"[NPU显存][{tag}] 记录失败: {e}")

def generate_layer_config(node_count: int) -> Dict[str, Any]:
    """
    生成指定节点数量的层配置

    Args:
        node_count: 总节点数量

    Returns:
        包含所有节点配置的字典
    """
    if node_count < 2:
        raise ValueError("节点数量必须至少为2")

    config = {"LayerConfig": {}}

    # 节点1: Embedding (起始节点)
    config["LayerConfig"][1] = {
        "name": "Embedding",
        "params": {},
        "state": "src",
        "layer_limits": ["embedding"],
        "to": [2],
        "from": [],
        "layer_nums": 1
    }

    # 节点2: Decoderlayer1
    config["LayerConfig"][2] = {
        "name": "Decoderlayer1",
        "params": {},
        "state": "none",
        "layer_limits": ["Qwen2DecoderLayer"],
        "to": [],
        "from": [1],
        "layer_nums": 1
    }

    # 生成节点3到n的配置
    for i in range(3, node_count + 1):
        config["LayerConfig"][i] = {
            "name": f"Decoderlayer{i - 1}",
            "params": {},
            "state": "none",
            "layer_limits": ["Qwen2DecoderLayer"],
            "to": [],
            "from": [],
            "layer_nums": 1
        }

    # 修正所有节点的to和from
    _fix_connections(config, node_count)

    return config


def _fix_connections(config: Dict[str, Any], node_count: int) -> None:
    """
    修正所有节点的连接关系

    Args:
        config: 配置字典
        node_count: 总节点数量
    """
    # 修正to字段
    for i in range(1, node_count):
        current_node = i
        next_node = i + 1
        config["LayerConfig"][current_node]["to"] = [next_node]

    # 最后一个节点的to为空列表
    config["LayerConfig"][node_count]["to"] = []

    # 修正from字段
    for i in range(2, node_count + 1):
        current_node = i
        prev_node = i - 1
        config["LayerConfig"][current_node]["from"] = [prev_node]


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    保存配置到YAML文件

    Args:
        config: 配置字典
        output_path: 输出文件路径
    """
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.preserve_quotes = True

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f)
        
        
        
if __name__ == "__main__":
    # parse arguments
    # module = "../model_config/baichuan.yaml" 
    # node_num = 1
    # config_dir = "model_config"
    # rounds = 2
    # mutating_num = 2
    
#     pipeline_dtype = torch.bfloat16
#     tensor_model_parallel_size = args.tensor_model_parallel_size
#     pipeline_model_parallel_size = args.pipeline_model_parallel_size

#     with open('assets/runtime/configs/template_config.yaml', 'r') as file:
#         data = yaml.load(file)  # 解析YAML内容为字典
#         data['config']['tensor_model_parallel_size'] = tensor_model_parallel_size
#         data['config']['pipeline_model_parallel_size'] = pipeline_model_parallel_size
#         data['config']['pipeline_dtype'] = "torch.bfloat16"
#     print(data)
        
#     with open('assets/runtime/configs/template_config.yaml', 'w') as file:
#         yaml.dump(data, file)
    
    # if args.module:
    #     node_num = 1
    # else:
    #     node_num = args.node_num
    # rounds = args.round
    # mutating_num = args.mutnm
    # config_dir = args.configs
    # seed_all()
    common_utils.print_seperate_line()
    print(f"开始{rounds}次语言模型整网配置变异（decoder层数={node_num}）")
    common_utils.print_seperate_line()
    
    # prepare dir
    
    # if args.module:
    #     filename = os.path.basename(args.module).split('.')[0]
    #     res_dir = os.path.join("./res", f"{common_utils.gen_timestamp_h()}_{filename}")
    # else:
    #     res_dir = os.path.join("./res", f"{common_utils.gen_timestamp_h()}_random{node_num}nodes")
    if module:
        filename = os.path.basename(module).split('.')[0]
        res_dir = os.path.join("./res", f"{filename}")
    else:
        res_dir = os.path.join("./res", f"random{node_num}nodes")
    successes = 0
    graph = Graph(
        config_path=TEMPLATE_CONFIG_PATH,
        nums=[int(i) for i in range(5)]
    )

    node_count = node_num + 1
    print(f"正在生成 {node_count} 个节点的配置文件...")
    config = generate_layer_config(node_count)
    output_path = STRUCTURE_CONFIG_PATH
    save_config(config, output_path)
    print(f"配置文件已成功生成: {output_path}")
    os.makedirs(res_dir, exist_ok=True)
    log_file_path = os.path.join(res_dir, "mutate_log.txt")
    sys.stdout = Logger(log_file_path)
    # select configs to be mutated, max = 12
    configs, selected_files = load_random_model_configs(count=min(node_num, 12))
    # mutate and forward
    
    for trial in range(rounds):

        iteration = iteration + 1
        print(f"\n当前第 {iteration}/{rounds} 次迭代")
        # run forward
        forward_res = demo_forward_with_random_configs(configs, selected_files, num_decoder_nodes=node_num)
        
        if forward_res:
            selected_files, configs, loss = forward_res
            record_file_path = os.path.join(res_dir, f"mutating-{iteration}.json")
        else:
            record_file_path = os.path.join(res_dir, f"mutating-{iteration}-err.json")
        
        common_utils.save_dict_into_json(mutating_record, record_file_path)
        print(f"\n变异信息存储已存储到{record_file_path}")
            
        yaml_file_name = f"mutated_config_iter_{iteration:03d}.yaml" 
        yaml_file_path = os.path.join(res_dir, yaml_file_name)
        success = 0
        json_file_path = os.path.join(res_dir, f"mutating-{iteration}.json")        


    # file output summary
    common_utils.print_seperate_line()
    if os.path.exists(res_dir):
        config_files = [f for f in os.listdir(res_dir) if f.endswith('.json')]
        print(f"\n保存的变异配置文件总结:")
        print(f"   目录: {res_dir}")
    common_utils.print_seperate_line()
