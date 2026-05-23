# RQ3 可变异参数列表

范围来源：`ICSE2027/experiment/fullnet/assets/runtime/configs/mutation_schema.yaml`。模型对应关系按当前整网实验的 12 个模型配置统计：`baichuan2`、`chatglm3`、`codellama`、`deepseekv3`、`glm4`、`grok1`、`llama2`、`mixtral`、`pangu`、`qwen2`、`qwen3`、`yi`。其中“预留”表示 schema 中已有可变异范围，但当前 baseline YAML 未显式启用，实验时需要在模型适配层补齐或确认框架支持。

## 通用/注意力与数值相关参数

| 参数 | 可变异范围 | 对应模型 | 备注 |
| --- | --- | --- | --- |
| `layernorm_epsilon` | `{1.0e-5, 1.0e-6, 1.0e-7}`；脚本侧约束为 `{1.0e-5, 1.0e-6}` | 全部模型 | 归一化数值稳定性参数。 |
| `rotary_base` | 模型侧 `{100000.0, 5000000.0}`；脚本侧 `{10000, 1000000}` | `codellama`、`grok1`、`mixtral`、`qwen2`、`qwen3`、`yi`；`chatglm3`、`glm4` 使用 RoPE 但当前配置未显式给出该字段 | 仅对 RoPE 位置编码模型有效。 |
| `rotary_percent` | `{0.0, 1.0}` | `chatglm3`、`glm4` | GLM 系模型当前显式配置为 0.5；schema 中用于 0/1 两端消融。 |
| `embedding_multiplier_scale` | `{0.0, 78.38}` | `grok1` | Grok-1 类缩放参数；当前 baseline YAML 未显式启用。 |
| `output_multiplier_scale` | `{0.0, 0.57}` | `grok1` | Grok-1 类缩放参数；当前 baseline YAML 未显式启用。 |
| `use_flash_attn` | `{true, false}` | `chatglm3`、`codellama`、`glm4`、`grok1`、`llama2`、`mixtral`、`qwen2`、`qwen3`、`yi` | Flash Attention 开关。 |
| `attention_softmax_in_fp32` | `{true, false}` | `baichuan2`、`deepseekv3`、`glm4`、`pangu`、`yi` | 注意力 softmax 精度控制。 |
| `no_masked_softmax_fusion` | `{true, false}` | `baichuan2`、`deepseekv3`、`glm4`、`pangu`、`yi` | 与配置里的 `masked_softmax_fusion` 语义相反；文档中按脚本参数名记录。 |
| `drop_path_rate` | `{0.0, 0.3}` | 预留，当前 baseline YAML 未显式启用 | 随机深度/DropPath 消融；需确认对应模型实现是否读取该字段。 |

## MoE 相关参数

| 参数 | 可变异范围 | 对应模型 | 备注 |
| --- | --- | --- | --- |
| `moe_router_load_balancing_type` | `{aux_loss, group_limited_greedy}` | `deepseekv3`、`grok1`、`mixtral` | MoE 路由负载均衡策略。 |
| `moe_aux_loss_coeff` | `{0.003, 0.01}` | `deepseekv3`、`grok1`、`mixtral` | MoE 辅助损失系数。 |
| `moe_router_pre_softmax` | `{true, false}` | `deepseekv3` | DeepSeekV3 当前显式启用；其他 MoE 模型未显式配置。 |
| `moe_router_topk` | `{1, 2, 4, 8}` | `deepseekv3`、`grok1`、`mixtral` | 需满足 `topk <= num_moe_experts`。 |
| `topk_group` | `{3, 6}` | `deepseekv3` | DeepSeekV3/group-limited routing 相关；当前 baseline YAML 未显式启用。 |
| `moe_grouped_gemm` | `{true, false}` | `deepseekv3`、`grok1`、`mixtral`；其他模型的 layer spec 中也以 `false` 占位 | 只有 MoE 模型上的变异结果有实际含义。 |
| `moe_permutation_async_comm` | `{true, false}` | `deepseekv3`、`grok1`、`mixtral` | MoE token permutation 通信优化；当前 baseline YAML 未显式启用。 |
| `use_fused_moe_token_permute_and_unpermute` | `{true, false}` | `deepseekv3`、`grok1`、`mixtral` | MoE token permute/unpermute 融合实现开关；当前 baseline YAML 未显式启用。 |
| `moe_device_level_aux_loss_coeff` | `{0.05, 0.03}` | `deepseekv3` | DeepSeekV3 类 device-level aux loss；当前 baseline YAML 未显式启用。 |
| `moe_comm_aux_loss_coeff` | `{0.02, 0.01}` | `deepseekv3` | DeepSeekV3 类通信辅助损失；当前 baseline YAML 未显式启用。 |
| `routed_scaling_factor` | `{16.0, 8.0}` | `deepseekv3` | DeepSeekV3 类 routed expert 缩放；当前 baseline YAML 未显式启用。 |
| `seq_aux` | `{true, false}` | `deepseekv3` | DeepSeekV3 类序列级辅助损失开关；当前 baseline YAML 未显式启用。 |
| `input_jitter` | `{true, false}` | `deepseekv3` | MoE/router 输入扰动开关；当前 baseline YAML 未显式启用。 |

## 可以变异但不保证 loss 差异的参数

| 参数 | 可变异范围 | 对应模型 | 备注 |
| --- | --- | --- | --- |
| `weight_decay` | `{0.01, 0.001, 0.0001, 0.0}` | 全部模型 | 优化器超参数；单步/短跑时不一定显著影响 loss。 |
| `sequence_parallel` | `{true, false}` | `baichuan2`、`deepseekv3`、`glm4`、`pangu`、`yi`；脚本侧也支持全部模型传参 | 并行策略参数；单卡或固定 world size 下可能不改变 loss。 |
| `use_cp_send_recv_overlap` | `{true, false}` | 预留，当前 baseline YAML 未显式启用 | Context Parallel 通信重叠开关；不保证 loss 差异。 |
