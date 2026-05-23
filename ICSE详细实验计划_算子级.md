# ICSE 详细实验计划 —— 算子级

> 本文档从源码实现角度，规定算子级实验（RQ1 差分测试 + RQ2 蜕变测试）的全部功能、接口、文件结构与代码要求。  
> 配套文档：`ICSE实验计划.md`、`ICSE论文提纲.md`、`ICSE总结稿.md`

---

## 一、实验目标

### 1.1 RQ1：误差的产生（差分测试）

在 PTA 和 MSA 双后端上，使用**完全相同的输入张量和权重**，分别执行同一算子，对比输出差异。差异张量的**最大值 / 平均值**表征误差的产生。

### 1.2 RQ2：误差的传播（蜕变测试）

在**单一后端**（MSA 或 PTA）内，对算子输入添加**微小扰动**（uniform fixed perturbation，每个元素加固定值 sigma，sigma ∈ [1e-7, 1e-6, 1e-5, 1e-4]），对比扰动前后的输出变化。差异张量的**最大值 / 平均值**表征误差的传播。

### 1.3 数据策略

算子级输入使用**随机生成的张量**，由 `TensorManager` 按确定性种子生成。不依赖真实数据，保证可复现。

**关键约束**：PTA 和 MSA 两端必须使用完全相同的随机输入张量（共享同一 seed 和生成参数）。

---

## 二、算子清单与 API 映射（47 个，无遗漏无重复）

以下算子覆盖大语言模型中**所有常见算子类型**，不局限于 11 个模型中出现的算子。同一 PyTorch API 只出现一次。

| 序号 | 算子名称 | PTA 真实接口 | MSA 真实接口 | 参数对齐说明 | 使用模型 |
|------|----------|-------------|-------------|-------------|----------|
| 1 | Embedding | `torch.nn.Embedding(num_embeddings, embedding_dim)` | `mindspore.nn.Embedding(vocab_size, embedding_size)` | `num_embeddings` ↔ `vocab_size`；`embedding_dim` ↔ `embedding_size` | 11 个 |
| 2 | LayerNorm | `torch.nn.LayerNorm(normalized_shape, eps=1e-5)` | `mindspore.nn.LayerNorm(normalized_shape, epsilon=1e-5)` | `eps` ↔ `epsilon` | PanGu |
| 3 | RMSNorm | Megatron-Core `RMSNorm(dim, eps)` | MindFormers `RMSNorm(dim, eps)` | 两端均使用框架内置实现，需确保参数一致 | 10 个 |
| 4 | Linear | `torch.nn.Linear(in_features, out_features, bias=True)` | `mindspore.nn.Dense(in_channels, out_channels, has_bias=True)` | `in_features` ↔ `in_channels`；`out_features` ↔ `out_channels`；`bias` ↔ `has_bias` | 11 个 |
| 5 | MatMul | `torch.matmul(input, other)` | `mindspore.ops.matmul(input, other)` | 参数完全一致 | 11 个 |
| 6 | ScaledDotProductAttention | `F.scaled_dot_product_attention(q, k, v, attn_mask=None)` | `ops.BatchMatMul` + `ops.Softmax` + `ops.masked_fill` 组合 | MSA 无原生等价物，需组合实现；scale 因子和 mask 填充值需一致 | 11 个 |
| 7 | FlashAttention | `flash_attn_func(q, k, v, causal=True)` | `flash_attn_func(q, k, v, causal=True)` | 两端调用同一第三方库 | 部分模型 |
| 8 | Softmax | `F.softmax(input, dim=-1)` | `mindspore.ops.softmax(input, axis=-1)` | `dim` ↔ `axis` | 11 个 |
| 9 | GELU | `torch.nn.GELU(approximate='tanh')` | `mindspore.nn.GELU(approximate=True)` | `approximate='tanh'` ↔ `approximate=True`（均为快速近似） | PanGu |
| 10 | SiLU | `torch.nn.SiLU()` | `mindspore.nn.SiLU()` | 无额外参数 | SwiGLU 模型 |
| 11 | SwiGLU | Megatron `SwiGLU(hidden, ffn)` | MindFormers `SwiGLU(hidden, ffn)` | gate/up/down 三个 Linear/Dense 按条目 4 对齐 | 10 个 |
| 12 | RoPE | Megatron `RotaryEmbedding(dim, max_len, base)` | MindFormers `RotaryEmbedding(dim, max_len, base)` | 两端均使用框架内置实现，参数一致 | 9 个 |
| 13 | ALiBi | 自定义偏置矩阵 + `torch.add(input, bias)` | 自定义偏置矩阵 + `mindspore.ops.add(input, bias)` | 两端均无原生 API，自定义实现 | Baichuan2 |
| 14 | Dropout | `torch.nn.Dropout(p=0.1)` | `mindspore.nn.Dropout(keep_prob=0.9)` | `p`（丢弃概率）↔ `keep_prob`（保留概率），语义相反：`keep_prob = 1.0 - p` | 11 个 |
| 15 | Add | `torch.add(input, other)` | `mindspore.ops.add(input, other)` | 参数完全一致 | 11 个 |
| 16 | Mul | `torch.mul(input, other)` | `mindspore.ops.mul(input, other)` | 参数完全一致 | 11 个 |
| 17 | CrossEntropy | `F.cross_entropy(input, target, reduction='mean')` | `mindspore.ops.cross_entropy(input, target, reduction='mean')` | 参数完全一致 | 11 个 |
| 18 | TopKGating | Megatron MoE Router | MindFormers MoE Router | 两端均使用框架内置 MoE 路由实现 | DeepSeekV3, Mixtral, Grok1 |
| 19 | MLA_Q_Projection | DeepSeekV3 自定义 `q_lora` projection | DeepSeekV3 MSA 自定义 `q_lora` projection | 自定义实现，`q_lora_rank` 维度对齐 | DeepSeekV3（生产配置） |
| 20 | MLA_KV_Projection | DeepSeekV3 自定义 `kv_lora` projection | DeepSeekV3 MSA 自定义 `kv_lora` projection | 自定义实现，`kv_lora_rank` 维度对齐 | DeepSeekV3（生产配置） |
| 21 | Transpose | `torch.transpose(input, -2, -1)` | `mindspore.ops.swapaxes(input, -2, -1)` | 转置最后两维 | 11 个 |
| 22 | MaskedFill | `tensor.masked_fill(mask == 0, -1e4)` | `mindspore.ops.masked_fill(tensor, mask == 0, -1e4)` | fill_value 一致，避免 `-inf` 导致 NaN | 11 个 |
| 23 | Concat | `torch.cat((a, b), dim=-1)` | `mindspore.ops.concat((a, b), axis=-1)` | 沿最后一维拼接两个张量 | 11 个 |
| 24 | Where | `torch.where(condition, x, y)` | `mindspore.ops.where(condition, x, y)` | 条件选择，condition 为 bool 类型 | 11 个 |
| 25 | Exp | `torch.exp(input)` | `mindspore.ops.exp(input)` | 指数函数 | 11 个 |
| 26 | Log | `torch.log(torch.clamp(input, min=1e-4))` | `mindspore.ops.log(mindspore.ops.clip_by_value(input, 1e-4, 1e9))` | 对数函数，输入 clamp 避免负数 | 11 个 |
| 27 | Sqrt | `torch.sqrt(input)` | `mindspore.ops.sqrt(input)` | 平方根 | 11 个 |
| 28 | Pow | `torch.pow(input, exponent)` | `mindspore.ops.pow(input, exponent)` | 幂运算，两个输入 | 11 个 |
| 29 | Clamp | `torch.clamp(input, min, max)` | `mindspore.ops.clip_by_value(input, min, max)` | 数值裁剪 | 11 个 |
| 30 | Split | `torch.split(input, split_size, dim)` | `mindspore.ops.split(input, split_size, axis=dim)` | 按大小分割张量 | 11 个 |
| 31 | Reshape | `torch.reshape(input, shape)` | `mindspore.ops.reshape(input, shape)` | 重塑张量形状 | 11 个 |
| 32 | Mean | `torch.mean(input, dim, keepdim)` | `mindspore.ops.mean(input, dim, keepdim)` | 均值归约 | 11 个 |
| 33 | Stack | `torch.stack((a, b), dim=0)` | `mindspore.ops.stack((a, b), axis=0)` | 沿新维度堆叠两个张量 | 11 个 |
| 34 | Expand | `tensor.expand(target_shape)` | `mindspore.ops.broadcast_to(input, shape)` | 维度广播扩展 | 11 个 |
| 35 | Tril | `torch.tril(input, diagonal=0)` | numpy 桥接 | 下三角矩阵，MSA CANN 不支持 | 11 个 |
| 36 | Sum | `torch.sum(input, dim, keepdim)` | `mindspore.ops.sum(input, dim, keepdim)` | 求和归约 | 11 个 |
| 37 | Max | `torch.max(input, dim)` | `mindspore.ops.max(input, dim, keepdim)` | 最大值归约 | 11 个 |
| 38 | Argmax | `torch.argmax(input, dim)` | numpy 桥接 | 取最大索引，MSA CANN 不支持 | 11 个 |
| 39 | Abs | `torch.abs(input)` | `mindspore.ops.abs(input)` | 绝对值 | 11 个 |
| 40 | Rsqrt | `torch.rsqrt(torch.clamp(input, min=1e-4))` | `mindspore.ops.rsqrt(clip_by_value(input, 1e-4, 1e9))` | 平方根倒数，输入 clamp 避免 0 | 11 个 |
| 41 | Eq | `torch.eq(a, b)` | `mindspore.ops.equal(a, b)` | 等于比较，输出 bool | 11 个 |
| 42 | Gather | `torch.gather(input, dim, index)` | `mindspore.ops.gather_elements(input, dim, index)` | 按索引取元素 | 11 个 |
| 43 | Pad | `F.pad(input, pad)` | `mindspore.ops.pad(input, pad)` | 边缘填充 | 11 个 |
| 44 | Sin | `torch.sin(input)` | `mindspore.ops.sin(input)` | 正弦函数 | 11 个 |
| 45 | Cos | `torch.cos(input)` | `mindspore.ops.cos(input)` | 余弦函数 | 11 个 |
| 46 | Permute | `tensor.permute(0, 2, 1)` | `mindspore.ops.transpose(input, (0, 2, 1))` | 维度重排 | 11 个 |
| 47 | Cumsum | `torch.cumsum(input, dim)` | numpy 桥接 | 前缀和，MSA CANN 不支持 | 11 个 |

**参数名差异速查表**：

| 语义 | PTA 参数名 | MSA 参数名 | 备注 |
|------|-----------|-----------|------|
| 词汇表大小 | `num_embeddings` | `vocab_size` | Embedding |
| 嵌入维度 | `embedding_dim` | `embedding_size` | Embedding |
| 输入维度 | `in_features` | `in_channels` | Linear / Dense |
| 输出维度 | `out_features` | `out_channels` | Linear / Dense |
| 是否加偏置 | `bias` | `has_bias` | Linear / Dense |
| 归一化稳定常数 | `eps` | `epsilon` | LayerNorm |
| 沿哪一维操作 | `dim` | `axis` | Softmax |
| 丢弃概率 | `p` | `1.0 - keep_prob` | Dropout（语义相反） |
| GELU 近似开关 | `approximate='tanh'` | `approximate=True` | GELU |
| 转置维度 | `dim0, dim1` | `axis1, axis2` | Transpose（`swapaxes`） |
| mask 填充值 | `value` | `value` | MaskedFill |

**重要约束**：所有算子必须使用 11 个模型在 PTA/MSA 中**实际安装的实现**（Megatron-Core / MindFormers 内置类），而非自定义简化版本。参数设置与整网配置完全一致。

---

## 三、源码级实现要求

### 3.1 实验目录结构（强制要求）

**所有实验代码必须放在 `/zyl/experiment/` 目录下**，与 `data2`、`lm-sv` 等其他文件夹完全解耦。

```
/zyl/experiment/
├── operator/                    # 算子级实验代码
│   ├── configs/
│   │   └── operator_experiment.yaml
│   ├── scripts/
│   │   ├── operator_diff_test.py       # RQ1 执行
│   │   ├── operator_metamorphic_test.py # RQ2 执行
│   │   ├── operator_diff_analysis.py    # RQ1 分析
│   │   └── operator_meta_analysis.py    # RQ2 分析
│   └── utils/
│       └── operator_registry.py
│
├── component/                   # 组件级实验代码
│   ├── configs/
│   │   └── component_experiment.yaml
│   ├── scripts/
│   │   ├── component_diff_test.py
│   │   └── component_metamorphic_test.py
│   └── utils/
│       ├── component_registry.py
│       └── metrics.py
│
└── common/                      # 算子级和组件级共用代码
    ├── config_loader.py         # 统一配置加载
    ├── tensor_manager.py        # 确定性张量生成
    ├── tensor_io.py             # 张量读写与格式转换
    ├── metrics.py               # 指标计算
    └── weight_sync.py           # 跨后端权重同步
```

**禁止**在 `lm-sv/`、`data2/` 或其他已有目录中创建或修改实验代码。实验代码必须完全自包含在 `experiment/` 目录中。

### 3.2 Conda 环境配置（强制要求）

算子级和组件级实验必须分别使用以下两个 conda 环境，环境定义参考 `zyl/lm-sv/task6_conda_envs_export/standard_env/` 中的 yml 文件。

| 环境名称 | 用途 | 环境定义文件 | Python 版本 | 核心包 |
|----------|------|-------------|------------|--------|
| `mindspeed` | PTA 后端实验 | `mindspeed_bare.yml` | 3.10 | torch==2.7.1, torch-npu==2.7.1.post2, transformers==4.55.2, numpy==1.26.0 |
| `msadapter` | MSA 后端实验 | `msadapter_bare.yml` | 3.10 | mindspore==2.8.0, torch==2.7.1, transformers==4.55.2, numpy==1.26.0 |

**环境创建命令**：

```bash
# PTA 环境（如果已存在则跳过）
conda env create -f zyl/lm-sv/task6_conda_envs_export/standard_env/mindspeed_bare.yml -n mindspeed

# MSA 环境（如果已存在则跳过）
conda env create -f zyl/lm-sv/task6_conda_envs_export/standard_env/msadapter_bare.yml -n msadapter
```

**环境激活与变量设置**：

```bash
# PTA 侧：激活 mindspeed 环境并设置变量
source zyl/lm-sv/task6_conda_envs_export/automated_setup/patches/envset/mm-pta-task6.sh

# MSA 侧：激活 msadapter 环境并设置变量
source zyl/lm-sv/task6_conda_envs_export/automated_setup/patches/envset/mm-msa-task6.sh
```

**关键环境变量说明**：

- `mm-pta-task6.sh`：激活 `mindspeed` conda 环境，设置 CANN 环境 (`/usr/local/Ascend/ascend-toolkit/set_env.sh`)，设置 `ASCEND_*` 系列 NPU 运行时变量，配置 PYTHONPATH（包含 Megatron-LM、MindSpeed、MindSpeed-MM）
- `mm-msa-task6.sh`：激活 `msadapter` conda 环境，设置 CANN 环境 (`/usr/local/Ascend/cann/set_env.sh`)，设置 `ASCEND_*` 系列 NPU 运行时变量，修复 libstdc++ 兼容性，配置 PYTHONPATH（包含 msadapter、Megatron-LM、MindSpeed）

**实验执行时必须显式指定 conda 环境**：

```bash
# RQ1 差分测试：分别在两端执行
conda run -n mindspeed python experiment/operator/scripts/operator_diff_test.py --backend pta
conda run -n msadapter python experiment/operator/scripts/operator_diff_test.py --backend msa

# RQ2 蜕变测试：在单端执行
conda run -n mindspeed python experiment/operator/scripts/operator_metamorphic_test.py --backend pta
conda run -n msadapter python experiment/operator/scripts/operator_metamorphic_test.py --backend msa
```

### 3.3 配置管理（强制要求）

**所有配置和路径禁止硬编码**，必须集中在一个统一的 YAML 配置文件中管理。

**配置文件路径**：`experiment/operator/configs/operator_experiment.yaml`

```yaml
# 实验基础配置
experiment:
  seed: 42
  device: "npu"
  output_dir: "res/operator_level"
  num_iterations: 10

# 扰动配置（RQ2 蜕变测试）
perturbation:
  sigmas: [1e-7, 1e-6, 1e-5, 1e-4]
  distribution: "uniform"  # 固定值扰动：每个元素 + sigma

# 算子输入形状配置
input_shapes:
  default: [32, 2, 1024]
  attention_score: [32, 16, 2, 2]
  ffn: [32, 2, 4096]
  vocab: [64]
  logits: [64, 40000]

# 指标配置
metrics:
  eps: 1e-8
  compute_ulp: false

# 模型配置映射
model_configs:
  qwen2: "model_config/qwen2.yaml"
  llama2: "model_config/llama2.yaml"
  baichuan2: "model_config/baichuan2.yaml"
  chatglm3: "model_config/chatglm3.yaml"
  glm4: "model_config/glm4.yaml"
  yi: "model_config/yi.yaml"
  codellama: "model_config/codellama.yaml"
  pangu: "model_config/pangu.yaml"
  deepseekv3: "model_config/deepseekv3.yaml"
  mixtral: "model_config/mixtral.yaml"
  grok1: "model_config/grok1.yaml"
```

**代码中读取配置**：

```python
from pathlib import Path
import yaml

_CONFIG = None

def get_config():
    global _CONFIG
    if _CONFIG is None:
        config_path = Path(__file__).parent.parent / "configs" / "operator_experiment.yaml"
        with open(config_path, "r") as f:
            _CONFIG = yaml.safe_load(f)
    return _CONFIG
```

### 3.4 张量保存规范（关键）

每个算子每次迭代需要记录 **4 个结果张量**，保存路径结构如下：

```
res/operator_level/
├── rq1_diff/                          # RQ1: 误差产生
│   └── {operator_name}/
│       ├── iter_{N}_pta_output.pt     # PTA 无扰动输出
│       └── iter_{N}_msa_output.pt     # MSA 无扰动输出
│
└── rq2_meta/                          # RQ2: 误差传播
    └── {operator_name}/
        ├── sigma_{sigma}/
        │   ├── iter_{N}_pta_baseline.pt   # PTA 扰动前输出
        │   ├── iter_{N}_pta_perturbed.pt  # PTA 扰动后输出
        │   ├── iter_{N}_msa_baseline.pt   # MSA 扰动前输出
        │   └── iter_{N}_msa_perturbed.pt  # MSA 扰动后输出
```

**保存代码示例**：

```python
from pathlib import Path
import torch

def save_tensor(tensor, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(tensor, "asnumpy"):  # MSA Tensor
        tensor = torch.from_numpy(tensor.asnumpy())
    torch.save(tensor.cpu(), path)
```

### 3.5 新增文件清单

| 文件路径 | 用途 | 行数预估 |
|----------|------|----------|
| `experiment/operator/configs/operator_experiment.yaml` | 统一配置 | ~50 行 |
| `experiment/operator/utils/operator_registry.py` | 47 个算子注册表 | ~600 行 |
| `experiment/operator/scripts/operator_diff_test.py` | RQ1 差分测试执行 | ~130 行 |
| `experiment/operator/scripts/operator_metamorphic_test.py` | RQ2 蜕变测试执行 | ~150 行 |
| `experiment/operator/scripts/operator_diff_analysis.py` | RQ1 差分指标分析 | ~100 行 |
| `experiment/operator/scripts/operator_meta_analysis.py` | RQ2 蜕变指标分析 | ~120 行 |
| `experiment/common/metrics.py` | 指标计算（与组件级共用） | ~80 行 |
| `experiment/common/tensor_io.py` | 张量读写与格式转换 | ~80 行 |
| `experiment/common/weight_sync.py` | 跨后端权重同步 | ~80 行 |

### 3.6 operator_registry.py 设计

```python
OPERATOR_REGISTRY = {
    "embedding": {
        "pta": lambda num_emb, emb_dim: torch.nn.Embedding(num_emb, emb_dim),
        "msa": lambda vocab_size, emb_size: mindspore.nn.Embedding(vocab_size, emb_size),
    },
    "layernorm": {
        "pta": lambda shape, eps=1e-5: torch.nn.LayerNorm(shape, eps=eps),
        "msa": lambda shape, eps=1e-5: mindspore.nn.LayerNorm(shape, epsilon=eps),
    },
    "rmsnorm": {
        "pta": lambda dim, eps=1e-6: _RMSNorm_PT(dim, eps),
        "msa": lambda dim, eps=1e-6: _RMSNorm_MS(dim, eps),
    },
    "linear": {
        "pta": lambda in_f, out_f, bias=True: torch.nn.Linear(in_f, out_f, bias=bias),
        "msa": lambda in_c, out_c, bias=True: mindspore.nn.Dense(in_c, out_c, has_bias=bias),
    },
    # ... 其余 16 个算子同理
}
```

### 3.7 metrics.py 设计

```python
def compute_diff_metrics(output_a, output_b, eps=1e-8):
    diff = torch.abs(output_a - output_b)
    return {
        "max_abs_diff": float(torch.max(diff)),
        "mean_abs_diff": float(torch.mean(diff)),
        "mean_rel_err": float(torch.mean(diff / (torch.abs(output_a) + eps))),
        "max_rel_err": float(torch.max(diff / (torch.abs(output_a) + eps))),
        "l2_norm": float(torch.norm(output_a - output_b, p=2)),
    }

def compute_metamorphic_metrics(baseline, perturbed, eps=1e-8):
    delta = torch.abs(perturbed - baseline)
    return {
        "max_abs_delta": float(torch.max(delta)),
        "mean_abs_delta": float(torch.mean(delta)),
        "rel_delta_max": float(torch.max(delta / (torch.abs(baseline) + eps))),
        "rel_delta_mean": float(torch.mean(delta / (torch.abs(baseline) + eps))),
    }
```

### 3.8 RQ1 主脚本设计（执行与分析分离）

**执行脚本**（`operator_diff_test.py`）只负责生成输入、运行算子、保存输出张量，不计算任何指标。

```python
def run_rq1(backend_filter=None):
    cfg = get_config("operator")
    out_dir = Path(cfg["experiment"]["output_dir"]) / "rq1_diff"
    tm = TensorManager(seed=cfg["experiment"]["seed"])

    for op_name, entry in OPERATOR_REGISTRY.items():
        for i in range(cfg["experiment"]["num_iterations"]):
            x = tm.generate(f"{op_name}_input", i)

            # 创建算子并同步权重（关键！否则误差 >1e-3）
            pta_op = entry["pta"](...)
            msa_op = entry["msa"](...)
            sync_weights(pta_op, msa_op, op_name, i)  # 或 save/load npz

            pta_out = pta_op(x)
            msa_out = msa_op(x)

            # 只保存 2 个张量，不计算指标
            save_tensor(pta_out, out_dir / op_name / f"iter_{i:03d}_pta_output.pt")
            save_tensor(msa_out, out_dir / op_name / f"iter_{i:03d}_msa_output.pt")
```

**分析脚本**（`operator_diff_analysis.py`）独立运行，加载保存的张量并计算指标：

```python
def run_analysis():
    for op_name in OPERATOR_REGISTRY:
        for i in range(num_iterations):
            pta = load_tensor(out_dir / op_name / f"iter_{i:03d}_pta_output.pt")
            msa = load_tensor(out_dir / op_name / f"iter_{i:03d}_msa_output.pt")
            metrics = compute_diff_metrics(pta, msa, eps)
            # metrics 写入 JSON
```

### 3.9 RQ2 主脚本设计（执行与分析分离）

**执行脚本**（`operator_metamorphic_test.py`）只负责运行算子并保存张量。

```python
def run_rq2(backend_filter=None):
    cfg = get_config("operator")
    out_dir = Path(cfg["experiment"]["output_dir"]) / "rq2_meta"
    tm = TensorManager(seed=cfg["experiment"]["seed"])

    for op_name, entry in OPERATOR_REGISTRY.items():
        for sigma in cfg["perturbation"]["sigmas"]:
            for i in range(cfg["experiment"]["num_iterations"]):
                x = tm.generate(f"{op_name}_input", i)
                x_perturbed = add_uniform_perturbation(x, sigma)  # 每个元素 + sigma

                for backend, factory in [("pta", entry["pta"]), ("msa", entry["msa"])]:
                    op = factory(...)
                    baseline = op(x)

                    # 独立实例运行扰动版本，并同步权重
                    op2 = factory(...)
                    copy_weights(op, op2, backend)
                    perturbed = op2(x_perturbed)

                    # 只保存 4 个张量
                    save_tensor(baseline, out_dir / op_name / f"sigma_{sigma}" / f"iter_{i:03d}_{backend}_baseline.pt")
                    save_tensor(perturbed, out_dir / op_name / f"sigma_{sigma}" / f"iter_{i:03d}_{backend}_perturbed.pt")
```

**分析脚本**（`operator_meta_analysis.py`）独立运行，加载保存的张量并计算指标。

---

## 四、指标定义汇总

### RQ1 差分测试指标

| 指标 | 公式 |
|------|------|
| Max Absolute Diff | `max(|pta - msa|)` |
| Mean Absolute Diff | `mean(|pta - msa|)` |
| Mean Relative Error | `mean(|pta - msa| / (|pta| + eps))` |
| Max Relative Error | `max(|pta - msa| / (|pta| + eps))` |
| L2 Norm | `||pta - msa||_2` |

### RQ2 蜕变测试指标

| 指标 | 公式 |
|------|------|
| Max Absolute Delta | `max(|perturbed - baseline|)` |
| Mean Absolute Delta | `mean(|perturbed - baseline|)` |
| Relative Delta Max | `max(|delta| / (|baseline| + eps))` |
| Relative Delta Mean | `mean(|delta| / (|baseline| + eps))` |
| 放大系数 | `Delta_output / Delta_input` |

---

## 五、权重同步约束（关键）

PTA 和 MSA 算子的默认随机初始化机制不同（不同的 RNG 种子、不同的分布参数），如果不做权重同步，RQ1 差分测试的误差会达到 ~4.0（embedding）和 ~2.0（linear），远超合理范围。

**强制要求**：
1. RQ1 执行脚本必须在每次迭代中，先创建 PTA 和 MSA 算子，然后同步权重，再执行前向传播
2. 由于 PTA 和 MSA 运行在不同 conda 环境中，采用 **npz 文件桥接**方案：
   - PTA 侧：保存权重到 `iter_{N}_pta_weights.npz`
   - MSA 侧：从 npz 加载并设置到对应参数
3. 参数名映射必须处理框架差异：`weight` ↔ `embedding_table` / `gamma` / `beta`
4. RQ2 执行脚本中，baseline 和 perturbed 两个独立实例的权重也必须相同（同后端内复制）

**实现文件**：`experiment/common/weight_sync.py`

## 六、代码风格约束

1. **简洁优先**：每个函数不超过 50 行，只实现要求的功能。
2. **禁止过度安全兜底**：信任输入合法性，不添加冗余 `try-except`。
3. **禁止过度设计**：不用工厂模式等；直接用字典和 lambda。
4. **配置集中管理**：所有路径、参数从 YAML 读取，代码中无字面量常量。
5. **类型注解**：核心函数添加类型注解。
6. **注释规范**：只注释非显而易见的逻辑。

---

## 六、执行命令

```bash
# ========== 前置步骤：设置环境变量 ==========
# PTA 侧（每次新终端都需要执行）
source zyl/lm-sv/task6_conda_envs_export/automated_setup/patches/envset/mm-pta-task6.sh

# MSA 侧（每次新终端都需要执行）
source zyl/lm-sv/task6_conda_envs_export/automated_setup/patches/envset/mm-msa-task6.sh

# ========== RQ1 差分测试（需在两端分别执行） ==========
# PTA 端（先执行，保存权重 npz）
conda run -n mindspeed python experiment/operator/scripts/operator_diff_test.py --backend pta

# MSA 端（后执行，加载 PTA 权重）
conda run -n msadapter python experiment/operator/scripts/operator_diff_test.py --backend msa

# RQ1 分析（与后端无关，纯张量计算）
python experiment/operator/scripts/operator_diff_analysis.py

# ========== RQ2 蜕变测试（在单端执行即可） ==========
# PTA 端
conda run -n mindspeed python experiment/operator/scripts/operator_metamorphic_test.py --backend pta

# MSA 端
conda run -n msadapter python experiment/operator/scripts/operator_metamorphic_test.py --backend msa

# RQ2 分析
python experiment/operator/scripts/operator_meta_analysis.py
```
