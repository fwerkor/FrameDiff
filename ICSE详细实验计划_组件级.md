# ICSE 详细实验计划 —— 组件级

> 本文档从源码实现角度，规定组件级实验（RQ1 差分测试 + RQ2 蜕变测试）的全部功能、接口、文件结构与代码要求。  
> 配套文档：`ICSE实验计划.md`、`ICSE论文提纲.md`、`ICSE详细实验计划_算子级.md`

---

## 一、实验目标

### 1.1 RQ1：误差的产生（差分测试）

在 PTA 和 MSA 双后端上，使用**完全相同的输入张量和权重**，分别执行同一组件块，对比输出差异。差异张量的**最大值 / 平均值**表征误差的产生。

### 1.2 RQ2：误差的传播（蜕变测试）

在**单一后端**（MSA 或 PTA）内，对组件输入添加**微小扰动**（Gaussian noise，sigma ∈ [1e-7, 1e-6, 1e-5, 1e-4]），对比扰动前后的输出变化。差异张量的**最大值 / 平均值**表征误差的传播。

### 1.3 数据策略

组件级输入使用**随机生成的张量**，由 `TensorManager` 按确定性种子生成。不依赖真实数据，保证可复现。

**关键约束**：PTA 和 MSA 两端必须使用完全相同的随机输入张量（共享同一 seed 和生成参数）。

---

## 二、组件清单与 API 映射（7 个，无遗漏无重复）

以下组件覆盖 11 个大语言模型中**所有实际存在的组件块**。每个组件对应 Megatron-Core（PTA）和 MindFormers（MSA）中的真实类。

| 编号 | 组件名称 | PTA 真实接口 | MSA 真实接口 | 参数对齐说明 | 使用模型 |
|------|----------|-------------|-------------|-------------|----------|
| 11 | Embedding Layer | `LanguageModelEmbedding(config, vocab_size, max_seq_len, position_embedding_type)` | `LanguageModelEmbedding(config, vocab_size, max_seq_len, position_embedding_type)` | 两端均为框架内置实现，参数完全一致 | 11 个 |
| 12 | Self-Attention Block | `TransformerBlock.layers[0].self_attention` | `TransformerBlock.layers[0].self_attention` | 两端均为 `TransformerBlock` 内嵌子模块，需确保 `num_attention_heads`、`num_query_groups`、`kv_channels` 一致 | 11 个 |
| 13 | FFN Block | `TransformerBlock.layers[0].mlp` | `TransformerBlock.layers[0].mlp` | 标准 MLP（非 MoE），两端均为 `TransformerBlock` 内嵌子模块 | 8 个标准模型 |
| 14 | Decoder Block | `TransformerBlock.layers[0]` (完整 DecoderLayer) | `TransformerBlock.layers[0]` (完整 DecoderLayer) | 完整 DecoderLayer = Self-Attention Block + FFN/MoE Block + 两个残差连接 + 两个 LayerNorm/RMSNorm | 11 个 |
| 15 | Output Layer | `tensor_parallel.ColumnParallelLinear(hidden_size, vocab_size, config=config, bias=False)` | `ColumnParallelLinear(hidden_size, vocab_size, config=config, bias=False)` | PTA 使用 `megatron.core.tensor_parallel.ColumnParallelLinear`；MSA 使用 `mindformers.parallel_core.inference.tensor_parallel.layers.ColumnParallelLinear` | 11 个 |
| 16 | MoE FFN Block | `TransformerBlock.layers[0].mlp` (MoE variant) | `TransformerBlock.layers[0].mlp` (MoE variant) | MoE 模型的 MLP 子模块，内含 router + top-k gating + expert computation。需确保 `num_experts`、`top_k` 与整网配置一致 | DeepSeekV3, Mixtral, Grok1 |
| 17 | MLA Self-Attention Block | DeepSeekV3 自定义 `MLASelfAttention` | DeepSeekV3 MSA 自定义 `MLASelfAttention` | 仅生产配置启用；`multi_latent_attention=True`。`q_lora_rank` / `kv_lora_rank` / `v_head_dim` 维度需对齐 | DeepSeekV3（生产配置） |

**重要约束**：
1. 所有组件必须使用 11 个模型在 PTA/MSA 中**实际安装的实现**（Megatron-Core / MindFormers 内置类），而非自定义简化版本。
2. 组件 12 的完整定义包括**残差连接**：`output = input + attention_output`。
3. 组件 13 的完整定义包括**残差连接**：`output = input + mlp_output`。
4. 组件 14（Decoder Block）在标准模型中包含 SA Block + FFN Block；在 MoE 模型中包含 SA Block + MoE FFN Block。
5. 组件 16（MoE FFN Block）替换普通 FFN Block，仅用于 3 个 MoE 模型。
6. 组件 17（MLA SA Block）替换普通 SA Block，仅用于 DeepSeekV3 生产配置；测试配置中 `multi_latent_attention=False`。

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
│   │   ├── operator_diff_test.py
│   │   └── operator_metamorphic_test.py
│   └── utils/
│       ├── operator_registry.py
│       └── metrics.py
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
    ├── tensor_manager.py        # 确定性张量生成
    └── config_loader.py         # 统一配置加载
```

**禁止**在 `lm-sv/`、`data2/` 或其他已有目录中创建或修改实验代码。实验代码必须完全自包含在 `experiment/` 目录中。

### 3.2 Conda 环境配置（强制要求）

组件级和算子级实验共用以下两个 conda 环境，环境定义参考 `zyl/lm-sv/task6_conda_envs_export/standard_env/` 中的 yml 文件。

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
conda run -n mindspeed python experiment/component/scripts/component_diff_test.py --backend pta
conda run -n msadapter python experiment/component/scripts/component_diff_test.py --backend msa

# RQ2 蜕变测试：在单端执行
conda run -n mindspeed python experiment/component/scripts/component_metamorphic_test.py --backend pta
conda run -n msadapter python experiment/component/scripts/component_metamorphic_test.py --backend msa
```

### 3.3 配置管理（强制要求）

**所有配置和路径禁止硬编码**，必须集中在一个统一的 YAML 配置文件中管理。

**配置文件路径**：`experiment/component/configs/component_experiment.yaml`

```yaml
# 实验基础配置
experiment:
  seed: 42
  device: "npu"
  output_dir: "res/component_level"
  num_iterations: 10

# 扰动配置（RQ2 蜕变测试）
perturbation:
  sigmas: [1e-7, 1e-6, 1e-5, 1e-4]
  distribution: "gaussian"

# 组件输入形状配置
input_shapes:
  default: [32, 2, 1024]        # (seq_len, batch_size, hidden_size)
  embedding_input: [32, 2]       # (seq_len, batch_size) -> input_ids
  output_layer_input: [32, 2, 1024]  # (seq_len, batch_size, hidden_size)
  output_layer_output: [32, 2, 40000] # (seq_len, batch_size, vocab_size)

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

# 组件编号映射
components:
  embedding_layer: 11
  self_attention_block: 12
  ffn_block: 13
  decoder_block: 14
  output_layer: 15
  moe_ffn_block: 16
  mla_self_attention_block: 17
```

**代码中读取配置**：

```python
from pathlib import Path
import yaml

_CONFIG = None

def get_config():
    global _CONFIG
    if _CONFIG is None:
        config_path = Path(__file__).parent.parent / "configs" / "component_experiment.yaml"
        with open(config_path, "r") as f:
            _CONFIG = yaml.safe_load(f)
    return _CONFIG
```

### 3.4 张量保存规范（关键）

每个组件每次迭代需要记录 **4 个结果张量**，保存路径结构如下：

```
res/component_level/
├── rq1_diff/                          # RQ1: 误差产生
│   └── {component_name}/
│       ├── iter_{N}_pta_output.pt     # PTA 无扰动输出
│       └── iter_{N}_msa_output.pt     # MSA 无扰动输出
│
└── rq2_meta/                          # RQ2: 误差传播
    └── {component_name}/
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
| `experiment/component/configs/component_experiment.yaml` | 统一配置 | ~50 行 |
| `experiment/component/utils/component_registry.py` | 7 个组件注册表 | ~300 行 |
| `experiment/component/scripts/component_diff_test.py` | RQ1 差分测试 | ~150 行 |
| `experiment/component/scripts/component_metamorphic_test.py` | RQ2 蜕变测试 | ~150 行 |
| `experiment/common/metrics.py` | 指标计算（与算子级共用） | ~80 行 |

### 3.6 component_registry.py 设计

```python
COMPONENT_REGISTRY = {
    "embedding_layer": {
        "pta": lambda config: LanguageModelEmbedding(
            config=config,
            vocab_size=config.vocab_size,
            max_sequence_length=config.max_sequence_length,
            position_embedding_type=config.position_embedding_type,
        ),
        "msa": lambda config: LanguageModelEmbedding(
            config=config,
            vocab_size=config.vocab_size,
            max_sequence_length=config.max_sequence_length,
            position_embedding_type=config.position_embedding_type,
        ),
        "input_shape": [32, 2],  # (seq_len, batch_size) -> input_ids
        "output_shape": [32, 2, 1024],  # (seq_len, batch_size, hidden_size)
        "models": ["qwen2", "llama2", "baichuan2", "chatglm3", "glm4", "yi", "codellama", "pangu", "deepseekv3", "mixtral", "grok1"],
    },
    "self_attention_block": {
        "pta": lambda config: TransformerBlock(config=config, spec=get_gpt_layer_local_spec()).layers[0].self_attention,
        "msa": lambda config: TransformerBlock(config=config, spec=get_gpt_layer_local_spec()).layers[0].self_attention,
        "input_shape": [32, 2, 1024],  # (seq_len, batch_size, hidden_size)
        "output_shape": [32, 2, 1024],
        "models": ["qwen2", "llama2", "baichuan2", "chatglm3", "glm4", "yi", "codellama", "pangu", "deepseekv3", "mixtral", "grok1"],
    },
    "ffn_block": {
        "pta": lambda config: TransformerBlock(config=config, spec=get_gpt_layer_local_spec()).layers[0].mlp,
        "msa": lambda config: TransformerBlock(config=config, spec=get_gpt_layer_local_spec()).layers[0].mlp,
        "input_shape": [32, 2, 1024],
        "output_shape": [32, 2, 1024],
        "models": ["qwen2", "llama2", "baichuan2", "chatglm3", "glm4", "yi", "codellama", "pangu"],
    },
    "decoder_block": {
        "pta": lambda config: TransformerBlock(config=config, spec=get_gpt_layer_local_spec()).layers[0],
        "msa": lambda config: TransformerBlock(config=config, spec=get_gpt_layer_local_spec()).layers[0],
        "input_shape": [32, 2, 1024],
        "output_shape": [32, 2, 1024],
        "models": ["qwen2", "llama2", "baichuan2", "chatglm3", "glm4", "yi", "codellama", "pangu", "deepseekv3", "mixtral", "grok1"],
    },
    "output_layer": {
        "pta": lambda config: ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            config=config,
            init_method=config.init_method,
            bias=False,
        ),
        "msa": lambda config: ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            config=config,
            init_method=config.init_method,
            bias=False,
        ),
        "input_shape": [32, 2, 1024],
        "output_shape": [32, 2, 40000],  # vocab_size
        "models": ["qwen2", "llama2", "baichuan2", "chatglm3", "glm4", "yi", "codellama", "pangu", "deepseekv3", "mixtral", "grok1"],
    },
    "moe_ffn_block": {
        "pta": lambda config: TransformerBlock(config=config, spec=get_gpt_layer_local_spec()).layers[0].mlp,
        "msa": lambda config: TransformerBlock(config=config, spec=get_gpt_layer_local_spec()).layers[0].mlp,
        "input_shape": [32, 2, 1024],
        "output_shape": [32, 2, 1024],
        "models": ["deepseekv3", "mixtral", "grok1"],
    },
    "mla_self_attention_block": {
        "pta": lambda config: DeepSeekV3_MLA_Attention(config),
        "msa": lambda config: DeepSeekV3_MSA_MLA_Attention(config),
        "input_shape": [32, 2, 1024],
        "output_shape": [32, 2, 1024],
        "models": ["deepseekv3"],
    },
}
```

**组件执行包装器**（确保残差连接等完整逻辑）：

```python
def run_self_attention_block(block, hidden_states, attention_mask=None):
    """组件 12：Self-Attention Block 完整执行（含残差连接）"""
    residual = hidden_states
    normed = block.input_layernorm(hidden_states)
    attn_out = block.self_attention(normed, attention_mask)[0]
    proj_out = block.linear_proj(attn_out)[0]
    dropped = block.attention_dropout(proj_out)
    return dropped + residual

def run_ffn_block(block, hidden_states):
    """组件 13：FFN Block 完整执行（含残差连接）"""
    residual = hidden_states
    normed = block.pre_mlp_layernorm(hidden_states)
    mlp_out = block.mlp(normed)[0]
    return mlp_out + residual

def run_decoder_block(block, hidden_states, attention_mask=None):
    """组件 14：Decoder Block 完整执行（SA + FFN + 两个残差）"""
    # Self-Attention Block
    residual = hidden_states
    normed = block.input_layernorm(hidden_states)
    attn_out = block.self_attention(normed, attention_mask)[0]
    proj_out = block.linear_proj(attn_out)[0]
    hidden_states = proj_out + residual
    # FFN Block
    residual = hidden_states
    normed = block.pre_mlp_layernorm(hidden_states)
    mlp_out = block.mlp(normed)[0]
    return mlp_out + residual
```

### 3.7 metrics.py 设计（与算子级共用）

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

### 3.8 RQ1 主脚本设计

```python
def run_rq1():
    cfg = get_config()
    out_dir = Path(cfg["experiment"]["output_dir"]) / "rq1_diff"
    tm = TensorManager(seed=cfg["experiment"]["seed"])

    for comp_name, entry in COMPONENT_REGISTRY.items():
        for i in range(cfg["experiment"]["num_iterations"]):
            x = tm.generate(f"{comp_name}_input", i)

            pta_comp = entry["pta"](config)
            msa_comp = entry["msa"](config)

            pta_out = pta_comp(x)
            msa_out = msa_comp(x)

            # 保存 2 个张量
            save_tensor(pta_out, out_dir / comp_name / f"iter_{i:03d}_pta_output.pt")
            save_tensor(msa_out, out_dir / comp_name / f"iter_{i:03d}_msa_output.pt")

            metrics = compute_diff_metrics(pta_out, to_torch(msa_out))
            # metrics 写入 JSON
```

### 3.9 RQ2 主脚本设计

```python
def run_rq2():
    cfg = get_config()
    out_dir = Path(cfg["experiment"]["output_dir"]) / "rq2_meta"
    tm = TensorManager(seed=cfg["experiment"]["seed"])

    for comp_name, entry in COMPONENT_REGISTRY.items():
        for sigma in cfg["perturbation"]["sigmas"]:
            for i in range(cfg["experiment"]["num_iterations"]):
                x = tm.generate(f"{comp_name}_input", i)
                x_perturbed = add_gaussian_noise(x, sigma)

                for backend, factory in [("pta", entry["pta"]), ("msa", entry["msa"])]:
                    comp = factory(config)
                    baseline = comp(x)
                    perturbed = comp(x_perturbed)

                    # 保存 4 个张量
                    save_tensor(baseline, out_dir / comp_name / f"sigma_{sigma}" / f"iter_{i:03d}_{backend}_baseline.pt")
                    save_tensor(perturbed, out_dir / comp_name / f"sigma_{sigma}" / f"iter_{i:03d}_{backend}_perturbed.pt")

                    metrics = compute_metamorphic_metrics(to_torch(baseline), to_torch(perturbed))
```

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

## 五、代码风格约束

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
# PTA 端
conda run -n mindspeed python experiment/component/scripts/component_diff_test.py --backend pta

# MSA 端
conda run -n msadapter python experiment/component/scripts/component_diff_test.py --backend msa

# ========== RQ2 蜕变测试（在单端执行即可） ==========
# PTA 端
conda run -n mindspeed python experiment/component/scripts/component_metamorphic_test.py --backend pta

# MSA 端
conda run -n msadapter python experiment/component/scripts/component_metamorphic_test.py --backend msa
```

---

## 七、与现有代码的关系说明

组件级实验代码**完全独立**于 `lm-sv/` 目录，但在实现上参考了以下已有代码的设计：

| 参考来源 | 参考内容 | 用途 |
|----------|----------|------|
| `lm-sv/lmsv_new/core/subgraph.py` | PTA 侧 `Graph` 类中 `submodule_num` 0~10 的执行逻辑 | 参考如何调用 Megatron-Core 组件 |
| `lm-sv/lmsv_rec/utils/runtime/mf_mutate_and_forward/sub_graph.py` | MSA 侧 `Graph` 类中 `submodule_num` 0~10 的执行逻辑 | 参考如何调用 MindFormers 组件 |
| `lm-sv/lmsv_new/utils/tensor_manager.py` | `TensorManager` 确定性张量生成 | 复用或迁移到 `experiment/common/` |

**重要**：组件级实验代码**不修改** `lm-sv/` 中的任何文件，也不依赖 `lm-sv/` 的运行时环境。所有依赖（Megatron-Core、MindFormers）在 `experiment/` 中独立导入。
