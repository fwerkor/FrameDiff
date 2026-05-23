# ICSE 论文实验计划

## 一、实验目标

### RQ1：误差的产生（差分测试）
在 PTA 和 MSA 双后端上，使用**完全相同的输入张量和权重**，分别执行同一算子、同一组件或同一模型，对比输出差异。差异张量的**最大值 / 平均值**表征误差的产生。

### RQ2：误差的传播（蜕变测试）
在**单一后端**（MSA 或 PTA）内，对算子输入、组件输入或模型输入添加**微小扰动**（如 Gaussian noise），对比扰动前后的输出变化。差异张量的**最大值 / 平均值**表征误差的传播。

---

## 二、三个实验层次总览

| 层次 | 定义 | 执行方式 | RQ1 差分测试 | RQ2 蜕变测试 | 现有支持 |
|------|------|----------|--------------|--------------|----------|
| **算子层** | 单个算子（MatMul、Softmax 等） | 独立脚本，确定性输入 | PTA vs MSA 输出差异 | 单一后端，输入加扰动 | ❌ 无，需新增 |
| **组件层** | 组件块（SA Block、FFN Block 等） | 单独执行组件 | PTA vs MSA 输出差异 | 单一后端，输入加扰动 | ⚠️ Task 2 可改 |
| **模型层** | 完整模型 | 整网执行 | PTA vs MSA loss 差异 | 单一后端，输入加扰动 | ✅ Task 1 已支持 |

**数据策略**：
- **算子层 / 组件层**：输入使用**随机生成的张量**（`TensorManager` 确定性种子），不依赖真实数据，保证可复现。
- **模型层**：输入使用**真实数据**（训练语料 / 评估样本），反映实际部署场景。

模型层整网执行时，仍通过前向钩子记录每个组件的输入张量，用于模型级分析；组件层独立实验时不再复用这些张量，改用随机输入。

---

## 三、模型清单（11 个）

| 架构类型 | 模型 | 配置文件 | 归一化 | 激活函数 | 位置编码 | FFN 类型 |
|----------|------|----------|--------|----------|----------|----------|
| Dense Transformer | Qwen2 | `model_config/qwen2.yaml` | RMSNorm | SwiGLU | RoPE | Standard MLP |
| Dense Transformer | Llama2 | `model_config/llama2.yaml` | RMSNorm | SwiGLU | RoPE | Standard MLP |
| Dense Transformer | Baichuan2 | `model_config/baichuan2.yaml` | RMSNorm | SwiGLU | ALiBi | Standard MLP |
| Dense Transformer | ChatGLM3 | `model_config/chatglm3.yaml` | RMSNorm | SwiGLU | RoPE (GLM) | Standard MLP |
| Dense Transformer | GLM4 | `model_config/glm4.yaml` | RMSNorm | SwiGLU | RoPE (GLM) | Standard MLP |
| Dense Transformer | Yi | `model_config/yi.yaml` | RMSNorm | SwiGLU | RoPE | Standard MLP |
| Dense Transformer | CodeLlama | `model_config/codellama.yaml` | RMSNorm | SwiGLU | RoPE | Standard MLP |
| Dense Transformer | PanGu | `model_config/pangu.yaml` | LayerNorm | fast_gelu (GELU) | Learned | Standard MLP |
| MoE Transformer | DeepSeekV3 | `model_config/deepseekv3.yaml` | RMSNorm | SwiGLU | RoPE | MoE (32 experts, topk=8) |
| MoE Transformer | Mixtral | `model_config/mixtral.yaml` | RMSNorm | SwiGLU | RoPE | MoE (4 experts, topk=2) |
| MoE Transformer | Grok1 | `model_config/grok1.yaml` | RMSNorm | SwiGLU | RoPE | MoE (4 experts, topk=2) |

**关键说明**：
- **LayerNorm**：仅 PanGu 使用；其余 10 个模型均使用 RMSNorm。
- **GELU**：仅 PanGu 使用 fast_gelu；其余 10 个模型均使用 SwiGLU。
- **ALiBi**：仅 Baichuan2 使用；其余 10 个模型使用 RoPE 或 Learned 位置编码。
- **MoE**：仅 DeepSeekV3、Mixtral、Grok1 使用；其余 8 个模型使用标准 MLP。
- **MLA**：DeepSeekV3 测试配置中 `multi_latent_attention: False`（禁用）；生产配置启用。概念上保留 MLA 算子和组件，但实验时仅测试生产配置。

**选取策略**：每个子实验从 Dense 和 MoE 中各选 2~3 个代表性模型，不必全部运行。

---

## 四、算子层实验

### 4.1 算子清单（去重后，一个不能少）

以下算子覆盖 11 个大语言模型中**所有实际使用的算子类型**。同一 PyTorch API 只出现一次，通过"出现位置"列描述所有功能位置。

| 序号 | 算子类别 | 算子名称 | PTA 真实接口 | MSA 真实接口 | 参数对齐说明 | 典型输入形状 | 出现位置 | 使用模型 |
|------|----------|----------|-------------|-------------|-------------|--------------|----------|----------|
| 1 | 嵌入 | Embedding | `torch.nn.Embedding(num_embeddings, embedding_dim)` | `mindspore.nn.Embedding(vocab_size, embedding_size)` | `num_embeddings` ↔ `vocab_size`（词汇表大小）；`embedding_dim` ↔ `embedding_size`（嵌入维度）。两对参数语义完全一致，仅命名不同。 | `(B, S)` | word embedding, position embedding, top_query embedding | 11 个 |
| 2 | 归一化 | LayerNorm | `torch.nn.LayerNorm(normalized_shape, eps=1e-5)` | `mindspore.nn.LayerNorm(normalized_shape, epsilon=1e-5)` | `normalized_shape` 完全一致；`eps` ↔ `epsilon`（归一化稳定常数）。语义一致，仅命名不同。 | `(B, S, H)` | input_layernorm, pre_mlp_layernorm | PanGu |
| 3 | 归一化 | RMSNorm | `class RMSNorm(torch.nn.Module)` | `class RMSNorm(mindspore.nn.Cell)` | 两端均无原生 API，均为框架自定义实现。需确保自定义类的前向逻辑完全一致（`x * rsqrt(mean(x^2) + eps) * weight`）。 | `(B, S, H)` | input_layernorm, pre_mlp_layernorm | 10 个（除 PanGu） |
| 4 | 线性变换 | Linear | `torch.nn.Linear(in_features, out_features, bias=True)` | `mindspore.nn.Dense(in_channels, out_channels, has_bias=True)` | `in_features` ↔ `in_channels`（输入维度）；`out_features` ↔ `out_channels`（输出维度）；`bias` ↔ `has_bias`（是否加偏置）。两对参数语义完全一致，仅命名不同。 | `(B, S, H)` / `(B, S, FFN)` | linear_qkv, linear_proj, linear_fc1, linear_fc2, output_layer, expert_linear | 11 个 |
| 5 | 矩阵乘法 | MatMul | `torch.matmul(input, other)` | `mindspore.ops.matmul(input, other)` | 参数 `input` / `other` 完全一致；广播语义一致。 | `(B, H, S, K)` / `(B, H, S, S)` | Attention Q@K^T, Score@V | 11 个 |
| 6 | 注意力 | ScaledDotProductAttention | `F.scaled_dot_product_attention(q, k, v, attn_mask=None)` | MindSpore 自定义实现（`ops.BatchMatMul` + `ops.Softmax` + `ops.masked_fill`） | PTA 为 PyTorch 原生融合算子；MSA 无原生等价物，需用 `BatchMatMul` + `Softmax` + `masked_fill` 组合实现。需确保 scale 因子（`1/sqrt(d_k)`）和 mask 填充值（`-inf`）完全一致。 | `(B, S, H)` | core_attention | 11 个 |
| 7 | 注意力 | FlashAttention | `flash_attn_func(q, k, v, causal=True)` | `flash_attn_func(q, k, v, causal=True)` | 两端调用同一第三方库 `flash-attn`，参数 `causal` 语义完全一致。 | `(B, S, H)` | core_attention (优化路径) | 部分模型 |
| 8 | Softmax | Softmax | `F.softmax(input, dim=-1)` | `mindspore.ops.softmax(input, axis=-1)` | `dim` ↔ `axis`（沿哪一维做归一化）。语义完全一致，仅命名不同。 | `(B, H, S, S)` | attention score normalization | 11 个 |
| 9 | 激活函数 | GELU | `torch.nn.GELU(approximate='tanh')` | `mindspore.nn.GELU(approximate=True)` | PTA `approximate='tanh'` 与 MSA `approximate=True` 均表示**快速近似版本**（tanh 近似，非精确 erf）。PanGu 使用的 `fast_gelu` 也是同一近似族。两端必须同时启用近似或同时禁用近似，才能对齐。 | `(B, S, FFN)` | MLP activation | PanGu |
| 10 | 激活函数 | SiLU | `torch.nn.SiLU()` | `mindspore.nn.SiLU()` | 两端均无额外参数，实现均为 `x * sigmoid(x)`，语义完全一致。 | `(B, S, FFN)` | SwiGLU gate | SwiGLU 模型 |
| 11 | 激活函数 | SwiGLU | `torch.nn.SiLU() * gate_proj(x) + up_proj(x)` | `mindspore.nn.SiLU() * gate + mindspore.nn.Dense()` | SwiGLU = SiLU(gate) * up + down，两端逻辑完全一致。gate/up/down 三个 Linear/Dense 的参数需按条目 4 对齐。 | `(B, S, FFN)` | MLP activation | 10 个（除 PanGu） |
| 12 | 位置编码 | RoPE | `RotaryEmbedding(dim, max_seq_len, base=10000)` | `RotaryEmbedding(dim, max_seq_len, base=10000)` | 两端均使用 MindFormers 的 `RotaryEmbedding` 实现，参数 `dim` / `max_seq_len` / `base` 语义完全一致。 | `(B, S, H)` | position encoding | 9 个（RoPE 模型） |
| 13 | 位置编码 | ALiBi | `自定义: 构建偏置矩阵 + torch.add(input, bias)` | `自定义: 构建偏置矩阵 + mindspore.ops.add(input, bias)` | 两端均无原生 API，均为自定义实现。需确保偏置矩阵的构建逻辑（基于 head 和距离）和加法操作完全一致。 | `(B, S, H)` | position encoding | Baichuan2 |
| 14 | 正则化 | Dropout | `torch.nn.Dropout(p=0.1)` | `mindspore.nn.Dropout(keep_prob=0.9)` | `p`（丢弃概率）↔ `keep_prob`（保留概率）。两者**语义相反、数值互补**：`keep_prob = 1.0 - p`。当 PTA `p=0.1` 时，MSA 必须设 `keep_prob=0.9`，才能实现完全相同的 dropout 行为。 | `(B, S, H)` | attention dropout, hidden dropout, embedding dropout | 11 个 |
| 15 | 逐元素运算 | Add | `torch.add(input, other)` | `mindspore.ops.add(input, other)` | 参数 `input` / `other` 完全一致；广播语义一致。 | `(B, S, H)` | residual connection, bias add | 11 个 |
| 16 | 逐元素运算 | Mul | `torch.mul(input, other)` | `mindspore.ops.mul(input, other)` | 参数 `input` / `other` 完全一致；广播语义一致。 | `(B, H, S, S)` | attention scale (1/sqrt(dk)) | 11 个 |
| 17 | 损失函数 | CrossEntropy | `F.cross_entropy(input, target, reduction='mean')` | `mindspore.ops.cross_entropy(input, target, reduction='mean')` | `input`（logits）/ `target`（类别索引）/ `reduction='mean'` 三参数语义完全一致。MSA 的 `ops.cross_entropy` 与 PTA 的 `F.cross_entropy` 同为函数式 API，比对类式 `nn.CrossEntropyLoss()` 更直接对齐。 | `(B*S, V), (B*S,)` | training loss | 11 个 |
| 18 | MoE 路由 | TopKGating | `自定义: router_logits → topk → load balancing` | `自定义: router_logits → topk → aux_loss` | 两端均无原生 API，均为自定义实现。需确保 top-k 选择逻辑、负载均衡系数、`aux_loss` 计算方式一致。 | `(B, S, E)` | MoE router | DeepSeekV3, Mixtral, Grok1 |
| 19 | MLA 投影 | MLA_Q_Projection | `自定义: low-rank projection + RoPE` | `自定义: low-rank projection + RoPE` | 两端均无原生 API，均为自定义实现。需确保低秩投影矩阵的维度（`q_lora_rank`）和 RoPE 应用逻辑一致。 | `(B, S, H)` | DeepSeekV3 q_lora_rank | DeepSeekV3（生产配置） |
| 20 | MLA 投影 | MLA_KV_Projection | `自定义: low-rank projection + RoPE` | `自定义: low-rank projection + RoPE` | 两端均无原生 API，均为自定义实现。需确保低秩投影矩阵的维度（`kv_lora_rank`）和 RoPE 应用逻辑一致。 | `(B, S, H)` | DeepSeekV3 kv_lora_rank | DeepSeekV3（生产配置） |

**注**：上表中 `B`=batch, `S`=seq_len, `H`=hidden_size, `FFN`=ffn_hidden_size, `V`=vocab_size, `E`=num_experts。

**参数名差异速查表**：
| 语义 | PTA 参数名 | MSA 参数名 | 备注 |
|------|-----------|-----------|------|
| 词汇表大小 | `num_embeddings` | `vocab_size` | Embedding |
| 嵌入维度 | `embedding_dim` | `embedding_size` | Embedding |
| 输入维度 | `in_features` | `in_channels` | Linear / Dense |
| 输出维度 | `out_features` | `out_channels` | Linear / Dense |
| 是否加偏置 | `bias` | `has_bias` | Linear / Dense |
|  egalization 稳定常数 | `eps` | `epsilon` | LayerNorm |
| 沿哪一维操作 | `dim` | `axis` | Softmax |
| 丢弃概率 | `p` | `1.0 - keep_prob` | Dropout（语义相反） |
| GELU 近似开关 | `approximate='tanh'`（字符串） | `approximate=True`（布尔） | GELU（都表示近似版本） |
| 函数式 API | `F.cross_entropy` | `ops.cross_entropy` | CrossEntropy |

### 4.2 RQ1 差分测试（算子级）

**执行流程**：
1. 使用 `TensorManager` 生成**随机输入张量**（seed=42，迭代编号作为确定性种子）
2. **关键：PTA 和 MSA 两端必须使用完全相同的随机输入张量**（共享同一 seed 和生成参数，确保跨后端输入一致性）
3. PTA 后端实例化算子，执行前向传播，保存输出张量
4. MSA 后端实例化**同一算子**，使用**完全相同输入**，执行前向传播，保存输出张量
5. 计算差异指标

**差异指标**（一个不能少）：
- **Max Absolute Difference**：`torch.max(torch.abs(output_pta - output_msa))`
- **Mean Absolute Difference**：`torch.mean(torch.abs(output_pta - output_msa))`
- **Mean Relative Error**：`torch.mean(torch.abs(output_pta - output_msa) / (torch.abs(output_pta) + 1e-8))`
- **Max Relative Error**：`torch.max(torch.abs(output_pta - output_msa) / (torch.abs(output_pta) + 1e-8))`
- **L2 Norm**：`torch.norm(output_pta - output_msa, p=2)`
- **ULP Distance**：浮点表示的 ULP 距离（用于 FP16/BF16 算子）

### 4.3 RQ2 蜕变测试（算子级）

**执行流程**：
1. 基线：确定性输入 `x` → 算子 → 输出 `y_baseline`
2. 变异：输入 `x' = x + epsilon`，其中 `epsilon ~ N(0, sigma^2)`，`sigma` 取值：`[1e-7, 1e-6, 1e-5, 1e-4]`
3. 同一后端执行：`x'` → 算子 → 输出 `y_perturbed`
4. 计算传播差异指标

**传播差异指标**（一个不能少）：
- **Max Absolute Delta**：`torch.max(torch.abs(y_perturbed - y_baseline))`
- **Mean Absolute Delta**：`torch.mean(torch.abs(y_perturbed - y_baseline))`
- **Relative Delta Max**：`torch.max(torch.abs(y_perturbed - y_baseline) / (torch.abs(y_baseline) + 1e-8))`
- **Relative Delta Mean**：`torch.mean(torch.abs(y_perturbed - y_baseline) / (torch.abs(y_baseline) + 1e-8))`
- **放大系数**：`Delta_output / Delta_input`（输出变化量 / 输入变化量）

### 4.4 代码修改

**新增文件**：

| 文件路径 | 用途 |
|----------|------|
| `scripts/analysis/operator_diff_test.py` | RQ1 算子级差分测试主脚本 |
| `scripts/analysis/operator_metamorphic_test.py` | RQ2 算子级蜕变测试主脚本 |
| `utils/rq1/operator_registry.py` | 算子注册表，定义 20 个算子的 PTA/MSA 实例化方式 |

**operator_registry.py 核心设计**：
```python
OPERATOR_REGISTRY = {
    "linear": {
        "pta": lambda in_f, out_f, bias: torch.nn.Linear(in_f, out_f, bias=bias),
        "msa": lambda in_c, out_c, bias: mindspore.nn.Dense(in_c, out_c, has_bias=bias),
        "variants": ["qkv", "proj", "fc1", "fc2", "output", "expert"],
    },
    "matmul": {
        "pta": lambda a, b: torch.matmul(a, b),
        "msa": lambda a, b: mindspore.ops.matmul(a, b),
        "variants": ["q_kt", "score_v"],
    },
    "rope": {
        "pta": lambda cfg: RotaryEmbedding(...),
        "msa": lambda cfg: RotaryEmbedding(...),
    },
    "softmax": {
        "pta": lambda: torch.nn.Softmax(dim=-1),
        "msa": lambda: mindspore.nn.Softmax(axis=-1),
    },
    "dropout": {
        "pta": lambda p: torch.nn.Dropout(p=p),
        "msa": lambda p: mindspore.nn.Dropout(keep_prob=1.0 - p),
    },
    "layernorm": {
        "pta": lambda shape: torch.nn.LayerNorm(shape, eps=1e-5),
        "msa": lambda shape: mindspore.nn.LayerNorm(shape, epsilon=1e-5),
    },
    "cross_entropy": {
        "pta": lambda input, target: F.cross_entropy(input, target),
        "msa": lambda input, target: mindspore.ops.cross_entropy(input, target),
    },
    "gelu": {
        "pta": lambda: torch.nn.GELU(approximate='tanh'),
        "msa": lambda: mindspore.nn.GELU(approximate=True),
    },
    "silu": {
        "pta": lambda: torch.nn.SiLU(),
        "msa": lambda: mindspore.nn.SiLU(),
    },
    "add": {
        "pta": lambda a, b: torch.add(a, b),
        "msa": lambda a, b: mindspore.ops.add(a, b),
    },
    "mul": {
        "pta": lambda a, b: torch.mul(a, b),
        "msa": lambda a, b: mindspore.ops.mul(a, b),
    },
    # ... 其余算子
}
```

---

## 五、组件层实验

### 5.1 组件清单（去重后，一个不能少）

以下组件覆盖 11 个大语言模型中**所有实际存在的组件块**。

| 编号 | 组件名称 | 包含的算子序列 | 输入形状 | 输出形状 | 使用模型 |
|------|---------|--------------|----------|----------|----------|
| 11 | **Embedding Layer** | word_embeddings + position_embeddings [+ top_query_embeddings] + embedding_dropout | `(B, S)` (input_ids) | `(B, S, H)` | 11 个 |
| 12 | **Self-Attention Block** | input_layernorm → linear_qkv → core_attention → linear_proj → dropout → **残差 Add** | `(B, S, H)` | `(B, S, H)` | 11 个 |
| 13 | **FFN Block** | pre_mlp_layernorm → linear_fc1 → [GELU/SwiGLU] → linear_fc2 → dropout → **残差 Add** | `(B, S, H)` | `(B, S, H)` | 8 个标准模型 |
| 14 | **Decoder Block** | Self-Attention Block → FFN/MoE Block（含两个残差） | `(B, S, H)` | `(B, S, H)` | 11 个 |
| 15 | **Output Layer** | ColumnParallelLinear (output_layer) | `(B, S, H)` | `(B, S, V)` | 11 个 |
| 16 | **MoE FFN Block** | pre_mlp_layernorm → router → top-k gating → expert computation → linear_fc2 → **残差 Add** | `(B, S, H)` | `(B, S, H)` | DeepSeekV3, Mixtral, Grok1 |
| 17 | **MLA Self-Attention Block** | input_layernorm → MLA_Q_proj → MLA_KV_proj → core_attention → linear_proj → **残差 Add** | `(B, S, H)` | `(B, S, H)` | DeepSeekV3（生产配置） |

**注**：
- 组件 12 的完整定义包括**残差连接**：`output = input + attention_output`。
- 组件 13 的完整定义包括**残差连接**：`output = input + mlp_output`。
- 组件 14（Decoder Block）在标准模型中包含 SA Block + FFN Block；在 MoE 模型中包含 SA Block + MoE FFN Block。
- 组件 16（MoE FFN Block）替换普通 FFN Block，仅用于 3 个 MoE 模型。
- 组件 17（MLA SA Block）替换普通 SA Block，仅用于 DeepSeekV3 生产配置；测试配置中禁用。

### 5.2 组件层输入策略

组件层实验采用**随机输入张量**，由 `TensorManager` 按确定性种子生成，与算子层一致。无需复用模型级记录的真实输入。

**模型级记录流程**（模型层执行时完成，仅用于模型级分析）：

```python
# utils/component_input_recorder.py
import torch
from pathlib import Path

class ComponentInputRecorder:
    """在整网模型前向传播过程中，记录每个组件的输入张量"""

    def __init__(self, model, save_dir: str, iteration: int, backend: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.iteration = iteration
        self.backend = backend
        self.hooks = []
        self._register_hooks(model)

    def _save(self, name: str, tensor: torch.Tensor):
        filepath = self.save_dir / f"{name}_iter_{self.iteration:04d}_{self.backend}.pt"
        torch.save(tensor.cpu(), filepath)

    def _register_hooks(self, model):
        # 11: Embedding Layer 输入
        self.hooks.append(
            model.embedding.register_forward_hook(
                lambda m, inp, out: self._save("component_11_embedding_input", inp[0])
            )
        )

        # 12: Self-Attention Block 输入（每个 decoder layer）
        for idx, layer in enumerate(model.decoder_layers):
            self.hooks.append(
                layer.input_layernorm.register_forward_hook(
                    lambda m, inp, out, i=idx: self._save(
                        f"component_12_sa_block_layer{i}_input", inp[0]
                    )
                )
            )

        # 13: FFN Block 输入（每个 decoder layer）
        for idx, layer in enumerate(model.decoder_layers):
            self.hooks.append(
                layer.pre_mlp_layernorm.register_forward_hook(
                    lambda m, inp, out, i=idx: self._save(
                        f"component_13_ffn_block_layer{i}_input", inp[0]
                    )
                )
            )

        # 14: Decoder Block 输入（每个 decoder layer）
        for idx, layer in enumerate(model.decoder_layers):
            self.hooks.append(
                layer.register_forward_hook(
                    lambda m, inp, out, i=idx: self._save(
                        f"component_14_decoder_block_layer{i}_input", inp[0]
                    )
                )
            )

        # 15: Output Layer 输入
        self.hooks.append(
            model.output_layer.register_forward_hook(
                lambda m, inp, out: self._save("component_15_output_layer_input", inp[0])
            )
        )

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
```

**保存路径**：
```
res/component_inputs/
├── component_11_embedding_input_iter_0001_pta.pt
├── component_11_embedding_input_iter_0001_msa.pt
├── component_12_sa_block_layer0_input_iter_0001_pta.pt
├── component_12_sa_block_layer0_input_iter_0001_msa.pt
├── component_13_ffn_block_layer0_input_iter_0001_pta.pt
├── ...
```

### 5.3 RQ1 差分测试（组件级）

**执行流程**：
1. 使用 `TensorManager` 生成**随机输入张量**（seed=42 + 组件编号 + 迭代编号作为确定性种子）
2. **关键：PTA 和 MSA 两端必须使用完全相同的随机输入张量**（共享同一 seed 和生成参数）
3. PTA 后端实例化组件，加载权重，执行前向 → `output_pta`
4. MSA 后端实例化同一组件，使用**完全相同输入**，加载权重，执行前向 → `output_msa`
5. 计算差异指标

**差异指标**（同算子层）：
- Max Absolute Difference
- Mean Absolute Difference
- Mean Relative Error
- Max Relative Error
- L2 Norm

### 5.4 RQ2 蜕变测试（组件级）

**执行流程**：
1. 使用 `TensorManager` 生成随机组件输入张量 `x`
2. 基线：同一后端执行组件 → `y_baseline`
3. 变异：输入 `x' = x + epsilon`，`epsilon ~ N(0, sigma^2)`
4. 同一后端执行组件 → `y_perturbed`
5. 计算传播差异指标

**传播指标**（同算子层）：
- Max Absolute Delta
- Mean Absolute Delta
- Relative Delta Max / Mean
- 放大系数

### 5.5 代码修改

**修改现有文件**：

| 文件路径 | 修改内容 |
|----------|----------|
| `core/subgraph.py` | 扩展 `submodule_num`（或新增 `component_num`）支持 11~17；`forward()` 新增组件块执行逻辑 |
| `utils/task/task1.py` | 在整网模型创建后注入 `ComponentInputRecorder`，记录组件输入张量 |
| `genconf.py` | 扩展 Task2 的 SUBMODULES 取值范围：`min_value=0, max_value=17` |

**新增文件**：

| 文件路径 | 用途 |
|----------|------|
| `utils/component_input_recorder.py` | 整网模型前向钩子，记录每个组件的输入张量 |
| `scripts/analysis/component_diff_test.py` | RQ1 组件级差分测试主脚本 |
| `scripts/analysis/component_metamorphic_test.py` | RQ2 组件级蜕变测试主脚本 |

**core/subgraph.py 修改示例**：
```python
# 在 forward() 方法中扩展
if component_num == 12:  # Self-Attention Block
    residual = hidden_states
    hidden_states = self.block.input_layernorm(hidden_states)
    hidden_states = self.block.self_attention(hidden_states)
    hidden_states = self.block.linear_proj(hidden_states)
    hidden_states = self.block.attention_dropout(hidden_states)
    hidden_states = hidden_states + residual  # 残差连接
    return hidden_states

elif component_num == 13:  # FFN Block
    residual = hidden_states
    hidden_states = self.block.pre_mlp_layernorm(hidden_states)
    hidden_states = self.block.mlp(hidden_states)
    hidden_states = hidden_states + residual  # 残差连接
    return hidden_states

elif component_num == 14:  # Decoder Block (完整 SA + FFN)
    # Self-Attention Block
    residual = hidden_states
    hidden_states = self.block.input_layernorm(hidden_states)
    hidden_states = self.block.self_attention(hidden_states)
    hidden_states = self.block.linear_proj(hidden_states)
    hidden_states = hidden_states + residual
    # FFN Block
    residual = hidden_states
    hidden_states = self.block.pre_mlp_layernorm(hidden_states)
    hidden_states = self.block.mlp(hidden_states)
    hidden_states = hidden_states + residual
    return hidden_states

elif component_num == 11:  # Embedding Layer
    hidden_states = self.embedding(hidden_states)
    return hidden_states

elif component_num == 15:  # Output Layer
    logits = self.output_layer(hidden_states)
    return logits

elif component_num == 16:  # MoE FFN Block
    residual = hidden_states
    hidden_states = self.block.pre_mlp_layernorm(hidden_states)
    hidden_states = self.block.moe_layer(hidden_states)  # MoE 路由 + 专家计算
    hidden_states = hidden_states + residual
    return hidden_states

elif component_num == 17:  # MLA Self-Attention Block
    residual = hidden_states
    hidden_states = self.block.input_layernorm(hidden_states)
    hidden_states = self.block.mla_self_attention(hidden_states)  # MLA Q/KV proj + core_attention
    hidden_states = self.block.linear_proj(hidden_states)
    hidden_states = hidden_states + residual
    return hidden_states
```

---

## 六、模型层实验

### 6.1 RQ1 差分测试（模型级）

**直接使用 Task 1（pta_msa 模式）**，无需新增代码。

**执行流程**：
1. 固定随机种子，确保 PTA 和 MSA 从完全相同的初始权重开始
2. PTA 后端执行完整模型训练 `SAVE_STEPS` 步
3. MSA 后端执行完整模型训练 `SAVE_STEPS` 步
4. 使用**相同的真实训练数据**（共享同一 dataloader，确保跨后端输入一致）
5. 记录每步 loss 和中间结果

**中间结果记录**：
- 每步 loss 保存到 `res/training_log_pta/training_log-{iter}.csv` 和 `res/training_log_msa/training_log-{iter}.csv`
- 组件输入张量保存到 `res/component_inputs/`（由 `ComponentInputRecorder` 完成）

**差异指标**：
- **Loss Max Diff**：`max(|loss_pta - loss_msa|)`
- **Loss Mean Diff**：`mean(|loss_pta - loss_msa|)`
- **Loss Relative Diff**：`mean(|loss_pta - loss_msa| / |loss_pta|)`
- **Loss Curve L2**：`||loss_curve_pta - loss_curve_msa||_2`
- **Final Loss Gap**：`|loss_pta[final] - loss_msa[final]|`

### 6.2 RQ2 蜕变测试（模型级）

**基于 Task 1，启用单一后端模式**。

**执行流程**：
1. 基线：默认配置，单一后端（MSA 或 PTA）执行 `SAVE_STEPS` 步 → `loss_baseline`
2. 变异：输入数据添加微小扰动 `x' = x + epsilon`
3. 同一后端执行 `SAVE_STEPS` 步 → `loss_perturbed`
4. 计算传播差异

**传播指标**：
- **Loss Delta Max**：`max(|loss_perturbed - loss_baseline|)`
- **Loss Delta Mean**：`mean(|loss_perturbed - loss_baseline|)`
- **Loss Relative Delta**：`mean(|loss_perturbed - loss_baseline| / |loss_baseline|)`
- **Loss Divergence L2**：`||loss_perturbed - loss_baseline||_2`

### 6.3 代码修改

**修改现有文件**：

| 文件路径 | 修改内容 |
|----------|----------|
| `utils/task/task1.py` | 添加 `SINGLE_BACKEND_MODE` 配置；启用时跳过 MSA/MF，仅保留 PTA 执行 |
| `msm_replace/new_training.py` | 在训练循环中支持保存中间 loss 张量（用于蜕变测试） |
| `utils/analyze/precision.py` | 新增 `compute_metamorphic_delta(baseline_csv, perturbed_csv)` 函数 |

---

## 七、三个层次的执行关系与数据流转

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             模型层（Task 1）                                  │
│  整网模型执行（PTA + MSA）                                                      │
│       │                                                                      │
│       ├── 注册 ComponentInputRecorder 前向钩子                                │
│       │       └── 记录每个组件的输入张量                                       │
│       │               res/component_inputs/                                    │
│       │               ├── component_11_embedding_input_iter_0001_pta.pt       │
│       │               ├── component_12_sa_block_layer0_input_iter_0001_pta.pt  │
│       │               ├── component_13_ffn_block_layer0_input_iter_0001_pta.pt │
│       │               └── ...                                                  │
│       │                                                                      │
│       ├── PTA 整网执行 → loss_pta_curve                                       │
│       └── MSA 整网执行 → loss_msa_curve                                       │
│               │                                                              │
│               ▼                                                              │
│       RQ1 差分：|loss_pta - loss_msa|                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ 真实训练数据（模型层）
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             组件层（Task 2 扩展）                              │
│  单独执行组件块                                                                │
│       │                                                                      │
│       ├── TensorManager 生成随机输入张量                                     │
│       ├── PTA 执行组件 → output_pta                                          │
│       └── MSA 执行组件 → output_msa                                          │
│               │                                                              │
│               ▼                                                              │
│       RQ1 差分：max(|output_pta - output_msa|)                               │
│       RQ2 蜕变：输入加扰动 → 对比扰动前后输出                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ 随机输入张量（TensorManager）
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             算子层（新增脚本）                                 │
│  单独执行单个算子                                                              │
│       │                                                                      │
│       ├── TensorManager 生成确定性输入                                        │
│       ├── PTA 执行算子 → output_pta                                          │
│       └── MSA 执行算子 → output_msa                                          │
│               │                                                              │
│               ▼                                                              │
│       RQ1 差分：max(|output_pta - output_msa|)                               │
│       RQ2 蜕变：输入加扰动 → 对比扰动前后输出                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 八、代码修改汇总表

### 新增文件

| 文件路径 | 用途 | 所属层次 |
|----------|------|----------|
| `scripts/analysis/operator_diff_test.py` | RQ1 算子级差分测试主脚本 | 算子层 |
| `scripts/analysis/operator_metamorphic_test.py` | RQ2 算子级蜕变测试主脚本 | 算子层 |
| `utils/rq1/operator_registry.py` | 20 个算子的 PTA/MSA 注册表 | 算子层 |
| `utils/component_input_recorder.py` | 整网模型前向钩子，记录组件输入张量 | 组件层 + 模型层 |
| `scripts/analysis/component_diff_test.py` | RQ1 组件级差分测试主脚本 | 组件层 |
| `scripts/analysis/component_metamorphic_test.py` | RQ2 组件级蜕变测试主脚本 | 组件层 |

### 修改现有文件

| 文件路径 | 修改内容 | 所属层次 |
|----------|----------|----------|
| `core/subgraph.py` | 扩展 `submodule_num` 支持 11~17；`forward()` 新增组件块执行逻辑 | 组件层 |
| `utils/task/task1.py` | 注入 `ComponentInputRecorder` 前向钩子；添加 `SINGLE_BACKEND_MODE` | 模型层 |
| `genconf.py` | 扩展 Task2 的 SUBMODULES 取值范围至 0~17 | 组件层 |
| `msm_replace/new_training.py` | 支持保存中间 loss 张量 | 模型层 |
| `utils/analyze/precision.py` | 新增 `compute_metamorphic_delta` 函数 | 模型层 |

### 不修改现有文件（直接复用）

| 功能 | 现有支持 | 所属层次 |
|------|----------|----------|
| 确定性张量生成 | `utils/tensor_manager.py` | 三层共用 |
| 模型级差分测试 | `utils/task/task1.py` pta_msa 模式 | 模型层 |
| Loss 曲线差异分析 | `utils/analyze/precision.py` | 模型层 |
| 子模块执行机制 | `core/subgraph.py` 现有 0~10 | 组件层（基础） |

---

## 九、指标定义汇总

### RQ1 差分测试指标（误差产生）

| 指标 | 公式 | 适用层次 |
|------|------|----------|
| Max Absolute Diff | `max(|output_pta - output_msa|)` | 算子 / 组件 |
| Mean Absolute Diff | `mean(|output_pta - output_msa|)` | 算子 / 组件 |
| Mean Relative Error | `mean(|pta - msa| / (|pta| + eps))` | 算子 / 组件 |
| Max Relative Error | `max(|pta - msa| / (|pta| + eps))` | 算子 / 组件 |
| L2 Norm | `||pta - msa||_2` | 算子 / 组件 |
| ULP Distance | 浮点 ULP 差 | 算子（FP16/BF16） |
| Loss Max Diff | `max(|loss_pta - loss_msa|)` | 模型 |
| Loss Mean Diff | `mean(|loss_pta - loss_msa|)` | 模型 |
| Loss Relative Diff | `mean(|loss_pta - loss_msa| / |loss_pta|)` | 模型 |

### RQ2 蜕变测试指标（误差传播）

| 指标 | 公式 | 适用层次 |
|------|------|----------|
| Max Absolute Delta | `max(|y_perturbed - y_baseline|)` | 算子 / 组件 / 模型 |
| Mean Absolute Delta | `mean(|y_perturbed - y_baseline|)` | 算子 / 组件 / 模型 |
| Relative Delta Max | `max(|delta| / (|y_baseline| + eps))` | 算子 / 组件 / 模型 |
| Relative Delta Mean | `mean(|delta| / (|y_baseline| + eps))` | 算子 / 组件 / 模型 |
| 放大系数 | `Delta_output / Delta_input` | 算子 / 组件 |
| Loss Delta Max | `max(|loss_perturbed - loss_baseline|)` | 模型 |
| Loss Divergence L2 | `||loss_perturbed - loss_baseline||_2` | 模型 |
