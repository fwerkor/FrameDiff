# 论文工作汇报：框架层精度误差的系统性实证研究

**汇报人：邹英龙**

---

## 一、研究背景与科学问题

大规模深度学习模型训练时，同一模型在不同框架后端上执行，即使功能完全正确，也常出现数值精度不一致。这种差异不触发任何错误信息，属于"静默精度漂移"，比功能缺陷更难检测和定位。

框架层精度误差的产生贯穿从底层算子到顶层模型的完整映射链：算子实现差异（如 LayerNorm 的数值稳定性处理不同）、框架优化差异（算子融合改变累加顺序）、分布式运行时差异（AllReduce 分块大小影响累加顺序）、混合精度策略差异（FP16 回退到 FP32 的触发条件不同）。同一模型在两种框架后端上以完全相同的配置训练，loss 差异可达到显著水平，却不会产生任何报错信息。

现有研究存在三个根本性的方法论缺失：

**（1）产生与传播的测量方法未被区分。** 误差的"产生"（不同后端对同一算子的实现差异）和"传播"（底层误差在多层网络中如何演变）是两个不同的研究对象，需要不同的测量方法。现有工作未对这一概念区分给予足够重视，导致测量方法与研究问题不匹配。Predoo [1] 仅覆盖 7 个 TensorFlow 算子的算子级差分测试，未连接到上层分析；CRADLE [2] 在模型级别做差分验证，但精度误差被多层累积淹没，无法追溯根因。ANPRED [28] 揭示了阶段集中性和非单调传播特征，但未明确采用蜕变测试范式，产生与传播的测量方法仍未被严格区分。

**（2）缺乏传播路径的刻画方法。** 算子级的微小数值差异在向上传播过程中可能经历叠加、放大或衰减，但没有任何现有工作提供系统性的方法论来刻画这一传播过程，尤其未覆盖训练阶段的大语言模型和多模态模型。

**（3）缺乏配置因素归因的量化方法。** 现有框架对比研究往往同时改变多个变量（后端、配置、硬件），缺乏将误差归因到单个因素的方法论，只能给出定性描述，无法提供定量度量。

ANPRED [28]（ASE 2026）提出基于锚点的精度测试框架，在模型执行阶段边界插入观测锚点，解决了输出级差分测试无法观察内部误差的问题。然而，ANPRED 仅覆盖推理阶段和 CV 模型（YOLO），未涉及训练阶段、大语言模型和多模态模型，也未做产生源归因和配置因素量化。

**核心科学问题**：在深度学习框架生态碎片化的背景下，如何系统性地分解、测量和量化框架层精度误差的产生机制、传播规律和影响因素？

---

## 二、研究问题（RQ）的完备性

围绕核心科学问题，设计三个递进式研究问题，形成"产生→传播→影响"的完整研究链条。

### RQ1：误差的产生

框架层精度误差无法直接从最终模型输出中定位根因。算子级测试可以识别最底层的误差来源，但孤立且规模小；组件级测试可以观测中间层输出的跨后端差异，但现有代码缺乏该能力；模型级观测可以捕获最终用户可见的结果，但黑盒且无法追溯根因。

RQ1 提出**差分测试方法**：在参考后端和对比后端上，使用完全相同的输入数据和模型权重，分别执行同一算子、同一组件或同一模型，比较输出差异。差分测试直接测量"误差的产生"，不引入任何配置变异，确保测量的是"纯框架实现差异"。

| 子实验 | 粒度 | 核心内容 |
|--------|------|---------|
| 1.1 | 算子级 | 核心算子（MatMul、Softmax、LayerNorm、RMSNorm、GELU、SiLU、RoPE、FlashAttention 等）在跨后端下的输出差异基准 |
| 1.2 | 组件级 | 关键组件（Self-Attention Block、FFN Block、LayerNorm 等）的中间输出差异，逐层误差累积曲线 |
| 1.3 | 模型级（语言） | Dense Transformer 和 MoE 模型在跨后端下的完整 loss 曲线差异 |
| 1.4 | 模型级（多模态） | 多模态模型在跨后端下的训练/推理 loss 差异 |
| 1.5 | 模态级 | 视觉分支、语言分支、完整模型的误差分解 |

### RQ2：误差的传播

算子级的微小数值差异在向上传播过程中可能经历叠加、放大或衰减。在训练阶段的大语言模型和多模态模型中，没有任何现有研究提供方法论来刻画这一传播过程。

RQ2 提出**蜕变测试方法**：在单一后端内，对算子输入、组件配置或模型配置引入可控变异，比较"变异前"与"变异后"的输出变化。通过比较"在不同粒度引入相同强度变异"的效果差异，推断误差传播规律。蜕变测试刻画的是"扰动影响的传播"，从而间接推断"精度误差的传播特性"。

现有框架（Task1/Task2/Task4/Task5）的现有实现均执行跨后端比较，不支持单一后端内的基线-变异对比。为实现蜕变测试，需要扩展这些任务以支持单一后端模式。

| 子实验 | 粒度 | 核心内容 |
|--------|------|---------|
| 2.1 | 算子级 | 单个算子对输入微扰的放大/衰减特性（Δ_op） |
| 2.2 | 组件级 | 同一组件在不同位置（浅层/中层/深层）变异后的传播敏感度差异 |
| 2.3 | 模型级 | 模型级配置（batch size、sequence length、learning rate）变异对最终输出的端到端影响 |
| 2.4 | 综合 | 基于 RQ1 和 RQ2 已有数据，计算算子和组件的误差贡献度，识别关键传播节点 |
| 2.5 | 跨模态 | 视觉组件变异对语言分支输出的影响，跨模态误差传递比例 |

### RQ3：配置因素的影响

多种配置因素可能同时影响精度误差，但现有研究混杂变量无法归因。

RQ3 采用**单因素控制实验方法**：在保持模型、数据、后端和其他配置完全相同的前提下，逐一改变一个配置因素，通过差分测试测量该因素对跨后端精度差异的影响。

| 子实验 | 因素类别 | 具体内容 |
|--------|---------|---------|
| 3.1 | 框架运行时优化机制 | FlashAttention、Fused RMSNorm、Fused SwiGLU、Attention Softmax FP32、Sequence Parallel 的独立影响 |
| 3.2 | 分布式并行策略 | TP（1, 2, 4, 8）、PP（1, 2, 4）、EP（1, 2, 4, 8）的系统改变 |
| 3.3 | 训练超参数 | batch size、sequence length、learning rate |
| 3.4 | 数值精度格式 | FP16 / BF16 / FP32 |
| 3.5 | 确定性控制 | 启用/禁用确定性模式对跨后端误差可复现性的影响 |

**诚实声明**：RQ3 测量的是配置因素与跨后端精度差异的**相关性**，而非严格的因果关系。无法独立控制框架内部的精度误差源。

---

## 三、方法论创新

### 创新一：严格分离"产生"与"传播"的实证方法论

现有工作未对"误差的产生"和"误差的传播"给予方法论层面的区分。Predoo 只做算子级差分测试（产生），CRADLE 只做模型级差分测试（产生），ANPRED 虽揭示了传播特征但未明确采用蜕变测试范式。产生和传播的混淆导致测量方法与研究问题不匹配。

本文明确区分差分测试（用于测量误差的产生）和蜕变测试（用于刻画误差的传播）：在算子级、组件级和模型级通过差分测试建立误差基准；在传播维度通过蜕变测试从算子级、组件级和模型级三个粒度揭示传播规律。两种方法论严格分离，互不混杂。

### 创新二：误差传播规律的刻画方法

没有任何现有工作提供方法论来刻画训练阶段大语言模型和多模态模型中的误差传播路径。ANPRED 揭示了阶段集中性和非单调传播特征，但未提供从算子到模型的完整传播图谱构建方法，也未覆盖训练阶段。

本文从位置敏感度、关键节点、跨模态传播三个维度构建误差传播规律的分析方法：通过在不同位置引入相同配置变异推断传播趋势，通过贡献度排序识别关键节点，通过跨模态变异量化多模态架构特有的传播风险。

### 创新三：单因素控制实验方法

现有研究同时改变多个变量，只能给出定性描述（如"并行策略会影响精度"），无法定量刻画各因素的独立贡献。

本文在统一基准条件下逐一改变单一配置因素，通过差分测试将误差贡献度从定性描述变为定量度量。通过 ANOVA 和多元线性回归计算各因素的相关性贡献度和权重排序（明确声明不做因果推断）。

---

## 四、实验框架与代码复用

### 现有代码复用情况

| 功能 | 现有支持 | 复用方式 |
|------|---------|---------|
| 模型级差分测试（RQ1.3，语言模型） | Task1 pta_msa | 直接复用，无需修改 |
| 模型级差分测试（RQ1.4，多模态模型） | Task6 pta_msa | 直接复用，无需修改 |
| 组件级位置敏感度（RQ2.2） | Task2 SUBMODULES | 直接复用，需添加单一后端模式 |
| 并行策略配置（RQ3.2） | Task1 TARGET_TENSOR_PARALLEL_SIZE | 直接复用，无需修改 |
| 脚本参数修改（RQ3.1, RQ3.3, RQ3.4） | Task1 `modify_script_param` | 直接复用，无需修改 |
| loss 曲线差异分析 | `precision.py` | 直接复用，无需修改 |
| 误差贡献度计算 | `precision.py` `compute_error_contribution_score` | 直接复用，无需修改 |

### 需要新增的代码

| 新增功能 | 说明 |
|---------|------|
| 算子级差分测试脚本 | RQ1.1 算子级差分测试；RQ2.1 算子级传播测试（perturbation 模式） |
| 组件级前向钩子模块 | RQ1.2 组件级中间输出提取（`utils/rq1/component_hook.py`） |
| 组件级输出比较脚本 | RQ1.2 跨后端中间输出差异分析 |
| Task1 单一后端模式 | RQ2.3 模型级传播测试 |
| Task2/4/5 单一后端模式 | RQ2 组件级和多模态传播测试 |
| 模态消融执行逻辑 | RQ1.5 多模态模态级误差分解 |
| 后处理分析脚本 | RQ1/RQ2/RQ3 结果分析 |

---

## 五、论文组织结构

| 章节 | 内容 |
|------|------|
| Abstract | 背景、目标、方法、贡献 |
| 1. Introduction | 研究动机、科学问题、RQ（附完备性论证）、Contributions |
| 2. Background | 执行技术栈、精度误差、变异测试、核心术语 |
| 3. Methodology | 总体概述、RQ1/RQ2/RQ3 研究设计、数据分析方法 |
| 4. Results | RQ1/RQ2/RQ3 实验结果（按子问题组织） |
| 5. Discussion | 对框架开发者、模型架构设计者、运行时开发者的启示 |
| 6. Threats to Validity | 内部效度、外部效度、构念效度 |
| 7. Related Work | 八个类别的 SOTA 工作系统性综述 |
| 8. Conclusion | 总结与展望 |

---

## 六、SOTA 工作系统性综述

### 6.1 深度学习系统测试（功能性正确性）

DeepXplore [3]、DeepGauge [4]、DLFuzz [5]、FreeFuzz [6]、EAGLE [7]、Muffin [8]、TitanFuzz [9]、NNSmith [10]、DocTer [11]、DevMuT [12]、FUEL [13] 等工作均聚焦于功能性缺陷（崩溃、NaN、输出不一致），不涉及精度误差的系统性分解与传播分析。

### 6.2 模型级蜕变测试

Y. Mu et al. [14]（ISSTA 2025）提出首个模型级蜕变测试方法 ModelMeta，设计四个 Structure Metamorphic Relations (SMRs)，使用 QR-DQN 指导测试输入生成，检测训练 loss/gradients、memory/GPU usage、execution time 等运行时指标，在 17 个模型上发现 31 个新 bug。其目标是检测框架实现 bug（内存泄漏、效率问题），而非研究精度误差的传播规律。

### 6.3 跨框架对比与验证

CRADLE [2]（ICSE 2019）使用 Keras 的多个后端进行差分测试，但在模型级别操作，浮点误差在多层中累积，难以隔离根因算子级精度缺陷，且需要预训练模型和昂贵的训练。TENSCOPE [15]（USENIX Security 2023）提出跨框架 API 差分测试，在 6 个框架的 1,658 个 API 上发现 257 个 bug。Xamt [16]（ICSE 2025）提出跨框架 API 匹配测试。这些工作均聚焦于功能性正确性，未系统性研究精度误差的产生机制、传播规律和影响因素。

### 6.4 模型压缩与量化

混合精度训练和模型量化 [17, 18] 聚焦主动引入的精度损失，不适用于框架实现差异分析。

### 6.5 深度学习中的数值分析

FP16 训练稳定性和数值稳定性分析 [19, 20] 聚焦算法级数值问题，无法刻画不同框架实现之间的差异。

### 6.6 数值 Bug 实证研究与数据库

Wang et al. [21]（ASE 2022）收集并分析 400 个真实数值 bug，分为 9 类。DeepStability [22]（ICSE 2022）建立数值 bug 数据库。Numerical Instability 2025 [23]（FSE 2025）构建 61 个数值不稳定函数数据库。这些工作揭示了数值问题的普遍性，但未提供分解和量化框架层精度误差的方法论。

### 6.7 深度学习库精度测试

Predoo [1]（ISSTA 2021）提出面向单个算子的精度误差引导模糊测试，仅覆盖 7 个 TensorFlow 算子，未连接到模型级分析，且在单一框架内测试。Duo [24]（IEEE TR 2021）对算子进行差分模糊测试，也未连接到模型级。

ANPRED [28]（ASE 2026）提出基于锚点的精度测试框架，解决了输出级差分测试无法观察内部误差的问题，揭示了阶段集中性和非单调传播特征。然而，ANPRED 仅覆盖推理阶段和 CV 模型（YOLO），未涉及训练阶段、大语言模型和多模态模型，也未做产生源归因和配置因素量化。

DL Library Testing Survey [25]（TSE 2025）首次全面综述深度学习库测试方法，指出数值精度问题是"研究不足的领域"。

### 6.8 与现有工作的本质区别

| 维度 | 现有工作 | 本文 |
|------|---------|------|
| 研究对象 | 功能性缺陷（崩溃、NaN）或算子级精度 | 框架层精度误差的产生、传播、影响因素 |
| 方法区分 | 未区分产生与传播 | 严格分离差分测试（产生）和蜕变测试（传播）|
| 覆盖范围 | 推理阶段 / CV 模型 / 算子级 | 训练阶段 / LLM+多模态 / 算子级+组件级+模型级 |
| 传播刻画 | ANPRED 揭示特征但未提供完整方法 | 从三个维度构建传播规律分析方法 |
| 因素量化 | 无单因素控制实验 | 逐一改变单一因素，独立量化贡献 |
| 配置因素 | 未覆盖 | 覆盖优化机制、并行策略、超参数、精度格式、确定性 |

---

## 七、关键诚实声明

1. **产生与传播的严格区分**：本文严格区分差分测试（跨后端比较，研究产生）和蜕变测试（同一后端内比较，研究传播）。RQ2 的蜕变测试测量的是配置变异的传播效应，而非精度误差本身的直接传播，在讨论中明确说明这一间接推断的局限性。

2. **因果推断的局限**：RQ3 的单因素控制实验只能建立配置因素与跨后端精度差异的相关性，不能做严格的因果推断。精度误差的根本来源是框架实现差异，这些因素无法独立控制。

3. **实验环境边界**：所有实验在单机八卡环境中完成。结果可能无法直接推广到多机场景或不同硬件平台。

4. **模型选择边界**：实验覆盖主流架构类别（Dense/MoE/多模态），但数量有限。结论的推广性需要在更多模型上验证。

5. **代码修改的诚实性**：RQ2 的传播实验需要扩展现有 Task1/2/4/5 以支持单一后端模式。RQ1.2 的组件级测试需要新增前向钩子模块并修改 Task1 执行流程以注入钩子。RQ1.5 需要扩展 Task6 的模态消融功能。这些修改在实验计划中已详细说明。

---

## 八、参考文献

[1] X. Zhang et al., "Predoo: Precision Testing of Deep Learning Operators," ISSTA, 2021.

[2] H. V. Pham et al., "CRADLE: Cross-Backend Validation to Detect and Localize Bugs in Deep Learning Libraries," ICSE, 2019.

[3] K. Pei et al., "DeepXplore: Automated Whitebox Testing of Deep Learning Systems," SOSP, 2017.

[4] L. Ma et al., "DeepGauge: Multi-Granularity Testing Criteria for Deep Learning Systems," ASE, 2018.

[5] J. Guo et al., "DLFuzz: Differential Fuzzing Testing of Deep Learning Systems," ESEC/FSE, 2018.

[6] A. Wei et al., "FreeFuzz: Fuzzing Deep Learning Libraries via Automated Relational API Inference," ASE, 2022.

[7] Q. Wang et al., "EAGLE: Creating Equivalent Graphs to Test Deep Learning Libraries," ICSE, 2022.

[8] J. Gu et al., "Muffin: Testing Deep Learning Libraries via Neural Architecture Fuzzing," ICSE, 2022.

[9] Y. Deng et al., "TitanFuzz: Fuzzing Deep-Learning Libraries via Large Language Models," ISSTA, 2023.

[10] J. Liu et al., "NNSmith: Generating Diverse and Valid Test Cases for Deep Learning Compilers," ASPLOS, 2023.

[11] X. Zhang et al., "DocTer: Documentation-Guided Fuzzing for Testing Deep Learning API Functions," ISSTA, 2022.

[12] Y. Mu et al., "DevMuT: Testing Deep Learning Framework via Developer Expertise-Based Mutation," arXiv:2507.04360, 2024.

[13] S. Yang et al., "May the Feedback Be with You! Breaking the Seal of Feedback-Driven Deep Learning Framework Fuzzing via Large Language Models," ISSTA, 2025.

[14] Y. Mu et al., "Improving Deep Learning Framework Testing with Model-Level Metamorphic Testing," ISSTA, 2025.

[15] Z. Chen et al., "TENSCOPE: Detecting DL Framework Implementation Bugs via Cross-framework Differential API Fuzzing," USENIX Security, 2023.

[16] Y. Liu et al., "Xamt: Cross-framework API Matching Testing for Deep Learning Libraries," ICSE, 2025.

[17] P. Micikevicius et al., "Mixed Precision Training," ICLR, 2018.

[18] B. Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference," CVPR, 2018.

[19] Y. N. Dauphin et al., "Language Modeling with Gated Convolutional Networks," ICML, 2017.

[20] P. K. Mogensen et al., "Deep Numerical Analysis: Stability, Convergence, and Error Propagation in Neural Networks," JMLR, 2022.

[21] G. Wang et al., "An Empirical Study of Numerical Bugs in Deep Learning Programs," ASE, 2022.

[22] Z. Chen et al., "DeepStability: A Study of Unstable Numerical Methods in Deep Learning," ICSE, 2022.

[23] J. Zhang et al., "Detecting Numerical Instability in Deep Learning Programs," ESEC/FSE, 2025.

[24] S. Dutta et al., "Duo: Differential Fuzzing for Deep Learning Operators," IEEE TR, 2021.

[25] X. Zhang et al., "Testing Deep Learning Libraries: A Comprehensive Survey," TSE, 2025.

[28] Anonymous, "ANPRED: Anchor-based Precision Testing for Deep Learning Frameworks," ASE, 2026.
