# FrameDiff FullNet

This directory contains the language-model full-network differential chain extracted and simplified for the FrameDiff paper workflow. It keeps model selection, prepared variant replay, shared-weight loading, and the PTA-MSA differential path, while removing the delivery-project shell, WebUI, multimodal workflows, historical outputs, unrelated task entrypoints, third-framework replay, and multi-machine orchestration.

## Quick Start

List available language-model presets:

```bash
cd fullnet
python fullnet.py models
```

Create a config for one language model:

```bash
python fullnet.py init --models qwen2
```

Check paths and selected models:

```bash
python fullnet.py doctor
```

Run the full-network chain:

```bash
python fullnet.py run
```

Preview the final runtime config without launching training:

```bash
python fullnet.py run --models qwen2 --perturb-eps 1e-5 --dry-run
```

Rebuild the lightweight paper report for the latest run:

```bash
python fullnet.py analyze --latest
```

## Paper-Oriented Path

FrameDiff uses this slice for language-model full-network precision experiments:

1. Read every prepared variant under `../mutated_config/<model>/`, with `ancestor` first.
2. Run `prepare` only for `ancestor` to produce the shared weight.
3. Run `pta-baseline` and `msa-baseline` for every variant using the same `ancestor` shared weight.
4. Check PTA/MSA baseline loss alignment before collecting perturbation data.
5. Run `pta-preturb` and `msa-preturb` with a one-way `+eps` tensor perturbation.
6. Write public tensor artifacts under `output` and keep scripts, logs, CSVs, variant inputs, trace indexes, summaries, and analysis artifacts under `records`.

The public config keeps PTA/MSA paths, output root, selected models, iteration count, load-step count, `PERTURB_EPS`, and baseline loss tolerance. There is no runtime mutation step: each variant supplies `mutating.json` + `mutated_config.yaml`. Public tensor artifacts are kept separate from trace indexes and runtime records.

RQ3 variants are pre-generated under `../mutated_config/<model>/<variant>/`, with `ancestor` and all enabled variants as sibling directories. `mutating.json` is both experiment metadata and the source for runtime overrides; it keeps the legacy numeric node records required by `Graph.load()` and adds `runtime_overrides` for environment variables, GPT runtime args, launcher parallelism, optimizer knobs, and per-variant runtime controls. The generated manifests record common variants plus model-specific MoE, MLA, GQA, MLP fusion, and structure variants, along with skipped variants that are not applicable to that model.

## RQ3 Prepared Variant Matrix

This matrix is generated from the actual prepared variant directories under `../mutated_config/<model>/<variant>/` and the per-model manifests. `Y` means the prepared variant exists for the model; `-` means the candidate is skipped or disabled for that model by the current generator constraints. The `ancestor` baseline is not shown as a matrix row.

Generated at: `2026-05-30T12:23:29`. Total prepared inputs including `ancestor`: **546** across **12** models. Candidate variants in generator scope: **69**.

| Model | Prepared variants incl. ancestor | Enabled variants | Skipped/disabled candidates |
|:---|:---:|:---:|:---:|
| `baichuan2` | 41 | 40 | 29 |
| `chatglm3` | 40 | 39 | 30 |
| `codellama` | 40 | 39 | 30 |
| `deepseekv3` | 67 | 66 | 3 |
| `glm4` | 41 | 40 | 29 |
| `grok1` | 59 | 58 | 11 |
| `llama2` | 38 | 37 | 32 |
| `mixtral` | 57 | 56 | 13 |
| `pangu` | 42 | 41 | 28 |
| `qwen2` | 40 | 39 | 30 |
| `qwen3` | 40 | 39 | 30 |
| `yi` | 41 | 40 | 29 |

| Variant | `baichuan2` | `chatglm3` | `codellama` | `deepseekv3` | `glm4` | `grok1` | `llama2` | `mixtral` | `pangu` | `qwen2` | `qwen3` | `yi` |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `runtime_flash_attn_on` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `runtime_softmax_fp32_on` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `runtime_masked_softmax_fusion_off` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `mlp_swiglu_fusion_on` | Y | Y | Y | Y | Y | Y | - | Y | Y | Y | Y | Y |
| `mlp_swiglu_fusion_off` | Y | Y | Y | Y | Y | Y | - | Y | Y | Y | Y | Y |
| `runtime_use_fused_rmsnorm_on` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `runtime_use_fused_rotary_pos_emb_on` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `runtime_no_gradient_accumulation_fusion` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `moe_grouped_gemm_on` | - | - | - | Y | - | Y | - | Y | - | - | - | - |
| `moe_use_fused_moe_token_permute_and_unpermute_on` | - | - | - | Y | - | Y | - | Y | - | - | - | - |
| `runtime_recompute_activation_function_on` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `runtime_recompute_granularity_full` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `runtime_recompute_method_uniform` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `runtime_recompute_num_layers_1` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `parallel_tp1` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `parallel_tp2` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `parallel_tp4` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `parallel_expert_parallel_2` | - | - | - | Y | - | Y | - | Y | - | - | - | - |
| `parallel_pipeline_parallel_2` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `parallel_data_parallel_2` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `runtime_sequence_parallel_on` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `parallel_context_parallel_2` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `moe_aux_loss_coeff_0` | - | - | - | Y | - | Y | - | Y | - | - | - | - |
| `moe_aux_loss_coeff_1e_2` | - | - | - | Y | - | Y | - | Y | - | - | - | - |
| `config_moe_device_level_aux_loss_coeff_0_03` | - | - | - | Y | - | Y | - | Y | - | - | - | - |
| `config_moe_comm_aux_loss_coeff_0_01` | - | - | - | Y | - | Y | - | Y | - | - | - | - |
| `config_seq_aux_on` | - | - | - | Y | - | Y | - | Y | - | - | - | - |
| `train_lr_1e_3` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `train_weight_decay_0` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `config_layernorm_eps_1e_7` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `precision_bf16_on` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `precision_fp16_on` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `precision_accumulate_grads_fp32_on` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `precision_fp32_residual_on` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `runtime_use_distributed_optimizer_on` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `runtime_reuse_fp32_param_on` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `config_normalization_layernorm` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `config_qk_layernorm_on` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `config_embedding_multiplier_scale_78_38` | - | - | - | - | - | Y | - | - | - | - | - | - |
| `config_output_multiplier_scale_0_57` | - | - | - | - | - | Y | - | - | - | - | - | - |
| `config_first_k_dense_replace_0` | - | - | - | Y | - | Y | - | Y | - | - | - | - |
| `config_moe_layer_freq_2` | - | - | - | Y | - | Y | - | Y | - | - | - | - |
| `config_n_shared_experts_2` | - | - | - | Y | - | Y | - | Y | - | - | - | - |
| `config_num_moe_experts_8` | - | - | - | Y | - | Y | - | Y | - | - | - | - |
| `config_moe_intermediate_size_1536` | - | - | - | Y | - | Y | - | Y | - | - | - | - |
| `moe_router_load_balancing_none` | - | - | - | Y | - | Y | - | Y | - | - | - | - |
| `moe_router_load_balancing_aux_loss` | - | - | - | Y | - | Y | - | Y | - | - | - | - |
| `moe_router_pre_softmax_on` | - | - | - | Y | - | Y | - | Y | - | - | - | - |
| `moe_router_pre_softmax_off` | - | - | - | Y | - | Y | - | Y | - | - | - | - |
| `config_input_jitter_on` | - | - | - | Y | - | - | - | - | - | - | - | - |
| `linear_bias_on` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `linear_bias_off` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `attention_qkv_bias_on` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `attention_qkv_bias_off` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `config_position_embedding_learned` | - | - | - | - | - | - | - | - | - | - | - | - |
| `mla_qk_head_dim_scaled_0_75` | - | - | - | Y | - | - | - | - | - | - | - | - |
| `mla_v_head_dim_scaled_0_75` | - | - | - | Y | - | - | - | - | - | - | - | - |
| `mla_q_lora_rank_scaled_0_5` | - | - | - | Y | - | - | - | - | - | - | - | - |
| `mla_kv_lora_rank_scaled_0_5` | - | - | - | Y | - | - | - | - | - | - | - | - |
| `mla_qk_rope_head_dim_128` | - | - | - | Y | - | - | - | - | - | - | - | - |
| `mla_qk_nope_head_dim_128` | - | - | - | Y | - | - | - | - | - | - | - | - |
| `mla_kv_channels_128` | - | - | - | Y | - | - | - | - | - | - | - | - |
| `config_num_query_groups_1` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `config_num_query_groups_half` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `config_num_query_groups_equal_heads` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `config_hidden_size_scaled_0_75` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `config_ffn_hidden_size_scaled_0_75` | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| `config_num_attention_heads_half` | - | - | - | Y | - | - | - | - | Y | - | - | - |
| `mlp_gated_linear_unit_toggle` | Y | - | - | Y | Y | - | - | - | Y | - | - | Y |

Notes:

- `config_position_embedding_learned` is intentionally disabled in the prepared-variant generator because the current fullnet script constraints only accept RoPE prepared variants.
- MoE variants are generated only for models detected as MoE (`deepseekv3`, `grok1`, and `mixtral`). MLA variants are generated only for `deepseekv3`.
- Grok-1-only numeric scaling variants and DeepSeekV3-only router jitter are represented directly in the matrix instead of relying on the approximate matrix in `RQ3_param_coverage.md`.


## Key Files

- `fullnet.py`: model selection, config generation, dry-run, run, and analysis CLI.
- `do.py`: direct entry that runs `config.json`.
- `config.json.example`: minimal language-model full-network configuration.
- `fullnet_core/`: CLI/config/runner layer.
- `utils/task/fullnet.py`: single-machine full-network orchestration core.
- `utils/analyze/fullnet_result.py`: lightweight paper-oriented result analysis.
- `../model_config/*.yaml`: supported language model presets shared with the other diff workflows.
- `../mutated_config/<model>/<variant>/`: prepared variant inputs consumed by this workflow.
- `assets/runtime/configs/*.yaml`: graph template still used by the runtime graph builder.
- `assets/runtime/tokenizers/baichuan2/`: the bootstrap tokenizer path used by the inherited PTA/MSA launch scripts.
- `scripts/runtime/submodule_entry.py`: runtime patch wrapper used for PTA/MSA graph execution.

## Outputs

Public DOI artifacts are archived under `output/<model>/<variant>/<training>/`, for example `output/qwen2/ancestor/pta-baseline/*.pt`. Shared ancestor weights are copied to `output/<model>/shared_weight.pth`. Runtime logs, scripts, copied configs, CSVs, trace indexes, summaries, and analysis files are written under the sibling `records/` tree.

The analysis step writes:

- `records/analysis/data/summary.json`
- `records/analysis/summary.md`
- `records/analysis/report.html`
