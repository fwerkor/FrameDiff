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
