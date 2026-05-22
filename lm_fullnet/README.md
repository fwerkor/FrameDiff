# FrameDiff Language FullNet

This directory contains the language-model fullnet chain extracted and simplified for the FrameDiff paper workflow. It keeps the model-selection, mutation, shared-weight replay, and PTA-MSA differential path, while removing the delivery-project shell, WebUI, multimodal workflows, historical outputs, unrelated task entrypoints, third-framework replay, and multi-machine orchestration.

## Quick Start

List available language-model presets:

```bash
cd lm_fullnet
python fullnet.py models
```

Create a config for one language model:

```bash
python fullnet.py init --models qwen2 --iters 3
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
python fullnet.py run --models qwen3 --iters 1 --dry-run
```

Rebuild the lightweight paper report for the latest run:

```bash
python fullnet.py analyze --latest
```

## Paper-Oriented Path

FrameDiff uses this slice for language-model full-network precision experiments:

1. Generate a single-model full-network graph using the selected model's configured decoder depth.
2. Run PTA-SAVE to produce shared weights.
3. Run PTA-LOAD as the same-backend baseline.
4. Run MSA-LOAD with the same graph and weights.
5. When trace is enabled, run PTA/MSA input-perturbation replays for metamorphic data.
6. Archive per-iteration logs, scripts, mutation inputs, step-level loss CSVs, full tensor/weight traces, and compact analysis artifacts.

Set `TRACE.ENABLED=true` in `config.json`, or pass `--trace`, when component-level diagnosis needs full tensor and weight exports. The trace index records 17 paper components, whole-network inputs/outputs, overall loss, mutation metadata, PTA/MSA baselines, and MSA perturbation data. The low-level runtime still uses a few `LMSV_*` environment flags internally because the existing shared-weight and training-log patches depend on those names.

## Key Files

- `fullnet.py`: model selection, config generation, dry-run, run, and analysis CLI.
- `do.py`: direct entry that runs `config.json`.
- `config.json.example`: minimal language-model full-network configuration.
- `frame_fullnet/`: external CLI/config/runner layer.
- `utils/task/fullnet.py`: single-machine full-network orchestration core.
- `utils/analyze/fullnet_result.py`: lightweight paper-oriented result analysis.
- `assets/runtime/model_config/*.yaml`: supported language model presets.
- `scripts/mutation/mutate-auto.sh`: mutation entry used by the full-network chain.
- `scripts/runtime/submodule_entry.py`: runtime patch wrapper used for PTA/MSA graph execution.

## Outputs

Runs are archived under `output/<timestamp>/`. Each iteration keeps runtime logs, generated shell scripts, mutation inputs, PTA/MSA training-log CSVs when available, and `status.json`.

The analysis step writes:

- `analysis/data/summary.json`
- `analysis/summary.md`
- `analysis/report.html`
