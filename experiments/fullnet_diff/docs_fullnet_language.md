# Language FullNet Workflow

The full-network workflow is the paper-facing language-model assembly path for FrameDiff. The extracted version exposes only language models and keeps model selection explicit through `fullnet.MODELS`.

## Model Selection

Model names map to YAML files in `../frame_diff_common/model_configs/`.

```json
"MODELS": ["qwen2"]
```

The CLI accepts either separate names or comma-separated names:

```bash
python fullnet.py run --models qwen2
```

The runner reads the selected model config and builds the corresponding decoder depth as a real full-network path. This directory exposes only the 11 models listed in `ICSE实验计划.md`; extra shared YAMLs are ignored by this CLI.

## Comparison Path

The extracted workflow keeps PTA baseline and MSA replay with shared weights. It also records PTA perturbation as backup metamorphic data and MSA perturbation as the primary metamorphic dataset. One launch runs exactly one full-network iteration.

## Trace Controls

For paper experiments, component-level trace and full weight export are always enabled. The metamorphic perturbation is a configurable one-way `+eps` applied to every floating-point tensor element; the default is `1e-5`.

```bash
python fullnet.py run --models qwen2 --perturb-eps 1e-5
```

This exports full tensors and module weights at component instrumentation points, plus `trace_index.jsonl` metadata for the 17 paper components, overall loss, mutation records, and baseline/perturbation run labels.

## Analysis Scope

The bundled analyzer summarizes executed iterations, mutation success, functional failures, precision hints, PTA-MSA loss deltas, MSA baseline-vs-perturbation deltas, and per-iteration trace coverage. Heavier delivery-project diagnostics were removed because they are not part of the current language-model paper workflow.
