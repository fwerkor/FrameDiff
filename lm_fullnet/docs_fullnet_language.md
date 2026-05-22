# Language FullNet Workflow

The full-network workflow is the paper-facing language-model assembly path for FrameDiff. The extracted version exposes only language models and keeps model selection explicit through `fullnet.MODELS`.

## Model Selection

Model names map to YAML files in `assets/runtime/model_config/`.

```json
"MODELS": ["qwen2"]
```

The CLI accepts either separate names or comma-separated names:

```bash
python fullnet.py run --models qwen2
```

By default `NODE_NUM=0`, so the runner reads the selected model config and builds the corresponding decoder depth as a real full-network path.

## Comparison Path

The extracted workflow keeps PTA baseline and MSA replay with shared weights. With trace enabled, it also records PTA perturbation as backup metamorphic data and MSA perturbation as the primary metamorphic dataset.

## Trace Controls

For paper experiments that need component-level diagnosis:

```bash
python fullnet.py run --trace --debug-compare
```

This exports full tensors and module weights at component instrumentation points, plus `trace_index.jsonl` metadata for the 17 paper components, overall loss, mutation records, and baseline/perturbation run labels.

## Analysis Scope

The bundled analyzer summarizes executed iterations, mutation success, functional failures, precision hints, PTA-MSA loss deltas, MSA baseline-vs-perturbation deltas, and per-iteration trace coverage. Heavier delivery-project diagnostics were removed because they are not part of the current language-model paper workflow.
