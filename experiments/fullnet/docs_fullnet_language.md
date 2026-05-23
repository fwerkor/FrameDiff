# Language FullNet Workflow

The full-network workflow is the paper-facing language-model assembly path for FrameDiff. The extracted version exposes only language models and keeps model selection explicit through `fullnet.MODELS`.

## Model Selection

Model names map to YAML files in `../model_config/`.

```json
"MODELS": ["baichuan2", "chatglm3", "codellama", "deepseekv3", "glm4", "grok1", "llama2", "mixtral", "pangu", "qwen2", "qwen3", "yi"]
```

The CLI accepts either separate names or comma-separated names:

```bash
python fullnet.py run --models qwen2 glm4
```

The runner reads the selected model config and builds the corresponding decoder depth as a real full-network path. Model names are discovered from `../model_config/*.yaml`; the default config enables all discovered models. A failure in one model is recorded and the runner continues with the remaining models.

## Comparison Path

The extracted workflow reads prepared variants from `../mutated_config/<model>/`. `ancestor` runs `prepare`; every variant then runs `pta-baseline`, `msa-baseline`, `pta-preturb`, and `msa-preturb` with the shared ancestor weight. The repeat count is configurable through `TOTAL_ITER` or `--iters` and defaults to `1`; load-mode steps default to `3`.

## Trace Controls

For paper experiments, component-level trace and full weight export are always enabled. The metamorphic perturbation is a configurable one-way `+eps` applied to every floating-point tensor element; the default is `1e-5`.

```bash
python fullnet.py run --models qwen2 glm4 --perturb-eps 1e-5
```

This exports full tensors and module weights at component instrumentation points, plus `trace_index.jsonl` metadata for the 17 paper components, overall loss, variant records, and the five training labels.

## Analysis Scope

The run writes `summary.json` and a console/Markdown overview with every model, variant, iteration, and training stage. The bundled analyzer summarizes executed model/variant runs, functional failures, precision hints, PTA-MSA loss deltas, baseline-vs-perturbation deltas, and trace coverage from the `output/<model>/<variant>/<training>` layout. Heavier delivery-project diagnostics were removed because they are not part of the current language-model paper workflow.
