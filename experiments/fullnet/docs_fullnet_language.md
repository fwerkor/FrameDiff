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

The extracted workflow reads prepared variants from `../mutated_config/<model>/`. `ancestor` runs `prepare`; every enabled variant then runs `pta-baseline`, `msa-baseline`, `pta-preturb`, and `msa-preturb` with the shared ancestor weight.

RQ3 variants are stored as sibling directories beside `ancestor`. Each variant supplies `mutating.json` and `mutated_config.yaml`; `mutating.json` preserves the legacy node records required by graph loading and adds paper metadata plus `runtime_overrides` for runtime args, launcher settings, environment variables, optimizer settings, and runtime controls. Per-model manifests record generated variants and skipped model-inapplicable variants.

## Trace Controls

For paper experiments, component-level trace and full weight export are always enabled. The metamorphic perturbation is a configurable one-way `+eps` applied to every floating-point tensor element; the default is `1e-5`.

```bash
python fullnet.py run --models qwen2 glm4 --perturb-eps 1e-5
```

This exports tensor artifacts at component instrumentation points into the public `output/<model>/<variant>/<training>/*.pt` layout. Trace metadata (`trace_index.jsonl` and component manifests) stays under `records` with logs and copied configs.

## Analysis Scope

The run writes `records/summary.json` and a console/Markdown overview with every model, variant, iteration, and training stage. The bundled analyzer summarizes executed model/variant runs, functional failures, precision hints, PTA-MSA loss deltas, baseline-vs-perturbation deltas, and trace coverage from the `records/<model>/<variant>/<training>` metadata layout while public tensors remain in `output`. Heavier delivery-project diagnostics were removed because they are not part of the current language-model paper workflow.
