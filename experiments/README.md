# FrameDiff Experiment Code

Reusable framework differential experiment workflows.

## Layout

- `experiments/operator/`: operator-level RQ1 differential tests and RQ2 metamorphic tests.
- `experiments/component/`: component-level RQ1 differential tests and RQ2 metamorphic tests.
- `experiments/fullnet/`: full-network language-model differential workflow.
- `experiments/common/`: shared config loading, tensor I/O, metrics, tensor generation, and weight sync.
- `experiments/model_config/`: shared language-model YAML presets.
- `experiments/mutated_config/`: prepared full-network variants.

## Entry Points

Run modules from the repository root:

```bash
python -m experiments.operator.scripts.operator_diff_test --backend pta
python -m experiments.operator.scripts.operator_diff_test --backend msa
python -m experiments.operator.scripts.operator_diff_analysis

python -m experiments.operator.scripts.operator_metamorphic_test --backend pta
python -m experiments.operator.scripts.operator_metamorphic_test --backend msa
python -m experiments.operator.scripts.operator_meta_analysis

python -m experiments.component.scripts.component_diff_test --backend pta
python -m experiments.component.scripts.component_diff_test --backend msa
python -m experiments.component.scripts.analyze_rq1

python -m experiments.component.scripts.component_metamorphic_test --backend pta
python -m experiments.component.scripts.component_metamorphic_test --backend msa
python -m experiments.component.scripts.analyze_rq2

python -m experiments.fullnet.fullnet models
python -m experiments.fullnet.fullnet run --models qwen2 --iters 1 --load-steps 3 --dry-run
```

The default output directories are configured in `experiments/operator/configs/operator_experiment.yaml`
and `experiments/component/configs/component_experiment.yaml`; relative paths are resolved from the
`experiments/` root.
