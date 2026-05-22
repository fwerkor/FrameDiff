# FrameDiff Experiment Code

Reusable framework differential experiment workflows.

## Layout

- `experiments/operator_diff/`: operator-level RQ1 differential tests and RQ2 metamorphic tests.
- `experiments/component_diff/`: component-level RQ1 differential tests and RQ2 metamorphic tests.
- `experiments/fullnet_diff/`: full-network language-model differential workflow.
- `experiments/frame_diff_common/`: shared config loading, tensor I/O, metrics, tensor generation, weight sync, and model configs.

## Entry Points

Run modules from the repository root:

```bash
python -m experiments.operator_diff.run_diff_test --backend pta
python -m experiments.operator_diff.run_diff_test --backend msa
python -m experiments.operator_diff.analyze_diff_results

python -m experiments.operator_diff.run_metamorphic_test --backend pta
python -m experiments.operator_diff.run_metamorphic_test --backend msa
python -m experiments.operator_diff.analyze_metamorphic_results

python -m experiments.component_diff.run_diff_test --backend pta
python -m experiments.component_diff.run_diff_test --backend msa
python -m experiments.component_diff.analyze_diff_results

python -m experiments.component_diff.run_metamorphic_test --backend pta
python -m experiments.component_diff.run_metamorphic_test --backend msa
python -m experiments.component_diff.analyze_metamorphic_results

python -m experiments.fullnet_diff.fullnet models
python -m experiments.fullnet_diff.fullnet run --models qwen2 --iters 1 --dry-run
```

The default output directories are configured in `experiments/operator_diff/config.yaml`
and `experiments/component_diff/config.yaml`; relative paths are resolved from the
`experiments/` root.
