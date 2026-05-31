from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class FullNetConfigTests(unittest.TestCase):
    def test_build_run_config_is_pta_msa_only(self) -> None:
        from fullnet_core.config import build_run_config

        config = build_run_config(
            {
                "M" + "F_NAME": "unused",
                "fullnet": {
                    "COMPARE_" + "MODE": "pta_" + "m" + "f",
                    "ENABLE_" + "M" + "F_WEIGHT_LOAD": True,
                },
            },
            models=["qwen2", "glm4"],
            perturb_eps="1e-4",
        )
        encoded = json.dumps(config, ensure_ascii=False)
        blocked = ("M" + "F", "pta_" + "mf", "COMPARE_" + "MODE", "TRACE", "PRECISION", "MUTNM")

        self.assertEqual(config["entry"], "fullnet")
        self.assertEqual(config["fullnet"]["MODELS"], ["qwen2", "glm4"])
        self.assertEqual(config["fullnet"]["TOTAL_ITER"], 1)
        self.assertEqual(config["fullnet"]["LOAD_STEPS"], 1)
        self.assertEqual(config["fullnet"]["PERTURB_EPS"], "1e-4")
        self.assertEqual(config["fullnet"]["BASELINE_LOSS_TOLERANCE"], 0.0)
        for token in blocked:
            self.assertNotIn(token, encoded)

    def test_available_models_match_icse_scope(self) -> None:
        from fullnet_core.models import available_models

        models = available_models()
        self.assertIn("qwen2", models)
        self.assertIn("grok1", models)
        self.assertIn("qwen3", models)

    def test_default_config_enables_all_models(self) -> None:
        from fullnet_core.config import DEFAULT_CONFIG
        from fullnet_core.models import available_models

        self.assertEqual(DEFAULT_CONFIG["fullnet"]["MODELS"], available_models())

    def test_prepare_trace_can_be_forced_off(self) -> None:
        from utils.task import fullnet

        old_trace = fullnet.Config.TRACE_ENABLED
        old_env = os.environ.get("LMSV_FULLNET_TRACE")
        try:
            fullnet.Config.TRACE_ENABLED = True
            os.environ["LMSV_FULLNET_TRACE"] = "1"

            block = fullnet._build_trace_env_block(
                iter_num=1,
                trace_dir=Path("/tmp/fullnet-trace"),
                backend="pta",
                run_name="prepare",
                trace_enabled=False,
            )

            self.assertIn("export LMSV_FULLNET_TRACE=0", block)
            self.assertIn("export LMSV_DEBUG_COMPARE=0", block)
        finally:
            fullnet.Config.TRACE_ENABLED = old_trace
            if old_env is None:
                os.environ.pop("LMSV_FULLNET_TRACE", None)
            else:
                os.environ["LMSV_FULLNET_TRACE"] = old_env

    def test_auto_parallel_uses_visible_cards_safely(self) -> None:
        from utils.task import fullnet

        old_env = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
        old_values = {
            name: getattr(fullnet.Config, name)
            for name in (
                "TARGET_TENSOR_PARALLEL_SIZE",
                "TARGET_PIPELINE_PARALLEL_SIZE",
                "TARGET_EXPERT_PARALLEL_SIZE",
                "TARGET_CONTEXT_PARALLEL_SIZE",
                "TARGET_NPUS_PER_NODE",
                "TARGET_WORLD_SIZE",
                "ENABLE_DATA_PARALLEL",
            )
        }
        try:
            os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
            fullnet.Config.TARGET_TENSOR_PARALLEL_SIZE = 0
            fullnet.Config.TARGET_PIPELINE_PARALLEL_SIZE = 0
            fullnet.Config.TARGET_EXPERT_PARALLEL_SIZE = 0
            fullnet.Config.TARGET_CONTEXT_PARALLEL_SIZE = 0
            fullnet.Config.TARGET_NPUS_PER_NODE = 0
            fullnet.Config.TARGET_WORLD_SIZE = 0
            fullnet.Config.ENABLE_DATA_PARALLEL = False
            fullnet.configure_auto_parallel_from_models([str(PROJECT_ROOT.parent / "model_config" / "qwen2.yaml")])
            dist_cfg = fullnet.resolve_distributed_config()

            self.assertEqual(dist_cfg["tp"], 8)
            self.assertEqual(dist_cfg["pp"], 1)
            self.assertEqual(dist_cfg["ep"], 1)
            self.assertEqual(dist_cfg["npus_per_node"], 8)
            self.assertEqual(dist_cfg["world_size"], 8)

            fullnet.Config.ENABLE_DATA_PARALLEL = True
            dist_cfg = fullnet.resolve_distributed_config()
            self.assertEqual(dist_cfg["npus_per_node"], 8)
            self.assertEqual(dist_cfg["world_size"], 8)

            fullnet.Config.TARGET_TENSOR_PARALLEL_SIZE = 0
            fullnet.Config.TARGET_PIPELINE_PARALLEL_SIZE = 0
            fullnet.Config.TARGET_EXPERT_PARALLEL_SIZE = 0
            fullnet.Config.TARGET_CONTEXT_PARALLEL_SIZE = 0
            fullnet.Config.TARGET_NPUS_PER_NODE = 0
            fullnet.Config.TARGET_WORLD_SIZE = 0
            fullnet.Config.ENABLE_DATA_PARALLEL = False
            fullnet.configure_auto_parallel_from_models([str(PROJECT_ROOT.parent / "model_config" / "llama2.yaml")])
            dist_cfg = fullnet.resolve_distributed_config()
            self.assertEqual(dist_cfg["tp"], 8)
            self.assertEqual(dist_cfg["npus_per_node"], 8)
            self.assertEqual(dist_cfg["world_size"], 8)

            fullnet.Config.TARGET_TENSOR_PARALLEL_SIZE = 0
            fullnet.Config.TARGET_PIPELINE_PARALLEL_SIZE = 0
            fullnet.Config.TARGET_EXPERT_PARALLEL_SIZE = 0
            fullnet.Config.TARGET_CONTEXT_PARALLEL_SIZE = 0
            fullnet.Config.TARGET_NPUS_PER_NODE = 0
            fullnet.Config.TARGET_WORLD_SIZE = 0
            fullnet.Config.ENABLE_DATA_PARALLEL = False
            fullnet.configure_auto_parallel_from_models([str(PROJECT_ROOT.parent / "model_config" / "chatglm3.yaml")])
            dist_cfg = fullnet.resolve_distributed_config()
            self.assertEqual(dist_cfg["tp"], 8)
            self.assertEqual(dist_cfg["npus_per_node"], 8)
            self.assertEqual(dist_cfg["world_size"], 8)
        finally:
            if old_env is None:
                os.environ.pop("ASCEND_RT_VISIBLE_DEVICES", None)
            else:
                os.environ["ASCEND_RT_VISIBLE_DEVICES"] = old_env
            for name, value in old_values.items():
                setattr(fullnet.Config, name, value)

    def test_variant_runtime_overrides_are_injected(self) -> None:
        from utils.task import fullnet

        cmd = fullnet.build_pta_verify_stage_cmd(
            1,
            "-c model_config -r 1 --mutnm 0 -n 1 -m model_config/qwen2.yaml",
            "res/qwen2/mutating-1.json",
            "pta",
            "/tmp/pta",
            "/tmp/shared.pth",
            "load",
            1,
            runtime_args_list=["--bf16", "--sequence-parallel"],
            launcher_overrides={
                "TARGET_TENSOR_PARALLEL_SIZE": 2,
                "TARGET_CONTEXT_PARALLEL_SIZE": 2,
                "TARGET_WORLD_SIZE": 4,
                "TARGET_NPUS_PER_NODE": 4,
            },
            env_overrides={"HCCL_DETERMINISTIC": False, "CUDA_DEVICE_MAX_CONNECTIONS": "default"},
            optimizer_env={"LMSV_RQ3_LR": "3e-4"},
        )

        self.assertIn("--bf16", cmd)
        self.assertIn("--sequence-parallel", cmd)
        self.assertIn("--context-parallel-size 2", cmd)
        self.assertIn("NPUS_PER_NODE=4", cmd)
        self.assertIn("export HCCL_DETERMINISTIC=false", cmd)
        self.assertIn("unset CUDA_DEVICE_MAX_CONNECTIONS", cmd)
        self.assertIn("export LMSV_RQ3_LR=3e-4", cmd)

    def test_cli_dry_run_outputs_fullnet_config(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "fullnet.py",
                "run",
                "--models",
                "qwen2",
                "glm4",
                "--iters",
                "2",
                "--load-steps",
                "4",
                "--perturb-eps",
                "1e-5",
                "--dry-run",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        config = json.loads(result.stdout)
        encoded = json.dumps(config, ensure_ascii=False)
        blocked = ("M" + "F", "COMPARE_" + "MODE", "TRACE", "PRECISION")

        self.assertEqual(config["fullnet"]["MODELS"], ["qwen2", "glm4"])
        self.assertEqual(config["fullnet"]["TOTAL_ITER"], 1)
        self.assertEqual(config["fullnet"]["LOAD_STEPS"], 1)
        self.assertEqual(config["fullnet"]["PERTURB_EPS"], "1e-5")
        for token in blocked:
            self.assertNotIn(token, encoded)

    def test_final_overview_includes_training_matrix(self) -> None:
        from utils.task.fullnet import build_final_overview_markdown

        summary = {
            "overall_status": "FAILED",
            "iterations": 1,
            "planned_variant_runs": 2,
            "failed_variant_runs": 1,
            "pta-baseline_success": 1,
            "msa-baseline_success": 0,
            "model_results": [
                {
                    "model": "qwen2",
                    "status": "PARTIAL",
                    "planned_variant_runs": 2,
                    "pta_baseline_success": 1,
                    "msa_baseline_success": 0,
                    "failed_variant_runs": 1,
                    "reason": "存在失败的训练阶段",
                }
            ],
            "records": [
                {
                    "model": "qwen2",
                    "variant": "ancestor",
                    "iteration": 1,
                    "overall_status": "FAILED",
                    "reason": "MSA baseline 失败",
                    "trainings": {
                        "prepare": "OK",
                        "pta-baseline": "OK",
                        "msa-baseline": "ERROR",
                        "baseline-align": "SKIP",
                        "pta-preturb": "SKIP",
                        "msa-preturb": "SKIP",
                    },
                }
            ],
        }

        markdown = build_final_overview_markdown(summary)

        self.assertIn("## Training Matrix", markdown)
        self.assertIn("qwen2 | ancestor | 1 | FAILED", markdown)
        self.assertIn("msa-baseline", markdown)
        self.assertIn("MSA baseline 失败", markdown)

    def test_graph_load_normalizes_dtype_strings(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed in this test environment")
        import torch
        from utils.runtime.core import graph

        config = {
            "params_dtype": "torch.float32",
            "autocast_dtype": "torch.float16",
            "pipeline_dtype": "torch.bfloat16",
        }
        graph._normalize_torch_dtype_fields(config)

        self.assertIs(config["params_dtype"], torch.float32)
        self.assertIs(config["autocast_dtype"], torch.float16)
        self.assertIs(config["pipeline_dtype"], torch.bfloat16)


class FullNetAnalysisTests(unittest.TestCase):
    def test_analyzer_reports_pta_msa_loss_delta(self) -> None:
        from utils.analyze.fullnet_result import analyze_fullnet_run

        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            variant_dir = output_root / "qwen2" / "ancestor"
            pta_dir = variant_dir / "pta-baseline"
            msa_dir = variant_dir / "msa-baseline"
            msa_perturb_dir = variant_dir / "msa-preturb"
            mutant_dir = output_root / "qwen2" / "mutant-a"
            mutant_pta_dir = mutant_dir / "pta-baseline"
            mutant_msa_dir = mutant_dir / "msa-baseline"
            pta_dir.mkdir(parents=True)
            msa_dir.mkdir(parents=True)
            msa_perturb_dir.mkdir(parents=True)
            mutant_pta_dir.mkdir(parents=True)
            mutant_msa_dir.mkdir(parents=True)
            (output_root / "summary.json").write_text(
                json.dumps({"models": ["qwen2"], "iterations": 1}, ensure_ascii=False),
                encoding="utf-8",
            )
            (variant_dir / "status.json").write_text(
                json.dumps(
                    {
                        "task_name": "fullnet",
                        "model": "qwen2",
                        "variant": "ancestor",
                        "iteration": 1,
                        "overall_status": "PASS",
                        "trainings": {
                            "prepare": "OK",
                            "pta-baseline": "OK",
                            "msa-baseline": "OK",
                            "pta-preturb": "OK",
                            "msa-preturb": "OK",
                            "baseline-align": "ERROR",
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (pta_dir / "execution.csv").write_text("Iteration,loss\n1,1.0\n", encoding="utf-8")
            (msa_dir / "execution.csv").write_text("Iteration,loss\n1,1.25\n", encoding="utf-8")
            (msa_perturb_dir / "execution.csv").write_text("Iteration,loss\n1,1.5\n", encoding="utf-8")
            (variant_dir / "baseline_alignment.json").write_text(
                json.dumps(
                    {
                        "model": "qwen2",
                        "variant": "ancestor",
                        "iteration": 1,
                        "aligned": False,
                        "required": True,
                        "tolerance": 0.0,
                        "issue": "diff",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (mutant_dir / "status.json").write_text(
                json.dumps(
                    {
                        "task_name": "fullnet",
                        "model": "qwen2",
                        "variant": "mutant-a",
                        "iteration": 1,
                        "overall_status": "PASS",
                        "trainings": {
                            "prepare": "SKIP",
                            "pta-baseline": "OK",
                            "msa-baseline": "OK",
                            "pta-preturb": "OK",
                            "msa-preturb": "OK",
                            "baseline-align": "WARN",
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            mutant_pta_dir.joinpath("execution.csv").write_text("Iteration,loss\n1,2.0\n", encoding="utf-8")
            mutant_msa_dir.joinpath("execution.csv").write_text("Iteration,loss\n1,3.0\n", encoding="utf-8")
            (mutant_dir / "baseline_alignment.json").write_text(
                json.dumps(
                    {
                        "model": "qwen2",
                        "variant": "mutant-a",
                        "iteration": 1,
                        "aligned": False,
                        "required": False,
                        "tolerance": 0.0,
                        "issue": "diff",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            trace_dir = pta_dir
            tensor_path = trace_dir / "x.pt"
            weight_path = trace_dir / "w_weight.pt"
            tensor_path.parent.mkdir(parents=True, exist_ok=True)
            weight_path.parent.mkdir(parents=True, exist_ok=True)
            tensor_path.write_bytes(b"tensor")
            weight_path.write_bytes(b"weights")
            (trace_dir / "trace_index.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "kind": "tensor",
                                "component_id": 11,
                                "component_name": "embedding_layer",
                                "run": "pta-baseline",
                                "backend": "pta",
                                "step": 0,
                            }
                        ),
                        json.dumps(
                            {
                                "kind": "weights",
                                "component_id": 11,
                                "component_name": "embedding_layer",
                                "run": "pta-baseline",
                                "backend": "pta",
                                "step": 0,
                            }
                        ),
                        json.dumps(
                            {
                                "kind": "event",
                                "event": "loss",
                                "run": "pta-baseline",
                                "backend": "pta",
                                "step": 0,
                                "payload": {"name": "overall_loss", "value": 1.0},
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = analyze_fullnet_run(output_root, model_name="qwen2", planned_iterations=1)
            payload = json.loads(result.summary_json.read_text(encoding="utf-8"))

        ancestor = next(item for item in payload["iterations"] if item["variant"] == "ancestor")
        mutant = next(item for item in payload["iterations"] if item["variant"] == "mutant-a")
        self.assertEqual(result.executed_iterations, 2)
        self.assertEqual(result.variant_success_count, 2)
        self.assertEqual(ancestor["model"], "qwen2")
        self.assertEqual(ancestor["variant"], "ancestor")
        self.assertAlmostEqual(ancestor["pta_msa_abs_loss_delta"], 0.25)
        self.assertAlmostEqual(ancestor["msa_metamorphic_abs_loss_delta"], 0.25)
        self.assertFalse(ancestor["baseline_aligned"])
        self.assertTrue(ancestor["baseline_alignment_required"])
        self.assertFalse(mutant["baseline_aligned"])
        self.assertFalse(mutant["baseline_alignment_required"])
        self.assertEqual(payload["baseline_alignment_failures"], 1)
        self.assertEqual(ancestor["trace"]["tensor_count"], 1)
        self.assertEqual(ancestor["trace"]["weight_count"], 1)
        self.assertEqual(payload["trace_tensor_count"], 1)
        self.assertNotIn("mf" + "_failures", payload)
        self.assertNotIn("pta_" + "m" + "f_abs_loss_delta", ancestor)

    def test_precision_check_falls_back_when_step_series_missing(self) -> None:
        from utils.analyze.precision import find_preferred_loss_mismatch

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pta_csv = root / "execution_pta.csv"
            msa_csv = root / "execution_msa.csv"
            pta_step = root / "pta_steps.csv"
            msa_step = root / "msa_steps.csv"
            pta_csv.write_text("Iteration,loss\n1,1.0\n", encoding="utf-8")
            msa_csv.write_text("Iteration,loss\n1,1.5\n", encoding="utf-8")
            pta_step.write_text("step,loss\n1,1.0\n", encoding="utf-8")
            msa_step.write_text("step,loss\n2,1.5\n", encoding="utf-8")

            issue = find_preferred_loss_mismatch(
                pta_csv,
                msa_csv,
                iteration=1,
                pta_step_csv_path=pta_step,
                msa_step_csv_path=msa_step,
            )

        self.assertIsNotNone(issue)
        self.assertIn("iter=1", issue)


class FullNetTraceTests(unittest.TestCase):
    def test_fullnet_component_classifier_uses_specific_leaf_components(self) -> None:
        from utils.runtime.fullnet_trace import classify_fullnet_component

        class IdentityOp:
            pass

        class Dropout:
            pass

        class DotProductAttention:
            pass

        class TransformerLayer:
            pass

        class Config:
            multi_latent_attention = False
            num_moe_experts = 4

        class MoERouter:
            config = Config()

        self.assertEqual(
            classify_fullnet_component("decoder_1.cross_attention", IdentityOp()),
            (10, "residual_elementwise_operator"),
        )
        self.assertEqual(
            classify_fullnet_component("decoder_1.self_attention.core_attention.scale_mask_softmax", object()),
            (7, "softmax_operator"),
        )
        self.assertEqual(
            classify_fullnet_component("decoder_1.self_attention.core_attention.attention_dropout", Dropout()),
            (10, "residual_elementwise_operator"),
        )
        self.assertEqual(
            classify_fullnet_component("decoder_1.self_attention.core_attention", DotProductAttention()),
            (5, "attention_core_operator"),
        )
        self.assertEqual(
            classify_fullnet_component("decoder_1.mlp.router", MoERouter()),
            (16, "moe_ffn_block"),
        )
        self.assertEqual(
            classify_fullnet_component("decoder_1.layers.0", TransformerLayer()),
            (14, "decoder_block"),
        )
        self.assertEqual(
            classify_fullnet_component("block0.layer0.output", TransformerLayer()),
            (14, "decoder_block"),
        )

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch unavailable")
    def test_trace_exports_full_tensor_and_weights(self) -> None:
        import torch

        from utils.runtime.fullnet_trace import maybe_perturb_tensor, set_trace_step, trace_loss, trace_module_weights, trace_tensor

        env_names = [
            "LMSV_FULLNET_TRACE",
            "LMSV_FULLNET_TRACE_DIR",
            "LMSV_FULLNET_TRACE_RECORD_DIR",
            "LMSV_FULLNET_TRACE_BACKEND",
            "LMSV_FULLNET_TRACE_RUN",
            "LMSV_FULLNET_TRACE_ITER",
            "LMSV_FULLNET_TRACE_FULL_WEIGHTS",
            "LMSV_FULLNET_PERTURB",
            "LMSV_FULLNET_PERTURB_EPS",
            "LMSV_FULLNET_TRACE_MODE",
        ]
        old_env = {name: os.environ.get(name) for name in env_names}
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                record_dir = str(Path(temp_dir) / "records")
                os.environ["LMSV_FULLNET_TRACE"] = "1"
                os.environ["LMSV_FULLNET_TRACE_DIR"] = temp_dir
                os.environ["LMSV_FULLNET_TRACE_RECORD_DIR"] = record_dir
                os.environ["LMSV_FULLNET_TRACE_BACKEND"] = "pta"
                os.environ["LMSV_FULLNET_TRACE_RUN"] = "pta-baseline"
                os.environ["LMSV_FULLNET_TRACE_ITER"] = "1"
                os.environ["LMSV_FULLNET_TRACE_FULL_WEIGHTS"] = "1"
                os.environ["LMSV_FULLNET_PERTURB"] = "1"
                os.environ["LMSV_FULLNET_PERTURB_EPS"] = "1e-5"
                os.environ["LMSV_FULLNET_TRACE_MODE"] = "full"
                set_trace_step(0)

                tensor_path = trace_tensor(3, "linear_operator", "x", torch.arange(4), stage="unit")
                weight_path = trace_module_weights(
                    3,
                    "linear_operator",
                    torch.nn.Linear(2, 2),
                    stage="unit",
                    module_name="linear",
                )
                trace_loss("overall_loss", torch.tensor(2.0))
                perturbed = maybe_perturb_tensor(
                    torch.zeros(2, dtype=torch.float32),
                    tensor_name="unit_input",
                    component_id=0,
                    component_name="full_network",
                )

                self.assertTrue(Path(tensor_path).exists())
                self.assertTrue(Path(weight_path).exists())
                self.assertTrue(torch.allclose(perturbed, torch.full((2,), 1e-5)))
                self.assertEqual(Path(tensor_path).parent, Path(temp_dir))
                self.assertEqual(Path(weight_path).parent, Path(temp_dir))
                index_path = Path(record_dir) / "trace_index.jsonl"
                rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
                self.assertTrue((Path(record_dir) / "components.json").exists())
                self.assertTrue(any(row.get("kind") == "tensor" for row in rows))
                self.assertTrue(any(row.get("kind") == "weights" for row in rows))
                self.assertTrue(any(row.get("event") == "loss" for row in rows))
        finally:
            for name, value in old_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch unavailable")
    def test_trace_defaults_to_precise_public_output_tensors(self) -> None:
        import torch

        from utils.runtime.fullnet_trace import set_trace_step, trace_module_weights, trace_tensor

        env_names = [
            "LMSV_FULLNET_TRACE",
            "LMSV_FULLNET_TRACE_DIR",
            "LMSV_FULLNET_TRACE_RECORD_DIR",
            "LMSV_FULLNET_TRACE_BACKEND",
            "LMSV_FULLNET_TRACE_RUN",
            "LMSV_FULLNET_TRACE_ITER",
            "LMSV_FULLNET_TRACE_FULL_WEIGHTS",
            "LMSV_FULLNET_TRACE_MODE",
        ]
        old_env = {name: os.environ.get(name) for name in env_names}
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                os.environ["LMSV_FULLNET_TRACE"] = "1"
                os.environ["LMSV_FULLNET_TRACE_DIR"] = temp_dir
                os.environ["LMSV_FULLNET_TRACE_BACKEND"] = "pta"
                os.environ["LMSV_FULLNET_TRACE_RUN"] = "pta-baseline"
                os.environ["LMSV_FULLNET_TRACE_ITER"] = "1"
                os.environ["LMSV_FULLNET_TRACE_FULL_WEIGHTS"] = "1"
                os.environ["LMSV_FULLNET_TRACE_MODE"] = "output_only"

                set_trace_step(0)
                skipped_input = trace_tensor(14, "decoder_block", "decoder_1.input", torch.ones(1), stage="decoder_input", node_id=1)
                module_output = trace_tensor(3, "linear_operator", "decoder_1.layers.0.self_attention.linear_qkv.output", torch.ones(1), stage="module_output", node_id=1)
                duplicate_module_output = trace_tensor(3, "linear_operator", "decoder_1.layers.0.mlp.linear_fc1.output", torch.ones(1), stage="module_output", node_id=1)
                decoder_output = trace_tensor(14, "decoder_block", "decoder_1.output", torch.ones(1), stage="decoder_output", node_id=1)
                duplicate_decoder_output = trace_tensor(14, "decoder_block", "decoder_1.output", torch.ones(1), stage="decoder_output", node_id=1)
                output_like_input = trace_tensor(15, "output_layer", "lm_head.input", torch.ones(1), stage="output_layer_input")
                logits = trace_tensor(15, "output_layer", "logits", torch.ones(1), stage="output_layer_output")
                hidden_before_head = trace_tensor(0, "full_network", "hidden_states_before_output_layer", torch.ones(1), stage="network_hidden")
                final_output = trace_tensor(0, "full_network", "final_output", torch.ones(1), stage="network_output")
                step_final_output = trace_tensor(0, "full_network", "step_final_output", torch.ones(1), stage="pta_step_output")
                block0_debug_output = trace_tensor(3, "linear_operator", "block0.mlp.linear_fc1.output", torch.ones(1), stage="block0_hook_output")
                perturb_delta = trace_tensor(11, "embedding_layer", "embedding_output.delta", torch.ones(1), stage="embedding_output_perturbation")
                perturb_baseline = trace_tensor(11, "embedding_layer", "embedding_output.baseline", torch.ones(1), stage="embedding_output_perturbation")
                perturb_perturbed = trace_tensor(11, "embedding_layer", "embedding_output.perturbed", torch.ones(1), stage="embedding_output_perturbation")
                second_instance = trace_tensor(14, "decoder_block", "decoder_2.output", torch.ones(1), stage="decoder_output", node_id=2)
                skipped_weight = trace_module_weights(14, "decoder_block", torch.nn.Linear(1, 1), stage="decoder_weights", node_id=1)
                set_trace_step(1)
                next_step = trace_tensor(14, "decoder_block", "decoder_1.output", torch.ones(1), stage="decoder_output", node_id=1)

                self.assertIsNone(skipped_input)
                self.assertIsNotNone(module_output)
                self.assertIsNone(duplicate_module_output)
                self.assertIsNotNone(decoder_output)
                self.assertIsNone(duplicate_decoder_output)
                self.assertIsNone(output_like_input)
                self.assertIsNotNone(logits)
                self.assertIsNone(hidden_before_head)
                self.assertIsNotNone(final_output)
                self.assertIsNotNone(step_final_output)
                self.assertIsNone(block0_debug_output)
                self.assertIsNone(perturb_delta)
                self.assertIsNotNone(perturb_baseline)
                self.assertIsNotNone(perturb_perturbed)
                self.assertIsNotNone(second_instance)
                self.assertIsNone(skipped_weight)
                self.assertIsNotNone(next_step)
        finally:
            for name, value in old_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value



if __name__ == "__main__":
    unittest.main()
