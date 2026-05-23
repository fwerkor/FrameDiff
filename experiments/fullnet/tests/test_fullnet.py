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
        self.assertEqual(config["fullnet"]["LOAD_STEPS"], 3)
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
        self.assertEqual(config["fullnet"]["TOTAL_ITER"], 2)
        self.assertEqual(config["fullnet"]["LOAD_STEPS"], 4)
        self.assertEqual(config["fullnet"]["PERTURB_EPS"], "1e-5")
        for token in blocked:
            self.assertNotIn(token, encoded)


class FullNetAnalysisTests(unittest.TestCase):
    def test_analyzer_reports_pta_msa_loss_delta(self) -> None:
        from utils.analyze.fullnet_result import analyze_fullnet_run

        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            variant_dir = output_root / "qwen2" / "ancestor"
            pta_dir = variant_dir / "pta-baseline"
            msa_dir = variant_dir / "msa-baseline"
            msa_perturb_dir = variant_dir / "msa-preturb"
            pta_dir.mkdir(parents=True)
            msa_dir.mkdir(parents=True)
            msa_perturb_dir.mkdir(parents=True)
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
            trace_dir = pta_dir / "traces"
            tensor_path = trace_dir / "pta" / "pta-baseline" / "iter_1" / "step_0" / "tensors" / "x.pt"
            weight_path = trace_dir / "pta" / "pta-baseline" / "iter_1" / "step_0" / "weights" / "w.pt"
            tensor_path.parent.mkdir(parents=True)
            weight_path.parent.mkdir(parents=True)
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

        self.assertEqual(result.executed_iterations, 1)
        self.assertEqual(result.variant_success_count, 1)
        self.assertEqual(payload["iterations"][0]["model"], "qwen2")
        self.assertEqual(payload["iterations"][0]["variant"], "ancestor")
        self.assertAlmostEqual(payload["iterations"][0]["pta_msa_abs_loss_delta"], 0.25)
        self.assertAlmostEqual(payload["iterations"][0]["msa_metamorphic_abs_loss_delta"], 0.25)
        self.assertFalse(payload["iterations"][0]["baseline_aligned"])
        self.assertEqual(payload["baseline_alignment_failures"], 1)
        self.assertEqual(payload["iterations"][0]["trace"]["tensor_count"], 1)
        self.assertEqual(payload["iterations"][0]["trace"]["weight_count"], 1)
        self.assertEqual(payload["trace_tensor_count"], 1)
        self.assertNotIn("mf" + "_failures", payload)
        self.assertNotIn("pta_" + "m" + "f_abs_loss_delta", payload["iterations"][0])

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
    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch unavailable")
    def test_trace_exports_full_tensor_and_weights(self) -> None:
        import torch

        from utils.runtime.fullnet_trace import maybe_perturb_tensor, set_trace_step, trace_loss, trace_module_weights, trace_tensor

        env_names = [
            "LMSV_FULLNET_TRACE",
            "LMSV_FULLNET_TRACE_DIR",
            "LMSV_FULLNET_TRACE_BACKEND",
            "LMSV_FULLNET_TRACE_RUN",
            "LMSV_FULLNET_TRACE_ITER",
            "LMSV_FULLNET_TRACE_FULL_WEIGHTS",
            "LMSV_FULLNET_PERTURB",
            "LMSV_FULLNET_PERTURB_EPS",
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
                os.environ["LMSV_FULLNET_PERTURB"] = "1"
                os.environ["LMSV_FULLNET_PERTURB_EPS"] = "1e-5"
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
                index_path = Path(temp_dir) / "trace_index.jsonl"
                rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
                self.assertTrue((Path(temp_dir) / "components.json").exists())
                self.assertTrue(any(row.get("kind") == "tensor" for row in rows))
                self.assertTrue(any(row.get("kind") == "weights" for row in rows))
                self.assertTrue(any(row.get("event") == "loss" for row in rows))
        finally:
            for name, value in old_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


if __name__ == "__main__":
    unittest.main()
