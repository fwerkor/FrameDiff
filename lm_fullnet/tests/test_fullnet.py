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
        from frame_fullnet.config import build_run_config

        config = build_run_config(
            {
                "M" + "F_NAME": "unused",
                "fullnet": {
                    "COMPARE_" + "MODE": "pta_" + "m" + "f",
                    "ENABLE_" + "M" + "F_WEIGHT_LOAD": True,
                },
            },
            models=["qwen2", "glm4"],
            total_iter=2,
        )
        encoded = json.dumps(config, ensure_ascii=False)
        blocked = ("M" + "F", "pta_" + "mf", "COMPARE_" + "MODE")

        self.assertEqual(config["entry"], "fullnet")
        self.assertEqual(config["fullnet"]["MODELS"], ["qwen2", "glm4"])
        self.assertEqual(config["fullnet"]["TOTAL_ITER"], 2)
        self.assertEqual(config["fullnet"]["NODE_NUM"], 0)
        self.assertEqual(config["fullnet"]["FULLNET_ASSEMBLY_MODE"], "single_model_fullnet")
        self.assertTrue(config["TRACE"]["EXPORT_FULL_WEIGHTS"])
        for token in blocked:
            self.assertNotIn(token, encoded)

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
                "1",
                "--dry-run",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        config = json.loads(result.stdout)
        encoded = json.dumps(config, ensure_ascii=False)
        blocked = ("M" + "F", "COMPARE_" + "MODE")

        self.assertEqual(config["fullnet"]["MODELS"], ["qwen2", "glm4"])
        self.assertEqual(config["fullnet"]["TOTAL_ITER"], 1)
        for token in blocked:
            self.assertNotIn(token, encoded)


class FullNetAnalysisTests(unittest.TestCase):
    def test_analyzer_reports_pta_msa_loss_delta(self) -> None:
        from utils.analyze.fullnet_result import analyze_fullnet_run

        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            iter_dir = output_root / "iters" / "iter_1"
            iter_dir.mkdir(parents=True)
            (iter_dir / "status.json").write_text(
                json.dumps(
                    {
                        "task_name": "fullnet",
                        "iteration": 1,
                        "overall_status": "PASS",
                        "components": {
                            "MUTATE": "OK",
                            "PTA_SAVE": "OK",
                            "PTA_LOAD": "OK",
                            "MSA_LOAD": "OK",
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (iter_dir / "execution_pta.csv").write_text("Iteration,loss\n1,1.0\n", encoding="utf-8")
            (iter_dir / "execution_msa.csv").write_text("Iteration,loss\n1,1.25\n", encoding="utf-8")
            (iter_dir / "execution_msa_perturb.csv").write_text("Iteration,loss\n1,1.5\n", encoding="utf-8")
            trace_dir = iter_dir / "traces"
            tensor_path = trace_dir / "pta" / "pta_baseline" / "iter_1" / "step_0" / "tensors" / "x.pt"
            weight_path = trace_dir / "pta" / "pta_baseline" / "iter_1" / "step_0" / "weights" / "w.pt"
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
                                "run": "pta_baseline",
                                "backend": "pta",
                                "step": 0,
                            }
                        ),
                        json.dumps(
                            {
                                "kind": "weights",
                                "component_id": 11,
                                "component_name": "embedding_layer",
                                "run": "pta_baseline",
                                "backend": "pta",
                                "step": 0,
                            }
                        ),
                        json.dumps(
                            {
                                "kind": "event",
                                "event": "loss",
                                "run": "pta_baseline",
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

            result = analyze_fullnet_run(output_root, model_name="qwen2,glm4", planned_iterations=1)
            payload = json.loads(result.summary_json.read_text(encoding="utf-8"))

        self.assertEqual(result.executed_iterations, 1)
        self.assertEqual(result.mutation_success_count, 1)
        self.assertAlmostEqual(payload["iterations"][0]["pta_msa_abs_loss_delta"], 0.25)
        self.assertAlmostEqual(payload["iterations"][0]["msa_metamorphic_abs_loss_delta"], 0.25)
        self.assertEqual(payload["iterations"][0]["trace"]["tensor_count"], 1)
        self.assertEqual(payload["iterations"][0]["trace"]["weight_count"], 1)
        self.assertEqual(payload["trace_tensor_count"], 1)
        self.assertNotIn("mf" + "_failures", payload)
        self.assertNotIn("pta_" + "m" + "f_abs_loss_delta", payload["iterations"][0])


class FullNetTraceTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch unavailable")
    def test_trace_exports_full_tensor_and_weights(self) -> None:
        import torch

        from utils.runtime.fullnet_trace import set_trace_step, trace_loss, trace_module_weights, trace_tensor

        env_names = [
            "LMSV_FULLNET_TRACE",
            "LMSV_FULLNET_TRACE_DIR",
            "LMSV_FULLNET_TRACE_BACKEND",
            "LMSV_FULLNET_TRACE_RUN",
            "LMSV_FULLNET_TRACE_ITER",
            "LMSV_FULLNET_TRACE_FULL_WEIGHTS",
        ]
        old_env = {name: os.environ.get(name) for name in env_names}
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                os.environ["LMSV_FULLNET_TRACE"] = "1"
                os.environ["LMSV_FULLNET_TRACE_DIR"] = temp_dir
                os.environ["LMSV_FULLNET_TRACE_BACKEND"] = "pta"
                os.environ["LMSV_FULLNET_TRACE_RUN"] = "pta_baseline"
                os.environ["LMSV_FULLNET_TRACE_ITER"] = "1"
                os.environ["LMSV_FULLNET_TRACE_FULL_WEIGHTS"] = "1"
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

                self.assertTrue(Path(tensor_path).exists())
                self.assertTrue(Path(weight_path).exists())
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
