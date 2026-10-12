from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.evaluation.provenance import config_digest, sha256_file
from src.rl.train import load_training_config, train_from_config


class TestTrainSmoke(unittest.TestCase):
    def test_train_small_config_saves_expected_artifacts(self) -> None:
        config_path = Path(__file__).resolve().parent.parent / "configs" / "train_small.yaml"
        config = load_training_config(config_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            summary = train_from_config(config, output_root=tmpdir)
            run_dir = Path(summary["run_dir"])

            self.assertTrue((run_dir / "selected_model.zip").exists())
            self.assertTrue((run_dir / "selected_vecnormalize.pkl").exists())
            self.assertTrue((run_dir / "last_model.zip").exists())
            self.assertTrue((run_dir / "selection.json").exists())
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "checkpoints").exists())
            self.assertTrue((run_dir / "eval" / "evaluations.npz").exists())
            self.assertTrue((run_dir / "logs" / "train_monitor.csv").exists())
            self.assertTrue((run_dir / "logs" / "progress.csv").exists())
            self.assertTrue((run_dir / "logs" / "progress.json").exists())

            checkpoint_files = list((run_dir / "checkpoints").glob("*.zip"))
            self.assertGreaterEqual(len(checkpoint_files), 1)

            with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
                saved_summary = json.load(handle)
            self.assertEqual(saved_summary["gae_lambda"], 1.0)
            self.assertEqual(saved_summary["n_envs"], 4)
            self.assertIn("selected_validation", saved_summary)
            self.assertIn("eval_seeds", saved_summary)
            self.assertIn("episode_rewards", saved_summary["selected_validation"])

            selection = json.loads(
                (run_dir / "selection.json").read_text(encoding="utf-8")
            )
            self.assertEqual(selection["selection_metric"], "social_welfare")
            self.assertEqual(selection["model_file"], "selected_model.zip")
            self.assertEqual(
                selection["model_sha256"],
                sha256_file(run_dir / "selected_model.zip"),
            )
            self.assertEqual(
                selection["normalization_sha256"],
                sha256_file(run_dir / "selected_vecnormalize.pkl"),
            )
            self.assertEqual(selection["config_fingerprint"], config_digest(config))
            with np.load(run_dir / "eval" / "evaluations.npz") as evaluations:
                self.assertIn("raw_rsu_reward", evaluations)
                self.assertIn("common_evaluation_reward", evaluations)
                self.assertEqual(
                    str(evaluations["reward_schema"].item()),
                    "common_evaluation_reward_v1",
                )
                indices = np.flatnonzero(
                    evaluations["timesteps"] == selection["selected_timestep"]
                )
                self.assertEqual(len(indices), 1)
                selected_mean = float(
                    np.mean(evaluations["social_welfare"][int(indices[0])])
                )
            self.assertAlmostEqual(
                selected_mean, selection["selected_metric_value"], places=10
            )

            monitor_text = (run_dir / "logs" / "train_monitor.csv").read_text(encoding="utf-8")
            self.assertIn("raw_rsu_reward", monitor_text)
            self.assertIn("te_ppo_reward", monitor_text)
            self.assertIn("penalty_fail", monitor_text)
            self.assertIn("penalty_lat", monitor_text)
            self.assertIn("penalty_part", monitor_text)


if __name__ == "__main__":
    unittest.main()
