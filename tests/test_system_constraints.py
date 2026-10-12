import unittest

import numpy as np

from src.examples.single_step_explain import (
    DEMO_SEED,
    make_demo_action,
    make_demo_config,
    make_demo_scenario,
)
from src.sim.env import RSULeaderEnv


def make_demo_env() -> RSULeaderEnv:
    config = make_demo_config()
    state = make_demo_scenario()
    return RSULeaderEnv(
        scenario_config=config,
        scenario_factory=lambda rng, scenario_config, fixed_state=state: fixed_state,
    )


class TestSystemConstraints(unittest.TestCase):
    def test_selected_tasks_satisfy_deadlines(self) -> None:
        env = make_demo_env()
        state = make_demo_scenario()
        config = make_demo_config()
        env.reset(seed=DEMO_SEED)
        _, _, _, _, info = env.step(make_demo_action())

        for response in info["follower_responses"]:
            if response.selected_task_id is None:
                continue
            selected_eval = next(
                evaluation
                for evaluation in response.task_evaluations
                if evaluation.task_id == response.selected_task_id
            )
            self.assertLessEqual(
                selected_eval.total_latency_s,
                state.tasks[selected_eval.task_id].tau_max_s + config.bisection_tol,
            )

    def test_each_vehicle_selects_at_most_one_task(self) -> None:
        env = make_demo_env()
        env.reset(seed=DEMO_SEED)
        _, _, _, _, info = env.step(make_demo_action())

        for response in info["follower_responses"]:
            selected_count = sum(
                int(response.selected_task_id == evaluation.task_id)
                for evaluation in response.task_evaluations
            )
            self.assertLessEqual(selected_count, 1)

    def test_valid_bandwidth_sum_is_accepted_and_invalid_sum_raises(self) -> None:
        env = make_demo_env()
        env.reset(seed=DEMO_SEED)
        env.step(make_demo_action())

        env_invalid = make_demo_env()
        env_invalid.reset(seed=DEMO_SEED)
        with self.assertRaises(ValueError):
            env_invalid.step(
                {
                    "prices": np.array([0.75, 0.55], dtype=np.float64),
                    "bandwidth": np.array([0.6, 0.3, 0.2], dtype=np.float64),
                }
            )

    def test_feasible_candidates_respect_eta_lower_bound(self) -> None:
        env = make_demo_env()
        env.reset(seed=DEMO_SEED)
        _, _, _, _, info = env.step(make_demo_action())

        for response in info["follower_responses"]:
            for evaluation in response.task_evaluations:
                if evaluation.feasible:
                    self.assertGreaterEqual(evaluation.eta, 1.0)

    def test_feasible_candidates_have_nonnegative_sensing_time(self) -> None:
        env = make_demo_env()
        env.reset(seed=DEMO_SEED)
        _, _, _, _, info = env.step(make_demo_action())

        for response in info["follower_responses"]:
            for evaluation in response.task_evaluations:
                if evaluation.feasible:
                    self.assertGreaterEqual(evaluation.tau_s_s, 0.0)


if __name__ == "__main__":
    unittest.main()
