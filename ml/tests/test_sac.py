"""Offline tests: no game construction or bridge connections."""
import csv
import json
import io
from contextlib import redirect_stdout
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rl_sac
import sac_stats


class CheckpointTests(unittest.TestCase):
    def test_resume_restores_next_sample_update_and_optimizer(self):
        rl_sac.seed_everything(123)
        agent = rl_sac.SAC(rl_sac.RcHeliEnv.N_OBS)
        replay = rl_sac.Replay(cap=8)
        for i in range(12):
            obs = np.full(rl_sac.RcHeliEnv.N_OBS, i / 12, dtype=np.float32)
            replay.add(obs, np.full(rl_sac.ACT_DIM, i / 12, dtype=np.float32), i / 12,
                       obs + .01, i % 3 == 0)
        agent.update(replay.sample(4))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            state = {"episode": 7, "total_steps": 12}
            config = {**rl_sac.default_config(123), "run_id": "test"}
            agent.save(path, replay, state, config)
            expected_random = random.random()
            expected_batch = replay.sample(4)
            expected_loss = agent.update(expected_batch)
            expected_action = agent.act(obs)
            expected_params = {
                name: {k: v.clone() for k, v in getattr(agent, name).state_dict().items()}
                for name in ("actor", "q1", "q2", "t1", "t2")}
            expected_alpha = agent.log_alpha.detach().clone()
            restored = rl_sac.SAC(rl_sac.RcHeliEnv.N_OBS)
            restored_replay = rl_sac.Replay(cap=2)
            metadata = restored.load(path, restored_replay)
            self.assertEqual(metadata["training"], state)
            self.assertEqual(metadata["config"], config)
            self.assertEqual(random.random(), expected_random)
            self.assertEqual((restored_replay.n, restored_replay.i), (8, 4))
            actual_batch = restored_replay.sample(4)
            for a, b in zip(actual_batch, expected_batch):
                self.assertTrue(torch.equal(a, b))
            self.assertEqual(restored.update(actual_batch), expected_loss)
            np.testing.assert_array_equal(restored.act(obs), expected_action)
            for name, parameters in expected_params.items():
                for key, value in getattr(restored, name).state_dict().items():
                    self.assertTrue(torch.equal(value, parameters[key]), (name, key))
            self.assertTrue(torch.equal(restored.log_alpha, expected_alpha))
            self.assertEqual(restored.updates, agent.updates)
            self.assertEqual(restored.target_entropy, -float(rl_sac.ACT_DIM))
            self.assertFalse(list(Path(tmp).glob("*.tmp")))

    def test_rejects_legacy_and_wrong_environment(self):
        agent = rl_sac.SAC(rl_sac.RcHeliEnv.N_OBS)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.pt"
            torch.save({"actor": agent.actor.state_dict()}, path)
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                agent.load(path)
            agent.save(path, rl_sac.Replay(2), {"episode": 0, "total_steps": 0},
                       rl_sac.default_config(0))
            payload = torch.load(path, weights_only=False)
            payload["env_version"] = "wrong-environment"
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                agent.load(path)

    def test_schema_threshold_and_reward_contracts_rejected_before_loading(self):
        agent = rl_sac.SAC(rl_sac.RcHeliEnv.N_OBS)
        before = {k: v.clone() for k, v in agent.actor.state_dict().items()}
        corruptions = {"checkpoint_version": 3, "env_version": "rc-heli-v3",
                       "obs_dim": 24, "observation_schema": ["outdated"],
                       "action_schema": ["roll", "pitch", "yaw", "collective"],
                       "release_threshold": 0., "phase_reward_version": "old-rewards",
                       "obstacle_schema_version": "blind-offset-rays", "damage_penalty_scale": 0.,
                       "progress_scope": "all_stages",
                       "damage_epsilon": .02,
                       "action_model": "mixed-discrete-continuous"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad-contract.pt"
            agent.save(path, rl_sac.Replay(2), {}, rl_sac.default_config(0))
            original = path.read_bytes()
            for location in ("checkpoint", "config"):
                for key, bad_value in corruptions.items():
                    with self.subTest(location=location, key=key):
                        path.write_bytes(original)
                        payload = torch.load(path, weights_only=False)
                        target = payload if location == "checkpoint" else payload["config"]
                        target[key] = bad_value
                        torch.save(payload, path)
                        with self.assertRaisesRegex(ValueError, key):
                            agent.load(path)
                        for name, value in agent.actor.state_dict().items():
                            self.assertTrue(torch.equal(value, before[name]))
            path.write_bytes(original)
            payload = torch.load(path, weights_only=False)
            del payload["observation_schema"]
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "observation_schema"):
                agent.load(path)

    def test_old_environment_rejected_before_evaluation_or_resume_constructs_game(self):
        agent = rl_sac.SAC(rl_sac.RcHeliEnv.N_OBS)
        replay_class = rl_sac.Replay
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v4.pt"
            agent.save(path, replay_class(2), {}, rl_sac.default_config(0))
            payload = torch.load(path, weights_only=False)
            # v4 has the same 5 actions/76 observations but old ray/damage
            # semantics, so dimension checks alone cannot establish compatibility.
            payload.update(checkpoint_version=3, env_version="rc-heli-v4",
                           phase_reward_version="verified-events-v1")
            payload.pop("obstacle_schema_version")
            torch.save(payload, path)
            with patch.object(rl_sac, "RcHeliEnv") as env, \
                    patch.object(rl_sac, "Replay", side_effect=lambda: replay_class(2)):
                env.N_OBS = agent.obs_dim
                with self.assertRaisesRegex(ValueError, "Incompatible"):
                    rl_sac.evaluate(checkpoint=path, output_dir=Path(tmp) / "eval")
                with self.assertRaisesRegex(ValueError, "Incompatible"):
                    rl_sac.train(resume=path, total_steps=1, run_dir=Path(tmp) / "train")
                env.assert_not_called()
            self.assertFalse((Path(tmp) / "eval").exists())
            self.assertFalse((Path(tmp) / "train").exists())

    def test_non_finite_checkpoint_rejected_before_model_mutation(self):
        agent = rl_sac.SAC(rl_sac.RcHeliEnv.N_OBS)
        before = {k: v.clone() for k, v in agent.actor.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.pt"
            agent.save(path, rl_sac.Replay(2), {}, rl_sac.default_config(0))
            payload = torch.load(path, weights_only=False)
            next(iter(payload["q2"].values())).fill_(float("nan"))
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                agent.load(path)
            for key, value in agent.actor.state_dict().items():
                self.assertTrue(torch.equal(value, before[key]))

    def test_missing_eval_model_never_constructs_game(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(rl_sac, "RcHeliEnv") as env:
            with self.assertRaises(FileNotFoundError):
                rl_sac.evaluate(checkpoint=Path(tmp) / "absent.pt")
            env.assert_not_called()

    def test_atomic_save_keeps_previous_checkpoint_if_replace_fails(self):
        agent = rl_sac.SAC(rl_sac.RcHeliEnv.N_OBS)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            path.write_bytes(b"original")
            with patch.object(rl_sac.os, "replace", side_effect=OSError("simulated failure")):
                with self.assertRaises(OSError):
                    agent.save(path, rl_sac.Replay(2), {}, rl_sac.default_config(0))
            self.assertEqual(path.read_bytes(), b"original")
            self.assertFalse(list(Path(tmp).glob("*.tmp")))


class ActionTests(unittest.TestCase):
    def test_five_dimensional_sampling_and_learner_gradients_remain_finite(self):
        rl_sac.seed_everything(42)
        self.assertEqual(tuple(rl_sac.ACTION_SCHEMA), ("roll", "pitch", "yaw", "collective", "release"))
        self.assertEqual(rl_sac.ACT_DIM, 5)
        self.assertEqual(len(rl_sac.OBS_SCHEMA), rl_sac.RcHeliEnv.N_OBS)
        agent = rl_sac.SAC(rl_sac.RcHeliEnv.N_OBS)
        obs = torch.zeros((256, rl_sac.RcHeliEnv.N_OBS))
        actions, logp, means = agent.actor.sample(obs)
        self.assertEqual(tuple(actions.shape), (256, 5))
        for value in (actions, logp, means):
            self.assertTrue(torch.isfinite(value).all())
        self.assertTrue((actions.abs() <= 1).all())
        self.assertTrue((actions[:, -1] > rl_sac.RELEASE_THRESHOLD).any())
        self.assertTrue((actions[:, -1] <= rl_sac.RELEASE_THRESHOLD).any())
        batch = (obs, actions.detach(), torch.ones(256), obs + .01, torch.zeros(256))
        losses = agent.update(batch)
        self.assertTrue(np.isfinite(losses).all())
        for model in (agent.actor, agent.q1, agent.q2):
            for parameter in model.parameters():
                self.assertTrue(torch.isfinite(parameter).all())
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(agent.actor.mu.weight.grad[-1].abs().sum().item(), 0.)
        self.assertGreater(agent.actor.log_std.weight.grad[-1].abs().sum().item(), 0.)


class FakeEnv:
    N_OBS = rl_sac.RcHeliEnv.N_OBS
    latest = None

    def __init__(self, **kwargs):
        FakeEnv.latest = self
        self.task = Mock()
        self.episode = 0
        self.target_id = (1, 2, 3)

    def reset(self):
        self.episode += 1
        return np.zeros(self.N_OBS, dtype=np.float32)

    def begin_episode(self):
        pass

    def launch_cont(self):
        return np.zeros(rl_sac.ACT_DIM, dtype=np.float32)

    def step_cont(self, action):
        if self.episode == 3:
            raise RuntimeError("simulated stale bridge")
        terminal = self.episode == 1
        return (np.zeros(self.N_OBS, dtype=np.float32), 1.0, True,
                dict(dist=1., target_id=self.target_id, terminated=terminal,
                     truncated=not terminal, success=terminal, success_known=True,
                     reason="success" if terminal else "time_limit", stage="test"))


class LifecycleTests(unittest.TestCase):
    def test_failed_transition_not_replayed_and_truncation_bootstraps(self):
        replay_class = rl_sac.Replay
        with tempfile.TemporaryDirectory() as tmp, patch.object(rl_sac, "RcHeliEnv", FakeEnv), \
                patch.object(rl_sac, "Replay", side_effect=lambda: replay_class(8)), \
                redirect_stdout(io.StringIO()):
            path = rl_sac.train(episodes=3, run_dir=Path(tmp) / "train")
            payload = torch.load(path / rl_sac.CKPT, weights_only=False)
            self.assertEqual(payload["training"]["total_steps"], 2)
            self.assertEqual(payload["replay"]["n"], 2)
            np.testing.assert_array_equal(payload["replay"]["don"], [1., 0.])
            with (path / "episodes.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([r["reason"] for r in rows], ["success", "time_limit", "environment_error"])
            FakeEnv.latest.task.release.assert_called_once()
            FakeEnv.latest.task.hard_reset.assert_not_called()

    def test_eval_reports_real_success_and_excludes_environment_errors(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(rl_sac, "RcHeliEnv", FakeEnv), \
                redirect_stdout(io.StringIO()):
            path = rl_sac.evaluate(episodes=3, baseline="random", output_dir=Path(tmp) / "eval")
            summary = json.loads((path / "summary.json").read_text())
            self.assertEqual(summary["success_rate"], .5)
            self.assertEqual(summary["environment_errors"], 1)
            self.assertAlmostEqual(summary["success_rate_all_attempts"], 1 / 3)
            FakeEnv.latest.task.release.assert_called_once()


class ContinuingEnv(FakeEnv):
    def reset(self):
        self.episode += 1
        self.tick = 0
        self.actions = []
        self._last_state = {"frame": 0, "heli": {"pos": [0., 0., 1.]}, "blips": [{"id": 4}]}
        return np.zeros(self.N_OBS, dtype=np.float32)

    def step_cont(self, action):
        self.tick += 1
        self.actions.append(np.asarray(action).copy())
        self._last_state = {"frame": self.tick * 10,
                            "heli": {"pos": [0., 0., 1. + self.tick * .01]},
                            "blips": [{"id": 4}]}
        return (np.full(self.N_OBS, self.tick / 100, dtype=np.float32), 1.0, False,
                dict(dist=10., target_id=self.target_id, terminated=False, truncated=False,
                     success=False, success_known=True, reason=None,
                     reward_terms={"progress": 1.0}, heli=self._last_state["heli"], stage="test"))


class BudgetAndTraceTests(unittest.TestCase):
    def test_verified_task_events_are_logged_without_counting_requests_or_proximity(self):
        class TaskEventsEnv(ContinuingEnv):
            def step_cont(self, action):
                obs, reward, done, info = super().step_cont(action)
                event = {2: "pickup", 4: "drop"}.get(self.tick)
                info.update(pickup_events=int(self.tick >= 2), drop_events=int(self.tick >= 4),
                            task_events=[{"type": event, "bomb_index": 0}] if event else [],
                            proximity_events=97, stage="carrying" if 2 <= self.tick < 4 else "pickup")
                return obs, reward, done, info
        with tempfile.TemporaryDirectory() as tmp, patch.object(rl_sac, "RcHeliEnv", TaskEventsEnv), \
                redirect_stdout(io.StringIO()):
            path = rl_sac.evaluate(steps=4, episodes=1, baseline="random", stall_limit=0,
                                   launch_steps=0, trace=True, output_dir=Path(tmp) / "events")
            summary = json.loads((path / "summary.json").read_text())
            self.assertEqual((summary["pickup_events"], summary["drop_events"]), (1, 1))
            self.assertEqual((summary["episodes_with_pickup"], summary["episodes_with_drop"]), (1, 1))
            self.assertEqual(summary["final_stage_counts"], {"pickup": 1})
            self.assertEqual(summary["successes"], 0)
            with (path / "episodes.csv").open() as f:
                row = next(csv.DictReader(f))
            self.assertEqual((row["pickup_events"], row["drop_events"], row["stage"]), ("1", "1", "pickup"))
            traces = [json.loads(line) for line in (path / "trace.jsonl").read_text().splitlines()]
            steps = [r for r in traces if r["event"] == "step"]
            self.assertEqual([r["info"]["task_events"] for r in steps],
                             [[], [{"type": "pickup", "bomb_index": 0}], [],
                              [{"type": "drop", "bomb_index": 0}]])

    def test_exact_run_step_budget_trace_and_resume_budget_restarts(self):
        replay_class = rl_sac.Replay
        with tempfile.TemporaryDirectory() as tmp, patch.object(rl_sac, "RcHeliEnv", ContinuingEnv), \
                patch.object(rl_sac, "Replay", side_effect=lambda: replay_class(8)), \
                patch.object(rl_sac.SAC, "act", return_value=np.full(rl_sac.ACT_DIM, .1, dtype=np.float32)), \
                redirect_stdout(io.StringIO()) as output:
            path = rl_sac.train(total_steps=5, stall_limit=0, min_progress=.2, launch_steps=2,
                                trace=True, run_dir=Path(tmp) / "first")
            summary = json.loads((path / "summary.json").read_text())
            self.assertEqual((summary["status"], summary["stop_reason"], summary["run_steps"]),
                             ("completed", "step_budget", 5))
            payload = torch.load(path / rl_sac.CKPT, weights_only=False)
            self.assertEqual(payload["replay"]["n"], 5)
            np.testing.assert_array_equal(payload["replay"]["don"], np.zeros(5))
            np.testing.assert_array_equal(payload["replay"]["act"][:2], np.zeros((2, rl_sac.ACT_DIM)))
            np.testing.assert_allclose(payload["replay"]["act"][2:], .1)
            traces = [json.loads(line) for line in (path / "trace.jsonl").read_text().splitlines()]
            steps = [row for row in traces if row["event"] == "step"]
            self.assertEqual(len(steps), 5)
            self.assertEqual(steps[-1]["reason"], "step_budget")
            self.assertTrue(steps[-1]["truncated"])
            self.assertFalse(steps[-1]["terminated"])
            self.assertEqual(steps[0]["reward_terms"], {"progress": 1.0})
            self.assertEqual(steps[0]["before_state"]["blips"], [{"id": 4}])
            self.assertEqual(steps[1]["obs"], steps[0]["next_obs"])
            self.assertIn('"event": "run_finished"', output.getvalue())
            resumed = rl_sac.train(resume=path / rl_sac.CKPT, total_steps=3,
                                  stall_limit=0, min_progress=.2, launch_steps=2,
                                  run_dir=Path(tmp) / "resumed")
            after = json.loads((resumed / "summary.json").read_text())
            self.assertEqual((after["run_steps"], after["total_steps"]), (3, 8))
            self.assertFalse((resumed / "trace.jsonl").exists())
            with self.assertRaisesRegex(ValueError, "min_progress"):
                rl_sac.train(resume=path / rl_sac.CKPT, total_steps=1,
                             stall_limit=0, min_progress=.3, launch_steps=2,
                             run_dir=Path(tmp) / "incompatible")
            self.assertFalse((Path(tmp) / "incompatible").exists())

    def test_eval_can_disable_stall_and_records_effective_limits(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(rl_sac, "RcHeliEnv", ContinuingEnv), \
                redirect_stdout(io.StringIO()):
            limited = rl_sac.evaluate(steps=75, episodes=1, baseline="random", stall_limit=2,
                                     launch_steps=0, output_dir=Path(tmp) / "limited")
            full = rl_sac.evaluate(steps=75, episodes=1, baseline="random", stall_limit=0,
                                  min_progress=.25, launch_steps=0, trace=True,
                                  output_dir=Path(tmp) / "full")
            a = json.loads((limited / "summary.json").read_text())
            b = json.loads((full / "summary.json").read_text())
            self.assertEqual(a["run_steps"], 4)
            self.assertEqual(a["end_reasons"], {"stall": 1})
            self.assertEqual(b["run_steps"], 75)
            self.assertEqual(b["end_reasons"], {"step_limit": 1})
            self.assertEqual((b["stall_limit"], b["min_progress"], b["launch_steps"]), (0, .25, 0))

    def test_min_progress_changes_stall_reset_behavior(self):
        class SlowProgressEnv(ContinuingEnv):
            def step_cont(self, action):
                obs, reward, done, info = super().step_cont(action)
                info["dist"] = 10. - self.tick * .1
                return obs, reward, done, info
        with tempfile.TemporaryDirectory() as tmp, patch.object(rl_sac, "RcHeliEnv", SlowProgressEnv), \
                redirect_stdout(io.StringIO()):
            limited = rl_sac.evaluate(steps=10, episodes=1, baseline="scripted", stall_limit=2,
                                     min_progress=1., output_dir=Path(tmp) / "large")
            full = rl_sac.evaluate(steps=10, episodes=1, baseline="scripted", stall_limit=2,
                                  min_progress=.05, output_dir=Path(tmp) / "small")
            self.assertEqual(json.loads((limited / "summary.json").read_text())["run_steps"], 4)
            self.assertEqual(json.loads((full / "summary.json").read_text())["run_steps"], 10)

    def test_time_budget_marks_last_valid_transition_as_truncated(self):
        clock = [0.0]
        class TimedEnv(ContinuingEnv):
            def step_cont(self, action):
                result = super().step_cont(action)
                clock[0] += .6
                return result
        replay_class = rl_sac.Replay
        with tempfile.TemporaryDirectory() as tmp, patch.object(rl_sac, "RcHeliEnv", TimedEnv), \
                patch.object(rl_sac, "Replay", side_effect=lambda: replay_class(8)), \
                patch.object(rl_sac.time, "perf_counter", side_effect=lambda: clock[0]), \
                redirect_stdout(io.StringIO()):
            path = rl_sac.train(minutes=1 / 60, stall_limit=0, trace=True, run_dir=Path(tmp) / "timed")
            summary = json.loads((path / "summary.json").read_text())
            self.assertEqual((summary["run_steps"], summary["stop_reason"]), (2, "time_budget"))
            payload = torch.load(path / rl_sac.CKPT, weights_only=False)
            np.testing.assert_array_equal(payload["replay"]["don"], [0., 0.])
            traces = [json.loads(line) for line in (path / "trace.jsonl").read_text().splitlines()]
            last = [row for row in traces if row["event"] == "step"][-1]
            self.assertEqual(last["reason"], "time_budget")
            self.assertTrue(last["truncated"])

    def test_reset_failure_still_writes_failed_summary_and_releases(self):
        class BrokenResetEnv(ContinuingEnv):
            def reset(self):
                raise RuntimeError("reset unavailable")
        replay_class = rl_sac.Replay
        with tempfile.TemporaryDirectory() as tmp, patch.object(rl_sac, "RcHeliEnv", BrokenResetEnv), \
                patch.object(rl_sac, "Replay", side_effect=lambda: replay_class(8)), \
                redirect_stdout(io.StringIO()):
            path = Path(tmp) / "failed"
            with self.assertRaisesRegex(RuntimeError, "reset failed"):
                rl_sac.train(total_steps=3, run_dir=path)
            summary = json.loads((path / "summary.json").read_text())
            self.assertEqual((summary["status"], summary["run_steps"]), ("failed", 0))
            FakeEnv.latest.task.release.assert_called_once()


class StatisticsTests(unittest.TestCase):
    def test_invalid_distance_is_not_a_close_approach(self):
        rows = [{"episode": str(i), "min_dist": d, "return": "0"}
                for i, d in enumerate(["-1", "n/a", "nan", "2", "8"], 1)]
        result = sac_stats.summarize(rows)
        self.assertEqual(result["valid_distances"], 2)
        self.assertEqual(result["distance_under_2_5m_fraction"], .5)
        self.assertIsNone(result["success_rate"])

    def test_run_id_and_episode_restart_split_runs(self):
        rows = [{"run_id": r, "episode": str(e)}
                for r, e in [("a", 1), ("a", 2), ("a", 1), ("b", 2)]]
        self.assertEqual([len(g) for g in sac_stats.split_runs(rows)], [2, 1, 1])


if __name__ == "__main__":
    unittest.main()
