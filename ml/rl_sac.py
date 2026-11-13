"""SAC for the versioned, fixed-step RC helicopter environment.

Actions: [roll, pitch, yaw, collective, release], each in [-1, 1].
The fifth output is a continuous SAC approximation to a thresholded request,
not a mixed discrete/continuous SAC policy. The environment applies release
only above RELEASE_THRESHOLD and when the pre-step state verifies carrying.
New runs never overwrite historical checkpoints. Resume starts a new run
with the learner/replay/RNG state restored; the game starts a fresh episode.

  python rl_sac.py --minutes 240 --seed 1
  python rl_sac.py --resume runs/<run>/checkpoint.pt --minutes 60
  python rl_sac.py --eval --checkpoint runs/<run>/checkpoint.pt --episodes 50
  python rl_sac.py --eval --baseline random --episodes 50 --seed 1
"""
import argparse
import csv
import os
import time
import json
import random
import tempfile
import uuid
import hashlib
import platform
from contextlib import nullcontext
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_train import (RcHeliEnv, LAUNCH_STEPS, MAX_STEPS, STALL_LIMIT,
                      ENV_VERSION, EpisodeProgress, OBS_SCHEMA, ACTION_SCHEMA,
                      RELEASE_THRESHOLD, PHASE_REWARD_VERSION, OBSTACLE_SCHEMA_VERSION,
                      DAMAGE_PENALTY_SCALE, DAMAGE_EPSILON, PROGRESS_SCOPE)

DEVICE = torch.device("cpu")
torch.set_num_threads(2)

ACT_DIM = len(ACTION_SCHEMA)
LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0
HIDDEN = 256
GAMMA = 0.99
TAU = 5e-3
LR = 3e-4
REPLAY_CAP = 200000
WARMUP = 600          # env steps before updates start
GRADS_PER_STEP = 2    # extract more from expensive env data
BATCH = 256
CHECKPOINT_VERSION = 4
CKPT = "checkpoint.pt"
ACTION_MODEL = "continuous-tanh-threshold-release-v1"


def _checkpoint_contract(obs_dim=None):
    return {"checkpoint_version": CHECKPOINT_VERSION, "env_version": ENV_VERSION,
            "obs_dim": RcHeliEnv.N_OBS if obs_dim is None else obs_dim,
            "observation_schema": list(OBS_SCHEMA), "action_schema": list(ACTION_SCHEMA),
            "release_threshold": RELEASE_THRESHOLD, "phase_reward_version": PHASE_REWARD_VERSION,
            "obstacle_schema_version": OBSTACLE_SCHEMA_VERSION,
            "progress_scope": PROGRESS_SCOPE,
            "damage_penalty_scale": DAMAGE_PENALTY_SCALE, "damage_epsilon": DAMAGE_EPSILON,
            "action_model": ACTION_MODEL}


def _validate_contract(value, expected, context):
    mismatches = [key for key, item in expected.items() if value.get(key) != item]
    if mismatches:
        raise ValueError("Incompatible %s contract: %s; historical checkpoints cannot be used "
                         "with this environment" % (context, ", ".join(mismatches)))

# ---------------- networks ----------------

class Actor(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, HIDDEN), nn.ReLU(),
                                 nn.Linear(HIDDEN, HIDDEN), nn.ReLU())
        self.mu = nn.Linear(HIDDEN, ACT_DIM)
        self.log_std = nn.Linear(HIDDEN, ACT_DIM)

    def forward(self, obs):
        h = self.net(obs)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs):
        mu, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        x = normal.rsample()                 # reparameterized
        a = torch.tanh(x)
        # log-prob with tanh correction (sum over action dims)
        logp = normal.log_prob(x) - torch.log(1 - a.pow(2) + 1e-6)
        return a, logp.sum(-1), torch.tanh(mu)


class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim + ACT_DIM, HIDDEN), nn.ReLU(),
                                 nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                                 nn.Linear(HIDDEN, 1))

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], -1)).squeeze(-1)


class SAC:
    def __init__(self, obs_dim, lr=LR):
        self.obs_dim = obs_dim
        self.actor = Actor(obs_dim)
        self.q1 = Critic(obs_dim)
        self.q2 = Critic(obs_dim)
        self.t1 = Critic(obs_dim)
        self.t2 = Critic(obs_dim)
        self.t1.load_state_dict(self.q1.state_dict())
        self.t2.load_state_dict(self.q2.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(list(self.q1.parameters()) +
                                      list(self.q2.parameters()), lr=lr)
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -float(ACT_DIM)
        self.updates = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def update(self, batch):
        obs, act, rew, nxt, don = batch
        with torch.no_grad():
            na, nlogp, _ = self.actor.sample(nxt)
            tq = torch.min(self.t1(nxt, na), self.t2(nxt, na))
            target = rew + GAMMA * (1 - don) * (tq - self.alpha.detach() * nlogp)
        q1 = self.q1(obs, act)
        q2 = self.q2(obs, act)
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        a, logp, _ = self.actor.sample(obs)
        q_new = torch.min(self.q1(obs, a), self.q2(obs, a))
        a_loss = (self.alpha.detach() * logp - q_new).mean()
        self.actor_opt.zero_grad()
        a_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha *
                       (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        for q, t in ((self.q1, self.t1), (self.q2, self.t2)):
            for p, pt in zip(q.parameters(), t.parameters()):
                pt.data.mul_(1 - TAU).add_(p.data, alpha=TAU)
        self.updates += 1
        return float(q_loss.detach()), float(a_loss.detach()), float(self.alpha.detach())

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            mu, log_std = self.actor(torch.as_tensor(obs, dtype=torch.float32))
            if deterministic:
                return torch.tanh(mu).numpy()
            std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp()
            return torch.tanh(mu + std * torch.randn_like(std)).numpy()

    def save(self, path, replay, training, config):
        contract = _checkpoint_contract(self.obs_dim)
        _validate_contract(config, contract, "checkpoint configuration")
        payload = {
            **contract,
            "actor": self.actor.state_dict(), "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(), "t1": self.t1.state_dict(),
            "t2": self.t2.state_dict(), "log_alpha": self.log_alpha.detach(),
            "actor_opt": self.actor_opt.state_dict(), "q_opt": self.q_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(), "updates": self.updates,
            "target_entropy": self.target_entropy, "replay": replay.state_dict(),
            "training": dict(training), "config": dict(config),
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state()},
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as f:
                torch.save(payload, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def load(self, path, replay=None, restore_rng=True):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError("Checkpoint does not exist: %s" % path)
        # Load only project-owned/trusted files: optimizer/replay/RNG use pickle.
        z = torch.load(path, map_location=DEVICE, weights_only=False)
        expected = _checkpoint_contract(self.obs_dim)
        _validate_contract(z, expected, "checkpoint")
        required = {"actor_opt", "q_opt", "alpha_opt", "replay", "training", "config", "rng"}
        if not required.issubset(z):
            raise ValueError("Checkpoint lacks complete training state")
        _validate_contract(z["config"], expected, "checkpoint configuration")
        # Reject corrupt/non-finite weights before mutating any live model.
        for key in ("actor", "q1", "q2", "t1", "t2"):
            if not all(torch.isfinite(value).all() for value in z[key].values()):
                raise ValueError("Checkpoint contains non-finite model state: %s" % key)
        if not torch.isfinite(z["log_alpha"]).all():
            raise ValueError("Checkpoint contains non-finite temperature")
        for key in ("actor", "q1", "q2", "t1", "t2"):
            getattr(self, key).load_state_dict(z[key])
        with torch.no_grad():
            self.log_alpha.copy_(z["log_alpha"])
        for key in ("actor_opt", "q_opt", "alpha_opt"):
            getattr(self, key).load_state_dict(z[key])
        self.target_entropy = float(z["target_entropy"])
        self.updates = int(z["updates"])
        if replay is not None:
            replay.load_state_dict(z["replay"])
        if restore_rng:
            random.setstate(z["rng"]["python"])
            np.random.set_state(z["rng"]["numpy"])
            torch.set_rng_state(z["rng"]["torch"])
        return {"training": z["training"], "config": z["config"]}


# ---------------- replay ----------------

class Replay:
    def __init__(self, cap=REPLAY_CAP):
        self.cap = cap
        self.obs = np.zeros((cap, RcHeliEnv.N_OBS), dtype=np.float32)
        self.act = np.zeros((cap, ACT_DIM), dtype=np.float32)
        self.rew = np.zeros(cap, dtype=np.float32)
        self.nxt = np.zeros((cap, RcHeliEnv.N_OBS), dtype=np.float32)
        self.don = np.zeros(cap, dtype=np.float32)
        self.n = 0
        self.i = 0

    def add(self, o, a, r, n, d):
        self.obs[self.i] = o; self.act[self.i] = a
        self.rew[self.i] = r; self.nxt[self.i] = n; self.don[self.i] = d
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, k):
        idx = np.random.randint(0, self.n, size=k)
        to = lambda x: torch.as_tensor(x, dtype=torch.float32)
        return (to(self.obs[idx]), to(self.act[idx]), to(self.rew[idx]),
                to(self.nxt[idx]), to(self.don[idx]))

    def state_dict(self):
        return {"cap": self.cap, "n": self.n, "i": self.i,
                **{k: getattr(self, k)[:self.n].copy()
                   for k in ("obs", "act", "rew", "nxt", "don")}}

    def load_state_dict(self, state):
        cap, n, i = int(state["cap"]), int(state["n"]), int(state["i"])
        if cap <= 0 or not 0 <= n <= cap or not 0 <= i < cap or (n < cap and i != n):
            raise ValueError("Invalid replay buffer metadata")
        expected = {"obs": (n, RcHeliEnv.N_OBS), "nxt": (n, RcHeliEnv.N_OBS),
                    "act": (n, ACT_DIM), "rew": (n,), "don": (n,)}
        for key, shape in expected.items():
            value = np.asarray(state[key])
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError("Invalid replay array: %s" % key)
        self.__init__(cap)
        for key in expected:
            getattr(self, key)[:n] = state[key]
        self.n, self.i = n, i


# ---------------- experiment lifecycle ----------------

LOG_FIELDS = ["run_id", "env_version", "episode", "seed", "steps", "total_steps", "run_steps",
              "return", "min_dist", "events", "proximity_events", "pickup_events", "drop_events",
              "success", "success_known", "failure",
              "terminated", "truncated", "reason", "stage", "alpha", "q_loss",
              "actor_loss", "updates", "reset_seconds", "episode_seconds", "error"]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _fingerprints():
    # These paths describe the configured installation, not a live game snapshot.
    from vc_task import GAME_DIR
    root = Path(__file__).parent
    paths = [root / name for name in ("rl_sac.py", "rl_train.py", "vc_task.py", "ml_client.py")]
    paths.extend([Path(GAME_DIR) / "reVC.exe", Path(GAME_DIR) / "data" / "main.scm"])
    results = {}
    for path in paths:
        try:
            digest = hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(chunk)
            results[str(path.resolve())] = digest.hexdigest()
        except OSError:
            results[str(path.resolve())] = None
    return results


def default_config(seed, stall_limit=STALL_LIMIT, min_progress=1.0, launch_steps=LAUNCH_STEPS):
    return {**_checkpoint_contract(), "hidden": HIDDEN, "gamma": GAMMA,
            "tau": TAU, "lr": LR, "replay_cap": REPLAY_CAP, "warmup": WARMUP,
            "grads_per_step": GRADS_PER_STEP, "batch": BATCH,
            "max_steps": MAX_STEPS, "stall_limit": stall_limit, "min_progress": min_progress,
            "launch_steps": launch_steps, "seed": seed,
            "sha256": _fingerprints(),
            "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                        "torch": torch.__version__},
            "seed_scope": "Python/NumPy/PyTorch only; game seed is not controlled"}


def _new_run(run_dir, config, prefix="sac"):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = "%s-%s-%s" % (prefix, stamp, uuid.uuid4().hex[:8])
    path = Path(run_dir) if run_dir else Path(__file__).parent / "runs" / run_id
    path.mkdir(parents=True, exist_ok=False)
    config = {**config, "run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat()}
    (path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path, config


def _checkpoint_path(checkpoint=None, run_dir=None):
    if checkpoint:
        path = Path(checkpoint)
    elif run_dir:
        path = Path(run_dir) / CKPT
    else:
        raise ValueError("Specify --checkpoint PATH or --run-dir DIR; no implicit legacy model")
    if not path.is_file():
        raise FileNotFoundError("Checkpoint does not exist: %s" % path)
    return path


def _release(env):
    if env is not None:
        try:
            env.task.release()
        except Exception as exc:
            print("control release failed: %s" % exc)


def _row(config, episode, steps, total, ret, mn, info, reason, truncated,
         reset_seconds, episode_seconds, error="", agent=None, losses=()):
    qloss = np.mean([x[0] for x in losses]) if losses else 0.0
    aloss = np.mean([x[1] for x in losses]) if losses else 0.0
    return dict(run_id=config["run_id"], env_version=ENV_VERSION, episode=episode,
                seed=config["seed"], steps=steps, total_steps=total, run_steps=total, **{"return": ret},
                min_dist=mn if np.isfinite(mn) else "", events=info.get("episode_events", 0),
                pickup_events=info.get("pickup_events", 0), drop_events=info.get("drop_events", 0),
                proximity_events=info.get("proximity_events", 0), success=bool(info.get("success", False)),
                success_known=bool(info.get("success_known", False)),
                failure=bool(info.get("failure", False)), terminated=bool(info.get("terminated", False)),
                truncated=truncated, reason=reason, stage=info.get("stage", "unknown"),
                alpha=float(agent.alpha.detach()) if agent else "", q_loss=float(qloss),
                actor_loss=float(aloss), updates=agent.updates if agent else 0,
                reset_seconds=reset_seconds, episode_seconds=episode_seconds, error=error)


def _reset(env):
    start = time.perf_counter()
    last_error = None
    for retry in range(3):
        try:
            obs = env.reset()
            if not np.isfinite(obs).all():
                raise ValueError("Non-finite reset observation")
            env.begin_episode()
            return obs, time.perf_counter() - start
        except Exception as exc:
            last_error = exc
            if retry < 2:
                env.task.hard_reset()
    raise RuntimeError("env.reset failed after 3 attempts") from last_error


def _validate_controls(stall_limit, min_progress, launch_steps):
    if stall_limit < 0 or int(stall_limit) != stall_limit:
        raise ValueError("stall_limit must be a nonnegative integer (0 disables stall)")
    if not np.isfinite(min_progress) or min_progress <= 0:
        raise ValueError("min_progress must be finite and positive")
    if launch_steps < 0 or int(launch_steps) != launch_steps:
        raise ValueError("launch_steps must be a nonnegative integer")


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _trace(stream, event, **values):
    if stream is not None:
        stream.write(json.dumps(_json_safe(dict(event=event, **values)),
                                ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()


def _outcomes(rows):
    valid = [r for r in rows if not r["error"]]
    known = [r for r in valid if r["success_known"]]
    successes = sum(r["success"] for r in known)
    return {"episodes": len(rows), "valid_episodes": len(valid),
            "environment_errors": len(rows) - len(valid),
            "outcome_known_episodes": len(known), "outcome_unknown_episodes": len(valid) - len(known),
            "successes": successes, "success_rate": successes / len(known) if known else None,
            "success_rate_denominator": "episodes without environment errors and with success telemetry",
            "success_rate_all_attempts": successes / len(rows) if rows else None,
            "pickup_events": sum(r["pickup_events"] for r in rows),
            "drop_events": sum(r["drop_events"] for r in rows),
            "episodes_with_pickup": sum(r["pickup_events"] > 0 for r in rows),
            "episodes_with_drop": sum(r["drop_events"] > 0 for r in rows),
            "task_event_count_scope": "last verified cumulative telemetry from all attempted episodes, "
                                      "including partial episodes ending in environment_error",
            "final_stage_counts": dict(Counter(r["stage"] for r in rows)),
            "end_reasons": dict(Counter(r["reason"] for r in rows))}


def _complete(path, summary):
    tmp = path / "summary.json.tmp"
    tmp.write_text(json.dumps(_json_safe(summary), indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path / "summary.json")
    print(json.dumps({"event": "run_finished", **summary}, ensure_ascii=False), flush=True)
    print("%s: %s; summary: %s" %
          (summary["status"], summary.get("stop_reason", "evaluation_finished"),
           (path / "summary.json").resolve()), flush=True)


def _budget_reason(deadline, run_steps, total_steps):
    if total_steps is not None and run_steps >= total_steps:
        return "step_budget"
    if time.perf_counter() >= deadline:
        return "time_budget"
    return None


def train(minutes=None, resume=False, episodes=None, seed=0, run_dir=None, checkpoint=None,
          stall_limit=STALL_LIMIT, min_progress=1.0, launch_steps=LAUNCH_STEPS,
          total_steps=None, trace=False):
    _validate_controls(stall_limit, min_progress, launch_steps)
    if episodes is not None and (episodes <= 0 or int(episodes) != episodes):
        raise ValueError("episodes must be a positive integer")
    if minutes is not None and (not np.isfinite(minutes) or minutes <= 0):
        raise ValueError("minutes must be finite and positive")
    if total_steps is not None and (total_steps <= 0 or int(total_steps) != total_steps):
        raise ValueError("total_steps must be a positive integer")
    if episodes is None and minutes is None and total_steps is None:
        minutes = 5.0
    seed_everything(seed)
    agent = SAC(RcHeliEnv.N_OBS)
    buf = Replay()
    state = {"episode": 0, "total_steps": 0}
    config = default_config(seed, stall_limit, min_progress, launch_steps)
    config.update(mode="training", minutes_budget=minutes, episodes_budget=episodes,
                  total_steps_budget=total_steps, trace=bool(trace))
    if resume:
        explicit = resume if isinstance(resume, (str, Path)) and resume else checkpoint
        source = _checkpoint_path(explicit, run_dir)
        restored = agent.load(source, buf)
        old_config = restored["config"]
        for key in ("gamma", "tau", "lr", "warmup", "grads_per_step", "batch",
                    "max_steps", "stall_limit", "min_progress", "launch_steps", "hidden"):
            old_value = old_config.get(key)
            if old_value != config[key]:
                raise ValueError("Resume configuration mismatch: %s" % key)
        state = dict(restored["training"])
        config["seed"] = old_config["seed"]
        config["resume_from"] = str(source.resolve())
        config["parent_run_id"] = old_config["run_id"]
        if run_dir and Path(run_dir).resolve() == source.parent.resolve():
            run_dir = None
    elif checkpoint:
        raise ValueError("Use --resume with --checkpoint when restoring training")
    path, config = _new_run(run_dir, config)
    env, rows = None, []
    run_start = time.perf_counter()
    deadline = run_start + minutes * 60 if minutes is not None else float("inf")
    run_steps, initial_updates = 0, agent.updates
    status, fatal_error, stop_reason = "completed", "", "episode_budget"
    print("run directory: %s" % path.resolve(), flush=True)
    try:
        env = RcHeliEnv(verbose=True)
        with (path / "episodes.csv").open("w", newline="", encoding="utf-8") as logf, \
                ((path / "trace.jsonl").open("w", encoding="utf-8") if trace else nullcontext(None)) as tracef:
            logger = csv.DictWriter(logf, fieldnames=LOG_FIELDS)
            logger.writeheader()
            while (not _budget_reason(deadline, run_steps, total_steps)
                   and (episodes is None or len(rows) < episodes)):
                episode = state["episode"] + 1
                obs, reset_seconds = _reset(env)
                progress = EpisodeProgress(stall_limit=stall_limit, min_progress=min_progress)
                progress.reset(target_id=getattr(env, "target_id", None))
                _trace(tracef, "reset", episode=episode, obs=obs,
                       state=getattr(env, "_last_state", None), target_id=getattr(env, "target_id", None))
                start = time.perf_counter()
                ret, mn, steps = 0.0, float("inf"), 0
                info, losses, error = {}, [], ""
                reason, truncated = "step_limit", True
                for t in range(MAX_STEPS):
                    budget = _budget_reason(deadline, run_steps, total_steps)
                    if budget:
                        reason = budget
                        break
                    action = env.launch_cont() if t < launch_steps else agent.act(obs)
                    before_state = getattr(env, "_last_state", None) if trace else None
                    try:
                        nxt, reward, done, info = env.step_cont(action)
                        if not (np.isfinite(nxt).all() and np.isfinite(reward)):
                            raise ValueError("Non-finite environment transition")
                    except Exception as exc:
                        reason, error = "environment_error", str(exc)
                        _trace(tracef, "environment_error", episode=episode, step=t + 1,
                               obs=obs, action=action, error=error, before_state=before_state)
                        break
                    terminated = bool(info.get("terminated", False))
                    # External budgets/stall are truncations, so bootstrap them.
                    buf.add(obs, action, reward, nxt, float(terminated))
                    ret += reward
                    steps += 1
                    run_steps += 1
                    state["total_steps"] += 1
                    dist = info.get("dist")
                    if dist is not None and np.isfinite(dist) and dist >= 0:
                        mn = min(mn, dist)
                    if buf.n >= WARMUP:
                        for _ in range(GRADS_PER_STEP):
                            losses.append(agent.update(buf.sample(BATCH)))
                    stalled = progress.update(info) if stall_limit else False
                    budget = _budget_reason(deadline, run_steps, total_steps)
                    ended = terminated or info.get("truncated", False) or done or stalled
                    if ended:
                        truncated = not terminated
                        reason = info.get("reason") or (progress.reason if stalled else "environment_truncated")
                    elif budget or t + 1 == MAX_STEPS:
                        ended, truncated = True, True
                        reason = budget or "step_limit"
                    _trace(tracef, "step", episode=episode, step=t + 1, run_steps=run_steps,
                           phase="launch" if t < launch_steps else "policy",
                           obs=obs, next_obs=nxt, action=action, reward=reward,
                           reward_terms=info.get("reward_terms"), target_id=info.get("target_id"),
                           info=info, before_state=before_state,
                           after_state=getattr(env, "_last_state", None),
                           terminated=terminated, truncated=bool(ended and truncated),
                           reason=reason if ended else None)
                    obs = nxt
                    if ended:
                        break
                state["episode"] += 1
                row = _row(config, state["episode"], steps, state["total_steps"], ret, mn,
                           info, reason, truncated, reset_seconds, time.perf_counter() - start,
                           error, agent, losses)
                row["run_steps"] = run_steps
                rows.append(row)
                logger.writerow(row)
                logf.flush()
                _trace(tracef, "episode_end", **row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                if len(rows) % 5 == 0:
                    agent.save(path / CKPT, buf, state, config)
                if error and not _budget_reason(deadline, run_steps, total_steps) and (episodes is None or len(rows) < episodes):
                    env.task.hard_reset()
            stop_reason = _budget_reason(deadline, run_steps, total_steps) or "episode_budget"
    except BaseException as exc:
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        stop_reason, fatal_error = status, repr(exc)
        raise
    finally:
        _release(env)
        try:
            agent.save(path / CKPT, buf, state, config)
        except BaseException as exc:
            status, stop_reason, fatal_error = "failed", "checkpoint_error", repr(exc)
            raise
        finally:
            summary = {"run_id": config["run_id"], "status": status, "stop_reason": stop_reason,
                       "error": fatal_error, "run_steps": run_steps, "total_steps": state["total_steps"],
                       "total_episodes": state["episode"], "updates": agent.updates,
                       "run_updates": agent.updates - initial_updates,
                       "elapsed_seconds": time.perf_counter() - run_start,
                       "reset_seconds": sum(r["reset_seconds"] for r in rows),
                       "total_steps_budget": total_steps, "minutes_budget": minutes,
                       "episodes_budget": episodes, "stall_limit": stall_limit,
                       "min_progress": min_progress, "launch_steps": launch_steps,
                       "checkpoint": str((path / CKPT).resolve()), "trace": bool(trace),
                       **_outcomes(rows)}
            _complete(path, summary)
    return path


def evaluate(steps=MAX_STEPS, episodes=10, seed=0, checkpoint=None,
             run_dir=None, baseline="policy", output_dir=None,
             stall_limit=STALL_LIMIT, min_progress=1.0, launch_steps=LAUNCH_STEPS, trace=False):
    _validate_controls(stall_limit, min_progress, launch_steps)
    if steps <= 0 or episodes <= 0:
        raise ValueError("steps and episodes must be positive")
    agent = None
    # Validate/load before constructing the live environment.
    if baseline == "policy":
        source = _checkpoint_path(checkpoint, run_dir)
        agent = SAC(RcHeliEnv.N_OBS)
        agent.load(source, restore_rng=False)
    elif baseline not in ("random", "scripted"):
        raise ValueError("Unknown baseline: %s" % baseline)
    seed_everything(seed)
    config = default_config(seed, stall_limit, min_progress, launch_steps)
    config.update(mode="evaluation", baseline=baseline, episodes=episodes, steps=steps, trace=bool(trace))
    if agent:
        config["checkpoint"] = str(source.resolve())
    path, config = _new_run(output_dir, config, prefix="eval")
    env, rows = None, []
    status, fatal_error, run_steps = "completed", "", 0
    start_run = time.perf_counter()
    try:
        env = RcHeliEnv(verbose=True)
        with (path / "episodes.csv").open("w", newline="", encoding="utf-8") as logf, \
                ((path / "trace.jsonl").open("w", encoding="utf-8") if trace else nullcontext(None)) as tracef:
            logger = csv.DictWriter(logf, fieldnames=LOG_FIELDS)
            logger.writeheader()
            for episode in range(1, episodes + 1):
                obs, reset_seconds = _reset(env)
                progress = EpisodeProgress(stall_limit=stall_limit, min_progress=min_progress)
                progress.reset(target_id=getattr(env, "target_id", None))
                _trace(tracef, "reset", episode=episode, obs=obs,
                       state=getattr(env, "_last_state", None), target_id=getattr(env, "target_id", None))
                start = time.perf_counter()
                ret, mn, valid_steps = 0.0, float("inf"), 0
                reason, truncated, info, error = "step_limit", True, {}, ""
                for t in range(steps):
                    if t < launch_steps or baseline == "scripted":
                        action = env.launch_cont()
                    elif baseline == "random":
                        action = np.random.uniform(-1, 1, ACT_DIM).astype(np.float32)
                    else:
                        action = agent.act(obs, deterministic=True)
                    before_state = getattr(env, "_last_state", None) if trace else None
                    try:
                        nxt, reward, done, info = env.step_cont(action)
                        if not (np.isfinite(nxt).all() and np.isfinite(reward)):
                            raise ValueError("Non-finite environment transition")
                    except Exception as exc:
                        reason, error = "environment_error", str(exc)
                        _trace(tracef, "environment_error", episode=episode, step=t + 1,
                               obs=obs, action=action, error=error, before_state=before_state)
                        break
                    ret += reward
                    valid_steps += 1
                    run_steps += 1
                    dist = info.get("dist")
                    if dist is not None and np.isfinite(dist) and dist >= 0:
                        mn = min(mn, dist)
                    stalled = progress.update(info) if stall_limit else False
                    terminated = bool(info.get("terminated", False))
                    ended = terminated or info.get("truncated", False) or done or stalled
                    if ended:
                        truncated = not terminated
                        reason = info.get("reason") or (progress.reason if stalled else "environment_truncated")
                    elif t + 1 == steps:
                        ended, truncated, reason = True, True, "step_limit"
                    _trace(tracef, "step", episode=episode, step=t + 1, run_steps=run_steps,
                           phase="launch" if t < launch_steps else baseline,
                           obs=obs, next_obs=nxt, action=action, reward=reward,
                           reward_terms=info.get("reward_terms"), target_id=info.get("target_id"),
                           info=info, before_state=before_state,
                           after_state=getattr(env, "_last_state", None),
                           terminated=terminated, truncated=bool(ended and truncated),
                           reason=reason if ended else None)
                    obs = nxt
                    if ended:
                        break
                row = _row(config, episode, valid_steps, run_steps, ret, mn, info, reason, truncated,
                           reset_seconds, time.perf_counter() - start, error, agent)
                rows.append(row)
                logger.writerow(row)
                logf.flush()
                _trace(tracef, "episode_end", **row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                if error and episode < episodes:
                    env.task.hard_reset()
    except BaseException as exc:
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        fatal_error = repr(exc)
        raise
    finally:
        _release(env)
        summary = {"run_id": config["run_id"], "status": status, "error": fatal_error,
                   "stop_reason": "episode_budget" if status == "completed" else status,
                   "run_steps": run_steps, "elapsed_seconds": time.perf_counter() - start_run,
                   "baseline": baseline, "evaluation_step_limit": steps, "stall_limit": stall_limit,
                   "min_progress": min_progress, "launch_steps": launch_steps, "trace": bool(trace),
                   "seed_scope": config["seed_scope"], **_outcomes(rows)}
        _complete(path, summary)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--minutes", type=float)
    ap.add_argument("--total-steps", type=int, help="Valid transitions collected in this run, including after resume")
    ap.add_argument("--stall-limit", type=int, default=STALL_LIMIT, help="Non-progress step limit; 0 disables external stall")
    ap.add_argument("--min-progress", type=float, default=1.0, help="Distance improvement in metres that resets stall")
    ap.add_argument("--launch-steps", type=int, default=LAUNCH_STEPS)
    ap.add_argument("--trace", action="store_true", help="Write compact per-step trace.jsonl")
    ap.add_argument("--episodes", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-dir")
    ap.add_argument("--checkpoint")
    ap.add_argument("--resume", nargs="?", const=True, default=False, metavar="CHECKPOINT")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--eval-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--baseline", choices=("policy", "random", "scripted"), default="policy")
    ap.add_argument("--output-dir", help="New evaluation output directory")
    args = ap.parse_args()
    if args.eval:
        if args.total_steps is not None:
            ap.error("--total-steps is training-only; use --eval-steps for the evaluation episode limit")
        evaluate(args.eval_steps, 10 if args.episodes is None else args.episodes, args.seed, args.checkpoint,
                 args.run_dir, args.baseline, args.output_dir,
                 stall_limit=args.stall_limit, min_progress=args.min_progress,
                 launch_steps=args.launch_steps, trace=args.trace)
    else:
        train(args.minutes, args.resume, args.episodes, args.seed, args.run_dir, args.checkpoint,
              stall_limit=args.stall_limit, min_progress=args.min_progress,
              launch_steps=args.launch_steps, total_steps=args.total_steps, trace=args.trace)


if __name__ == "__main__":
    main()
