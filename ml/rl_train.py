"""Versioned DQN baseline and shared RC-helicopter environment.

The bridge executes 100ms simulated locksteps. Observations use relative target
geometry, kinematics and native mission state. Task rewards use verified native
pickup/placement transitions; release is a separate, carrying-gated action.

DQN policy artifacts support explicit --warm-start only, with fresh optimizer,
replay and exploration schedule. Legacy checkpoints are incompatible.
"""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np

import vc_task
from ml_client import GameDeadError

# ---------------- env ----------------

ENV_VERSION = "rc-heli-v5"
ACTION_SCHEMA = ("roll", "pitch", "yaw", "collective", "release")
RELEASE_THRESHOLD = 0.5
PHASE_REWARD_VERSION = "verified-events-damage-no-carry-shaping-v3"
OBSTACLE_SCHEMA_VERSION = "collision-rays-v2"
PROGRESS_SCOPE = "pickup_only"
DAMAGE_PENALTY_SCALE = 150.  # a 0.02 health loss costs 3; smaller damage still counts
DAMAGE_EPSILON = 1e-6       # ignore single-precision telemetry jitter, not minor collisions
OBS_SCHEMA = (
    "target_dx_100", "target_dy_100", "target_dz_20", "heading_error_sin", "heading_error_cos",
    "velocity_x_5", "velocity_y_5", "velocity_z_5", "world_z_30", "speed_5",
) + tuple("ray_%d" % i for i in range(13)) + ("health",) + tuple(
    "%s_%s" % (vector, axis) for vector in ("up", "forward", "turnVel") for axis in ("x", "y", "z")
) + ("mission_valid",) + tuple(
    "bomb_%d_%s" % (i, state) for i in range(4) for state in ("available", "carried", "falling", "placed")
) + tuple("completed_%d" % i for i in range(4)) + (
    "timer_not_started", "timer_pending", "timer_running", "timer_remaining_fraction",
    "stage_pickup", "stage_carrying", "stage_falling", "stage_complete",
    "objective_kind_bomb", "objective_kind_drop",
) + tuple("objective_index_%d" % i for i in range(4)) + tuple(
    "pickup_rewarded_%d" % i for i in range(4)
) + tuple("placement_rewarded_%d" % i for i in range(4))
# Physical roll/pitch/yaw signs follow Vehicle.cpp. Circle is independent
# from positive Cross collective and negative Square collective.
ACTION_VECTORS = np.asarray([
    [0, 0, 0, 0, 0], [0, 0, 0, .35, 0], [0, 0, 0, -.35, 0],
    [.25, 0, 0, .2, 0], [-.25, 0, 0, .2, 0],
    [0, -.25, 0, .2, 0], [0, .25, 0, .2, 0],
    [0, 0, -.25, .2, 0], [0, 0, .25, .2, 0],
    [0, 0, 0, 0, 1],
], dtype=np.float32)
ACTIONS = list(zip(("neutral", "climb", "descend", "roll+", "roll-",
                    "pitch-", "pitch+", "yaw-", "yaw+", "release"), ACTION_VECTORS))
N_ACT = len(ACTIONS)

STEP_FRAMES = 10     # fixed lockstep: 10 x 10ms simulated time
MAX_STEPS = 1500     # step cap: 150 seconds of simulated time
STALL_LIMIT = 60     # external truncation only; no hidden stall reward penalty
EPS_FLOOR = 0.15     # minimum exploration probability for the DQN baseline
LAUNCH_STEPS = 16    # mild vertical input only; real-flight calibration pending


class EpisodeProgress:
    """Per-target progress for external truncation, not MDP termination."""
    def __init__(self, stall_limit=STALL_LIMIT, min_progress=1.0):
        self.stall_limit = stall_limit
        self.min_progress = min_progress
        self.reset()

    def reset(self, target_id=None):
        self.target_id = target_id
        self.best = None
        self.stall = 0
        self.reason = None

    def update(self, info):
        target = info.get("target_id")
        if target != self.target_id or info.get("target_changed", False):
            self.reset(target)
        distance = info.get("dist")
        if distance is not None and np.isfinite(distance):
            if self.best is None or distance < self.best - self.min_progress:
                self.best = float(distance)
                self.stall = 0
            else:
                self.stall += 1
        else:
            self.stall += 1
        stopped = self.stall > self.stall_limit
        self.reason = "stall" if stopped else None
        return stopped

class RcHeliEnv:
    N_OBS = len(OBS_SCHEMA)

    def __init__(self, verbose=True, task=None, require_telemetry=True):
        self.task = task if task is not None else vc_task.VcDemolitionTask(verbose=verbose)
        self.require_telemetry = require_telemetry
        self.total_events = 0
        self.step_frames = STEP_FRAMES
        if verbose:
            print("lockstep: %d frames/step (100ms simulated time)" % self.step_frames)

    @staticmethod
    def _integer(value, lower, upper=None):
        return (not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer))
                and value >= lower and (upper is None or value <= upper))

    def _validate_metadata(self, st):
        for key in ("episodeId", "missionSuccessCount"):
            if self.require_telemetry and not self._integer(st.get(key), 0):
                raise RuntimeError("bridge telemetry missing or invalid: " + key)

    def _mission(self, st, required=True):
        """Only validated native task globals define the mission phase."""
        mission = st.get("mission")
        valid = isinstance(mission, dict) and mission.get("valid") is True
        if valid:
            states = mission.get("bomb_states")
            valid = (mission.get("id") == 19 and self._integer(mission.get("epoch"), 0)
                     and isinstance(states, (list, tuple)) and len(states) == 4
                     and all(self._integer(x, 0, 3) for x in states)
                     and sum(x == 1 for x in states) <= 1
                     and self._integer(mission.get("completed_mask"), 0, 15)
                     and self._integer(mission.get("timer_state"), 0, 2)
                     and self._integer(mission.get("remaining_ms"), 0, 420000)
                     and (mission["timer_state"] != 0 or mission["remaining_ms"] == 0))
        if not valid:
            if required:
                raise RuntimeError("native mission telemetry missing or invalid; discard transition")
            return None
        return dict(mission, bomb_states=list(mission["bomb_states"]))

    @staticmethod
    def _vector(value, length=3):
        try:
            array = np.asarray(value, dtype=np.float64)
            return array.shape == (length,) and np.isfinite(array).all()
        except (TypeError, ValueError):
            return False

    def _validate_heli(self, st, required=True):
        h = st.get("heli")
        valid = isinstance(h, dict) and all(self._vector(h.get(key))
                    for key in ("pos", "vel", "up", "forward", "turnVel"))
        if valid:
            valid = self._vector(h.get("rays"), 13) and all(
                isinstance(h.get(key), (int, float, np.number)) and np.isfinite(h[key])
                for key in ("heading", "speed", "health"))
        if not valid and required:
            raise RuntimeError("heli observation missing or invalid; discard transition")
        return bool(valid)

    def _objectives(self, st, required=True):
        raw = st.get("objectives")
        valid = isinstance(raw, (list, tuple)) and len(raw) == 8
        result = {}
        if valid:
            for item in raw:
                if (not isinstance(item, dict) or not self._integer(item.get("kind"), 1, 2)
                        or not self._integer(item.get("index"), 0, 3)
                        or not isinstance(item.get("valid"), (bool, np.bool_))
                        or not self._vector(item.get("pos"))):
                    valid = False
                    break
                key = (int(item["kind"]), int(item["index"]))
                if key in result:
                    valid = False
                    break
                result[key] = dict(item, kind=key[0], index=key[1], pos=tuple(item["pos"]))
            valid = valid and len(result) == 8
        if not valid:
            if required:
                raise RuntimeError("native objectives missing or invalid; discard transition")
            return {}
        return result

    @staticmethod
    def _stage(mission, success=False):
        if success:
            return "complete"
        if mission is None:
            return "unknown"
        if mission["completed_mask"] == 15:
            return "complete"
        if 1 in mission["bomb_states"]:
            return "carrying"
        if 2 in mission["bomb_states"]:
            return "falling"
        return "pickup"

    @staticmethod
    def _tgt_key(tgt):
        return None if tgt is None else (int(tgt["kind"]), int(tgt["index"]))

    @property
    def target_id(self):
        key = self._tgt_key(getattr(self, "_tgt", None))
        return None if key is None else (self.episode_id,) + key

    def _pick_target(self, st, mission, objectives, old=None):
        stage = self._stage(mission)
        if stage == "complete":
            return None
        if stage == "pickup":
            eligible = [(1, i) for i, state in enumerate(mission["bomb_states"]) if state == 0]
        else:
            # Carrying and falling both refer to a destination. Do not point
            # a falling bomb episode back toward a new pickup until resolved.
            eligible = [(2, i) for i in range(4) if not mission["completed_mask"] & (1 << i)]
        available = {key: objectives[key] for key in eligible if objectives[key]["valid"]}
        key = self._tgt_key(old)
        if key in available:
            return available[key]  # refresh coordinates, preserve semantic ID
        if not available:
            raise RuntimeError("no valid objective for native mission phase; discard transition")
        pos = np.asarray(st["heli"]["pos"])
        return min(available.values(), key=lambda target: float(np.linalg.norm(np.asarray(target["pos"]) - pos)))

    def _hdg(self, h):
        return float(h["heading"])

    def _features(self, st, tgt, mission=None, stage=None):
        h = st.get("heli")
        base = np.zeros(24, dtype=np.float32)
        attitude = np.zeros(9, dtype=np.float32)
        if self._validate_heli(st, required=False):
            hx, hy, hz = h["pos"]
            if tgt is not None:
                dx, dy, dz = np.asarray(tgt["pos"]) - np.asarray(h["pos"])
                angle = np.arctan2(dy, dx) - self._hdg(h)
                base[:5] = [dx / 100., dy / 100., dz / 20., np.sin(angle), np.cos(angle)]
            base[5:10] = [*(np.asarray(h["vel"]) * 5.), hz / 30., h["speed"] * 5.]
            base[10:23] = h["rays"]
            base[23] = h["health"]
            attitude[:] = [v for key in ("up", "forward", "turnVel") for v in h[key]]
        if mission is None:
            mission_features = [0.] * 25
        else:
            mission_features = [1.] + [float(state == category) for state in mission["bomb_states"] for category in range(4)]
            mission_features += [float(bool(mission["completed_mask"] & (1 << i))) for i in range(4)]
            mission_features += [float(mission["timer_state"] == i) for i in range(3)]
            mission_features += [mission["remaining_ms"] / 420000.]
        stage = stage or self._stage(mission)
        phase = [float(stage == name) for name in ("pickup", "carrying", "falling", "complete")]
        key = self._tgt_key(tgt)
        goal = [float(key is not None and key[0] == kind) for kind in (1, 2)]
        goal += [float(key is not None and key[1] == i) for i in range(4)]
        history = [float(i in self._pickup_rewarded) for i in range(4)]
        history += [float(i in self._placement_rewarded) for i in range(4)]
        features = np.concatenate([base, attitude, mission_features, phase, goal, history])
        if features.shape != (self.N_OBS,) or not np.isfinite(features).all():
            raise RuntimeError("invalid observation schema or values; discard transition")
        return np.clip(features, -3., 3.).astype(np.float32)

    def reset(self):
        self.task.reset_force(natural_wait=5.0)
        st = self.task.observe()  # reset_force returns a frozen native snapshot
        self._validate_metadata(st)
        mission = self._mission(st)
        objectives = self._objectives(st)
        self._validate_heli(st)
        if (mission["completed_mask"] != 0 or any(mission["bomb_states"])
                or mission["timer_state"] != 0):
            raise RuntimeError("reset returned stale native mission progress")
        target = self._pick_target(st, mission, objectives)
        self.episode_id = st.get("episodeId")
        self._pid = st.get("pid")
        self._mission_epoch = mission["epoch"]
        self._mission_prev = mission
        self._success_baseline = st.get("missionSuccessCount")
        self._success_awarded = False
        self._pickup_rewarded = set()
        self._placement_rewarded = set()
        self.episode_events = self.pickup_events = self.drop_events = 0
        self.proximity_events = 0  # retained log field; proximity no longer pays
        self.target_changes = 0
        self._launch_k = 0
        self._tgt = target
        self._dist_prev = self._dist(st)
        self._health_prev = st["heli"]["health"]
        self._last_state = st
        self._finished = False
        return self._features(st, target, mission)

    def _dist(self, st):
        if not st.get("heli") or self._tgt is None:
            return None
        return float(np.linalg.norm(np.asarray(self._tgt["pos"]) - np.asarray(st["heli"]["pos"])))

    def step(self, action):
        name, vector = ACTIONS[action]
        obs, reward, done, info = self.step_cont(vector)
        info["action"] = name
        return obs, reward, done, info

    def step_cont(self, act):
        """Five normalized inputs; release > .5 sends Circle only if carrying."""
        if getattr(self, "_finished", False):
            raise RuntimeError("step called after termination; reset first")
        vector = np.asarray(act, dtype=np.float32)
        if vector.shape != (len(ACTION_SCHEMA),) or not np.isfinite(vector).all():
            raise ValueError("action must contain five finite values")
        roll, pitch, yaw, collective, release = np.clip(vector, -1., 1.)
        requested = bool(release > RELEASE_THRESHOLD)
        effective = requested and 1 in self._mission_prev["bomb_states"]
        st = self.task.step(frames=self.step_frames, lx=-float(roll), ly=float(pitch), rx=float(yaw),
                            cross=max(float(collective), 0.), brake_val=max(-float(collective), 0.),
                            buttons=("Circle",) if effective else ())
        result = self._finish_step(st, action_name="c(%+.2f,%+.2f,%+.2f,%+.2f,%+.2f)" % tuple(vector))
        result[3].update(release_requested=requested, release_effective=effective,
                         release_threshold=RELEASE_THRESHOLD)
        return result

    def _finish_step(self, st_now, action_name=""):
        if getattr(self, "_finished", False):
            raise RuntimeError("step called after termination; reset first")
        self._validate_metadata(st_now)
        if (st_now.get("recovered") or st_now.get("episodeId") != self.episode_id
                or st_now.get("pid") != self._pid):
            raise RuntimeError("episode changed during step; discard transition")
        count = st_now.get("missionSuccessCount")
        success_known = self._success_baseline is not None and count is not None
        if success_known and count < self._success_baseline:
            raise RuntimeError("success counter regressed; discard transition")
        success = bool(success_known and count > self._success_baseline)
        ended = bool(vc_task.VcDemolitionTask.is_done(st_now))
        done = success or ended
        failure = bool(ended and not success and (success_known or st_now.get("heli") is None))
        reason = "success" if success else "failure" if failure else "mission_ended_unknown" if ended else None
        mission = self._mission(st_now, required=not done)
        if mission is not None and mission["epoch"] != self._mission_epoch:
            raise RuntimeError("native mission epoch changed; discard transition")
        if mission is not None and self._mission_prev["completed_mask"] & ~mission["completed_mask"]:
            if not done:
                raise RuntimeError("native completion mask regressed; discard transition")
            mission = None  # terminal cleanup is not a task event
        objectives = self._objectives(st_now, required=not done)
        self._validate_heli(st_now, required=not done)
        old_id, old_target = self.target_id, self._tgt
        previous_distance = self._dist_prev
        target = None if done else self._pick_target(st_now, mission, objectives, old_target)
        current_key = self._tgt_key(target)
        current_id = None if current_key is None else (self.episode_id,) + current_key
        target_changed = old_id != current_id
        distance = (float(np.linalg.norm(np.asarray(target["pos"]) - np.asarray(st_now["heli"]["pos"])))
                    if target is not None else None)
        # Validate the prospective observation before committing event history.
        # It is rebuilt below after the reward-history bits have been updated.
        self._features(st_now, target, mission, self._stage(mission, success))
        terms = dict(time_cost=-.15, progress=0., pickup=0., drop=0., collision=0., success=0., failure=0.)
        events = []
        # Straight-line distance to an indoor destination is not traversable
        # path length. Until route-conditioned observations/rewards are verified,
        # carrying/falling can earn only native events, not wallward progress.
        if (not done and self._stage(self._mission_prev) == "pickup"
                and self._stage(mission) == "pickup" and current_id is not None
                and not target_changed and previous_distance is not None):
            weight = 2.5 if target["pos"][2] > 15. else .5
            if min(distance, previous_distance) < 8.:
                weight *= 3.
            terms["progress"] = (previous_distance - distance) * weight
        # Pickup is a native available -> carried edge, once per bomb identity.
        # An invalid/terminal cleanup snapshot can never manufacture pickup.
        if mission is not None:
            if not done:
                for index, (before, after) in enumerate(zip(self._mission_prev["bomb_states"], mission["bomb_states"])):
                    if before == 0 and after == 1 and index not in self._pickup_rewarded:
                        self._pickup_rewarded.add(index)
                        terms["pickup"] += 5.
                        self.pickup_events += 1
                        events.append(dict(type="pickup", bomb_index=index, reward=5.))
            # A valid same-epoch final native snapshot may contain the fourth
            # completed destination as well as the global success increment.
            new_mask = mission["completed_mask"] & ~self._mission_prev["completed_mask"]
            for index in range(4):
                if new_mask & (1 << index) and index not in self._placement_rewarded:
                    self._placement_rewarded.add(index)
                    terms["drop"] += 25.
                    self.drop_events += 1
                    events.append(dict(type="drop", objective_index=index, reward=25.))
        h = st_now.get("heli")
        health = h.get("health") if h else None
        # Preserve the raw nonnegative observed loss for diagnostics. The
        # epsilon affects only the penalty, not this recorded measurement.
        health_loss = 0.
        if health is not None and np.isfinite(health) and self._health_prev is not None:
            health_loss = max(0., self._health_prev - health)
            if health_loss > DAMAGE_EPSILON:
                terms["collision"] = -DAMAGE_PENALTY_SCALE * health_loss
        if success and not self._success_awarded:
            terms["success"] = 100.
            self._success_awarded = True
            events.append(dict(type="success", reward=100.))
        elif ended:
            terms["failure"] = -10.
        self.episode_events += len(events)
        self.total_events += len(events)
        self.target_changes += int(target_changed)
        self._health_prev = health if health is not None and np.isfinite(health) else None
        self._tgt = target
        self._dist_prev = distance
        if mission is not None:
            self._mission_prev = mission
        self._last_state = st_now
        self._finished = done
        stage = self._stage(mission, success)
        obs = self._features(st_now, target, mission, stage)
        native_ids = [b.get("blipId") for b in st_now.get("blips", [])
                      if current_key is not None and (b.get("objectiveKind"), b.get("objectiveIndex")) == current_key]
        info = dict(dist=distance, previous_target_dist=previous_distance,
                    target_id=current_id, previous_target_id=old_id, target_changed=target_changed,
                    target_pos=None if target is None else target["pos"],
                    previous_target_pos=None if old_target is None else old_target["pos"],
                    target_visible=target is not None, target_native_blip_ids=native_ids,
                    target_tiers=[], awarded_tiers=[], reward_terms=terms, task_events=events,
                    health_loss=health_loss, progress_scope=PROGRESS_SCOPE,
                    action=action_name, done=done, terminated=done, truncated=False, reason=reason,
                    success=success, failure=failure, success_known=success_known, stage=stage,
                    mission=st_now.get("mission"), mission_valid=mission is not None,
                    mission_epoch=self._mission_epoch, objectives=st_now.get("objectives", []),
                    episode_id=self.episode_id, episode_events=self.episode_events, events=self.episode_events,
                    pickup_events=self.pickup_events, drop_events=self.drop_events, proximity_events=0,
                    pickup_rewarded=sorted(self._pickup_rewarded), placement_rewarded=sorted(self._placement_rewarded),
                    target_changes=self.target_changes, heli=h, blips=st_now.get("blips", []), frame=st_now["frame"])
        def plain(value):
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, dict):
                return {key: plain(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return type(value)(plain(item) for item in value)
            return value
        info = plain(info)
        return obs, sum(info["reward_terms"].values()), done, info

    def begin_episode(self):
        self._launch_k = 0

    def launch_cont(self):
        self._launch_k += 1
        return ACTION_VECTORS[1].copy()

    def launch_action(self):
        self._launch_k += 1
        return 1

    def demo_action(self):
        return 1


# ---------------- numpy DQN ----------------

class DQN:
    def __init__(self, n_obs, n_act, hidden=64, lr=1e-3, seed=0):
        rng = np.random.default_rng(seed)
        self.n_act = n_act
        def init(inp, out):
            return (rng.normal(0, (2.0 / inp) ** 0.5, (inp, out)).astype(np.float32),
                    np.zeros(out, dtype=np.float32))
        self.w1, self.b1 = init(n_obs, hidden)
        self.w2, self.b2 = init(hidden, hidden)
        self.w3, self.b3 = init(hidden, n_act)
        self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
        self.tparams = [p.copy() for p in self.params]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0
        self.lr, self.beta1, self.beta2, self.eps = lr, 0.9, 0.999, 1e-8

    def forward(self, x, params=None, cache=False):
        p = params if params is not None else self.params
        w1, b1, w2, b2, w3, b3 = p
        h1 = np.tanh(x @ w1 + b1)
        h2 = np.tanh(h1 @ w2 + b2)
        q = h2 @ w3 + b3
        if cache:
            return q, (x, h1, h2)
        return q

    def act(self, obs, eps):
        if np.random.rand() < eps:
            return np.random.randint(self.n_act)
        q = self.forward(obs[None, :])[0]
        return int(np.argmax(q))

    def train_step(self, batch, gamma=0.95):
        obs, act, rew, nxt, don = batch
        p = self.params
        q_next = self.forward(nxt, self.tparams).max(axis=1)
        target = rew + gamma * q_next * (1.0 - don)
        # forward with cache on full batch
        q_all, (x, h1, h2) = self.forward(obs, cache=True)
        idx = np.arange(len(obs))
        q_sa = q_all[idx, act]
        err = q_sa - target
        # huber grad dL/dq
        dq = np.where(np.abs(err) <= 1.0, err, np.sign(err))
        grads = [np.zeros_like(pp) for pp in p]
        dz3 = np.zeros_like(q_all)
        dz3[idx, act] = dq / len(obs)
        grads[4] = h2.T @ dz3          # w3
        grads[5] = dz3.sum(axis=0)     # b3
        dh2 = dz3 @ p[4].T * (1.0 - h2 * h2)
        grads[2] = h1.T @ dh2
        grads[3] = dh2.sum(axis=0)
        dh1 = dh2 @ p[2].T * (1.0 - h1 * h1)
        grads[0] = x.T @ dh1
        grads[1] = dh1.sum(axis=0)
        # adam
        self.t += 1
        for i, g in enumerate(grads):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g * g
            mh = self.m[i] / (1 - self.beta1 ** self.t)
            vh = self.v[i] / (1 - self.beta2 ** self.t)
            p[i] -= self.lr * mh / (np.sqrt(vh) + self.eps)
        return float(np.mean(np.abs(err)))

    def sync_target(self):
        self.tparams = [p.copy() for p in self.params]

    def save(self, path):
        """Versioned policy artifact. DQN exposes warm starts, not resume."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as f:
            np.savez(f, *self.params, env_version=np.asarray(ENV_VERSION),
                     artifact_kind=np.asarray("dqn-warm-start"),
                     action_schema=np.asarray(ACTION_SCHEMA), obs_schema=np.asarray(OBS_SCHEMA),
                     release_threshold=np.asarray(RELEASE_THRESHOLD),
                     phase_reward_version=np.asarray(PHASE_REWARD_VERSION),
                     obstacle_schema_version=np.asarray(OBSTACLE_SCHEMA_VERSION),
                     progress_scope=np.asarray(PROGRESS_SCOPE),
                     damage_penalty_scale=np.asarray(DAMAGE_PENALTY_SCALE),
                     damage_epsilon=np.asarray(DAMAGE_EPSILON))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def load(self, path, *, warm_start=False):
        if not warm_start:
            raise ValueError("DQN supports explicit warm_start=True only; replay/optimizer resume is unavailable")
        with np.load(path, allow_pickle=False) as z:
            if "env_version" not in z or str(z["env_version"]) != ENV_VERSION:
                raise ValueError("incompatible environment checkpoint; retrain for " + ENV_VERSION)
            expected = dict(artifact_kind=np.asarray("dqn-warm-start"),
                            action_schema=np.asarray(ACTION_SCHEMA), obs_schema=np.asarray(OBS_SCHEMA),
                            release_threshold=np.asarray(RELEASE_THRESHOLD),
                            phase_reward_version=np.asarray(PHASE_REWARD_VERSION),
                            obstacle_schema_version=np.asarray(OBSTACLE_SCHEMA_VERSION),
                            progress_scope=np.asarray(PROGRESS_SCOPE),
                            damage_penalty_scale=np.asarray(DAMAGE_PENALTY_SCALE),
                            damage_epsilon=np.asarray(DAMAGE_EPSILON))
            if any(key not in z or not np.array_equal(z[key], value) for key, value in expected.items()):
                raise ValueError("incompatible DQN action/observation/reward contract")
            ps = [z["arr_%d" % i].copy() for i in range(len(self.params))]
        if any(p.shape != q.shape or not np.isfinite(q).all()
               for p, q in zip(self.params, ps)):
            raise ValueError("checkpoint network shape or values are invalid")
        for p, q in zip(self.params, ps):
            p[...] = q
        self.t = 0
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.sync_target()


class Replay:
    def __init__(self, cap=20000):
        self.cap = cap
        self.buf = []

    def add(self, *transition):
        if len(self.buf) >= self.cap:
            self.buf.pop(0)
        self.buf.append(transition)

    def sample(self, n):
        idx = np.random.randint(0, len(self.buf), size=n)
        rows = [self.buf[i] for i in idx]
        obs = np.stack([r[0] for r in rows])
        act = np.array([r[1] for r in rows], dtype=np.int64)
        rew = np.array([r[2] for r in rows], dtype=np.float32)
        nxt = np.stack([r[3] for r in rows])
        don = np.array([r[4] for r in rows], dtype=np.float32)
        return obs, act, rew, nxt, don


# ---------------- training loop ----------------

def train(minutes=None, episodes=None, resume=False, budget_fallback=True,
          warm_start=None, run_dir=None, seed=0):
    if minutes is not None and (not np.isfinite(minutes) or minutes <= 0):
        raise ValueError("minutes must be finite and positive")
    if episodes is not None and (not isinstance(episodes, int) or episodes <= 0):
        raise ValueError("episodes must be a positive integer")
    if minutes is None and episodes is None:
        minutes = 5.0
    if resume:
        raise ValueError("DQN --resume is unsupported; use --warm-start PATH for a new run with fresh optimizer/replay")
    np.random.seed(seed)
    agent = DQN(RcHeliEnv.N_OBS, N_ACT, seed=seed)
    if warm_start:
        agent.load(warm_start, warm_start=True)
    out = Path(run_dir) if run_dir else Path(__file__).parent / "runs" / (
        "dqn-%s-%s" % (time.strftime("%Y%m%d-%H%M%S"), time.time_ns() % 1000000))
    out.mkdir(parents=True, exist_ok=False)
    import json
    (out / "config.json").write_text(json.dumps(dict(
        env_version=ENV_VERSION, seed=seed, game_seed_controlled=False,
        action_schema=ACTION_SCHEMA, obs_schema=OBS_SCHEMA, release_threshold=RELEASE_THRESHOLD,
        phase_reward_version=PHASE_REWARD_VERSION, obstacle_schema_version=OBSTACLE_SCHEMA_VERSION,
        progress_scope=PROGRESS_SCOPE,
        damage_penalty_scale=DAMAGE_PENALTY_SCALE, damage_epsilon=DAMAGE_EPSILON,
        step_frames=STEP_FRAMES, warm_start=str(warm_start) if warm_start else None,
        algorithm="DQN", artifact_kind="warm-start-only"), indent=2), encoding="utf-8")
    buf = Replay()
    env = None
    logf = None
    step_count = ep_i = 0
    t_end = time.perf_counter() + minutes * 60 if minutes else None
    try:
        env = RcHeliEnv(verbose=True)
        logf = (out / "episodes.csv").open("w", newline="", encoding="utf-8")
        logger = csv.writer(logf)
        logger.writerow(["episode", "env_steps", "steps", "return", "min_dist",
                         "events", "proximity_events", "pickup_events", "drop_events", "stage",
                         "success", "success_known", "failure",
                         "terminated", "truncated", "reason", "epsilon", "td_err"])
        while not (t_end and time.perf_counter() >= t_end):
            if episodes is not None and ep_i >= episodes:
                break
            obs = env.reset()
            env.begin_episode()
            ep_i += 1
            tracker = EpisodeProgress()
            tracker.reset(env.target_id)
            ret, minimum, steps, td = 0.0, None, 0, 0.0
            info = dict(success=False, failure=False, terminated=False,
                        truncated=True, reason="max_steps")
            eps = max(EPS_FLOOR, 1.0 - step_count / 1500.0)
            for t in range(MAX_STEPS):
                if t_end and time.perf_counter() >= t_end:
                    info.update(truncated=True, reason="budget")
                    break
                action = env.launch_action() if t < LAUNCH_STEPS else agent.act(obs, eps)
                try:
                    nxt, reward, done, transition = env.step(action)
                except (GameDeadError, TimeoutError, RuntimeError) as exc:
                    # There is no valid successor for the attempted action.
                    # Never pair it with a recovered/reset initial observation.
                    info.update(terminated=False, truncated=True, reason="bridge_error")
                    print("step discarded: %s" % exc)
                    env.task.hard_reset()
                    break
                info = dict(transition)
                # External stall/budget/max-step truncations still bootstrap.
                buf.add(obs.copy(), action, reward, nxt.copy(),
                        float(info["terminated"]))
                ret += reward
                steps += 1
                step_count += 1
                distance = info["dist"]
                if distance is not None:
                    minimum = distance if minimum is None else min(minimum, distance)
                stalled = tracker.update(info)
                obs = nxt
                eps = max(EPS_FLOOR, 1.0 - step_count / 1500.0)
                if len(buf.buf) >= 500:
                    td = agent.train_step(buf.sample(64))
                if step_count % 200 == 0:
                    agent.sync_target()
                if done:
                    break
                if stalled:
                    info.update(truncated=True, reason="stall")
                    break
            else:
                info.update(truncated=True, reason="max_steps")
            logger.writerow([ep_i, step_count, steps, ret,
                             -1 if minimum is None else minimum,
                             env.episode_events, env.proximity_events,
                             env.pickup_events, env.drop_events, info.get("stage", "unknown"),
                             info["success"], info.get("success_known", False), info["failure"], info["terminated"],
                             info["truncated"], info["reason"], eps, td])
            logf.flush()
            print("episode=%d steps=%d return=%.2f reason=%s" %
                  (ep_i, steps, ret, info["reason"]))
            if ep_i % 5 == 0:
                agent.save(out / "policy.npz")
    except KeyboardInterrupt:
        print("interrupted; saving warm-start policy")
    finally:
        try:
            if env is not None:
                env.task.release()
        finally:
            if logf is not None:
                logf.close()
            agent.save(out / "policy.npz")
    print("saved %s; episodes=%d environment_steps=%d" % (out, ep_i, step_count))
    return out


def evaluate(steps=120, checkpoint=None):
    if checkpoint is None:
        raise ValueError("evaluation requires an explicit --checkpoint PATH")
    agent = DQN(RcHeliEnv.N_OBS, N_ACT)
    agent.load(checkpoint, warm_start=True)  # validate before launching a game
    env = RcHeliEnv(verbose=True)
    try:
        obs = env.reset()
        env.begin_episode()
        for t in range(steps):
            action = env.launch_action() if t < LAUNCH_STEPS else agent.act(obs, 0.0)
            obs, reward, done, info = env.step(action)
            print("t=%d action=%s dist=%s reward=%+.2f success=%s reason=%s" %
                  (t, info["action"], info["dist"], reward, info["success"], info["reason"]))
            if done:
                break
    finally:
        env.task.release()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float)
    ap.add_argument("--episodes", type=int)
    ap.add_argument("--resume", action="store_true",
                    help="unsupported; use an explicit --warm-start instead")
    ap.add_argument("--warm-start", metavar="PATH")
    ap.add_argument("--run-dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--checkpoint")
    ap.add_argument("--eval-steps", type=int, default=120)
    args = ap.parse_args()
    if args.eval:
        evaluate(args.eval_steps, checkpoint=args.checkpoint)
    else:
        train(minutes=args.minutes if args.minutes is not None else
              (None if args.episodes is not None else 5.0), episodes=args.episodes,
              resume=args.resume, warm_start=args.warm_start,
              run_dir=args.run_dir, seed=args.seed)
