"""Offline summaries for interleaved policy evaluation blocks.

Usage: python -B ml/summarize_policy_eval.py ml/runs/v3-eval-20261106

Only status.json, block config.json, episodes.csv and trace.jsonl are read.
Reports go to ROOT/analysis/summary.json and summary.md by default. No game,
learner or checkpoint is imported or executed. Running evaluations are supported;
an incomplete trailing record is ignored and explicitly reported.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import statistics


THRESHOLDS = (2.5, 1.5)
CANDIDATE_THRESHOLDS = (2.5, 1.5, 1.0)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def truth(value):
    return value is True or (isinstance(value, str) and value.lower() == "true")


def target_key(value):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        if all(isinstance(x, int) and not isinstance(x, bool) for x in value):
            return tuple(value)
    return None


def position(heli):
    value = heli.get("pos") if isinstance(heli, dict) else None
    if isinstance(value, (list, tuple)) and len(value) == 3 and all(finite(x) for x in value):
        return tuple(value)
    return None


def distribution(values):
    values = [float(x) for x in values if finite(x)]
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {"n": len(values), "mean": statistics.fmean(values),
            "median": statistics.median(values), "min": min(values), "max": max(values)}


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def read_json(path, warnings, required=False):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        if required:
            raise
        warnings.append("Missing file: " + str(path))
    except json.JSONDecodeError as exc:
        if required:
            raise ValueError("Cannot read a complete status snapshot: " + str(path)) from exc
        warnings.append("Incomplete or invalid JSON: " + str(path))
    return {}


def read_csv(path, warnings):
    if not path.is_file():
        warnings.append("Missing episode CSV: " + str(path))
        return []
    # One read gives a bounded snapshot while another process appends rows.
    text = path.read_text(encoding="utf-8-sig")
    if text and not text.endswith("\n"):
        text = text.rsplit("\n", 1)[0] + "\n" if "\n" in text else ""
        warnings.append("Ignored incomplete trailing CSV line: " + str(path))
    rows = []
    for number, row in enumerate(csv.DictReader(io.StringIO(text)), 2):
        if None in row or any(value is None for value in row.values()):
            warnings.append("Invalid CSV row %d: %s" % (number, path))
            continue
        try:
            int(row["episode"])
        except (KeyError, TypeError, ValueError):
            warnings.append("Missing episode number on CSV row %d: %s" % (number, path))
            continue
        rows.append(row)
    return rows


class Episode:
    def __init__(self, number):
        self.number = number
        self.reset_seen = False
        self.end_seen = False
        self.row = {}
        self.end_row = {}
        self.steps = 0
        self.reward = 0.0
        self.terms = Counter()
        self.issues = Counter()
        self.issue_examples = []
        self.max_reward_error = 0.0
        self.frame_deltas = Counter()
        self.time_deltas = Counter()
        self.tier_counts = Counter()
        self.tier_history = Counter()
        self.stages = Counter()
        self.phase_counts = Counter()
        self.initial_target = None
        self.initial_pid = None
        self.initial_episode_id = None
        self.initial_distance = None
        self.minimum_distance = None
        self.final_distance = None
        self.last_observed_distance = None
        self.last_observed_step = None
        self.first_hits = {str(t): None for t in THRESHOLDS}
        self.candidate_ids = []
        self.candidate_initial_distance = None
        self.candidate_min_distance = None
        self.candidate_final_distance = None
        self.candidate_last_distance = None
        self.candidate_last_step = None
        self.candidate_first_hits = {str(t): None for t in CANDIDATE_THRESHOLDS}
        self.first_position = None
        self.last_position = None
        self.previous_position = None
        self.horizontal_path = 0.0
        self.world_z_min = None
        self.world_z_max = None
        self.world_z_values = []
        self.initial_health = None
        self.last_health = None
        self.min_health = None
        self.cumulative_health_drop = 0.0
        self.collision_steps = 0
        self.native_switches = []
        self.last_selected = None
        self.previous_state = None
        self.previous_obs = None
        self.last_info = {}
        self.last_record = {}
        self.trace_errors = []

    def issue(self, name, step=None):
        self.issues[name] += 1
        if len(self.issue_examples) < 8:
            self.issue_examples.append({"issue": name, "step": step})

    def observe_state(self, state, step):
        if not isinstance(state, dict):
            self.issue("missing_raw_state", step)
            self.final_distance = None
            self.candidate_final_distance = None
            return
        heli = state.get("heli")
        pos = position(heli)
        if pos is not None:
            if self.first_position is None:
                self.first_position = pos
            if self.previous_position is not None:
                self.horizontal_path += math.hypot(pos[0] - self.previous_position[0],
                                                   pos[1] - self.previous_position[1])
            self.previous_position = self.last_position = pos
            self.world_z_min = pos[2] if self.world_z_min is None else min(self.world_z_min, pos[2])
            self.world_z_max = pos[2] if self.world_z_max is None else max(self.world_z_max, pos[2])
            self.world_z_values.append(pos[2])
        else:
            self.previous_position = None
        health = heli.get("health") if isinstance(heli, dict) else None
        if finite(health):
            if self.initial_health is None:
                self.initial_health = health
            if self.last_health is not None:
                self.cumulative_health_drop += max(self.last_health - health, 0.)
            self.last_health = health
            self.min_health = health if self.min_health is None else min(self.min_health, health)
        self.final_distance = None
        self.candidate_final_distance = None
        if pos is None:
            return
        if state.get("pid") != self.initial_pid or state.get("episodeId") != self.initial_episode_id:
            self.issue("initial_target_epoch_changed", step)
            return
        # This set is frozen at reset. Only positions of its still-visible IDs
        # are refreshed; later floor markers cannot enter the candidate set.
        candidate_distances = [math.dist(pos, position(b)) for b in state.get("blips", [])
                               if b.get("blipId") in self.candidate_ids and position(b) is not None]
        if candidate_distances:
            distance = min(candidate_distances)
            if step == 0:
                self.candidate_initial_distance = distance
            self.candidate_final_distance = self.candidate_last_distance = distance
            self.candidate_last_step = step
            self.candidate_min_distance = distance if self.candidate_min_distance is None else min(self.candidate_min_distance, distance)
            for threshold in CANDIDATE_THRESHOLDS:
                key = str(threshold)
                if self.candidate_first_hits[key] is None and distance <= threshold:
                    self.candidate_first_hits[key] = step
        if self.initial_target is None:
            return
        matches = [b for b in state.get("blips", []) if b.get("blipId") == self.initial_target[1]]
        if len(matches) != 1:
            return  # An absent original target is not the newly selected target.
        goal = position(matches[0])
        if goal is None:
            return
        distance = math.dist(pos, goal)
        if step == 0:
            self.initial_distance = distance
        self.final_distance = self.last_observed_distance = distance
        self.last_observed_step = step
        self.minimum_distance = distance if self.minimum_distance is None else min(self.minimum_distance, distance)
        for threshold in THRESHOLDS:
            key = str(threshold)
            if self.first_hits[key] is None and distance <= threshold:
                self.first_hits[key] = step

    def reset(self, record):
        if self.reset_seen:
            self.issue("duplicate_reset_record", 0)
        self.reset_seen = True
        self.initial_target = target_key(record.get("target_id"))
        state = record.get("state", {})
        self.initial_pid = state.get("pid")
        self.initial_episode_id = state.get("episodeId")
        pos = position(state.get("heli"))
        if pos is not None:
            self.candidate_ids = sorted(b["blipId"] for b in state.get("blips", [])
                                        if b.get("blipType") == 3 and b.get("color") == 4
                                        and b.get("sprite") == 0 and position(b) is not None
                                        and math.dist(pos, position(b)) <= 20
                                        and target_key([self.initial_episode_id, b.get("blipId")]) is not None)
        self.previous_state = state
        self.previous_obs = record.get("obs")
        self.last_selected = self.initial_target
        self.observe_state(state, 0)

    def step(self, record):
        self.steps += 1
        number = record.get("step")
        if number != self.steps:
            self.issue("nonconsecutive_step_index", number)
        info = record.get("info", {})
        before, after = record.get("before_state"), record.get("after_state")
        self.phase_counts[str(record.get("phase", "unknown"))] += 1
        self.stages[str(info.get("stage", "unknown"))] += 1
        if self.previous_state is not None and before != self.previous_state:
            self.issue("raw_snapshot_discontinuity", number)
        if self.previous_obs is not None and record.get("obs") != self.previous_obs:
            self.issue("observation_discontinuity", number)
        for name in ("obs", "next_obs", "action"):
            values = record.get(name)
            if not isinstance(values, list) or not values or not all(finite(x) for x in values):
                self.issue("invalid_" + name, number)
        if not self.reset_seen:
            self.issue("missing_reset_record", number)
        terms = record.get("reward_terms", info.get("reward_terms"))
        reward = record.get("reward")
        if finite(reward):
            self.reward += reward
        else:
            self.issue("invalid_reward", number)
        if not isinstance(terms, dict) or not all(finite(v) for v in terms.values()):
            self.issue("invalid_reward_terms", number)
            terms = {}
        elif finite(reward):
            difference = abs(reward - sum(terms.values()))
            self.max_reward_error = max(self.max_reward_error, difference)
            if difference > 1e-9:
                self.issue("reward_sum_mismatch", number)
        if "reward_terms" in info and info["reward_terms"] != terms:
            self.issue("info_reward_terms_mismatch", number)
        self.terms.update(terms)
        if terms.get("collision", 0) < 0:
            self.collision_steps += 1
        current = target_key(info.get("target_id"))
        previous = target_key(info.get("previous_target_id"))
        awarded = []
        for tier in ("reach", "near"):
            if terms.get(tier, 0) > 0:
                awarded.append(tier)
                self.tier_counts[tier] += 1
                key = (current, tier)
                self.tier_history[key] += 1
                if current is None:
                    self.issue("tier_without_target", number)
                if self.tier_history[key] > 1:
                    self.issue("duplicate_target_tier", number)
        if "awarded_tiers" in info and sorted(info["awarded_tiers"]) != sorted(awarded):
            self.issue("awarded_tier_list_mismatch", number)
        if info.get("target_changed", False) != (current != previous):
            self.issue("target_changed_flag_mismatch", number)
        if info.get("target_changed") and terms.get("progress", 0) != 0:
            self.issue("progress_reward_across_target_change", number)
        if isinstance(before, dict) and isinstance(after, dict):
            self.check_transition(before, after, info, current, number)
        else:
            self.issue("missing_transition_snapshot", number)
        self.observe_state(after, number)
        self.previous_state = after
        self.previous_obs = record.get("next_obs")
        self.last_info, self.last_record = info, record

    def check_transition(self, before, after, info, current, number):
        for key in ("pid", "episodeId"):
            if before.get(key) != after.get(key):
                self.issue("cross_" + key, number)
        frames = after.get("stepFrames")
        sim_ms = after.get("simStepMs")
        if not finite(frames) or frames <= 0 or not finite(sim_ms) or sim_ms <= 0:
            self.issue("invalid_step_size", number)
        for key, start_key, counter, expected in (
                ("frame", "stepStartFrame", self.frame_deltas, frames),
                ("gameTimerMs", "stepStartGameMs", self.time_deltas,
                 frames * sim_ms if finite(frames) and finite(sim_ms) else None)):
            if not finite(before.get(key)) or not finite(after.get(key)):
                self.issue("missing_" + key, number)
                continue
            delta = (after[key] - before[key]) % (2 ** 32)
            counter[str(delta)] += 1
            if delta != expected:
                self.issue(key + "_delta_mismatch", number)
            if after.get(start_key) != before[key]:
                self.issue(start_key + "_mismatch", number)
        if (not after.get("lockstep") or after.get("stepActive")
                or not after.get("injectActive") or not after.get("controllerPid")):
            self.issue("unfrozen_step_result", number)
        if finite(before.get("lastStepSeq")) and finite(after.get("lastStepSeq")):
            if (after["lastStepSeq"] - before["lastStepSeq"]) % (2 ** 32) != 1:
                self.issue("step_ack_sequence_mismatch", number)
        raw = after.get("blips", [])
        ids = [b.get("blipId") for b in raw]
        if any(not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < 2 ** 32 for i in ids):
            self.issue("invalid_native_blip_id", number)
        if len(set(ids)) != len(ids):
            self.issue("duplicate_native_blip_id", number)
        matches = [b for b in raw if current is not None and b.get("blipId") == current[1]]
        valid = current is not None and current[0] == after.get("episodeId") and len(matches) == 1
        if current is not None and not valid:
            self.issue("selected_target_not_in_raw_blips", number)
        if valid:
            if info.get("target_pos") != matches[0].get("pos"):
                self.issue("stale_target_position", number)
            if self.last_selected is not None and current != self.last_selected:
                before_by_id = {b.get("blipId"): b for b in before.get("blips", [])}
                after_by_id = {b.get("blipId"): b for b in raw}
                self.native_switches.append({"step": number, "previous": self.last_selected,
                                             "current": current, "target_pos": matches[0].get("pos"),
                                             "removed_native_blips": [b for key, b in before_by_id.items() if key not in after_by_id],
                                             "added_native_blips": [b for key, b in after_by_id.items() if key not in before_by_id]})
            self.last_selected = current

    def finish(self):
        row = self.row or self.end_row
        complete = bool(row) or self.end_seen
        if row:
            try:
                if int(row["steps"]) != self.steps:
                    self.issue("csv_trace_step_count_mismatch")
                if abs(float(row["return"]) - self.reward) > 1e-8:
                    self.issue("csv_trace_return_mismatch")
            except (KeyError, ValueError):
                self.issue("invalid_csv_metrics")
        if complete and not self.end_seen:
            self.issue("missing_episode_end_record")
        info, record = self.last_info, self.last_record
        error = row.get("error", "") or "; ".join(self.trace_errors)
        success = truth(row.get("success", info.get("success", False)))
        known = truth(row.get("success_known", info.get("success_known", False)))
        net = None
        if self.first_position is not None and self.last_position is not None:
            net = math.hypot(self.last_position[0] - self.first_position[0],
                             self.last_position[1] - self.first_position[1])
        health_drop = (max(self.initial_health - self.min_health, 0.)
                       if self.initial_health is not None and self.min_health is not None else None)
        return {"episode": self.number, "complete": complete, "steps": self.steps,
                "error": error, "success": success, "success_known": known,
                "terminated": truth(row.get("terminated", info.get("terminated", False))),
                "truncated": truth(row.get("truncated", record.get("truncated", False))),
                "reason": row.get("reason") or record.get("reason"),
                "initial_target_id": self.initial_target,
                "initial_target_initial_distance": self.initial_distance,
                "initial_target_min_distance": self.minimum_distance,
                "initial_target_final_distance": self.final_distance,
                "initial_target_last_observed_distance": self.last_observed_distance,
                "initial_target_last_observed_step": self.last_observed_step,
                "first_reach_steps": self.first_hits,
                "initial_nearby_object_candidates": {
                    "native_ids": self.candidate_ids, "episode_id": self.initial_episode_id,
                    "initial_distance": self.candidate_initial_distance,
                    "min_distance": self.candidate_min_distance,
                    "final_distance": self.candidate_final_distance,
                    "last_observed_distance": self.candidate_last_distance,
                    "last_observed_step": self.candidate_last_step,
                    "first_reach_steps": self.candidate_first_hits},
                "native_target_switch_count": len(self.native_switches),
                "native_target_switches": self.native_switches,
                "health": {"initial": self.initial_health, "last_observed": self.last_health,
                           "min": self.min_health, "drop_from_initial": health_drop,
                           "cumulative_decreases": self.cumulative_health_drop},
                "horizontal_sampled_path": self.horizontal_path,
                "horizontal_net_displacement": net,
                "world_z_min": self.world_z_min, "world_z_max": self.world_z_max,
                "world_z_initial": self.first_position[2] if self.first_position is not None else None,
                "world_z_last_observed": self.last_position[2] if self.last_position is not None else None,
                "world_z_mean": statistics.fmean(self.world_z_values) if self.world_z_values else None,
                "reward": self.reward, "reward_per_step": ratio(self.reward, self.steps),
                "reward_terms": dict(self.terms), "collision_penalty_steps": self.collision_steps,
                "phase_counts": dict(self.phase_counts), "stage_counts": dict(self.stages),
                "tier_counts": dict(self.tier_counts),
                "integrity": {"issues": dict(self.issues), "examples": self.issue_examples,
                              "max_reward_sum_error": self.max_reward_error,
                              "frame_delta_counts": dict(self.frame_deltas),
                              "game_ms_delta_counts": dict(self.time_deltas)}}


def analyze_block(block, root, warnings):
    path = Path(block["path"])
    if not path.is_absolute():
        path = root / path
    config = read_json(path / "config.json", warnings) if path.is_dir() else {}
    episodes = {}
    event_counts = Counter()
    trace = path / "trace.jsonl"
    if trace.is_file():
        # Stop at the initial byte length so a live append cannot make this
        # read unbounded or mix several snapshots of the same running block.
        limit = trace.stat().st_size
        with trace.open("rb") as f:
            line_number = 0
            while f.tell() < limit:
                raw = f.readline(limit - f.tell())
                line_number += 1
                if not raw.endswith(b"\n"):
                    warnings.append("Ignored incomplete trailing trace record: " + str(trace))
                    break
                try:
                    record = json.loads(raw)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    warnings.append("Invalid trace JSON line %d: %s" % (line_number, trace))
                    continue
                event = record.get("event", "unknown")
                event_counts[event] += 1
                number = record.get("episode")
                if not isinstance(number, int):
                    warnings.append("Trace record without integer episode: %s:%d" % (trace, line_number))
                    continue
                ep = episodes.setdefault(number, Episode(number))
                if event == "reset":
                    ep.reset(record)
                elif event == "step":
                    ep.step(record)
                elif event == "episode_end":
                    if ep.end_seen:
                        ep.issue("duplicate_episode_end_record")
                    ep.end_seen = True
                    ep.end_row = record
                elif "error" in event:
                    ep.trace_errors.append(str(record.get("error", event)))
    elif block.get("status") != "pending":
        warnings.append("Missing trace: " + str(trace))
    if (path / "episodes.csv").is_file() or block.get("status") != "pending":
        for row in read_csv(path / "episodes.csv", warnings):
            number = int(row["episode"])
            ep = episodes.setdefault(number, Episode(number))
            # CSV may advance after the bounded trace snapshot was read. Such
            # a row must not turn a partial trajectory into a completed one.
            if block.get("status") == "running" and int(row.get("steps", 0)) > ep.steps:
                warnings.append("CSV ahead of trace snapshot in block %s episode %s" % (block["index"], number))
                continue
            if ep.row:
                ep.issue("duplicate_csv_episode")
            ep.row = row
    records = [ep.finish() for _, ep in sorted(episodes.items())]
    for record in records:
        record.update(block_index=block["index"], source=block["source"],
                      run_id=config.get("run_id"), block_path=str(path))
    return {"index": block["index"], "source": block["source"], "path": str(path),
            "status": block.get("status", "unknown"), "event_counts": dict(event_counts),
            "planned_episodes": block.get("episodes"), "episodes": records}


def summarize_model(records):
    completed = [r for r in records if r["complete"]]
    valid = [r for r in completed if not r["error"]]
    known = [r for r in valid if r["success_known"]]
    distance_known = [r for r in valid if r["initial_target_min_distance"] is not None]
    terms, issues, frames, milliseconds, tiers, stages, phases = (Counter() for _ in range(7))
    for r in records:
        issues.update(r["integrity"]["issues"])
        frames.update(r["integrity"]["frame_delta_counts"])
        milliseconds.update(r["integrity"]["game_ms_delta_counts"])
        tiers.update(r["tier_counts"])
        stages.update(r["stage_counts"])
        phases.update(r["phase_counts"])
    for r in valid:
        terms.update(r["reward_terms"])
    steps = sum(r["steps"] for r in valid)
    total_reward = sum(r["reward"] for r in valid)
    thresholds = {}
    for threshold in THRESHOLDS:
        key = str(threshold)
        hits = [r["first_reach_steps"][key] for r in distance_known
                if r["first_reach_steps"][key] is not None]
        thresholds[key] = {"episodes_reached": len(hits), "denominator": len(distance_known),
                           "rate": ratio(len(hits), len(distance_known)),
                           "at_reset": sum(step == 0 for step in hits),
                           "first_reach_step": distribution(hits)}
    switching = sum(r["native_target_switch_count"] > 0 for r in valid)
    candidates = [r["initial_nearby_object_candidates"] for r in valid
                  if r["initial_nearby_object_candidates"]["min_distance"] is not None]
    candidate_thresholds = {}
    for threshold in CANDIDATE_THRESHOLDS:
        key = str(threshold)
        hits = [c["first_reach_steps"][key] for c in candidates if c["first_reach_steps"][key] is not None]
        candidate_thresholds[key] = {"episodes_reached": len(hits), "denominator": len(candidates),
                                     "rate": ratio(len(hits), len(candidates)),
                                     "at_reset": sum(step == 0 for step in hits),
                                     "first_reach_step": distribution(hits)}
    return {"observed_episodes": len(records), "completed_episodes": len(completed),
            "in_progress_episodes": len(records) - len(completed),
            "environment_error_episodes": sum(bool(r["error"]) for r in completed),
            "valid_completed_episodes": len(valid),
            "terminated_episodes": sum(r["terminated"] for r in completed),
            "truncated_episodes": sum(r["truncated"] for r in completed),
            "successes": sum(r["success"] for r in known),
            "success_denominator": len(known),
            "success_rate": ratio(sum(r["success"] for r in known), len(known)),
            "success_unknown_episodes": len(valid) - len(known),
            "end_reasons": dict(Counter(str(r["reason"]) for r in completed)),
            "observed_trace_steps": sum(r["steps"] for r in records),
            "valid_completed_steps": steps,
            "initial_target_min_distance": distribution(r["initial_target_min_distance"] for r in distance_known),
            "initial_target_final_distance": distribution(r["initial_target_final_distance"] for r in distance_known),
            "initial_target_unobservable_at_end": sum(r["initial_target_final_distance"] is None for r in distance_known),
            "initial_target_last_observed_distance": distribution(r["initial_target_last_observed_distance"] for r in distance_known),
            "reach_thresholds": thresholds,
            "initial_nearby_object_candidates": {
                "episode_denominator": len(candidates),
                "initial_candidate_count": distribution(len(c["native_ids"]) for c in candidates),
                "min_distance": distribution(c["min_distance"] for c in candidates),
                "final_distance": distribution(c["final_distance"] for c in candidates),
                "last_observed_distance": distribution(c["last_observed_distance"] for c in candidates),
                "unobservable_at_end": sum(c["final_distance"] is None for c in candidates),
                "reach_thresholds": candidate_thresholds},
            "native_target_switches": sum(r["native_target_switch_count"] for r in valid),
            "native_target_switch_episodes": switching,
            "native_target_switch_episode_rate": ratio(switching, len(valid)),
            "health_drop_episodes": sum((r["health"]["drop_from_initial"] or 0) > 1e-6 for r in valid),
            "health_drop_from_initial": distribution(r["health"]["drop_from_initial"] for r in valid),
            "cumulative_health_decreases": sum(r["health"]["cumulative_decreases"] for r in valid),
            "horizontal_sampled_path": distribution(r["horizontal_sampled_path"] for r in valid),
            "horizontal_net_displacement": distribution(r["horizontal_net_displacement"] for r in valid),
            "world_z_range": {"min": min((r["world_z_min"] for r in valid if r["world_z_min"] is not None), default=None),
                              "max": max((r["world_z_max"] for r in valid if r["world_z_max"] is not None), default=None)},
            "world_z_initial": distribution(r["world_z_initial"] for r in valid),
            "world_z_last_observed": distribution(r["world_z_last_observed"] for r in valid),
            "world_z_episode_mean": distribution(r["world_z_mean"] for r in valid),
            "reward_per_step": ratio(total_reward, steps), "reward_total": total_reward,
            "reward_terms_total": dict(terms),
            "reward_terms_per_step": {key: ratio(value, steps) for key, value in terms.items()},
            "collision_penalty_steps": sum(r["collision_penalty_steps"] for r in valid),
            "observed_tier_counts": dict(tiers), "observed_stage_counts": dict(stages),
            "observed_phase_counts": dict(phases),
            "integrity": {"issues": dict(issues),
                          "max_reward_sum_error": max((r["integrity"]["max_reward_sum_error"] for r in records), default=0.),
                          "frame_delta_counts": dict(frames), "game_ms_delta_counts": dict(milliseconds)}}


def analyze(root):
    root = Path(root).resolve()
    warnings = []
    status = read_json(root / "status.json", warnings, required=True)
    blocks = [analyze_block(block, root, warnings) for block in status.get("blocks", [])]
    grouped = defaultdict(list)
    for name in status.get("sources", {}):
        grouped[name] = []
    for block in blocks:
        grouped[block["source"]].extend(block["episodes"])
    complete = status.get("status") == "completed" and bool(blocks) and all(
        b["status"] == "completed" and len(b["episodes"]) == b["planned_episodes"]
        and all(e["complete"] for e in b["episodes"]) for b in blocks)
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "root": str(root), "source_status": status.get("status", "unknown"),
            "is_complete": complete and not warnings, "active_block": status.get("active_block"),
            "env_version": status.get("env_version"), "protocol": status.get("protocol", {}),
            "sources": status.get("sources", {}), "warnings": warnings,
            "metric_definitions": {
                "rates": "Completed episodes without environment errors; success additionally requires telemetry. Threshold rates require an observable initial target.",
                "initial_target": "The reset target identity, followed by native blip ID in the same process/episode. A new selected target is never substituted.",
                "initial_nearby_object_candidates": "Heuristic set of reset blips with blipType=3, color=4, sprite=0 and 3D distance <=20m. Freeze their native IDs; each snapshot uses the nearest still-visible member. Later floor markers never enter this set. Proximity is not a verified task event.",
                "final_distance": "Distance to the initial target at the last recorded state; null when it is absent. last_observed_distance is reported separately.",
                "first_reach": "First observed distance <= threshold; reset is step 0. No interpolation between 100ms snapshots.",
                "native_target_switch": "A different non-null selected native ID confirmed in raw blips. It is not a verified bomb pickup or mission success.",
                "horizontal_path": "Sum of horizontal distances between recorded heli positions; within-step curvature is not measured.",
                "world_z": "Absolute world Z, not height above terrain.",
                "health": "Native normalized health; missing heli is not silently assigned zero health.",
                "tier_scope": "Block run + trace episode + target identity + tier; reset IDs may repeat after process relaunch.",
                "integrity": "All observed trace steps including an in-progress episode. Other group metrics use valid completed episodes."},
            "models": {name: summarize_model(records) for name, records in sorted(grouped.items())},
            "blocks": blocks}


def markdown(report):
    def fmt(value, digits=3):
        return "—" if value is None else (str(value) if isinstance(value, int) else f"{value:.{digits}f}")
    def rate(value):
        return "—" if value is None else f"{value * 100:.1f}%"
    state = "已完成" if report["is_complete"] else "未完整结束：%s（只读快照）" % report["source_status"]
    lines = ["# 策略评估离线分析", "", "状态：**%s**。快照UTC：%s。" % (state, report["created_utc"]), "",
             "以下比例只统计已完成且无环境错误的回合；进行中的回合保留在JSON明细中。目标切换不等同已验证拾取。", "",
             "本轮每回合最多30秒，含相同的1.6秒起飞。窗口内未记录通关不等于任务失败；这也不是完整任务时限评估。结果限于这两个冻结模型及当前重置分布。", "",
             "| 模型来源 | 完成/进行中 | 环境错误 | 终止 | 成功 | 初始目标≤2.5m | 初始目标≤1.5m | 原生目标切换回合 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, m in report["models"].items():
        hits = m["reach_thresholds"]
        def hit(t):
            h = hits[t]
            return f"{h['episodes_reached']}/{h['denominator']} ({rate(h['rate'])})"
        lines.append(f"| {name} | {m['completed_episodes']}/{m['in_progress_episodes']} | {m['environment_error_episodes']} | {m['terminated_episodes']} | {m['successes']}/{m['success_denominator']} | {hit('2.5')} | {hit('1.5')} | {m['native_target_switch_episodes']}/{m['valid_completed_episodes']} |")
    lines += ["", "| 模型来源 | 每回合最小距离均值 | 末次距离均值（可见回合） | 首达2.5m/1.5m步数中位数 | 水平路程/净位移均值 | 世界Z范围 | 伤害回合 | 奖励/步 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, m in report["models"].items():
        h = m["reach_thresholds"]
        z = m["world_z_range"]
        lines.append(f"| {name} | {fmt(m['initial_target_min_distance']['mean'])} | {fmt(m['initial_target_final_distance']['mean'])} (n={m['initial_target_final_distance']['n']}) | {fmt(h['2.5']['first_reach_step']['median'])}/{fmt(h['1.5']['first_reach_step']['median'])} | {fmt(m['horizontal_sampled_path']['mean'])}/{fmt(m['horizontal_net_displacement']['mean'])} | {fmt(z['min'])}–{fmt(z['max'])} | {m['health_drop_episodes']} | {fmt(m['reward_per_step'])} |")
    lines += ["", "初始目标消失后的末次距离记为空，不使用新目标距离，也不把最后一次可见距离冒充回合结束距离。世界Z不是离地高度；路程按采样位置累计。", "",
              "固定首目标可能不是实际靠近的对象；下表另看初始附近候选集合：reset时type3/color4/sprite0、三维距离≤20m的原生ID，逐步取仍可见成员的最近距离。它是启发式候选，接近或消失均不视为验证任务事件，新的楼层标记不加入集合。", "",
              "候选集合指标是在发现60组接近另一对象后增加的描述性后验分析；固定首目标指标同时保留。这不是预注册显著性检验，也不证明某个停滞参数普遍更好。首达步数中位数仅在达到阈值的回合中计算，不能据此直接比较所有回合的整体到达速度。", "",
              "| 模型来源 | 候选数均值 | 集合最小距离均值 | ≤2.5m | ≤1.5m | ≤1.0m | 首达步数中位数（2.5/1.5/1.0m） |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, m in report["models"].items():
        c = m["initial_nearby_object_candidates"]
        h = c["reach_thresholds"]
        hit_cells = [f"{h[t]['episodes_reached']}/{h[t]['denominator']} ({rate(h[t]['rate'])})" for t in ("2.5", "1.5", "1.0")]
        first = "/".join(fmt(h[t]["first_reach_step"]["median"]) for t in ("2.5", "1.5", "1.0"))
        lines.append(f"| {name} | {fmt(c['initial_candidate_count']['mean'])} | {fmt(c['min_distance']['mean'])} | " + " | ".join(hit_cells) + f" | {first} |")
    lines += ["",
              "| 模型来源 | 时间成本/步 | 进展奖励/步 | 碰撞惩罚/步 | 碰撞惩罚步数 | 重复tier | 奖励分项最大误差 | 完整性异常 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, m in report["models"].items():
        t, integrity = m["reward_terms_per_step"], m["integrity"]
        issues = integrity["issues"]
        lines.append(f"| {name} | {fmt(t.get('time_cost'))} | {fmt(t.get('progress'))} | {fmt(t.get('collision'))} | {m['collision_penalty_steps']} | {issues.get('duplicate_target_tier', 0)} | {integrity['max_reward_sum_error']:.3g} | {sum(issues.values())} |")
    lines += ["", "完整性检查覆盖所有已读取step，包括未结束回合；帧/时间差分布、stage、目标切换详情、逐回合指标见summary.json。"]
    for name, m in report["models"].items():
        lines.append("- %s：frame差=%s；游戏毫秒差=%s；stage=%s。" %
                     (name, m["integrity"]["frame_delta_counts"], m["integrity"]["game_ms_delta_counts"], m["observed_stage_counts"]))
    if report["warnings"]:
        lines += ["", "读取提示（可能来自写入中的尾行）："]
        lines.extend("- " + warning for warning in report["warnings"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = analyze(args.root)
    output = args.output_dir or args.root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    # Only derived report files are written; evaluation status and data stay untouched.
    for name, content in (("summary.json", json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"),
                          ("summary.md", markdown(report))):
        destination = output / name
        temporary = output / (name + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(destination)
    print(json.dumps({"source_status": report["source_status"], "is_complete": report["is_complete"],
                      "models": {name: {"completed": m["completed_episodes"], "in_progress": m["in_progress_episodes"],
                                        "errors": m["environment_error_episodes"], "integrity": m["integrity"]["issues"]}
                                 for name, m in report["models"].items()},
                      "summary": str((output / "summary.json").resolve()),
                      "markdown": str((output / "summary.md").resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
