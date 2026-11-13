"""Summarize SAC CSVs without mixing runs or treating invalid distances as success.

  python sac_stats.py runs/<run>/episodes.csv
  python sac_stats.py rl_sac_log_run1.csv rl_sac_log_run2.csv
"""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean


def number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def truth(value):
    return str(value).lower() in ("true", "1")


def split_runs(rows):
    groups = []
    for row in rows:
        if not groups or row.get("run_id") != groups[-1][-1].get("run_id"):
            groups.append([])
        elif number(row.get("episode")) is not None and groups[-1]:
            prev = number(groups[-1][-1].get("episode"))
            if prev is not None and number(row["episode"]) <= prev:
                groups.append([])
        groups[-1].append(row)
    return groups


def summarize(rows):
    values = lambda key: [v for row in rows if (v := number(row.get(key))) is not None]
    distances = [v for v in values("min_dist") if v >= 0]
    returns = values("return")
    qlosses = values("q_loss")
    result = {"episodes": len(rows), "mean_return": mean(returns) if returns else None,
              "valid_distances": len(distances),
              "mean_min_distance": mean(distances) if distances else None,
              "distance_under_4m_fraction": mean(d < 4 for d in distances) if distances else None,
              "distance_under_2_5m_fraction": mean(d < 2.5 for d in distances) if distances else None,
              "mean_q_loss": mean(qlosses) if qlosses else None,
              "exactly_62_steps": sum(number(r.get("steps")) == 62 for r in rows),
              "end_reasons": dict(Counter(r.get("reason", "unrecorded") for r in rows))}
    if all("success" in row for row in rows):
        valid = [r for r in rows if not r.get("error") and r.get("reason") != "environment_error"]
        known = [r for r in valid if truth(r.get("success_known", True))]
        result.update(valid_episodes=len(valid), environment_errors=len(rows) - len(valid),
                      outcome_known_episodes=len(known), outcome_unknown_episodes=len(valid) - len(known),
                      successes=sum(truth(r["success"]) for r in known),
                      success_rate=mean(truth(r["success"]) for r in known) if known else None,
                      success_rate_all_attempts=sum(truth(r["success"]) for r in known) / len(rows) if rows else None)
    else:
        result["success_rate"] = None
        result["note"] = "Legacy distances/events are proxies, not mission success labels"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        if path.is_dir():
            path = path / "episodes.csv"
        with path.open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            print(json.dumps({"file": str(path), "episodes": 0}))
        for segment, group in enumerate(split_runs(rows), 1):
            print(json.dumps({"file": str(path), "segment": segment,
                              "run_id": group[0].get("run_id", "legacy"),
                              "summary": summarize(group),
                              "first_100": summarize(group[:100]),
                              "last_100": summarize(group[-100:])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
