"""Summarize CSV runs without inferring versions from file-tail lengths."""
import argparse
import csv
import json
import math
from pathlib import Path


def split_runs(rows):
    runs = []
    previous_episode, previous_id = None, None
    for row in rows:
        episode = int(row['episode'])
        run_id = row.get('run_id')
        if (not runs or run_id != previous_id or
                (previous_episode is not None and episode <= previous_episode)):
            runs.append([])
        runs[-1].append(row)
        previous_episode, previous_id = episode, run_id
    return runs


def summarize(rows):
    distances = []
    for row in rows:
        try:
            value = float(row['min_dist'])
        except (ValueError, TypeError, KeyError):
            continue
        if math.isfinite(value) and value >= 0:
            distances.append(value)
    result = dict(episodes=len(rows), run_id=rows[0].get('run_id'),
                  mean_return=sum(float(r['return']) for r in rows) / len(rows),
                  valid_distances=len(distances),
                  mean_min_distance=sum(distances) / len(distances) if distances else None,
                  proximity_lt_2_5=sum(d < 2.5 for d in distances) / len(distances)
                  if distances else None,
                  steps_62=sum(int(r['steps']) == 62 for r in rows))
    if all(r.get('success', '').lower() in ('true', 'false', '1', '0') for r in rows):
        result['success_rate'] = sum(r['success'].lower() in ('true', '1')
                                     for r in rows) / len(rows)
    else:
        result['success_rate'] = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_path', type=Path,
                        help='Explicit CSV file; versions are never guessed from row offsets.')
    parser.add_argument('--segment', type=int, help='One-based segment to summarize')
    args = parser.parse_args()
    with args.csv_path.open(encoding='utf-8-sig', newline='') as f:
        runs = split_runs(list(csv.DictReader(f)))
    if args.segment is not None:
        if not 1 <= args.segment <= len(runs):
            parser.error('segment is outside the available run range')
        selected = [(args.segment, runs[args.segment - 1])]
    else:
        selected = list(enumerate(runs, 1))
    print(json.dumps([dict(segment=i, **summarize(r)) for i, r in selected], indent=2))


if __name__ == '__main__':
    main()
