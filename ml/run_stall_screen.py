"""Run a bounded, one-factor SAC screen: stall 60 versus 200 steps.

Both conditions start new models with identical learner settings. This is a
single-seed mechanism screen, not a statistical claim of task performance.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from rl_sac import train
from rl_train import ENV_VERSION


def run(output, seed=1, steps=3000, minutes=30.0):
    if steps <= 0 or minutes <= 0:
        raise ValueError('Both step and time budgets must be positive')
    output.mkdir(parents=True, exist_ok=False)
    status = dict(status='running', pid=os.getpid(), env_version=ENV_VERSION,
                  started_utc=datetime.now(timezone.utc).isoformat(), seed=seed,
                  steps_per_condition=steps, minutes_per_condition=minutes,
                  conditions=[60, 200], active_condition=None, completed=[],
                  comparison='Only stall_limit differs; no checkpoint/replay is reused.')

    def save():
        temporary = output / 'status.json.tmp'
        temporary.write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')
        os.replace(temporary, output / 'status.json')

    save()
    try:
        for stall in status['conditions']:
            status['active_condition'] = stall
            status['active_run'] = str((output / ('stall-%d' % stall)).resolve())
            save()
            directory = train(run_dir=status['active_run'], seed=seed,
                              total_steps=steps, minutes=minutes,
                              stall_limit=stall, min_progress=1.0,
                              launch_steps=16, trace=True)
            summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
            status['completed'].append(dict(stall_limit=stall, path=str(directory), summary=summary))
            save()
            if summary['status'] != 'completed' or summary['environment_errors']:
                raise RuntimeError('Condition ended with an error; inspect its summary before continuing')
        status['status'] = 'completed'
        status['equal_step_budget_met'] = all(
            item['summary']['run_steps'] == steps for item in status['completed'])
        status['active_condition'] = None
        status['active_run'] = None
    except BaseException as exc:
        status['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        status['error'] = repr(exc)
        raise
    finally:
        status['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
        print(json.dumps(status, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--minutes', type=float, default=30.0)
    args = parser.parse_args()
    run(args.output.resolve(), args.seed, args.steps, args.minutes)
