"""Opt-in isolated live training/resume/evaluation integration check.

Uses small, explicitly recorded test-only replay warmup/batch settings.
Artifacts are verification data, not a trained task-solving policy.
"""
import argparse
import csv
import json
from pathlib import Path
from unittest.mock import patch

import torch

import rl_sac
from rl_train import RcHeliEnv
from calibrate_controls import _OwnedCalibrationTask, _cleanup_owned_processes
from vc_task import GAME_DIR


def run(output, game_dir):
    output.mkdir(parents=True, exist_ok=False)
    task, primary_error = None, None
    result = {'status': 'started', 'test_only_hyperparameters': True}

    class IsolatedEnv(RcHeliEnv):
        def __init__(self, **kwargs):
            super().__init__(task=task, **kwargs)

    try:
        task = _OwnedCalibrationTask(game_dir=str(game_dir), auto=False)
        pid = task._launch()
        task.ensure_game(pid)
        with patch.object(rl_sac, 'RcHeliEnv', IsolatedEnv), \
                patch.multiple(rl_sac, WARMUP=16, BATCH=8, GRADS_PER_STEP=1):
            train_dir = rl_sac.train(total_steps=70, minutes=5, seed=17,
                                     stall_limit=0, trace=True, run_dir=output / 'train')
            before = torch.load(train_dir / rl_sac.CKPT, weights_only=False)
            assert before['updates'] > 0, 'learner never performed an update'
            resumed = rl_sac.train(total_steps=30, minutes=5,
                                   resume=str(train_dir / rl_sac.CKPT),
                                   stall_limit=0, trace=True, run_dir=output / 'resumed')
            after = torch.load(resumed / rl_sac.CKPT, weights_only=False)
            assert after['updates'] > before['updates'], 'resume did not continue updating'
            assert after['training']['total_steps'] > before['training']['total_steps']
            assert before['training']['total_steps'] == 70
            assert after['training']['total_steps'] == 100
            evaluation = rl_sac.evaluate(steps=20, episodes=1, seed=17,
                                        checkpoint=resumed / rl_sac.CKPT,
                                        stall_limit=0, trace=True,
                                        output_dir=output / 'evaluation')
        for directory in (train_dir, resumed, evaluation):
            with (directory / 'episodes.csv').open(encoding='utf-8', newline='') as f:
                rows = list(csv.DictReader(f))
            assert all(not r['error'] for r in rows), rows
            assert all(int(r['steps']) > 0 for r in rows), rows
            run_summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
            assert run_summary['status'] == 'completed', run_summary
            assert (directory / 'trace.jsonl').stat().st_size > 0
        summary = json.loads((evaluation / 'summary.json').read_text())
        result.update(status='passed', updates_before_resume=before['updates'],
                      updates_after_resume=after['updates'],
                      total_steps=after['training']['total_steps'], evaluation=summary)
    except BaseException as exc:
        primary_error = exc
        result.update(status='failed', error=repr(exc))
        raise
    finally:
        records, cleanup_errors = [], []
        try:
            if task is not None:
                records, cleanup_errors = _cleanup_owned_processes(task)
        except BaseException as exc:
            cleanup_errors.append('unexpected cleanup failure: %r' % exc)
        result.update(process_cleanup=records, cleanup_errors=cleanup_errors)
        if cleanup_errors:
            result['status'] = 'failed'
            result.setdefault('error', 'owned-process cleanup was incomplete')
        (output / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result, indent=2), flush=True)
        if cleanup_errors and primary_error is None:
            raise RuntimeError('training verification cleanup incomplete; see verification.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--game-dir', type=Path, default=Path(GAME_DIR))
    args = parser.parse_args()
    run(args.output.resolve(), args.game_dir.resolve())
