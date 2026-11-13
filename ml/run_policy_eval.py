"""Frozen-policy evaluation in balanced blocks, with owned-game cleanup."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import rl_sac
from calibrate_controls import _OwnedCalibrationTask, _cleanup_owned_processes
from rl_train import ENV_VERSION, RcHeliEnv
from vc_task import GAME_DIR


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(source_dir, output, episodes_per_block=5, steps=300, seed=1):
    if episodes_per_block <= 0 or steps <= 0:
        raise ValueError('Episode and step limits must be positive')
    order = [60, 200, 200, 60, 60, 200, 200, 60]
    sources = {}
    for stall in (60, 200):
        checkpoint = (source_dir / ('stall-%d' % stall) / 'checkpoint.pt').resolve()
        # Validate both frozen artifacts before creating a game process.
        rl_sac.SAC(RcHeliEnv.N_OBS).load(checkpoint, restore_rng=False)
        sources['stall-%d' % stall] = dict(checkpoint=str(checkpoint), sha256=fingerprint(checkpoint))
    output.mkdir(parents=True, exist_ok=False)
    status = dict(status='running', pid=os.getpid(), env_version=ENV_VERSION,
                  started_utc=datetime.now(timezone.utc).isoformat(), sources=sources,
                  protocol=dict(steps_per_episode=steps,
                                episodes_per_model=episodes_per_block * 4,
                                episodes_per_block=episodes_per_block,
                                stall_limit=0, launch_steps=16,
                                deterministic=True, seed=seed,
                                order='ABBAABBA', game_rng_controlled=False),
                  active_block=None, blocks=[])
    for index, stall in enumerate(order, 1):
        source = 'stall-%d' % stall
        status['blocks'].append(dict(index=index, source=source,
                                     path=str(output / ('block-%02d-%s' % (index, source))),
                                     episodes=episodes_per_block, status='pending'))

    def save():
        temporary = output / 'status.json.tmp'
        temporary.write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')
        os.replace(temporary, output / 'status.json')

    task = _OwnedCalibrationTask(game_dir=GAME_DIR, auto=False)

    class OwnedEvalEnv(RcHeliEnv):
        def __init__(self, **kwargs):
            super().__init__(task=task, **kwargs)

    save()
    try:
        task.ensure_game(task._launch())
        with patch.object(rl_sac, 'RcHeliEnv', OwnedEvalEnv):
            for block in status['blocks']:
                source = sources[block['source']]
                checkpoint = Path(source['checkpoint'])
                if fingerprint(checkpoint) != source['sha256']:
                    raise RuntimeError('Frozen checkpoint changed during evaluation')
                status['active_block'] = block['index']
                block['status'] = 'running'
                block['started_utc'] = datetime.now(timezone.utc).isoformat()
                save()
                print(json.dumps(dict(event='block_started', **block)), flush=True)
                path = rl_sac.evaluate(steps=steps, episodes=episodes_per_block,
                                       seed=seed, checkpoint=checkpoint,
                                       output_dir=block['path'], stall_limit=0,
                                       launch_steps=16, trace=True)
                summary = json.loads((path / 'summary.json').read_text(encoding='utf-8'))
                block.update(summary=summary, status=summary['status'],
                             finished_utc=datetime.now(timezone.utc).isoformat())
                save()
                if summary['status'] != 'completed' or summary['environment_errors']:
                    raise RuntimeError('Evaluation block has errors; inspect summary before continuing')
                if fingerprint(checkpoint) != source['sha256']:
                    raise RuntimeError('Frozen checkpoint changed during evaluation')
        status['status'] = 'completed'
        status['active_block'] = None
    except BaseException as exc:
        status['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        status['error'] = repr(exc)
        raise
    finally:
        try:
            processes, errors = _cleanup_owned_processes(task)
            status['owned_process_cleanup'] = processes
            status['cleanup_errors'] = errors
        except BaseException as exc:
            status['cleanup_errors'] = [repr(exc)]
        finally:
            status['finished_utc'] = datetime.now(timezone.utc).isoformat()
            save()
            print(json.dumps(dict(event='evaluation_finished', status=status['status'],
                                  path=str(output), cleanup_errors=status.get('cleanup_errors'))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes-per-block', type=int, default=5)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    run(args.source_dir.resolve(), args.output.resolve(), args.episodes_per_block, args.steps, args.seed)
