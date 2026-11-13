"""Launch an isolated game to verify bridge-v6 stepping and native controls.

This is an opt-in integration check, never imported by offline test discovery.
It terminates only the game instance it created (including its replacement PID).
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from rl_train import RcHeliEnv, ENV_VERSION
from calibrate_controls import _OwnedCalibrationTask, _cleanup_owned_processes
from vc_task import GAME_DIR


def check(game_dir, output):
    task = _OwnedCalibrationTask(game_dir=str(game_dir), auto=False)
    result = dict(env_version=ENV_VERSION, steps=[])
    try:
        pid = task._launch()
        result['initial_pid'] = pid
        task.ensure_game(pid)
        env = RcHeliEnv(task=task)
        obs = env.reset()
        env.begin_episode()
        assert np.isfinite(obs).all(), 'non-finite reset observation'
        initial = task.observe()
        assert initial['lockstep'], 'reset did not pause the simulator'
        time.sleep(.35)
        paused = task.observe()
        assert paused['frame'] == initial['frame'], 'simulator advanced during learner delay'
        assert paused['gameTimerMs'] == initial['gameTimerMs'], 'clock advanced while paused'
        actions = ([0, 0, 0, .2, 0], [.1, 0, 0, .2, 0], [0, -.1, 0, .2, 0],
                   [0, 0, .1, .2, 0], [0, 0, 0, -.1, 0])
        previous = paused
        for action in actions:
            obs, reward, done, info = env.step_cont(action)
            current = task.observe()
            delta_frames = current['frame'] - previous['frame']
            delta_ms = current['gameTimerMs'] - previous['gameTimerMs']
            assert delta_frames == env.step_frames, (delta_frames, env.step_frames)
            assert delta_ms == env.step_frames * 10, delta_ms
            assert np.isfinite(obs).all() and np.isfinite(reward), 'non-finite transition'
            assert not done, 'small-control probe unexpectedly terminated the mission'
            distance = float(np.linalg.norm(np.asarray(current['heli']['pos']) -
                                            previous['heli']['pos']))
            assert distance < 10, 'implausible 100ms motion under small control'
            result['steps'].append(dict(action=action, delta_frames=delta_frames,
                                        delta_ms=delta_ms, distance=distance, reward=reward))
            time.sleep(.2)
            after_delay = task.observe()
            assert after_delay['frame'] == current['frame'], 'step terminal state did not remain paused'
            previous = current
        old_episode = (previous['pid'], previous['episodeId'])
        env.reset()
        reset = task.observe()
        assert (reset['pid'], reset['episodeId']) != old_episode, 'reset reused the prior episode identity'
        assert reset['lockstep'], 'second reset did not pause'
        assert env.episode_events == 0 and reset['mission']['completed_mask'] == 0, 'episode-local state leaked'
        task.release()
        released = task.observe()
        time.sleep(.2)
        resumed = task.observe()
        assert resumed['frame'] > released['frame'], 'release left simulator locked'
        result.update(status='passed', initial_episode=initial['episodeId'],
                      final_episode=reset['episodeId'], final_pid=task.game.pid,
                      reset_hard_restarted=reset['pid'] != previous['pid'])
    except BaseException as exc:
        result.update(status='failed', error=repr(exc))
        raise
    finally:
        result['cleanup'], result['cleanup_errors'] = _cleanup_owned_processes(task)
        if result['cleanup_errors']:
            result['status'] = 'failed'
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result, indent=2), flush=True)
        if result['cleanup_errors']:
            raise RuntimeError('owned game cleanup did not complete: %s' % result['cleanup_errors'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game-dir', type=Path, default=Path(GAME_DIR))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    check(args.game_dir.resolve(), args.output)
