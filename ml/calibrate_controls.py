"""Opt-in, isolated lockstep control calibration. Does not train a policy.

Every probe starts from a fresh game episode. Raw telemetry and actual inputs
are retained so world height is never confused with measured ground clearance.
"""
import argparse
import ctypes
import ctypes.wintypes as wt
import json
import math
from pathlib import Path
import time

import numpy as np

from ml_client import GameDeadError, ML_VERSION
from rl_train import ENV_VERSION, RcHeliEnv
from vc_task import GAME_DIR, VcDemolitionTask


def _process_api():
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.CreateProcessW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p,
                                  ctypes.c_void_p, ctypes.c_void_p, wt.BOOL,
                                  wt.DWORD, ctypes.c_void_p, ctypes.c_wchar_p,
                                  ctypes.c_void_p, ctypes.c_void_p]
    api.CreateProcessW.restype = wt.BOOL
    api.CreateMutexA.argtypes = [ctypes.c_void_p, wt.BOOL, ctypes.c_char_p]
    api.CreateMutexA.restype = wt.HANDLE
    api.WaitForSingleObject.argtypes = [wt.HANDLE, wt.DWORD]
    api.WaitForSingleObject.restype = wt.DWORD
    api.TerminateProcess.argtypes = [wt.HANDLE, wt.UINT]
    api.TerminateProcess.restype = wt.BOOL
    api.GetExitCodeProcess.argtypes = [wt.HANDLE, ctypes.POINTER(wt.DWORD)]
    api.GetExitCodeProcess.restype = wt.BOOL
    api.ReleaseMutex.argtypes = [wt.HANDLE]
    api.ReleaseMutex.restype = wt.BOOL
    api.CloseHandle.argtypes = [wt.HANDLE]
    api.CloseHandle.restype = wt.BOOL
    return api


class _OwnedCalibrationTask(VcDemolitionTask):
    """Retain original process handles; cleanup never resolves an old PID anew."""

    def __init__(self, *args, **kwargs):
        self.process_api = _process_api()
        self.owned_processes = []
        self.process_cleanup_errors = []
        super().__init__(*args, **kwargs)

    def _launch(self):
        class StartupInfo(ctypes.Structure):
            _fields_ = [('cb', wt.DWORD), ('lpReserved', ctypes.c_wchar_p),
                        ('lpDesktop', ctypes.c_wchar_p), ('lpTitle', ctypes.c_wchar_p),
                        ('dwX', wt.DWORD), ('dwY', wt.DWORD),
                        ('dwXSize', wt.DWORD), ('dwYSize', wt.DWORD),
                        ('dwXCountChars', wt.DWORD), ('dwYCountChars', wt.DWORD),
                        ('dwFillAttribute', wt.DWORD), ('dwFlags', wt.DWORD),
                        ('wShowWindow', wt.WORD), ('cbReserved2', wt.WORD),
                        ('lpReserved2', ctypes.c_void_p), ('hStdInput', wt.HANDLE),
                        ('hStdOutput', wt.HANDLE), ('hStdError', wt.HANDLE)]

        class ProcessInfo(ctypes.Structure):
            _fields_ = [('hProcess', wt.HANDLE), ('hThread', wt.HANDLE),
                        ('dwProcessId', wt.DWORD), ('dwThreadId', wt.DWORD)]

        # Earlier hard resets may have exited these processes. Drop their
        # handles before another launch, but retain their diagnostic records.
        for owned in self.owned_processes:
            handle = owned['handle']
            if handle and self.process_api.WaitForSingleObject(handle, 0) == 0:
                if self.process_api.CloseHandle(handle):
                    owned.update(handle=None, outcome='already_exited')
                else:
                    self.process_cleanup_errors.append(
                        'close exited process %d: Win32 %d' %
                        (owned['pid'], ctypes.get_last_error()))

        startup = StartupInfo()
        startup.cb = ctypes.sizeof(startup)
        startup.dwFlags = 1  # STARTF_USESHOWWINDOW
        startup.wShowWindow = 4  # SW_SHOWNOACTIVATE; matches the ordinary launcher
        process = ProcessInfo()
        exe = str(Path(self.game_dir) / 'reVC.exe')
        if not self.process_api.CreateProcessW(
                exe, None, None, None, False, 0, None, self.game_dir,
                ctypes.byref(startup), ctypes.byref(process)):
            raise ctypes.WinError(ctypes.get_last_error())
        pid = int(process.dwProcessId)
        # Record before any following operation can fail. The returned handle
        # names this exact process even if its PID is later reused.
        self.owned_pids.add(pid)
        self.owned_processes.append(dict(pid=pid, handle=process.hProcess,
                                         outcome='created'))
        if not self.process_api.CloseHandle(process.hThread):
            self.process_cleanup_errors.append(
                'close launch thread for %d: Win32 %d' % (pid, ctypes.get_last_error()))
        return pid


def _wait_owned_process_exit(api, handle, timeout_ms):
    """Observe this exact process, distinguishing timeout from wait failure."""
    remaining = timeout_ms
    while remaining > 0:
        interval = min(1000, remaining)
        result = api.WaitForSingleObject(handle, interval)
        if result == 0:
            return True
        if result != 0x102:
            raise ctypes.WinError(ctypes.get_last_error())
        remaining -= interval
    # Cover exit on the timeout boundary before declaring cleanup incomplete.
    result = api.WaitForSingleObject(handle, 0)
    if result not in (0, 0x102):
        raise ctypes.WinError(ctypes.get_last_error())
    return result == 0


def _cleanup_owned_processes(task, exit_timeout_ms=30000):
    """Terminate retained handles; keep ownership until exit is confirmed.

    TerminateProcess is asynchronous, including cancellation of pending I/O.
    An unconfirmed exit retains both handles for retry on this same controller
    thread, instead of losing the only safe process identity or opening a race.
    """
    if not isinstance(exit_timeout_ms, int) or exit_timeout_ms < 0:
        raise ValueError('exit_timeout_ms must be a nonnegative integer')
    errors = list(task.process_cleanup_errors)
    records = []
    game = task.game
    closed_game = False
    api = task.process_api

    def close_game():
        nonlocal closed_game
        if game is None or closed_game:
            return
        closed_game = True
        try:
            game.close()
        except GameDeadError:
            pass  # Expected after successful termination; close still unmaps.
        except BaseException as exc:
            errors.append('close bridge: %r' % exc)

    for owned in task.owned_processes:
        handle = owned['handle']
        record = dict(pid=owned['pid'], outcome=owned['outcome'])
        records.append(record)
        if not handle:
            continue
        mutex = owned.get('cleanup_mutex')
        locked, exited = bool(mutex), False
        try:
            wait = api.WaitForSingleObject(handle, 0)
            if wait == 0:
                exited = True
                record['outcome'] = 'already_exited'
            elif wait != 0x102:  # WAIT_TIMEOUT means the process is alive.
                raise ctypes.WinError(ctypes.get_last_error())
            else:
                if not locked:
                    mutex = api.CreateMutexA(
                        None, False, ('ReVC_ML_Controller_PID%d' % owned['pid']).encode())
                    if not mutex:
                        raise ctypes.WinError(ctypes.get_last_error())
                    acquired = api.WaitForSingleObject(mutex, 0)
                    if acquired == 0x102:
                        record['outcome'] = 'controller_busy_skipped'
                        errors.append('process %d is controlled elsewhere; not terminated' % owned['pid'])
                        continue
                    if acquired not in (0, 0x80):  # WAIT_OBJECT_0 / WAIT_ABANDONED
                        raise ctypes.WinError(ctypes.get_last_error())
                    locked = True
                # The task's existing ownership stays held until the process
                # exits. This extra named mutex also covers half-attached games.
                if not owned.get('termination_requested') and game is not None and game.pid == owned['pid']:
                    game._claim(timeout=0.0)
                    try:
                        game._stop_refresher()
                    except BaseException as exc:
                        errors.append('stop refresher for %d: %r' % (owned['pid'], exc))
                if not owned.get('termination_requested'):
                    if api.TerminateProcess(handle, 1):
                        owned['termination_requested'] = True
                    else:
                        terminate_error = ctypes.get_last_error()
                        if api.WaitForSingleObject(handle, 0) != 0:
                            raise ctypes.WinError(terminate_error)
                        exited = True
                wait_start = time.perf_counter()
                try:
                    exited = exited or _wait_owned_process_exit(api, handle, exit_timeout_ms)
                finally:
                    record['exit_wait_seconds'] = time.perf_counter() - wait_start
                    record['exit_timeout_ms'] = exit_timeout_ms
                if not exited:
                    raise TimeoutError('owned process %d exit not confirmed after %d ms; '
                                       'process handle and controller mutex retained' %
                                       (owned['pid'], exit_timeout_ms))
                record['outcome'] = 'terminated'
        except BaseException as exc:
            # The process may exit between a failed claim/wait and its error
            # handler. Preserve that diagnostic, but report the observed exit.
            if api.WaitForSingleObject(handle, 0) == 0:
                exited = True
                record.update(outcome='terminated' if owned.get('termination_requested')
                              else 'already_exited', exit_confirmed_after_error=repr(exc))
            else:
                record.update(outcome='cleanup_error', error=repr(exc))
                errors.append('process %d: %r' % (owned['pid'], exc))
        finally:
            # A request is not an exit: release ownership only after the exact
            # retained process handle becomes signaled, including late exits.
            if exited and game is not None and game.pid == owned['pid']:
                close_game()
            if locked and not exited:
                owned['cleanup_mutex'] = mutex
            else:
                if locked and not api.ReleaseMutex(mutex):
                    errors.append('release cleanup mutex for %d: Win32 %d' %
                                  (owned['pid'], ctypes.get_last_error()))
                if mutex and not api.CloseHandle(mutex):
                    errors.append('close cleanup mutex for %d: Win32 %d' %
                                  (owned['pid'], ctypes.get_last_error()))
                owned.pop('cleanup_mutex', None)
            if exited:
                if api.CloseHandle(handle):
                    owned['handle'] = None
                else:
                    errors.append('close process %d: Win32 %d' %
                                  (owned['pid'], ctypes.get_last_error()))
            owned['outcome'] = record['outcome']
            record['exit_confirmed'] = exited
            record['handles_retained'] = bool(owned['handle'])
    if game is None or not any(owned['pid'] == game.pid and owned['handle']
                               for owned in task.owned_processes):
        close_game()
    return records, errors


def tilt(heli):
    up = np.asarray(heli['up'], dtype=float)
    return math.degrees(math.acos(float(np.clip(up[2] / np.linalg.norm(up), -1, 1))))


def probes(axes=False):
    for collective in (0.0, 0.25, 0.35, 0.60):
        yield 'collective_%g' % collective, [[0., 0., 0., collective, 0.]] * 30
    if axes:
        for axis, name in enumerate(('roll', 'pitch', 'yaw')):
            for sign in (-1, 1):
                action = [0., 0., 0., .35, 0.]
                action[axis] = sign * .15
                # Keep the vertical baseline fixed across signs. The measured
                # trajectories, not the nominal baseline, define the outcome.
                yield '%s_%+d' % (name, sign), (
                    [[0., 0., 0., .35, 0.]] * 16 + [action] * 5 +
                    [[0., 0., 0., .35, 0.]] * 10)


def run(output, game_dir, axes=False):
    output.mkdir(parents=True, exist_ok=False)
    task = None
    result = dict(env_version=ENV_VERSION, bridge_version=ML_VERSION,
                  status='started', probes=[], axes=axes,
                  note='World Z is not ground clearance; game RNG is uncontrolled.')
    try:
        task = _OwnedCalibrationTask(game_dir=str(game_dir), auto=False)
        pid = task._launch()
        task.ensure_game(pid)
        env = RcHeliEnv(task=task, verbose=False)
        with (output / 'trace.jsonl').open('w', encoding='utf-8') as trace:
            for name, actions in probes(axes):
                reset_start = time.perf_counter()
                env.reset()
                reset_seconds = time.perf_counter() - reset_start
                initial = task.observe()
                previous = initial
                initial_pos = np.asarray(initial['heli']['pos'], dtype=float)
                initial_health = initial['heli']['health']
                reason = 'probe_complete'
                steps = 0
                maximum_tilt = tilt(initial['heli'])
                for index, action in enumerate(actions):
                    obs, reward, done, info = env.step_cont(action)
                    current = task.observe()
                    assert current['frame'] - previous['frame'] == 10
                    assert current['gameTimerMs'] - previous['gameTimerMs'] == 100
                    assert current['pid'] == initial['pid']
                    assert current['episodeId'] == initial['episodeId']
                    trace.write(json.dumps(dict(probe=name, step=index + 1,
                                               action=action, before=previous,
                                               after=current, obs=obs.tolist(),
                                               reward=reward,
                                               reward_terms=info.get('reward_terms'),
                                               target_id=info.get('target_id')),
                                           separators=(',', ':')) + '\n')
                    trace.flush()
                    steps += 1
                    previous = current
                    if done or current['heli'] is None:
                        reason = info.get('reason') or 'terminated'
                        break
                    heli = current['heli']
                    maximum_tilt = max(maximum_tilt, tilt(heli))
                    delta = np.asarray(heli['pos']) - initial_pos
                    if (maximum_tilt > 35 or heli['health'] < initial_health - .05
                            or abs(delta[2]) > 3 or np.linalg.norm(delta[:2]) > 5):
                        reason = 'probe_safety_bound'
                        break
                last = previous['heli']
                summary = dict(name=name, model_id=initial['heli']['modelId'],
                               steps=steps, reset_seconds=reset_seconds, reason=reason,
                               initial_pos=initial['heli']['pos'],
                               final_pos=last['pos'] if last else None,
                               delta_z=last['pos'][2] - initial_pos[2] if last else None,
                               horizontal_displacement=float(np.linalg.norm(
                                   np.asarray(last['pos'])[:2] - initial_pos[:2])) if last else None,
                               max_tilt_degrees=maximum_tilt,
                               final_native_velocity=last['vel'] if last else None,
                               initial_health=initial_health,
                               final_health=last['health'] if last else None,
                               initial_blips=initial['blips'],
                               final_blips=previous['blips'])
                result['probes'].append(summary)
                print(json.dumps(summary), flush=True)
        result['status'] = 'completed'
    except BaseException as exc:
        result.update(status='failed', error=repr(exc))
        raise
    finally:
        cleanup_errors = []
        try:
            if task is not None:
                records, cleanup_errors = _cleanup_owned_processes(task)
                result['process_cleanup'] = records
        except BaseException as exc:
            cleanup_errors.append('unexpected cleanup failure: %r' % exc)
        finally:
            result['cleanup_errors'] = cleanup_errors
            if cleanup_errors:
                result['status'] = 'failed'
                result.setdefault('error', 'owned-process cleanup was incomplete')
            # Always attempt the summary, including failed attachment, game
            # loss, keyboard interruption, or any individual cleanup failure.
            (output / 'summary.json').write_text(
                json.dumps(result, indent=2) + '\n', encoding='utf-8')
        if cleanup_errors and not result.get('error', '').startswith(('KeyboardInterrupt',)):
            # Preserve a primary exception already propagating from the try.
            import sys
            if sys.exc_info()[0] is None:
                raise RuntimeError('calibration cleanup incomplete; see summary.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--game-dir', type=Path, default=Path(GAME_DIR))
    parser.add_argument('--axes', action='store_true')
    args = parser.parse_args()
    run(args.output.resolve(), args.game_dir.resolve(), args.axes)
