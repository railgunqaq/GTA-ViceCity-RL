"""Owned-process cleanup and smoke harness tests; no game or Win32 calls."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import calibrate_controls as cleanup
import smoke_training as smoke
from ml_client import GameDeadError


class ProcessApiStub:
    process = 11
    mutex = 22

    def __init__(self, exit_after=6000, *, busy=False, fail_wait=False, boundary=False):
        self.exit_after, self.busy = exit_after, busy
        self.fail_wait, self.boundary = fail_wait, boundary
        self.elapsed = 0
        self.terminated = self.exit_seen = False
        self.calls = []

    def WaitForSingleObject(self, handle, milliseconds):
        self.calls.append(('wait', handle, milliseconds))
        if handle == self.mutex:
            return 0x102 if self.busy else 0
        assert handle == self.process, 'unexpected process identity'
        if self.terminated and self.fail_wait and milliseconds:
            return 0xffffffff
        if self.terminated:
            self.elapsed += milliseconds
        if self.elapsed >= self.exit_after and (not self.boundary or not milliseconds):
            self.exit_seen = True
            return 0
        return 0x102

    def CreateMutexA(self, *args):
        self.calls.append(('create_mutex',))
        return self.mutex

    def TerminateProcess(self, handle, code):
        assert handle == self.process, 'termination must use original handle'
        self.calls.append(('terminate', handle))
        self.terminated = True
        return True

    def ReleaseMutex(self, handle):
        assert self.exit_seen, 'controller released before confirmed exit'
        self.calls.append(('release_mutex', handle))
        return True

    def CloseHandle(self, handle):
        if handle == self.process:
            assert self.exit_seen, 'original process handle lost before confirmed exit'
        self.calls.append(('close_handle', handle))
        return True


class GameStub:
    pid = 73

    def __init__(self, api):
        self.api, self.closed = api, False

    def _claim(self, timeout):
        self.api.calls.append(('claim',))

    def _stop_refresher(self):
        self.api.calls.append(('stop_refresher',))

    def close(self):
        assert self.api.exit_seen, 'bridge released before confirmed exit'
        self.api.calls.append(('close_game',))
        self.closed = True


def task_stub(api, game=None):
    return SimpleNamespace(process_api=api, game=game or GameStub(api),
                           process_cleanup_errors=[],
                           owned_processes=[dict(pid=73, handle=11, outcome='created')])


class CleanupTests(unittest.TestCase):
    def setUp(self):
        # These attributes do not exist on non-Windows ctypes builds. Error
        # conversion is stubbed too, so no test depends on host Win32 state.
        self.enterContext(patch.object(cleanup.ctypes, 'get_last_error', return_value=6, create=True))
        self.enterContext(patch.object(cleanup.ctypes, 'WinError',
                                      side_effect=lambda code: OSError(code, 'simulated Win32 error'),
                                      create=True))

    def test_exit_later_than_five_seconds_keeps_ownership_until_confirmation(self):
        api = ProcessApiStub(exit_after=6000)
        task = task_stub(api)
        records, errors = cleanup._cleanup_owned_processes(task)
        self.assertEqual(errors, [])
        self.assertTrue(records[0]['exit_confirmed'])
        self.assertEqual(api.elapsed, 6000)
        self.assertTrue(task.game.closed)
        self.assertIsNone(task.owned_processes[0]['handle'])
        self.assertLess(api.calls.index(('close_game',)),
                        api.calls.index(('release_mutex', api.mutex)))

    def test_timeout_retains_handles_and_same_thread_retry_does_not_terminate_again(self):
        api = ProcessApiStub(exit_after=6000)
        task = task_stub(api)
        records, errors = cleanup._cleanup_owned_processes(task, exit_timeout_ms=5000)
        self.assertEqual(len(errors), 1)
        self.assertIn('TimeoutError', errors[0])
        self.assertFalse(records[0]['exit_confirmed'])
        self.assertFalse(task.game.closed)
        self.assertEqual(task.owned_processes[0]['handle'], api.process)
        self.assertEqual(task.owned_processes[0]['cleanup_mutex'], api.mutex)
        self.assertFalse(any(c[0] in ('release_mutex', 'close_handle') for c in api.calls))
        records, errors = cleanup._cleanup_owned_processes(task, exit_timeout_ms=3000)
        self.assertEqual(errors, [])
        self.assertTrue(records[0]['exit_confirmed'])
        self.assertEqual(sum(c[0] == 'terminate' for c in api.calls), 1)
        self.assertEqual(sum(c[0] == 'create_mutex' for c in api.calls), 1)
        self.assertIsNone(task.owned_processes[0]['handle'])
        self.assertNotIn('cleanup_mutex', task.owned_processes[0])

    def test_timeout_boundary_is_rechecked(self):
        task = task_stub(ProcessApiStub(exit_after=5000, boundary=True))
        records, errors = cleanup._cleanup_owned_processes(task, exit_timeout_ms=5000)
        self.assertEqual(errors, [])
        self.assertTrue(records[0]['exit_confirmed'])

    def test_busy_controller_is_not_terminated_or_released(self):
        api = ProcessApiStub(busy=True)
        task = task_stub(api)
        records, errors = cleanup._cleanup_owned_processes(task)
        self.assertTrue(errors)
        self.assertEqual(records[0]['outcome'], 'controller_busy_skipped')
        self.assertFalse(api.terminated)
        self.assertFalse(task.game.closed)
        self.assertEqual(task.owned_processes[0]['handle'], api.process)
        self.assertNotIn('cleanup_mutex', task.owned_processes[0])
        self.assertIn(('close_handle', api.mutex), api.calls)

    def test_wait_failure_is_not_timeout_and_keeps_ownership(self):
        api = ProcessApiStub(fail_wait=True)
        task = task_stub(api)
        records, errors = cleanup._cleanup_owned_processes(task)
        self.assertTrue(errors)
        self.assertNotIn('TimeoutError', errors[0])
        self.assertIn('OSError', errors[0])
        self.assertFalse(records[0]['exit_confirmed'])
        self.assertFalse(task.game.closed)
        self.assertEqual(task.owned_processes[0]['cleanup_mutex'], api.mutex)

    def test_already_exited_process_is_closed_without_termination(self):
        api = ProcessApiStub(exit_after=0)
        task = task_stub(api)
        records, errors = cleanup._cleanup_owned_processes(task)
        self.assertEqual(errors, [])
        self.assertEqual(records[0]['outcome'], 'already_exited')
        self.assertTrue(task.game.closed)
        self.assertFalse(api.terminated)

    def sequence_task(self, waits):
        api = Mock()
        api.WaitForSingleObject.side_effect = waits
        api.CreateMutexA.return_value = 22
        api.TerminateProcess.return_value = True
        api.ReleaseMutex.return_value = api.CloseHandle.return_value = True
        return task_stub(api, game=Mock(pid=73))

    def test_exit_during_claim_is_confirmed_using_original_handle(self):
        task = self.sequence_task([0x102, 0, 0])
        task.game._claim.side_effect = GameDeadError('exited during claim')
        records, errors = cleanup._cleanup_owned_processes(task)
        self.assertEqual(errors, [])
        self.assertEqual(records[0]['outcome'], 'already_exited')
        self.assertIn('GameDeadError', records[0]['exit_confirmed_after_error'])
        task.process_api.TerminateProcess.assert_not_called()
        task.process_api.ReleaseMutex.assert_called_once_with(22)
        task.game.close.assert_called_once()

    def test_exit_in_error_handler_clears_false_timeout(self):
        task = self.sequence_task([0x102, 0, 0x102, 0x102, 0])
        records, errors = cleanup._cleanup_owned_processes(task, exit_timeout_ms=1000)
        self.assertEqual(errors, [])
        self.assertTrue(records[0]['exit_confirmed'])
        self.assertEqual(records[0]['outcome'], 'terminated')
        self.assertIn('TimeoutError', records[0]['exit_confirmed_after_error'])
        task.process_api.ReleaseMutex.assert_called_once_with(22)


class SmokeHarnessTests(unittest.TestCase):
    def exercise(self, cleanup_errors=(), cleanup_exception=None, training_exception=None,
                 attachment_exception=None):
        def artifacts(path, steps):
            path.mkdir()
            (path / 'episodes.csv').write_text('steps,error\n%d,\n' % steps, encoding='utf-8')
            (path / 'summary.json').write_text('{"status":"completed"}', encoding='utf-8')
            (path / 'trace.jsonl').write_text('{}\n', encoding='utf-8')
            return path

        def train(**kwargs):
            if training_exception:
                raise training_exception
            return artifacts(kwargs['run_dir'], kwargs['total_steps'])

        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
                patch.object(smoke, '_OwnedCalibrationTask') as constructor, \
                patch.object(smoke, '_cleanup_owned_processes',
                             return_value=([{'pid': 73, 'outcome': 'terminated'}], list(cleanup_errors)),
                             side_effect=cleanup_exception) as cleaned, \
                patch.object(smoke.rl_sac, 'train', side_effect=train) as trained, \
                patch.object(smoke.rl_sac, 'evaluate', side_effect=lambda **k: artifacts(k['output_dir'], k['steps'])) as evaluated, \
                patch.object(smoke.torch, 'load', side_effect=[
                    {'updates': 55, 'training': {'total_steps': 70}},
                    {'updates': 85, 'training': {'total_steps': 100}}]):
            constructor.return_value._launch.return_value = 73
            constructor.return_value.ensure_game.side_effect = attachment_exception
            output, game_dir = Path(tmp) / 'verification', Path(tmp) / 'game'
            caught = None
            try:
                smoke.run(output, game_dir)
            except BaseException as exc:
                caught = exc
            result = json.loads((output / 'verification.json').read_text(encoding='utf-8'))
            constructor.assert_called_once_with(game_dir=str(game_dir), auto=False)
            constructor.return_value._launch.assert_called_once()
            constructor.return_value.ensure_game.assert_called_once_with(73)
            cleaned.assert_called_once_with(constructor.return_value)
            return result, caught, trained, evaluated

    def test_owned_cleanup_is_recorded_and_original_step_budgets_preserved(self):
        result, error, trained, evaluated = self.exercise()
        self.assertIsNone(error)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['cleanup_errors'], [])
        self.assertEqual(result['process_cleanup'][0]['outcome'], 'terminated')
        self.assertEqual([c.kwargs['total_steps'] for c in trained.call_args_list], [70, 30])
        self.assertEqual([c.kwargs['minutes'] for c in trained.call_args_list], [5, 5])
        self.assertEqual(evaluated.call_args.kwargs['steps'], 20)

    def test_cleanup_errors_and_exceptions_fail_verification(self):
        for kwargs in ({'cleanup_errors': ['not exited']},
                       {'cleanup_exception': RuntimeError('cleanup unavailable')}):
            with self.subTest(kwargs=kwargs):
                result, error, _, _ = self.exercise(**kwargs)
                self.assertEqual(result['status'], 'failed')
                self.assertTrue(result['cleanup_errors'])
                self.assertIsInstance(error, RuntimeError)
                self.assertIn('cleanup incomplete', str(error))

    def test_primary_training_error_is_preserved_and_cleanup_still_recorded(self):
        original = ValueError('training failed')
        result, error, _, _ = self.exercise(cleanup_errors=['not exited'], training_exception=original)
        self.assertIs(error, original)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['error'], repr(original))
        self.assertEqual(result['cleanup_errors'], ['not exited'])

    def test_failed_attachment_still_cleans_the_launched_owned_process(self):
        original = RuntimeError('bridge attachment failed')
        result, error, trained, evaluated = self.exercise(attachment_exception=original)
        self.assertIs(error, original)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['cleanup_errors'], [])
        self.assertEqual(result['process_cleanup'][0]['outcome'], 'terminated')
        trained.assert_not_called()
        evaluated.assert_not_called()


if __name__ == '__main__':
    unittest.main()
