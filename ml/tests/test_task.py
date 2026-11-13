"""Pure task-reset stubs: no game launch or shared-memory connection."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vc_task import VcDemolitionTask


def snapshot(ready):
    return dict(missionRunning=1, playerState=2, heli=dict(health=1.4),
                mission=dict(valid=ready, id=19 if ready else -1))


class TutorialResetTests(unittest.TestCase):
    def task(self, ready=False):
        task = VcDemolitionTask(auto=False, verbose=False)
        task.game = Mock()
        task.game.get_state.return_value = snapshot(ready)
        return task

    def test_ready_native_mission_needs_no_tutorial_input(self):
        task = self.task(ready=True)
        self.assertTrue(task._advance_native_tutorial(timeout=1.))
        task.game.hold.assert_not_called()

    def test_exception_restores_neutral_after_minimum_pressure(self):
        task = self.task()
        with patch.object(task, 'wait_until', side_effect=[True, RuntimeError('read failed')]):
            with self.assertRaisesRegex(RuntimeError, 'read failed'):
                task._advance_native_tutorial(timeout=1.)
        self.assertEqual(len(task.game.hold.call_args_list), 2)
        self.assertEqual(task.game.hold.call_args_list[0].kwargs['cross_val'], 1. / 255.)
        self.assertEqual(task.game.hold.call_args_list[1].kwargs, {})

    def test_invalid_telemetry_timeout_is_not_ready_and_restores_neutral(self):
        task = self.task()
        with patch('vc_task.time.sleep'), patch('vc_task.time.perf_counter',
                                               side_effect=[0., 0., 0., 0., 0., 0., 0., 2.]):
            with self.assertRaises(TimeoutError):
                task._advance_native_tutorial(timeout=1.)
        self.assertEqual(len(task.game.hold.call_args_list), 2)
        self.assertEqual(task.game.hold.call_args_list[0].kwargs['cross_val'], 1. / 255.)
        self.assertEqual(task.game.hold.call_args_list[1].kwargs, {})


if __name__ == '__main__':
    unittest.main()
