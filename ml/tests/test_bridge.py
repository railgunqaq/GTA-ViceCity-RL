"""Bridge contract tests using in-memory ctypes structures, never a live game."""
from contextlib import nullcontext
import ctypes
import os
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ml_client as bridge
from vc_task import VcDemolitionTask


def client():
    c = bridge.ReVCInstance.__new__(bridge.ReVCInstance)
    c.shared = bridge.MlShared()
    c._pending = bridge.MlAction()
    c._pending.magic = bridge.ML_MAGIC
    c._pending.version = bridge.ML_VERSION
    c._pending.controllerPid = os.getpid()
    c._lock = threading.RLock()
    c._action_mutex = c._state_mutex = None
    c._mutex = lambda *args, **kwargs: nullcontext()
    c._claim = lambda timeout=3: None
    c._drop_claim = lambda: None
    c._require_alive = lambda: None
    c._start_refresher = lambda: None
    c._stop_refresher = lambda: None
    c._action_seq = c._command_seq = c._step_seq = c._query_seq = 0
    return c


def mission_state():
    s = bridge.MlState()
    s.missionTelemetryValid = 1
    s.missionId = 19
    s.missionEpoch = 7
    for i, objective in enumerate(s.objectives):
        objective.kind, objective.index = 1 + i // 4, i % 4
        objective.valid = 1
        objective.x, objective.y, objective.z = 10 + i, 20 + i, 30 + i
    return s


class BridgeTests(unittest.TestCase):
    def test_collision_diagnostics_preserve_submetre_physical_hits(self):
        s = bridge.MlState()
        s.heliExists, s.sensorVersion = 1, 2
        s.clearanceRadius, s.bodyBoundRadius = 1.2, .7
        hit = s.rayHits[0]
        hit.valid, hit.hit, hit.entityType, hit.modelId, hit.surface = 1, 1, 1, 123, 44
        hit.distance = .25
        hit.point[:], hit.normal[:] = (1., 2., 3.), (-1., 0., 0.)
        s.rays[0] = .25 / 20.
        st = bridge.ReVCInstance._decode_state(s)
        self.assertEqual(st['heli']['sensor_schema'], 'collision-rays-v2')
        self.assertAlmostEqual(st['heli']['rays'][0] * 20., .25)
        self.assertEqual(st['heli']['ray_hits'][0], dict(valid=True, hit=True, entity_type=1,
                         model_id=123, surface=44, distance=.25, point=(1., 2., 3.), normal=(-1., 0., 0.)))
        self.assertFalse(st['heli']['ray_hits'][1]['valid'])
        hit.normal[0] = float('nan')
        with self.assertRaisesRegex(RuntimeError, 'collision hit geometry'):
            bridge.ReVCInstance._decode_state(s)

    def test_query_unknown_or_unloaded_is_never_clear(self):
        q = bridge.MlQueryResult()
        for status in (0, -1, -2, -3, -4):
            q.status = status
            decoded = bridge.ReVCInstance._decode_query(q)
            self.assertFalse(decoded['valid'])
            self.assertTrue(decoded['blocked'])
        q.status, q.radius, q.sampleSpacing, q.length = 1, 1., .25, 2.
        with self.assertRaisesRegex(RuntimeError, 'path-query geometry'):
            bridge.ReVCInstance._decode_query(q)
        q.line.valid = q.sweep.valid = 1
        self.assertFalse(bridge.ReVCInstance._decode_query(q)['blocked'])
        q.sweep.hit = 1
        self.assertTrue(bridge.ReVCInstance._decode_query(q)['blocked'])

    @staticmethod
    def query_client():
        c = client()
        c._pending.active, c._pending.lockstep, c._pending.stepSeq = 1, 1, 9
        c._pending.actionSeq, c._pending.LeftStickX = 4, 13
        base = dict(frame=30, gameTimerMs=300, episodeId=2, pid=42,
                    lockstep=True, stepActive=False, controllerPid=os.getpid(),
                    heli=dict(clearance_radius=1.1))
        def state():
            return dict(base, query=dict(seq=c._pending.querySeq, status=1, valid=True,
                        blocked=False, frame=30, episode_id=2, radius=1.1, sample_spacing=.25))
        c.get_state = state
        return c, base

    def test_path_query_preserves_actions_step_and_simulation(self):
        c, _ = self.query_client()
        result = c.query_path((1, 2, 3), (4, 5, 6))
        self.assertTrue(result['valid'])
        self.assertEqual(result['seq'], 1)
        self.assertEqual(c.shared.action.querySeq, 1)
        self.assertEqual(tuple(c.shared.action.queryStart), (1, 2, 3))
        self.assertEqual((c.shared.action.actionSeq, c.shared.action.stepSeq, c.shared.action.LeftStickX), (4, 9, 13))
        self.assertEqual(c.query_path((1, 2, 3), (1, 2, 3))['seq'], 2)

    def test_path_query_rejects_invalid_points_small_radius_and_free_running(self):
        c, base = self.query_client()
        for start, end, radius in [((float('nan'), 2, 3), (4, 5, 6), None),
                                  ((1, 2), (4, 5, 6), None),
                                  ((1, 2, 3), (401, 5, 6), None),
                                  ((1, 2, 3), (4, 5, 6), .5)]:
            with self.assertRaises(ValueError):
                c.query_path(start, end, radius)
            self.assertEqual(c.shared.action.seq, 0)
        base['lockstep'] = False
        with self.assertRaisesRegex(RuntimeError, 'frozen lockstep'):
            c.query_path((1, 2, 3), (4, 5, 6))
        self.assertEqual(c.shared.action.seq, 0)

    def test_path_query_rejects_old_ack_time_change_and_unloaded_geometry(self):
        for variant in ('old', 'advanced', 'unloaded', 'stale'):
            c, _ = self.query_client()
            original = c.get_state
            def state():
                st = original()
                if c.shared.action.querySeq:
                    if variant == 'old': st['query']['seq'] = 0
                    elif variant == 'advanced': st['gameTimerMs'] += 10
                    elif variant == 'unloaded': st['query'].update(valid=False, status=-3)
                    else: st['query']['frame'] -= 1
                return st
            c.get_state = state
            with self.subTest(variant=variant), self.assertRaises((RuntimeError, TimeoutError)):
                c.query_path((1, 2, 3), (4, 5, 6), timeout=.005)

    def test_unloaded_geometry_has_structured_exception_only_for_status_minus_three(self):
        for status in (-1, -2, -3, -4):
            c, _ = self.query_client()
            original = c.get_state
            def state():
                st = original()
                if c.shared.action.querySeq:
                    st['query'].update(valid=False, blocked=True, status=status)
                return st
            c.get_state = state
            with self.subTest(status=status), self.assertRaises(RuntimeError) as caught:
                c.query_path((1, 2, 3), (4, 5, 6))
            if status == -3:
                self.assertIsInstance(caught.exception, bridge.GeometryUnavailableError)
                self.assertEqual(caught.exception.query_status, -3)
                self.assertEqual(caught.exception.result['seq'], 1)
                self.assertFalse(caught.exception.result['valid'])
                self.assertTrue(caught.exception.result['blocked'])
            else:
                self.assertNotIsInstance(caught.exception, bridge.GeometryUnavailableError)

    def test_native_controller_ranges_and_continuous_brake(self):
        c = client()
        c._controls(lx=1, ly=-1, rx=.5, ry=-.5, cross_val=.5, brake_val=.25,
                    buttons={'Circle'}, dpad=(0, 1, 0, 0))
        a = c.shared.action
        self.assertEqual((a.LeftStickX, a.LeftStickY, a.RightStickX, a.RightStickY),
                         (128, -128, 64, -64))
        self.assertEqual((a.Cross, a.Square, a.Circle, a.DPadDown), (128, 64, 255, 255))
        self.assertEqual(a.seq % 2, 0)
        self.assertEqual(a.actionSeq, 1)

    def test_nonfinite_input_is_rejected_before_publication(self):
        c = client()
        for kwargs in ({'lx': float('nan')}, {'cross_val': float('inf')}, {'brake_val': -float('inf')}):
            with self.assertRaises(ValueError):
                c._controls(**kwargs)
            self.assertEqual(c.shared.action.seq, 0)

    def test_new_command_does_not_accept_previous_ack(self):
        c = client()
        c._command_seq = 12
        c.get_state = lambda: dict(lastCommandSeq=12, lastCommandResult=1, controllerPid=0)
        with self.assertRaises(TimeoutError):
            c.send_mission(19, wait_ack=.005)
        self.assertEqual(c.shared.action.commandSeq, 13)

    def test_step_accepts_only_matching_complete_snapshot(self):
        c = client()
        def get_state():
            return dict(lastActionSeq=c._action_seq, lastStepSeq=c._step_seq,
                        stepActive=False, lockstep=True, controllerPid=os.getpid(),
                        stepFrames=10, frame=30, stepStartFrame=20,
                        gameTimerMs=300, stepStartGameMs=200, simStepMs=10)
        c.get_state = get_state
        self.assertEqual(c.step(frames=10)['frame'], 30)
        original = c.get_state
        c.get_state = lambda: dict(original(), gameTimerMs=350)
        with self.assertRaisesRegex(RuntimeError, 'game time mismatch'):
            c.step(frames=10)

    def test_step_refuses_old_step_ack(self):
        c = client()
        c.get_state = lambda: dict(lastActionSeq=c._action_seq, lastStepSeq=0,
                                  stepActive=False, lockstep=True, controllerPid=os.getpid())
        with self.assertRaises(TimeoutError):
            c.step(frames=1, timeout=.005)

    def test_pause_is_zero_frame_transaction(self):
        c = client()
        c.get_state = lambda: dict(lastActionSeq=c._action_seq, lastStepSeq=c._step_seq,
                                  stepActive=False, lockstep=True, controllerPid=os.getpid(),
                                  stepFrames=0, frame=30, stepStartFrame=30,
                                  gameTimerMs=300, stepStartGameMs=300, simStepMs=10)
        c.pause()
        self.assertEqual(c.shared.action.stepFrames, 0)
        self.assertEqual(c.shared.action.lockstep, 1)

    def test_unrecorded_frames_between_steps_invalidate_transition(self):
        c = client()
        c._last_step_state = dict(frame=20, gameTimerMs=200, episodeId=1, pid=42)
        c.get_state = lambda: dict(lastActionSeq=c._action_seq, lastStepSeq=c._step_seq,
                                  stepActive=False, lockstep=True, controllerPid=os.getpid(),
                                  stepFrames=10, frame=40, stepStartFrame=30,
                                  gameTimerMs=400, stepStartGameMs=300, simStepMs=10,
                                  episodeId=1, pid=42)
        with self.assertRaisesRegex(RuntimeError, 'advanced between'):
            c.step(frames=10)
        self.assertIsNone(c._last_step_state)
        with self.assertRaisesRegex(RuntimeError, 'previous step failed'):
            c.step(frames=10)

    def test_wait_frames_timeout_is_error(self):
        c = client()
        c.get_state = lambda: dict(lockstep=False, stepActive=False, frame=1)
        with self.assertRaises(TimeoutError):
            c.wait_frames(10, timeout=.005)

    def test_even_snapshot_includes_native_success_and_epoch(self):
        c = client()
        s = c.shared.state
        s.magic = bridge.ML_MAGIC
        s.version = bridge.ML_VERSION
        s.seq = 2
        s.episodeId = 11
        s.missionSuccessCount = 3
        s.lastStepSeq = 18
        s.lockstep = 1
        st = c.get_state()
        self.assertEqual((st['episodeId'], st['missionSuccessCount'], st['lastStepSeq']), (11, 3, 18))
        s.seq = 3
        with self.assertRaises(TimeoutError):
            c.get_state(retries=1, stable_gap=0)

    def test_observer_release_does_not_write(self):
        c = client()
        c._owns_controller = False
        c._view = 1
        c.shared.action.controllerPid = 999
        c.shared.action.active = 1
        c.release()
        self.assertEqual(c.shared.action.controllerPid, 999)
        self.assertEqual(c.shared.action.active, 1)

    def test_layout_has_explicit_v7_abi(self):
        self.assertEqual(bridge.ML_VERSION, 7)
        self.assertEqual(ctypes.sizeof(bridge.MlBlip), 36)
        self.assertEqual(bridge.MlBlip.blipId.offset, 24)
        self.assertEqual(bridge.MlBlip.objectiveKind.offset, 28)
        self.assertEqual(bridge.MlBlip.objectiveIndex.offset, 32)
        self.assertEqual(ctypes.sizeof(bridge.MlObjective), 24)
        self.assertEqual(ctypes.sizeof(bridge.MlHit), 48)
        self.assertEqual(ctypes.sizeof(bridge.MlQueryResult), 124)
        self.assertEqual(bridge.MlState.rayHits.offset, 1652)
        self.assertEqual(bridge.MlState.query.offset, 2300)
        self.assertEqual(bridge.MlAction.querySeq.offset, 136)
        self.assertEqual(ctypes.sizeof(bridge.MlState), 2424)
        self.assertEqual(ctypes.sizeof(bridge.MlAction), 168)
        self.assertEqual(ctypes.sizeof(bridge.MlShared), 2592)
        self.assertEqual(bridge.MlState.blips.offset, 160)
        self.assertEqual(bridge.MlState.lastActionSeq.offset, 1312)
        self.assertEqual(bridge.MlState.missionTelemetryValid.offset, 1420)
        self.assertEqual(bridge.MlState.missionEpoch.offset, 1428)
        self.assertEqual(bridge.MlState.bombState.offset, 1432)
        self.assertEqual(bridge.MlState.targetCompleteMask.offset, 1448)
        self.assertEqual(bridge.MlState.objectives.offset, 1460)
        self.assertEqual(bridge.MlShared.action.offset, 2424)

    def test_attach_rejects_old_protocol_before_reading_larger_layout(self):
        for version, state_size in ((3, 1012), (4, 1036), (5, 1164), (6, 1652), (7, 1652)):
            with self.subTest(version=version, state_size=state_size):
                # Only a common header exists: do not interpret it as v6 state.
                header = (ctypes.c_uint32 * 4)(bridge.ML_MAGIC, version, state_size, 42)
                kernel = Mock()
                kernel.OpenFileMappingA.return_value = 123
                kernel.MapViewOfFile.return_value = ctypes.addressof(header)
                with patch.object(bridge, '_kernel32', return_value=kernel), \
                        patch.object(bridge.ReVCInstance, 'get_state') as get_state:
                    with self.assertRaisesRegex(RuntimeError, 'layout/version mismatch'):
                        bridge.ReVCInstance(pid=42)
                    get_state.assert_not_called()
                    kernel.CreateMutexA.assert_not_called()
                kernel.UnmapViewOfFile.assert_called_once()
                kernel.CloseHandle.assert_called_once_with(123)

    def test_state_rejects_old_protocol_even_with_v7_sized_buffer(self):
        c = client()
        c.shared.state.magic = bridge.ML_MAGIC
        c.shared.state.version = 5
        c.shared.state.seq = 2
        with self.assertRaisesRegex(RuntimeError, 'invalid state layout'):
            c.get_state()

    def test_blip_identity_survives_position_motion_and_array_reordering(self):
        s = bridge.MlState()
        s.numBlips = 2
        s.blips[0].blipId = (7 << 16) | 12
        s.blips[0].x = 2.49
        s.blips[1].blipId = (3 << 16) | 13
        s.blips[1].x = 9.0
        before = bridge.ReVCInstance._decode_state(s)['blips']
        old_first = bridge.MlBlip.from_buffer_copy(bytes(s.blips[0]))
        s.blips[0] = bridge.MlBlip.from_buffer_copy(bytes(s.blips[1]))
        s.blips[1] = old_first
        s.blips[1].x = 2.51  # crosses the old rounded-coordinate identity boundary
        after = bridge.ReVCInstance._decode_state(s)['blips']
        self.assertEqual(before[0]['blipId'], after[1]['blipId'])
        self.assertNotEqual(before[0]['pos'], after[1]['pos'])
        self.assertEqual(before[1]['blipId'], after[0]['blipId'])

    def test_same_slot_new_generation_and_coincident_blips_stay_distinct(self):
        s = bridge.MlState()
        s.numBlips = 2
        s.blips[0].blipId = (10 << 16) | 4
        s.blips[1].blipId = (10 << 16) | 5
        before = bridge.ReVCInstance._decode_state(s)['blips']
        self.assertEqual(before[0]['pos'], before[1]['pos'])
        self.assertNotEqual(before[0]['blipId'], before[1]['blipId'])
        s.blips[0].blipId = (11 << 16) | 4
        after = bridge.ReVCInstance._decode_state(s)['blips']
        self.assertNotEqual(before[0]['blipId'], after[0]['blipId'])
        self.assertEqual(before[0]['blipId'] & 0xffff, after[0]['blipId'] & 0xffff)

    def test_full_blip_array_preserves_unsigned_handles_and_following_fields(self):
        s = bridge.MlState()
        s.numBlips = 32
        for i in range(32):
            s.blips[i].blipId = (0xfffe << 16) | i
        s.lastActionSeq = 1234
        s.episodeId = 19
        st = bridge.ReVCInstance._decode_state(s)
        self.assertEqual(len(st['blips']), 32)
        self.assertEqual(st['blips'][-1]['blipId'], 0xfffe001f)
        self.assertGreater(st['blips'][-1]['blipId'], 0x7fffffff)
        self.assertEqual(st['lastActionSeq'], 1234)
        self.assertEqual(st['episodeId'], 19)

    def test_invalid_mission_never_exposes_stale_globals_or_objective_identity(self):
        s = mission_state()
        s.missionTelemetryValid = 0
        s.bombState[0] = 1
        s.targetCompleteMask = 15
        s.timerState, s.missionTimerMs = 2, 123456
        s.numBlips = 1
        s.blips[0].objectiveKind, s.blips[0].objectiveIndex = 2, 3
        st = bridge.ReVCInstance._decode_state(s)
        self.assertEqual(st['mission'], dict(valid=False, id=-1, epoch=7,
                                            bomb_states=[-1] * 4, completed_mask=0,
                                            timer_state=-1, remaining_ms=-1))
        self.assertTrue(all(not o['valid'] and o['pos'] == (0., 0., 0.) for o in st['objectives']))
        self.assertEqual((st['blips'][0]['objectiveKind'], st['blips'][0]['objectiveIndex']), (0, -1))

    def test_valid_mission_preserves_carry_drop_completion_and_game_timer(self):
        s = mission_state()
        s.bombState[:] = (1, 2, 3, 0)
        s.targetCompleteMask = 8  # bomb 2 delivered to target 3, not target 2
        s.timerState, s.missionTimerMs = 2, 419900
        st = bridge.ReVCInstance._decode_state(s)
        self.assertEqual(st['mission'], dict(valid=True, id=19, epoch=7,
                                            bomb_states=[1, 2, 3, 0], completed_mask=8,
                                            timer_state=2, remaining_ms=419900))
        self.assertEqual(st['objectives'][7], dict(kind=2, index=3, valid=True, pos=(17., 27., 37.)))
        s.bombState[:] = (3, 3, 3, 3)
        s.targetCompleteMask = 15
        self.assertEqual(bridge.ReVCInstance._decode_state(s)['mission']['completed_mask'], 15)

    def test_invalid_objective_coordinates_are_zero_and_missing_blips_keep_objectives(self):
        s = mission_state()
        s.objectives[2].valid = 0
        s.objectives[2].x = float('nan')
        st = bridge.ReVCInstance._decode_state(s)
        self.assertEqual(st['blips'], [])
        self.assertEqual(len(st['objectives']), 8)
        self.assertEqual(st['objectives'][2]['pos'], (0., 0., 0.))
        self.assertTrue(st['objectives'][4]['valid'])

    def test_semantic_target_identity_survives_native_blip_recreation(self):
        s = mission_state()
        s.numBlips = 1
        s.blips[0].blipId = (7 << 16) | 12
        s.blips[0].objectiveKind, s.blips[0].objectiveIndex = 2, 1
        before = bridge.ReVCInstance._decode_state(s)['blips'][0]
        s.blips[0].blipId = (8 << 16) | 20
        after = bridge.ReVCInstance._decode_state(s)['blips'][0]
        self.assertNotEqual(before['blipId'], after['blipId'])
        self.assertEqual((before['objectiveKind'], before['objectiveIndex']),
                         (after['objectiveKind'], after['objectiveIndex']))

    def test_corrupt_valid_mission_is_rejected_instead_of_used_as_transition(self):
        changes = [dict(missionId=18), dict(missionEpoch=0), dict(targetCompleteMask=16),
                   dict(timerState=3), dict(missionTimerMs=1),
                   dict(timerState=2, missionTimerMs=420001), dict(missionTelemetryValid=2)]
        for change in changes:
            with self.subTest(change=change):
                s = mission_state()
                for key, value in change.items():
                    setattr(s, key, value)
                with self.assertRaisesRegex(RuntimeError, 'invalid mission'):
                    bridge.ReVCInstance._decode_state(s)
        for states in ((1, 1, 0, 0), (4, 0, 0, 0), (-1, 0, 0, 0)):
            s = mission_state()
            s.bombState[:] = states
            with self.assertRaisesRegex(RuntimeError, 'invalid mission'):
                bridge.ReVCInstance._decode_state(s)

    def test_valid_mission_rejects_bad_objective_and_blip_schemas(self):
        for field, value in (('kind', 0), ('index', 9), ('valid', 2), ('x', float('inf'))):
            s = mission_state()
            setattr(s.objectives[0], field, value)
            with self.assertRaisesRegex(RuntimeError, 'invalid mission objective'):
                bridge.ReVCInstance._decode_state(s)
        for kind, index in ((0, 0), (1, -1), (2, 4), (3, 0)):
            s = mission_state()
            s.numBlips = 1
            s.blips[0].objectiveKind, s.blips[0].objectiveIndex = kind, index
            with self.assertRaisesRegex(RuntimeError, 'invalid blip objective'):
                bridge.ReVCInstance._decode_state(s)


class TaskTests(unittest.TestCase):
    def test_task_defaults_to_lockstep_and_uses_square_without_dpad(self):
        task = VcDemolitionTask(auto=False, verbose=False)
        task.game = Mock()
        task.game.step.return_value = {'frame': 12}
        task.step(throttle=True, brake=True, brake_val=.2)
        kwargs = task.game.step.call_args.kwargs
        self.assertEqual(kwargs['frames'], 10)
        self.assertEqual(kwargs['buttons'], {'Cross', 'Square'})
        self.assertEqual(kwargs['brake_val'], .2)
        self.assertNotIn('dpad', kwargs)
        task.game.hold.assert_not_called()

    def test_game_death_never_returns_reset_state(self):
        task = VcDemolitionTask(auto=False, verbose=False)
        task.game = Mock()
        task.game.step.side_effect = bridge.GameDeadError('dead')
        task.hard_reset = Mock()
        with self.assertRaises(bridge.GameDeadError):
            task.step()
        task.hard_reset.assert_not_called()
        with self.assertRaises(bridge.GameDeadError):
            task.step(auto_recover=True)
        task.hard_reset.assert_called_once()

    def test_false_legacy_frame_wait_is_not_a_step(self):
        task = VcDemolitionTask(auto=False, verbose=False)
        task.game = Mock()
        task.game.wait_frames.return_value = False
        with self.assertRaises(TimeoutError):
            task.step(lockstep=False)

    def test_explicit_attach_failure_never_kills_or_discovers_other_game(self):
        task = VcDemolitionTask(auto=False, verbose=False)
        task._attach = Mock(side_effect=RuntimeError('failed attach'))
        with patch('vc_task.discover') as discover, patch('vc_task.subprocess.run') as run:
            with self.assertRaises(RuntimeError):
                task.ensure_game(pid=1234)
            discover.assert_not_called()
            run.assert_not_called()

    def test_attach_retries_protocol_timeout(self):
        task = VcDemolitionTask(auto=False, verbose=False)
        expected = object()
        with patch('vc_task.ReVCInstance', side_effect=[TimeoutError('not ready'), expected]) as constructor, \
                patch('vc_task._pid_alive_quiet', return_value=True), patch('vc_task.time.sleep'):
            self.assertIs(task._attach(1234), expected)
            self.assertEqual(constructor.call_count, 2)

    def test_hard_reset_refuses_another_controller_before_kill(self):
        task = VcDemolitionTask(auto=False, verbose=False)
        task.game = Mock()
        task.game.alive.return_value = True
        task.game._claim.side_effect = bridge.ControllerBusyError('owned')
        with patch('vc_task.subprocess.run') as run:
            with self.assertRaises(bridge.ControllerBusyError):
                task.hard_reset()
            run.assert_not_called()

    def test_active_mission_reset_never_issues_force_switch(self):
        task = VcDemolitionTask(auto=False, verbose=False)
        task.game = Mock()
        task.game.get_state.return_value = {'missionRunning': 1}
        task.hard_reset = Mock(return_value=True)
        self.assertTrue(task.reset_force())
        task.game.send_mission.assert_not_called()
        task.hard_reset.assert_called_once_with(reset_mode='hard_active_mission')

    def test_busy_soft_reset_uses_clean_process_instead_of_force_switch(self):
        task = VcDemolitionTask(auto=False, verbose=False)
        task.game = Mock()
        task.game.get_state.return_value = {'missionRunning': 0, 'heli': None, 'episodeId': 1}
        task.game.send_mission.return_value = -1
        task.hard_reset = Mock(return_value=True)
        self.assertTrue(task.reset_force())
        task.game.send_mission.assert_called_once_with(19, wait_ack=2.0)
        task.hard_reset.assert_called_once_with(reset_mode='hard_busy_mission_cleanup')


if __name__ == '__main__':
    unittest.main()
