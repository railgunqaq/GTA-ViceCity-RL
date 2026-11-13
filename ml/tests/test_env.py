"""Offline tests: no process launch, shared memory or game access."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl_train import (ACTION_SCHEMA, ACTION_VECTORS, ACTIONS, OBS_SCHEMA, RELEASE_THRESHOLD,
                      PHASE_REWARD_VERSION, OBSTACLE_SCHEMA_VERSION, DAMAGE_EPSILON, PROGRESS_SCOPE,
                      DAMAGE_PENALTY_SCALE, DQN, ENV_VERSION, EpisodeProgress, N_ACT, RcHeliEnv,
                      Replay, STEP_FRAMES, train)


def state(targets=None, *, x=0., heading=0., alive=True, mission=1, success_count=7,
          episode=1, pid=100, epoch=1, ids=None, bomb_states=(0, 0, 0, 0), completed=0,
          timer_state=0, remaining=0, mission_valid=True, health=1.):
    bombs = targets if targets is not None else ((2., 0., 0.), (20., 0., 0.), (30., 0., 0.), (40., 0., 0.))
    objectives = [dict(kind=1, index=i, valid=i < len(bombs), pos=bombs[i] if i < len(bombs) else (0., 0., 0.)) for i in range(4)]
    objectives += [dict(kind=2, index=i, valid=True, pos=(100. + 10*i, 0., 20.)) for i in range(4)]
    return dict(heli=dict(pos=(x, 0., 0.), vel=(0., 0., 0.), heading=heading, speed=0., health=health,
                         rays=[1.] * 13, up=(0., 0., 1.), forward=(1., 0., 0.), turnVel=(0., 0., 0.)) if alive else None,
                blips=[dict(pos=pos, blipId=ids[i] if ids is not None else i+101, objectiveKind=1, objectiveIndex=i)
                       for i, pos in enumerate(bombs)], objectives=objectives, frame=10, pid=pid,
                missionRunning=mission, cutsceneProcessing=0, missionSuccessCount=success_count, episodeId=episode,
                mission=dict(valid=mission_valid, id=19 if mission_valid else -1, epoch=epoch,
                             bomb_states=list(bomb_states), completed_mask=completed,
                             timer_state=timer_state, remaining_ms=remaining))


class FakeTask:
    def __init__(self, initial=None):
        self.initial = initial or state()
        self.current = copy.deepcopy(self.initial)
        self.calls = []
        self.released = False

    def reset_force(self, **kwargs):
        self.current = copy.deepcopy(self.initial)

    def observe(self):
        return copy.deepcopy(self.current)

    def step(self, **kwargs):
        self.calls.append(kwargs)
        return self.observe()

    def release(self):
        self.released = True

    def hard_reset(self):
        self.reset_force()


def make_env(initial=None, require_telemetry=True):
    task = FakeTask(initial)
    env = RcHeliEnv(verbose=False, task=task, require_telemetry=require_telemetry)
    env.reset()
    env.begin_episode()
    return env, task


class EnvironmentTests(unittest.TestCase):
    def test_schema_is_explicit_and_observation_retains_base_features(self):
        env, task = make_env()
        obs = env._features(task.observe(), env._tgt, env._mission_prev)
        self.assertEqual(ENV_VERSION, 'rc-heli-v5')
        self.assertEqual(ACTION_SCHEMA, ('roll', 'pitch', 'yaw', 'collective', 'release'))
        self.assertEqual(RELEASE_THRESHOLD, .5)
        self.assertEqual(PHASE_REWARD_VERSION, 'verified-events-damage-no-carry-shaping-v3')
        self.assertEqual(OBSTACLE_SCHEMA_VERSION, 'collision-rays-v2')
        self.assertEqual(PROGRESS_SCOPE, 'pickup_only')
        self.assertEqual(len(OBS_SCHEMA), len(set(OBS_SCHEMA)))
        self.assertEqual(obs.shape, (76,))
        self.assertTrue(np.isfinite(obs).all())
        self.assertAlmostEqual(obs[0], .02)
        np.testing.assert_array_equal(obs[10:23], np.ones(13))
        self.assertEqual(obs[OBS_SCHEMA.index('up_z')], 1.)
        self.assertEqual(obs[OBS_SCHEMA.index('mission_valid')], 1.)
        self.assertEqual(obs[OBS_SCHEMA.index('stage_pickup')], 1.)

    def test_proximity_and_stall_never_create_extra_reward(self):
        env, _ = make_env()
        for _ in range(120):
            _, reward, done, info = env._finish_step(state())
            self.assertAlmostEqual(reward, -.15)
            self.assertEqual(info['reward_terms']['time_cost'], -.15)
            self.assertFalse(done)
            self.assertEqual(info['task_events'], [])
        self.assertEqual(env.proximity_events, 0)
        self.assertEqual(env.total_events, 0)

    def test_recorded_minor_damage_is_penalized_on_each_real_step(self):
        # Physical placement probe: steps 447 and 456 each lost 0.018;
        # the old >0.02 per-step threshold silently paid zero both times.
        env, _ = make_env(state(health=1.399999976158142))
        snapshots = (state(health=1.3819999694824219),
                     state(health=1.3819999694824219),
                     state(health=1.3639999628067017))
        penalties = []
        for snapshot in snapshots:
            _, reward, _, info = env._finish_step(snapshot)
            penalties.append(info['reward_terms']['collision'])
            self.assertAlmostEqual(reward, sum(info['reward_terms'].values()))
        self.assertAlmostEqual(penalties[0], -2.7, places=4)
        self.assertEqual(penalties[1], 0.)
        self.assertAlmostEqual(penalties[2], -2.7, places=4)
        self.assertAlmostEqual(info['health_loss'], .018, places=6)

    def test_damage_cost_is_proportional_not_dependent_on_partition(self):
        penalties = []
        for healths in ((.96,), (.98, .96), (.982, .964, .96)):
            env, _ = make_env(state(health=1.))
            penalties.append(sum(env._finish_step(state(health=h))[3]['reward_terms']['collision']
                                 for h in healths))
        np.testing.assert_allclose(penalties, [-.04 * DAMAGE_PENALTY_SCALE] * 3)

    def test_health_jitter_and_recovery_do_not_pay_or_trigger_damage_cost(self):
        env, _ = make_env(state(health=1.))
        for health in (1. - DAMAGE_EPSILON / 2, 1., 1.1, 1.1):
            info = env._finish_step(state(health=health))[3]
            self.assertEqual(info['reward_terms']['collision'], 0.)
            self.assertGreaterEqual(info['health_loss'], 0.)
        info = env._finish_step(state(health=1.098))[3]
        self.assertAlmostEqual(info['reward_terms']['collision'], -.3)

    def test_missing_terminal_vehicle_does_not_fabricate_health_loss(self):
        env, _ = make_env(state(health=1.4))
        info = env._finish_step(state(alive=False, mission=0, mission_valid=False))[3]
        self.assertEqual(info['health_loss'], 0.)
        self.assertEqual(info['reward_terms']['collision'], 0.)
        self.assertEqual(info['reward_terms']['failure'], -10.)

    def test_carrying_and_falling_do_not_reward_approaching_an_indoor_target(self):
        env, _ = make_env()
        env._finish_step(state(bomb_states=(1,0,0,0)))
        previous_distance = env._dist_prev
        for x, bomb in ((10., 1), (40., 1), (70., 2), (90., 2)):
            _, reward, _, info = env._finish_step(state(x=x, bomb_states=(bomb,0,0,0)))
            self.assertLess(info['dist'], previous_distance)
            self.assertEqual(info['reward_terms']['progress'], 0.)
            self.assertEqual(info['progress_scope'], 'pickup_only')
            self.assertAlmostEqual(reward, -.15)
            previous_distance = info['dist']

    def test_pickup_still_rewards_same_objective_distance_progress(self):
        env, _ = make_env()
        _, reward, _, info = env._finish_step(state(x=1.))
        self.assertEqual(info['stage'], 'pickup')
        self.assertAlmostEqual(info['reward_terms']['progress'], 1.5)
        self.assertAlmostEqual(reward, 1.35)

    def test_return_to_pickup_resets_progress_baseline_without_switch_bonus(self):
        env, _ = make_env()
        for x, bomb in ((1., 1), (10., 1), (10., 2), (10., 0)):
            info = env._finish_step(state(x=x, bomb_states=(bomb,0,0,0)))[3]
            self.assertEqual(info['reward_terms']['progress'], 0.)
        self.assertEqual(info['stage'], 'pickup')
        self.assertTrue(info['target_changed'])
        info = env._finish_step(state(x=9.))[3]
        self.assertFalse(info['target_changed'])
        self.assertAlmostEqual(info['reward_terms']['progress'], 1.5)

    def test_semantic_identity_survives_native_blip_reconstruction(self):
        env, _ = make_env()
        for handles in ((999, 998, 997, 996), (1, 2, 3, 4)):
            _, reward, _, info = env._finish_step(state(ids=handles))
            self.assertEqual(info['target_id'], (1, 1, 0))
            self.assertEqual(info['target_native_blip_ids'], [handles[0]])
            self.assertFalse(info['target_changed'])
            self.assertAlmostEqual(reward, -.15)
        no_blips = state(); no_blips['blips'] = []
        self.assertEqual(env._finish_step(no_blips)[3]['target_id'], (1, 1, 0))

    def test_same_semantic_target_refreshes_coordinates(self):
        env, _ = make_env(state(((2.49, 0., 0.),), x=1.))
        for value in (2.51, 2.49, 4.):
            obs, reward, _, info = env._finish_step(state(((value, 0., 0.),), x=1.))
            self.assertFalse(info['target_changed'])
            self.assertEqual(info['target_pos'], (value, 0., 0.))
            self.assertAlmostEqual(info['dist'], value-1.)
            self.assertAlmostEqual(obs[0], (value-1.)/100.)
            self.assertEqual(info['task_events'], [])
            self.assertAlmostEqual(reward, sum(info['reward_terms'].values()))

    def test_real_target_switch_has_no_distance_jump_reward(self):
        env, _ = make_env()
        snapshot = state(); snapshot['objectives'][0]['valid'] = False
        info = env._finish_step(snapshot)[3]
        self.assertEqual(info['target_id'], (1, 1, 1))
        self.assertEqual(info['previous_target_id'], (1, 1, 0))
        self.assertEqual(info['previous_target_dist'], 2.)
        self.assertEqual(info['dist'], 20.)
        self.assertEqual(info['reward_terms']['progress'], 0.)

    def test_pickup_is_native_edge_once_per_bomb(self):
        env, _ = make_env()
        paid = []
        for states in ((1,0,0,0), (1,0,0,0), (2,0,0,0), (0,0,0,0), (1,0,0,0)):
            obs, _, _, info = env._finish_step(state(bomb_states=states, timer_state=2, remaining=400000))
            paid.append(info['reward_terms']['pickup'])
        self.assertEqual(paid, [5.,0.,0.,0.,0.])
        self.assertEqual(info['pickup_events'], 1)
        self.assertEqual(info['stage'], 'carrying')
        self.assertEqual(info['target_id'], (1,2,0))
        self.assertEqual(obs[OBS_SCHEMA.index('pickup_rewarded_0')], 1.)
        self.assertEqual(obs[OBS_SCHEMA.index('bomb_0_carried')], 1.)
        self.assertEqual(obs[OBS_SCHEMA.index('timer_running')], 1.)
        self.assertAlmostEqual(obs[OBS_SCHEMA.index('timer_remaining_fraction')], 400000/420000)

    def test_falling_keeps_destination_and_does_not_reward_release(self):
        env, _ = make_env()
        env._finish_step(state(bomb_states=(1,0,0,0)))
        info = env._finish_step(state(bomb_states=(2,0,0,0)))[3]
        self.assertEqual(info['stage'], 'falling')
        self.assertEqual(info['target_id'], (1,2,0))
        self.assertFalse(info['target_changed'])
        self.assertEqual(info['reward_terms']['drop'], 0.)
        self.assertEqual(info['task_events'], [])

    def test_each_of_four_bombs_and_destinations_pays_once_and_final_success_stacks(self):
        env, _ = make_env()
        bombs = [0] * 4
        mask = 0
        for i in range(4):
            bombs[i] = 1
            info = env._finish_step(state(bomb_states=bombs, completed=mask))[3]
            self.assertEqual(info['reward_terms']['pickup'], 5.)
            self.assertEqual(info['target_id'], (1,2,i))
            bombs[i] = 2
            env._finish_step(state(bomb_states=bombs, completed=mask))
            bombs[i] = 3
            mask |= 1 << i
            obs, reward, done, info = env._finish_step(state(bomb_states=bombs, completed=mask,
                                                            success_count=8 if i == 3 else 7))
            self.assertEqual(info['reward_terms']['drop'], 25.)
            self.assertEqual(info['drop_events'], i+1)
            self.assertEqual(obs[OBS_SCHEMA.index('placement_rewarded_%d' % i)], 1.)
            if i < 3:
                repeated = env._finish_step(state(bomb_states=bombs, completed=mask))[3]
                self.assertEqual(repeated['reward_terms']['drop'], 0.)
            else:
                self.assertTrue(done)
                self.assertAlmostEqual(reward, 124.85)
                self.assertTrue(info['success'])
                self.assertEqual(info['stage'], 'complete')
        self.assertEqual(env.pickup_events, 4)
        self.assertEqual(env.drop_events, 4)
        self.assertEqual(env.episode_events, 9)
        with self.assertRaisesRegex(RuntimeError, 'after termination'):
            env._finish_step(state(success_count=8))

    def test_placement_reward_is_native_mask_not_radar_disappearance(self):
        env, _ = make_env()
        snapshot = state(); snapshot['blips'] = []
        info = env._finish_step(snapshot)[3]
        self.assertEqual(info['drop_events'], 0)
        self.assertEqual(info['reward_terms']['drop'], 0.)

    def test_reset_clears_event_history_and_rejects_dirty_native_state(self):
        env, task = make_env()
        env._finish_step(state(bomb_states=(1,0,0,0)))
        task.initial = state(episode=2, epoch=2, pid=200)
        obs = env.reset()
        self.assertEqual(env.target_id, (2,1,0))
        self.assertEqual(env.pickup_events, 0)
        self.assertEqual(obs[OBS_SCHEMA.index('pickup_rewarded_0')], 0.)
        self.assertEqual(env._finish_step(state(episode=2, epoch=2, pid=200, bomb_states=(1,0,0,0)))[3]['reward_terms']['pickup'], 5.)
        for dirty in (state(bomb_states=(1,0,0,0)), state(completed=1), state(timer_state=1, remaining=420000)):
            with self.subTest(mission=dirty['mission']), self.assertRaisesRegex(RuntimeError, 'stale'):
                make_env(dirty)

    def test_failure_cleanup_invalid_mission_never_pays_task_events(self):
        env, _ = make_env()
        snapshot = state((), alive=False, mission=0, mission_valid=False, bomb_states=(1,3,3,3), completed=15)
        obs, reward, done, info = env._finish_step(snapshot)
        self.assertTrue(done)
        self.assertAlmostEqual(reward, -10.15)
        self.assertTrue(info['failure'])
        self.assertFalse(info['success'])
        self.assertFalse(info['mission_valid'])
        self.assertEqual(info['task_events'], [])
        self.assertEqual(obs[OBS_SCHEMA.index('mission_valid')], 0.)
        self.assertEqual(env.total_events, 0)

    def test_success_survives_invalid_cleaned_up_mission(self):
        env, _ = make_env()
        snapshot = state((), alive=False, mission=0, success_count=8, mission_valid=False)
        snapshot.pop('objectives')
        obs, reward, done, info = env._finish_step(snapshot)
        self.assertTrue(done)
        self.assertTrue(info['success'])
        self.assertFalse(info['failure'])
        self.assertAlmostEqual(reward, 99.85)
        self.assertEqual(info['reward_terms']['drop'], 0.)
        self.assertEqual(obs[OBS_SCHEMA.index('stage_complete')], 1.)
        self.assertEqual(env.total_events, 1)

    def test_valid_same_epoch_terminal_can_report_last_completed_target(self):
        env, _ = make_env()
        env._finish_step(state(bomb_states=(3,3,3,1), completed=7))
        snapshot = state((), alive=False, mission=0, success_count=8, bomb_states=(3,3,3,3), completed=15)
        _, reward, _, info = env._finish_step(snapshot)
        self.assertAlmostEqual(reward, 124.85)
        self.assertEqual(info['reward_terms']['drop'], 25.)
        self.assertEqual(info['drop_events'], 4)

    def test_invalid_nonterminal_telemetry_is_discarded_before_reward_history_changes(self):
        bad = []
        for field, value in [('valid',False), ('id',18), ('epoch',-1), ('bomb_states',[0,0,0]),
                             ('bomb_states',[1,1,0,0]), ('bomb_states',[True,0,0,0]),
                             ('completed_mask',16), ('timer_state',-1), ('remaining_ms',420001),
                             ('remaining_ms',1)]:
            snapshot=state(); snapshot['mission'][field]=value; bad.append(snapshot)
        missing=state(); missing.pop('mission'); bad.append(missing)
        for snapshot in bad:
            env, _ = make_env()
            with self.subTest(mission=snapshot.get('mission')), self.assertRaisesRegex(RuntimeError, 'mission telemetry'):
                env._finish_step(snapshot)
            self.assertEqual(env._pickup_rewarded, set())
            self.assertEqual(env._placement_rewarded, set())
            self.assertEqual(env.total_events, 0)

    def test_mask_regression_and_epoch_changes_are_not_transitions(self):
        env, _ = make_env()
        env._finish_step(state(completed=1, bomb_states=(3,0,0,0)))
        for snapshot in (state(), state(epoch=2), state(episode=2), state(pid=2), dict(state(), recovered=True)):
            with self.subTest(snapshot=snapshot), self.assertRaises(RuntimeError):
                env._finish_step(snapshot)
        self.assertEqual(env.drop_events, 1)

    def test_bad_objectives_or_attitude_discard_before_rewards(self):
        for mutate in (lambda s:s.pop('objectives'), lambda s:s['objectives'][0].update(index=9),
                       lambda s:s['objectives'][0].update(pos=(float('nan'),0,0)),
                       lambda s:s['heli'].pop('up'), lambda s:s['heli'].update(rays=[1.]*12)):
            env, _ = make_env()
            snapshot=state(bomb_states=(1,0,0,0)); mutate(snapshot)
            with self.assertRaisesRegex(RuntimeError, 'invalid'):
                env._finish_step(snapshot)
            self.assertEqual(env.pickup_events, 0)

    def test_missing_global_metadata_is_rejected(self):
        for key in ('missionSuccessCount','episodeId'):
            initial=state(); initial.pop(key)
            with self.assertRaisesRegex(RuntimeError, 'telemetry missing'):
                make_env(initial)

    def test_missing_counter_does_not_fabricate_success_for_explicit_offline_adapter(self):
        initial=state(); initial.pop('missionSuccessCount')
        env, _ = make_env(initial, require_telemetry=False)
        snapshot=state(mission=0, mission_valid=False); snapshot.pop('missionSuccessCount')
        info=env._finish_step(snapshot)[3]
        self.assertFalse(info['success_known'])
        self.assertFalse(info['success'])
        self.assertFalse(info['failure'])
        self.assertEqual(info['reason'], 'mission_ended_unknown')

    def test_five_actions_map_independently_and_release_is_prestate_gated(self):
        env, task = make_env()
        task.current=state(bomb_states=(1,0,0,0))
        info=env.step_cont([.2,-.3,.4,-.5,1.])[3]
        call=task.calls[-1]
        for key,value in dict(lx=-.2,ly=-.3,rx=.4,cross=0.,brake_val=.5).items():
            self.assertAlmostEqual(call[key],value)
        self.assertEqual(call['frames'],STEP_FRAMES)
        self.assertEqual(call['buttons'],())  # pickup during this step is too late
        self.assertTrue(info['release_requested'])
        self.assertFalse(info['release_effective'])
        for value, expected in ((.5,()),(.5001,('Circle',)),(-1.,())):
            info=env.step_cont([0,0,0,.75,value])[3]
            self.assertEqual(task.calls[-1]['buttons'],expected)
            self.assertEqual(info['release_effective'],bool(expected))
            self.assertAlmostEqual(task.calls[-1]['cross'],.75)
            self.assertEqual(task.calls[-1]['brake_val'],0.)
        task.current=state(bomb_states=(2,0,0,0)); env.step_cont([0,0,0,0,1])
        env.step_cont([0,0,0,0,1]); self.assertEqual(task.calls[-1]['buttons'],())

    def test_invalid_or_post_terminal_action_is_rejected_before_sending(self):
        env,task=make_env()
        for action in ([0]*4,[0,0,0,0,float('nan')],[1,2]):
            with self.assertRaises(ValueError): env.step_cont(action)
        self.assertFalse(task.calls)
        env._finish_step(state((),alive=False,mission=0,mission_valid=False))
        with self.assertRaisesRegex(RuntimeError,'after termination'): env.step_cont([0]*5)
        self.assertFalse(task.calls)

    def test_dqn_and_launch_vectors_have_explicit_release_axis(self):
        env,_=make_env()
        self.assertEqual(ACTION_VECTORS.shape,(10,5))
        np.testing.assert_array_equal(ACTION_VECTORS[:-1,4],np.zeros(9))
        self.assertEqual(ACTIONS[-1][0],'release')
        np.testing.assert_array_equal(ACTION_VECTORS[-1],[0,0,0,0,1])
        for _ in range(16):
            action=env.launch_cont()
            np.testing.assert_array_equal(action[:3],[0,0,0])
            self.assertGreater(action[3],0.)
            self.assertLess(action[3],.5)
            self.assertEqual(action[4],0.)

    def test_logs_are_json_serializable_with_raw_native_fields(self):
        env,_=make_env()
        snapshot=state(bomb_states=(1,0,0,0))
        snapshot['blips'][0]['blipId']=np.uint32(101)
        snapshot['blips'][0]['pos']=np.array([2.,0.,0.],dtype=np.float32)
        _,reward,_,info=env._finish_step(snapshot)
        encoded=json.loads(json.dumps(info,allow_nan=False))
        self.assertEqual(encoded['target_id'],[1,2,0])
        self.assertEqual(encoded['mission']['bomb_states'],[1,0,0,0])
        self.assertEqual(encoded['task_events'],[dict(type='pickup',bomb_index=0,reward=5.)])
        self.assertAlmostEqual(reward,sum(encoded['reward_terms'].values()))


class ProgressTests(unittest.TestCase):
    def test_new_target_progress_ignores_old_minimum(self):
        progress = EpisodeProgress()
        progress.update(dict(target_id='old', dist=1.))
        for distance in range(100, 39, -1):
            self.assertFalse(progress.update(dict(target_id='new', dist=distance)))
        self.assertLess(progress.stall, 2)

    def test_stall_is_external_truncation(self):
        progress = EpisodeProgress(stall_limit=2)
        info = dict(target_id='same', dist=1., terminated=False)
        for _ in range(3):
            self.assertFalse(progress.update(info))
        self.assertTrue(progress.update(info))
        self.assertFalse(info['terminated'])
        self.assertEqual(progress.reason, 'stall')


class DQNArtifactTests(unittest.TestCase):
    def test_old_artifact_is_rejected(self):
        agent = DQN(RcHeliEnv.N_OBS, N_ACT)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'old.npz'
            np.savez(path, *agent.params)
            with self.assertRaisesRegex(ValueError, 'incompatible environment'):
                agent.load(path, warm_start=True)

    def test_earlier_version_policies_are_rejected_after_contract_change(self):
        agent = DQN(RcHeliEnv.N_OBS, N_ACT)
        with tempfile.TemporaryDirectory() as tmp:
            for version in ('rc-heli-v2', 'rc-heli-v3', 'rc-heli-v4'):
                path = Path(tmp) / (version + '.npz')
                np.savez(path, *agent.params, env_version=np.asarray(version))
                with self.assertRaisesRegex(ValueError, 'incompatible environment'):
                    agent.load(path, warm_start=True)

    def test_dqn_rejects_mutated_action_observation_and_reward_contracts(self):
        agent = DQN(RcHeliEnv.N_OBS, N_ACT)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'policy.npz'
            agent.save(path)
            with np.load(path, allow_pickle=False) as z:
                original = {key: z[key].copy() for key in z.files}
            for key, value in (('action_schema', np.asarray(ACTION_SCHEMA[::-1])),
                               ('obs_schema', np.asarray(OBS_SCHEMA[::-1])),
                               ('release_threshold', np.asarray(.4)),
                               ('phase_reward_version', np.asarray('old-proximity')),
                               ('obstacle_schema_version', np.asarray('blind-offset-rays')),
                               ('progress_scope', np.asarray('all_stages')),
                               ('damage_penalty_scale', np.asarray(0.)),
                               ('damage_epsilon', np.asarray(.02))):
                changed = dict(original); changed[key] = value
                np.savez(path, **changed)
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'contract'):
                    agent.load(path, warm_start=True)
            changed = dict(original); changed.pop('obstacle_schema_version')
            np.savez(path, **changed)
            with self.assertRaisesRegex(ValueError, 'contract'):
                agent.load(path, warm_start=True)

    def test_warm_start_copies_policy_and_resets_adam(self):
        agent = DQN(RcHeliEnv.N_OBS, N_ACT, seed=4)
        other = DQN(RcHeliEnv.N_OBS, N_ACT, seed=99)
        other.t = 100
        other.m[0][:] = 10.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'policy.npz'
            agent.save(path)
            with np.load(path) as z:
                self.assertEqual(str(z['env_version']), ENV_VERSION)
            with self.assertRaisesRegex(ValueError, 'explicit warm_start'):
                other.load(path)
            other.load(path, warm_start=True)
        self.assertEqual(other.t, 0)
        self.assertTrue(all(not p.any() for p in other.m + other.v))
        for expected, actual in zip(agent.params, other.params):
            np.testing.assert_array_equal(expected, actual)

    def test_resume_rejected_before_constructing_environment(self):
        with patch('rl_train.RcHeliEnv') as env_class:
            with self.assertRaisesRegex(ValueError, 'unsupported'):
                train(resume=True)
            env_class.assert_not_called()

    def test_invalid_budget_rejected_before_constructing_environment(self):
        with patch('rl_train.RcHeliEnv') as env_class:
            for minutes in (0, -1, float('nan'), float('inf')):
                with self.subTest(minutes=minutes), self.assertRaises(ValueError):
                    train(minutes=minutes)
            for episodes in (0, -1, 1.5):
                with self.subTest(episodes=episodes), self.assertRaises(ValueError):
                    train(episodes=episodes)
            env_class.assert_not_called()


class DQNLoopTests(unittest.TestCase):
    def exercise_loop(self, mode):
        original_env = RcHeliEnv
        task = FakeTask()
        captured = []

        class TestEnv(original_env):
            def __init__(self, **kwargs):
                super().__init__(verbose=False, task=task)

            def step(self, action):
                if mode == 'error':
                    raise RuntimeError('simulated missing action acknowledgement')
                if mode == 'failure':
                    task.current = state((), mission=0, alive=False)
                return super().step(action)

        class RecordingReplay(Replay):
            def add(self, *transition):
                captured.append(transition)
                return super().add(*transition)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'run'
            with patch('rl_train.RcHeliEnv', TestEnv), \
                    patch('rl_train.Replay', RecordingReplay), \
                    patch('rl_train.MAX_STEPS', 2):
                train(episodes=1, run_dir=output)
            with (output / 'episodes.csv').open(newline='', encoding='utf-8') as f:
                row = next(csv.DictReader(f))
            self.assertTrue((output / 'policy.npz').is_file())
        self.assertTrue(task.released)
        return row, captured

    def test_step_limit_bootstraps_and_releases(self):
        row, captured = self.exercise_loop('limit')
        self.assertEqual(row['reason'], 'max_steps')
        self.assertEqual(row['truncated'], 'True')
        self.assertEqual([t[-1] for t in captured], [0., 0.])

    def test_failure_masks_bootstrap_and_releases(self):
        row, captured = self.exercise_loop('failure')
        self.assertEqual(row['reason'], 'failure')
        self.assertEqual(row['terminated'], 'True')
        self.assertEqual([t[-1] for t in captured], [1.])

    def test_failed_step_never_enters_replay_and_releases(self):
        row, captured = self.exercise_loop('error')
        self.assertEqual(row['reason'], 'bridge_error')
        self.assertEqual(row['truncated'], 'True')
        self.assertEqual(captured, [])


if __name__ == '__main__':
    unittest.main()
