"""Opt-in bounded native-geometry survey and physical route diagnostic.

Only normal controls move the helicopter. Planning runs while simulation time is
frozen; every route edge is checked against loaded game collision geometry.
The optional flight uses native-object feedback and a separate PD controller.
It neither trains nor evaluates a learned task-solving policy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from calibrate_controls import _OwnedCalibrationTask, _cleanup_owned_processes
from navigation import plan_route
from ml_client import GeometryUnavailableError
from rl_train import RcHeliEnv, ENV_VERSION, OBSTACLE_SCHEMA_VERSION
from vc_task import GAME_DIR


def control(state, target, vmax=1.0):
    """Conservative velocity/altitude feedback in the native 50Hz speed units."""
    heli = state['heli']
    pos, vel = np.asarray(heli['pos']), np.asarray(heli['vel']) * 50.
    error = np.asarray(target) - pos
    desired = error[:2] * .5
    desired *= min(1., vmax / max(1e-8, np.linalg.norm(desired)))
    forward = np.asarray(heli['forward'][:2], dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.array([forward[1], -forward[0]])
    v_error = desired - vel[:2]
    roll = -float(np.clip(.55 * np.dot(v_error, right), -.25, .25))
    pitch = -float(np.clip(.55 * np.dot(v_error, forward), -.25, .25))
    collective = float(np.clip(.25 + .18 * error[2] - .12 * vel[2], -.15, .6))
    return [roll, pitch, 0., collective, 0.]


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    task = _OwnedCalibrationTask(game_dir=GAME_DIR, auto=False)
    result = dict(status='started', env_version=ENV_VERSION,
                  obstacle_schema=OBSTACLE_SCHEMA_VERSION, diagnostic_only=True,
                  teleported=False, mode='flight' if args.fly else 'survey', plans=[])
    steps = 0
    state = None
    trace = (output / 'trace.jsonl').open('w', encoding='utf-8')
    queries = (output / 'queries.jsonl').open('w', encoding='utf-8')
    try:
        task.ensure_game(task._launch())
        env = RcHeliEnv(task=task, verbose=True)
        env.reset()
        state = task.observe()
        result['initial'] = state
        health_start = state['heli']['health']

        def step(action, phase):
            nonlocal steps, state
            if steps >= args.steps:
                raise RuntimeError('Physical diagnostic step budget reached')
            before = state
            _, reward, done, info = env.step_cont(action)
            state = task.observe()
            steps += 1
            trace.write(json.dumps(dict(step=steps, phase=phase, action=action,
                                        before=before, after=state, reward=reward, info=info)) + '\n')
            trace.flush()
            if done:
                raise RuntimeError('Mission ended during navigation diagnostic')
            if health_start - state['heli']['health'] > 1e-5:
                raise RuntimeError('Physical diagnostic stopped on observed damage')
            if state['heli']['up'][2] < np.cos(np.deg2rad(15)):
                raise RuntimeError('Physical diagnostic stopped on excessive tilt')
            if steps % 100 == 0:
                print(json.dumps(dict(event='flight_progress', step=steps, phase=phase,
                                      pos=state['heli']['pos'], mission=state['mission'])), flush=True)
            return info

        def query(start, end, radius=None):
            before = task.observe()
            record = dict(start=list(start), end=list(end), radius=radius,
                          before={k: before[k] for k in ('frame','gameTimerMs','episodeId','pid')})
            try:
                try:
                    answer = task.game.query_path(tuple(start), tuple(end), radius=radius)
                except GeometryUnavailableError as exc:
                    # Only a typed native -3 is a locally unavailable map edge.
                    # It remains blocked; transport or malformed results still abort.
                    answer = exc.result
                record['result'] = answer
                after = task.observe()
                for key in ('frame', 'gameTimerMs', 'episodeId', 'pid'):
                    if before[key] != after[key]:
                        raise RuntimeError('Read-only geometry query advanced or changed the simulator')
                if before['mission'] != after['mission']:
                    raise RuntimeError('Mission state changed during geometry query')
                return answer
            except BaseException as exc:
                record['error'] = repr(exc)
                raise
            finally:
                queries.write(json.dumps(record) + '\n')
                queries.flush()

        if args.fly:
            # Spawned bomb positions vary. Use live native coordinates instead
            # of replaying inputs from a different episode or migrating a policy.
            bombs = [o for o in state['objectives'] if o['kind'] == 1 and o['valid']]
            bomb = min(bombs, key=lambda o: np.linalg.norm(np.asarray(o['pos'])[:2] - np.asarray(state['heli']['pos'])[:2]))
            result['pickup_controller'] = 'bounded_native_position_feedback'
            result['pickup_objective'] = bomb['index']
            for _ in range(350):
                if 1 in state['mission']['bomb_states']:
                    break
                bomb = next(o for o in state['objectives'] if o['kind'] == 1 and o['index'] == bomb['index'] and o['valid'])
                pickup = list(bomb['pos'])
                pickup[2] += .45
                step(control(state, pickup, vmax=.4), 'pickup_feedback')
            if 1 not in state['mission']['bomb_states']:
                raise RuntimeError('Bounded native feedback did not pick up a bomb in this episode')
            result['verified_pickup_step'] = steps

        # Planning uses a genuinely airborne start; no teleport or collision bypass.
        hover = list(state['heli']['pos'])
        hover[2] = max(14.5, hover[2] + 3.)
        for _ in range(220):
            error = np.asarray(hover) - np.asarray(state['heli']['pos'])
            if np.linalg.norm(error) < .45 and np.linalg.norm(state['heli']['vel']) * 50 < .3:
                break
            if not state['heli']['ray_hits'][8]['valid']:
                raise RuntimeError('Overhead collision geometry is unavailable')
            if state['heli']['ray_hits'][8]['hit'] and state['heli']['ray_hits'][8]['distance'] < error[2] + state['heli']['clearance_radius']:
                raise RuntimeError('Overhead obstruction prevents the initial climb')
            step(control(state, hover, vmax=.4), 'climb')
        else:
            raise RuntimeError('Initial climb did not stabilize within its budget')
        base_radius = state['heli']['clearance_radius']
        radius = base_radius + args.margin
        result['clearance_radius'] = base_radius
        result['planning_radius'] = radius
        result['survey_state'] = state
        # Revisit the previous failed approach using current game geometry.
        result['old_direct_approach'] = query((228., -343., 19.35), (175., -348., 19.35), radius)
        result['old_contact_point'] = query((210.657, -343.584, 19.537),
                                            (210.657, -343.584, 19.537), base_radius)
        result['query_time_frozen'] = True
        indices = [args.goal] if args.goal is not None else list(range(4))
        selected = None
        for index in indices:
            native = next(o for o in state['objectives'] if o['kind'] == 2 and o['index'] == index and o['valid'])
            candidates = []
            goal = None
            # The marker is near a barrel, not a safe helicopter-center pose.
            # Retain the full clearance sphere and inspect several drop heights.
            for height in (3.5, 4., 3., 2.5, 4.5):
                candidate = list(native['pos'])
                candidate[2] += height
                if candidate[2] > 40:
                    continue
                occupancy = query(candidate, candidate, radius)
                candidates.append(dict(goal=candidate, query=occupancy))
                if occupancy['valid'] and not occupancy['blocked']:
                    goal = candidate
                    break
            if goal is None:
                result['plans'].append(dict(status='goal_blocked', objective_index=index,
                                            path=[], hover_candidates=candidates))
                if args.fly:
                    raise RuntimeError('No tested hover pose has the required clearance')
                continue
            print(json.dumps(dict(event='plan_start', goal_index=index, goal=goal, radius=radius)), flush=True)
            route_points = []
            if args.via_file:
                raw = args.via_file.read_bytes()
                hints = json.loads(raw)
                route_points = hints['waypoints']
                result['via_source'] = dict(path=str(args.via_file.resolve()),
                                           sha256=hashlib.sha256(raw).hexdigest(),
                                           evidence=hints.get('evidence'))
            endpoints = route_points + [goal]
            current_start = state['heli']['pos']
            legs, combined, raw_combined, checked = [], [list(current_start)], [list(current_start)], []
            for leg_index, endpoint in enumerate(endpoints):
                remaining = args.queries - sum(leg['queries'] for leg in legs)
                if remaining <= 0:
                    break
                leg = plan_route(query, current_start, endpoint, radius=radius,
                                 max_queries=max(1, remaining // (len(endpoints) - leg_index)),
                                 max_nodes=args.nodes, resolutions=tuple(args.resolutions),
                                 progress_callback=lambda x: print(json.dumps(dict(event='plan_progress', **x)), flush=True))
                legs.append(leg)
                checked.extend(leg.get('checked_segments', []))
                if leg['status'] != 'found':
                    break
                combined.extend(leg['path'][1:])
                raw_combined.extend(leg['raw_path'][1:])
                current_start = endpoint
            successful = len(legs) == len(endpoints) and all(leg['status'] == 'found' for leg in legs)
            planned = dict(status='found' if successful else legs[-1]['status'],
                           start=list(state['heli']['pos']), goal=goal,
                           path=combined if successful else [], raw_path=raw_combined if successful else [],
                           queries=sum(leg['queries'] for leg in legs),
                           snapshot=legs[0]['snapshot'],
                           legs=[{k: v for k, v in leg.items() if k != 'checked_segments'} for leg in legs],
                           checked_segments=checked, source='native_checked_geometry_hints' if route_points else 'bounded_search')
            planned['objective_index'] = index
            planned['hover_candidates'] = candidates
            (output / ('route-%d.json' % index)).write_text(json.dumps(planned, indent=2), encoding='utf-8')
            result['plans'].append({k: v for k, v in planned.items() if k not in ('checked_segments',)})
            print(json.dumps(dict(event='plan_finished', goal_index=index, status=planned['status'],
                                  queries=planned['queries'], path=planned['path'])), flush=True)
            if args.fly:
                selected = planned
                break

        if args.fly:
            if selected is None or selected['status'] != 'found':
                raise RuntimeError('No geometry-validated route available for physical execution')
            stable = 0
            released_at = None
            release_hover = None
            path = selected['path']
            waypoint = 1
            best_distance, stagnant = float('inf'), 0
            while steps < args.steps:
                pos = np.asarray(state['heli']['pos'])
                velocity = np.asarray(state['heli']['vel']) * 50.
                if released_at is not None:
                    # Release creates the payload inside the conservative full-
                    # vehicle sphere. Do not start another navigation segment
                    # through that dynamic object: hold the already reached pose
                    # briefly while the native mission resolves the falling bomb.
                    if np.linalg.norm(pos - release_hover) > .5:
                        raise RuntimeError('Release hover drift exceeded its bound')
                    if steps - released_at >= 30:
                        raise RuntimeError('Release did not produce a placement within 3 simulated seconds')
                    info = step(control(state, release_hover, vmax=.15), 'release_observation')
                    if state['mission']['completed_mask'] & (1 << args.goal):
                        result['placement'] = dict(step=steps, mission=state['mission'], events=info['task_events'])
                        break
                    continue
                endpoint = np.asarray(path[waypoint])
                error = endpoint - pos
                if waypoint < len(path) - 1 and np.linalg.norm(error) < .5:
                    waypoint += 1
                    endpoint = np.asarray(path[waypoint])
                    error = endpoint - pos
                    best_distance, stagnant = float('inf'), 0
                if np.linalg.norm(error) < best_distance - .15:
                    best_distance, stagnant = np.linalg.norm(error), 0
                else:
                    stagnant += 1
                if stagnant > 200 and released_at is None:
                    raise RuntimeError('No waypoint progress for 20 simulated seconds')
                segment_start = np.asarray(path[waypoint - 1])
                delta = endpoint - segment_start
                length = np.linalg.norm(delta)
                sloped = abs(delta[2]) / max(length, 1e-8) > .15
                along = np.clip(np.dot(pos - segment_start, delta) / max(length * length, 1e-8), 0., 1.)
                target = segment_start + delta * min(1., along + (1.5 if sloped else 3.) / max(length, 1e-8))
                # Track a short point ON the validated 3D segment. A final-height
                # setpoint would climb prematurely into a floor while approaching stairs.
                clearance = query(pos, target, base_radius + .1)
                if not clearance['valid'] or clearance['blocked']:
                    raise RuntimeError('Current-to-waypoint clearance failed; stopped before continuing')
                stopping = query(pos, pos + velocity * 1.5, base_radius + .1)
                if not stopping['valid'] or stopping['blocked']:
                    raise RuntimeError('Projected motion lacks clearance; stopped before continuing')
                final = waypoint == len(path) - 1
                action = control(state, target, vmax=.35 if final and np.linalg.norm(error) < 4 else .5 if sloped else 1.2)
                quiet = (final and np.max(np.abs(error[:2])) < .7 and abs(error[2]) < .4
                         and np.linalg.norm(velocity[:2]) < .3 and abs(velocity[2]) < .25)
                stable = stable + 1 if quiet else 0
                if stable >= 3 and released_at is None:
                    action[4] = 1.
                    released_at = steps + 1
                    release_hover = pos.copy()
                    result['verified_arrival_step'] = steps
                info = step(action, 'route_following' if released_at is None else 'release_observation')
                if state['mission']['completed_mask'] & (1 << args.goal):
                    result['placement'] = dict(step=steps, mission=state['mission'], events=info['task_events'])
                    break
                if released_at is not None and steps - released_at > 30:
                    raise RuntimeError('Release did not produce a placement within 3 simulated seconds')
            if not result.get('placement'):
                raise RuntimeError('No verified placement before the physical step budget')
        result['status'] = 'passed' if all(p['status'] == 'found' for p in result['plans']) else 'survey_incomplete'
    except BaseException as exc:
        result.update(status='failed', error=repr(exc))
        raise
    finally:
        result.update(steps=steps, final_state=state)
        trace.close()
        queries.close()
        try:
            result['cleanup'], result['cleanup_errors'] = _cleanup_owned_processes(task)
        except BaseException as cleanup_error:
            result['cleanup_errors'] = [repr(cleanup_error)]
        if result['cleanup_errors']:
            result['status'] = 'failed'
        (output / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps({k: v for k, v in result.items() if k not in ('initial','survey_state','final_state','plans','cleanup')}, indent=2), flush=True)
        if result['cleanup_errors'] and not result.get('error'):
            raise RuntimeError('Owned process cleanup failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fly', action='store_true')
    parser.add_argument('--goal', type=int, choices=range(4))
    parser.add_argument('--queries', type=int, default=15000)
    parser.add_argument('--nodes', type=int, default=10000)
    parser.add_argument('--steps', type=int, default=2400)
    parser.add_argument('--margin', type=float, default=.35)
    parser.add_argument('--resolutions', type=float, nargs='+', default=[4., 2., 1.])
    parser.add_argument('--via-file', type=Path)
    args = parser.parse_args()
    if args.fly and args.goal is None:
        parser.error('--fly requires one --goal')
    if args.margin < 0 or min(args.queries, args.nodes, args.steps, *args.resolutions) <= 0:
        parser.error('Budgets must be positive and margin nonnegative')
    run(args)
