"""Synthetic exact swept-AABB geometry; never launches or attaches to a game."""
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navigation import plan_route


def intersects(a, b, box, radius):
    low, high = 0., 1.
    for index, (minimum, maximum) in enumerate(box):
        minimum, maximum = minimum - radius, maximum + radius
        delta = b[index] - a[index]
        if abs(delta) < 1e-12:
            if a[index] < minimum or a[index] > maximum:
                return False
        else:
            enter, leave = sorted(((minimum - a[index]) / delta, (maximum - a[index]) / delta))
            low, high = max(low, enter), min(high, leave)
            if low > high:
                return False
    return True


class World:
    def __init__(self, boxes=(), radius=.15):
        self.boxes, self.radius, self.calls = boxes, radius, []

    def query(self, start, end, radius=None):
        radius = self.radius if radius is None else radius
        self.calls.append((tuple(start), tuple(end), radius))
        return dict(valid=True, blocked=any(intersects(start, end, box, radius) for box in self.boxes),
                    radius=radius, frame=42, episode_id=7)


class NavigationTests(unittest.TestCase):
    bounds = ((0, 10), (0, 10), (0, 6))

    def plan(self, world, start=(1, 1, 2), goal=(9, 1, 2), **kwargs):
        return plan_route(world.query, start, goal, bounds=self.bounds, resolutions=(2, 1), **kwargs)

    def assert_clear_path(self, result, world, start, goal):
        self.assertEqual(result['status'], 'found', result.get('failure', result['status']))
        self.assertEqual(result['path'][0], list(start))
        self.assertEqual(result['path'][-1], list(goal))
        for a, b in zip(result['path'], result['path'][1:]):
            self.assertFalse(any(intersects(a, b, box, world.radius) for box in world.boxes))
        json.dumps(result, allow_nan=False)

    def test_direct_path_uses_real_goal_and_native_clearance(self):
        world = World()
        start, goal = (1.3, 1.7, 2.1), (8.7, 2.3, 3.2)
        result = self.plan(world, start, goal)
        self.assert_clear_path(result, world, start, goal)
        self.assertEqual(len(result['path']), 2)
        self.assertEqual(result['queries'], 3)

    def test_wall_requires_detour(self):
        world = World([((4, 6), (0, 6), (-1, 7))])
        result = self.plan(world)
        self.assert_clear_path(result, world, (1, 1, 2), (9, 1, 2))
        self.assertGreater(max(p[1] for p in result['path']), 6)

    def test_refinement_discovers_narrow_door_missed_by_coarse_grid(self):
        # Only doorway y in (4.5,5.5), z in (1.5,2.5). A 2m grid has no y=5.
        world = World([((4.8, 5.2), (-1, 4.5), (-1, 7)),
                       ((4.8, 5.2), (5.5, 11), (-1, 7)),
                       ((4.8, 5.2), (4.5, 5.5), (-1, 1.5)),
                       ((4.8, 5.2), (4.5, 5.5), (2.5, 7))], radius=.1)
        result = self.plan(world, max_queries=15000, visibility_detour=False)
        self.assert_clear_path(result, world, (1, 1, 2), (9, 1, 2))
        self.assertEqual(result['used_resolution'], 1)
        self.assertEqual(result['attempts'][0]['status'], 'grid_exhausted')

    def test_floor_requires_reaching_opening_then_changing_level(self):
        # Floor opening is the remote corner x>6,y>6.
        world = World([((-1, 6), (-1, 11), (2.8, 3.2)),
                       ((6, 11), (-1, 6), (2.8, 3.2))])
        start, goal = (1, 1, 1), (1, 1, 5)
        result = self.plan(world, start, goal)
        self.assert_clear_path(result, world, start, goal)
        self.assertTrue(any(p[0] > 6 and p[1] > 6 for p in result['path']))

    def test_clearance_closes_door_that_center_ray_would_pass(self):
        walls = [((4.8, 5.2), (-1, 4.5), (-1, 7)), ((4.8, 5.2), (5.5, 11), (-1, 7))]
        world = World(walls, radius=.6)
        result = self.plan(world, (1, 5, 2), (9, 5, 2))
        self.assertEqual(result['status'], 'no_route_at_resolutions')
        self.assertEqual(result['path'], [])

    def test_blocked_start_and_goal_are_distinct(self):
        start_wall = World([((0, 2), (0, 2), (0, 3))])
        self.assertEqual(self.plan(start_wall)['status'], 'start_blocked')
        goal_wall = World([((8, 10), (0, 2), (0, 3))])
        self.assertEqual(self.plan(goal_wall)['status'], 'goal_blocked')

    def test_query_budget_cannot_return_unverified_partial_route(self):
        world = World([((4, 6), (0, 6), (-1, 7))])
        result = self.plan(world, max_queries=4)
        self.assertEqual(result['status'], 'query_budget_exhausted')
        self.assertEqual(result['queries'], 4)
        self.assertEqual(result['path'], [])

    def test_node_limit_is_not_claim_of_no_route(self):
        world = World([((4, 6), (0, 6), (-1, 7))])
        result = self.plan(world, max_nodes=1, visibility_detour=False)
        self.assertEqual(result['status'], 'search_limit')
        self.assertEqual(len(result['attempts']), 2)

    def test_unknown_query_is_not_free_space(self):
        for answer in (None, {}, {'valid': False, 'blocked': False}, {'valid': True, 'blocked': 0}):
            result = plan_route(lambda *a, **k: answer, (1, 1, 2), (9, 1, 2), bounds=self.bounds)
            self.assertEqual(result['status'], 'query_invalid')
            self.assertEqual(result['path'], [])

    def test_frame_change_invalidates_plan(self):
        world = World()
        def changing(*args, **kwargs):
            result = world.query(*args, **kwargs)
            result['frame'] += len(world.calls)
            return result
        result = plan_route(changing, (1, 1, 2), (9, 1, 2), bounds=self.bounds)
        self.assertEqual(result['status'], 'snapshot_changed')

    def test_too_small_radius_or_query_exception_fails_closed(self):
        world = World(radius=.1)
        def too_small(*args, **kwargs):
            return world.query(*args, radius=.1)
        result = plan_route(too_small, (1, 1, 2), (9, 1, 2), bounds=self.bounds, radius=.5)
        self.assertEqual(result['status'], 'query_invalid')
        def fail(*args, **kwargs):
            raise TimeoutError('missing acknowledgement')
        result = plan_route(fail, (1, 1, 2), (9, 1, 2), bounds=self.bounds)
        self.assertEqual(result['status'], 'query_error')

    def test_long_edges_are_split_for_native_limit(self):
        world = World()
        result = plan_route(world.query, (0, 0, 1), (450, 0, 1),
                            bounds=((-1, 451), (-1, 1), (0, 2)))
        self.assertEqual(result['status'], 'found')
        self.assertTrue(all(math.dist(a, b) <= 200 for a, b, _ in world.calls))

    def test_input_validation_does_not_query_game(self):
        world = World()
        with self.assertRaises(ValueError):
            self.plan(world, goal=(100, 0, 0))
        with self.assertRaises(ValueError):
            self.plan(world, max_queries=0)
        with self.assertRaises(ValueError):
            self.plan(world, radius=float('nan'))
        self.assertEqual(world.calls, [])

    def test_progress_callback_reports_actual_query_count(self):
        world, progress = World(), []
        result = self.plan(world, progress_callback=progress.append, progress_every=2)
        self.assertEqual(result['status'], 'found')
        self.assertEqual([row['queries'] for row in progress], [2])
        self.assertEqual(progress[0]['frame'], 42)

    def test_unavailable_region_is_skipped_but_never_used_in_route(self):
        world = World([((4, 6), (0, 6), (-1, 7))])
        def query(*args, **kwargs):
            result = world.query(*args, **kwargs)
            if result['blocked']:
                result.update(status=-3, valid=False)
            return result
        result = plan_route(query, (1, 1, 2), (9, 1, 2), bounds=self.bounds, resolutions=(2, 1))
        self.assert_clear_path(result, world, (1, 1, 2), (9, 1, 2))
        self.assertGreater(result['unknown_edge_count'], 0)
        self.assertEqual(result['unknown_edge_count'], len(result['unknown_edges']))
        self.assertTrue(all(row['status'] == -3 for row in result['unknown_edges']))

    def test_no_route_with_unavailable_geometry_is_not_physical_no_route(self):
        world = World([((4, 6), (-1, 11), (-1, 7))])
        def query(*args, **kwargs):
            result = world.query(*args, **kwargs)
            if result['blocked']:
                result.update(status=-3, valid=False)
            return result
        result = plan_route(query, (1, 1, 2), (9, 1, 2), bounds=self.bounds, resolutions=(2, 1))
        self.assertEqual(result['status'], 'no_route_with_unavailable_geometry')
        self.assertEqual(result['path'], [])

    def test_unknown_exception_must_still_be_structured_and_identified(self):
        for answer in (dict(valid=False, blocked=True, status=-4, frame=42, episode_id=7, radius=.1),
                       dict(valid=False, blocked=False, status=-3, frame=42, episode_id=7, radius=.1),
                       dict(valid=False, blocked=True, status=-3)):
            result = plan_route(lambda *a, **k: answer, (1, 1, 2), (9, 1, 2), bounds=self.bounds)
            self.assertEqual(result['status'], 'query_invalid')
        unavailable = dict(valid=False, blocked=True, status=-3, frame=42, episode_id=7, radius=.1)
        result = plan_route(lambda *a, **k: unavailable, (1, 1, 2), (9, 1, 2), bounds=self.bounds)
        self.assertEqual(result['status'], 'start_geometry_unavailable')

    def test_coarse_budget_reserves_opportunity_for_all_finer_levels(self):
        world = World([((4, 6), (-1, 11), (-1, 7))])
        progress = []
        result = plan_route(world.query, (1, 1, 2), (9, 1, 2), bounds=self.bounds,
                            resolutions=(2, 1, .5), max_queries=100,
                            progress_callback=progress.append, progress_every=10,
                            visibility_detour=False)
        self.assertEqual(result['status'], 'query_budget_exhausted')
        self.assertEqual([item['resolution'] for item in result['attempts']], [2, 1, .5])
        self.assertEqual(result['attempts'][0]['status'], 'resolution_query_limit')
        self.assertLessEqual(result['queries'], 100)
        self.assertTrue(all(item['queries'] > 0 for item in result['attempts']))
        self.assertEqual({row['resolution'] for row in progress}, {2, 1, .5})

    def test_transport_error_preserves_partial_attempt_statistics(self):
        world = World([((4, 6), (0, 6), (-1, 7))])
        def failing(*args, **kwargs):
            if len(world.calls) >= 20:
                raise TimeoutError('transport lost')
            return world.query(*args, **kwargs)
        result = plan_route(failing, (1, 1, 2), (9, 1, 2), bounds=self.bounds, resolutions=(2, 1),
                            visibility_detour=False)
        self.assertEqual(result['status'], 'query_error')
        self.assertEqual(len(result['attempts']), 1)
        self.assertEqual(result['attempts'][0]['status'], 'query_error')
        self.assertGreater(result['expanded_nodes'], 0)
        self.assertGreater(result['attempts'][0]['queries'], 0)

    def test_visibility_detour_finds_checked_corner_without_grid_search(self):
        world = World([((4.8, 5.2), (-1, 3), (-1, 7))])
        result = self.plan(world)
        self.assert_clear_path(result, world, (1, 1, 2), (9, 1, 2))
        self.assertEqual(result['planner'], 'visibility_detour')
        self.assertIsNone(result['used_resolution'])
        self.assertEqual(result['expanded_nodes'], 0)
        self.assertEqual(len(result['attempts']), 1)
        attempt = result['attempts'][0]
        self.assertEqual(attempt['status'], 'found')
        self.assertLessEqual(attempt['queries'], min(150, (15000 - 3) // 10))
        self.assertEqual(result['queries'], len(world.calls))
        self.assertEqual(len(result['path']), 3)
        for a, b in zip(result['path'], result['path'][1:]):
            self.assertTrue(any(tuple(a) == start and tuple(b) == end for start, end, _ in world.calls))

    def test_visibility_detour_budget_preserves_grid_search(self):
        for budget in (20, 100):
            with self.subTest(budget=budget):
                world = World([((4, 6), (-1, 11), (-1, 7))])
                result = self.plan(world, max_queries=budget)
                prepass, *grid = result['attempts']
                self.assertEqual(prepass['planner'], 'visibility_detour')
                self.assertLessEqual(prepass['queries'], (budget - 3) // 10)
                self.assertEqual([item['resolution'] for item in grid], [2, 1])
                self.assertTrue(all(item['queries'] > 0 for item in grid))
                self.assertLessEqual(result['queries'], budget)
                self.assertEqual(result['path'], [])

    def test_visibility_detour_unknown_segments_never_become_a_path(self):
        world = World([((4.8, 5.2), (-1, 3), (-1, 7))])
        def unavailable_detour(start, end, **kwargs):
            answer = world.query(start, end, **kwargs)
            if max(start[1], end[1]) > 1. or max(start[2], end[2]) > 2.:
                answer.update(valid=False, blocked=True, status=-3)
            return answer
        result = plan_route(unavailable_detour, (1, 1, 2), (9, 1, 2), bounds=self.bounds,
                            max_queries=2000, resolutions=(2, 1))
        self.assertNotEqual(result['status'], 'found')
        self.assertEqual(result['path'], [])
        self.assertGreater(result['unknown_edge_count'], 0)
        self.assertEqual(result['attempts'][0]['planner'], 'visibility_detour')

    def test_visibility_detour_snapshot_change_still_invalidates_plan(self):
        world = World([((4.8, 5.2), (-1, 3), (-1, 7))])
        def changed(start, end, **kwargs):
            answer = world.query(start, end, **kwargs)
            if len(world.calls) > 3:
                answer['frame'] += 1
            return answer
        result = plan_route(changed, (1, 1, 2), (9, 1, 2), bounds=self.bounds)
        self.assertEqual(result['status'], 'snapshot_changed')
        self.assertEqual(result['path'], [])
        self.assertEqual(result['attempts'][0]['status'], 'snapshot_changed')


if __name__ == '__main__':
    unittest.main()
