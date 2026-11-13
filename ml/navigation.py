"""Bounded geometry-checked routing for frozen reVC snapshots.

This is diagnostic planning, not a flight controller. Every returned segment
has passed the supplied native swept-clearance query. Searches are limited to
the supplied bounds and resolutions: failure never proves that the world has
no route. The query covers loaded static geometry, not future dynamic actors.
"""
from bisect import bisect_left
import heapq
import itertools
import math
import time


class _StopPlanning(Exception):
    def __init__(self, status, detail):
        self.status, self.detail = status, detail


def _point(value, label):
    try:
        point = tuple(float(x) for x in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('%s must contain three finite coordinates' % label) from exc
    if len(point) != 3 or not all(math.isfinite(x) for x in point):
        raise ValueError('%s must contain three finite coordinates' % label)
    return point


def _distance(a, b):
    return math.dist(a, b)


class _Queries:
    def __init__(self, callback, radius, budget, max_segment, record_queries,
                 progress_callback, progress_every):
        self.callback, self.radius = callback, radius
        self.budget, self.max_segment = budget, max_segment
        self.cache, self.records = {}, []
        self.count, self.identity = 0, None
        self.record_queries = record_queries
        self.progress_callback, self.progress_every = progress_callback, progress_every
        self.search_ceiling = self.active_resolution = None
        self.active_planner = 'initial'
        self.unknown_edges = []

    def _one(self, a, b):
        key = tuple(sorted((a, b)))
        if key in self.cache:
            return self.cache[key]
        if self.count >= self.budget:
            raise _StopPlanning('query_budget_exhausted', 'Native query budget exhausted')
        if self.search_ceiling is not None and self.count >= self.search_ceiling:
            status = 'prepass_query_limit' if self.active_planner == 'visibility_detour' else 'resolution_query_limit'
            raise _StopPlanning(status, 'Reserved query budget for this planning phase exhausted')
        self.count += 1
        try:
            answer = self.callback(a, b, radius=self.radius)
        except Exception as exc:
            raise _StopPlanning('query_error', '%s: %s' % (type(exc).__name__, exc)) from exc
        unavailable = (isinstance(answer, dict) and answer.get('status') == -3
                       and answer.get('valid') is False and answer.get('blocked') is True)
        if (not isinstance(answer, dict) or (answer.get('valid') is not True and not unavailable)
                or not isinstance(answer.get('blocked'), bool)):
            raise _StopPlanning('query_invalid', 'Query did not explicitly establish valid/blocked')
        frame, episode = answer.get('frame'), answer.get('episode_id')
        radius = answer.get('radius')
        if (isinstance(frame, bool) or not isinstance(frame, int) or frame < 0
                or isinstance(episode, bool) or not isinstance(episode, int) or episode < 0
                or isinstance(radius, bool) or not isinstance(radius, (int, float))
                or not math.isfinite(radius) or radius <= 0):
            raise _StopPlanning('query_invalid', 'Missing snapshot identity or clearance radius')
        if self.radius is not None and radius + 1e-6 < self.radius:
            raise _StopPlanning('query_invalid', 'Native clearance radius smaller than requested')
        identity = frame, episode, float(radius)
        if self.identity is None:
            self.identity = identity
        elif (identity[:2] != self.identity[:2]
              or abs(identity[2] - self.identity[2]) > 1e-5):
            raise _StopPlanning('snapshot_changed', 'Frame, episode or clearance changed during planning')
        blocked = answer['blocked']
        self.cache[key] = blocked
        if unavailable:
            # This is an explicitly classified unavailable region, not an
            # obstacle measurement or a free-space certificate. The edge is
            # unusable, while other fully validated regions may still connect.
            self.unknown_edges.append(dict(start=list(a), end=list(b), frame=frame,
                                           episode_id=episode, radius=radius, status=-3))
        if self.record_queries:
            self.records.append(dict(start=list(a), end=list(b), **answer))
        if self.progress_callback is not None and self.count % self.progress_every == 0:
            try:
                self.progress_callback(dict(queries=self.count, max_queries=self.budget,
                                            frame=frame, episode_id=episode, radius=radius,
                                            last_start=list(a), last_end=list(b), blocked=blocked,
                                            resolution=self.active_resolution,
                                            planner=self.active_planner,
                                            resolution_query_ceiling=self.search_ceiling,
                                            unknown_edge_count=len(self.unknown_edges)))
            except Exception as exc:
                raise _StopPlanning('progress_callback_error', '%s: %s' % (type(exc).__name__, exc)) from exc
        return blocked

    def blocked(self, a, b):
        # The native ABI limits segment length. All pieces must pass; none of
        # them advances the simulator or replaces a failed query with a ray.
        pieces = max(1, math.ceil(_distance(a, b) / self.max_segment))
        previous = a
        for index in range(1, pieces + 1):
            following = tuple(a[axis] + (b[axis] - a[axis]) * index / pieces for axis in range(3))
            if self._one(previous, following):
                return True
            previous = following
        return False


def _axis(lower, upper, step):
    count = int(math.floor((upper - lower) / step + 1e-9))
    points = [lower + index * step for index in range(count + 1)]
    if upper - points[-1] > 1e-8:
        points.append(upper)
    return points


def _connectors(point, axes):
    candidates = []
    for value, axis in zip(point, axes):
        upper = min(len(axis) - 1, bisect_left(axis, value))
        lower = max(0, upper - 1)
        candidates.append(sorted({lower, upper}))
    return list(itertools.product(*candidates))


def _detour_candidates(start, goal, bounds):
    """Cheap geometric hypotheses only; none is a free-space certificate."""
    dx, dy = goal[0] - start[0], goal[1] - start[1]
    horizontal = math.hypot(dx, dy)
    normals = [(-dy / horizontal, dx / horizontal)] if horizontal > 1e-9 else [(1., 0.), (0., 1.)]
    candidates = {}
    for fraction, side, rise, normal in itertools.product(
            (.25, .5, .75), (-24., -16., -8., -4., 0., 4., 8., 16., 24.),
            (0., 4., 8., 12.), normals):
        if side == 0 and rise == 0:
            continue  # a point on the already blocked direct segment adds no route
        point = (start[0] + fraction * dx + side * normal[0],
                 start[1] + fraction * dy + side * normal[1],
                 start[2] + fraction * (goal[2] - start[2]) + rise)
        if (not all(bounds[i][0] <= point[i] <= bounds[i][1] for i in range(3))
                or min(_distance(start, point), _distance(goal, point)) <= 1e-8):
            continue
        candidates.setdefault(tuple(round(value, 8) for value in point), point)
    return sorted(candidates.values(), key=lambda point: (_distance(start, point) + _distance(point, goal), point))


def _visibility_detour(start, goal, bounds, query, attempt):
    candidates = _detour_candidates(start, goal, bounds)
    before = query.count
    allowance = min(150, (query.budget - before) // 10)
    attempt.update(planner='visibility_detour', resolution=None, candidates=len(candidates),
                   candidates_tested=0, query_allowance=allowance, queries=0, expanded_nodes=0)
    if not candidates or allowance < 2:
        attempt['status'] = 'no_candidates' if not candidates else 'skipped_budget'
        return None
    previous = query.search_ceiling, query.active_resolution, query.active_planner
    query.search_ceiling, query.active_resolution, query.active_planner = before + allowance, None, 'visibility_detour'
    try:
        for point in candidates:
            attempt['candidates_tested'] += 1
            # Both edges, including any native-limit subsegments, must pass the
            # same clearance, loading and snapshot checks as the grid search.
            if not query.blocked(start, point) and not query.blocked(point, goal):
                attempt['status'] = 'found'
                return [start, point, goal]
        attempt['status'] = 'candidates_exhausted'
    except _StopPlanning as exc:
        attempt['status'] = exc.status
        if exc.status != 'prepass_query_limit':
            raise  # identity, transport and invalid telemetry remain fatal
    finally:
        attempt['queries'] = query.count - before
        query.search_ceiling, query.active_resolution, query.active_planner = previous
    return None


def _search(start, goal, bounds, resolution, query, max_nodes, connectivity):
    axes = [_axis(lo, hi, resolution) for lo, hi in bounds]
    shape = tuple(len(axis) for axis in axes)
    offsets = [(x, y, z) for x, y, z in itertools.product((-1, 0, 1), repeat=3)
               if (x or y or z) and (connectivity == 26 or abs(x) + abs(y) + abs(z) == 1)]
    start_key, goal_key = 'start', 'goal'

    def coordinates(key):
        if key == start_key:
            return start
        if key == goal_key:
            return goal
        return tuple(axes[i][key[i]] for i in range(3))

    def neighbors(key):
        if key == start_key:
            yield from _connectors(start, axes)
            return
        here = coordinates(key)
        if _distance(here, goal) <= math.sqrt(3) * resolution + 1e-8:
            yield goal_key
        for offset in offsets:
            other = tuple(key[i] + offset[i] for i in range(3))
            if all(0 <= other[i] < shape[i] for i in range(3)):
                yield other

    # Queue optimistic *edges*, not unverified node parents. An edge is checked
    # only when it could settle its child. A blocked edge cannot hide an
    # alternative parent because all competing candidates stay in the queue.
    counter = itertools.count()
    frontier, costs, parents = [], {start_key: 0.}, {}
    expanded = 0

    def enqueue(parent):
        here, parent_cost = coordinates(parent), costs[parent]
        for child in neighbors(parent):
            point = coordinates(child)
            cost = parent_cost + _distance(here, point)
            if cost + 1e-9 < costs.get(child, float('inf')):
                heapq.heappush(frontier, (cost + _distance(point, goal), next(counter),
                                         cost, parent_cost, parent, child))

    enqueue(start_key)
    try:
        while frontier:
            _, _, cost, parent_cost, parent, child = heapq.heappop(frontier)
            if parent_cost != costs[parent] or cost + 1e-9 >= costs.get(child, float('inf')):
                continue
            if query.blocked(coordinates(parent), coordinates(child)):
                continue
            costs[child], parents[child] = cost, parent
            if child == goal_key:
                path = [coordinates(child)]
                while child != start_key:
                    child = parents[child]
                    point = coordinates(child)
                    if _distance(point, path[-1]) > 1e-9:
                        path.append(point)
                return list(reversed(path)), expanded, 'found'
            expanded += 1
            if expanded >= max_nodes:
                return None, expanded, 'node_limit'
            enqueue(child)
    except _StopPlanning as exc:
        exc.expanded_nodes = expanded
        raise
    return None, expanded, 'grid_exhausted'


def _simplify(path, query):
    result, index = [path[0]], 0
    while index < len(path) - 1:
        following = index + 1
        # Greedy shortcuts are accepted only after their own clearance query.
        # Existing neighbor edges were already checked, so budget exhaustion
        # can safely retain the unshortened remainder of a found route.
        for candidate in range(len(path) - 1, index + 1, -1):
            try:
                if not query.blocked(path[index], path[candidate]):
                    following = candidate
                    break
            except _StopPlanning as exc:
                if exc.status == 'query_budget_exhausted':
                    return result + path[index + 1:], False
                raise
        result.append(path[following])
        index = following
    return result, True


def plan_route(query_path, start, goal, *, bounds=((75., 240.), (-400., -325.), (8., 40.)),
               resolutions=(4., 2., 1.), radius=None, max_queries=15000, max_nodes=10000,
               connectivity=6, max_segment=200., simplify=True, record_queries=True,
               progress_callback=None, progress_every=200, visibility_detour=True):
    """Find one route using only explicitly validated native-clear segments.

    ``query_path(a,b,radius=...)`` must return valid/blocked booleans, the queried
    nominal radius, frame, and episode_id. Its snapshot must stay frozen for the
    entire call. An explicitly classified ``status=-3, valid=False,
    blocked=True`` means unavailable geometry: that edge is unusable and is
    reported separately, while other validated edges may still form a route.
    All other invalid query results or transport errors terminate planning.
    ``max_nodes`` is a per-resolution expansion limit; native-query
    budget is shared across the optional single-waypoint detour prepass, all
    resolutions and simplification. The prepass uses at most 150 queries and 10%
    of the remaining budget; every returned edge still requires native clearance.
    A failed hypothesis falls through to the grid. Each resolution
    receives an equal share of the remaining budget, reserving later levels an
    opportunity to search; unused allowance rolls forward. Bounds describe
    admissible vehicle-center coordinates. The caller must allow sufficient
    space outside/above the building and choose a physically clear final hover
    position, rather than asking the vehicle center to occupy the bomb target.

    A returned route is a clearance-checked geometric reference, not proof of
    dynamic flyability. Re-query the intended segment during execution and stop
    on invalid telemetry, newly blocked geometry, or tracking/collision limits.
    """
    started = time.perf_counter()
    start, goal = _point(start, 'start'), _point(goal, 'goal')
    try:
        bounds = tuple(tuple(float(v) for v in axis) for axis in bounds)
        resolutions = tuple(float(value) for value in resolutions)
    except (TypeError, ValueError) as exc:
        raise ValueError('Invalid bounds/resolutions') from exc
    if (len(bounds) != 3 or any(len(axis) != 2 or not all(map(math.isfinite, axis))
                               or axis[0] >= axis[1] for axis in bounds)):
        raise ValueError('Bounds must be three finite (minimum, maximum) pairs')
    if not all(bounds[i][0] <= point[i] <= bounds[i][1] for point in (start, goal) for i in range(3)):
        raise ValueError('Start and goal must be within planning bounds')
    if not resolutions or any(not math.isfinite(value) or value <= 0 for value in resolutions):
        raise ValueError('Resolutions must be positive finite values')
    if any(resolutions[i] <= resolutions[i + 1] for i in range(len(resolutions) - 1)):
        raise ValueError('Resolutions must be strictly decreasing')
    if (not isinstance(max_queries, int) or isinstance(max_queries, bool) or max_queries < 1
            or not isinstance(max_nodes, int) or isinstance(max_nodes, bool) or max_nodes < 1):
        raise ValueError('Query/node budgets must be positive integers')
    if radius is not None and (isinstance(radius, bool) or not math.isfinite(radius) or radius <= 0):
        raise ValueError('Radius must be positive and finite, or None for native vehicle bounds')
    if connectivity not in (6, 26) or not math.isfinite(max_segment) or not 0 < max_segment <= 200:
        raise ValueError('Connectivity must be 6/26 and max_segment in (0,200]')
    if (progress_callback is not None and not callable(progress_callback)) or (
            isinstance(progress_every, bool) or not isinstance(progress_every, int) or progress_every < 1):
        raise ValueError('Progress callback must be callable and interval a positive integer')
    if not isinstance(visibility_detour, bool):
        raise ValueError('visibility_detour must be a boolean')
    query = _Queries(query_path, radius, max_queries, max_segment, record_queries,
                     progress_callback, progress_every)
    result = dict(status='not_started', path=[], raw_path=[], start=list(start), goal=list(goal),
                  bounds=[list(axis) for axis in bounds], resolutions=list(resolutions),
                  requested_radius=radius, max_queries=max_queries, max_nodes_per_resolution=max_nodes,
                  connectivity=connectivity, attempts=[], expanded_nodes=0,
                  visibility_detour=visibility_detour,
                  scope='Frozen loaded static geometry; not dynamic flight or global reachability')
    try:
        if query.blocked(start, start):
            result['status'] = 'start_geometry_unavailable' if query.unknown_edges else 'start_blocked'
        elif query.blocked(goal, goal):
            result['status'] = 'goal_geometry_unavailable' if query.unknown_edges else 'goal_blocked'
        elif not query.blocked(start, goal):
            result.update(status='found', raw_path=[start, goal], path=[start, goal], simplified=True,
                          planner='direct', used_resolution=None)
        else:
            detour = None
            if visibility_detour:
                attempt = {}
                result['attempts'].append(attempt)
                detour = _visibility_detour(start, goal, bounds, query, attempt)
            if detour is not None:
                result.update(status='found', raw_path=detour, path=detour, simplified=True,
                              planner='visibility_detour', used_resolution=None)
            else:
                any_limited = False
                for level, resolution in enumerate(resolutions):
                    before = query.count
                    levels_left = len(resolutions) - level
                    allowance = (query.budget - before) // levels_left
                    query.active_resolution = resolution
                    query.active_planner = 'grid'
                    query.search_ceiling = before + allowance
                    path, expanded, status, error = None, 0, 'not_started', None
                    try:
                        path, expanded, status = _search(start, goal, bounds, resolution, query, max_nodes, connectivity)
                    except _StopPlanning as exc:
                        expanded, status, error = getattr(exc, 'expanded_nodes', 0), exc.status, exc
                    finally:
                        result['attempts'].append(dict(planner='grid', resolution=resolution, status=status,
                                                       query_allowance=allowance, queries=query.count - before,
                                                       expanded_nodes=expanded))
                        result['expanded_nodes'] += expanded
                        query.active_resolution = query.search_ceiling = None
                    if error is not None and status != 'resolution_query_limit':
                        raise error
                    any_limited |= status in ('node_limit', 'resolution_query_limit')
                    if path is not None:
                        short, complete = _simplify(path, query) if simplify else (path, False)
                        result.update(status='found', raw_path=path, path=short, simplified=complete,
                                      used_resolution=resolution, planner='grid')
                        break
                else:
                    result['status'] = ('no_route_with_unavailable_geometry' if query.unknown_edges else
                                        'search_limit' if any_limited else 'no_route_at_resolutions')
    except _StopPlanning as exc:
        # Invalidation during search or shortcut validation invalidates the
        # complete proposed route. Never return partially checked waypoints.
        result.update(status=exc.status, failure=exc.detail, path=[], raw_path=[])
    result['path'] = [list(point) for point in result['path']]
    result['raw_path'] = [list(point) for point in result['raw_path']]
    result.update(queries=query.count, elapsed_seconds=time.perf_counter() - started,
                  unknown_edge_count=len(query.unknown_edges), unknown_edges=query.unknown_edges,
                  snapshot=None if query.identity is None else dict(frame=query.identity[0],
                      episode_id=query.identity[1], radius=query.identity[2]))
    if record_queries:
        result['checked_segments'] = query.records
    return result
