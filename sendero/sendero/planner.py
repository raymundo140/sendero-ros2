"""
planner.py — global path planner
================================
Cost-aware grid search on the global costmap, followed by line-of-sight
smoothing and resampling.

Search: 8-connected, no corner cutting. Moving through a cell costs
    step_length * (1 + cost_weight * cost / 252)
so paths keep away from obstacles where there is room and still pass
through narrow gaps. INSCRIBED and LETHAL cells are never entered. UNKNOWN
cells are entered only when the active profile allows unknown space (during
exploration), at the cost of `unknown_cost`.

Two interchangeable back-ends return the same optimal path:
  * "dijkstra": SciPy's compiled Dijkstra on a sparse grid graph (fast,
    also on large maps and for unreachable goals; needs SciPy);
  * "astar":    pure-Python A* with the octile heuristic (no SciPy).
"auto" picks dijkstra when SciPy is importable.

If the start or goal cell is blocked (the robot pressed against a shelf,
a frontier goal inside the inflation band), the nearest free cell within a
search radius is used instead and the result says so.

Error codes follow nav2_msgs/action/ComputePathToPose.
"""
# Author: Raymundo Gomez

import heapq
import math
import time
from dataclasses import dataclass, field

import numpy as np

from .costmap import INSCRIBED, LETHAL, MAX_NON_OBSTACLE, UNKNOWN

try:  # optional fast back-end
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra as _scipy_dijkstra
    HAVE_SCIPY = True
except ImportError:  # pragma: no cover - depends on the installation
    HAVE_SCIPY = False

PLAN_OK = 0
PLAN_UNKNOWN = 200
PLAN_TF_ERROR = 202
START_OUTSIDE_MAP = 203
GOAL_OUTSIDE_MAP = 204
START_OCCUPIED = 205
GOAL_OCCUPIED = 206
PLAN_TIMEOUT = 207
NO_VALID_PATH = 208

PLAN_CODE_NAMES = {
    PLAN_OK: 'OK', PLAN_UNKNOWN: 'UNKNOWN', PLAN_TF_ERROR: 'TF_ERROR',
    START_OUTSIDE_MAP: 'START_OUTSIDE_MAP', GOAL_OUTSIDE_MAP: 'GOAL_OUTSIDE_MAP',
    START_OCCUPIED: 'START_OCCUPIED', GOAL_OCCUPIED: 'GOAL_OCCUPIED',
    PLAN_TIMEOUT: 'TIMEOUT', NO_VALID_PATH: 'NO_VALID_PATH',
}

_SQRT2 = math.sqrt(2.0)


@dataclass
class PlannerSettings:
    """Settings of one planning request (from the config and the profile)."""
    allow_unknown: bool = True
    unknown_cost: float = 80.0
    cost_weight: float = 3.0
    goal_search_radius: float = 0.5
    start_search_radius: float = 0.6
    timeout_s: float = 2.0
    smooth: bool = True
    smoothing_cost_tolerance: float = 10.0
    path_resolution: float = 0.05
    backend: str = 'auto'


@dataclass
class PlanResult:
    """Outcome of a planning request."""
    ok: bool
    code: int = PLAN_OK
    message: str = ''
    path: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    length: float = 0.0
    time_s: float = 0.0
    expansions: int = 0
    start_adjusted: bool = False
    goal_adjusted: bool = False
    backend: str = ''


def traversal_costs(grid, allow_unknown, unknown_cost, cost_weight):
    """Per-cell traversal multiplier (inf = not traversable).

    Args:
        grid: CostGrid.
        allow_unknown: Whether UNKNOWN cells may be entered.
        unknown_cost: Cost (0..252) assigned to UNKNOWN cells.
        cost_weight: Weight of the cell cost.

    Returns:
        Float64 array (H, W), >= 1 where traversable.
    """
    c = grid.data
    trav = 1.0 + cost_weight * np.minimum(c, MAX_NON_OBSTACLE) / 252.0
    trav[(c == INSCRIBED) | (c == LETHAL)] = np.inf
    unk = c == UNKNOWN
    if allow_unknown:
        trav[unk] = 1.0 + cost_weight * min(max(unknown_cost, 0.0), 252.0) / 252.0
    else:
        trav[unk] = np.inf
    return trav


def _nearest_free(trav, ix, iy, radius_cells):
    """Nearest traversable cell to (ix, iy) within a radius, or None."""
    h, w = trav.shape
    r = max(0, int(math.ceil(radius_cells)))
    x0, x1 = max(0, ix - r), min(w, ix + r + 1)
    y0, y1 = max(0, iy - r), min(h, iy + r + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    sub = np.isfinite(trav[y0:y1, x0:x1])
    if not np.any(sub):
        return None
    yy, xx = np.nonzero(sub)
    d2 = (xx + x0 - ix) ** 2 + (yy + y0 - iy) ** 2
    k = int(np.argmin(d2))
    if d2[k] > r * r:
        return None
    return int(xx[k] + x0), int(yy[k] + y0)


def _astar(trav, start, goal, timeout_s):
    """Pure-Python A* on the traversal array.

    Returns:
        (cells, expansions, timed_out) with cells a list of flat indices
        from start to goal, or None if no path exists.
    """
    h, w = trav.shape
    flat = trav.ravel().tolist()
    inf = math.inf
    n = h * w
    g = [inf] * n
    parent = [-1] * n
    sx, sy = start
    gx, gy = goal
    s = sy * w + sx
    t = gy * w + gx
    g[s] = 0.0
    nbrs = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, _SQRT2), (1, -1, _SQRT2), (-1, 1, _SQRT2), (-1, -1, _SQRT2))
    k2 = _SQRT2 - 2.0

    def heur(x, y):
        dx = abs(x - gx)
        dy = abs(y - gy)
        return dx + dy + k2 * (dx if dx < dy else dy)

    open_heap = [(heur(sx, sy), 0.0, s)]
    push, pop = heapq.heappush, heapq.heappop
    expansions = 0
    deadline = time.monotonic() + max(0.05, timeout_s)
    while open_heap:
        _, gc, i = pop(open_heap)
        if gc > g[i]:
            continue
        if i == t:
            break
        expansions += 1
        if (expansions & 2047) == 0 and time.monotonic() > deadline:
            return None, expansions, True
        y, x = divmod(i, w)
        ti = flat[i]
        for dx, dy, step in nbrs:
            nx = x + dx
            ny = y + dy
            if nx < 0 or ny < 0 or nx >= w or ny >= h:
                continue
            j = ny * w + nx
            tj = flat[j]
            if tj == inf:
                continue
            if dx and dy and (flat[y * w + nx] == inf or flat[ny * w + x] == inf):
                continue
            ng = gc + step * 0.5 * (ti + tj)
            if ng < g[j]:
                g[j] = ng
                parent[j] = i
                push(open_heap, (ng + heur(nx, ny), ng, j))
    if g[t] == inf:
        return None, expansions, False
    cells = [t]
    while cells[-1] != s:
        cells.append(parent[cells[-1]])
    cells.reverse()
    return cells, expansions, False


def _dijkstra(trav, start, goal):
    """SciPy Dijkstra on the 8-connected grid graph.

    Returns:
        (cells, expansions, False) like _astar().
    """
    h, w = trav.shape
    fin = np.isfinite(trav)
    idx = np.arange(h * w).reshape(h, w)
    rows, cols, wts = [], [], []
    # Each undirected edge once: right, up, up-right, up-left.
    for dx, dy, step in ((1, 0, 1.0), (0, 1, 1.0), (1, 1, _SQRT2), (-1, 1, _SQRT2)):
        ys0, ys1 = 0, h - dy
        xs0, xs1 = max(0, -dx), w - max(0, dx)
        a = (slice(ys0, ys1), slice(xs0, xs1))
        b = (slice(ys0 + dy, ys1 + dy), slice(xs0 + dx, xs1 + dx))
        ok = fin[a] & fin[b]
        if dx and dy:  # no corner cutting
            c1 = (slice(ys0, ys1), slice(xs0 + dx, xs1 + dx))
            c2 = (slice(ys0 + dy, ys1 + dy), slice(xs0, xs1))
            ok &= fin[c1] & fin[c2]
        rows.append(idx[a][ok])
        cols.append(idx[b][ok])
        wts.append(step * 0.5 * (trav[a][ok] + trav[b][ok]))
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    wts = np.concatenate(wts)
    graph = coo_matrix((wts, (rows, cols)), shape=(h * w, h * w)).tocsr()
    s = start[1] * w + start[0]
    t = goal[1] * w + goal[0]
    dist, pred = _scipy_dijkstra(graph, directed=False, indices=s,
                                 return_predecessors=True)
    if not np.isfinite(dist[t]):
        return None, int(np.count_nonzero(np.isfinite(dist))), False
    cells = [t]
    while cells[-1] != s:
        p = int(pred[cells[-1]])
        if p < 0:
            return None, 0, False
        cells.append(p)
    cells.reverse()
    return cells, int(np.count_nonzero(np.isfinite(dist))), False


def _smooth(points, grid, trav_ok, tolerance):
    """Greedy line-of-sight shortcutting.

    A shortcut replaces a stretch of the path only if every cell on the
    straight line is traversable and no costlier than the costliest cell of
    the stretch it replaces plus `tolerance`, so a smoothed path never passes
    noticeably closer to obstacles than the search chose to.

    Args:
        points: Array (N, 2) of cell centers along the raw path.
        grid: CostGrid.
        trav_ok: Boolean array (H, W) of traversable cells.
        tolerance: Allowed cost increase (0..252).

    Returns:
        Array (M, 2) of corner points.
    """
    n = len(points)
    if n < 3:
        return points
    raw_cost = grid.cost_at(points[:, 0], points[:, 1]).astype(float)
    raw_cost[raw_cost == UNKNOWN] = 0.0
    res = grid.resolution
    out = [points[0]]
    anchor = 0
    j = 2
    while j < n:
        ok = _line_ok(points[anchor], points[j], grid, trav_ok,
                      raw_cost[anchor:j + 1].max() + tolerance, res)
        if ok:
            j += 1
            continue
        anchor = j - 1
        out.append(points[anchor])
        j = anchor + 2
    out.append(points[-1])
    return np.asarray(out)


def _line_ok(a, b, grid, trav_ok, max_cost, res):
    """True if the straight segment a-b is traversable and cheap enough."""
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    m = max(2, int(math.ceil(length / (0.5 * res))) + 1)
    t = np.linspace(0.0, 1.0, m)
    xs = a[0] + t * (b[0] - a[0])
    ys = a[1] + t * (b[1] - a[1])
    ix, iy = grid.world_to_cell(xs, ys)
    if not np.all(grid.in_bounds(ix, iy)):
        return False
    if not np.all(trav_ok[iy, ix]):
        return False
    cost = grid.data[iy, ix].astype(float)
    cost[cost == UNKNOWN] = 0.0
    return bool(cost.max() <= max_cost)


def resample(points, spacing):
    """Resample a polyline at (about) fixed spacing, keeping both ends.

    Args:
        points: Array (N, 2).
        spacing: Target spacing [m].

    Returns:
        Array (M, 2).
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(points) < 2:
        return points.copy()
    seg = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
    keep = np.concatenate(([True], seg > 1e-9))
    points = points[keep]
    if len(points) < 2:
        return points.copy()
    s = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(points[:, 0]),
                                                  np.diff(points[:, 1])))))
    total = s[-1]
    n = max(2, int(math.ceil(total / max(1e-3, spacing))) + 1)
    q = np.linspace(0.0, total, n)
    return np.column_stack((np.interp(q, s, points[:, 0]),
                            np.interp(q, s, points[:, 1])))


def add_headings(points, final_yaw=None):
    """Append a heading to each point (direction of travel).

    Args:
        points: Array (N, 2).
        final_yaw: Heading of the last point; None = last segment heading.

    Returns:
        Array (N, 3).
    """
    n = len(points)
    yaw = np.zeros(n)
    if n >= 2:
        d = np.diff(points, axis=0)
        yaw[:-1] = np.arctan2(d[:, 1], d[:, 0])
        yaw[-1] = yaw[-2]
    if final_yaw is not None and n:
        yaw[-1] = final_yaw
    return np.column_stack((points, yaw))


def plan_path(grid, start, goal, goal_yaw, settings):
    """Plan a path on a costmap.

    Args:
        grid: CostGrid (global costmap).
        start: (x, y) robot position in the grid frame.
        goal: (x, y) goal position in the grid frame.
        goal_yaw: Heading at the goal [rad] or None.
        settings: PlannerSettings.

    Returns:
        PlanResult (path is (N, 3): x, y, yaw).
    """
    t0 = time.monotonic()
    trav = traversal_costs(grid, settings.allow_unknown, settings.unknown_cost,
                           settings.cost_weight)
    res = grid.resolution
    sx, sy = (int(v) for v in grid.world_to_cell(start[0], start[1]))
    gx, gy = (int(v) for v in grid.world_to_cell(goal[0], goal[1]))

    def fail(code, msg, expansions=0):
        return PlanResult(False, code, msg, time_s=time.monotonic() - t0,
                          expansions=expansions)

    if not (0 <= sx < grid.width and 0 <= sy < grid.height):
        return fail(START_OUTSIDE_MAP, 'start is outside the costmap')
    if not (0 <= gx < grid.width and 0 <= gy < grid.height):
        return fail(GOAL_OUTSIDE_MAP, 'goal is outside the costmap')

    start_adj = goal_adj = False
    if not np.isfinite(trav[sy, sx]):
        cell = _nearest_free(trav, sx, sy, settings.start_search_radius / res)
        if cell is None:
            return fail(START_OCCUPIED, 'no free cell near the start')
        sx, sy = cell
        start_adj = True
    if not np.isfinite(trav[gy, gx]):
        cell = _nearest_free(trav, gx, gy, settings.goal_search_radius / res)
        if cell is None:
            return fail(GOAL_OCCUPIED, 'goal is in collision and no free '
                        'cell lies within goal_search_radius')
        gx, gy = cell
        goal_adj = True

    backend = settings.backend
    if backend == 'auto':
        backend = 'dijkstra' if HAVE_SCIPY else 'astar'
    if backend == 'dijkstra' and not HAVE_SCIPY:
        backend = 'astar'

    if (sx, sy) == (gx, gy):
        cells = [sy * grid.width + sx]
        expansions = 0
    elif backend == 'dijkstra':
        cells, expansions, _ = _dijkstra(trav, (sx, sy), (gx, gy))
    else:
        cells, expansions, timed_out = _astar(trav, (sx, sy), (gx, gy),
                                              settings.timeout_s)
        if timed_out:
            return fail(PLAN_TIMEOUT, 'planner timed out', expansions)
    if cells is None:
        return fail(NO_VALID_PATH, 'no path to the goal', expansions)

    cy, cx = np.divmod(np.asarray(cells), grid.width)
    wx, wy = grid.cell_to_world(cx, cy)
    pts = np.column_stack((wx, wy))
    # Exact ends: the real start position and the requested goal (when the
    # goal cell itself was free).
    pts[0] = (start[0], start[1]) if not start_adj else pts[0]
    if not goal_adj:
        pts[-1] = (goal[0], goal[1])
    if settings.smooth and len(pts) > 2:
        pts = _smooth(pts, grid, np.isfinite(trav),
                      settings.smoothing_cost_tolerance)
    pts = resample(pts, settings.path_resolution)
    if start_adj:
        pts = np.vstack(([start[0], start[1]], pts))
    path = add_headings(pts, goal_yaw)
    length = float(np.sum(np.hypot(np.diff(path[:, 0]), np.diff(path[:, 1]))))
    return PlanResult(True, PLAN_OK, 'ok', path, length,
                      time.monotonic() - t0, expansions, start_adj, goal_adj,
                      backend)
