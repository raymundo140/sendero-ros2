"""
costmap.py — cost grids, inflation and collision checking
=========================================================
Cost values follow Nav2 (costmap_2d) so tools and other frameworks can read
the published costmaps:

    0          free
    1 .. 252   inflated cost, decaying with distance to the nearest obstacle
    253        INSCRIBED: the robot center here means a collision
    254        LETHAL: an obstacle cell
    255        UNKNOWN: no information

Two costmaps are built from the same pieces:

* the global costmap — the SLAM occupancy grid (static layer) plus recently
  seen sensor obstacles (obstacle layer), inflated; used by the planner;
* the local costmap — a rolling window around the robot holding recent
  sensor obstacles (and, optionally, the static map), inflated; used by the
  controller and the recoveries to check the footprint.

Inflation uses an exact Euclidean distance transform bounded by the
inflation radius (two separable passes), so no SciPy is required.
"""
# Author: Raymundo Gomez

import math

import numpy as np

FREE = 0
MAX_NON_OBSTACLE = 252
INSCRIBED = 253
LETHAL = 254
UNKNOWN = 255

# costmap_2d translation of costs to OccupancyGrid values (0..100, -1).
_TRANSLATION = np.zeros(256, dtype=np.int8)
_TRANSLATION[1:253] = (1 + (97 * (np.arange(1, 253) - 1)) // 251).astype(np.int8)
_TRANSLATION[INSCRIBED] = 99
_TRANSLATION[LETHAL] = 100
_TRANSLATION[UNKNOWN] = -1


def bounded_edt(mask, max_cells):
    """Exact Euclidean distance [cells] to the nearest True cell, bounded.

    Distances up to `max_cells` are exact; anything farther is inf.

    Args:
        mask: Boolean array (H, W); True marks obstacle cells.
        max_cells: Largest distance of interest [cells].

    Returns:
        Float32 array (H, W).
    """
    r = max(0, int(math.ceil(max_cells)))
    far = float(r + 1)  # "no obstacle within r" during the passes
    h, w = mask.shape
    # Pass 1: horizontal distance to the nearest obstacle in the same row.
    g = np.full((h, w), far, dtype=np.float32)
    g[mask] = 0.0
    for d in range(1, min(r, w - 1) + 1):
        sub = g[:, d:]
        np.minimum(sub, float(d), out=sub, where=mask[:, :-d])
        sub = g[:, :-d]
        np.minimum(sub, float(d), out=sub, where=mask[:, d:])
    # Pass 2: combine rows, D^2 = min over dy of g(y + dy)^2 + dy^2.
    g2 = g * g
    out = g2.copy()
    for d in range(1, min(r, h - 1) + 1):
        dd = float(d * d)
        np.minimum(out[d:], g2[:-d] + dd, out=out[d:])
        np.minimum(out[:-d], g2[d:] + dd, out=out[:-d])
    dist = np.sqrt(out)
    dist[dist > r] = np.inf
    return dist


def inflate(lethal, unknown, resolution, inscribed_radius, inflation_radius,
            cost_scaling_factor, distance_range=0.0):
    """Turn obstacle and unknown masks into a Nav2-style cost array.

    Args:
        lethal: Boolean array (H, W) of obstacle cells.
        unknown: Boolean array (H, W) of unknown cells (or None).
        resolution: Cell size [m].
        inscribed_radius: Footprint inscribed radius [m].
        inflation_radius: Distance at which the cost reaches 0 [m].
        cost_scaling_factor: Exponential decay rate of the cost [1/m].
        distance_range: Compute distances at least this far [m] (for speed
            regulation near obstacles); 0 = up to the inflation radius.

    Returns:
        (cost, dist): uint8 cost array and float32 distance to the nearest
        obstacle [m] (inf beyond the computed range).
    """
    radius = max(inflation_radius, inscribed_radius)
    max_cells = max(radius, distance_range) / resolution + 1.0
    dist = bounded_edt(lethal, max_cells) * np.float32(resolution)
    cost = np.zeros(lethal.shape, dtype=np.uint8)
    band = (dist > inscribed_radius) & (dist <= radius)
    if np.any(band):
        decay = np.exp(-cost_scaling_factor * (dist[band] - inscribed_radius))
        cost[band] = np.clip(np.round(MAX_NON_OBSTACLE * decay),
                             1, MAX_NON_OBSTACLE).astype(np.uint8)
    cost[dist <= inscribed_radius] = INSCRIBED
    cost[lethal] = LETHAL
    if unknown is not None:
        # Unknown space stays unknown unless the robot center cannot be there.
        cost[unknown & (cost < INSCRIBED)] = UNKNOWN
    return cost, dist


class CostGrid:
    """A cost array with its geometry (axis-aligned, origin at cell 0,0 corner).

    Attributes:
        data: uint8 array (H, W), row = y index, column = x index.
        dist: float32 array (H, W), distance to the nearest obstacle [m].
        resolution: Cell size [m].
        origin_x, origin_y: World position of the corner of cell (0, 0).
        frame: Frame the grid is expressed in.
    """

    def __init__(self, data, dist, resolution, origin_x, origin_y, frame=''):
        """Store the arrays and geometry."""
        self.data = data
        self.dist = dist
        self.resolution = float(resolution)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.frame = frame
        self.height, self.width = data.shape

    # ── indexing ─────────────────────────────────────────────────────────
    def world_to_cell(self, x, y):
        """Return integer cell indices (ix, iy) of world points."""
        ix = np.floor((np.asarray(x) - self.origin_x) / self.resolution)
        iy = np.floor((np.asarray(y) - self.origin_y) / self.resolution)
        return ix.astype(np.int64), iy.astype(np.int64)

    def cell_to_world(self, ix, iy):
        """Return the world position of cell centers."""
        return (self.origin_x + (np.asarray(ix) + 0.5) * self.resolution,
                self.origin_y + (np.asarray(iy) + 0.5) * self.resolution)

    def in_bounds(self, ix, iy):
        """Boolean mask of cells inside the grid."""
        return (ix >= 0) & (ix < self.width) & (iy >= 0) & (iy < self.height)

    def contains_point(self, x, y):
        """True if the world point lies on the grid."""
        ix, iy = self.world_to_cell(x, y)
        return bool(self.in_bounds(ix, iy))

    def cost_at(self, x, y):
        """Cost of world points; points off the grid are UNKNOWN."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        ix, iy = self.world_to_cell(x, y)
        ok = self.in_bounds(ix, iy)
        out = np.full(x.shape, UNKNOWN, dtype=np.uint8)
        out[ok] = self.data[iy[ok], ix[ok]]
        return out

    def distance_at(self, x, y, default=np.inf):
        """Distance to the nearest obstacle at world points [m]."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        ix, iy = self.world_to_cell(x, y)
        ok = self.in_bounds(ix, iy)
        out = np.full(x.shape, default, dtype=float)
        out[ok] = self.dist[iy[ok], ix[ok]]
        return out

    # ── collision checking ───────────────────────────────────────────────
    def footprint_collisions(self, poses, footprint, allow_unknown=True,
                             samples=None):
        """Check the footprint at several poses.

        A pose collides if the robot center is within the inscribed radius
        of an obstacle, if any footprint sample point lies on a LETHAL cell,
        or, when unknown space is not allowed, on an UNKNOWN cell.

        Args:
            poses: Array (K, 3) of (x, y, yaw) in the grid frame.
            footprint: Footprint.
            allow_unknown: False makes unknown cells count as collisions.
            samples: Optional precomputed footprint sample points (M, 2).

        Returns:
            Boolean array (K,).
        """
        poses = np.asarray(poses, dtype=float).reshape(-1, 3)
        if len(poses) == 0:
            return np.zeros(0, dtype=bool)
        center = self.cost_at(poses[:, 0], poses[:, 1])
        hit = (center == LETHAL) | (center == INSCRIBED)
        if not allow_unknown:
            hit |= center == UNKNOWN
        if footprint.is_circle:
            if not allow_unknown:
                hit |= self._sample_hits(poses, footprint, samples,
                                         unknown_only=True)
            return hit
        d = self.distance_at(poses[:, 0], poses[:, 1])
        need = ~hit & ((d <= footprint.circumscribed_radius + self.resolution)
                       | (not allow_unknown))
        if np.any(need):
            hit[need] = self._sample_hits(poses[need], footprint, samples,
                                          unknown_only=False,
                                          allow_unknown=allow_unknown)
        return hit

    def _sample_hits(self, poses, footprint, samples, unknown_only=False,
                     allow_unknown=True):
        """Test footprint sample points of each pose against the grid."""
        if samples is None:
            samples = footprint_samples(footprint, self.resolution)
        k, m = len(poses), len(samples)
        c = np.cos(poses[:, 2])[:, None]
        s = np.sin(poses[:, 2])[:, None]
        px = poses[:, 0][:, None] + c * samples[:, 0] - s * samples[:, 1]
        py = poses[:, 1][:, None] + s * samples[:, 0] + c * samples[:, 1]
        cost = self.cost_at(px.ravel(), py.ravel()).reshape(k, m)
        if unknown_only:
            return np.any(cost == UNKNOWN, axis=1)
        bad = cost == LETHAL
        if not allow_unknown:
            bad |= cost == UNKNOWN
        return np.any(bad, axis=1)

    def line_max_cost(self, x0, y0, x1, y1, treat_unknown_as=UNKNOWN):
        """Largest cost along a straight segment (sampled at half a cell)."""
        length = math.hypot(x1 - x0, y1 - y0)
        n = max(2, int(math.ceil(length / (0.5 * self.resolution))) + 1)
        t = np.linspace(0.0, 1.0, n)
        cost = self.cost_at(x0 + t * (x1 - x0), y0 + t * (y1 - y0))
        cost = np.where(cost == UNKNOWN, treat_unknown_as, cost)
        return int(cost.max())

    # ── output ───────────────────────────────────────────────────────────
    def to_occupancy_values(self):
        """Return the grid as OccupancyGrid values (row-major list)."""
        return _TRANSLATION[self.data].ravel()


def footprint_samples(footprint, spacing):
    """Return points covering the footprint outline and interior (base frame).

    Args:
        footprint: Footprint.
        spacing: Sample spacing [m] (normally the costmap resolution).

    Returns:
        Array (M, 2).
    """
    outline = footprint.outline(spacing)
    if footprint.is_circle:
        r = footprint.radius
    else:
        r = footprint.circumscribed_radius
    n = int(math.ceil(r / spacing))
    g = np.arange(-n, n + 1) * spacing
    gx, gy = np.meshgrid(g, g)
    inner = np.column_stack((gx.ravel(), gy.ravel()))
    inner = inner[footprint.contains(inner, 0.0)]
    return np.vstack((outline, inner))


def classify_occupancy(values, occupied_threshold, free_threshold):
    """Split OccupancyGrid values into obstacle and unknown masks.

    Args:
        values: int array (H, W) of OccupancyGrid values (-1, 0..100).
        occupied_threshold: Values >= this are obstacles.
        free_threshold: Values <= this are free; values in between and
            negative values are unknown.

    Returns:
        (lethal, unknown) boolean arrays.
    """
    lethal = values >= occupied_threshold
    unknown = (values < 0) | ((values > free_threshold) & ~lethal)
    return lethal, unknown


def downsample_masks(lethal, unknown, factor):
    """Merge factor x factor blocks (any obstacle wins, mostly-unknown wins).

    Args:
        lethal: Boolean array (H, W).
        unknown: Boolean array (H, W).
        factor: Integer block size (1 = unchanged).

    Returns:
        (lethal, unknown) of shape (ceil(H/f), ceil(W/f)).
    """
    if factor <= 1:
        return lethal, unknown
    h, w = lethal.shape
    hh, ww = -(-h // factor), -(-w // factor)
    pad_l = np.zeros((hh * factor, ww * factor), dtype=bool)
    pad_u = np.ones((hh * factor, ww * factor), dtype=bool)
    pad_l[:h, :w] = lethal
    pad_u[:h, :w] = unknown
    bl = pad_l.reshape(hh, factor, ww, factor).any(axis=(1, 3))
    frac_u = pad_u.reshape(hh, factor, ww, factor).mean(axis=(1, 3))
    bu = (frac_u > 0.5) & ~bl
    return bl, bu


class StaticMap:
    """The latest SLAM occupancy grid, kept as obstacle / unknown masks."""

    def __init__(self, values, resolution, origin_x, origin_y, frame,
                 occupied_threshold, free_threshold, stamp=0.0):
        """Store a map.

        Args:
            values: int array (H, W) of OccupancyGrid values.
            resolution: Cell size [m].
            origin_x, origin_y: World position of the corner of cell (0, 0).
            frame: Map frame.
            occupied_threshold, free_threshold: See classify_occupancy().
            stamp: Receive time [s].
        """
        self.resolution = float(resolution)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.frame = frame
        self.stamp = float(stamp)
        self.height, self.width = values.shape
        self.lethal, self.unknown = classify_occupancy(
            values, occupied_threshold, free_threshold)

    def window_masks(self, origin_x, origin_y, width, height, resolution):
        """Sample the map masks on another axis-aligned grid.

        Cells outside the map are unknown and free of obstacles.

        Returns:
            (lethal, unknown) boolean arrays (height, width).
        """
        xs = origin_x + (np.arange(width) + 0.5) * resolution
        ys = origin_y + (np.arange(height) + 0.5) * resolution
        ix = np.floor((xs - self.origin_x) / self.resolution).astype(np.int64)
        iy = np.floor((ys - self.origin_y) / self.resolution).astype(np.int64)
        okx = (ix >= 0) & (ix < self.width)
        oky = (iy >= 0) & (iy < self.height)
        lethal = np.zeros((height, width), dtype=bool)
        unknown = np.ones((height, width), dtype=bool)
        if np.any(okx) and np.any(oky):
            sub = np.ix_(iy[oky], ix[okx])
            dst = np.ix_(np.nonzero(oky)[0], np.nonzero(okx)[0])
            lethal[dst] = self.lethal[sub]
            unknown[dst] = self.unknown[sub]
        return lethal, unknown


def mark_points(lethal, points, origin_x, origin_y, resolution):
    """Set the cells hit by world points to True (points off the grid ignored).

    Args:
        lethal: Boolean array (H, W), modified in place.
        points: Array (N, 2) in the grid frame.
        origin_x, origin_y, resolution: Grid geometry.

    Returns:
        Number of marked points.
    """
    if points is None or len(points) == 0:
        return 0
    ix = np.floor((points[:, 0] - origin_x) / resolution).astype(np.int64)
    iy = np.floor((points[:, 1] - origin_y) / resolution).astype(np.int64)
    h, w = lethal.shape
    ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    lethal[iy[ok], ix[ok]] = True
    return int(np.count_nonzero(ok))


def build_global_costmap(static_map, obstacle_points, footprint,
                         inflation_radius, cost_scaling_factor,
                         downsample=1, clear_center=None, extra_points=None):
    """Build the planner's costmap from the map and sensor obstacles.

    Args:
        static_map: StaticMap.
        obstacle_points: Array (N, 2) in the map frame, or None.
        footprint: Footprint (gives the inscribed radius).
        inflation_radius: [m].
        cost_scaling_factor: [1/m].
        downsample: Integer factor applied to the map resolution.
        clear_center: Optional (x, y, radius): obstacle points inside this
            circle are ignored (the robot's own position).
        extra_points: Array (M, 2) of obstacle points that are always
            marked (not affected by clear_center), or None.

    Returns:
        CostGrid in the map frame.
    """
    lethal = static_map.lethal.copy()
    if obstacle_points is not None and len(obstacle_points):
        pts = obstacle_points
        if clear_center is not None:
            cx, cy, cr = clear_center
            keep = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) > cr
            pts = pts[keep]
        mark_points(lethal, pts, static_map.origin_x, static_map.origin_y,
                    static_map.resolution)
    if extra_points is not None and len(extra_points):
        mark_points(lethal, extra_points, static_map.origin_x,
                    static_map.origin_y, static_map.resolution)
    unknown = static_map.unknown & ~lethal
    lethal, unknown = downsample_masks(lethal, unknown, int(downsample))
    res = static_map.resolution * max(1, int(downsample))
    cost, dist = inflate(lethal, unknown, res, footprint.inscribed_radius,
                         inflation_radius, cost_scaling_factor)
    return CostGrid(cost, dist, res, static_map.origin_x, static_map.origin_y,
                    static_map.frame)


def build_local_costmap(center_x, center_y, size, resolution, obstacle_points,
                        footprint, inflation_radius, cost_scaling_factor,
                        static_map=None, frame='', distance_range=0.0):
    """Build a rolling-window costmap centered on the robot.

    The window is aligned with the frame axes and snapped to the resolution,
    so consecutive windows share cells. Unknown space is free here: the
    local costmap only answers "is there an obstacle".

    Args:
        center_x, center_y: Robot position in `frame`.
        size: Window side [m].
        resolution: Cell size [m].
        obstacle_points: Array (N, 2) in `frame`, or None.
        footprint: Footprint.
        inflation_radius, cost_scaling_factor: Inflation settings.
        static_map: Optional StaticMap whose obstacles are added.
        frame: Frame name.
        distance_range: See inflate().

    Returns:
        CostGrid.
    """
    n = max(4, int(math.ceil(size / resolution)))
    ox = math.floor(center_x / resolution) * resolution - 0.5 * n * resolution
    oy = math.floor(center_y / resolution) * resolution - 0.5 * n * resolution
    if static_map is not None:
        lethal, _ = static_map.window_masks(ox, oy, n, n, resolution)
    else:
        lethal = np.zeros((n, n), dtype=bool)
    mark_points(lethal, obstacle_points, ox, oy, resolution)
    cost, dist = inflate(lethal, None, resolution, footprint.inscribed_radius,
                         inflation_radius, cost_scaling_factor, distance_range)
    return CostGrid(cost, dist, resolution, ox, oy, frame)
