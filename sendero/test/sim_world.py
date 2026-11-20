"""
sim_world.py — a small 2D simulator for testing Sendero without ROS
====================================================================
A warehouse occupancy grid, a 360-degree LiDAR (ray marching), and a
differential-drive or omnidirectional base that integrates the commanded
velocities. The NavigationManager is driven exactly as the ROS node drives
it: scans in, tick(), command out.
"""
# Author: Raymundo Gomez

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sendero.config import NavigationConfig  # noqa: E402
from sendero.geometry import (compose, distance, integrate_velocity,  # noqa: E402
                              transform_points)
from sendero.navigator import ACTIVE, NavigationManager  # noqa: E402

RES = 0.05


class World:
    """Ground-truth occupancy grid (True = obstacle)."""

    def __init__(self, width_m, height_m, res=RES):
        """Create an empty world with walls around it."""
        self.res = res
        self.w = int(round(width_m / res))
        self.h = int(round(height_m / res))
        self.occ = np.zeros((self.h, self.w), dtype=bool)
        self.occ[0, :] = self.occ[-1, :] = True
        self.occ[:, 0] = self.occ[:, -1] = True

    def box(self, x0, y0, x1, y1, value=True):
        """Fill (or clear) an axis-aligned rectangle [m]."""
        i0, i1 = int(round(x0 / self.res)), int(round(x1 / self.res))
        j0, j1 = int(round(y0 / self.res)), int(round(y1 / self.res))
        self.occ[max(0, j0):min(self.h, j1), max(0, i0):min(self.w, i1)] = value

    def occupied(self, x, y):
        """Vectorized occupancy lookup (outside = occupied)."""
        ix = np.floor(np.asarray(x) / self.res).astype(int)
        iy = np.floor(np.asarray(y) / self.res).astype(int)
        ok = (ix >= 0) & (ix < self.w) & (iy >= 0) & (iy < self.h)
        out = np.ones(ix.shape, dtype=bool)
        out[ok] = self.occ[iy[ok], ix[ok]]
        return out

    def scan(self, pose, n_beams=360, max_range=8.0, noise=0.0, rng=None):
        """Simulate a 360-degree scan; returns ranges (inf = no return)."""
        ang = pose[2] + np.linspace(-math.pi, math.pi, n_beams, endpoint=False)
        steps = np.arange(0.05, max_range, self.res * 0.5)
        xs = pose[0] + np.cos(ang)[:, None] * steps[None, :]
        ys = pose[1] + np.sin(ang)[:, None] * steps[None, :]
        hit = self.occupied(xs, ys)
        first = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
        r = np.where(first >= 0, steps[np.maximum(first, 0)], np.inf)
        if noise > 0.0 and rng is not None:
            r = r + rng.normal(0.0, noise, r.shape)
        return r

    def map_values(self, known_mask=None):
        """OccupancyGrid values (0 / 100, -1 outside `known_mask`)."""
        v = np.where(self.occ, 100, 0).astype(np.int16)
        if known_mask is not None:
            v[~known_mask] = -1
        return v

    def clearance(self, pose, footprint):
        """Smallest distance from the footprint to any obstacle cell [m]."""
        r = footprint.circumscribed_radius + 0.6
        i0 = max(0, int((pose[0] - r) / self.res))
        i1 = min(self.w, int((pose[0] + r) / self.res) + 1)
        j0 = max(0, int((pose[1] - r) / self.res))
        j1 = min(self.h, int((pose[1] + r) / self.res) + 1)
        jj, ii = np.nonzero(self.occ[j0:j1, i0:i1])
        if len(ii) == 0:
            return math.inf
        pts = np.column_stack(((ii + i0 + 0.5) * self.res, (jj + j0 + 0.5) * self.res))
        c, s = math.cos(pose[2]), math.sin(pose[2])
        dx, dy = pts[:, 0] - pose[0], pts[:, 1] - pose[1]
        local = np.column_stack((c * dx + s * dy, -s * dx + c * dy))
        return float(footprint.signed_distance(local).min()) - 0.5 * self.res


def warehouse():
    """A 14 m x 10 m warehouse with shelf rows and a few boxes."""
    w = World(14.0, 10.0)
    for x0 in (3.0, 6.0, 9.0):          # three shelf rows
        w.box(x0, 1.6, x0 + 0.6, 4.2)
        w.box(x0, 5.8, x0 + 0.6, 8.4)
    w.box(12.0, 0.0, 12.1, 3.8)         # partition wall with a door
    w.box(12.0, 4.8, 12.1, 10.0)
    w.box(1.2, 8.6, 1.8, 9.2)           # a box near the start corner
    return w


class Sim:
    """Closed loop: world + robot + NavigationManager."""

    def __init__(self, world, pose, cfg=None, overrides=None, rate=10.0,
                 omni=False, scan_noise=0.01, seed=1, known_mask=None,
                 map_period=1.0, threaded=False):
        """Create the simulation.

        Args:
            world: World.
            pose: Initial true pose (x, y, yaw).
            cfg: NavigationConfig (planner_threaded is forced off).
            overrides: Profile overrides.
            rate: Control rate [Hz].
            omni: Omnidirectional base.
            scan_noise: Range noise sigma [m].
            seed: RNG seed.
            known_mask: Callable(world, pose) -> bool mask of known cells
                (None = the whole map is known), to mimic exploration.
            map_period: How often a new map is "published" [s].
        """
        self.world = world
        self.pose = tuple(pose)
        self.cfg = cfg or NavigationConfig()
        self.cfg.planner_threaded = threaded
        self.threaded = threaded
        if omni:
            self.cfg.motion_model = 'omnidirectional'
            if self.cfg.max_vel_y <= 0.0:
                self.cfg.max_vel_y = self.cfg.max_vel_x
        self.nav = NavigationManager(self.cfg, overrides)
        self.dt = 1.0 / rate
        self.t = 0.0
        self.rng = np.random.default_rng(seed)
        self.scan_noise = scan_noise
        self.known_mask = known_mask
        self.map_period = map_period
        self._last_map = -math.inf
        self.map_to_odom = (0.0, 0.0, 0.0)  # odom == map unless shifted
        # Error of the robot's map -> odom estimate (what SLAM believes);
        # setting it back to zero mimics a loop-closure correction.
        self.loc_error = (0.0, 0.0, 0.0)
        self.trace = [self.pose]
        self.min_clearance = math.inf
        self.cmds = []
        self.localization_ok = True
        self.known = None
        self.publish_map()

    def odom_pose(self):
        """Pose of the base in odom (true pose seen through map->odom)."""
        inv = (-self.map_to_odom[0], -self.map_to_odom[1], -self.map_to_odom[2])
        c, s = math.cos(inv[2]), math.sin(inv[2])
        tx = c * inv[0] - s * inv[1]
        ty = s * inv[0] + c * inv[1]
        return compose((tx, ty, inv[2]), self.pose)

    def publish_map(self):
        """Give the manager the current map."""
        mask = None
        if self.known_mask is not None:
            m = self.known_mask(self.world, self.pose)
            self.known = m if self.known is None else (self.known | m)
            mask = self.known
        self.nav.set_map(self.world.map_values(mask), self.world.res, 0.0, 0.0,
                         'map', self.t)
        self._last_map = self.t

    def step(self):
        """Advance one control period."""
        if self.t - self._last_map >= self.map_period:
            self.publish_map()
        r = self.world.scan(self.pose, noise=self.scan_noise, rng=self.rng)
        ang = np.linspace(-math.pi, math.pi, len(r), endpoint=False)
        ok = np.isfinite(r) & (r <= self.cfg.obstacle_max_range)
        pts = np.column_stack((r[ok] * np.cos(ang[ok]), r[ok] * np.sin(ang[ok])))
        self.nav.add_scan(self.t, pts, self.odom_pose(), 'lidar')
        m2o_est = compose(self.loc_error, self.map_to_odom)
        pose_g = compose(m2o_est, self.odom_pose()) if self.localization_ok else None
        cmd = self.nav.tick(self.t, pose_g, self.odom_pose(), m2o_est)
        if cmd is None:
            cmd = (0.0, 0.0, 0.0)
        self.cmds.append(cmd)
        if self.threaded:
            # Give a planner thread a chance to finish (wall time).
            import time
            time.sleep(0.002)
        vy = cmd[1] if self.cfg.motion_model == 'omnidirectional' else 0.0
        self.pose = integrate_velocity(self.pose, cmd[0], vy, cmd[2], self.dt)
        self.t += self.dt
        self.trace.append(self.pose)
        fp = self.nav.active_profile.footprint
        self.min_clearance = min(self.min_clearance, self.world.clearance(self.pose, fp))
        if self.world.clearance(self.pose, fp) < -0.01:
            raise AssertionError(f'collision at t={self.t:.1f} pose={self.pose}')

    def run_until_done(self, task_id, timeout):
        """Step until the task ends; return its snapshot."""
        end = self.t + timeout
        while self.t < end:
            self.step()
            snap = self.nav.get_task(task_id)
            if snap is None or snap['status'] != ACTIVE:
                # a few more steps so the stop command goes out
                for _ in range(3):
                    self.step()
                return snap
        return self.nav.get_task(task_id)

    def navigate(self, goal, profile='', timeout=120.0, task_id=None):
        """Send a goal and run until it ends."""
        task_id = task_id or f'goal-{self.t:.1f}'
        self.nav.start_navigate(task_id, goal, profile, self.t)
        return self.run_until_done(task_id, timeout)


def radius_known(radius):
    """known_mask factory: cells within `radius` of the robot become known."""
    def mask(world, pose):
        jj, ii = np.mgrid[0:world.h, 0:world.w]
        x = (ii + 0.5) * world.res
        y = (jj + 0.5) * world.res
        return np.hypot(x - pose[0], y - pose[1]) <= radius
    return mask


def path_length(trace):
    """Length of a pose trace [m]."""
    return float(sum(distance(a, b) for a, b in zip(trace, trace[1:])))


__all__ = ['World', 'warehouse', 'Sim', 'radius_known', 'path_length',
           'transform_points']


class ScanMapper:
    """A stand-in for SLAM: carves the map from the simulated LiDAR.

    Cells along each beam become free, the cell at the return becomes
    occupied; everything never seen stays unknown (-1).
    """

    def __init__(self, world, max_range=6.0):
        """Start with an all-unknown map the size of the world."""
        self.world = world
        self.max_range = max_range
        self.values = np.full((world.h, world.w), -1, dtype=np.int16)

    def update(self, pose, ranges):
        """Insert one 360-degree scan taken at `pose`."""
        w = self.world
        n = len(ranges)
        ang = pose[2] + np.linspace(-math.pi, math.pi, n, endpoint=False)
        r = np.where(np.isfinite(ranges), ranges, self.max_range)
        r = np.minimum(r, self.max_range)
        steps = np.arange(0.0, self.max_range, w.res * 0.5)
        xs = pose[0] + np.cos(ang)[:, None] * steps[None, :]
        ys = pose[1] + np.sin(ang)[:, None] * steps[None, :]
        free = steps[None, :] < (r[:, None] - w.res)
        ix = np.floor(xs / w.res).astype(int)
        iy = np.floor(ys / w.res).astype(int)
        ok = free & (ix >= 0) & (ix < w.w) & (iy >= 0) & (iy < w.h)
        self.values[iy[ok], ix[ok]] = 0
        hit = np.isfinite(ranges) & (ranges < self.max_range)
        hx = np.floor((pose[0] + np.cos(ang[hit]) * ranges[hit]) / w.res).astype(int)
        hy = np.floor((pose[1] + np.sin(ang[hit]) * ranges[hit]) / w.res).astype(int)
        ok = (hx >= 0) & (hx < w.w) & (hy >= 0) & (hy < w.h)
        self.values[hy[ok], hx[ok]] = 100
