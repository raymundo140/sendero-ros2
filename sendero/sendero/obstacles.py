"""
obstacles.py — sensor obstacles
===============================
Converts range scans to obstacle points and keeps a short, time-decayed
history of them. Points are stored in a smooth frame (normally odom), so a
SLAM correction of map -> odom moves all of them together and none is
smeared. Old points expire instead of being ray-traced away: a moving
obstacle disappears from the costmaps after the decay time.
"""
# Author: Raymundo Gomez

import collections
import math

import numpy as np


def scan_to_points(ranges, angle_min, angle_increment, range_min, range_max,
                   max_obstacle_range=None, beam_step=1):
    """Convert a planar range scan to points in the sensor frame.

    Invalid returns (NaN, inf, outside [range_min, range_max]) are dropped.

    Args:
        ranges: Sequence of ranges [m].
        angle_min: Angle of the first beam [rad].
        angle_increment: Angle between beams [rad].
        range_min, range_max: Valid range of the sensor [m].
        max_obstacle_range: Only returns closer than this are obstacles
            (None = range_max).
        beam_step: Use every n-th beam.

    Returns:
        Array (N, 2).
    """
    r = np.asarray(ranges, dtype=float)
    idx = np.arange(0, len(r), max(1, int(beam_step)))
    r = r[idx]
    limit = range_max if max_obstacle_range is None else min(
        range_max, max_obstacle_range)
    ok = np.isfinite(r) & (r >= max(range_min, 1e-3)) & (r <= limit)
    ang = angle_min + idx[ok] * angle_increment
    r = r[ok]
    return np.column_stack((r * np.cos(ang), r * np.sin(ang)))


def voxel_filter(points, cell):
    """Keep one point per square cell (reduces dense scans)."""
    if points is None or len(points) == 0 or cell <= 0.0:
        return points
    keys = np.floor(points / cell).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(first)]


class ObstacleBuffer:
    """Time-stamped obstacle points from one or more sensors."""

    def __init__(self, max_age, voxel=0.0):
        """Create an empty buffer.

        Args:
            max_age: Longest age any consumer will ask for [s].
            voxel: Voxel size used to thin each scan [m] (0 = off).
        """
        self.max_age = float(max_age)
        self.voxel = float(voxel)
        self._scans = collections.deque()
        self._latest = {}

    def add(self, stamp, points, source=''):
        """Add the points of one scan.

        Args:
            stamp: Scan time [s].
            points: Array (N, 2) in the storage frame.
            source: Sensor name, used to keep the latest scan per sensor.
        """
        pts = voxel_filter(np.asarray(points, dtype=float).reshape(-1, 2),
                           self.voxel)
        self._scans.append((float(stamp), pts))
        self._latest[source] = (float(stamp), pts)
        self._prune(float(stamp))

    def _prune(self, now):
        """Drop scans older than max_age relative to `now`."""
        while self._scans and now - self._scans[0][0] > self.max_age:
            self._scans.popleft()

    def points(self, now, max_age):
        """Return all points newer than max_age (storage frame).

        Args:
            now: Current time [s].
            max_age: Age limit [s].

        Returns:
            Array (N, 2).
        """
        sel = [p for t, p in self._scans
               if now - t <= max_age and t - now <= 1.0 and len(p)]
        if not sel:
            return np.zeros((0, 2))
        return np.vstack(sel)

    def latest(self, now, max_age):
        """Return the newest scan of each sensor, if fresh enough.

        Returns:
            (points, age): stacked points (storage frame) and the age of
            the oldest of the used scans [s]; (empty, inf) if none is fresh.
        """
        fresh = [(t, p) for t, p in self._latest.values()
                 if now - t <= max_age]
        if not fresh:
            return np.zeros((0, 2)), math.inf
        age = max(now - t for t, _ in fresh)
        pts = [p for _, p in fresh if len(p)]
        return (np.vstack(pts) if pts else np.zeros((0, 2))), max(0.0, age)

    def newest_stamp(self):
        """Time of the newest scan, or None."""
        if not self._latest:
            return None
        return max(t for t, _ in self._latest.values())

    def clear(self):
        """Forget the point history (a recovery behavior).

        The newest scan of each sensor is kept for the safety monitor and
        returns to the history with the next scan.
        """
        self._scans.clear()

    def __len__(self):
        """Number of stored points."""
        return int(sum(len(p) for _, p in self._scans))
