"""
safety.py — velocity limits and the last-line safety monitor
============================================================
Every command leaves the framework through two stages:

1. VelocityLimiter: clamps the command to the profile's velocity limits and
   to its acceleration / deceleration limits.
2. SafetyMonitor: checks the command against the newest raw sensor points
   (not the costmap). The footprint is simulated forward for a short
   horizon; if a point would come within `stop_margin` of the footprint the
   command becomes zero, within `slow_margin` it is scaled down. Only points
   the motion approaches count, so a robot standing next to a wall can
   still drive away from it. If no fresh scan exists the robot stops.

The safety monitor is independent of planning and control: a bug or a stale
costmap upstream cannot drive the robot into something the sensor sees.
"""
# Author: Raymundo Gomez

import math
from dataclasses import dataclass

import numpy as np

from .geometry import integrate_velocity, inverse_transform_points

CLEAR = 'CLEAR'
SLOW = 'SLOW'
STOP = 'STOP'
NO_SCAN = 'NO_SCAN'


@dataclass
class Limits:
    """Velocity and acceleration limits."""
    max_vel_x: float = 0.2
    min_vel_x: float = -0.1
    max_vel_y: float = 0.0
    max_vel_theta: float = 1.0
    acc_lim_x: float = 0.5
    acc_lim_y: float = 0.5
    acc_lim_theta: float = 1.5
    decel_lim_x: float = 1.0
    decel_lim_y: float = 1.0
    decel_lim_theta: float = 3.0


def _ramp(target, current, acc, dec, dt):
    """Move `current` toward `target` within the acceleration limits."""
    if dt <= 0.0:
        return target
    speeding_up = abs(target) > abs(current) and target * current >= 0.0
    lim = (acc if speeding_up else dec) * dt
    return current + max(-lim, min(lim, target - current))


class VelocityLimiter:
    """Velocity and acceleration limiting of commands."""

    def __init__(self):
        """Start from rest."""
        self.last = (0.0, 0.0, 0.0)

    def reset(self, current=(0.0, 0.0, 0.0)):
        """Set the last command (for example after a stop)."""
        self.last = tuple(float(v) for v in current)

    def apply(self, cmd, limits, dt, omnidirectional):
        """Limit a command.

        For a differential drive the forward and angular velocities are
        scaled together, so the commanded curvature (and the path) is kept.

        Args:
            cmd: (vx, vy, wz) requested.
            limits: Limits.
            dt: Time since the last command [s].
            omnidirectional: False forces vy = 0.

        Returns:
            (vx, vy, wz) limited.
        """
        vx, vy, wz = (float(v) for v in cmd)
        vx = min(limits.max_vel_x, max(limits.min_vel_x, vx))
        vy = 0.0 if not omnidirectional else max(-limits.max_vel_y,
                                                 min(limits.max_vel_y, vy))
        wz = max(-limits.max_vel_theta, min(limits.max_vel_theta, wz))
        lx, ly, lw = self.last
        nx = _ramp(vx, lx, limits.acc_lim_x, limits.decel_lim_x, dt)
        ny = _ramp(vy, ly, limits.acc_lim_y, limits.decel_lim_y, dt)
        nw = _ramp(wz, lw, limits.acc_lim_theta, limits.decel_lim_theta, dt)
        if not omnidirectional and abs(vx) > 1e-6 and abs(wz) > 1e-6:
            # Keep the curvature: use the most restrictive ratio.
            rx = nx / vx if abs(vx) > 1e-9 else 1.0
            rw = nw / wz if abs(wz) > 1e-9 else 1.0
            r = max(0.0, min(rx, rw))
            if (nx * vx) >= 0.0 and (nw * wz) >= 0.0:
                nx, nw = vx * r, wz * r
        self.last = (nx, ny, nw)
        return self.last


@dataclass
class SafetySettings:
    """Settings of the safety monitor."""
    enabled: bool = True
    stop_margin: float = 0.05
    slow_margin: float = 0.10
    slow_factor: float = 0.5
    horizon: float = 0.6
    scan_timeout: float = 0.5
    require_scan: bool = True


class SafetyMonitor:
    """Command check against the newest raw sensor points."""

    def __init__(self):
        """Create the monitor."""
        self.state = CLEAR
        self.closest = math.inf

    def check(self, cmd, points_base, scan_age, footprint, s):
        """Return a safe version of a command.

        Args:
            cmd: (vx, vy, wz).
            points_base: Array (N, 2) of sensor points in the base frame.
            scan_age: Age of the newest scan [s] (inf if none).
            footprint: Footprint.
            s: SafetySettings.

        Returns:
            (vx, vy, wz) after the check; self.state tells what happened.
        """
        vx, vy, wz = cmd
        moving = abs(vx) + abs(vy) + abs(wz) > 1e-6
        if not s.enabled or not moving:
            self.state = CLEAR
            return cmd
        if s.require_scan and not scan_age <= s.scan_timeout:
            self.state = NO_SCAN
            return (0.0, 0.0, 0.0)
        pts = np.asarray(points_base, dtype=float).reshape(-1, 2)
        if len(pts) == 0:
            self.state = CLEAR
            self.closest = math.inf
            return cmd
        # Only points that could matter within the horizon.
        reach = (math.hypot(vx, vy) * s.horizon + footprint.circumscribed_radius
                 + s.slow_margin)
        pts = pts[np.hypot(pts[:, 0], pts[:, 1]) <= reach]
        if len(pts) == 0:
            self.state = CLEAR
            self.closest = math.inf
            return cmd
        d0 = footprint.signed_distance(pts)
        self.closest = float(d0.min())
        n = 6
        dt = s.horizon / n
        pose = (0.0, 0.0, 0.0)
        worst = np.inf
        for _ in range(n):
            pose = integrate_velocity(pose, vx, vy, wz, dt)
            dk = footprint.signed_distance(inverse_transform_points(pose, pts))
            approaching = dk < d0 - 1e-4
            if np.any(approaching):
                worst = min(worst, float(dk[approaching].min()))
        if worst < s.stop_margin:
            self.state = STOP
            return (0.0, 0.0, 0.0)
        if worst < s.slow_margin:
            self.state = SLOW
            return (vx * s.slow_factor, vy * s.slow_factor, wz)
        self.state = CLEAR
        return cmd
