"""
controller.py — path following
==============================
Regulated Pure Pursuit (Macenski et al., 2023), the default path follower
of Nav2 for differential-drive robots, with an omnidirectional variant.

Each control cycle:
  1. find the path point nearest to the robot (searching forward only, so
     a path that passes the same place twice is followed in order);
  2. take a lookahead ("carrot") point at a distance that grows with speed;
  3. differential drive: steer on the arc through the carrot
     (curvature = 2 * y_carrot / L^2), or rotate in place first if the
     carrot is far off to the side;
     omnidirectional: drive straight toward the carrot and turn the body
     toward the path heading at the same time;
  4. regulate the speed: slower on tight curves, near obstacles and when
     approaching the goal;
  5. project the command forward and check the footprint against the
     local costmap; a collision ahead stops the robot (BLOCKED);
  6. check the next metres of the path against the local costmap, so the
     navigator can replan around a new obstacle before reaching it.

Goal checking is stateful: once the position is within tolerance it stays
accepted while the robot turns to the final heading.
"""
# Author: Raymundo Gomez

import math
from dataclasses import dataclass

import numpy as np

from .geometry import integrate_velocity, normalize_angle

TRACKING = 'TRACKING'
ROTATING_TO_PATH = 'ROTATING_TO_PATH'
ROTATING_TO_GOAL = 'ROTATING_TO_GOAL'
GOAL_REACHED = 'GOAL_REACHED'
BLOCKED = 'BLOCKED'
OFF_PATH = 'OFF_PATH'
NO_PATH = 'NO_PATH'

DIFFERENTIAL = 'differential'
OMNIDIRECTIONAL = 'omnidirectional'


@dataclass
class ControllerSettings:
    """Settings of the path follower (from the config and the profile)."""
    motion_model: str = DIFFERENTIAL
    max_vel_x: float = 0.2
    max_vel_y: float = 0.2
    max_vel_theta: float = 1.0
    min_rotation_speed: float = 0.1
    lookahead_time: float = 1.5
    min_lookahead: float = 0.3
    max_lookahead: float = 0.9
    use_rotate_to_heading: bool = True
    rotate_to_heading_angle: float = 0.785
    rotation_kp: float = 1.5
    heading_kp: float = 1.5
    regulated_min_radius: float = 0.6
    min_regulated_speed: float = 0.05
    proximity_slowdown_distance: float = 0.25
    proximity_min_scale: float = 0.4
    approach_distance: float = 0.5
    min_approach_speed: float = 0.05
    collision_check_time: float = 1.0
    path_check_distance: float = 1.5
    max_path_deviation: float = 0.5
    xy_goal_tolerance: float = 0.15
    yaw_goal_tolerance: float = 0.15
    ignore_goal_yaw: bool = False
    omni_face_path: bool = True


@dataclass
class ControlOutput:
    """Result of one control cycle."""
    cmd: tuple = (0.0, 0.0, 0.0)
    state: str = NO_PATH
    remaining: float = 0.0
    distance_to_goal: float = 0.0
    nearest_index: int = 0
    deviation: float = 0.0
    lookahead: tuple = (0.0, 0.0)
    collision_in: float = None
    collision_pose: tuple = None
    path_blocked: bool = False


def _rotation_speed(error, kp, min_speed, max_speed):
    """P control of an angle error with a minimum and maximum speed."""
    mag = min(max_speed, max(min_speed, kp * abs(error)))
    return math.copysign(mag, error)


def project_collision(pose, cmd, horizon, grid, footprint, samples,
                      allow_unknown=True):
    """Forward-simulate a command and check the footprint.

    If the robot already overlaps an obstacle (pressed against a shelf), only
    poses that move deeper into it count, so the robot can always back out.

    Args:
        pose: Current pose (x, y, yaw) in the grid frame.
        cmd: (vx, vy, wz).
        horizon: Simulated time [s].
        grid: CostGrid (local costmap).
        footprint: Footprint.
        samples: Footprint sample points for this grid's resolution.
        allow_unknown: Passed to the footprint check.

    Returns:
        Time of the first collision [s], or None.
    """
    hit = project_collision_pose(pose, cmd, horizon, grid, footprint, samples,
                                 allow_unknown)
    return None if hit is None else hit[0]


def project_collision_pose(pose, cmd, horizon, grid, footprint, samples,
                           allow_unknown=True):
    """Like project_collision() but returns (time, pose) of the collision."""
    vx, vy, wz = cmd
    reach = abs(vx) + abs(vy) + abs(wz) * footprint.circumscribed_radius
    if reach < 1e-6 or horizon <= 0.0:
        return None
    n = int(min(60, max(1, math.ceil(horizon * reach / grid.resolution))))
    dt = horizon / n
    poses = np.empty((n, 3))
    p = pose
    for k in range(n):
        p = integrate_velocity(p, vx, vy, wz, dt)
        poses[k] = p
    hits = grid.footprint_collisions(poses, footprint, allow_unknown, samples)
    if not np.any(hits):
        return None
    if grid.footprint_collisions(np.asarray([pose]), footprint,
                                 allow_unknown, samples)[0]:
        d0 = float(grid.distance_at(pose[0], pose[1])[0])
        deeper = grid.distance_at(poses[:, 0], poses[:, 1]) < d0 - 1e-6
        hits &= deeper
        if not np.any(hits):
            return None
    k = int(np.argmax(hits))
    return float((k + 1) * dt), tuple(float(v) for v in poses[k])


class PathFollower:
    """Regulated Pure Pursuit path follower (differential or omnidirectional)."""

    def __init__(self):
        """Create a follower with no path."""
        self.path = None
        self._cum = None
        self._idx = 0
        self.xy_reached = False

    def set_path(self, path):
        """Start following a new path.

        Args:
            path: Array (N, 3) of (x, y, yaw) in the global frame.
        """
        path = np.asarray(path, dtype=float).reshape(-1, 3)
        self.path = path
        seg = np.hypot(np.diff(path[:, 0]), np.diff(path[:, 1]))
        self._cum = np.concatenate(([0.0], np.cumsum(seg)))
        self._idx = 0

    def clear(self):
        """Forget the path and the goal-check state."""
        self.path = None
        self._cum = None
        self._idx = 0
        self.xy_reached = False

    def reset_goal_check(self):
        """Forget that the goal position was reached (new goal)."""
        self.xy_reached = False

    @property
    def length(self):
        """Path length [m]."""
        return 0.0 if self._cum is None else float(self._cum[-1])

    def _nearest(self, x, y, s):
        """Nearest path index, searching forward from the last one."""
        path, cum = self.path, self._cum
        window = max(2.0 * s.max_lookahead, 1.0)
        end = int(np.searchsorted(cum, cum[self._idx] + window, side='right'))
        end = max(end, self._idx + 1)
        seg = path[self._idx:end]
        d = np.hypot(seg[:, 0] - x, seg[:, 1] - y)
        k = int(np.argmin(d))
        idx, dev = self._idx + k, float(d[k])
        if dev > s.max_path_deviation:
            # Possibly a localization correction: look along the whole rest.
            rest = path[self._idx:]
            d = np.hypot(rest[:, 0] - x, rest[:, 1] - y)
            k = int(np.argmin(d))
            idx, dev = self._idx + k, float(d[k])
        self._idx = idx
        return idx, dev

    def step(self, pose, speed, grid, footprint, samples, s):
        """Compute one velocity command.

        Args:
            pose: Robot pose (x, y, yaw) in the global frame.
            speed: Current forward speed [m/s] (sets the lookahead).
            grid: Local costmap (CostGrid) in the global frame, or None.
            footprint: Footprint.
            samples: Footprint samples for `grid`.
            s: ControllerSettings.

        Returns:
            ControlOutput.
        """
        out = ControlOutput()
        if self.path is None or len(self.path) == 0:
            return out
        x, y, yaw = pose
        path, cum = self.path, self._cum
        idx, dev = self._nearest(x, y, s)
        goal = path[-1]
        out.nearest_index = idx
        out.deviation = dev
        out.remaining = float(cum[-1] - cum[idx] + dev)
        out.distance_to_goal = math.hypot(goal[0] - x, goal[1] - y)

        if dev > s.max_path_deviation and not self.xy_reached:
            out.state = OFF_PATH
            return out

        # ── goal check ───────────────────────────────────────────────────
        if not self.xy_reached and out.distance_to_goal <= s.xy_goal_tolerance \
                and out.remaining <= s.xy_goal_tolerance + 0.3:
            self.xy_reached = True
        if self.xy_reached:
            err = normalize_angle(goal[2] - yaw)
            if s.ignore_goal_yaw or abs(err) <= s.yaw_goal_tolerance:
                out.state = GOAL_REACHED
                return out
            wz = _rotation_speed(err, s.rotation_kp, s.min_rotation_speed,
                                 s.max_vel_theta)
            vx = vy = 0.0
            if s.motion_model == OMNIDIRECTIONAL:
                c, sn = math.cos(yaw), math.sin(yaw)
                ex = c * (goal[0] - x) + sn * (goal[1] - y)
                ey = -sn * (goal[0] - x) + c * (goal[1] - y)
                vx = float(np.clip(ex, -0.05, 0.05))
                vy = float(np.clip(ey, -0.05, 0.05))
            out.cmd = (vx, vy, wz)
            out.state = ROTATING_TO_GOAL
            return self._check(out, pose, grid, footprint, samples, s)

        # ── lookahead point ──────────────────────────────────────────────
        lookahead = float(np.clip(abs(speed) * s.lookahead_time,
                                  s.min_lookahead, s.max_lookahead))
        ahead = path[idx:]
        d = np.hypot(ahead[:, 0] - x, ahead[:, 1] - y)
        far = np.nonzero(d >= lookahead)[0]
        j = idx + (int(far[0]) if len(far) else len(ahead) - 1)
        carrot = path[j]
        c, sn = math.cos(yaw), math.sin(yaw)
        lx = c * (carrot[0] - x) + sn * (carrot[1] - y)
        ly = -sn * (carrot[0] - x) + c * (carrot[1] - y)
        out.lookahead = (float(carrot[0]), float(carrot[1]))

        # ── speed regulation shared by both motion models ────────────────
        v = s.max_vel_x
        if grid is not None:
            d_obs = float(grid.distance_at(x, y)[0]) - footprint.inscribed_radius
            if d_obs < s.proximity_slowdown_distance:
                v *= max(s.proximity_min_scale,
                         d_obs / max(1e-6, s.proximity_slowdown_distance))

        if s.motion_model == OMNIDIRECTIONAL:
            v = max(v, s.min_regulated_speed)
            if out.remaining < s.approach_distance:
                v = min(v, max(s.min_approach_speed,
                               s.max_vel_x * out.remaining / s.approach_distance))
            norm = math.hypot(lx, ly)
            vx, vy = (v * lx / norm, v * ly / norm) if norm > 1e-6 else (0.0, 0.0)
            scale = max(1.0, abs(vx) / max(1e-6, s.max_vel_x),
                        abs(vy) / max(1e-6, s.max_vel_y))
            vx, vy = vx / scale, vy / scale
            target_yaw = carrot[2] if s.omni_face_path else goal[2]
            wz = float(np.clip(s.heading_kp * normalize_angle(target_yaw - yaw),
                               -s.max_vel_theta, s.max_vel_theta))
            out.cmd = (vx, vy, wz)
            out.state = TRACKING
            return self._check(out, pose, grid, footprint, samples, s)

        # ── differential drive ───────────────────────────────────────────
        angle = math.atan2(ly, lx)
        if s.use_rotate_to_heading and abs(angle) > s.rotate_to_heading_angle:
            out.cmd = (0.0, 0.0, _rotation_speed(angle, s.rotation_kp,
                                                 s.min_rotation_speed,
                                                 s.max_vel_theta))
            out.state = ROTATING_TO_PATH
            return self._check(out, pose, grid, footprint, samples, s)
        d2 = lx * lx + ly * ly
        kappa = 2.0 * ly / d2 if d2 > 1e-9 else 0.0
        if abs(kappa) > 1e-9:
            radius = 1.0 / abs(kappa)
            if radius < s.regulated_min_radius:
                v *= radius / s.regulated_min_radius
        v = max(v, s.min_regulated_speed)
        if out.remaining < s.approach_distance:
            v = min(v, max(s.min_approach_speed,
                           s.max_vel_x * out.remaining / s.approach_distance))
        wz = v * kappa
        if abs(wz) > s.max_vel_theta:
            wz = math.copysign(s.max_vel_theta, wz)
            v = abs(wz / kappa)
        out.cmd = (float(v), 0.0, float(wz))
        out.state = TRACKING
        return self._check(out, pose, grid, footprint, samples, s)

    def _check(self, out, pose, grid, footprint, samples, s):
        """Collision projection of the command and of the path ahead."""
        if grid is None:
            return out
        hit = project_collision_pose(pose, out.cmd, s.collision_check_time,
                                     grid, footprint, samples)
        if hit is not None:
            out.collision_in, out.collision_pose = hit
            out.cmd = (0.0, 0.0, 0.0)
            out.state = BLOCKED
        if self.path is not None and s.path_check_distance > 0.0:
            cum = self._cum
            i0 = out.nearest_index
            i1 = int(np.searchsorted(cum, cum[i0] + s.path_check_distance,
                                     side='right'))
            seg = self.path[i0:max(i1, i0 + 1)]
            # Skip the part of the path still under the robot.
            dist = np.hypot(seg[:, 0] - pose[0], seg[:, 1] - pose[1])
            seg = seg[dist > footprint.circumscribed_radius]
            if len(seg):
                step = max(1, int(len(seg) / 30))
                hits = grid.footprint_collisions(seg[::step], footprint, True,
                                                 samples)
                out.path_blocked = bool(np.any(hits))
        return out
