"""
aligner.py — precision alignment
================================
Moves the robot to an exact pose relative to a target seen by perception
(a box, a pallet, a shelf slot, an ArUco marker), for forklift pickup and
placement. Runs after global navigation has brought the robot close.

The desired robot pose is given in the target frame, so it follows the
target as perception refines it. The target is tracked in a smooth frame
(odom), filtered, and kept for a while when it drops out of view.

Differential drive, three phases:
  APPROACH  polar pose controller (Siegwart & Nourbakhsh) to a pre-pose
            `standoff` behind the final pose, on its approach line;
  DOCK      straight final approach along the line, steering to remove the
            lateral and heading errors; it re-approaches if the lateral error
            cannot be removed in the remaining distance;
  ROTATE    final in-place correction of the heading.
Omnidirectional: one proportional controller on x, y and heading.

Errors are reported in the frame of the desired pose:
  longitudinal  along the approach direction (negative = not there yet)
  lateral       sideways
  yaw           heading
"""
# Author: Raymundo Gomez

import math
from dataclasses import dataclass

from .geometry import compose, normalize_angle, relative

APPROACH = 'APPROACH'
DOCK = 'DOCK'
ROTATE = 'ROTATE'
DONE = 'DONE'
FAILED = 'FAILED'


@dataclass
class AlignSettings:
    """Settings of one alignment (config defaults, goal overrides)."""
    omnidirectional: bool = False
    max_vel_x: float = 0.08
    max_vel_y: float = 0.08
    max_vel_theta: float = 0.4
    min_vel_x: float = 0.01
    min_rotation_speed: float = 0.05
    xy_tolerance: float = 0.02
    yaw_tolerance: float = 0.035
    standoff: float = 0.25
    approach_xy_tolerance: float = 0.04
    approach_yaw_tolerance: float = 0.10
    k_rho: float = 0.6
    k_alpha: float = 1.8
    k_beta: float = -0.6
    k_longitudinal: float = 1.0
    k_lateral: float = 3.0
    k_heading: float = 1.5
    rotation_kp: float = 1.5
    allow_reverse: bool = True
    max_reapproaches: int = 2


class TargetTracker:
    """Filtered estimate of the target pose in a smooth frame."""

    def __init__(self, gain=0.3, jump_reject=0.3, reject_limit=5):
        """Create an empty tracker.

        Args:
            gain: Weight of a new measurement in (0, 1]; 1 = no filtering.
            jump_reject: Measurements this far from the estimate are
                treated as outliers [m].
            reject_limit: After this many consecutive outliers the
                estimate restarts from the new measurement.
        """
        self.gain = min(1.0, max(1e-3, float(gain)))
        self.jump_reject = float(jump_reject)
        self.reject_limit = int(reject_limit)
        self.reset()

    def reset(self):
        """Forget the estimate."""
        self.pose = None
        self.stamp = None
        self.updates = 0
        self._rejects = 0

    def update(self, pose, stamp):
        """Add a measurement (x, y, yaw) taken at `stamp` [s]."""
        if self.pose is None:
            self.pose, self.stamp = tuple(pose), float(stamp)
            self.updates = 1
            return True
        jump = math.hypot(pose[0] - self.pose[0], pose[1] - self.pose[1])
        if jump > self.jump_reject:
            self._rejects += 1
            if self._rejects < self.reject_limit:
                return False
            self.pose = tuple(pose)
        else:
            a = self.gain
            yaw = self.pose[2] + a * normalize_angle(pose[2] - self.pose[2])
            self.pose = ((1 - a) * self.pose[0] + a * pose[0],
                         (1 - a) * self.pose[1] + a * pose[1],
                         normalize_angle(yaw))
        self._rejects = 0
        self.stamp = float(stamp)
        self.updates += 1
        return True

    def age(self, now):
        """Seconds since the last accepted measurement (inf if none)."""
        return math.inf if self.stamp is None else max(0.0, now - self.stamp)


def _scale(v, w, vmax, wmax):
    """Scale (v, w) together so both stay within their limits."""
    k = 1.0
    if abs(v) > vmax > 0.0:
        k = min(k, vmax / abs(v))
    if abs(w) > wmax > 0.0:
        k = min(k, wmax / abs(w))
    return v * k, w * k


class AlignController:
    """Pose stabilization to the desired robot pose."""

    def __init__(self):
        """Create the controller."""
        self.reset()

    def reset(self):
        """Start a new alignment."""
        self.phase = APPROACH
        self.reapproaches = 0
        self.message = ''

    @staticmethod
    def errors(robot, goal):
        """(longitudinal, lateral, yaw) error of the robot in the goal frame."""
        ex, ey, eyaw = relative(goal, robot)
        return ex, ey, eyaw

    def step(self, robot, goal, s):
        """Compute one command.

        Args:
            robot: Robot pose (x, y, yaw) in the tracking frame.
            goal: Desired robot pose in the same frame.
            s: AlignSettings.

        Returns:
            (cmd, errors): cmd = (vx, vy, wz); errors = (long, lat, yaw).
            self.phase becomes DONE or FAILED when finished.
        """
        err = self.errors(robot, goal)
        if self.phase in (DONE, FAILED):
            return (0.0, 0.0, 0.0), err
        if s.omnidirectional:
            return self._omni(robot, goal, err, s), err
        ex, ey, eyaw = err

        if self.phase == APPROACH:
            if s.standoff <= 0.0:
                pre = goal
            else:
                pre = compose(goal, (-s.standoff, 0.0, 0.0))
            # Already on the approach line behind the goal: dock directly.
            if (ex <= 0.02 and abs(ey) <= s.approach_xy_tolerance
                    and abs(eyaw) <= s.approach_yaw_tolerance):
                self.phase = DOCK
            else:
                px, py, pyaw = relative(pre, robot)
                rho = math.hypot(px, py)
                if rho <= s.approach_xy_tolerance:
                    if abs(pyaw) <= s.approach_yaw_tolerance:
                        self.phase = DOCK if s.standoff > 0.0 else ROTATE
                    else:
                        w = -math.copysign(
                            min(s.max_vel_theta,
                                max(s.min_rotation_speed,
                                    s.rotation_kp * abs(pyaw))), pyaw)
                        return (0.0, 0.0, w), err
                else:
                    return self._polar(robot, pre, s), err

        if self.phase == DOCK:
            if abs(ex) <= s.xy_tolerance:
                if abs(ey) <= s.xy_tolerance:
                    self.phase = ROTATE
                elif self.reapproaches < s.max_reapproaches:
                    self.reapproaches += 1
                    self.phase = APPROACH
                    return (0.0, 0.0, 0.0), err
                else:
                    self.phase = FAILED
                    self.message = (f'lateral error {ey:.3f} m remains after '
                                    f'{self.reapproaches} re-approaches')
                    return (0.0, 0.0, 0.0), err
            else:
                # Signed speed toward the goal along the line.
                v = -s.k_longitudinal * ex
                v = math.copysign(min(s.max_vel_x, max(s.min_vel_x, abs(v))), v)
                if v < 0.0 and not s.allow_reverse:
                    self.phase = FAILED
                    self.message = 'overshot the goal and reversing is disabled'
                    return (0.0, 0.0, 0.0), err
                if v > 0.0:
                    desired = -math.atan(s.k_lateral * ey)
                else:
                    desired = 0.0
                w = s.k_heading * normalize_angle(desired - eyaw)
                v, w = _scale(v, w, s.max_vel_x, s.max_vel_theta)
                return (v, 0.0, w), err

        if self.phase == ROTATE:
            if abs(eyaw) <= s.yaw_tolerance:
                self.phase = DONE
                return (0.0, 0.0, 0.0), err
            w = -math.copysign(min(s.max_vel_theta,
                                   max(s.min_rotation_speed,
                                       s.rotation_kp * abs(eyaw))), eyaw)
            return (0.0, 0.0, w), err
        return (0.0, 0.0, 0.0), err

    def _polar(self, robot, target, s):
        """Polar-coordinate pose controller toward `target`."""
        dx = target[0] - robot[0]
        dy = target[1] - robot[1]
        rho = math.hypot(dx, dy)
        alpha = normalize_angle(math.atan2(dy, dx) - robot[2])
        direction = 1.0
        if s.allow_reverse and abs(alpha) > math.pi / 2.0:
            direction = -1.0
            alpha = normalize_angle(alpha + math.pi)
        beta = normalize_angle(target[2] - robot[2] - alpha)
        v = direction * s.k_rho * rho
        w = s.k_alpha * alpha + s.k_beta * beta
        v, w = _scale(v, w, s.max_vel_x, s.max_vel_theta)
        if 0.0 < abs(v) < s.min_vel_x:
            k = s.min_vel_x / abs(v)
            v, w = _scale(v * k, w * k, s.max_vel_x, s.max_vel_theta)
        return (v, 0.0, w)

    def _omni(self, robot, goal, err, s):
        """Proportional x, y, heading control for an omnidirectional base."""
        ex, ey, eyaw = err
        if (math.hypot(ex, ey) <= s.xy_tolerance
                and abs(eyaw) <= s.yaw_tolerance):
            self.phase = DONE
            return (0.0, 0.0, 0.0)
        self.phase = DOCK
        # Error vector in the robot frame: goal position seen from the robot.
        gx, gy, _ = relative(robot, goal)
        k = s.k_longitudinal
        vx, vy = k * gx, k * gy
        norm = math.hypot(vx, vy)
        vmax = min(s.max_vel_x, s.max_vel_y) if s.max_vel_y > 0 else s.max_vel_x
        if norm > vmax:
            vx, vy = vx * vmax / norm, vy * vmax / norm
        w = -s.rotation_kp * eyaw
        w = max(-s.max_vel_theta, min(s.max_vel_theta, w))
        return (vx, vy, w)
