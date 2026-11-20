"""
recovery.py — navigation-level recovery behaviors
=================================================
Short, local behaviors that try to free the robot so the navigator can plan
again. Each behavior is a small state machine driven by step(); it returns
a command and whether it has finished.

    clear   forget the sensor-obstacle history (stale or ghost obstacles)
    wait    stand still for a while (a person or another robot may move away)
    backup  drive straight backward, only into space the costmap shows free
    spin    rotate in place, only if the rotated footprint is free

These recoveries are deliberately bounded (count and duration). Deciding
that a goal is not worth pursuing is the client's job (for example
Frontera's region memory), not the navigator's.
"""
# Author: Raymundo Gomez

import math

import numpy as np

from .controller import project_collision
from .geometry import normalize_angle

CLEAR = 'clear'
WAIT = 'wait'
BACKUP = 'backup'
SPIN = 'spin'
KNOWN_BEHAVIORS = (CLEAR, WAIT, BACKUP, SPIN)


class Behavior:
    """Base class. Subclasses implement start() and step()."""

    name = ''

    def __init__(self):
        """Create an idle behavior."""
        self.done = False
        self.success = False
        self.message = ''
        self.t0 = 0.0
        self.start_pose = None

    def start(self, now, pose):
        """Begin the behavior at time `now` and pose `pose`."""
        self.t0 = now
        self.start_pose = pose
        self.done = False
        self.success = False
        self.message = ''

    def finish(self, success, message=''):
        """Mark the behavior as finished; returns a zero command."""
        self.done = True
        self.success = success
        self.message = message
        return (0.0, 0.0, 0.0)

    def step(self, now, pose, grid, footprint, samples):
        """Return the next command (vx, vy, wz)."""
        raise NotImplementedError


class ClearBehavior(Behavior):
    """Forget the obstacle history through a callback."""

    name = CLEAR

    def __init__(self, clear_callback):
        """Store the callback that clears the obstacle buffer."""
        super().__init__()
        self._clear = clear_callback

    def step(self, now, pose, grid, footprint, samples):
        """Clear immediately and finish."""
        self._clear()
        return self.finish(True, 'obstacle history cleared')


class WaitBehavior(Behavior):
    """Stand still for a fixed time."""

    name = WAIT

    def __init__(self, duration):
        """Store the duration [s]."""
        super().__init__()
        self.duration = float(duration)

    def step(self, now, pose, grid, footprint, samples):
        """Zero command until the time is up."""
        if now - self.t0 >= self.duration:
            return self.finish(True, f'waited {self.duration:.1f} s')
        return (0.0, 0.0, 0.0)


class BackUpBehavior(Behavior):
    """Drive straight backward for a distance."""

    name = BACKUP

    def __init__(self, distance, speed, time_allowance, check_time=0.5):
        """Store the motion settings.

        Args:
            distance: Distance to travel backward [m] (> 0).
            speed: Backward speed magnitude [m/s] (> 0).
            time_allowance: Give up after this long [s].
            check_time: Collision projection horizon [s].
        """
        super().__init__()
        self.distance = abs(float(distance))
        self.speed = abs(float(speed))
        self.time_allowance = float(time_allowance)
        self.check_time = float(check_time)
        self.travelled = 0.0

    def step(self, now, pose, grid, footprint, samples):
        """Back up until the distance is covered or something blocks."""
        self.travelled = math.hypot(pose[0] - self.start_pose[0],
                                    pose[1] - self.start_pose[1])
        if self.travelled >= self.distance:
            return self.finish(True, f'backed up {self.travelled:.2f} m')
        if now - self.t0 > self.time_allowance:
            return self.finish(False, 'backup timed out')
        remaining = self.distance - self.travelled
        speed = min(self.speed, max(0.02, 2.0 * remaining))
        cmd = (-speed, 0.0, 0.0)
        if grid is not None and project_collision(
                pose, cmd, self.check_time, grid, footprint, samples) is not None:
            return self.finish(False, 'obstacle behind the robot')
        return cmd


class SpinBehavior(Behavior):
    """Rotate in place by an angle."""

    name = SPIN

    def __init__(self, angle, speed, time_allowance, check_time=0.5):
        """Store the motion settings (angle [rad], speed [rad/s])."""
        super().__init__()
        self.angle = float(angle)
        self.speed = abs(float(speed))
        self.time_allowance = float(time_allowance)
        self.check_time = float(check_time)
        self.turned = 0.0
        self._last_yaw = None

    def start(self, now, pose):
        """Begin the rotation."""
        super().start(now, pose)
        self.turned = 0.0
        self._last_yaw = pose[2]

    def step(self, now, pose, grid, footprint, samples):
        """Rotate until the angle is covered or the footprint would collide."""
        self.turned += abs(normalize_angle(pose[2] - self._last_yaw))
        self._last_yaw = pose[2]
        if self.turned >= abs(self.angle):
            return self.finish(True, f'rotated {math.degrees(self.turned):.0f} deg')
        if now - self.t0 > self.time_allowance:
            return self.finish(False, 'spin timed out')
        remaining = abs(self.angle) - self.turned
        w = math.copysign(min(self.speed, max(0.15, 1.5 * remaining)), self.angle)
        cmd = (0.0, 0.0, w)
        if grid is not None and not footprint.is_circle and project_collision(
                pose, cmd, self.check_time, grid, footprint, samples) is not None:
            return self.finish(False, 'rotation would collide')
        return cmd


class RecoverySequence:
    """The configured recovery behaviors, tried in order, cycling."""

    def __init__(self, names, factory, max_recoveries):
        """Create the sequence.

        Args:
            names: Behavior names in order (see KNOWN_BEHAVIORS).
            factory: Callable(name) -> Behavior.
            max_recoveries: Total behaviors allowed per goal.
        """
        self.names = [n for n in names if n in KNOWN_BEHAVIORS]
        self.factory = factory
        self.max_recoveries = int(max_recoveries)
        self.count = 0
        self._next = 0

    def reset(self):
        """Start again from the first behavior (new goal)."""
        self.count = 0
        self._next = 0

    def exhausted(self):
        """True once the per-goal limit is used up."""
        return not self.names or self.count >= self.max_recoveries

    def next_behavior(self):
        """Return the next Behavior instance, or None if exhausted."""
        if self.exhausted():
            return None
        name = self.names[self._next % len(self.names)]
        self._next += 1
        self.count += 1
        return self.factory(name)


def free_space_behind(pose, distance, grid, footprint, samples):
    """True if the footprint can move `distance` straight backward."""
    if grid is None:
        return True
    n = max(2, int(math.ceil(distance / grid.resolution)))
    c, s = math.cos(pose[2]), math.sin(pose[2])
    d = np.linspace(distance / n, distance, n)
    poses = np.column_stack((pose[0] - c * d, pose[1] - s * d,
                             np.full(n, pose[2])))
    return not np.any(grid.footprint_collisions(poses, footprint, True, samples))
