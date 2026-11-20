"""
geometry.py — planar (SE(2)) helpers
====================================
A pose is a tuple (x, y, yaw) in metres and radians. Point sets are NumPy
arrays of shape (N, 2). Nothing here depends on ROS.
"""
# Author: Raymundo Gomez

import math

import numpy as np


def normalize_angle(angle):
    """Wrap an angle to [-pi, pi).

    Args:
        angle: Angle [rad], scalar or NumPy array.

    Returns:
        The wrapped angle, same type as the input.
    """
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quaternion(x, y, z, w):
    """Return the yaw (rotation about z) of a quaternion [rad]."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quaternion_from_yaw(yaw):
    """Return the quaternion (x, y, z, w) of a pure rotation about z."""
    half = 0.5 * float(yaw)
    return 0.0, 0.0, math.sin(half), math.cos(half)


def compose(a, b):
    """Return a (+) b: pose b expressed in the frame of pose a.

    Args:
        a: Pose (x, y, yaw) of frame B in frame A.
        b: Pose (x, y, yaw) expressed in frame B.

    Returns:
        Pose b expressed in frame A.
    """
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1],
            a[1] + s * b[0] + c * b[1],
            normalize_angle(a[2] + b[2]))


def inverse(a):
    """Return the inverse pose a^-1 (so that a (+) a^-1 is the identity)."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return (-c * a[0] - s * a[1],
            s * a[0] - c * a[1],
            normalize_angle(-a[2]))


def relative(a, b):
    """Return a^-1 (+) b: pose b expressed in the frame of pose a."""
    return compose(inverse(a), b)


def transform_points(pose, points):
    """Map points from the frame of `pose` into the frame `pose` lives in.

    Args:
        pose: Pose (x, y, yaw) of the source frame.
        points: Array (N, 2) in the source frame.

    Returns:
        Array (N, 2) in the parent frame.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    c, s = math.cos(pose[2]), math.sin(pose[2])
    out = np.empty_like(points)
    out[:, 0] = pose[0] + c * points[:, 0] - s * points[:, 1]
    out[:, 1] = pose[1] + s * points[:, 0] + c * points[:, 1]
    return out


def inverse_transform_points(pose, points):
    """Map points from the parent frame into the frame of `pose`."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    c, s = math.cos(pose[2]), math.sin(pose[2])
    dx = points[:, 0] - pose[0]
    dy = points[:, 1] - pose[1]
    out = np.empty_like(points)
    out[:, 0] = c * dx + s * dy
    out[:, 1] = -s * dx + c * dy
    return out


def distance(a, b):
    """Return the planar distance between two poses or points [m]."""
    return math.hypot(b[0] - a[0], b[1] - a[1])


def path_lengths(path_xy):
    """Return the cumulative arc length along a polyline.

    Args:
        path_xy: Array (N, 2).

    Returns:
        Array (N,) with 0 at the first point.
    """
    path_xy = np.asarray(path_xy, dtype=float).reshape(-1, 2)
    if len(path_xy) < 2:
        return np.zeros(len(path_xy))
    seg = np.hypot(np.diff(path_xy[:, 0]), np.diff(path_xy[:, 1]))
    return np.concatenate(([0.0], np.cumsum(seg)))


def integrate_velocity(pose, vx, vy, wz, dt):
    """Integrate a body-frame velocity over dt (exact for constant twist).

    Args:
        pose: Start pose (x, y, yaw).
        vx: Forward velocity [m/s].
        vy: Lateral velocity [m/s] (0 for a differential drive).
        wz: Angular velocity [rad/s].
        dt: Duration [s].

    Returns:
        The end pose.
    """
    if abs(wz) < 1e-9:
        dx, dy = vx * dt, vy * dt
    else:
        th = wz * dt
        s, c = math.sin(th), math.cos(th)
        dx = (vx * s - vy * (1.0 - c)) / wz
        dy = (vx * (1.0 - c) + vy * s) / wz
    return compose(pose, (dx, dy, wz * dt))
