"""
footprint.py — robot footprint model
====================================
The footprint is either a circle (radius) or a polygon given in the robot
base frame. It provides the inscribed and circumscribed radii used by the
costmap, outline points used for collision checking, and a vectorized
signed distance from points to the footprint used by the safety monitor.

Footprint strings follow the Nav2 convention:
    "[[x1, y1], [x2, y2], [x3, y3], ...]"   (metres, base frame, >= 3 vertices)
An empty string selects the circular footprint.
"""
# Author: Raymundo Gomez

import ast
import math

import numpy as np


def parse_footprint(text):
    """Parse a Nav2-style footprint string into a list of (x, y) vertices.

    Args:
        text: "[[x, y], ...]" or "" (circular footprint).

    Returns:
        List of (x, y) tuples, or None for the circular footprint.

    Raises:
        ValueError: If the string is malformed or has fewer than 3 vertices.
    """
    text = (text or '').strip()
    if not text:
        return None
    try:
        value = ast.literal_eval(text)
        vertices = [(float(p[0]), float(p[1])) for p in value]
    except (ValueError, SyntaxError, TypeError, IndexError) as exc:
        raise ValueError(f'footprint "{text}" is not [[x, y], ...]') from exc
    if len(vertices) < 3:
        raise ValueError('a footprint polygon needs at least 3 vertices')
    return vertices


def _pad_polygon(vertices, padding):
    """Grow a polygon outward by `padding` (Nav2 style, per coordinate)."""
    if padding == 0.0:
        return list(vertices)
    out = []
    for x, y in vertices:
        out.append((x + math.copysign(padding, x) if x != 0.0 else x,
                    y + math.copysign(padding, y) if y != 0.0 else y))
    return out


class Footprint:
    """Circle or polygon footprint in the robot base frame.

    Attributes:
        radius: Circle radius [m] (circular footprint only).
        vertices: Array (M, 2) of polygon vertices, or None for a circle.
        inscribed_radius: Radius of the largest circle inside the footprint.
        circumscribed_radius: Radius of the smallest circle around it.
    """

    def __init__(self, radius=0.2, vertices=None, padding=0.0):
        """Create a footprint.

        Args:
            radius: Circle radius [m]; used when vertices is None.
            vertices: Polygon vertices [(x, y), ...] or None.
            padding: Extra margin added to every side [m].
        """
        self.padding = float(padding)
        if vertices is None:
            if radius <= 0.0:
                raise ValueError('robot_radius must be > 0')
            self.radius = float(radius) + self.padding
            self.vertices = None
            self.inscribed_radius = self.radius
            self.circumscribed_radius = self.radius
        else:
            self.radius = 0.0
            self.vertices = np.asarray(
                _pad_polygon(vertices, self.padding), dtype=float)
            # The origin must lie inside the polygon for the radii to mean
            # anything; a forklift footprint may be off-center, which is fine.
            dist = self._edge_distance(np.zeros((1, 2)))[0]
            if not self._inside(np.zeros((1, 2)))[0]:
                raise ValueError('footprint must contain the base origin')
            self.inscribed_radius = float(dist)
            self.circumscribed_radius = float(
                np.max(np.hypot(self.vertices[:, 0], self.vertices[:, 1])))

    @classmethod
    def from_config(cls, footprint_text, radius, padding):
        """Build from the configuration values (string, radius, padding)."""
        return cls(radius=radius, vertices=parse_footprint(footprint_text),
                   padding=padding)

    @property
    def is_circle(self):
        """True for a circular footprint."""
        return self.vertices is None

    # ── geometry ─────────────────────────────────────────────────────────
    def outline(self, spacing):
        """Return points along the footprint outline, base frame.

        Args:
            spacing: Maximum distance between consecutive points [m].

        Returns:
            Array (K, 2).
        """
        spacing = max(1e-3, float(spacing))
        if self.is_circle:
            n = max(8, int(math.ceil(2.0 * math.pi * self.radius / spacing)))
            ang = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
            return np.column_stack((self.radius * np.cos(ang),
                                    self.radius * np.sin(ang)))
        pts = []
        verts = self.vertices
        for i in range(len(verts)):
            a, b = verts[i], verts[(i + 1) % len(verts)]
            n = max(1, int(math.ceil(np.hypot(*(b - a)) / spacing)))
            t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
            pts.append(a + t * (b - a))
        return np.vstack(pts)

    def _inside(self, points):
        """Even-odd point-in-polygon test (polygon footprint only)."""
        x, y = points[:, 0], points[:, 1]
        inside = np.zeros(len(points), dtype=bool)
        verts = self.vertices
        j = len(verts) - 1
        for i in range(len(verts)):
            xi, yi = verts[i]
            xj, yj = verts[j]
            crosses = (yi > y) != (yj > y)
            with np.errstate(divide='ignore', invalid='ignore'):
                x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            inside ^= crosses & (x < x_cross)
            j = i
        return inside

    def _edge_distance(self, points):
        """Unsigned distance from points to the polygon boundary."""
        best = np.full(len(points), np.inf)
        verts = self.vertices
        for i in range(len(verts)):
            a, b = verts[i], verts[(i + 1) % len(verts)]
            ab = b - a
            denom = float(ab @ ab) or 1e-12
            t = np.clip(((points - a) @ ab) / denom, 0.0, 1.0)
            proj = a + t[:, None] * ab
            best = np.minimum(best, np.hypot(points[:, 0] - proj[:, 0],
                                             points[:, 1] - proj[:, 1]))
        return best

    def signed_distance(self, points):
        """Signed distance from points (base frame) to the footprint [m].

        Negative inside the footprint, positive outside.

        Args:
            points: Array (N, 2) in the base frame.

        Returns:
            Array (N,).
        """
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        if len(points) == 0:
            return np.zeros(0)
        if self.is_circle:
            return np.hypot(points[:, 0], points[:, 1]) - self.radius
        d = self._edge_distance(points)
        return np.where(self._inside(points), -d, d)

    def contains(self, points, margin=0.0):
        """True for points inside the footprint grown by `margin`."""
        return self.signed_distance(points) <= margin

    def as_list(self):
        """Return the outline as [(x, y), ...] (a polygon for a circle)."""
        if self.is_circle:
            return [tuple(p) for p in self.outline(self.radius * 0.4)]
        return [tuple(v) for v in self.vertices]
