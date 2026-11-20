"""Unit tests of the ROS-free building blocks (no simulation loop)."""
# Author: Raymundo Gomez

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sendero import planner as pl  # noqa: E402
from sendero.aligner import DONE, AlignController, AlignSettings, TargetTracker  # noqa: E402
from sendero.config import NavigationConfig, build_profile  # noqa: E402
from sendero.costmap import (INSCRIBED, LETHAL, UNKNOWN, CostGrid,  # noqa: E402
                             StaticMap, bounded_edt, build_global_costmap,
                             footprint_samples, inflate)
from sendero.footprint import Footprint, parse_footprint  # noqa: E402
from sendero.geometry import (compose, integrate_velocity, inverse,  # noqa: E402
                              normalize_angle, relative)
from sendero.safety import (CLEAR, NO_SCAN, STOP, Limits, SafetyMonitor,  # noqa: E402
                            SafetySettings, VelocityLimiter)


# ── geometry ─────────────────────────────────────────────────────────────
def test_compose_inverse_identity():
    a = (1.2, -0.4, 2.5)
    e = compose(a, inverse(a))
    assert np.allclose(e[:2], 0.0, atol=1e-12) and abs(e[2]) < 1e-12
    b = (0.3, 0.7, -1.0)
    assert np.allclose(compose(a, relative(a, b)), b)


def test_integrate_velocity_arc():
    p = integrate_velocity((0.0, 0.0, 0.0), 1.0, 0.0, math.pi / 2, 1.0)
    r = 1.0 / (math.pi / 2)
    assert np.allclose(p, (r, r, math.pi / 2), atol=1e-9)


# ── footprint ────────────────────────────────────────────────────────────
def test_footprint_polygon_radii_and_distance():
    fp = Footprint(vertices=parse_footprint('[[0.3, 0.2], [0.3, -0.2], [-0.2, -0.2], [-0.2, 0.2]]'))
    assert fp.inscribed_radius == pytest.approx(0.2)
    assert fp.circumscribed_radius == pytest.approx(math.hypot(0.3, 0.2))
    d = fp.signed_distance(np.array([[0.0, 0.0], [0.5, 0.0], [0.3, 0.0]]))
    assert d[0] == pytest.approx(-0.2) and d[1] == pytest.approx(0.2)
    assert abs(d[2]) < 1e-9


def test_footprint_rejects_bad_input():
    with pytest.raises(ValueError):
        parse_footprint('[[0, 0], [1, 1]]')
    with pytest.raises(ValueError):
        Footprint(vertices=[(1.0, 1.0), (2.0, 1.0), (2.0, 2.0)])


# ── costmap ──────────────────────────────────────────────────────────────
def test_bounded_edt_matches_brute_force():
    rng = np.random.default_rng(0)
    mask = rng.random((40, 50)) < 0.02
    d = bounded_edt(mask, 8)
    oy, ox = np.nonzero(mask)
    yy, xx = np.mgrid[0:40, 0:50]
    brute = np.sqrt(((yy[..., None] - oy) ** 2 + (xx[..., None] - ox) ** 2).min(axis=2))
    near = brute <= 8
    assert np.allclose(d[near], brute[near])
    assert np.all(np.isinf(d[~near]))


def test_inflation_values():
    lethal = np.zeros((41, 41), dtype=bool)
    lethal[20, 20] = True
    cost, dist = inflate(lethal, None, 0.05, 0.2, 0.5, 5.0)
    assert cost[20, 20] == LETHAL
    assert cost[20, 24] == INSCRIBED          # 0.20 m
    assert 0 < cost[20, 26] < INSCRIBED       # 0.30 m
    assert cost[20, 26] > cost[20, 29]        # decays outward
    assert cost[20, 31] == 0                  # 0.55 m: beyond the radius
    assert dist[20, 30] == pytest.approx(0.5)


def test_unknown_cells_stay_unknown_unless_inscribed():
    lethal = np.zeros((20, 20), dtype=bool)
    unknown = np.zeros((20, 20), dtype=bool)
    unknown[:, 10:] = True
    lethal[10, 9] = True
    cost, _ = inflate(lethal, unknown, 0.05, 0.1, 0.3, 5.0)
    assert cost[10, 10] == INSCRIBED and cost[10, 15] == UNKNOWN


def test_occupancy_translation():
    data = np.array([[0, 100, INSCRIBED, LETHAL, UNKNOWN]], dtype=np.uint8)
    grid = CostGrid(data, np.zeros(data.shape), 0.05, 0.0, 0.0)
    v = grid.to_occupancy_values()
    assert list(v) == [0, 39, 99, 100, -1]


# ── planner ──────────────────────────────────────────────────────────────
def _corridor_map(unknown_right=False):
    v = np.zeros((60, 100), dtype=np.int16)
    v[0, :] = v[-1, :] = 100
    v[:, 0] = v[:, -1] = 100
    v[:45, 50] = 100                     # a wall with a gap at the top
    if unknown_right:
        v[:, 70:] = -1
    return StaticMap(v, 0.05, 0.0, 0.0, 'map', 65, 50)


@pytest.mark.parametrize('backend', ['astar', 'dijkstra'])
def test_planner_goes_around_wall(backend):
    if backend == 'dijkstra' and not pl.HAVE_SCIPY:
        pytest.skip('SciPy not installed')
    sm = _corridor_map()
    fp = Footprint(radius=0.15)
    grid = build_global_costmap(sm, None, fp, 0.4, 5.0)
    s = pl.PlannerSettings(backend=backend)
    r = pl.plan_path(grid, (1.0, 1.0), (4.0, 1.0), 0.0, s)
    assert r.ok, r.message
    assert r.path[:, 1].max() > 2.2       # through the gap near the top
    cost = grid.cost_at(r.path[:, 0], r.path[:, 1])
    assert not np.any((cost == LETHAL) | (cost == INSCRIBED))
    assert np.allclose(r.path[-1, :2], (4.0, 1.0))


def test_backends_agree_on_length():
    if not pl.HAVE_SCIPY:
        pytest.skip('SciPy not installed')
    sm = _corridor_map()
    grid = build_global_costmap(sm, None, Footprint(radius=0.15), 0.4, 5.0)
    a = pl.plan_path(grid, (1.0, 1.0), (4.0, 1.0), 0.0,
                     pl.PlannerSettings(backend='astar', smooth=False))
    d = pl.plan_path(grid, (1.0, 1.0), (4.0, 1.0), 0.0,
                     pl.PlannerSettings(backend='dijkstra', smooth=False))
    assert a.ok and d.ok
    assert a.length == pytest.approx(d.length, rel=0.05)


def test_unknown_space_policy():
    sm = _corridor_map(unknown_right=True)
    grid = build_global_costmap(sm, None, Footprint(radius=0.15), 0.4, 5.0)
    ok = pl.plan_path(grid, (1.0, 1.0), (4.0, 1.5), None,
                      pl.PlannerSettings(allow_unknown=True))
    no = pl.plan_path(grid, (1.0, 1.0), (4.0, 1.5), None,
                      pl.PlannerSettings(allow_unknown=False))
    assert ok.ok
    assert not no.ok and no.code == pl.GOAL_OCCUPIED


def test_unreachable_and_snapping():
    v = np.zeros((60, 100), dtype=np.int16)
    v[:, 50] = 100                       # full wall: right side unreachable
    sm = StaticMap(v, 0.05, 0.0, 0.0, 'map', 65, 50)
    grid = build_global_costmap(sm, None, Footprint(radius=0.15), 0.4, 5.0)
    r = pl.plan_path(grid, (1.0, 1.5), (4.0, 1.5), None, pl.PlannerSettings())
    assert not r.ok and r.code == pl.NO_VALID_PATH
    # Goal 5 cm from the wall (inside the inscribed band): snapped.
    r = pl.plan_path(grid, (1.0, 1.5), (2.42, 1.5), None, pl.PlannerSettings())
    assert r.ok and r.goal_adjusted
    r = pl.plan_path(grid, (1.0, 1.5), (5.2, 1.5), None, pl.PlannerSettings())
    assert not r.ok and r.code == pl.GOAL_OUTSIDE_MAP


# ── safety and limits ────────────────────────────────────────────────────
def test_safety_monitor():
    fp = Footprint(radius=0.2)
    s = SafetySettings()
    mon = SafetyMonitor()
    ahead = np.array([[0.30, 0.0]])
    assert mon.check((0.2, 0.0, 0.0), ahead, 0.05, fp, s) == (0.0, 0.0, 0.0)
    assert mon.state == STOP
    # Moving away from a close point is allowed.
    assert mon.check((-0.1, 0.0, 0.0), ahead, 0.05, fp, s) == (-0.1, 0.0, 0.0)
    assert mon.state == CLEAR
    # A stale scan stops the robot.
    assert mon.check((0.1, 0.0, 0.0), ahead, 5.0, fp, s) == (0.0, 0.0, 0.0)
    assert mon.state == NO_SCAN


def test_velocity_limiter_keeps_curvature():
    lim = VelocityLimiter()
    limits = Limits(max_vel_x=0.5, acc_lim_x=0.5, acc_lim_theta=0.5)
    vx, _, wz = lim.apply((0.4, 0.0, 0.8), limits, 0.1, False)
    assert vx == pytest.approx(0.025) and wz == pytest.approx(0.05)
    assert wz / vx == pytest.approx(2.0)


# ── alignment ────────────────────────────────────────────────────────────
def test_align_controller_converges_differential():
    ctrl = AlignController()
    s = AlignSettings()
    goal = (2.0, 1.0, 0.3)
    for start in [(1.3, 0.6, 0.0), (1.5, 1.4, -0.6), (2.3, 1.2, 0.5)]:
        ctrl.reset()
        pose = start
        for _ in range(3000):
            cmd, err = ctrl.step(pose, goal, s)
            if ctrl.phase == DONE:
                break
            pose = integrate_velocity(pose, cmd[0], 0.0, cmd[2], 0.05)
        assert ctrl.phase == DONE, start
        ex, ey, eyaw = AlignController.errors(pose, goal)
        assert math.hypot(ex, ey) < 0.03 and abs(eyaw) < s.yaw_tolerance + 1e-3


def test_target_tracker_rejects_outliers():
    tr = TargetTracker(gain=0.5, jump_reject=0.2, reject_limit=3)
    tr.update((1.0, 0.0, 0.0), 0.0)
    assert not tr.update((2.0, 0.0, 0.0), 0.1)
    assert tr.pose[0] == pytest.approx(1.0)
    tr.update((1.1, 0.0, 0.0), 0.2)
    assert tr.pose[0] == pytest.approx(1.05)


# ── configuration ────────────────────────────────────────────────────────
def test_config_validation():
    NavigationConfig().validate()
    with pytest.raises(ValueError):
        NavigationConfig(motion_model='ackermann').validate()
    with pytest.raises(ValueError):
        NavigationConfig(recovery_behaviors=['dance']).validate()
    with pytest.raises(ValueError):
        NavigationConfig(default_profile='nope').validate()
    with pytest.raises(ValueError):
        build_profile(NavigationConfig(), 'x', {'controller_frequency': 5.0})
    p = build_profile(NavigationConfig(), 'carrying',
                      {'max_vel_x': 0.1, 'footprint': '[[0.4, 0.2], [0.4, -0.2], [-0.2, -0.2], [-0.2, 0.2]]'})
    assert p.limits.max_vel_x == 0.1 and not p.footprint.is_circle
    assert normalize_angle(3 * math.pi) == pytest.approx(-math.pi)


def test_footprint_samples_cover_interior():
    fp = Footprint(vertices=[(0.3, 0.2), (0.3, -0.2), (-0.2, -0.2), (-0.2, 0.2)])
    pts = footprint_samples(fp, 0.05)
    assert pts[:, 0].min() >= -0.2 - 1e-9 and pts[:, 0].max() <= 0.3 + 1e-9
    assert np.any(np.hypot(pts[:, 0], pts[:, 1]) < 0.05)
