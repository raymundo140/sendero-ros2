"""Closed-loop tests: the NavigationManager drives a simulated robot.

Each test runs the manager exactly as the ROS node does (scans in, tick(),
command out) against the 2D warehouse in sim_world.py. A collision of the
true footprint with a true obstacle fails any test.
"""
# Author: Raymundo Gomez

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from sim_world import Sim, radius_known, warehouse  # noqa: E402

from sendero import navigator as nv  # noqa: E402
from sendero import planner as pl  # noqa: E402
from sendero.config import NavigationConfig  # noqa: E402
from sendero.geometry import compose, inverse, relative  # noqa: E402

GOAL = (13.0, 8.5, math.pi / 2)
FORKLIFT = '[[0.32, 0.15], [0.32, -0.15], [-0.12, -0.15], [-0.12, 0.15]]'


def _done(snap, status=nv.SUCCEEDED):
    assert snap is not None
    assert snap['status'] == status, (snap['status'], snap['code'], snap['message'])


def test_differential_reaches_goal_across_the_warehouse():
    sim = Sim(warehouse(), (1.0, 1.0, 0.0))
    snap = sim.navigate(GOAL, timeout=150)
    _done(snap)
    assert math.hypot(sim.pose[0] - GOAL[0], sim.pose[1] - GOAL[1]) <= 0.16
    assert abs(math.remainder(sim.pose[2] - GOAL[2], 2 * math.pi)) <= 0.16
    assert sim.min_clearance > 0.05
    assert all(c == (0.0, 0.0, 0.0) for c in sim.cmds[-3:])


def test_omnidirectional_reaches_goal():
    sim = Sim(warehouse(), (1.0, 1.0, 0.0), omni=True)
    _done(sim.navigate(GOAL, timeout=150))
    assert sim.min_clearance > 0.05


def test_polygon_forklift_footprint():
    cfg = NavigationConfig(footprint=FORKLIFT, inflation_radius=0.5)
    sim = Sim(warehouse(), (1.0, 1.0, 0.0), cfg=cfg)
    _done(sim.navigate((10.6, 7.0, -math.pi / 2), timeout=150))
    assert sim.min_clearance > 0.02


def test_exploration_through_unknown_space_vs_strict_profile():
    cfg = NavigationConfig(profile_names=['exploration', 'transport'],
                           default_profile='exploration')
    over = {'exploration': {'allow_unknown': True, 'ignore_goal_yaw': True},
            'transport': {'allow_unknown': False}}
    sim = Sim(warehouse(), (1.0, 1.0, 0.0), cfg=cfg, overrides=over,
              known_mask=radius_known(3.0))
    # The goal is in unknown space at the start; the map grows as it drives.
    strict = sim.navigate((7.5, 1.0, 0.0), profile='transport', timeout=20)
    _done(strict, nv.ABORTED)
    assert strict['code'] in (pl.GOAL_OCCUPIED, pl.NO_VALID_PATH)
    _done(sim.navigate((7.5, 1.0, 0.0), profile='exploration', timeout=80))
    assert sim.min_clearance > 0.05


def test_replans_around_a_new_obstacle():
    world = warehouse()
    sim = Sim(world, (1.0, 1.0, 0.0))
    sim.nav.start_navigate('g', (10.5, 1.0, 0.0), '', sim.t)
    for _ in range(30):
        sim.step()
    world.box(5.0, 0.3, 5.5, 1.4)   # a pallet drops in the aisle (not on the map)
    snap = sim.run_until_done('g', 150)
    _done(snap)                     # Sim fails the test on any collision
    trace = np.array(sim.trace)
    near = (trace[:, 0] > 4.8) & (trace[:, 0] < 5.7)
    assert trace[near, 1].min() > 1.4   # went around the pallet


def test_blocked_door_aborts_after_bounded_recoveries():
    world = warehouse()
    world.box(12.0, 3.7, 12.1, 4.9)          # door closed (not on the map)
    cfg = NavigationConfig(recovery_wait_s=1.0)
    sim = Sim(world, (11.0, 4.3, 0.0), cfg=cfg)
    snap = sim.navigate((13.2, 4.3, 0.0), timeout=200)
    _done(snap, nv.ABORTED)
    assert snap['code'] in (nv.NO_VALID_CONTROL, nv.FAILED_TO_MAKE_PROGRESS,
                            pl.NO_VALID_PATH, pl.GOAL_OCCUPIED)
    assert sim.t < 120


def test_cancel_stops_the_robot():
    sim = Sim(warehouse(), (1.0, 1.0, 0.0))
    sim.nav.start_navigate('g', GOAL, '', sim.t)
    for _ in range(40):
        sim.step()
    assert sim.nav.cancel('g', sim.t)
    for _ in range(5):
        sim.step()
    _done(sim.nav.get_task('g'), nv.CANCELED)
    assert sim.cmds[-1] == (0.0, 0.0, 0.0)
    p = sim.pose
    for _ in range(10):
        sim.step()
    assert sim.pose == p


def test_preemption_and_reject_policy():
    sim = Sim(warehouse(), (1.0, 1.0, 0.0))
    sim.nav.start_navigate('a', GOAL, '', sim.t)
    for _ in range(20):
        sim.step()
    sim.nav.start_navigate('b', (2.5, 1.0, 0.0), '', sim.t)
    old = sim.nav.get_task('a')                   # superseded
    assert old['status'] == nv.ABORTED and old['code'] == nv.NAV_PREEMPTED
    snap = sim.run_until_done('b', 60)
    _done(snap)
    cfg = NavigationConfig(goal_policy='reject')
    sim2 = Sim(warehouse(), (1.0, 1.0, 0.0), cfg=cfg)
    sim2.nav.start_navigate('a', GOAL, '', sim2.t)
    sim2.step()
    ok, why = sim2.nav.can_accept('')
    assert not ok and 'busy' in why


def test_profiles_unknown_name():
    cfg = NavigationConfig(unknown_profile_policy='reject')
    sim = Sim(warehouse(), (1.0, 1.0, 0.0), cfg=cfg)
    assert not sim.nav.can_accept('flying')[0]
    sim2 = Sim(warehouse(), (1.0, 1.0, 0.0))
    assert sim2.nav.can_accept('flying')[0]


def test_localization_loss_short_and_long():
    sim = Sim(warehouse(), (1.0, 1.0, 0.0))
    sim.nav.start_navigate('g', (6.0, 1.0, 0.0), '', sim.t)
    for _ in range(30):
        sim.step()
    sim.localization_ok = False
    for _ in range(10):              # 1 s: hold still and wait
        sim.step()
        assert sim.cmds[-1] == (0.0, 0.0, 0.0) or sim.cmds[-1][0] <= sim.cmds[-2][0]
    sim.localization_ok = True
    _done(sim.run_until_done('g', 80))
    sim.nav.start_navigate('h', (1.0, 1.0, math.pi), '', sim.t)
    for _ in range(10):
        sim.step()
    sim.localization_ok = False
    snap = sim.run_until_done('h', 10)
    _done(snap, nv.ABORTED)
    assert snap['code'] == nv.NAV_TF_ERROR


def test_slam_correction_pose_jump():
    sim = Sim(warehouse(), (1.0, 0.8, 0.0))
    sim.loc_error = (0.0, 0.25, 0.0)       # drifted estimate
    sim.nav.start_navigate('g', (10.5, 1.0, 0.0), '', sim.t)
    for _ in range(80):
        sim.step()
    # A loop closure corrects map -> odom: the estimated pose jumps by
    # 0.25 m while odometry stays smooth.
    sim.loc_error = (0.0, 0.0, 0.0)
    _done(sim.run_until_done('g', 120))
    assert math.hypot(sim.pose[0] - 10.5, sim.pose[1] - 1.0) < 0.2
    assert 'POSE_JUMP' in '\n'.join(sim.nav.drain_events())


def test_pause_and_resume():
    sim = Sim(warehouse(), (1.0, 1.0, 0.0))
    sim.nav.start_navigate('g', (6.0, 1.0, 0.0), '', sim.t)
    for _ in range(20):
        sim.step()
    sim.nav.set_paused(True, sim.t)
    for _ in range(5):
        sim.step()
    p = sim.pose
    for _ in range(100):             # 10 s paused: no progress timeout
        sim.step()
    assert sim.pose == p
    sim.nav.set_paused(False, sim.t)
    _done(sim.run_until_done('g', 80))
    assert sim.nav.get_task('g')['state'] == nv.SUCCEEDED


def test_backup_free_and_blocked():
    world = warehouse()
    sim = Sim(world, (2.0, 1.0, 0.0))
    sim.nav.start_backup('b', 0.3, 0.05, 0.0, sim.t)
    _done(sim.run_until_done('b', 30))
    assert sim.pose[0] == pytest.approx(1.7, abs=0.03)
    world.box(1.2, 0.5, 1.3, 1.5)    # wall right behind the robot
    sim.nav.start_backup('c', 0.5, 0.05, 0.0, sim.t)
    snap = sim.run_until_done('c', 30)
    _done(snap, nv.ABORTED)
    assert snap['code'] == nv.BACKUP_COLLISION_AHEAD


def _align_run(omni=False, noise=0.01, start=(4.9, 1.25, 0.35), lose_at=None,
               static=False):
    """Align in front of a box at (6.0, 1.0); perception sees its front face."""
    world = warehouse()
    world.box(6.0, 0.8, 6.3, 1.2)
    sim = Sim(world, start, omni=omni)
    target = (6.0, 1.0, math.pi)                  # x axis points out of the box
    robot_in_target = (0.30, 0.0, math.pi)        # 0.30 m in front, facing it
    rng = np.random.default_rng(3)
    static_t = target if static else None
    sim.nav.start_align('al', robot_in_target, static_t, {}, '', 0.0, sim.t)
    while sim.t < 60.0:
        visible = lose_at is None or sim.t < lose_at
        if not static and visible and int(round(sim.t / sim.dt)) % 2 == 0:
            meas = (target[0] + rng.normal(0, noise), target[1] + rng.normal(0, noise),
                    target[2] + rng.normal(0, 0.01))
            # the manager tracks in odom; here odom == map
            sim.nav.add_target('al', relative(sim.map_to_odom, meas), sim.t)
        sim.step()
        snap = sim.nav.get_task('al')
        if snap['status'] != nv.ACTIVE:
            break
    goal = compose(target, robot_in_target)
    return snap, relative(goal, sim.pose), sim


@pytest.mark.parametrize('omni', [False, True])
def test_alignment_with_noisy_perception(omni):
    snap, err, sim = _align_run(omni=omni)
    _done(snap)
    assert math.hypot(err[0], err[1]) < 0.03
    assert abs(err[2]) < 0.05
    assert sim.min_clearance > -0.005


def test_alignment_static_target_and_failures():
    snap, err, _ = _align_run(static=True, noise=0.0)
    _done(snap)
    snap, _, _ = _align_run(lose_at=1.0)
    _done(snap, nv.ABORTED)
    assert snap['code'] == nv.ALIGN_TARGET_LOST
    snap, _, _ = _align_run(start=(2.5, 1.0, 0.0))
    _done(snap, nv.ABORTED)
    assert snap['code'] == nv.ALIGN_TOO_FAR


def test_compute_path_without_moving():
    sim = Sim(warehouse(), (1.0, 1.0, 0.0))
    r = sim.nav.compute_path((1.0, 1.0), GOAL[:2], GOAL[2], '', sim.t,
                             sim.map_to_odom)
    assert r.ok and r.length > 15.0
    assert sim.pose == (1.0, 1.0, 0.0)


def test_no_map_aborts():
    sim = Sim(warehouse(), (1.0, 1.0, 0.0))
    sim.nav.static_map = None
    sim.map_period = 1e9
    snap = sim.navigate((2.0, 1.0, 0.0), timeout=15)
    _done(snap, nv.ABORTED)
    assert snap['code'] == nv.NAV_NO_MAP


def test_inverse_helper_roundtrip():
    a = (1.0, 2.0, 0.5)
    assert np.allclose(compose(inverse(a), a), (0, 0, 0), atol=1e-12)


def test_threaded_planner_and_astar_backend():
    sim = Sim(warehouse(), (1.0, 1.0, 0.0), threaded=True)
    _done(sim.navigate((10.5, 1.0, 0.0), timeout=120))
    cfg = NavigationConfig(planner_backend='astar')
    sim = Sim(warehouse(), (1.0, 1.0, 0.0), cfg=cfg)
    _done(sim.navigate((7.5, 5.0, 0.0), timeout=120))


def test_start_next_to_a_wall_goal_behind_and_trivial_goal():
    sim = Sim(warehouse(), (2.0, 0.3, math.pi / 2))   # 0.3 m from the wall
    _done(sim.navigate((2.0, 1.2, math.pi), timeout=60))
    p = sim.pose
    _done(sim.navigate((p[0] + 0.05, p[1], p[2]), timeout=5))   # within tolerance
    _done(sim.navigate((0.7, 1.2, 0.0), timeout=60))  # behind the robot
