"""Frontera + Sendero: autonomous exploration of the simulated warehouse.

Frontera's ROS-free ExplorationController selects frontiers and supervises
goals; Sendero's NavigationManager executes them; a LiDAR-carved map stands
in for SLAM. This is the same goal / feedback / result / cancel exchange the
two ROS nodes have over NavigateToPose, without the transport.

Skipped when Frontera is not importable. From source:
    FRONTERA_SRC=/path/to/frontera-ros2 python3 -m pytest test/ -v
"""
# Author: Raymundo Gomez

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
if os.environ.get('FRONTERA_SRC'):
    sys.path.insert(0, os.environ['FRONTERA_SRC'])

frontera = pytest.importorskip('frontera')

from frontera import navigation_supervisor as ns  # noqa: E402
from frontera.config import ExplorationConfig  # noqa: E402
from frontera.controller import ExplorationController  # noqa: E402
from frontera.grid import GridMap  # noqa: E402
from sim_world import ScanMapper, Sim, warehouse  # noqa: E402

from sendero import navigator as nv  # noqa: E402
from sendero.config import NavigationConfig  # noqa: E402

_RESULT = {nv.SUCCEEDED: ns.RESULT_SUCCEEDED, nv.CANCELED: ns.RESULT_CANCELED,
           nv.ABORTED: ns.RESULT_ABORTED}


class Ports:
    """Frontera's ports, wired to the Sendero simulation."""

    def __init__(self, sim):
        self.sim = sim
        self.events = []
        self.logs = []
        self.saves = 0
        self.pending = []        # (txn_id, task key) waiting for a response
        self.active = {}         # txn_id -> task key
        self.finished = set()
        self.statuses = []

    def ros_now(self): return self.sim.t
    def wall_now(self): return self.sim.t
    def get_pose(self): return self.sim.pose
    def reset_tf(self): pass
    def nav_ready(self): return True

    def nav_send(self, txn_id, x, y, yaw):
        key = f'txn-{txn_id}'
        self.sim.nav.start_navigate(key, (x, y, yaw), '', self.sim.t)
        self.pending.append((txn_id, key))
        return True

    def nav_cancel(self, handle):
        self.sim.nav.cancel(handle, self.sim.t)

    def cmd_rotate(self, wz): pass
    def publish_goal(self, x, y, yaw): pass
    def publish_status(self, text): self.statuses.append(text)
    def publish_stats(self, text): pass
    def publish_event(self, text): self.events.append(json.loads(text))
    def publish_markers(self, items): pass
    def point_sensor(self, bearing): pass
    def request_map_save(self): self.saves += 1
    def publish_map_save_status(self, text): pass
    def log(self, level, text): self.logs.append((level, text))

    def pump(self, ctrl):
        """Deliver goal responses, feedback and results (like the action client)."""
        for txn_id, key in self.pending:
            ctrl.nav_goal_response(txn_id, True, key)
            self.active[txn_id] = key
        self.pending = []
        for txn_id, key in list(self.active.items()):
            snap = self.sim.nav.get_task(key)
            if snap is None:
                continue
            if snap['status'] == nv.ACTIVE:
                fb = snap['feedback']
                if fb:
                    ctrl.nav_feedback(txn_id, fb['distance_remaining'],
                                      fb['number_of_recoveries'])
            else:
                ctrl.nav_result(txn_id, _RESULT[snap['status']])
                del self.active[txn_id]


MAPPING_RANGE = 3.0


def big_warehouse():
    """A 20 m x 14 m warehouse: shelf rows, two side rooms with doors."""
    w = warehouse.__globals__['World'](20.0, 14.0)
    for x0 in (3.0, 6.0, 9.0):
        w.box(x0, 1.6, x0 + 0.6, 5.0)
        w.box(x0, 6.6, x0 + 0.6, 10.0)
    w.box(12.0, 0.0, 12.1, 5.5)           # wall to the right-hand rooms
    w.box(12.0, 6.5, 12.1, 14.0)
    w.box(12.0, 7.0, 20.0, 7.1)           # splits them into two rooms
    w.box(15.0, 7.0, 16.0, 7.1, value=False)   # door between the rooms
    w.box(0.0, 11.5, 8.0, 11.6)           # office strip at the top-left
    w.box(3.5, 11.5, 4.5, 11.6, value=False)
    w.box(16.0, 2.0, 17.0, 3.0)           # pallets in the rooms
    w.box(14.0, 10.0, 15.2, 11.0)
    return w


def test_frontera_explores_the_warehouse_with_sendero():
    world = big_warehouse()
    cfg = NavigationConfig(profile_names=['exploration'],
                           default_profile='exploration')
    over = {'exploration': {'allow_unknown': True, 'ignore_goal_yaw': True}}
    sim = Sim(world, (1.0, 1.0, 0.0), cfg=cfg, overrides=over,
              map_period=1e9)          # the ScanMapper provides the map
    mapper = ScanMapper(world, max_range=MAPPING_RANGE)

    fcfg = ExplorationConfig()
    fcfg.goal_preview_s = 0.0
    fcfg.rotate_first_enabled = False
    fcfg.expected_speed_mps = 0.2
    fcfg.sensor_range_m = 4.0
    fcfg.goal_clearance_m = 0.3
    fcfg.reachability_inflation_m = 0.25
    ports = Ports(sim)
    ctrl = ExplorationController(fcfg, ports)

    def publish_map():
        v = mapper.values
        sim.nav.set_map(v, world.res, 0.0, 0.0, 'map', sim.t)
        ctrl.on_map(GridMap(v.ravel().tolist(), world.w, world.h, world.res,
                            0.0, 0.0))

    mapper.update(sim.pose, world.scan(sim.pose))
    publish_map()
    ctrl.on_command('START')
    last_map = sim.t
    while sim.t < 3000.0 and not ctrl.ended:
        if sim.t - last_map >= 1.0:
            mapper.update(sim.pose, world.scan(sim.pose))
            publish_map()
            last_map = sim.t
        sim.step()
        if int(round(sim.t / sim.dt)) % 2 == 0:     # Frontera at 5 Hz
            ports.pump(ctrl)
            ctrl.tick()
    known_free = np.count_nonzero(mapper.values == 0)
    true_free = np.count_nonzero(~world.occ)
    coverage = known_free / true_free
    names = [e['event'] for e in ports.events]
    reached = names.count('FRONTIER_REACHED') + names.count('FRONTIER_RESOLVED')
    failed = names.count('NAVIGATION_FAILED')
    print(f'\nend={ctrl.end_class} t={sim.t:.0f}s coverage={coverage:.1%} '
          f'reached={reached} failed={failed} saves={ports.saves}')
    assert ctrl.ended, ports.statuses[-3:]
    assert ctrl.end_class in ('COMPLETE', 'COMPLETE_PARTIAL')
    assert coverage > 0.85
    assert ports.saves >= 1
