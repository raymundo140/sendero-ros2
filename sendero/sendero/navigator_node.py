#!/usr/bin/env python3
"""
navigator_node.py — Sendero ROS 2 node
======================================
The ROS adapter of the navigation manager: parameters, subscriptions, TF,
action servers and publishers. All navigation logic lives in navigator.py
and the modules it uses, which have no ROS dependency.

Action servers (names configurable):
  /navigate_to_pose       nav2_msgs/action/NavigateToPose   (Nav2-compatible)
  /compute_path_to_pose   nav2_msgs/action/ComputePathToPose
  /backup                 nav2_msgs/action/BackUp
  /align                  sendero_msgs/action/Align

The `behavior_tree` field of NavigateToPose (and `planner_id` of
ComputePathToPose) selects a navigation profile by name; empty selects the
default profile. Any NavigateToPose client, Frontera included, works
unchanged.

Inputs: the occupancy grid (map_topic), LaserScan topics (scan_topics),
TF global_frame -> odom_frame -> robot_base_frame, control commands
(PAUSE, RESUME, CANCEL, CLEAR_COSTMAPS) on control_topic, and goal poses
(RViz "2D Goal Pose") on goal_pose_topic.
Outputs: velocity commands, the global plan, both costmaps, RViz markers,
a one-word status, key=value statistics and JSON events.

The node is safe to launch at startup: it publishes no velocity until it
has accepted a goal.
"""
# Author: Raymundo Gomez

import math
import time

import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import (MutuallyExclusiveCallbackGroup,
                                   ReentrantCallbackGroup)
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy,
                       qos_profile_sensor_data)
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, PoseStamped, Twist, TwistStamped
from nav2_msgs.action import BackUp, ComputePathToPose, NavigateToPose
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from sendero import navigator as nv
from sendero.config import PROFILE_FIELDS, NavigationConfig
from sendero.geometry import (compose, quaternion_from_yaw,
                              yaw_from_quaternion)
from sendero.obstacles import scan_to_points

try:  # the alignment action is optional (sendero_msgs may not be built)
    from sendero_msgs.action import Align
except ImportError:  # pragma: no cover - depends on the workspace
    Align = None


def _coerce(name, value, default):
    """Convert a parameter value to the type of its default.

    YAML writes 1 for 1.0 and [] for an empty list; parameters are read
    loosely and converted here, with a clear error when impossible.
    """
    try:
        if isinstance(default, bool):
            if isinstance(value, str):
                return value.strip().lower() in ('true', '1', 'yes', 'on')
            return bool(value)
        if isinstance(default, int):
            return int(value)
        if isinstance(default, float):
            return float(value)
        if isinstance(default, str):
            return '' if value is None else str(value)
        if isinstance(default, list):
            if value is None:
                return list(default)
            if isinstance(value, (str, bytes)):
                value = [value]
            item = type(default[0]) if default else str
            return [item(v) for v in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f'parameter "{name}": cannot use {value!r}') from exc
    return value


def _to_sec(stamp):
    """builtin_interfaces/Time -> seconds."""
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


def _duration(seconds):
    """Seconds -> builtin_interfaces/Duration."""
    seconds = max(0.0, float(seconds)) if math.isfinite(seconds) else 0.0
    sec = int(seconds)
    return Duration(sec=sec, nanosec=int((seconds - sec) * 1e9))


def _pose_of(transform):
    """geometry_msgs/TransformStamped -> (x, y, yaw)."""
    t = transform.transform
    q = t.rotation
    return (t.translation.x, t.translation.y,
            yaw_from_quaternion(q.x, q.y, q.z, q.w))


def _set_result_error(result, code, message):
    """Fill error_code / error_msg if this distribution's message has them."""
    if hasattr(result, 'error_code'):
        try:
            result.error_code = int(code) & 0xFFFF
        except (TypeError, ValueError):
            pass
    if hasattr(result, 'error_msg'):
        result.error_msg = str(message)


class SenderoNavigator(Node):
    """ROS 2 node around NavigationManager."""

    def __init__(self):
        """Load parameters and create every interface."""
        super().__init__(
            'sendero',
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True)
        self.cfg, overrides = self._load_parameters()
        cfg = self.cfg
        self.manager = nv.NavigationManager(cfg, overrides, self._log)
        self.storage_frame = cfg.odom_frame or cfg.global_frame

        self.tf_buffer = Buffer()
        # TF callbacks run on this node's multi-threaded executor.
        self.tf_listener = TransformListener(self.tf_buffer, self,
                                             spin_thread=False)

        self._control_group = MutuallyExclusiveCallbackGroup()
        self._sensor_group = MutuallyExclusiveCallbackGroup()
        self._io_group = MutuallyExclusiveCallbackGroup()
        self._action_group = ReentrantCallbackGroup()

        latched = QoSProfile(depth=1,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=QoSReliabilityPolicy.RELIABLE)
        map_qos = latched if cfg.map_qos_transient_local else QoSProfile(
            depth=1, reliability=QoSReliabilityPolicy.RELIABLE)

        # ── publishers ───────────────────────────────────────────────────
        cmd_type = TwistStamped if cfg.cmd_vel_stamped else Twist
        self.cmd_pub = self.create_publisher(cmd_type, cfg.cmd_vel_topic, 10)
        self.plan_pub = self.create_publisher(Path, cfg.plan_topic, latched)
        self.global_costmap_pub = self.create_publisher(
            OccupancyGrid, cfg.global_costmap_topic, latched)
        self.local_costmap_pub = self.create_publisher(
            OccupancyGrid, cfg.local_costmap_topic, 1)
        self.marker_pub = self.create_publisher(MarkerArray, cfg.markers_topic, 1)
        self.status_pub = self.create_publisher(String, cfg.status_topic, 10)
        self.stats_pub = self.create_publisher(String, cfg.stats_topic, 10)
        self.events_pub = self.create_publisher(String, cfg.events_topic, 100)

        # ── subscriptions ────────────────────────────────────────────────
        self.create_subscription(OccupancyGrid, cfg.map_topic, self._on_map,
                                 map_qos, callback_group=self._io_group)
        for topic in cfg.scan_topics:
            self.create_subscription(
                LaserScan, topic,
                lambda msg, t=topic: self._on_scan(msg, t),
                qos_profile_sensor_data, callback_group=self._sensor_group)
        self.create_subscription(String, cfg.control_topic, self._on_control,
                                 10, callback_group=self._io_group)
        self._topic_goals = 0
        if cfg.goal_pose_topic:
            self.create_subscription(PoseStamped, cfg.goal_pose_topic,
                                     self._on_goal_pose, 10,
                                     callback_group=self._io_group)

        # ── action servers ───────────────────────────────────────────────
        self._nav_server = ActionServer(
            self, NavigateToPose, cfg.navigate_action,
            execute_callback=self._execute_navigate,
            goal_callback=self._goal_navigate,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=self._action_group)
        if cfg.compute_path_action:
            self._path_server = ActionServer(
                self, ComputePathToPose, cfg.compute_path_action,
                execute_callback=self._execute_compute_path,
                callback_group=self._action_group)
        if cfg.backup_action:
            self._backup_server = ActionServer(
                self, BackUp, cfg.backup_action,
                execute_callback=self._execute_backup,
                goal_callback=lambda _: self._accept(''),
                cancel_callback=lambda _: CancelResponse.ACCEPT,
                callback_group=self._action_group)
        if cfg.align_action:
            if Align is None:
                self.get_logger().warn(
                    'sendero_msgs is not available: the Align action is '
                    'disabled. Build sendero_msgs to enable it.')
            else:
                self._align_server = ActionServer(
                    self, Align, cfg.align_action,
                    execute_callback=self._execute_align,
                    goal_callback=lambda g: self._accept(g.profile),
                    cancel_callback=lambda _: CancelResponse.ACCEPT,
                    callback_group=self._action_group)

        # ── timers ───────────────────────────────────────────────────────
        self._last_path_version = -1
        self._tf_warned = {}
        self.create_timer(1.0 / cfg.controller_frequency, self._control_tick,
                          callback_group=self._control_group)
        self.create_timer(1.0 / cfg.status_rate_hz, self._publish_status,
                          callback_group=self._io_group)
        if cfg.global_costmap_publish_period_s > 0.0:
            self.create_timer(cfg.global_costmap_publish_period_s,
                              self._publish_global_costmap,
                              callback_group=self._io_group)
        if cfg.local_costmap_publish_period_s > 0.0:
            self.create_timer(cfg.local_costmap_publish_period_s,
                              self._publish_local_costmap,
                              callback_group=self._io_group)

        self.get_logger().info(
            f'Sendero ready: {cfg.motion_model} base, profiles '
            f'{list(self.manager.profiles)} (default "{cfg.default_profile}"), '
            f'action {cfg.navigate_action}, cmd {cfg.cmd_vel_topic}')

    # =====================================================================
    # parameters
    # =====================================================================
    def _param(self, name, default):
        """Declare (if needed) and read one parameter, coerced to its type."""
        if self.has_parameter(name):
            value = self.get_parameter(name).value
        else:
            value = self.declare_parameter(name, default).value
        return _coerce(name, value, default)

    def _load_parameters(self):
        """Read the configuration and the profile overrides.

        Returns:
            (NavigationConfig, {profile: {field: value}}).
        """
        defaults = NavigationConfig()
        values = {}
        for f in NavigationConfig.__dataclass_fields__.values():
            values[f.name] = self._param(f.name, getattr(defaults, f.name))
        cfg = NavigationConfig(**values)
        overrides = {}
        # ROS 2 itself declares a few parameters on every node (Jazzy:
        # start_type_description_service); they are not Sendero's.
        known = set(values) | {'use_sim_time', 'start_type_description_service'}
        for name in cfg.profile_names:
            prefix = f'profiles.{name}.'
            prof = {}
            for field in PROFILE_FIELDS:
                pname = prefix + field
                if self.has_parameter(pname):
                    raw = self.get_parameter(pname).value
                    if raw is not None:
                        prof[field] = _coerce(pname, raw, getattr(defaults, field))
                known.add(pname)
            overrides[name] = prof
        for pname in list(getattr(self, '_parameters', {})):
            if pname not in known and not pname.startswith('qos_overrides'):
                self.get_logger().warn(
                    f'unknown parameter "{pname}" ignored (typo, or a profile '
                    'not listed in profile_names?)')
        return cfg, overrides

    # =====================================================================
    # helpers
    # =====================================================================
    def _now(self):
        """Current ROS time [s] (simulation time with use_sim_time)."""
        return self.get_clock().now().nanoseconds * 1e-9

    def _log(self, level, message):
        """Logger used by the manager."""
        getattr(self.get_logger(), {'warn': 'warn', 'error': 'error'}.get(
            level, 'info'))(message)

    def _warn_throttled(self, key, message, period=5.0):
        """Warn at most once per `period` seconds of wall time per key."""
        now = time.monotonic()
        if now - self._tf_warned.get(key, -1e9) >= period:
            self._tf_warned[key] = now
            self.get_logger().warn(message)

    def _lookup(self, target, source, stamp=None):
        """Return (x, y, yaw, stamp_s) of `source` in `target`, or None."""
        try:
            tr = self.tf_buffer.lookup_transform(
                target, source, Time() if stamp is None else stamp)
        except TransformException as exc:
            self._warn_throttled((target, source), f'TF {target} <- {source}: {exc}')
            return None
        x, y, yaw = _pose_of(tr)
        return x, y, yaw, _to_sec(tr.header.stamp)

    def _robot_poses(self, now):
        """Robot pose in the global and storage frames, and storage->global.

        Returns:
            (pose_global, pose_storage, storage_to_global); entries are None
            when unavailable or older than pose_timeout_s.
        """
        cfg = self.cfg
        if not cfg.odom_frame:
            p = self._lookup(cfg.global_frame, cfg.robot_base_frame)
            if p is None or not self._fresh(now, p[3], cfg.pose_timeout_s):
                return None, None, None
            return p[:3], p[:3], (0.0, 0.0, 0.0)
        ob = self._lookup(cfg.odom_frame, cfg.robot_base_frame)
        if ob is None or not self._fresh(now, ob[3], cfg.pose_timeout_s):
            return None, None, None
        mo = self._lookup(cfg.global_frame, cfg.odom_frame)
        if mo is None or not self._fresh(now, mo[3], cfg.map_odom_timeout_s):
            return None, ob[:3], None
        return compose(mo[:3], ob[:3]), ob[:3], mo[:3]

    @staticmethod
    def _fresh(now, stamp, max_age):
        """True if a transform stamp is recent enough (0 = no limit)."""
        if max_age <= 0.0 or stamp == 0.0:
            return True
        return now - stamp <= max_age

    def _to_frame(self, pose_stamped, target, at_time=False):
        """Transform a PoseStamped into `target` -> (x, y, yaw) or None."""
        p = pose_stamped.pose
        local = (p.position.x, p.position.y,
                 yaw_from_quaternion(p.orientation.x, p.orientation.y,
                                     p.orientation.z, p.orientation.w))
        frame = pose_stamped.header.frame_id.lstrip('/') or target
        if frame == target:
            return local
        stamp = Time.from_msg(pose_stamped.header.stamp) if at_time else None
        tr = self._lookup(target, frame, stamp)
        if tr is None and at_time:
            tr = self._lookup(target, frame)
        if tr is None:
            return None
        return compose(tr[:3], local)

    def _accept(self, profile_name):
        """Goal callback shared by the action servers."""
        ok, why = self.manager.can_accept(profile_name)
        if not ok:
            self.get_logger().warn(f'goal rejected: {why}')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    @staticmethod
    def _goal_key(goal_handle):
        """A string ID for an action goal."""
        return bytes(goal_handle.goal_id.uuid).hex()

    def _pose_stamped(self, pose, frame):
        """(x, y, yaw) -> PoseStamped in `frame`, stamped now."""
        msg = PoseStamped()
        msg.header.frame_id = frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(pose[0])
        msg.pose.position.y = float(pose[1])
        q = quaternion_from_yaw(pose[2])
        msg.pose.orientation.x, msg.pose.orientation.y = q[0], q[1]
        msg.pose.orientation.z, msg.pose.orientation.w = q[2], q[3]
        return msg

    def _wait_for(self, goal_handle, key, on_feedback):
        """Run an action goal until the manager finishes it.

        Args:
            goal_handle: rclpy ServerGoalHandle.
            key: Task ID in the manager.
            on_feedback: Callable(snapshot) publishing feedback.

        Returns:
            (snapshot, canceled_by_client).
        """
        period = 1.0 / self.cfg.feedback_rate_hz
        while rclpy.ok():
            snap = self.manager.get_task(key)
            if snap is None:
                return {'status': nv.ABORTED, 'code': nv.NAV_UNKNOWN,
                        'message': 'task lost', 'feedback': {},
                        'result': {}}, False
            if goal_handle.is_cancel_requested and snap['status'] == nv.ACTIVE:
                self.manager.cancel(key, self._now())
                return self.manager.get_task(key), True
            if snap['status'] != nv.ACTIVE:
                return snap, False
            if snap['feedback']:
                on_feedback(snap)
            time.sleep(period)
        self.manager.cancel(key, self._now())
        return self.manager.get_task(key), False

    def _close(self, goal_handle, snap, canceled):
        """Set the terminal state of an action goal from a task snapshot."""
        if snap['status'] == nv.SUCCEEDED:
            goal_handle.succeed()
        elif canceled:
            goal_handle.canceled()
        else:
            goal_handle.abort()

    # =====================================================================
    # inputs
    # =====================================================================
    def _on_map(self, msg):
        """Store a new occupancy grid."""
        info = msg.info
        q = info.origin.orientation
        if abs(yaw_from_quaternion(q.x, q.y, q.z, q.w)) > 1e-3:
            self._warn_throttled('map_yaw', 'map origin is rotated; Sendero '
                                 'assumes an axis-aligned map', 60.0)
        frame = msg.header.frame_id.lstrip('/') or self.cfg.global_frame
        if frame != self.cfg.global_frame:
            self._warn_throttled('map_frame', f'map frame "{frame}" differs '
                                 f'from global_frame "{self.cfg.global_frame}"',
                                 60.0)
        values = np.asarray(msg.data, dtype=np.int16).reshape(info.height,
                                                               info.width)
        self.manager.set_map(values, info.resolution, info.origin.position.x,
                             info.origin.position.y, frame, self._now())

    def _on_scan(self, msg, topic):
        """Convert a scan to obstacle points and hand them to the manager."""
        cfg = self.cfg
        stamp = Time.from_msg(msg.header.stamp)
        stamp_s = stamp.nanoseconds * 1e-9
        now = self._now()
        if stamp_s == 0.0:
            stamp_s, stamp = now, None
        pts = scan_to_points(msg.ranges, msg.angle_min, msg.angle_increment,
                             max(msg.range_min, cfg.obstacle_min_range),
                             msg.range_max, cfg.obstacle_max_range,
                             cfg.scan_beam_step)
        sensor = msg.header.frame_id.lstrip('/') or cfg.robot_base_frame
        bs = self._lookup(cfg.robot_base_frame, sensor)        # static mount
        if bs is None:
            if not cfg.use_scan_mount_fallback:
                return
            bs = tuple(float(v) for v in cfg.scan_mount_fallback)
        pts_base = np.empty_like(pts)
        c, s = math.cos(bs[2]), math.sin(bs[2])
        pts_base[:, 0] = bs[0] + c * pts[:, 0] - s * pts[:, 1]
        pts_base[:, 1] = bs[1] + s * pts[:, 0] + c * pts[:, 1]
        base = None
        if stamp is not None:
            try:
                tr = self.tf_buffer.lookup_transform(self.storage_frame,
                                                     cfg.robot_base_frame, stamp)
                base = _pose_of(tr)
            except TransformException:
                base = None
        if base is None:
            latest = self._lookup(self.storage_frame, cfg.robot_base_frame)
            if latest is None or abs(latest[3] - stamp_s) > cfg.scan_tf_tolerance_s \
                    and latest[3] != 0.0:
                self._warn_throttled(('scan', topic), f'no {self.storage_frame} '
                                     f'pose at the time of a scan on {topic}')
                return
            base = latest[:3]
        self.manager.add_scan(stamp_s, pts_base, base, topic)

    def _on_goal_pose(self, msg):
        """Start a navigation goal from a PoseStamped (RViz 2D Goal Pose)."""
        ok, why = self.manager.can_accept('')
        if not ok:
            self.get_logger().warn(f'goal pose ignored: {why}')
            return
        goal = self._to_frame(msg, self.cfg.global_frame)
        if goal is None:
            self.get_logger().warn('goal pose ignored: TF error')
            return
        self._topic_goals += 1
        self.manager.start_navigate(f'goal_pose_{self._topic_goals}', goal, '',
                                    self._now())
        self.get_logger().info(f'goal pose received: ({goal[0]:.2f}, '
                               f'{goal[1]:.2f}, {goal[2]:.2f})')

    def _on_control(self, msg):
        """PAUSE | RESUME | CANCEL | CLEAR_COSTMAPS (case-insensitive)."""
        cmd = msg.data.strip().upper()
        now = self._now()
        if cmd == 'PAUSE':
            self.manager.set_paused(True, now)
        elif cmd == 'RESUME':
            self.manager.set_paused(False, now)
        elif cmd in ('CANCEL', 'STOP'):
            self.manager.cancel_all(now)
        elif cmd in ('CLEAR_COSTMAPS', 'CLEAR'):
            self.manager.clear_obstacles()
        else:
            self.get_logger().warn(f'unknown control command "{msg.data}"')
            return
        self.get_logger().info(f'control: {cmd}')

    # =====================================================================
    # control loop and outputs
    # =====================================================================
    def _control_tick(self):
        """One control cycle: poses -> manager -> velocity command."""
        now = self._now()
        pose_g, pose_s, s2g = self._robot_poses(now)
        cmd = self.manager.tick(now, pose_g, pose_s, s2g)
        if cmd is not None:
            self._publish_cmd(cmd)
        if self.manager.path_version != self._last_path_version:
            self._last_path_version = self.manager.path_version
            self._publish_plan(self.manager.path)
        if self.cfg.log_events:
            for ev in self.manager.drain_events():
                self.events_pub.publish(String(data=ev))
        else:
            self.manager.drain_events()

    def _publish_cmd(self, cmd):
        """Publish a velocity command (Twist or TwistStamped)."""
        twist = Twist()
        twist.linear.x = float(cmd[0])
        twist.linear.y = float(cmd[1])
        twist.angular.z = float(cmd[2])
        if self.cfg.cmd_vel_stamped:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.cfg.robot_base_frame
            msg.twist = twist
            self.cmd_pub.publish(msg)
        else:
            self.cmd_pub.publish(twist)

    def _publish_plan(self, path):
        """Publish the global plan (an empty path clears it in RViz)."""
        msg = Path()
        msg.header.frame_id = self.cfg.global_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        if path is not None:
            msg.poses = [self._pose_stamped(p, self.cfg.global_frame)
                         for p in path]
        self.plan_pub.publish(msg)

    def _grid_msg(self, grid):
        """CostGrid -> nav_msgs/OccupancyGrid."""
        msg = OccupancyGrid()
        msg.header.frame_id = grid.frame or self.cfg.global_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = float(grid.resolution)
        msg.info.width = int(grid.width)
        msg.info.height = int(grid.height)
        msg.info.origin.position.x = float(grid.origin_x)
        msg.info.origin.position.y = float(grid.origin_y)
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.to_occupancy_values().astype(int).tolist()
        return msg

    def _publish_global_costmap(self):
        """Publish the global costmap when it changed."""
        now = self._now()
        _, _, s2g = self._robot_poses(now)
        grid = self.manager.global_costmap_for_publish(now, s2g)
        if grid is not None:
            self.global_costmap_pub.publish(self._grid_msg(grid))

    def _publish_local_costmap(self):
        """Publish the local costmap and the RViz markers."""
        grid = self.manager.local_grid
        if grid is not None:
            self.local_costmap_pub.publish(self._grid_msg(grid))
        self._publish_markers()

    def _publish_markers(self):
        """Footprint, lookahead point and alignment goal for RViz."""
        snap = self.manager.markers_snapshot()
        stamp = self.get_clock().now().to_msg()
        arr = MarkerArray()
        if snap['pose'] is not None:
            m = Marker()
            m.header.frame_id = self.cfg.global_frame
            m.header.stamp = stamp
            m.ns, m.id, m.type, m.action = 'footprint', 0, Marker.LINE_STRIP, Marker.ADD
            m.scale.x = 0.02
            m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.6, 1.0, 1.0
            m.pose.orientation.w = 1.0
            px, py, pyaw = snap['pose']
            c, s = math.cos(pyaw), math.sin(pyaw)
            pts = list(snap['footprint']) + [snap['footprint'][0]]
            m.points = [Point(x=px + c * x - s * y, y=py + s * x + c * y, z=0.02)
                        for x, y in pts]
            arr.markers.append(m)
        if 'lookahead' in snap:
            m = Marker()
            m.header.frame_id = self.cfg.global_frame
            m.header.stamp = stamp
            m.ns, m.id, m.type, m.action = 'lookahead', 0, Marker.SPHERE, Marker.ADD
            m.scale.x = m.scale.y = m.scale.z = 0.08
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.5, 0.0, 1.0
            m.pose.position.x, m.pose.position.y = snap['lookahead']
            m.pose.orientation.w = 1.0
            arr.markers.append(m)
        goal = snap.get('align_goal_storage')
        if goal is not None:
            m = Marker()
            m.header.frame_id = self.storage_frame
            m.header.stamp = stamp
            m.ns, m.id, m.type, m.action = 'align_goal', 0, Marker.ARROW, Marker.ADD
            m.scale.x, m.scale.y, m.scale.z = 0.25, 0.04, 0.04
            m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 1.0, 0.2, 1.0
            m.pose = self._pose_stamped(goal, self.storage_frame).pose
            arr.markers.append(m)
        if arr.markers:
            self.marker_pub.publish(arr)

    def _publish_status(self):
        """Status (one word) and statistics (key=value CSV)."""
        self.status_pub.publish(String(data=self.manager.status_text()))
        self.stats_pub.publish(String(data=self.manager.stats_text(self._now())))

    # =====================================================================
    # NavigateToPose
    # =====================================================================
    def _goal_navigate(self, goal_request):
        """Accept or reject a NavigateToPose goal."""
        return self._accept(goal_request.behavior_tree)

    def _execute_navigate(self, goal_handle):
        """Run a NavigateToPose goal to its end."""
        req = goal_handle.request
        key = self._goal_key(goal_handle)
        result = NavigateToPose.Result()
        goal = self._to_frame(req.pose, self.cfg.global_frame)
        if goal is None:
            _set_result_error(result, nv.NAV_TF_ERROR,
                              f'cannot transform the goal from '
                              f'"{req.pose.header.frame_id}"')
            goal_handle.abort()
            return result
        self.manager.start_navigate(key, goal, req.behavior_tree, self._now())

        def feedback(snap):
            fb = snap['feedback']
            msg = NavigateToPose.Feedback()
            if fb.get('pose') is not None:
                msg.current_pose = self._pose_stamped(fb['pose'],
                                                      self.cfg.global_frame)
            msg.navigation_time = _duration(fb.get('navigation_time', 0.0))
            msg.estimated_time_remaining = _duration(
                fb.get('estimated_time_remaining', 0.0))
            msg.number_of_recoveries = int(fb.get('number_of_recoveries', 0))
            dr = fb.get('distance_remaining', 0.0)
            msg.distance_remaining = float(dr) if math.isfinite(dr) else 0.0
            goal_handle.publish_feedback(msg)

        snap, canceled = self._wait_for(goal_handle, key, feedback)
        _set_result_error(result, snap['code'], snap['message'])
        self._close(goal_handle, snap, canceled)
        return result

    # =====================================================================
    # ComputePathToPose
    # =====================================================================
    def _execute_compute_path(self, goal_handle):
        """Plan a path without moving the robot."""
        req = goal_handle.request
        result = ComputePathToPose.Result()
        now = self._now()
        pose_g, _, s2g = self._robot_poses(now)
        goal = self._to_frame(req.goal, self.cfg.global_frame)
        start = (self._to_frame(req.start, self.cfg.global_frame)
                 if req.use_start else pose_g)
        if goal is None or start is None:
            _set_result_error(result, 202, 'TF error: start or goal unknown')
            goal_handle.abort()
            return result
        r = self.manager.compute_path(start[:2], goal[:2], goal[2],
                                      req.planner_id, now, s2g)
        result.planning_time = _duration(r.time_s)
        _set_result_error(result, r.code, r.message)
        if r.ok:
            result.path.header.frame_id = self.cfg.global_frame
            result.path.header.stamp = self.get_clock().now().to_msg()
            result.path.poses = [self._pose_stamped(p, self.cfg.global_frame)
                                 for p in r.path]
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return result

    # =====================================================================
    # BackUp
    # =====================================================================
    def _execute_backup(self, goal_handle):
        """Drive straight backward by |target.x| metres."""
        req = goal_handle.request
        key = self._goal_key(goal_handle)
        t0 = self._now()
        allowance = req.time_allowance.sec + 1e-9 * req.time_allowance.nanosec
        self.manager.start_backup(key, req.target.x, req.speed, allowance, t0)

        def feedback(snap):
            msg = BackUp.Feedback()
            msg.distance_traveled = float(snap['feedback'].get('distance_traveled', 0.0))
            goal_handle.publish_feedback(msg)

        snap, canceled = self._wait_for(goal_handle, key, feedback)
        result = BackUp.Result()
        result.total_elapsed_time = _duration(self._now() - t0)
        _set_result_error(result, snap['code'], snap['message'])
        self._close(goal_handle, snap, canceled)
        return result

    # =====================================================================
    # Align
    # =====================================================================
    def _execute_align(self, goal_handle):
        """Precision alignment to a perceived (or fixed) target."""
        req = goal_handle.request
        key = self._goal_key(goal_handle)
        result = Align.Result()
        p = req.robot_pose_in_target
        robot_in_target = (p.position.x, p.position.y,
                           yaw_from_quaternion(p.orientation.x, p.orientation.y,
                                               p.orientation.z, p.orientation.w))
        overrides = {'xy_tolerance': req.xy_tolerance,
                     'yaw_tolerance': req.yaw_tolerance,
                     'max_vel_x': req.max_linear_speed}
        if req.forward_only:
            overrides['allow_reverse'] = False
        topic = req.target_topic.strip()
        static_target = None
        if not topic:
            if not req.target_pose.header.frame_id:
                result.error_code = Align.Result.INVALID_INPUT
                result.error_msg = 'set target_topic or target_pose.header.frame_id'
                goal_handle.abort()
                return result
            static_target = self._to_frame(req.target_pose, self.storage_frame)
            if static_target is None:
                result.error_code = Align.Result.TF_ERROR
                result.error_msg = 'cannot transform target_pose'
                goal_handle.abort()
                return result
        self.manager.start_align(key, robot_in_target, static_target, overrides,
                                 req.profile, req.time_allowance, self._now())
        sub = None
        if topic:
            if req.target_pose.header.frame_id:
                first = self._to_frame(req.target_pose, self.storage_frame)
                if first is not None:
                    self.manager.add_target(key, first, self._now())

            def on_target(msg, k=key):
                pose = self._to_frame(msg, self.storage_frame, at_time=True)
                if pose is not None:
                    stamp = _to_sec(msg.header.stamp) or self._now()
                    self.manager.add_target(k, pose, stamp)

            sub = self.create_subscription(PoseStamped, topic, on_target, 10,
                                           callback_group=self._action_group)

        def feedback(snap):
            fb = snap['feedback']
            msg = Align.Feedback()
            msg.phase = str(fb.get('phase', ''))
            msg.longitudinal_error = float(fb.get('longitudinal_error', 0.0))
            msg.lateral_error = float(fb.get('lateral_error', 0.0))
            age = fb.get('target_age', 0.0)
            msg.yaw_error = float(fb.get('yaw_error', 0.0))
            msg.target_age = float(age) if math.isfinite(age) else -1.0
            goal_handle.publish_feedback(msg)

        try:
            snap, canceled = self._wait_for(goal_handle, key, feedback)
        finally:
            if sub is not None:
                self.destroy_subscription(sub)
        res = snap.get('result', {})
        result.error_code = int(snap['code']) & 0xFFFF
        result.error_msg = snap['message']
        result.longitudinal_error = float(res.get('longitudinal_error', 0.0))
        result.lateral_error = float(res.get('lateral_error', 0.0))
        result.yaw_error = float(res.get('yaw_error', 0.0))
        self._close(goal_handle, snap, canceled)
        return result

    # =====================================================================
    def stop_robot(self):
        """Publish zero velocity (shutdown)."""
        try:
            for _ in range(3):
                self._publish_cmd((0.0, 0.0, 0.0))
        except Exception:  # the context may already be gone
            pass


def main(args=None):
    """Entry point: spin the node on a multi-threaded executor."""
    rclpy.init(args=args)
    node = SenderoNavigator()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    external = getattr(rclpy.executors, 'ExternalShutdownException',
                       KeyboardInterrupt)
    try:
        executor.spin()
    except (KeyboardInterrupt, external):
        pass
    finally:
        # A second Ctrl+C during the shutdown must not print a traceback.
        try:
            if node.manager.task is not None and node.manager.task.status == nv.ACTIVE:
                node.stop_robot()
            executor.shutdown()
            node.destroy_node()
        except (KeyboardInterrupt, external):
            pass
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == '__main__':
    main()
