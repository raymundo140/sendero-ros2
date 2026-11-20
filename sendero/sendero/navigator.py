"""
navigator.py — the navigation manager (ROS-free core)
=====================================================
Owns every navigation task and is the only source of velocity commands.

Task kinds (one active at a time; a new one preempts or is rejected,
depending on `goal_policy`):

  navigate  plan to a goal and follow the path (NavigateToPose)
  backup    drive straight backward (nav2_msgs BackUp)
  align     precision alignment to a perceived target (sendero_msgs Align)

Navigate task phases:

    PLANNING ──path──> FOLLOWING ──goal reached──> SUCCEEDED
       ^  │               │   ^
       │  └─no path─> ABORTED (after plan_failure_tolerance)
       │                  │   │ obstacle cleared / new path
       │                  v   │
       │               BLOCKED ──blocked_wait_s──┐
       │                  ^                       v
       └──after each── RECOVERING <──no progress for progress_timeout_s
                          │
                          └──max_recoveries used──> ABORTED

Also: replanning every replan_period_s, when a new obstacle lies on the
path ahead, when the robot leaves the path, and after a localization jump
(a SLAM loop closure moves map -> odom); a pose lost for longer than
localization_lost_timeout_s aborts the goal.

The node calls tick() at the controller frequency and publishes what it
returns. All public methods are thread-safe.
"""
# Author: Raymundo Gomez

import collections
import json
import math
import threading

import numpy as np

from . import planner as pl
from .aligner import DONE as ALIGN_DONE
from .aligner import FAILED as ALIGN_FAILED
from .aligner import AlignController, TargetTracker
from .config import align_settings, build_profiles
from .controller import (BLOCKED, GOAL_REACHED, OFF_PATH, ROTATING_TO_GOAL,
                         ROTATING_TO_PATH, PathFollower)
from .costmap import (StaticMap, build_global_costmap, build_local_costmap,
                      footprint_samples)
from .geometry import (compose, distance, inverse_transform_points,
                       normalize_angle, transform_points)
from .obstacles import ObstacleBuffer
from .recovery import (BACKUP, CLEAR, SPIN, WAIT, BackUpBehavior,
                       ClearBehavior, RecoverySequence, SpinBehavior,
                       WaitBehavior, free_space_behind)
from .safety import SafetyMonitor, SafetySettings, VelocityLimiter

ZERO = (0.0, 0.0, 0.0)

NAVIGATE = 'navigate'
BACKUP_TASK = 'backup'
ALIGN = 'align'

ACTIVE = 'ACTIVE'
SUCCEEDED = 'SUCCEEDED'
ABORTED = 'ABORTED'
CANCELED = 'CANCELED'

PLANNING = 'PLANNING'
FOLLOWING = 'FOLLOWING'
RECOVERING = 'RECOVERING'

# ── error codes (Nav2 numbering where Nav2 defines one) ──────────────────
NONE = 0
CONTROLLER_UNKNOWN = 100           # nav2_msgs FollowPath
CONTROLLER_TF_ERROR = 102
FAILED_TO_MAKE_PROGRESS = 105
NO_VALID_CONTROL = 106
BACKUP_UNKNOWN = 710               # nav2_msgs BackUp
BACKUP_TIMEOUT = 711
BACKUP_TF_ERROR = 712
BACKUP_INVALID_INPUT = 713
BACKUP_COLLISION_AHEAD = 714
ALIGN_UNKNOWN = 1000               # sendero_msgs Align
ALIGN_TARGET_NOT_FOUND = 1001
ALIGN_TARGET_LOST = 1002
ALIGN_TOO_FAR = 1003
ALIGN_TIMEOUT = 1004
ALIGN_COLLISION = 1005
ALIGN_TF_ERROR = 1006
ALIGN_INVALID_INPUT = 1007
ALIGN_FAILED_TO_CONVERGE = 1008
NAV_UNKNOWN = 9000                 # navigator level
NAV_INVALID_PROFILE = 9001
NAV_TF_ERROR = 9002
NAV_TIMEOUT = 9003
NAV_PREEMPTED = 9050
NAV_NO_MAP = 9051
NAV_INVALID_GOAL = 9052


class Task:
    """One navigation, backup or alignment request."""

    def __init__(self, task_id, kind, profile, now):
        """Create an active task."""
        self.id = task_id
        self.kind = kind
        self.profile = profile
        self.status = ACTIVE
        self.code = NONE
        self.message = ''
        self.state = 'STARTING'
        self.t_start = float(now)
        self.t_end = None
        self.recoveries = 0
        self.feedback = {}
        self.result = {}
        self.nav = None
        self.behavior = None
        self.align = None

    def snapshot(self):
        """Copy of what action servers need (safe to read without a lock)."""
        return {'status': self.status, 'code': self.code,
                'message': self.message, 'state': self.state,
                'feedback': dict(self.feedback), 'result': dict(self.result),
                'kind': self.kind}


class _NavState:
    """Bookkeeping of a navigate task."""

    def __init__(self, goal, recovery):
        self.goal = goal
        self.phase = PLANNING
        self.recovery = recovery
        self.behavior = None
        self.blocked_since = None
        self.lost_since = None
        self.checkpoint = None
        self.last_pose = None
        self.travelled = 0.0
        self.plan_failures = 0
        self.last_plan_request = -math.inf
        self.last_plan_time = -math.inf
        self.reason = 'initial'
        self.remaining = math.nan


class _AlignState:
    """Bookkeeping of an align task."""

    def __init__(self, robot_in_target, static_target, settings, tracker,
                 time_allowance):
        self.robot_in_target = robot_in_target
        self.static = static_target is not None
        self.tracker = tracker
        if static_target is not None:
            tracker.update(static_target, 0.0)
        self.settings = settings
        self.ctrl = AlignController()
        self.time_allowance = time_allowance
        self.distance_checked = False
        self.lost_since = None
        self.goal = None


class _PlanJob:
    """One global planning request, run inline or in a worker thread."""

    def __init__(self, task_id, static_map, points, start, goal, goal_yaw,
                 profile, downsample, clear_center, reason, static_only,
                 marks=None):
        self.task_id = task_id
        self.static_map = static_map
        self.points = points
        self.marks = marks
        self.start = start
        self.goal = goal
        self.goal_yaw = goal_yaw
        self.profile = profile
        self.downsample = downsample
        self.clear_center = clear_center
        self.reason = reason
        self.static_only = static_only
        self.had_obstacles = points is not None and len(points) > 0
        self.grid = None
        self.result = None
        self.done = threading.Event()

    def run(self):
        """Build the global costmap and plan (never raises)."""
        try:
            p = self.profile
            self.grid = build_global_costmap(
                self.static_map, self.points, p.footprint, p.inflation_radius,
                p.cost_scaling_factor, self.downsample, self.clear_center,
                self.marks)
            self.result = pl.plan_path(self.grid, self.start, self.goal,
                                       self.goal_yaw, p.planner)
        except Exception as exc:  # report, do not kill the worker silently
            self.result = pl.PlanResult(False, pl.PLAN_UNKNOWN,
                                        f'planner error: {exc!r}')
        finally:
            self.done.set()


class NavigationManager:
    """The navigation core: tasks, planning, control, recovery, safety."""

    def __init__(self, cfg, profile_overrides=None, logger=None):
        """Create the manager.

        Args:
            cfg: NavigationConfig.
            profile_overrides: dict profile name -> dict of field overrides.
            logger: Callable(level, message) or None.

        Raises:
            ValueError: If the configuration is invalid.
        """
        cfg.validate()
        self.cfg = cfg
        self.profiles = build_profiles(cfg, profile_overrides or {})
        self.active_profile = self.profiles[cfg.default_profile]
        self._log = logger or (lambda level, msg: None)
        self.lock = threading.RLock()

        self.static_map = None
        self.map_version = 0
        keep = max(cfg.global_obstacle_decay_s, cfg.local_obstacle_decay_s,
                   cfg.scan_timeout_s) + 0.5
        self.obstacles = ObstacleBuffer(keep, cfg.obstacle_voxel)
        self.follower = PathFollower()
        self.limiter = VelocityLimiter()
        self.safety = SafetyMonitor()

        self.task = None
        self.paused = False
        self._job = None
        self._events = collections.deque(maxlen=1000)
        self._samples = {}

        self.path = None
        self.path_version = 0
        self.global_grid = None
        self.global_grid_version = 0
        self._published_global_key = None
        self.local_grid = None
        self.last_cmd = ZERO
        self._stop_repeats = 0
        self._last_tick = None
        self._now = 0.0
        self.pose = None
        self.stats = collections.Counter()
        self.last_plan = None
        # Places where the robot was stopped by a predicted collision,
        # (x, y, time) in the global frame. Planning avoids them for
        # blocked_memory_s, so a passage the controller could not get
        # through is not chosen again and again.
        self._blocked_marks = collections.deque(maxlen=50)
        self._safety_stop_since = None
        # Outcomes of recent tasks, so an action server can still read the
        # result of a goal that was preempted by a newer one.
        self._history = collections.OrderedDict()

    # =====================================================================
    # inputs
    # =====================================================================
    def set_map(self, values, resolution, origin_x, origin_y, frame, stamp):
        """Receive a new occupancy grid.

        Args:
            values: int array (H, W) of OccupancyGrid values.
            resolution: [m/cell].
            origin_x, origin_y: Corner of cell (0, 0) in `frame`.
            frame: Map frame.
            stamp: Receive time [s].
        """
        sm = StaticMap(np.asarray(values), resolution, origin_x, origin_y,
                       frame, self.cfg.occupied_threshold,
                       self.cfg.free_threshold, stamp)
        with self.lock:
            self.static_map = sm
            self.map_version += 1

    def add_scan(self, stamp, points_base, base_pose, source=''):
        """Receive one scan.

        Args:
            stamp: Scan time [s].
            points_base: Array (N, 2), obstacle points in the robot base frame.
            base_pose: Pose of the base in the storage frame (odom, or the
                global frame when no odom frame is configured) at `stamp`.
            source: Sensor name.
        """
        pts = np.asarray(points_base, dtype=float).reshape(-1, 2)
        with self.lock:
            fp = self.active_profile.body
            if len(pts):
                pts = pts[~fp.contains(pts, self.cfg.self_filter_margin)]
            self.obstacles.add(stamp, transform_points(base_pose, pts), source)

    def add_target(self, task_id, pose_storage, stamp):
        """Receive a perception measurement of an alignment target.

        Args:
            task_id: The align task it belongs to.
            pose_storage: Target pose (x, y, yaw) in the storage frame.
            stamp: Measurement time [s].

        Returns:
            True if the measurement was accepted.
        """
        with self.lock:
            t = self.task
            if t is None or t.id != task_id or t.kind != ALIGN \
                    or t.status != ACTIVE or t.align.static:
                return False
            return t.align.tracker.update(pose_storage, stamp)

    # =====================================================================
    # tasks
    # =====================================================================
    def resolve_profile(self, name):
        """Return (Profile or None, message) for a requested profile name."""
        name = (name or '').strip()
        if not name:
            return self.profiles[self.cfg.default_profile], ''
        if name in self.profiles:
            return self.profiles[name], ''
        if self.cfg.unknown_profile_policy == 'default':
            return (self.profiles[self.cfg.default_profile],
                    f'unknown profile "{name}", using '
                    f'"{self.cfg.default_profile}"')
        return None, f'unknown profile "{name}"'

    def can_accept(self, profile_name=''):
        """Decide whether a new goal may be accepted (goal callback).

        Returns:
            (accept, reason).
        """
        with self.lock:
            prof, msg = self.resolve_profile(profile_name)
            if prof is None:
                return False, msg
            if (self.cfg.goal_policy == 'reject' and self.task is not None
                    and self.task.status == ACTIVE):
                return False, 'busy: another goal is active (goal_policy=reject)'
            return True, msg

    def _begin(self, task_id, kind, profile, now):
        """Preempt the current task and register a new one."""
        old = self.task
        if old is not None and old.status == ACTIVE:
            code = {NAVIGATE: NAV_PREEMPTED, BACKUP_TASK: BACKUP_UNKNOWN,
                    ALIGN: ALIGN_UNKNOWN}[old.kind]
            self._finish(old, ABORTED, code, 'preempted by a new goal', now)
        task = Task(task_id, kind, profile, now)
        self.task = task
        self.active_profile = profile
        self.limiter.reset(self.last_cmd)
        return task

    def start_navigate(self, task_id, goal, profile_name, now):
        """Start a navigation task.

        Args:
            task_id: Unique ID (any hashable).
            goal: (x, y, yaw) in the global frame.
            profile_name: Profile name ('' = default).
            now: Current time [s].

        Returns:
            The task snapshot.
        """
        with self.lock:
            prof, msg = self.resolve_profile(profile_name)
            if prof is None:
                prof = self.active_profile
            task = self._begin(task_id, NAVIGATE, prof, now)
            if msg:
                self._event('PROFILE_WARNING', {'message': msg}, task)
            if not all(math.isfinite(v) for v in goal):
                self._finish(task, ABORTED, NAV_INVALID_GOAL,
                             'goal contains NaN or inf', now)
                return task.snapshot()
            seq = RecoverySequence(self.cfg.recovery_behaviors,
                                   lambda name: self._make_behavior(name, prof),
                                   self.cfg.max_recoveries)
            task.nav = _NavState(tuple(float(v) for v in goal), seq)
            self.follower.clear()
            self.path = None
            self.path_version += 1
            self._event('GOAL_RECEIVED', {'x': round(goal[0], 3),
                                          'y': round(goal[1], 3),
                                          'yaw': round(goal[2], 3),
                                          'profile': prof.name}, task)
            return task.snapshot()

    def start_backup(self, task_id, dist, speed, time_allowance, now):
        """Start a backup task (drive straight backward).

        Args:
            task_id: Unique ID.
            dist: Distance [m] (sign ignored).
            speed: Speed [m/s] (sign ignored; 0 = backup_speed).
            time_allowance: [s] (0 = recovery_time_allowance_s).
            now: Current time [s].
        """
        with self.lock:
            task = self._begin(task_id, BACKUP_TASK, self.active_profile, now)
            dist = abs(float(dist))
            speed = abs(float(speed)) or self.cfg.backup_speed
            speed = min(speed, abs(self.active_profile.limits.min_vel_x) or speed)
            allowance = float(time_allowance) or self.cfg.recovery_time_allowance_s
            if dist <= 0.0 or not math.isfinite(dist):
                self._finish(task, ABORTED, BACKUP_INVALID_INPUT,
                             'backup distance must be > 0', now)
                return task.snapshot()
            if self.active_profile.limits.min_vel_x >= 0.0:
                self._finish(task, ABORTED, BACKUP_INVALID_INPUT,
                             'reversing is disabled (min_vel_x = 0)', now)
                return task.snapshot()
            task.behavior = BackUpBehavior(dist, speed, allowance,
                                           self.cfg.collision_check_time)
            task.behavior.start(now, None)
            self._event('BACKUP_RECEIVED', {'distance': dist, 'speed': speed},
                        task)
            return task.snapshot()

    def start_align(self, task_id, robot_in_target, static_target, overrides,
                    profile_name, time_allowance, now):
        """Start a precision alignment task.

        Args:
            task_id: Unique ID.
            robot_in_target: Desired robot pose (x, y, yaw) in the target frame.
            static_target: Target pose in the storage frame, or None to track
                it from perception measurements (add_target()).
            overrides: dict of AlignSettings fields from the goal.
            profile_name: Profile ('' = keep the active one).
            time_allowance: [s] (0 = align_time_allowance_s).
            now: Current time [s].
        """
        with self.lock:
            if profile_name:
                prof, _ = self.resolve_profile(profile_name)
                prof = prof or self.active_profile
            else:
                prof = self.active_profile
            task = self._begin(task_id, ALIGN, prof, now)
            settings = align_settings(self.cfg, prof, overrides)
            tracker = TargetTracker(self.cfg.align_target_filter_gain,
                                    self.cfg.align_target_jump_reject)
            task.align = _AlignState(
                tuple(float(v) for v in robot_in_target), static_target,
                settings, tracker,
                float(time_allowance) or self.cfg.align_time_allowance_s)
            self._event('ALIGN_RECEIVED', {
                'static': static_target is not None,
                'robot_in_target': [round(v, 3) for v in robot_in_target]},
                task)
            return task.snapshot()

    def cancel(self, task_id, now):
        """Cancel a task; the robot stops on the next tick."""
        with self.lock:
            t = self.task
            if t is not None and t.id == task_id and t.status == ACTIVE:
                self._finish(t, CANCELED, NONE, 'canceled by the client', now)
                return True
            return False

    def cancel_all(self, now):
        """Cancel whatever task is active."""
        with self.lock:
            t = self.task
            if t is not None and t.status == ACTIVE:
                self._finish(t, CANCELED, NONE, 'canceled (control topic)', now)

    def set_paused(self, paused, now):
        """Hold (True) or release (False) the robot without ending the task."""
        with self.lock:
            if bool(paused) == self.paused:
                return
            self.paused = bool(paused)
            t = self.task
            if not self.paused and t is not None and t.nav is not None:
                # Paused time is not "no progress" and not "blocked".
                if self.pose is not None:
                    t.nav.checkpoint = (self.pose, now)
                t.nav.blocked_since = None
            self._event('PAUSED' if self.paused else 'RESUMED', {}, t)

    def clear_obstacles(self):
        """Forget the sensor obstacles and blocked places (control topic)."""
        with self.lock:
            self.obstacles.clear()
            self._blocked_marks.clear()
            self._event('OBSTACLES_CLEARED', {}, self.task)

    def get_task(self, task_id):
        """Snapshot of a task (current or recently finished), or None."""
        with self.lock:
            t = self.task
            if t is not None and t.id == task_id:
                return t.snapshot()
            return self._history.get(task_id)

    def compute_path(self, start, goal, goal_yaw, profile_name, now,
                     storage_to_global=None):
        """Plan without executing (ComputePathToPose).

        Args:
            start: (x, y) in the global frame.
            goal: (x, y) in the global frame.
            goal_yaw: [rad] or None.
            profile_name: Profile ('' = default).
            now: Current time [s].
            storage_to_global: Pose of the storage frame in the global frame
                (to include recent sensor obstacles), or None.

        Returns:
            PlanResult.
        """
        with self.lock:
            if self.static_map is None:
                return pl.PlanResult(False, pl.PLAN_UNKNOWN, 'no map received')
            prof, msg = self.resolve_profile(profile_name)
            if prof is None:
                return pl.PlanResult(False, pl.PLAN_UNKNOWN, msg)
            sm = self.static_map
            pts = None
            if self.cfg.global_obstacles_enabled and storage_to_global is not None:
                raw = self.obstacles.points(now, self.cfg.global_obstacle_decay_s)
                if len(raw):
                    pts = transform_points(storage_to_global, raw)
        clear = (start[0], start[1], prof.footprint.circumscribed_radius + 0.05)
        job = _PlanJob(None, sm, pts, start, goal, goal_yaw, prof,
                       self.cfg.global_downsample, clear, 'compute_path', False)
        job.run()
        return job.result

    # =====================================================================
    # control loop
    # =====================================================================
    def tick(self, now, pose_global, pose_storage, storage_to_global):
        """Run one control cycle.

        Args:
            now: Current time [s].
            pose_global: Robot pose (x, y, yaw) in the global frame, or None
                if it is unavailable or stale.
            pose_storage: Robot pose in the storage frame (odom), or None.
            storage_to_global: Pose of the storage frame in the global frame,
                or None.

        Returns:
            (vx, vy, wz) to publish, or None when nothing must be published.
        """
        with self.lock:
            dt = 0.0 if self._last_tick is None else min(0.5, max(0.0, now - self._last_tick))
            self._last_tick = now
            self._now = now
            self.pose = pose_global
            self._collect_plan(now)
            task = self.task
            if task is None or task.status != ACTIVE:
                return self._idle_output()
            if self.paused:
                cmd = ZERO
                task.state = 'PAUSED'
            else:
                try:
                    if task.kind == NAVIGATE:
                        cmd = self._tick_navigate(task, now, pose_global,
                                                  pose_storage, storage_to_global)
                    elif task.kind == BACKUP_TASK:
                        cmd = self._tick_backup(task, now, pose_storage)
                    else:
                        cmd = self._tick_align(task, now, pose_storage)
                except Exception as exc:  # stop the robot, report, go on
                    self._log('error', f'navigation tick failed: {exc!r}')
                    code = {NAVIGATE: NAV_UNKNOWN, BACKUP_TASK: BACKUP_UNKNOWN,
                            ALIGN: ALIGN_UNKNOWN}[task.kind]
                    self._finish(task, ABORTED, code, f'internal error: {exc!r}', now)
                    cmd = ZERO
            if task.status != ACTIVE:
                cmd = ZERO
            return self._output(cmd, dt, task, pose_storage, now)

    def _idle_output(self):
        """Publish a few zero commands after a task ends, then nothing."""
        if self._stop_repeats > 0:
            self._stop_repeats -= 1
            self.last_cmd = ZERO
            self.limiter.reset(ZERO)
            return ZERO
        return None

    def _output(self, cmd, dt, task, pose_storage, now):
        """Velocity limiting and the safety monitor."""
        prof = task.profile
        cmd = self.limiter.apply(cmd, prof.limits, dt, prof.omnidirectional)
        settings = prof.safety
        pts_s, age = self.obstacles.latest(now, max(self.cfg.scan_timeout_s, 1.0))
        if task.kind == ALIGN and task.align is not None:
            settings = SafetySettings(
                enabled=prof.safety.enabled, stop_margin=self.cfg.align_stop_margin,
                slow_margin=self.cfg.align_stop_margin, slow_factor=1.0,
                horizon=prof.safety.horizon, scan_timeout=prof.safety.scan_timeout,
                require_scan=True)
            tgt = task.align.tracker.pose
            if tgt is not None and len(pts_s):
                keep = np.hypot(pts_s[:, 0] - tgt[0], pts_s[:, 1] - tgt[1]) \
                    > self.cfg.align_ignore_target_radius
                pts_s = pts_s[keep]
        if pose_storage is not None:
            pts_b = inverse_transform_points(pose_storage, pts_s) if len(pts_s) else pts_s
            safe = self.safety.check(cmd, pts_b, age, prof.body, settings)
        else:
            safe = ZERO
        if safe != cmd:
            self.stats['safety_' + self.safety.state.lower()] += 1
        if self.safety.state in ('STOP', 'NO_SCAN') and safe == ZERO and cmd != ZERO:
            if self._safety_stop_since is None:
                self._safety_stop_since = now
        else:
            self._safety_stop_since = None
        self.limiter.reset(safe)
        self.last_cmd = safe
        self._stop_repeats = 3
        return safe

    # ── navigate ─────────────────────────────────────────────────────────
    def _tick_navigate(self, task, now, pose, pose_s, s2g):
        """One cycle of a navigate task."""
        cfg = self.cfg
        nav = task.nav
        prof = task.profile
        if self.static_map is None:
            task.state = 'WAITING_FOR_MAP'
            if now - task.t_start > cfg.map_wait_timeout_s:
                self._finish(task, ABORTED, NAV_NO_MAP, 'no map received', now)
            return ZERO
        if pose is None:
            if nav.lost_since is None:
                nav.lost_since = now
                self._event('LOCALIZATION_LOST', {}, task)
            task.state = 'LOCALIZATION_LOST'
            if now - nav.lost_since > cfg.localization_lost_timeout_s:
                self._finish(task, ABORTED, NAV_TF_ERROR,
                             'robot pose unavailable (TF)', now)
            return ZERO
        if nav.lost_since is not None:
            self._event('LOCALIZATION_RESTORED',
                        {'after_s': round(now - nav.lost_since, 2)}, task)
            nav.lost_since = None
            nav.checkpoint = (pose, now)
            nav.blocked_since = None
            self._replan(task, pose, now, 'localization_restored', s2g, force=True)
        if nav.checkpoint is None:
            nav.checkpoint = (pose, now)
        if nav.last_pose is not None:
            step = distance(nav.last_pose, pose)
            if step > cfg.pose_jump_replan_m:
                self._event('POSE_JUMP', {'m': round(step, 3)}, task)
                nav.checkpoint = (pose, now)
                self._replan(task, pose, now, 'pose_jump', s2g, force=True)
            else:
                nav.travelled += step
        nav.last_pose = pose
        if cfg.max_goal_time_s > 0.0 and now - task.t_start > cfg.max_goal_time_s:
            self._finish(task, ABORTED, NAV_TIMEOUT,
                         f'goal not reached within {cfg.max_goal_time_s:.0f} s', now)
            return ZERO

        local = self._local_costmap(pose, s2g, now, prof)
        samples = self._samples_for(prof, local.resolution)
        self._update_nav_feedback(task, pose, now)

        if nav.phase == PLANNING:
            task.state = 'PLANNING'
            if self._job is None and now - nav.last_plan_request >= cfg.replan_min_interval_s:
                self._launch_plan(task, pose, now, nav.reason, s2g)
            return ZERO

        if nav.phase == RECOVERING:
            b = nav.behavior
            cmd = b.step(now, pose, local, prof.footprint, samples)
            task.state = 'RECOVERING_' + b.name.upper()
            if b.done:
                self._event('RECOVERY_END', {'behavior': b.name,
                                             'success': b.success,
                                             'message': b.message}, task)
                nav.behavior = None
                nav.phase = PLANNING
                nav.reason = 'after_recovery'
                nav.last_plan_request = -math.inf
                nav.checkpoint = (pose, now)
            return cmd

        out = self.follower.step(pose, self.last_cmd[0], local, prof.footprint,
                                 samples, prof.controller)
        if (out.state != BLOCKED and self._safety_stop_since is not None
                and now - self._safety_stop_since >= cfg.safety_block_s
                and out.cmd != ZERO):
            # The safety monitor keeps stopping a command the costmap allows
            # (an obstacle the costmap misses, or a too-tight passage):
            # handle it as a blocked path.
            out.state = BLOCKED
            out.collision_in = 0.0
            ahead = 0.5 * prof.footprint.circumscribed_radius
            out.collision_pose = (pose[0] + ahead * math.cos(pose[2]),
                                  pose[1] + ahead * math.sin(pose[2]), pose[2])
            out.cmd = ZERO
            self._safety_stop_since = None
        nav.remaining = out.remaining
        self._update_nav_feedback(task, pose, now)
        self._last_out = out
        if out.state == GOAL_REACHED:
            self._finish(task, SUCCEEDED, NONE, 'goal reached', now,
                         {'distance_error': round(out.distance_to_goal, 4)})
            return ZERO
        if out.state == OFF_PATH:
            self._event('OFF_PATH', {'deviation': round(out.deviation, 3)}, task)
            nav.phase = PLANNING
            nav.reason = 'off_path'
            return ZERO
        if out.state == BLOCKED:
            if nav.blocked_since is None:
                nav.blocked_since = now
                self._event('BLOCKED', {'collision_in_s': round(out.collision_in, 2)},
                            task)
                cp = out.collision_pose
                if cp is not None and self.cfg.blocked_memory_s > 0.0 \
                        and distance(cp, nav.goal) > prof.controller.xy_goal_tolerance:
                    self._blocked_marks.append((cp[0], cp[1], now))
                    nav.last_plan_request = -math.inf   # replan right away
            task.state = 'BLOCKED'
            self._replan(task, pose, now, 'blocked', s2g)
            if now - nav.blocked_since >= cfg.blocked_wait_s:
                nav.blocked_since = None
                return self._start_recovery(task, now, pose, local, samples,
                                            'blocked')
            return ZERO
        if nav.blocked_since is not None:
            self._event('UNBLOCKED', {}, task)
            nav.blocked_since = None
        task.state = {ROTATING_TO_PATH: 'ROTATING_TO_PATH',
                      ROTATING_TO_GOAL: 'ROTATING_TO_GOAL'}.get(out.state,
                                                                'FOLLOWING')
        if out.path_blocked:
            self._replan(task, pose, now, 'path_blocked', s2g)
        elif cfg.replan_period_s > 0.0 and \
                now - nav.last_plan_time >= cfg.replan_period_s:
            self._replan(task, pose, now, 'periodic', s2g)

        cp_pose, cp_t = nav.checkpoint
        if (distance(cp_pose, pose) >= cfg.progress_min_distance or
                abs(normalize_angle(pose[2] - cp_pose[2])) >= cfg.progress_min_rotation):
            nav.checkpoint = (pose, now)
        elif now - cp_t > cfg.progress_timeout_s:
            self._event('NO_PROGRESS', {'for_s': round(now - cp_t, 1)}, task)
            nav.checkpoint = (pose, now)
            return self._start_recovery(task, now, pose, local, samples,
                                        'no_progress')
        return out.cmd

    def _update_nav_feedback(self, task, pose, now):
        """Fill the NavigateToPose feedback fields."""
        nav = task.nav
        remaining = nav.remaining
        if not math.isfinite(remaining):
            remaining = distance(pose, nav.goal)
        elapsed = now - task.t_start
        speed = nav.travelled / elapsed if elapsed > 2.0 else 0.0
        speed = max(speed, 0.5 * task.profile.limits.max_vel_x, 0.02)
        task.feedback = {
            'pose': pose, 'navigation_time': elapsed,
            'estimated_time_remaining': remaining / speed,
            'number_of_recoveries': task.recoveries,
            'distance_remaining': remaining,
        }

    def _replan(self, task, pose, now, reason, s2g, force=False):
        """Request a replan (rate limited unless forced)."""
        nav = task.nav
        if nav.phase == PLANNING:
            nav.reason = reason
            return
        if not force and now - nav.last_plan_request < self.cfg.replan_min_interval_s:
            return
        if self._job is not None:
            return
        self._launch_plan(task, pose, now, reason, s2g)

    def _launch_plan(self, task, pose, now, reason, s2g, static_only=False):
        """Start a planning job for the navigate task."""
        cfg = self.cfg
        nav = task.nav
        prof = task.profile
        pts = None
        if cfg.global_obstacles_enabled and not static_only and s2g is not None:
            raw = self.obstacles.points(now, cfg.global_obstacle_decay_s)
            if len(raw):
                pts = transform_points(s2g, raw)
        marks = self._blocked_points(now)
        clear = (pose[0], pose[1], prof.footprint.circumscribed_radius + 0.05)
        job = _PlanJob(task.id, self.static_map, pts, (pose[0], pose[1]),
                       nav.goal[:2], nav.goal[2], prof, cfg.global_downsample,
                       clear, reason, static_only, marks)
        nav.last_plan_request = now
        self._job = job
        self.stats['plans'] += 1
        if cfg.planner_threaded:
            threading.Thread(target=job.run, name='sendero_planner',
                             daemon=True).start()
        else:
            job.run()
            self._collect_plan(now)

    def _collect_plan(self, now):
        """Take the result of a finished planning job."""
        job = self._job
        if job is None or not job.done.is_set():
            return
        self._job = None
        if job.grid is not None:
            self.global_grid = job.grid
            self.global_grid_version += 1
        task = self.task
        r = job.result
        self.last_plan = r
        if task is None or task.id != job.task_id or task.status != ACTIVE \
                or task.kind != NAVIGATE:
            return
        nav = task.nav
        if r.ok:
            nav.plan_failures = 0
            self.follower.set_path(r.path)
            self.path = r.path
            self.path_version += 1
            nav.last_plan_time = now
            if nav.phase == PLANNING:
                nav.phase = FOLLOWING
                if nav.last_pose is not None:
                    nav.checkpoint = (nav.last_pose, now)
            if job.reason != 'periodic':
                self._event('PATH_PLANNED', {
                    'reason': job.reason, 'length_m': round(r.length, 2),
                    'ms': round(1000.0 * r.time_s, 1),
                    'goal_adjusted': r.goal_adjusted,
                    'start_adjusted': r.start_adjusted,
                    'static_only': job.static_only}, task)
            return
        self.stats['plan_failures'] += 1
        self._event('PLAN_FAILED', {'reason': job.reason,
                                    'code': pl.PLAN_CODE_NAMES.get(r.code, r.code),
                                    'message': r.message,
                                    'static_only': job.static_only}, task)
        if (not job.static_only and job.had_obstacles and r.code in
                (pl.NO_VALID_PATH, pl.GOAL_OCCUPIED, pl.START_OCCUPIED)):
            # Blocked by sensor obstacles only? Plan on the map alone; the
            # controller will stop at the obstacle and the recoveries run.
            if nav.last_pose is not None:
                s2g = None  # static only: no obstacles needed
                self._launch_plan(task, nav.last_pose, now, job.reason, s2g,
                                  static_only=True)
            return
        nav.plan_failures += 1
        fatal = r.code in (pl.START_OUTSIDE_MAP, pl.GOAL_OUTSIDE_MAP)
        if fatal or nav.plan_failures >= self.cfg.plan_failure_tolerance:
            self._finish(task, ABORTED, r.code, r.message, now)

    def _blocked_points(self, now):
        """Recent blocked places as points (global frame)."""
        keep = self.cfg.blocked_memory_s
        if keep <= 0.0:
            return np.zeros((0, 2))
        while self._blocked_marks and now - self._blocked_marks[0][2] > keep:
            self._blocked_marks.popleft()
        if not self._blocked_marks:
            return np.zeros((0, 2))
        return np.asarray([(x, y) for x, y, _ in self._blocked_marks])

    def _make_behavior(self, name, prof):
        """Create a recovery behavior by name."""
        cfg = self.cfg
        if name == CLEAR:
            return ClearBehavior(self.obstacles.clear)
        if name == WAIT:
            return WaitBehavior(cfg.recovery_wait_s)
        if name == BACKUP:
            return BackUpBehavior(cfg.backup_distance, cfg.backup_speed,
                                  cfg.recovery_time_allowance_s,
                                  cfg.collision_check_time)
        if name == SPIN:
            return SpinBehavior(cfg.spin_angle, cfg.spin_speed,
                                cfg.recovery_time_allowance_s,
                                cfg.collision_check_time)
        raise ValueError(name)

    def _start_recovery(self, task, now, pose, local, samples, reason):
        """Start the next recovery behavior, or abort when none is left."""
        nav = task.nav
        prof = task.profile
        while True:
            b = nav.recovery.next_behavior()
            task.recoveries = nav.recovery.count
            if b is None:
                code = FAILED_TO_MAKE_PROGRESS if reason == 'no_progress' \
                    else NO_VALID_CONTROL
                self._finish(task, ABORTED, code,
                             f'{reason}: {self.cfg.max_recoveries} recoveries '
                             'did not free the robot', now)
                return ZERO
            if b.name == BACKUP and (prof.limits.min_vel_x >= 0.0 or not
                                     free_space_behind(pose, self.cfg.backup_distance,
                                                       local, prof.footprint, samples)):
                self._event('RECOVERY_SKIPPED', {'behavior': b.name,
                                                 'why': 'no free space behind'}, task)
                continue
            break
        b.start(now, pose)
        nav.behavior = b
        nav.phase = RECOVERING
        self._event('RECOVERY_START', {'behavior': b.name, 'reason': reason,
                                       'number': nav.recovery.count}, task)
        return ZERO

    # ── backup ───────────────────────────────────────────────────────────
    def _tick_backup(self, task, now, pose_s):
        """One cycle of a backup task (in the storage frame)."""
        b = task.behavior
        if pose_s is None:
            task.state = 'LOCALIZATION_LOST'
            if b.start_pose is None and now - task.t_start > self.cfg.localization_lost_timeout_s:
                self._finish(task, ABORTED, BACKUP_TF_ERROR, 'odometry unavailable', now)
            return ZERO
        if b.start_pose is None:
            b.start(now, pose_s)
        local = self._local_costmap(pose_s, None, now, task.profile)
        samples = self._samples_for(task.profile, local.resolution)
        cmd = b.step(now, pose_s, local, task.profile.footprint, samples)
        task.state = 'BACKING_UP'
        task.feedback = {'distance_traveled': b.travelled}
        if b.done:
            if b.success:
                self._finish(task, SUCCEEDED, NONE, b.message, now)
            else:
                code = BACKUP_TIMEOUT if 'timed out' in b.message \
                    else BACKUP_COLLISION_AHEAD
                self._finish(task, ABORTED, code, b.message, now)
            return ZERO
        return cmd

    # ── align ────────────────────────────────────────────────────────────
    def _tick_align(self, task, now, pose_s):
        """One cycle of an align task (in the storage frame)."""
        cfg = self.cfg
        al = task.align
        if pose_s is None:
            task.state = 'LOCALIZATION_LOST'
            if al.lost_since is None:
                al.lost_since = now
            if now - al.lost_since > cfg.localization_lost_timeout_s:
                self._finish(task, ABORTED, ALIGN_TF_ERROR, 'odometry unavailable', now)
            return ZERO
        al.lost_since = None
        tr = al.tracker
        if tr.pose is None:
            task.state = 'WAITING_FOR_TARGET'
            if now - task.t_start > cfg.align_wait_for_target_s:
                self._finish(task, ABORTED, ALIGN_TARGET_NOT_FOUND,
                             'no target measurement received', now)
            return ZERO
        age = 0.0 if al.static else tr.age(now)
        if age > cfg.align_target_lost_s:
            self._finish(task, ABORTED, ALIGN_TARGET_LOST,
                         f'target not seen for {age:.1f} s', now)
            return ZERO
        if now - task.t_start > al.time_allowance:
            self._finish(task, ABORTED, ALIGN_TIMEOUT,
                         f'not aligned within {al.time_allowance:.0f} s', now)
            return ZERO
        goal = compose(tr.pose, al.robot_in_target)
        al.goal = goal
        if not al.distance_checked:
            al.distance_checked = True
            d = distance(pose_s, goal)
            if d > cfg.align_max_start_distance:
                self._finish(task, ABORTED, ALIGN_TOO_FAR,
                             f'target pose is {d:.2f} m away (limit '
                             f'{cfg.align_max_start_distance:.2f} m); '
                             'navigate closer first', now)
                return ZERO
        if age > cfg.align_target_timeout_s:
            # Hold still until perception sees the target again.
            err = AlignController.errors(pose_s, goal)
            task.state = 'ALIGNING_TARGET_STALE'
            task.feedback = {'phase': al.ctrl.phase, 'longitudinal_error': err[0],
                             'lateral_error': err[1], 'yaw_error': err[2],
                             'target_age': age}
            return ZERO
        cmd, err = al.ctrl.step(pose_s, goal, al.settings)
        task.feedback = {'phase': al.ctrl.phase, 'longitudinal_error': err[0],
                         'lateral_error': err[1], 'yaw_error': err[2],
                         'target_age': age}
        task.state = 'ALIGNING_' + al.ctrl.phase
        if al.ctrl.phase == ALIGN_DONE:
            self._finish(task, SUCCEEDED, NONE, 'aligned', now,
                         {'longitudinal_error': err[0], 'lateral_error': err[1],
                          'yaw_error': err[2]})
            return ZERO
        if al.ctrl.phase == ALIGN_FAILED:
            self._finish(task, ABORTED, ALIGN_FAILED_TO_CONVERGE,
                         al.ctrl.message, now,
                         {'longitudinal_error': err[0], 'lateral_error': err[1],
                          'yaw_error': err[2]})
            return ZERO
        return cmd

    # =====================================================================
    # helpers
    # =====================================================================
    def _samples_for(self, prof, resolution):
        """Footprint sample points for a profile and grid resolution (cached)."""
        key = (prof.name, round(resolution, 6))
        s = self._samples.get(key)
        if s is None:
            s = footprint_samples(prof.footprint, resolution)
            self._samples[key] = s
        return s

    def _local_costmap(self, center, s2g, now, prof):
        """Build the rolling local costmap around `center`.

        Args:
            center: Robot pose in the frame the grid is built in.
            s2g: Storage-to-grid-frame pose (None = grid frame is storage).
            now: Current time [s].
            prof: Profile.
        """
        cfg = self.cfg
        raw = self.obstacles.points(now, cfg.local_obstacle_decay_s)
        pts = transform_points(s2g, raw) if (s2g is not None and len(raw)) else raw
        static = self.static_map if (cfg.local_use_static_map and s2g is not None) else None
        frame = cfg.global_frame if (s2g is not None or not cfg.odom_frame) \
            else cfg.odom_frame
        grid = build_local_costmap(center[0], center[1], cfg.local_costmap_size,
                                   cfg.local_costmap_resolution, pts,
                                   prof.footprint, prof.inflation_radius,
                                   prof.cost_scaling_factor, static, frame,
                                   prof.footprint.inscribed_radius
                                   + prof.controller.proximity_slowdown_distance)
        self.local_grid = grid
        return grid

    def _finish(self, task, status, code, message, now, result=None):
        """End a task and record the outcome."""
        if task.status != ACTIVE:
            return
        task.status = status
        task.code = int(code)
        task.message = message
        task.t_end = now
        task.state = status
        if result:
            task.result.update(result)
        if task.nav is not None:
            task.nav.behavior = None
        if task is self.task and task.kind == NAVIGATE:
            self.follower.clear()
        self.stats[f'{task.kind}_{status.lower()}'] += 1
        self._history[task.id] = task.snapshot()
        while len(self._history) > 50:
            self._history.popitem(last=False)
        self._event('GOAL_' + status, {'code': int(code), 'message': message,
                                       'duration_s': round(now - task.t_start, 2)},
                    task)
        self._log('info' if status != ABORTED else 'warn',
                  f'{task.kind} {status.lower()}: {message}')

    def _event(self, name, data, task=None):
        """Queue one event (JSON-serializable)."""
        ev = {'t': round(self._now, 3), 'event': name}
        if task is not None:
            ev['task'] = str(task.id)[:8]
            ev['kind'] = task.kind
        ev.update(data)
        self._events.append(ev)

    def drain_events(self):
        """Return and forget the queued events as JSON strings."""
        with self.lock:
            out = [json.dumps(e, default=str) for e in self._events]
            self._events.clear()
            return out

    # =====================================================================
    # status for the node
    # =====================================================================
    def status_text(self):
        """One-word status for the status topic."""
        with self.lock:
            t = self.task
            if t is None or t.status != ACTIVE:
                return 'IDLE' if self.static_map is not None else 'WAITING_FOR_MAP'
            if self.paused:
                return 'PAUSED'
            return t.state

    def stats_text(self, now):
        """Flat key=value statistics (CSV), like Frontera's."""
        with self.lock:
            t = self.task
            kv = [('state', self.status_text()),
                  ('task', t.kind if t is not None and t.status == ACTIVE else 'none'),
                  ('profile', self.active_profile.name)]
            if t is not None and t.nav is not None and t.status == ACTIVE:
                g = t.nav.goal
                kv += [('goal_x', f'{g[0]:.2f}'), ('goal_y', f'{g[1]:.2f}'),
                       ('distance_remaining',
                        f"{t.feedback.get('distance_remaining', float('nan')):.2f}"),
                       ('recoveries', t.recoveries)]
            if t is not None and t.align is not None and t.status == ACTIVE:
                fb = t.feedback
                kv += [('align_phase', fb.get('phase', '')),
                       ('align_long', f"{fb.get('longitudinal_error', float('nan')):.3f}"),
                       ('align_lat', f"{fb.get('lateral_error', float('nan')):.3f}"),
                       ('align_yaw', f"{fb.get('yaw_error', float('nan')):.3f}")]
            if self.path is not None and len(self.path):
                kv.append(('path_length', f'{self.follower.length:.2f}'))
            if self.last_plan is not None:
                kv += [('last_plan_ms', f'{1000.0 * self.last_plan.time_s:.0f}'),
                       ('last_plan', pl.PLAN_CODE_NAMES.get(self.last_plan.code,
                                                            self.last_plan.code))]
            newest = self.obstacles.newest_stamp()
            scan_age = (now - newest) if newest is not None else float('inf')
            kv += [('scan_age_s', f'{scan_age:.2f}'),
                   ('obstacle_points', len(self.obstacles)),
                   ('safety', self.safety.state),
                   ('closest_obstacle_m', f'{self.safety.closest:.2f}')]
            if self.static_map is not None:
                sm = self.static_map
                kv += [('map_width', sm.width), ('map_height', sm.height),
                       ('map_res', f'{sm.resolution:.3f}'),
                       ('map_age_s', f'{now - sm.stamp:.1f}')]
            c = self.last_cmd
            kv += [('cmd_vx', f'{c[0]:.3f}'), ('cmd_vy', f'{c[1]:.3f}'),
                   ('cmd_wz', f'{c[2]:.3f}'), ('paused', int(self.paused))]
            for k in sorted(self.stats):
                kv.append((k, self.stats[k]))
            return ','.join(f'{k}={v}' for k, v in kv)

    def global_costmap_for_publish(self, now, storage_to_global=None):
        """Return a fresh global costmap for visualization / clients.

        Rebuilt when the map, the active profile or a planner result changed.

        Returns:
            CostGrid or None if nothing changed or no map exists.
        """
        with self.lock:
            if self.static_map is None:
                return None
            key = (self.map_version, self.active_profile.name,
                   self.global_grid_version)
            if key == self._published_global_key:
                return None
            self._published_global_key = key
            sm, prof = self.static_map, self.active_profile
            pts = None
            if self.cfg.global_obstacles_enabled and storage_to_global is not None:
                raw = self.obstacles.points(now, self.cfg.global_obstacle_decay_s)
                if len(raw):
                    pts = transform_points(storage_to_global, raw)
            clear = None
            if self.pose is not None:
                clear = (self.pose[0], self.pose[1],
                         prof.footprint.circumscribed_radius + 0.05)
        return build_global_costmap(sm, pts, prof.footprint, prof.inflation_radius,
                                    prof.cost_scaling_factor,
                                    self.cfg.global_downsample, clear)

    def markers_snapshot(self):
        """Data for RViz markers: footprint, carrot, alignment goal."""
        with self.lock:
            out = {'footprint': self.active_profile.footprint.as_list(),
                   'pose': self.pose}
            lo = getattr(self, '_last_out', None)
            t = self.task
            if lo is not None and t is not None and t.status == ACTIVE \
                    and t.kind == NAVIGATE:
                out['lookahead'] = lo.lookahead
            if t is not None and t.status == ACTIVE and t.align is not None:
                out['align_goal_storage'] = t.align.goal
                out['align_target_storage'] = t.align.tracker.pose
            return out
