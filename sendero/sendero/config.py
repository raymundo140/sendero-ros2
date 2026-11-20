"""
config.py — configuration and navigation profiles
=================================================
Every field of NavigationConfig is a ROS 2 parameter of the same name,
documented line by line in config/sendero.yaml. Nothing robot-specific is
hard-coded anywhere else.

A *profile* is a named set of overrides for the fields in PROFILE_FIELDS
(footprint, speed and acceleration limits, inflation, unknown-space
handling, goal tolerances, safety margins). A NavigateToPose client selects
a profile through the goal's `behavior_tree` string, for example
"carrying"; an empty string selects `default_profile`. This keeps the Nav2
action unchanged while letting a program manager switch between, say,
exploring, driving empty and carrying a box. Profiles are declared as
    profiles.<name>.<field>
parameters; any field a profile does not set keeps the base value.
"""
# Author: Raymundo Gomez

import dataclasses
import math
from dataclasses import dataclass, field

from .aligner import AlignSettings
from .controller import DIFFERENTIAL, OMNIDIRECTIONAL, ControllerSettings
from .footprint import Footprint
from .planner import PlannerSettings
from .recovery import KNOWN_BEHAVIORS
from .safety import Limits, SafetySettings


@dataclass
class NavigationConfig:
    """All parameters of the framework (see config/sendero.yaml)."""

    # 1. robot / system interfaces
    global_frame: str = 'map'
    odom_frame: str = 'odom'
    robot_base_frame: str = 'base_link'
    map_topic: str = '/map'
    map_qos_transient_local: bool = True
    scan_topics: list = field(default_factory=lambda: ['/scan'])
    cmd_vel_topic: str = '/cmd_vel'
    cmd_vel_stamped: bool = False
    motion_model: str = DIFFERENTIAL
    navigate_action: str = '/navigate_to_pose'
    compute_path_action: str = '/compute_path_to_pose'
    backup_action: str = '/backup'
    align_action: str = '/align'
    goal_pose_topic: str = '/goal_pose'
    footprint: str = ''
    robot_radius: float = 0.20
    footprint_padding: float = 0.05
    max_vel_x: float = 0.20
    min_vel_x: float = -0.10
    max_vel_y: float = 0.0
    max_vel_theta: float = 1.0
    min_rotation_speed: float = 0.10
    acc_lim_x: float = 0.5
    acc_lim_y: float = 0.5
    acc_lim_theta: float = 1.5
    decel_lim_x: float = 1.0
    decel_lim_y: float = 1.0
    decel_lim_theta: float = 3.0

    # 2. framework topics
    status_topic: str = '/navigation/status'
    stats_topic: str = '/navigation/stats'
    events_topic: str = '/navigation/events'
    control_topic: str = '/navigation/control'
    plan_topic: str = '/navigation/plan'
    global_costmap_topic: str = '/navigation/global_costmap'
    local_costmap_topic: str = '/navigation/local_costmap'
    markers_topic: str = '/navigation/markers'
    global_costmap_publish_period_s: float = 2.0
    local_costmap_publish_period_s: float = 0.5
    status_rate_hz: float = 1.0

    # 3. profiles
    profile_names: list = field(default_factory=lambda: ['default'])
    default_profile: str = 'default'
    unknown_profile_policy: str = 'default'

    # 4. map and costmaps
    occupied_threshold: int = 65
    free_threshold: int = 50
    inflation_radius: float = 0.45
    cost_scaling_factor: float = 6.0
    allow_unknown: bool = True
    unknown_cost: float = 80.0
    global_downsample: int = 1
    global_obstacles_enabled: bool = True
    global_obstacle_decay_s: float = 3.0
    local_costmap_size: float = 4.0
    local_costmap_resolution: float = 0.05
    local_obstacle_decay_s: float = 0.6
    local_use_static_map: bool = False
    obstacle_min_range: float = 0.05
    obstacle_max_range: float = 4.0
    scan_beam_step: int = 1
    obstacle_voxel: float = 0.03
    self_filter_margin: float = 0.02
    scan_tf_tolerance_s: float = 0.2
    use_scan_mount_fallback: bool = False
    scan_mount_fallback: list = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # 5. global planner
    planner_backend: str = 'auto'
    planner_threaded: bool = True
    planner_cost_weight: float = 3.0
    planner_timeout_s: float = 2.0
    goal_search_radius: float = 0.5
    start_search_radius: float = 0.6
    path_smoothing: bool = True
    smoothing_cost_tolerance: float = 10.0
    path_resolution: float = 0.05
    replan_period_s: float = 2.0
    replan_min_interval_s: float = 0.5
    plan_failure_tolerance: int = 2

    # 6. path following
    controller_frequency: float = 10.0
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
    omni_face_path: bool = True
    xy_goal_tolerance: float = 0.15
    yaw_goal_tolerance: float = 0.15
    ignore_goal_yaw: bool = False

    # 7. goal supervision and recovery
    goal_policy: str = 'preempt'
    progress_timeout_s: float = 10.0
    progress_min_distance: float = 0.10
    progress_min_rotation: float = 0.35
    blocked_wait_s: float = 2.0
    blocked_memory_s: float = 20.0
    max_goal_time_s: float = 0.0
    recovery_behaviors: list = field(
        default_factory=lambda: ['clear', 'wait', 'backup', 'spin'])
    max_recoveries: int = 4
    recovery_wait_s: float = 3.0
    backup_distance: float = 0.15
    backup_speed: float = 0.05
    spin_angle: float = 1.57
    spin_speed: float = 0.6
    recovery_time_allowance_s: float = 10.0

    # 8. safety monitor
    safety_enabled: bool = True
    safety_stop_margin: float = 0.02
    safety_slow_margin: float = 0.08
    safety_slow_factor: float = 0.5
    safety_horizon_s: float = 0.6
    safety_block_s: float = 1.0
    scan_timeout_s: float = 0.5

    # 9. precision alignment
    align_max_vel_x: float = 0.08
    align_max_vel_y: float = 0.08
    align_max_vel_theta: float = 0.4
    align_min_vel_x: float = 0.01
    align_min_rotation_speed: float = 0.05
    align_xy_tolerance: float = 0.02
    align_yaw_tolerance: float = 0.035
    align_standoff: float = 0.25
    align_approach_xy_tolerance: float = 0.04
    align_approach_yaw_tolerance: float = 0.10
    align_k_rho: float = 0.6
    align_k_alpha: float = 1.8
    align_k_beta: float = -0.6
    align_k_longitudinal: float = 1.0
    align_k_lateral: float = 3.0
    align_k_heading: float = 1.5
    align_max_reapproaches: int = 2
    align_target_filter_gain: float = 0.3
    align_target_jump_reject: float = 0.3
    align_target_timeout_s: float = 1.0
    align_target_lost_s: float = 5.0
    align_wait_for_target_s: float = 3.0
    align_max_start_distance: float = 1.5
    align_time_allowance_s: float = 60.0
    align_ignore_target_radius: float = 0.35
    align_stop_margin: float = 0.01

    # 10. timing, TF, diagnostics
    pose_timeout_s: float = 0.5
    map_odom_timeout_s: float = 0.0
    localization_lost_timeout_s: float = 3.0
    map_wait_timeout_s: float = 10.0
    pose_jump_replan_m: float = 0.25
    feedback_rate_hz: float = 5.0
    log_events: bool = True

    def validate(self):
        """Check the values; raise ValueError naming the first bad one."""
        def need(cond, msg):
            if not cond:
                raise ValueError(msg)

        need(self.motion_model in (DIFFERENTIAL, OMNIDIRECTIONAL),
             f'motion_model must be "{DIFFERENTIAL}" or "{OMNIDIRECTIONAL}"')
        need(self.goal_policy in ('preempt', 'reject'),
             'goal_policy must be "preempt" or "reject"')
        need(self.unknown_profile_policy in ('default', 'reject'),
             'unknown_profile_policy must be "default" or "reject"')
        need(self.planner_backend in ('auto', 'astar', 'dijkstra'),
             'planner_backend must be "auto", "astar" or "dijkstra"')
        need(self.global_frame != '', 'global_frame must not be empty')
        need(self.robot_base_frame != '', 'robot_base_frame must not be empty')
        need(len(self.scan_topics) > 0 and all(self.scan_topics),
             'scan_topics needs at least one topic')
        need(0 <= self.free_threshold < self.occupied_threshold <= 100,
             'need 0 <= free_threshold < occupied_threshold <= 100')
        for name in ('controller_frequency', 'feedback_rate_hz',
                     'status_rate_hz', 'local_costmap_size',
                     'local_costmap_resolution', 'path_resolution',
                     'robot_radius', 'max_vel_x', 'max_vel_theta',
                     'acc_lim_x', 'acc_lim_theta', 'decel_lim_x',
                     'decel_lim_theta', 'min_lookahead', 'max_lookahead',
                     'xy_goal_tolerance', 'yaw_goal_tolerance',
                     'progress_timeout_s', 'scan_timeout_s',
                     'align_xy_tolerance', 'align_yaw_tolerance',
                     'align_max_vel_x', 'align_max_vel_theta'):
            need(getattr(self, name) > 0.0, f'{name} must be > 0')
        need(self.min_vel_x <= 0.0, 'min_vel_x must be <= 0 (reverse limit)')
        need(self.min_lookahead <= self.max_lookahead,
             'min_lookahead must be <= max_lookahead')
        need(self.global_downsample >= 1, 'global_downsample must be >= 1')
        need(len(self.scan_mount_fallback) == 3,
             'scan_mount_fallback must be [x, y, yaw]')
        need(self.scan_beam_step >= 1, 'scan_beam_step must be >= 1')
        need(self.max_recoveries >= 0, 'max_recoveries must be >= 0')
        bad = [b for b in self.recovery_behaviors if b not in KNOWN_BEHAVIORS]
        need(not bad, f'unknown recovery behaviors {bad}; '
             f'known: {list(KNOWN_BEHAVIORS)}')
        need(0.0 < self.align_target_filter_gain <= 1.0,
             'align_target_filter_gain must be in (0, 1]')
        need(self.align_k_rho > 0.0 and self.align_k_beta < 0.0
             and self.align_k_alpha > self.align_k_rho,
             'alignment gains need k_rho > 0, k_beta < 0, k_alpha > k_rho')
        need(self.default_profile in self.profile_names,
             f'default_profile "{self.default_profile}" is not in '
             f'profile_names {self.profile_names}')
        if self.motion_model == OMNIDIRECTIONAL:
            need(self.max_vel_y > 0.0,
                 'an omnidirectional base needs max_vel_y > 0')


# Fields a profile may override.
PROFILE_FIELDS = (
    'footprint', 'robot_radius', 'footprint_padding',
    'max_vel_x', 'min_vel_x', 'max_vel_y', 'max_vel_theta',
    'acc_lim_x', 'acc_lim_y', 'acc_lim_theta',
    'decel_lim_x', 'decel_lim_y', 'decel_lim_theta',
    'inflation_radius', 'cost_scaling_factor', 'allow_unknown', 'unknown_cost',
    'goal_search_radius', 'xy_goal_tolerance', 'yaw_goal_tolerance',
    'ignore_goal_yaw', 'safety_stop_margin', 'safety_slow_margin',
    'proximity_slowdown_distance', 'approach_distance',
)


@dataclass
class Profile:
    """Resolved settings of one navigation profile."""
    name: str
    values: NavigationConfig
    footprint: Footprint      # padded: costmaps, planning, control
    body: Footprint           # unpadded: self-filter and safety monitor
    limits: Limits
    planner: PlannerSettings
    controller: ControllerSettings
    safety: SafetySettings
    inflation_radius: float
    cost_scaling_factor: float

    @property
    def omnidirectional(self):
        """True for an omnidirectional base."""
        return self.controller.motion_model == OMNIDIRECTIONAL


def build_profile(cfg, name, overrides=None):
    """Resolve a profile from the base config and its overrides.

    Args:
        cfg: NavigationConfig (base values).
        name: Profile name.
        overrides: dict field -> value (only PROFILE_FIELDS).

    Returns:
        Profile.

    Raises:
        ValueError: On an unknown field or invalid value.
    """
    overrides = dict(overrides or {})
    bad = [k for k in overrides if k not in PROFILE_FIELDS]
    if bad:
        raise ValueError(f'profile "{name}": fields {bad} cannot be set per '
                         f'profile; allowed: {list(PROFILE_FIELDS)}')
    v = dataclasses.replace(cfg, **overrides)
    for f in ('max_vel_x', 'max_vel_theta', 'xy_goal_tolerance',
              'yaw_goal_tolerance', 'robot_radius'):
        if getattr(v, f) <= 0.0:
            raise ValueError(f'profile "{name}": {f} must be > 0')
    if v.footprint_padding < 0.0:
        raise ValueError(f'profile "{name}": footprint_padding must be >= 0')
    fp = Footprint.from_config(v.footprint, v.robot_radius, v.footprint_padding)
    body = Footprint.from_config(v.footprint, v.robot_radius, 0.0)
    omni = v.motion_model == OMNIDIRECTIONAL
    limits = Limits(
        max_vel_x=v.max_vel_x, min_vel_x=min(0.0, v.min_vel_x),
        max_vel_y=v.max_vel_y if omni else 0.0, max_vel_theta=v.max_vel_theta,
        acc_lim_x=v.acc_lim_x, acc_lim_y=v.acc_lim_y,
        acc_lim_theta=v.acc_lim_theta, decel_lim_x=v.decel_lim_x,
        decel_lim_y=v.decel_lim_y, decel_lim_theta=v.decel_lim_theta)
    planner = PlannerSettings(
        allow_unknown=v.allow_unknown, unknown_cost=v.unknown_cost,
        cost_weight=v.planner_cost_weight,
        goal_search_radius=v.goal_search_radius,
        start_search_radius=v.start_search_radius,
        timeout_s=v.planner_timeout_s, smooth=v.path_smoothing,
        smoothing_cost_tolerance=v.smoothing_cost_tolerance,
        path_resolution=v.path_resolution, backend=v.planner_backend)
    controller = ControllerSettings(
        motion_model=v.motion_model, max_vel_x=v.max_vel_x,
        max_vel_y=v.max_vel_y if omni else 0.0, max_vel_theta=v.max_vel_theta,
        min_rotation_speed=v.min_rotation_speed,
        lookahead_time=v.lookahead_time, min_lookahead=v.min_lookahead,
        max_lookahead=v.max_lookahead,
        use_rotate_to_heading=v.use_rotate_to_heading,
        rotate_to_heading_angle=v.rotate_to_heading_angle,
        rotation_kp=v.rotation_kp, heading_kp=v.heading_kp,
        regulated_min_radius=v.regulated_min_radius,
        min_regulated_speed=min(v.min_regulated_speed, v.max_vel_x),
        proximity_slowdown_distance=v.proximity_slowdown_distance,
        proximity_min_scale=v.proximity_min_scale,
        approach_distance=max(1e-3, v.approach_distance),
        min_approach_speed=min(v.min_approach_speed, v.max_vel_x),
        collision_check_time=v.collision_check_time,
        path_check_distance=v.path_check_distance,
        max_path_deviation=v.max_path_deviation,
        xy_goal_tolerance=v.xy_goal_tolerance,
        yaw_goal_tolerance=v.yaw_goal_tolerance,
        ignore_goal_yaw=v.ignore_goal_yaw, omni_face_path=v.omni_face_path)
    safety = SafetySettings(
        enabled=v.safety_enabled, stop_margin=v.safety_stop_margin,
        slow_margin=max(v.safety_slow_margin, v.safety_stop_margin),
        slow_factor=v.safety_slow_factor, horizon=v.safety_horizon_s,
        scan_timeout=v.scan_timeout_s, require_scan=True)
    if v.inflation_radius < fp.inscribed_radius:
        raise ValueError(f'profile "{name}": inflation_radius '
                         f'{v.inflation_radius} is below the footprint '
                         f'inscribed radius {fp.inscribed_radius:.3f}')
    return Profile(name, v, fp, body, limits, planner, controller, safety,
                   v.inflation_radius, v.cost_scaling_factor)


def build_profiles(cfg, overrides_by_name):
    """Resolve every profile listed in cfg.profile_names.

    Args:
        cfg: NavigationConfig.
        overrides_by_name: dict name -> dict of overrides.

    Returns:
        dict name -> Profile.
    """
    return {name: build_profile(cfg, name, overrides_by_name.get(name))
            for name in cfg.profile_names}


def align_settings(cfg, profile, overrides=None):
    """AlignSettings from the config, the active profile and goal overrides.

    Args:
        cfg: NavigationConfig.
        profile: Profile (gives the motion model).
        overrides: dict of AlignSettings fields (goal values > 0).

    Returns:
        AlignSettings.
    """
    s = AlignSettings(
        omnidirectional=profile.omnidirectional,
        max_vel_x=cfg.align_max_vel_x,
        max_vel_y=cfg.align_max_vel_y if profile.omnidirectional else 0.0,
        max_vel_theta=cfg.align_max_vel_theta, min_vel_x=cfg.align_min_vel_x,
        min_rotation_speed=cfg.align_min_rotation_speed,
        xy_tolerance=cfg.align_xy_tolerance,
        yaw_tolerance=cfg.align_yaw_tolerance, standoff=cfg.align_standoff,
        approach_xy_tolerance=cfg.align_approach_xy_tolerance,
        approach_yaw_tolerance=cfg.align_approach_yaw_tolerance,
        k_rho=cfg.align_k_rho, k_alpha=cfg.align_k_alpha,
        k_beta=cfg.align_k_beta, k_longitudinal=cfg.align_k_longitudinal,
        k_lateral=cfg.align_k_lateral, k_heading=cfg.align_k_heading,
        rotation_kp=cfg.rotation_kp, allow_reverse=True,
        max_reapproaches=cfg.align_max_reapproaches)
    for k, val in (overrides or {}).items():
        if isinstance(val, bool) or (isinstance(val, (int, float))
                                     and math.isfinite(val) and val > 0):
            setattr(s, k, val)
    return s
