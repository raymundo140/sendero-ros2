"""
Sendero — supervised navigation for ROS 2
=========================================
A Nav2-compatible navigation framework: NavigateToPose, ComputePathToPose
and BackUp action servers, plus a precision-alignment action, built on a
layered costmap, a cost-aware grid planner, a Regulated Pure Pursuit
controller, bounded recoveries and an independent safety monitor.

Module map (all plain Python, no ROS dependency, except the node):

  geometry.py         SE(2) helpers
  footprint.py        circle / polygon footprint, signed distance
  costmap.py          static + obstacle layers, inflation, collision checks
  obstacles.py        scan -> points, time-decayed obstacle history
  planner.py          A* / Dijkstra on the costmap, smoothing, resampling
  controller.py       Regulated Pure Pursuit (differential / omnidirectional)
  safety.py           velocity / acceleration limits, raw-scan safety monitor
  recovery.py         clear, wait, backup, spin
  aligner.py          target tracking and precision alignment
  config.py           all parameters and navigation profiles
  navigator.py        the navigation manager (tasks, supervision, recovery)
  navigator_node.py   the ROS 2 node (parameters, topics, TF, actions)
"""
# Author: Raymundo Gomez

__version__ = '1.0.0'
