"""Launch the Sendero navigation node.

Arguments:
    params_file: Parameter file (default: the package's config/sendero.yaml).
    use_sim_time: 'true' in simulation, so the node follows /clock.

The node publishes no velocity command before it has accepted a goal.
"""
# Author: Raymundo Gomez

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Return the launch description with its two arguments and the node."""
    default_params = os.path.join(
        get_package_share_directory('sendero'), 'config', 'sendero.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file', default_value=default_params,
            description='Sendero parameter file'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Use the /clock topic (simulation)'),
        Node(
            package='sendero',
            executable='navigator',
            name='sendero',
            output='screen',
            parameters=[LaunchConfiguration('params_file'),
                        {'use_sim_time': LaunchConfiguration('use_sim_time')}],
        ),
    ])
