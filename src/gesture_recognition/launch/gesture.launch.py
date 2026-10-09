"""ros2 launch gesture_recognition gesture.launch.py [params_file:=...]"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(get_package_share_directory("gesture_recognition"), "config", "params.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=default_params),
        Node(
            package="gesture_recognition",
            executable="gesture_node",
            name="gesture_recognition",
            output="screen",
            parameters=[LaunchConfiguration("params_file")],
        ),
    ])
