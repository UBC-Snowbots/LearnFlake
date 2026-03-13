import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    rname = LaunchConfiguration("rname")
    world_name = LaunchConfiguration("world")
    joy_topic = LaunchConfiguration("joy_topic")

    robot_desc_pkg = get_package_share_directory("rover2026_description")
    urdf_file = os.path.join(robot_desc_pkg, "urdf", "rover2026.urdf")
    with open(urdf_file, "r", encoding="utf-8") as f:
        robot_description_content = f.read()

    robot_description = ParameterValue(robot_description_content, value_type=str)
    gazebo_model_name = ParameterValue([rname, TextSubstitution(text="_gazebo")], value_type=str)
    robot_name = ParameterValue(rname, value_type=str)

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("gazebo_ros"), "launch", "gazebo.launch.py")
        ),
        launch_arguments={
            "world": os.path.join(get_package_share_directory("rl_sar"), "worlds", world_name),
        }.items(),
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}],
        remappings=[("/joint_states", "/rover2026_gazebo/joint_states")],
    )

    spawn_entity = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-topic", "/robot_description",
            "-entity", "rover2026_gazebo",
            "-z", "1.0",
        ],
        output="screen",
    )

    joint_state_broadcaster_node = Node(
        package="controller_manager",
        executable="spawner.py" if os.environ.get("ROS_DISTRO", "") == "foxy" else "spawner",
        arguments=["joint_state_broadcaster"],
        output="screen",
    )

    joy_node = Node(
        package="joy",
        executable="joy_node",
        name="joy_node",
        output="screen",
        remappings=[("/joy", joy_topic)],
        parameters=[{"deadzone": 0.1, "autorepeat_rate": 0.0}],
    )

    param_node = Node(
        package="demo_nodes_cpp",
        executable="parameter_blackboard",
        name="param_node",
        parameters=[{
            "robot_name": robot_name,
            "gazebo_model_name": gazebo_model_name,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument("rname", default_value=TextSubstitution(text="rover2026")),
        DeclareLaunchArgument("world", default_value=TextSubstitution(text="stairs.world")),
        DeclareLaunchArgument("joy_topic", default_value=TextSubstitution(text="/joy")),
        robot_state_publisher_node,
        gazebo,
        spawn_entity,
        joint_state_broadcaster_node,
        joy_node,
        param_node,
    ])

