from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare arguments
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            "debug",
            default_value="false",
            description="Enable debug output for the particle filter."
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "viz",
            default_value="false",
            description="Enable visualization tools for the filter."
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "pose_pub_topic",
            default_value="/tracked_pose",
            description="Topic name for publishing tracked poses."
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "scan_topic",
            default_value="/autodrive/roboracer_1/lidar",
            description="Laser scan topic to use for measurements."
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "odometry_topic",
            default_value="/odom_rf2o",
            description="Odometry topic for motion prediction."
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "initial_pose_x",
            default_value="0.7406",
            description="Initial X position for particle filter."
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "initial_pose_y",
            default_value="3.1583",
            description="Initial Y position for particle filter."
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "initial_pose_theta",
            default_value="-1.5707963",
            description="Initial orientation (theta) for particle filter."
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "publish_tf",
            default_value="False",
            description="Whether to publish transform."
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "pub_covariance",
            default_value="False",
            description="Publish covariance matrix."
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "publish_pose",
            default_value="True",
            description="Enable pose publication."
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "publish_odom",
            default_value="False",
            description="Enable odometry publication."
        )
    )

    # Configuration variables
    debug = LaunchConfiguration("debug")
    viz = LaunchConfiguration("viz")
    pose_pub_topic = LaunchConfiguration("pose_pub_topic")
    scan_topic = LaunchConfiguration("scan_topic")
    odometry_topic = LaunchConfiguration("odometry_topic")
    initial_pose_x = LaunchConfiguration("initial_pose_x")
    initial_pose_y = LaunchConfiguration("initial_pose_y")
    initial_pose_theta = LaunchConfiguration("initial_pose_theta")
    publish_tf = LaunchConfiguration("publish_tf")
    pub_covariance = LaunchConfiguration("pub_covariance")
    publish_pose = LaunchConfiguration("publish_pose")
    publish_odom = LaunchConfiguration("publish_odom")

    # Particle Filter Node
    particle_filter_node = Node(
        package="particle_filter",
        executable="synPF",
        name="particle_filter",
        output="screen",
        parameters=[
            PathJoinSubstitution([
                FindPackageShare("particle_filter"),
                "config",
                "pf2_params.yaml"
            ]),
            {
                "debug": debug,
                "viz": viz,
                "pose_pub_topic": pose_pub_topic,
                "scan_topic": scan_topic,
                "odometry_topic": odometry_topic,
                "initial_pose_x": initial_pose_x,
                "initial_pose_y": initial_pose_y,
                "initial_pose_theta": initial_pose_theta,
                "publish_tf": publish_tf,
                "pub_covariance": pub_covariance,
                "publish_pose": publish_pose,
                "publish_odom": publish_odom
            }
        ]
    )

    return LaunchDescription(
        declared_arguments +
        [
            particle_filter_node,
        ]
    )
