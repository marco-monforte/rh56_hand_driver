from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    controlled_hand_arg = DeclareLaunchArgument(
        "controlled_hand",
        default_value="left",
        description="Hand to control: left, right, or both",
    )

    calibration_arg = DeclareLaunchArgument(
        "calibration",
        default_value="false",
        description="Run mapper calibration at startup",
    )

    input_topic_arg = DeclareLaunchArgument(
        "input_topic",
        default_value="/hand_landmarks",
        description="Input topic for mapped hand landmarks",
    )

    command_topic_arg = DeclareLaunchArgument(
        "command_topic",
        default_value="/rh56dftp/angle_command",
        description="Output topic for RH56 angle commands",
    )

    feedback_topic_arg = DeclareLaunchArgument(
        "feedback_topic",
        default_value="/rh56dftp/feedback",
        description="Feedback topic published by the hand driver",
    )

    network_interface_arg = DeclareLaunchArgument(
        "network_interface",
        default_value="",
        description="Kept for backward compatibility; ignored by the direct Modbus driver",
    )

    reset_errors_on_startup_arg = DeclareLaunchArgument(
        "reset_errors_on_startup",
        default_value="true",
        description="Attempt Modbus TCP error reset before starting control",
    )

    open_hand_on_startup_arg = DeclareLaunchArgument(
        "open_hand_on_startup",
        default_value="true",
        description="Send a full open command at startup, matching the reference Python script",
    )

    left_hand_ip_arg = DeclareLaunchArgument(
        "left_hand_ip",
        default_value="192.168.123.210",
        description="Modbus TCP IP for left hand control and feedback",
    )

    right_hand_ip_arg = DeclareLaunchArgument(
        "right_hand_ip",
        default_value="192.168.123.211",
        description="Modbus TCP IP for right hand control and feedback",
    )

    modbus_port_arg = DeclareLaunchArgument(
        "modbus_port",
        default_value="6000",
        description="Modbus TCP port used for control and feedback",
    )

    left_modbus_unit_id_arg = DeclareLaunchArgument(
        "left_modbus_unit_id",
        default_value="1",
        description="Modbus unit id for left hand startup reset; SDK TCP example uses 1, serial example uses 2",
    )

    right_modbus_unit_id_arg = DeclareLaunchArgument(
        "right_modbus_unit_id",
        default_value="1",
        description="Modbus unit id for right hand startup reset; SDK TCP and serial examples use 1",
    )

    polling_hz_arg = DeclareLaunchArgument(
        "polling_hz",
        default_value="200.0",
        description="Internal Modbus polling rate",
    )

    feedback_rate_hz_arg = DeclareLaunchArgument(
        "feedback_rate_hz",
        default_value="100.0",
        description="ROS feedback publish rate",
    )

    command_rate_hz_arg = DeclareLaunchArgument(
        "command_rate_hz",
        default_value="200.0",
        description="Command write loop rate",
    )

    touch_poll_divider_arg = DeclareLaunchArgument(
        "touch_poll_divider",
        default_value="50",
        description="Read tactile matrices every N polling cycles to reduce Modbus load; set 0 to disable touch polling",
    )

    mapper_node = Node(
        package="rh56_hand_driver",
        executable="hand_mapper_node",
        name="hand_mapper_node",
        output="screen",
        parameters=[{
            "calibration": ParameterValue(LaunchConfiguration("calibration"), value_type=bool),
            "controlled_hand": LaunchConfiguration("controlled_hand"),
            "input_topic": LaunchConfiguration("input_topic"),
            "command_topic": LaunchConfiguration("command_topic"),
        }],
    )

    driver_node = Node(
        package="rh56_hand_driver",
        executable="rh56dftp_driver_node",
        name="rh56dftp_driver_node",
        output="screen",
        parameters=[{
            "controlled_hand": LaunchConfiguration("controlled_hand"),
            "network_interface": LaunchConfiguration("network_interface"),
            "command_topic": LaunchConfiguration("command_topic"),
            "feedback_topic": LaunchConfiguration("feedback_topic"),
            "reset_errors_on_startup": ParameterValue(
                LaunchConfiguration("reset_errors_on_startup"),
                value_type=bool,
            ),
            "open_hand_on_startup": ParameterValue(
                LaunchConfiguration("open_hand_on_startup"),
                value_type=bool,
            ),
            "left_hand_ip": LaunchConfiguration("left_hand_ip"),
            "right_hand_ip": LaunchConfiguration("right_hand_ip"),
            "modbus_port": ParameterValue(LaunchConfiguration("modbus_port"), value_type=int),
            "left_modbus_unit_id": ParameterValue(
                LaunchConfiguration("left_modbus_unit_id"),
                value_type=int,
            ),
            "right_modbus_unit_id": ParameterValue(
                LaunchConfiguration("right_modbus_unit_id"),
                value_type=int,
            ),
            "polling_hz": ParameterValue(LaunchConfiguration("polling_hz"), value_type=float),
            "feedback_rate_hz": ParameterValue(
                LaunchConfiguration("feedback_rate_hz"),
                value_type=float,
            ),
            "command_rate_hz": ParameterValue(
                LaunchConfiguration("command_rate_hz"),
                value_type=float,
            ),
            "touch_poll_divider": ParameterValue(
                LaunchConfiguration("touch_poll_divider"),
                value_type=int,
            ),
        }],
    )

    return LaunchDescription([
        controlled_hand_arg,
        calibration_arg,
        input_topic_arg,
        command_topic_arg,
        feedback_topic_arg,
        network_interface_arg,
        reset_errors_on_startup_arg,
        open_hand_on_startup_arg,
        left_hand_ip_arg,
        right_hand_ip_arg,
        modbus_port_arg,
        left_modbus_unit_id_arg,
        right_modbus_unit_id_arg,
        polling_hz_arg,
        feedback_rate_hz_arg,
        command_rate_hz_arg,
        touch_poll_divider_arg,
        mapper_node,
        driver_node,
    ])
