import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from hand_msgs.msg import HandLandmarks
import numpy as np
import sys
import time

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from inspire_sdkpy import inspire_hand_defaut, inspire_dds


class HandMapperNode(Node):

    def __init__(self):
        super().__init__('hand_mapper_node')

        self.get_logger().info("=== RH56 PRO Teleoperation ===")

        # DDS INIT
        if len(sys.argv) > 1:
            ChannelFactoryInitialize(0, sys.argv[1])
        else:
            ChannelFactoryInitialize(0)

        self.pub = ChannelPublisher(
            "rt/inspire_hand/ctrl/l",
            inspire_dds.inspire_hand_ctrl
        )
        self.pub.Init()

        self.cmd = inspire_hand_defaut.get_inspire_hand_ctrl()

        self.open_hand()

        # ROS SUB
        self.subscription = self.create_subscription(
            HandLandmarks,
            '/hand_landmarks',
            self.callback,
            qos_profile_sensor_data
        )

        # Teleop state
        self.min_val = np.full(5, np.inf)
        self.max_val = np.zeros(5)

        self.filtered = np.zeros(5)
        self.alpha = 0.25
        self.deadband = 0.015

        self.debug_counter = 0

        self.get_logger().info("=== READY ===")

    # -----------------------------------------------------

    def open_hand(self):
        self.cmd.angle_set = [0]*6
        self.cmd.mode = 0b0001
        for _ in range(5):
            self.pub.Write(self.cmd)
            time.sleep(0.05)

    # -----------------------------------------------------

    def normalize_dynamic(self, value, idx):

        self.min_val[idx] = min(self.min_val[idx], value)
        self.max_val[idx] = max(self.max_val[idx], value)

        if self.max_val[idx] - self.min_val[idx] < 1e-4:
            return 0.0

        norm = (value - self.min_val[idx]) / (
            self.max_val[idx] - self.min_val[idx]
        )

        return np.clip(1 - norm, 0.0, 1.0)

    # -----------------------------------------------------

    def low_pass(self, new, idx):

        filtered = self.alpha * new + (1 - self.alpha) * self.filtered[idx]

        if abs(filtered - self.filtered[idx]) < self.deadband:
            return self.filtered[idx]

        self.filtered[idx] = filtered
        return filtered

    # -----------------------------------------------------

    def callback(self, msg):

        if msg.hand_id != "Left":
            return

        if len(msg.landmarks) != 21:
            return

        def p(i):
            return np.array([
                msg.landmarks[i].position.x,
                msg.landmarks[i].position.y,
                msg.landmarks[i].position.z
            ])

        wrist = p(0)
        middle_mcp = p(9)

        # Scala mano (compensazione prospettiva)
        hand_scale = np.linalg.norm(middle_mcp - wrist)

        if hand_scale < 1e-6:
            return

        tips = [4, 8, 12, 16, 20]  # thumb → pinky
        closes = []

        for i, tip_idx in enumerate(tips):

            rel_dist = np.linalg.norm(p(tip_idx) - wrist) / hand_scale

            norm = self.normalize_dynamic(rel_dist, i)
            smooth = self.low_pass(norm, i)

            closes.append(smooth)

        # Robot order: pinky → thumb
        servo = [
            int(closes[4]*1000),
            int(closes[3]*1000),
            int(closes[2]*1000),
            int(closes[1]*1000),
            int(closes[0]*1000),
            int(closes[0]*1000)
        ]

        self.cmd.angle_set = servo
        self.cmd.mode = 0b0001

        self.pub.Write(self.cmd)

        # DEBUG
        self.debug_counter += 1
        if self.debug_counter % 20 == 0:
            self.get_logger().info(f"Servo: {servo}")
            self.get_logger().info(f"Filtered close: {np.round(closes,2)}")


# -----------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = HandMapperNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
