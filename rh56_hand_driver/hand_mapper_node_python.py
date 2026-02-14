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

        self.get_logger().info("=== Initializing RH56 Hand Driver ===")

        # ================= DDS INIT =================
        if len(sys.argv) > 1:
            ChannelFactoryInitialize(0, sys.argv[1])
        else:
            ChannelFactoryInitialize(0)

        self.pub_r = ChannelPublisher(
            "rt/inspire_hand/ctrl/l",
            inspire_dds.inspire_hand_ctrl
        )
        self.pub_r.Init()

        self.cmd = inspire_hand_defaut.get_inspire_hand_ctrl()

        self.get_logger().info("DDS Channel initialized")

        # ============ OPEN HAND AT START ============
        self.open_hand()

        # ================= ROS SUB ==================
        self.subscription = self.create_subscription(
            HandLandmarks,
            '/hand_landmarks',
            self.landmarks_callback,
            qos_profile_sensor_data
        )

        self.debug_counter = 0
        self.debug_every_n = 10

        self.get_logger().info("Subscribed to /hand_landmarks")
        self.get_logger().info("=== RH56 READY ===")

    # =========================================================

    def open_hand(self):
        """Apre completamente la mano robotica"""
        self.cmd.angle_set = [0, 0, 0, 0, 0, 0]
        self.cmd.mode = 0b0001

        self.get_logger().info("Opening hand...")

        for _ in range(5):
            self.pub_r.Write(self.cmd)
            time.sleep(0.05)

        self.get_logger().info("Hand fully opened.")

    # =========================================================

    def normalize(self, value, min_val, max_val):
        norm = (value - min_val) / (max_val - min_val)
        return np.clip(norm, 0.0, 1.0)

    def to_servo(self, close_norm):
        """
        close_norm:
            0 → dito aperto
            1 → dito chiuso
        output:
            0 → aperto
            1000 → chiuso
        """
        return int(np.clip(close_norm, 0.0, 1.0) * 1000)

    # =========================================================

    def landmarks_callback(self, msg):

        if msg.hand_id != "Left":
            return

        if len(msg.landmarks) != 21:
            self.get_logger().warn("Invalid landmark size")
            return

        self.debug_counter += 1

        def get_tip(idx):
            return np.array([
                msg.landmarks[idx].position.x,
                msg.landmarks[idx].position.y,
                msg.landmarks[idx].position.z
            ])

        wrist = get_tip(0)

        # ===== Distances (thumb → pinky) =====
        thumb_dist  = np.linalg.norm(get_tip(4)  - wrist)
        index_dist  = np.linalg.norm(get_tip(8)  - wrist)
        middle_dist = np.linalg.norm(get_tip(12) - wrist)
        ring_dist   = np.linalg.norm(get_tip(16) - wrist)
        pinky_dist  = np.linalg.norm(get_tip(20) - wrist)

        # ===== Normalize → 0=open, 1=closed =====
        thumb_close  = 1.0 - self.normalize(thumb_dist, 0.05, 0.20)
        index_close  = 1.0 - self.normalize(index_dist, 0.05, 0.25)
        middle_close = 1.0 - self.normalize(middle_dist, 0.05, 0.25)
        ring_close   = 1.0 - self.normalize(ring_dist, 0.05, 0.25)
        pinky_close  = 1.0 - self.normalize(pinky_dist, 0.05, 0.25)

        # ===== Convert to servo =====
        # IMPORTANTE: robot order = pinky → thumb
        angles = [
            self.to_servo(pinky_close),
            self.to_servo(ring_close),
            self.to_servo(middle_close),
            self.to_servo(index_close),
            self.to_servo(thumb_close),
            self.to_servo(thumb_close)  # opposizione
        ]

        # Per proteggere i giunti
        angles = np.clip(angles, 10, 990).tolist()

        self.cmd.angle_set = angles
        self.cmd.mode = 0b0001

        # ================= DEBUG =================
        if self.debug_counter % self.debug_every_n == 0:

            self.get_logger().info("----- HAND DEBUG -----")
            self.get_logger().info(
                f"Distances  T:{thumb_dist:.3f} "
                f"I:{index_dist:.3f} "
                f"M:{middle_dist:.3f} "
                f"R:{ring_dist:.3f} "
                f"P:{pinky_dist:.3f}"
            )
            self.get_logger().info(
                f"CloseNorm  T:{thumb_close:.2f} "
                f"I:{index_close:.2f} "
                f"M:{middle_close:.2f} "
                f"R:{ring_close:.2f} "
                f"P:{pinky_close:.2f}"
            )
            self.get_logger().info(f"[TX] Servo angles: {angles}")
            self.get_logger().info("----------------------")

        # =============== SEND DDS ===============
        if not self.pub_r.Write(self.cmd):
            self.get_logger().warn("Waiting for DDS subscriber...")

# =========================================================

def main(args=None):
    rclpy.init(args=args)
    node = HandMapperNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
