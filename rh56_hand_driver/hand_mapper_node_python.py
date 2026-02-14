import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from hand_msgs.msg import HandLandmarks
import numpy as np
import sys

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from inspire_sdkpy import inspire_hand_defaut, inspire_dds


class HandMapperNode(Node):

    def __init__(self):
        super().__init__('hand_mapper_node')

        self.get_logger().info("=== Initializing RH56 Hand Driver ===")

        # === INIT DDS ===
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

        self.get_logger().info("DDS Channel initialized on rt/inspire_hand/ctrl/l")

        # === ROS SUBSCRIPTION ===
        self.subscription = self.create_subscription(
            HandLandmarks,
            '/hand_landmarks',
            self.landmarks_callback,
            qos_profile_sensor_data   # importante per topic sensori
        )

        self.debug_counter = 0
        self.debug_every_n = 10

        self.get_logger().info("Subscribed to /hand_landmarks")
        self.get_logger().info("=== RH56 Hand Driver READY ===")

    # -------------------------------------------------

    def normalize(self, value, min_val, max_val):
        norm = (value - min_val) / (max_val - min_val)
        return np.clip(norm, 0.0, 1.0)

    def to_short(self, close_norm):
        close_norm = np.clip(close_norm, 0.0, 1.0)
        return int(200 + close_norm * 600)

    # -------------------------------------------------

    def landmarks_callback(self, msg):

        self.debug_counter += 1

        # ===== DEBUG RICEZIONE =====
        # self.get_logger().info(
        #     f"[RX] hand_id={msg.hand_id} | landmarks={len(msg.landmarks)}"
        # )

        if len(msg.landmarks) != 21:
            self.get_logger().warn(
                f"[ERROR] Invalid landmark size: {len(msg.landmarks)}"
            )
            return

        if msg.hand_id != "Left":
            self.get_logger().info(
                f"[SKIP] Ignoring hand_id={msg.hand_id}"
            )
            return

        # ===== PROCESSING =====
        def get_tip(idx):
            return np.array([
                msg.landmarks[idx].position.x,
                msg.landmarks[idx].position.y,
                msg.landmarks[idx].position.z
            ])

        wrist = get_tip(0)

        thumb_dist  = np.linalg.norm(get_tip(4)  - wrist)
        index_dist  = np.linalg.norm(get_tip(8)  - wrist)
        middle_dist = np.linalg.norm(get_tip(12) - wrist)
        ring_dist   = np.linalg.norm(get_tip(16) - wrist)
        pinky_dist  = np.linalg.norm(get_tip(20) - wrist)

        pinky_close  = 1.0 - self.normalize(thumb_dist, 0.05, 0.20)
        ring_close   = 1.0 - self.normalize(index_dist, 0.05, 0.25)
        middle_close = 1.0 - self.normalize(middle_dist, 0.05, 0.25)
        index_close  = 1.0 - self.normalize(ring_dist, 0.05, 0.25)
        thumb_close  = 1.0 - self.normalize(pinky_dist, 0.05, 0.25)

        angles = [
            self.to_short(thumb_close),
            self.to_short(index_close),
            self.to_short(middle_close),
            self.to_short(ring_close),
            self.to_short(pinky_close),
            self.to_short(thumb_close)
        ]

        angles = np.clip(angles, 10, 990).tolist()

        self.cmd.angle_set = angles
        self.cmd.mode = 0b0001

        # ===== DEBUG DETTAGLIATO =====
        if self.debug_counter % self.debug_every_n == 0:

            self.get_logger().info("------ HAND DEBUG ------")

            self.get_logger().info(
                f"Distances: "
                f"T:{thumb_dist:.3f} "
                f"I:{index_dist:.3f} "
                f"M:{middle_dist:.3f} "
                f"R:{ring_dist:.3f} "
                f"P:{pinky_dist:.3f}"
            )

            self.get_logger().info(
                f"Close norm: "
                f"T:{thumb_close:.2f} "
                f"I:{index_close:.2f} "
                f"M:{middle_close:.2f} "
                f"R:{ring_close:.2f} "
                f"P:{pinky_close:.2f}"
            )

            self.get_logger().info(f"[TX] Writing angles: {angles}")
            self.get_logger().info("------------------------")

        # ===== SCRITTURA DDS =====
        if self.pub_r.Write(self.cmd):
            pass
            # self.get_logger().info("[DDS] Command sent successfully")
        else:
            self.get_logger().warn("[DDS] Waiting for hand subscriber...")

# ---------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = HandMapperNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
