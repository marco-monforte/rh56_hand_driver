import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from hand_msgs.msg import HandLandmarks
import numpy as np
import time

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from inspire_sdkpy import inspire_hand_defaut, inspire_dds


class HandMapperNode(Node):

    def __init__(self):
        super().__init__('hand_mapper_node')

        # ================= PARAMETERS =================
        self.declare_parameter("calibration", False)
        self.declare_parameter("dds_domain", 0)

        self.calibration_enabled = self.get_parameter("calibration").value
        dds_domain = self.get_parameter("dds_domain").value

        ChannelFactoryInitialize(dds_domain)

        self.pub = ChannelPublisher(
            "rt/inspire_hand/ctrl/l",
            inspire_dds.inspire_hand_ctrl
        )
        self.pub.Init()

        self.cmd = inspire_hand_defaut.get_inspire_hand_ctrl()

        self.cmd.angle_set = [0, 0, 0, 0, 0, 0]
        self.cmd.mode = 0b0001
        self.pub.Write(self.cmd)

        self.subscription = self.create_subscription(
            HandLandmarks,
            '/hand_landmarks',
            self.landmarks_callback,
            qos_profile_sensor_data
        )

        # ================= CONTROL =================
        self.alpha = 0.25
        self.deadband = 15
        self.filtered_angles = np.zeros(6)

        # ================= CALIBRATION =================
        self.open_close_data = []
        self.opposition_data = []

        self.open_ref = None
        self.close_ref = None
        self.opp_ref = None

        self.debug_counter = 0
        self.debug_every_n = 10

        if self.calibration_enabled:
            self.run_calibration()

        self.get_logger().info("=== RH56 Hand Driver READY ===")

    # =========================================================
    # Utility
    # =========================================================

    def low_pass_filter(self, target):
        self.filtered_angles = (
            self.alpha * target + (1 - self.alpha) * self.filtered_angles
        )
        return self.filtered_angles

    def apply_deadband(self, target):
        delta = np.abs(target - self.filtered_angles)
        mask = delta > self.deadband
        result = self.filtered_angles.copy()
        result[mask] = target[mask]
        return result

    def normalize(self, value, min_val, max_val):
        denom = max_val - min_val
        denom[np.abs(denom) < 1e-6] = 1e-6
        return np.clip((value - min_val) / denom, 0.0, 1.0)

    def progress_bar(self, elapsed, total):
        percent = min(elapsed / total, 1.0)
        bars = int(percent * 20)
        bar = "[" + "#" * bars + "-" * (20 - bars) + "]"
        print(f"\r{bar} {percent*100:5.1f}% ", end="")

    # =========================================================
    # Calibration
    # =========================================================

    def run_calibration(self):

        self.get_logger().warn("=== CALIBRATION MODE ===")

        # -------- OPEN HAND --------
        self.get_logger().warn("Keep hand OPEN")
        input("Press ENTER to start OPEN acquisition (5s)...")

        self.open_close_data = []
        start = time.time()
        while time.time() - start < 5.0:
            self.progress_bar(time.time() - start, 5.0)
            rclpy.spin_once(self, timeout_sec=0.01)
        print()

        self.open_ref = np.mean(self.open_close_data, axis=0)
        self.get_logger().info(f"OPEN reference: {np.round(self.open_ref,3)}")

        # -------- CLOSED HAND --------
        self.get_logger().warn("Keep hand CLOSED")
        input("Press ENTER to start CLOSED acquisition (5s)...")

        self.open_close_data = []
        start = time.time()
        while time.time() - start < 5.0:
            self.progress_bar(time.time() - start, 5.0)
            rclpy.spin_once(self, timeout_sec=0.01)
        print()

        self.close_ref = np.mean(self.open_close_data, axis=0)
        self.get_logger().info(f"CLOSE reference: {np.round(self.close_ref,3)}")

        # -------- THUMB OPPOSITION --------
        self.get_logger().warn("Move thumb in MAXIMUM OPPOSITION")
        input("Press ENTER to start OPPOSITION acquisition (5s)...")

        self.opposition_data = []
        start = time.time()
        while time.time() - start < 5.0:
            self.progress_bar(time.time() - start, 5.0)
            rclpy.spin_once(self, timeout_sec=0.01)
        print()

        self.opp_ref = np.mean(self.opposition_data)
        self.get_logger().info(f"OPPOSITION reference: {self.opp_ref:.4f}")

        self.get_logger().info("=== CALIBRATION COMPLETE ===")
        self.calibration_enabled = False

    # =========================================================
    # Main Callback
    # =========================================================

    def landmarks_callback(self, msg):

        if len(msg.landmarks) != 21:
            return
        if msg.hand_id != "Left":
            return

        def get_pt(i):
            return np.array([
                msg.landmarks[i].position.x,
                msg.landmarks[i].position.y,
                msg.landmarks[i].position.z
            ])

        wrist = get_pt(0)
        palm_size = np.linalg.norm(get_pt(9) - wrist)
        if palm_size < 1e-6:
            return

        # ================= NORMAL FINGERS =================
        index  = np.linalg.norm(get_pt(8)  - wrist) / palm_size
        middle = np.linalg.norm(get_pt(12) - wrist) / palm_size
        ring   = np.linalg.norm(get_pt(16) - wrist) / palm_size
        pinky  = np.linalg.norm(get_pt(20) - wrist) / palm_size

        # ================= THUMB FLEXION =================
        v1 = get_pt(3) - get_pt(2)
        v2 = get_pt(4) - get_pt(3)
        angle = np.arccos(
            np.clip(
                np.dot(v1, v2) /
                (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6),
                -1.0, 1.0
            )
        )

        # ================= THUMB OPPOSITION =================
        opposition = np.linalg.norm(get_pt(4) - get_pt(17)) / palm_size

        # ================= CALIBRATION DATA COLLECTION =================
        if self.calibration_enabled:

            self.open_close_data.append(
                np.array([index, middle, ring, pinky, angle])
            )

            self.opposition_data.append(opposition)
            return

        # ================= NORMALIZATION =================
        fingers_raw = np.array([index, middle, ring, pinky, angle])

        if self.open_ref is not None and self.close_ref is not None:

            fingers_norm = self.normalize(
                fingers_raw,
                self.open_ref,
                self.close_ref
            )

        else:
            fingers_norm = np.clip(fingers_raw, 0, 1)

        thumb_flex = fingers_norm[4]

        if self.opp_ref is not None:
            thumb_opp = np.clip(opposition / self.opp_ref, 0, 1)
        else:
            thumb_opp = np.clip(opposition, 0, 1)

        # ================= SCALE =================
        angles = fingers_norm[:4] * 1000.0
        thumb_close = thumb_flex * 1000.0
        thumb_opp_angle = thumb_opp * 1000.0

        robot_angles = np.array([
            angles[3],  # pinky
            angles[2],  # ring
            angles[1],  # middle
            angles[0],  # index
            thumb_close,
            thumb_opp_angle
        ])

        robot_angles = self.low_pass_filter(robot_angles)
        robot_angles = self.apply_deadband(robot_angles)
        robot_angles = np.clip(robot_angles, 50, 950)

        self.cmd.angle_set = robot_angles.astype(int).tolist()
        self.cmd.mode = 0b0001
        self.pub.Write(self.cmd)

        # ================= DEBUG =================
        self.debug_counter += 1
        if self.debug_counter % self.debug_every_n == 0:

            self.get_logger().info("----- DEBUG -----")
            self.get_logger().info(
                f"Fingers norm: {np.round(fingers_norm,2).tolist()}"
            )
            self.get_logger().info(
                f"Thumb flex: {thumb_flex:.2f} | Thumb opp: {thumb_opp:.2f}"
            )
            self.get_logger().info(
                f"Robot angles: {np.round(robot_angles,2).tolist()}"
            )
            self.get_logger().info("-----------------")


def main(args=None):
    rclpy.init(args=args)
    node = HandMapperNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()