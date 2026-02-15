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

        self.get_logger().info("=== Initializing RH56 Hand Driver ===")

        # ================= DDS INIT =================
        ChannelFactoryInitialize(dds_domain)

        self.pub = ChannelPublisher(
            "rt/inspire_hand/ctrl/l",
            inspire_dds.inspire_hand_ctrl
        )
        self.pub.Init()

        self.cmd = inspire_hand_defaut.get_inspire_hand_ctrl()

        # Open hand at startup
        self.cmd.angle_set = [0, 0, 0, 0, 0, 0]
        self.cmd.mode = 0b0001
        self.pub.Write(self.cmd)

        self.get_logger().info("Hand opened. Waiting for landmarks...")

        # ================= ROS SUB =================
        self.subscription = self.create_subscription(
            HandLandmarks,
            '/hand_landmarks',
            self.landmarks_callback,
            qos_profile_sensor_data
        )

        # ================= CONTROL =================
        self.alpha = 0.2
        self.deadband = 15
        self.filtered_angles = np.zeros(6)

        # ================= CALIBRATION =================
        self.calibration_phase = 0
        self.calibration_start = None
        self.calibration_data = []
        self.open_norm = None
        self.close_norm = None

        if self.calibration_enabled:
            self.start_calibration()

        self.debug_counter = 0
        self.debug_every_n = 10

        self.get_logger().info("=== RH56 Hand Driver READY ===")

    # =========================================================
    # Utility
    # =========================================================

    def normalize(self, value, min_val, max_val):
        return np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0)

    def scale_to_angle(self, norm):
        return int(np.clip(norm * 1000.0, 0, 1000))

    def low_pass_filter(self, target):
        self.filtered_angles = self.alpha * target + (1 - self.alpha) * self.filtered_angles
        return self.filtered_angles

    def apply_deadband(self, target):
        delta = np.abs(target - self.filtered_angles)
        mask = delta > self.deadband
        result = self.filtered_angles.copy()
        result[mask] = target[mask]
        return result

    # =========================================================
    # ASCII Progress + Countdown
    # =========================================================

    def progress_bar(self, elapsed, total):
        percent = min(elapsed / total, 1.0)
        bars = int(percent * 20)
        bar = "[" + "#" * bars + "-" * (20 - bars) + "]"
        countdown = int(total - elapsed)
        print(f"\r{bar} {percent*100:5.1f}%  {countdown}s ", end="")

    # =========================================================
    # Calibration
    # =========================================================

    def start_calibration(self):
        self.get_logger().info("=== CALIBRATION STARTED ===")
        self.calibration_phase = 0
        self.calibration_start = time.time()

    def handle_calibration(self, norm_values):

        elapsed = time.time() - self.calibration_start

        # -------- Phase 0: Initial wait --------
        if self.calibration_phase == 0:
            self.progress_bar(elapsed, 5)
            if elapsed > 5:
                print()
                self.get_logger().info("Keep hand OPEN")
                self.calibration_phase = 1
                self.calibration_start = time.time()
                self.calibration_data = []
            return True

        # -------- Phase 1: Open acquisition --------
        if self.calibration_phase == 1:
            self.progress_bar(elapsed, 5)
            self.calibration_data.append(norm_values)
            if elapsed > 5:
                print()
                self.open_norm = np.mean(self.calibration_data, axis=0)
                self.get_logger().info(f"OPEN calibration: {self.open_norm}")
                self.calibration_phase = 2
                self.calibration_start = time.time()
            return True

        # -------- Phase 2: Pause --------
        if self.calibration_phase == 2:
            self.progress_bar(elapsed, 5)
            if elapsed > 5:
                print()
                self.get_logger().info("Now CLOSE hand")
                self.calibration_phase = 3
                self.calibration_start = time.time()
                self.calibration_data = []
            return True

        # -------- Phase 3: Closed acquisition --------
        if self.calibration_phase == 3:
            self.progress_bar(elapsed, 5)
            self.calibration_data.append(norm_values)
            if elapsed > 5:
                print()
                self.close_norm = np.mean(self.calibration_data, axis=0)
                self.get_logger().info(f"CLOSE calibration: {self.close_norm}")
                self.calibration_phase = 4
                self.calibration_start = time.time()
            return True

        # -------- Phase 4: Final wait --------
        if self.calibration_phase == 4:
            self.progress_bar(elapsed, 5)
            if elapsed > 5:
                print()
                self.get_logger().info("=== CALIBRATION COMPLETE ===")
                self.calibration_enabled = False
            return True

        return False

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

        # Camera scale compensation
        palm_size = np.linalg.norm(get_pt(9) - wrist)

        thumb  = np.linalg.norm(get_pt(4)  - wrist) / palm_size
        index  = np.linalg.norm(get_pt(8)  - wrist) / palm_size
        middle = np.linalg.norm(get_pt(12) - wrist) / palm_size
        ring   = np.linalg.norm(get_pt(16) - wrist) / palm_size
        pinky  = np.linalg.norm(get_pt(20) - wrist) / palm_size

        norm_values = np.array([thumb, index, middle, ring, pinky])

        # ================= CALIBRATION =================
        if self.calibration_enabled:
            self.handle_calibration(norm_values)
            return

        # ================= MAPPING =================
        if self.open_norm is not None and self.close_norm is not None:
            norm = (norm_values - self.open_norm) / (self.close_norm - self.open_norm)
            norm = np.clip(norm, 0, 1)
        else:
            norm = self.normalize(norm_values, 0.3, 1.5)

        # 0=open, 1000=closed
        angles = norm * 1000.0

        # Robot order: mignolo→pollice
        robot_angles = np.array([
            angles[4],
            angles[3],
            angles[2],
            angles[1],
            angles[0],
            angles[0]
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
            self.get_logger().info(f"Norm values: {np.round(norm_values, 2)}")
            self.get_logger().info(f"Mapped angles: {robot_angles}")
            self.get_logger().info("-----------------")


# =========================================================

def main(args=None):
    rclpy.init(args=args)
    node = HandMapperNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
