#!/usr/bin/env python3

import time
import threading
import numpy as np

import rclpy
from rclpy.node import Node

from hand_msgs.msg import RH56DFTPAngleCommand, RH56DFTPFeedback

from inspire_sdkpy import inspire_sdk
from inspire_sdkpy.inspire_dds import inspire_hand_ctrl


class RH56DFTPModbusNode(Node):

    def __init__(self):
        super().__init__('rh56dftp_modbus_node')

        # =============================
        # Parameters
        # =============================
        self.declare_parameter("ip", "192.168.123.210")
        self.declare_parameter("controlled_hand", "left")
        self.declare_parameter("device_id", 1)
        self.declare_parameter("polling_hz", 200.0)

        # Safety parameters
        self.declare_parameter("current_limit_ma", 1500)
        self.declare_parameter("contact_current_ma", 650)
        self.declare_parameter("soft_grasp_enabled", True)

        self.ip = self.get_parameter("ip").value
        self.controlled_hand = self.get_parameter("controlled_hand").value
        self.device_id = self.get_parameter("device_id").value
        self.polling_hz = self.get_parameter("polling_hz").value

        self.current_limit_ma = self.get_parameter("current_limit_ma").value
        self.contact_current_ma = self.get_parameter("contact_current_ma").value
        self.soft_grasp_enabled = self.get_parameter("soft_grasp_enabled").value

        # unlock logic
        self.unlock_current_ma = 500
        self.unlock_time = 1.0

        # =============================
        # Finger metadata
        # =============================
        self.finger_names = [
            "pinky",
            "ring",
            "middle",
            "index",
            "thumb_bend",
            "thumb_rotate"
        ]

        # =============================
        # Internal state
        # =============================
        self.lock = threading.Lock()
        self.latest_data = None
        self.running = True

        # command states
        self.desired_angles = [1000, 1000, 1000, 1000, 1000, 0]
        self.safe_angles = self.desired_angles.copy()

        # safety states
        self.finger_locked = [False]*6
        self.contact_locked = [False]*6
        self.unlock_timer = [None]*6

        # =============================
        # Init Modbus
        # =============================
        self.get_logger().info(f"Connecting to RH56DFTP at {self.ip} ...")

        self.handler = inspire_sdk.ModbusDataHandler(
            ip=self.ip,
            LR=self.controlled_hand,
            device_id=self.device_id
        )

        time.sleep(0.5)

        self.open_hand()

        # =============================
        # ROS Interfaces
        # =============================
        self.cmd_sub = self.create_subscription(
            RH56DFTPAngleCommand,
            "/rh56dftp/angle_command",
            self.command_callback,
            10
        )

        self.feedback_pub = self.create_publisher(
            RH56DFTPFeedback,
            "/rh56dftp/feedback",
            10
        )

        # =============================
        # Threads
        # =============================
        self.read_thread = threading.Thread(target=self.read_loop)
        self.read_thread.start()

        self.control_thread = threading.Thread(target=self.control_loop)
        self.control_thread.start()

        # feedback publisher
        self.timer = self.create_timer(0.02, self.publish_feedback)

        self.get_logger().info("RH56DFTP Modbus Node started")

    # ==========================================================
    # OPEN HAND
    # ==========================================================
    def open_hand(self):

        self.get_logger().info("Opening hand at startup...")

        open_angles = [1000, 1000, 1000, 1000, 1000, 0]

        self.desired_angles = open_angles.copy()
        self.safe_angles = open_angles.copy()

        ctrl_msg = inspire_hand_ctrl(
            pos_set=[0]*6,
            angle_set=open_angles,
            force_set=[0]*6,
            speed_set=[0]*6,
            mode=0b0001
        )

        try:
            self.handler.write_registers_callback(ctrl_msg)
            time.sleep(1.0)
            self.get_logger().info("Hand fully opened.")
        except Exception as e:
            self.get_logger().error(f"Failed to open hand: {e}")

    # ==========================================================
    # COMMAND CALLBACK
    # ==========================================================
    def command_callback(self, msg):

        requested = np.array(msg.angles).astype(int)

        with self.lock:

            for i in range(6):

                if not self.finger_locked[i]:
                    self.desired_angles[i] = int(requested[i])

                # unlock contact if opening
                if requested[i] > self.safe_angles[i]:
                    self.contact_locked[i] = False

    # ==========================================================
    # READ LOOP
    # ==========================================================
    def read_loop(self):

        period = 1.0 / self.polling_hz

        while self.running:

            try:

                data_dict = self.handler.read()

                with self.lock:
                    self.latest_data = data_dict
                    self.check_current_safety(data_dict)

            except Exception as e:
                self.get_logger().error(f"Read error: {e}")

            time.sleep(period)

    # ==========================================================
    # SAFETY CHECK
    # ==========================================================
    def check_current_safety(self, data):

        currents = data['states']['CURRENT']
        angles = data['states']['ANGLE_ACT']

        now = time.time()

        for i in range(6):

            current = currents[i]

            # -----------------------------------------
            # OVERCURRENT PROTECTION
            # -----------------------------------------
            if current > self.current_limit_ma and not self.finger_locked[i]:

                self.finger_locked[i] = True
                self.safe_angles[i] = angles[i]

                self.get_logger().warn(
                    f"[OVERCURRENT] {self.finger_names[i]} "
                    f"{current} mA -> locking at {angles[i]}"
                )

            # -----------------------------------------
            # CONTACT DETECTION
            # -----------------------------------------
            if self.soft_grasp_enabled:

                if (current > self.contact_current_ma and
                        not self.contact_locked[i] and
                        not self.finger_locked[i]):

                    self.contact_locked[i] = True
                    self.safe_angles[i] = angles[i]

                    self.get_logger().info(
                        f"[CONTACT] {self.finger_names[i]} "
                        f"{current} mA -> grasp contact"
                    )

            # -----------------------------------------
            # AUTO UNLOCK LOGIC
            # -----------------------------------------
            if self.finger_locked[i]:

                if current < self.unlock_current_ma:

                    if self.unlock_timer[i] is None:
                        self.unlock_timer[i] = now

                    if now - self.unlock_timer[i] > self.unlock_time:

                        self.finger_locked[i] = False
                        self.unlock_timer[i] = None

                        self.get_logger().info(
                            f"[UNLOCK] {self.finger_names[i]} current normal"
                        )

                else:
                    self.unlock_timer[i] = None

    # ==========================================================
    # CONTROL LOOP (ONLY WRITER)
    # ==========================================================
    def control_loop(self):

        rate = 200.0
        period = 1.0 / rate

        while self.running:

            try:

                with self.lock:

                    for i in range(6):

                        if not self.finger_locked[i] and not self.contact_locked[i]:
                            self.safe_angles[i] = self.desired_angles[i]

                    ctrl_msg = inspire_hand_ctrl(
                        pos_set=[0]*6,
                        angle_set=self.safe_angles,
                        force_set=[0]*6,
                        speed_set=[0]*6,
                        mode=0b0001
                    )

                self.handler.write_registers_callback(ctrl_msg)

            except Exception as e:
                self.get_logger().error(f"Control write error: {e}")

            time.sleep(period)

    # ==========================================================
    # PUBLISH FEEDBACK
    # ==========================================================
    def publish_feedback(self):

        if self.latest_data is None:
            return

        feedback = RH56DFTPFeedback()
        feedback.header.stamp = self.get_clock().now().to_msg()

        with self.lock:
            data = self.latest_data

        feedback.position = data['states']['POS_ACT']
        feedback.angle = data['states']['ANGLE_ACT']
        feedback.force = data['states']['FORCE_ACT']
        feedback.current = data['states']['CURRENT']
        feedback.error = data['states']['ERROR']
        feedback.status = data['states']['STATUS']
        feedback.temperature = data['states']['TEMP']

        touch = data['touch']

        feedback.pinky_tip_touch = touch['fingerone_tip_touch'].flatten().tolist()
        feedback.pinky_top_touch = touch['fingerone_top_touch'].flatten().tolist()
        feedback.pinky_palm_touch = touch['fingerone_palm_touch'].flatten().tolist()

        feedback.ring_tip_touch = touch['fingertwo_tip_touch'].flatten().tolist()
        feedback.ring_top_touch = touch['fingertwo_top_touch'].flatten().tolist()
        feedback.ring_palm_touch = touch['fingertwo_palm_touch'].flatten().tolist()

        feedback.middle_tip_touch = touch['fingerthree_tip_touch'].flatten().tolist()
        feedback.middle_top_touch = touch['fingerthree_top_touch'].flatten().tolist()
        feedback.middle_palm_touch = touch['fingerthree_palm_touch'].flatten().tolist()

        feedback.index_tip_touch = touch['fingerfour_tip_touch'].flatten().tolist()
        feedback.index_top_touch = touch['fingerfour_top_touch'].flatten().tolist()
        feedback.index_palm_touch = touch['fingerfour_palm_touch'].flatten().tolist()

        feedback.thumb_tip_touch = touch['fingerfive_tip_touch'].flatten().tolist()
        feedback.thumb_top_touch = touch['fingerfive_top_touch'].flatten().tolist()
        feedback.thumb_middle_touch = touch['fingerfive_middle_touch'].flatten().tolist()
        feedback.thumb_palm_touch = touch['fingerfive_palm_touch'].flatten().tolist()

        feedback.palm_touch = touch['palm_touch'].flatten().tolist()

        self.feedback_pub.publish(feedback)

    # ==========================================================
    # SHUTDOWN
    # ==========================================================
    def destroy_node(self):

        self.running = False

        self.read_thread.join()
        self.control_thread.join()

        super().destroy_node()


# ==========================================================
# MAIN
# ==========================================================
def main(args=None):

    rclpy.init(args=args)

    node = RH56DFTPModbusNode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()