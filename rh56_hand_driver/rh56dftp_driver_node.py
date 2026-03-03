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
        self.declare_parameter("lr", "l")
        self.declare_parameter("device_id", 1)
        self.declare_parameter("polling_hz", 200.0)

        self.ip = self.get_parameter("ip").value
        self.lr = self.get_parameter("lr").value
        self.device_id = self.get_parameter("device_id").value
        self.polling_hz = self.get_parameter("polling_hz").value

        # =============================
        # Init Modbus Handler
        # =============================
        self.get_logger().info(f"Connecting to RH56DFTP at {self.ip} ...")

        self.handler = inspire_sdk.ModbusDataHandler(
            ip=self.ip,
            LR=self.lr,
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
        # Internal state
        # =============================
        self.lock = threading.Lock()
        self.latest_data = None
        self.running = True

        # =============================
        # Start polling thread
        # =============================
        self.thread = threading.Thread(target=self.read_loop)
        self.thread.start()

        # Publish timer (50 Hz)
        self.timer = self.create_timer(0.02, self.publish_feedback)

        self.get_logger().info("RH56DFTP Modbus Node started")

    # ==========================================================
    # FULLY OPEN HAND
    # ==========================================================
    def open_hand(self):
        """
        Open the hand completely at startup:
        - All fingers fully open → 1000
        - Thumb opposition fully open → 0
        """
        self.get_logger().info("Opening hand at startup...")

        open_angles = [1000, 1000, 1000, 1000, 1000, 0]

        try:
            # self.handler.write_angle(open_angles)
            ctrl_msg = inspire_hand_ctrl(
                pos_set=[0]*6,
                angle_set=open_angles,
                force_set=[0]*6,
                speed_set=[0]*6,
                mode=0b0001
            )
            self.handler.write_registers_callback(ctrl_msg)
            
            time.sleep(1.0)  # give time to physically move
            self.get_logger().info("Hand fully opened.")
        except Exception as e:
            self.get_logger().error(f"Failed to open hand: {e}")

    # ==========================================================
    # COMMAND CALLBACK (Write Angles)
    # ==========================================================
    def command_callback(self, msg: RH56DFTPAngleCommand):

        angles = np.array(msg.angles)

        try:
            # Scrittura registri angoli
            # self.handler.write_angle(angles.astype(int).tolist())
            ctrl_msg = inspire_hand_ctrl(
                pos_set=[0]*6,
                angle_set=angles.astype(int).tolist(),
                force_set=[0]*6,
                speed_set=[0]*6,
                mode=0b0001
            )
            self.handler.write_registers_callback(ctrl_msg)
        except Exception as e:
            self.get_logger().error(f"Write error: {e}")

    # ==========================================================
    # READ LOOP THREAD
    # ==========================================================
    def read_loop(self):

        period = 1.0 / self.polling_hz

        call_count = 0
        start_time = time.perf_counter()

        while self.running:

            try:
                data_dict = self.handler.read()

                with self.lock:
                    self.latest_data = data_dict

                call_count += 1

                # Debug frequency
                if call_count % 200 == 0:
                    elapsed = time.perf_counter() - start_time
                    freq = call_count / elapsed
                    self.get_logger().info(f"Polling freq: {freq:.1f} Hz")

            except Exception as e:
                self.get_logger().error(f"Read error: {e}")

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

        # =============================
        # Joint State
        # =============================
        feedback.position = data['states']['POS_ACT']
        feedback.angle = data['states']['ANGLE_ACT']
        feedback.force = data['states']['FORCE_ACT']
        feedback.current = data['states']['CURRENT']
        feedback.error = data['states']['ERROR']
        feedback.status = data['states']['STATUS']
        feedback.temperature = data['states']['TEMP']

        # =============================
        # Touch Mapping
        # =============================
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
    # Shutdown
    # ==========================================================
    def destroy_node(self):
        self.running = False
        self.thread.join()
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