#!/usr/bin/env python3

import time
import threading
import numpy as np

import rclpy
from rclpy.node import Node

from hand_msgs.msg import RH56DFXAngleCommand, RH56DFXFeedback

from inspire_sdkpy import inspire_sdk


class RH56DFXModbusNode(Node):

    def __init__(self):
        super().__init__('rh56dfx_modbus_node')

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
        self.get_logger().info(f"Connecting to RH56DFX at {self.ip} ...")

        self.handler = inspire_sdk.ModbusDataHandler(
            ip=self.ip,
            LR=self.lr,
            device_id=self.device_id
        )

        time.sleep(0.5)

        # =============================
        # ROS Interfaces
        # =============================
        self.cmd_sub = self.create_subscription(
            RH56DFXAngleCommand,
            "angle_command",
            self.command_callback,
            10
        )

        self.feedback_pub = self.create_publisher(
            RH56DFXFeedback,
            "feedback",
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

        self.get_logger().info("RH56DFX Modbus Node started")

    # ==========================================================
    # COMMAND CALLBACK (Write Angles)
    # ==========================================================
    def command_callback(self, msg: RH56DFXAngleCommand):

        angles = np.array(msg.angles)

        try:
            # Scrittura registri angoli
            self.handler.write_angle(angles.astype(int).tolist())
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

        feedback = RH56DFXFeedback()
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

        feedback.pinky_tip_touch = touch['pinky_tip_touch'].flatten().tolist()
        feedback.pinky_top_touch = touch['pinky_top_touch'].flatten().tolist()
        feedback.pinky_palm_touch = touch['pinky_palm_touch'].flatten().tolist()

        feedback.ring_tip_touch = touch['ring_tip_touch'].flatten().tolist()
        feedback.ring_top_touch = touch['ring_top_touch'].flatten().tolist()
        feedback.ring_palm_touch = touch['ring_palm_touch'].flatten().tolist()

        feedback.middle_tip_touch = touch['middle_tip_touch'].flatten().tolist()
        feedback.middle_top_touch = touch['middle_top_touch'].flatten().tolist()
        feedback.middle_palm_touch = touch['middle_palm_touch'].flatten().tolist()

        feedback.index_tip_touch = touch['index_tip_touch'].flatten().tolist()
        feedback.index_top_touch = touch['index_top_touch'].flatten().tolist()
        feedback.index_palm_touch = touch['index_palm_touch'].flatten().tolist()

        feedback.thumb_tip_touch = touch['thumb_tip_touch'].flatten().tolist()
        feedback.thumb_top_touch = touch['thumb_top_touch'].flatten().tolist()
        feedback.thumb_middle_touch = touch['thumb_middle_touch'].flatten().tolist()
        feedback.thumb_palm_touch = touch['thumb_palm_touch'].flatten().tolist()

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
    node = RH56DFXModbusNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()