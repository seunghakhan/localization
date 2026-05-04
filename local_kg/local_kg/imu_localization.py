#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion


def quaternion_to_rotation_matrix(q: Quaternion):
    x = q.x
    y = q.y
    z = q.z
    w = q.w

    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)

    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - x * w)

    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x * x + y * y)

    return [
        [r00, r01, r02],
        [r10, r11, r12],
        [r20, r21, r22]
    ]


def mat_vec_mul(R, v):
    return [
        R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
        R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
        R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2]
    ]


class ImuDeadReckoningNode(Node):
    def __init__(self):
        super().__init__('imu_dead_reckoning_node')

        self.declare_parameter('imu_topic', '/imu/data')
        self.declare_parameter('odom_topic', '/imu_odom')
        self.declare_parameter('world_frame', 'odom')
        self.declare_parameter('child_frame', 'base_link')

        # 아주 작은 가속도는 0으로 처리해서 노이즈 적분 줄이기
        self.declare_parameter('acc_threshold', 0.05)

        # 아주 작은 속도는 0으로 처리
        self.declare_parameter('vel_threshold', 0.02)

        # 정지 상태일 때 속도 서서히 죽이기
        self.declare_parameter('damping', 0.99)

        imu_topic = self.get_parameter('imu_topic').get_parameter_value().string_value
        odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value

        self.world_frame = self.get_parameter('world_frame').get_parameter_value().string_value
        self.child_frame = self.get_parameter('child_frame').get_parameter_value().string_value
        self.acc_threshold = self.get_parameter('acc_threshold').get_parameter_value().double_value
        self.vel_threshold = self.get_parameter('vel_threshold').get_parameter_value().double_value
        self.damping = self.get_parameter('damping').get_parameter_value().double_value

        self.sub_imu = self.create_subscription(
            Imu,
            imu_topic,
            self.imu_callback,
            10
        )

        self.pub_odom = self.create_publisher(Odometry, odom_topic, 10)

        self.prev_time = None

        # 위치
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

        # 속도
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0

        self.get_logger().info('IMU dead reckoning node started')
        self.get_logger().info(f'Subscribing IMU topic: {imu_topic}')
        self.get_logger().info(f'Publishing odom topic: {odom_topic}')

    def imu_callback(self, msg: Imu):
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.prev_time is None:
            self.prev_time = current_time
            return

        dt = current_time - self.prev_time
        self.prev_time = current_time

        if dt <= 0.0 or dt > 1.0:
            return

        # IMU 선형가속도 (body frame)
        ax_body = msg.linear_acceleration.x
        ay_body = msg.linear_acceleration.y
        az_body = msg.linear_acceleration.z

        # orientation으로 body -> world 회전
        R = quaternion_to_rotation_matrix(msg.orientation)

        # world frame 가속도
        ax_world, ay_world, az_world = mat_vec_mul(R, [ax_body, ay_body, az_body])

        # gravity 제거
        # ROS에서 보통 world z-up 기준으로 중력은 +9.81이 측정에 포함되므로 빼줌
        az_world -= 9.81

        # 너무 작은 가속도는 0 처리
        if abs(ax_world) < self.acc_threshold:
            ax_world = 0.0
        if abs(ay_world) < self.acc_threshold:
            ay_world = 0.0
        if abs(az_world) < self.acc_threshold:
            az_world = 0.0

        # 속도 적분
        self.vx += ax_world * dt
        self.vy += ay_world * dt
        self.vz += az_world * dt

        # 가속도가 거의 없으면 속도 조금 감쇠
        if ax_world == 0.0:
            self.vx *= self.damping
        if ay_world == 0.0:
            self.vy *= self.damping
        if az_world == 0.0:
            self.vz *= self.damping

        # 너무 작은 속도는 0 처리
        if abs(self.vx) < self.vel_threshold:
            self.vx = 0.0
        if abs(self.vy) < self.vel_threshold:
            self.vy = 0.0
        if abs(self.vz) < self.vel_threshold:
            self.vz = 0.0

        # 위치 적분
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt

        # Odometry publish
        odom = Odometry()
        odom.header.stamp = msg.header.stamp
        odom.header.frame_id = self.world_frame
        odom.child_frame_id = self.child_frame

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = self.z

        odom.pose.pose.orientation = msg.orientation

        odom.twist.twist.linear.x = self.vx
        odom.twist.twist.linear.y = self.vy
        odom.twist.twist.linear.z = self.vz

        odom.twist.twist.angular = msg.angular_velocity

        self.pub_odom.publish(odom)

        self.get_logger().info(
            f'x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f}, '
            f'vx={self.vx:.3f}, vy={self.vy:.3f}, vz={self.vz:.3f}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = ImuDeadReckoningNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()