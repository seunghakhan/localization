#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PointStamped


# WGS84 constants
A = 6378137.0
F = 1 / 298.257223563
E2 = F * (2 - F)


class GNSS2ENU:
    def __init__(self, origin_sample_count=10):
        self.origin_set = False
        self.lat0 = None
        self.lon0 = None
        self.alt0 = None

        self.origin_sample_count = origin_sample_count
        self.origin_buffer = []

    def geodetic_to_ecef(self, lat, lon, alt):
        sin_lat = math.sin(lat)
        cos_lat = math.cos(lat)
        sin_lon = math.sin(lon)
        cos_lon = math.cos(lon)

        N = A / math.sqrt(1 - E2 * sin_lat * sin_lat)

        X = (N + alt) * cos_lat * cos_lon
        Y = (N + alt) * cos_lat * sin_lon
        Z = (N * (1 - E2) + alt) * sin_lat

        return X, Y, Z

    def convert(self, lat_deg, lon_deg, alt):
        lat = math.radians(lat_deg)
        lon = math.radians(lon_deg)

        # 처음 GNSS 여러 개를 모아서 평균 origin 설정
        if not self.origin_set:
            self.origin_buffer.append((lat, lon, alt))

            if len(self.origin_buffer) < self.origin_sample_count:
                return None

            self.lat0 = sum(p[0] for p in self.origin_buffer) / len(self.origin_buffer)
            self.lon0 = sum(p[1] for p in self.origin_buffer) / len(self.origin_buffer)
            self.alt0 = sum(p[2] for p in self.origin_buffer) / len(self.origin_buffer)

            self.origin_set = True
            print("Origin set by average!")
            return 0.0, 0.0, 0.0

        lat0 = self.lat0
        lon0 = self.lon0
        alt0 = self.alt0

        X, Y, Z = self.geodetic_to_ecef(lat, lon, alt)
        X0, Y0, Z0 = self.geodetic_to_ecef(lat0, lon0, alt0)

        dx = X - X0
        dy = Y - Y0
        dz = Z - Z0

        sin_lat0 = math.sin(lat0)
        cos_lat0 = math.cos(lat0)
        sin_lon0 = math.sin(lon0)
        cos_lon0 = math.cos(lon0)

        east = -sin_lon0 * dx + cos_lon0 * dy
        north = -sin_lat0 * cos_lon0 * dx - sin_lat0 * sin_lon0 * dy + cos_lat0 * dz
        up = cos_lat0 * cos_lon0 * dx + cos_lat0 * sin_lon0 * dy + sin_lat0 * dz

        return east, north, up


class GNSSToENUNode(Node):
    def __init__(self):
        super().__init__('gnss_to_enu_node')

        self.converter = GNSS2ENU(origin_sample_count=10)

        self.sub = self.create_subscription(
            NavSatFix,
            '/fix',
            self.gnss_callback,
            10
        )

        self.pub = self.create_publisher(
            PointStamped,
            '/enu_position',
            10
        )

        self.get_logger().info('GNSS to ENU node started')

    def gnss_callback(self, msg):
        if msg.status.status < 0:
            self.get_logger().warn('Invalid GNSS fix')
            return

        lat = msg.latitude
        lon = msg.longitude
        alt = msg.altitude

        result = self.converter.convert(lat, lon, alt)

        if result is None:
            self.get_logger().info(
                f'Collecting origin samples: '
                f'{len(self.converter.origin_buffer)}/'
                f'{self.converter.origin_sample_count}'
            )
            return

        E, N, U = result

        enu_msg = PointStamped()
        enu_msg.header.stamp = self.get_clock().now().to_msg()
        enu_msg.header.frame_id = 'enu'
        enu_msg.point.x = E
        enu_msg.point.y = N
        enu_msg.point.z = U

        self.pub.publish(enu_msg)

        self.get_logger().info(
            f'E: {E:.3f} m, N: {N:.3f} m, U: {U:.3f} m'
        )


def main(args=None):
    rclpy.init(args=args)

    node = GNSSToENUNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()