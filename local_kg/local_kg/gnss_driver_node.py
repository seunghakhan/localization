#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
import serial


class GNSSDriver(Node):
    def __init__(self):
        super().__init__('gnss_driver_node')

        self.serial_port = '/dev/ttyACM0'
        self.baudrate = 9600

        self.ser = serial.Serial(self.serial_port, self.baudrate, timeout=1)

        self.pub = self.create_publisher(NavSatFix, '/fix', 10)

        self.timer = self.create_timer(0.1, self.read_gnss)

        self.get_logger().info('GNSS Driver Node Started')

    def read_gnss(self):
        try:
            line = self.ser.readline().decode('utf-8', errors='ignore').strip()

            if "$GNGGA" in line or "$GPGGA" in line:
                data = line.split(',')

                if len(data) < 10:
                    return

                lat = self.convert_lat(data[2], data[3])
                lon = self.convert_lon(data[4], data[5])
                alt = float(data[9]) if data[9] != '' else 0.0
                fix_quality = int(data[6]) if data[6] != '' else 0
                satellite_count = int(data[7]) if data[7] != '' else 0
                hdop = float(data[8]) if data[8] != '' else 0.0

                msg = NavSatFix()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = 'gps'

                msg.latitude = lat
                msg.longitude = lon
                msg.altitude = alt
                msg.status.status = fix_quality

                self.pub.publish(msg)

                self.get_logger().info(
                    f"lat={lat:.8f}, lon={lon:.8f}, alt={alt:.2f} m, "
                    f"fix={fix_quality}, sats={satellite_count}, hdop={hdop:.2f}"
                )

        except Exception as e:
            self.get_logger().warn(f"Error: {e}")

    def convert_lat(self, lat_str, direction):
        if lat_str == '':
            return 0.0

        deg = float(lat_str[:2])
        minutes = float(lat_str[2:])
        lat = deg + minutes / 60.0

        if direction == 'S':
            lat *= -1

        return lat

    def convert_lon(self, lon_str, direction):
        if lon_str == '':
            return 0.0

        deg = float(lon_str[:3])
        minutes = float(lon_str[3:])
        lon = deg + minutes / 60.0

        if direction == 'W':
            lon *= -1

        return lon


def main(args=None):
    rclpy.init(args=args)
    node = GNSSDriver()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()