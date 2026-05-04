import cv2
import numpy as np
import time

import rclpy
from rclpy.node import Node


class ArucoNode(Node):
    def __init__(self):
        super().__init__('aruco_node')
        self.get_logger().info('✅ ArUco node started')
        self.live_aruco_detection()

    def live_aruco_detection(self):
        # =========================
        # 1. 카메라 캘리브레이션 파라미터
        # =========================
        camera_matrix = np.array([
            [340.8725662168225, 0.0, 319.00615441641145],
            [0.0, 318.7517218331414, 266.4956924279892],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        dist_coeffs = np.array([
            -0.007831276384506362,
            0.0032045866437473355,
            -0.04355424933501001,
            0.03502030948622766
        ], dtype=np.float32)

        # =========================
        # 2. ArUco 설정
        # =========================
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        # =========================
        # 3. 마커 정보
        # 기준점(원점) = ID 2 마커의 윗변 중간점 = (0, 0, 0)
        # 단위: mm
        # =========================
        MARKERS = {
            0: {"size": 124.0, "center": (-118.0,  62.0, 0.0)},
            1: {"size": 124.0, "center": ( 118.0,  62.0, 0.0)},
            2: {"size":  62.0, "center": (   0.0, -31.0, 0.0)},
        }

        # =========================
        # 4. 카메라 열기
        # =========================
        cap = cv2.VideoCapture(2)

        if not cap.isOpened():
            self.get_logger().error("❌ 카메라를 열 수 없습니다. VideoCapture 번호 확인")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        time.sleep(2)

        self.get_logger().info("✅ 실행 시작 (q 누르면 종료)")
        self.get_logger().info("📌 기준점 = ID 2 마커의 윗변 중간점 (0,0,0)")
        self.get_logger().info(f"📌 마커 설정: {MARKERS}")

        while rclpy.ok():
            ret, frame = cap.read()
            if not ret:
                self.get_logger().error("❌ 프레임 읽기 실패")
                break

            # =========================
            # 5. ArUco 마커 검출
            # =========================
            corners, ids, rejected = detector.detectMarkers(frame)

            if ids is not None and len(ids) > 0:
                ids = ids.flatten()

                # 검출된 마커 테두리 표시
                cv2.aruco.drawDetectedMarkers(frame, corners, ids.reshape(-1, 1))

                all_object_points = []
                all_image_points = []

                for corner, marker_id in zip(corners, ids):

                    if marker_id not in MARKERS:
                        continue

                    marker_info = MARKERS[marker_id]
                    marker_size = marker_info["size"]
                    cx, cy, cz = marker_info["center"]
                    half = marker_size / 2.0

                    # =========================
                    # 6. 2D 코너 정리
                    # corner 순서:
                    # topLeft, topRight, bottomRight, bottomLeft
                    # =========================
                    corner = corner.reshape((4, 2)).astype(np.float32)
                    topLeft, topRight, bottomRight, bottomLeft = corner

                    # 디버깅용 점 표시
                    cv2.circle(frame, tuple(topLeft.astype(int)), 5, (255, 0, 0), -1)
                    cv2.circle(frame, tuple(topRight.astype(int)), 5, (255, 0, 0), -1)
                    cv2.circle(frame, tuple(bottomRight.astype(int)), 5, (255, 0, 0), -1)
                    cv2.circle(frame, tuple(bottomLeft.astype(int)), 5, (255, 0, 0), -1)

                    # =========================
                    # 7. 3D 좌표 정의
                    # 기준: ID 2 마커의 윗변 중간점 = (0,0,0)
                    # =========================
                    object_points = np.array([
                        [cx - half, cy + half, cz],  # topLeft
                        [cx + half, cy + half, cz],  # topRight
                        [cx + half, cy - half, cz],  # bottomRight
                        [cx - half, cy - half, cz],  # bottomLeft
                    ], dtype=np.float32)

                    image_points = np.array([
                        topLeft,
                        topRight,
                        bottomRight,
                        bottomLeft
                    ], dtype=np.float32)

                    all_object_points.append(object_points)
                    all_image_points.append(image_points)

                    # 마커 ID 표시
                    tx = int(topLeft[0])
                    ty = int(topLeft[1]) - 10
                    if ty < 20:
                        ty = int(topLeft[1]) + 20

                    cv2.putText(frame, f"ID:{marker_id}", (tx, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # =========================
                # 8. pose 계산
                # 검출된 모든 마커를 합쳐서 한 번에 solvePnP
                # =========================
                if len(all_object_points) > 0:
                    all_object_points = np.vstack(all_object_points)
                    all_image_points = np.vstack(all_image_points)

                    success, rvec, tvec = cv2.solvePnP(
                        all_object_points,
                        all_image_points,
                        camera_matrix,
                        dist_coeffs,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )

                    if success:
                        # =========================
                        # 8-1. 좌표계 변환
                        # x축만 반전해서
                        # "카메라가 마커 기준 오른쪽으로 갈 때 x가 +"
                        # 되도록 수정
                        # =========================
                        t = tvec.reshape(3, 1)

                        T_coord = np.array([
                            [-1.0, 0.0, 0.0],
                            [ 0.0, 1.0, 0.0],
                            [ 0.0, 0.0, 1.0]
                        ], dtype=np.float32)

                        t_new = T_coord @ t
                        x, y, z = t_new.flatten()

                        # =========================
                        # 9. 기준점 좌표축 표시
                        # =========================
                        cv2.drawFrameAxes(
                            frame,
                            camera_matrix,
                            dist_coeffs,
                            rvec,
                            tvec,
                            50.0
                        )

                        # =========================
                        # 10. 기준점(원점)의 이미지상 위치 표시
                        # =========================
                        target_3d = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
                        projected, _ = cv2.projectPoints(
                            target_3d,
                            rvec,
                            tvec,
                            camera_matrix,
                            dist_coeffs
                        )
                        px, py = projected[0][0].astype(int)

                        cv2.circle(frame, (px, py), 6, (255, 0, 255), -1)
                        cv2.putText(frame, "TARGET", (px + 10, py),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                        # =========================
                        # 11. 텍스트 출력
                        # =========================
                        text1 = "TARGET POSE (ID2 TOP MIDDLE)"
                        text2 = f"X:{x:.1f} Y:{y:.1f} Z:{z:.1f} mm"

                        cv2.putText(frame, text1, (20, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        cv2.putText(frame, text2, (20, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # =========================
            # 12. 화면 출력
            # =========================
            cv2.imshow("ArUco Target Pose", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # =========================
        # 종료 처리
        # =========================
        cap.release()
        cv2.destroyAllWindows()
        self.get_logger().info("🛑 ArUco node 종료")


def main(args=None):
    rclpy.init(args=args)
    node = ArucoNode()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()