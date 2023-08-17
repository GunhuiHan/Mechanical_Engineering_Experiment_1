import cv2
import numpy as np
import pickle
import time
import serial
import threading


py_serial = serial.Serial(
    port = "/dev/ttyACM0",
    # 보드 레이트 (통신 속도)
    baudrate=9600,
    timeout = 0.1
)

position = 90 # degrees 
rotation_degree = 0 # degrees

last_adjustment_time = 0

CAMERA_WIDTH = 1280 # 1920 1280 640
CAMERA_HEIGHT = 720 # 1080 720 360

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

# get focal length from calibration.pkl
calibration_file = f'calibrations/calibration_{CAMERA_WIDTH}_{CAMERA_HEIGHT}'

with open(calibration_file + '.pkl', 'rb') as f:
    data = pickle.load(f)
    cameraMatrix = data[0]
    dist = data[1]

    focal_length = data[0][0][0]

detector = cv2.QRCodeDetector()

def control_motor():
    global position, rotation_degree, last_adjustment_time

    current_time = time.time()
    elapsed_time = current_time - last_adjustment_time

    if (elapsed_time > 0.5):        
        goal_degree = int(position - rotation_degree)
        print(f'goal_degree : {goal_degree}')
        py_serial.write(str(goal_degree).encode())

        if py_serial.readable():
            response = py_serial.readline()
            print(response[:len(response) - 1].decode())
            position = goal_degree
            last_adjustment_time = current_time

try:
    while True:
        success, img = cap.read()

        start = time.perf_counter()

        value, points, qrcode = detector.detectAndDecode(img)

        if value != "":
            top_left, top_right, bottom_right, bottom_left = points[0]

            # QRCode center position (pixel)
            image_center_x = (top_left[0] + bottom_right[0]) / 2
            image_center_y = (top_left[1] + bottom_right[1]) / 2

            # Camera cneter position (pixel)
            CAMERA_CENTER_X = CAMERA_WIDTH / 2
            CAMERA_CENTER_Y = CAMERA_HEIGHT / 2

            center_difference_x = image_center_x - CAMERA_CENTER_X
            center_difference_y = image_center_y - CAMERA_CENTER_Y

            # Distance from QRCode to Camera
            image_width = np.linalg.norm(np.array(bottom_left) - np.array(top_left))
            pysical_width = 92 # mm
            distance = pysical_width * focal_length / abs(image_width)

            # Rotation degree
            pysical_center_difference_x = center_difference_x * pysical_width / image_width
            rotation_degree = np.arctan(pysical_center_difference_x / distance)
            rotation_degree = rotation_degree * 180 / np.pi

            pts = np.array(points, dtype=np.int32)
            cv2.polylines(img, pts, isClosed=True, color=(0, 255, 0), thickness=5)
            cv2.circle(img, (int(image_center_x), int(image_center_y)), color = (0, 0, 255), radius=3, thickness=3)
            cv2.putText(img, str(distance), (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            cv2.putText(img, f'center difference : ({int(center_difference_x)},{int(center_difference_y)})', (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

            end = time.perf_counter()
            totalTime = end - start
            fps = 1 / totalTime

            cv2.putText(img, f'FPS: {int(fps)}', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            cv2.imshow('img', img)

            if(abs(rotation_degree) > 1):
                control_motor()

            if cv2.waitKey(1) & 0xFF == 27:
                break
finally:
    cap.release()
    cv2.destroyAllWindows()
    py_serial.close()