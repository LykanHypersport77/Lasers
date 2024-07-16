import cv2
import numpy as np
import serial
import time

# Adjustable calibration variables
X_OFFSET = 13
Y_OFFSET = 48
X_SCALE = 0.4257
Y_SCALE = 0.4556
Z_SCALE = 1.0  # Initial scale factor for Z, adjust as needed

# Tolerance for calibration in pixels
CALIBRATION_TOLERANCE = 5
ADJUSTMENT_FACTOR = 0.1

# Camera parameters (focal length, sensor size, etc.)
FOCAL_LENGTH = 3.6  # in mm, adjust based on your camera
SENSOR_WIDTH = 4.8  # in mm, adjust based on your camera
SENSOR_HEIGHT = 3.6  # in mm, adjust based on your camera
CAMERA_WIDTH = 3840  # in pixels (4K resolution), adjust based on your camera
CAMERA_HEIGHT = 2160  # in pixels (4K resolution), adjust based on your camera

# Pre-defined drone width (in meters) for distance estimation
DRONE_WIDTH = 0.5  # adjust based on the actual drone size

def write_to_serial(port, baudrate, coords):
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        gcode_command = f"G1 X{45 + (250 - (coords[0] + X_OFFSET) * X_SCALE):.2f} Y{75 + (250 - (coords[1] + Y_OFFSET) * Y_SCALE):.2f} Z{(coords[2] * Z_SCALE):.2f} F15000\n"
        ser.write(gcode_command.encode())
        ser.close()
        print(f"Sent to serial: {gcode_command.strip()}")
    except Exception as e:
        print(f"An error occurred: {e}")

def average_coordinates(minx, miny, maxx, maxy):
    avg_x = (minx + maxx) / 2
    avg_y = (miny + maxy) / 2
    return avg_x, avg_y

def estimate_distance(known_width, focal_length, per_width, sensor_width, image_width):
    # Adjust focal length based on the image width and sensor width
    adjusted_focal_length = (focal_length / sensor_width) * image_width
    return (known_width * adjusted_focal_length) / per_width

def compute_velocity(prev_coords, curr_coords, time_elapsed):
    velocity = (curr_coords - prev_coords) / time_elapsed
    return velocity

serial_port = 'COM3'
baud_rate = 115200

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

prev_frame = None
prev_time = time.time()
prev_coords = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    if prev_frame is None:
        prev_frame = gray_frame
        continue

    # Calculate the absolute difference between the current frame and previous frame
    frame_diff = cv2.absdiff(prev_frame, gray_frame)
    thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
        avg_coords_2d = np.array(average_coordinates(x, y, x + w, y + h))

        distance = estimate_distance(DRONE_WIDTH, FOCAL_LENGTH, w, SENSOR_WIDTH, CAMERA_WIDTH)
        coords_3d = np.array([avg_coords_2d[0], avg_coords_2d[1], distance])
        print(f"3D Coordinates: {coords_3d}")

        curr_time = time.time()
        if prev_coords is not None:
            time_elapsed = curr_time - prev_time
            velocity = compute_velocity(prev_coords, coords_3d, time_elapsed)
            print(f"Velocity: {velocity} m/s")
        prev_time = curr_time
        prev_coords = coords_3d

        write_to_serial(serial_port, baud_rate, coords_3d)

    prev_frame = gray_frame
    cv2.imshow('webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
