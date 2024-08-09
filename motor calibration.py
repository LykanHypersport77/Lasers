import cv2
import numpy as np
import serial
import math

# Initial Transformation matrix coefficients
A = 0.1867  # Scale factor for X
B = 0.2085  # Scale factor for Y
C_X = 20  # Offset for X
C_Y = -10  # Offset for Y

# Serial communication parameters
serial_port = 'COM3'
baud_rate = 115200

# Background Subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)

def write_to_serial(port, baudrate, avg_coords):
    new_y = 540 + (avg_coords[1] - 540) * math.cos(math.radians(1))  # to project bounding box onto a plane not 1 degree off
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        # Apply the transformation matrix
        galvo_x = (A * avg_coords[0]) + C_X
        galvo_y = (B * new_y) + C_Y
        galvo_x = max(0, min(3000, galvo_x))  # Ensure coordinates are within galvo limits
        galvo_y = max(0, min(3000, galvo_y))  # Ensure coordinates are within galvo limits
        gcode_command = f"G1 X{galvo_x:.3f} Y{galvo_y:.3f} F9000\n"
        ser.write(gcode_command.encode())
        ser.close()
        print(f"Sent to serial: {gcode_command.strip()}")
    except Exception as e:
        print(f"An error occurred: {e}")

def average_coordinates(minx, miny, maxx, maxy):
    avg_x = (minx + maxx) / 2
    avg_y = (miny + maxy) / 2
    return avg_x, avg_y

def calibrate_transformation(reference_points_camera, reference_points_galvo):
    """
    Calibrate the transformation matrix using reference points.
    :param reference_points_camera: List of reference points in camera coordinates [(x1, y1), (x2, y2), ...]
    :param reference_points_galvo: List of reference points in galvo coordinates [(X1, Y1), (X2, Y2), ...]
    :return: Transformation matrix coefficients A, B, C_X, C_Y
    """
    camera_points = np.array(reference_points_camera, dtype=np.float32)
    galvo_points = np.array(reference_points_galvo, dtype=np.float32)
    h, _ = cv2.findHomography(camera_points, galvo_points)
    A, B = h[0, 0], h[1, 1]
    C_X, C_Y = h[0, 2], h[1, 2]
    return A, B, C_X, C_Y

cap = cv2.VideoCapture(0)  # Use the correct camera index

# Set camera resolution to 1920x1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Create a window and set it to fullscreen
cv2.namedWindow('webcam', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('webcam', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Calibration reference points
# Format: [(camera_x1, camera_y1), (camera_x2, camera_y2), ...]
reference_points_camera = [(299, 166), (1632, 174), (882, 568), (1086, 670)]
# Format: [(galvo_x1, galvo_y1), (galvo_x2, galvo_y2), ...]
reference_points_galvo = [(0, 50), (250, 0), (110, 130), (150, 150)]

# Calibrate the transformation matrix
A, B, C_X, C_Y = calibrate_transformation(reference_points_camera, reference_points_galvo)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Apply the background subtractor
    fg_mask = bg_subtractor.apply(frame)

    # Apply some morphological operations to eliminate noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Draw the bounding box in a different color (red)
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        avg_coords = average_coordinates(x, y, x + w, y + h)
        print(f"Average coordinates: {avg_coords}")

        # Check if the average x-coordinate is within the valid range
        if 303 <= avg_coords[0] <= 1802:
            # Send to serial
            write_to_serial(serial_port, baud_rate, avg_coords)
        else:
            print("Coordinates out of range, not sending to galvo.")
    
    cv2.imshow('webcam', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('a'):  # Adjust left
        C_X -= 1
        print(f"Adjusted C_X: {C_X}")
    elif key == ord('d'):  # Adjust right
        C_X += 1
        print(f"Adjusted C_X: {C_X}")
    elif key == ord('w'):  # Adjust up
        C_Y -= 1
        print(f"Adjusted C_Y: {C_Y}")
    elif key == ord('s'):  # Adjust down
        C_Y += 1
        print(f"Adjusted C_Y: {C_Y}")

cap.release()
cv2.destroyAllWindows()
