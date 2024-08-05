import cv2
import numpy as np
import serial

# Initial calibration offsets
X_OFFSET = 12.5  # Initial X offset when the laser is centered
Y_OFFSET = 19.23 # Initial Y offset when the laser is centered

# Motor units per pixel
MOTOR_UNITS_PER_PIXEL_X = 0.1667  # Motor units per pixel in the X direction
MOTOR_UNITS_PER_PIXEL_Y = 0.1825  # Motor units per pixel in the Y direction

# Distance and bounding box sizes
DISTANCES = [5, 10, 20]  # Distances in feet
BOX_SIZES = [(260, 150), (140, 50), (70, 25)]  # Bounding box sizes at corresponding distances

# Serial communication parameters
serial_port = 'COM3'
baud_rate = 115200

# Background Subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)

def estimate_distance(w, h):
    # Estimate distance based on bounding box size
    for i in range(len(DISTANCES)):
        box_w, box_h = BOX_SIZES[i]
        if w > box_w * 0.8 and h > box_h * 0.8:
            return DISTANCES[i]
    return DISTANCES[-1]  # Default to the farthest distance

def write_to_serial(port, baudrate, avg_coords):
    try:
        ser = serial.Serial(port, baudrate, timeout=1)

        # Calculate the distance and adjust scaling factors
        distance = estimate_distance(avg_coords[2] - avg_coords[0], avg_coords[3] - avg_coords[1])
        X_SCALE = MOTOR_UNITS_PER_PIXEL_X / distance
        Y_SCALE = MOTOR_UNITS_PER_PIXEL_Y / distance

        # Map the camera coordinates to the galvo coordinates
        galvo_x = (avg_coords[0] * X_SCALE) + X_OFFSET
        galvo_y = (avg_coords[1] * Y_SCALE) + Y_OFFSET
        galvo_x = max(0, min(200, galvo_x))  # Ensure coordinates are within galvo limits
        galvo_y = max(0, min(200, galvo_y))  # Ensure coordinates are within galvo limits

        gcode_command = f"G1 X{galvo_x:.2f} Y{galvo_y:.2f} F15000\n"
        ser.write(gcode_command.encode())
        ser.close()
        print(f"Sent to serial: {gcode_command.strip()}")
    except Exception as e:
        print(f"An error occurred: {e}")

def average_coordinates(minx, miny, maxx, maxy):
    avg_x = (minx + maxx) / 2
    avg_y = (miny + maxy) / 2
    return avg_x, avg_y, maxx - minx, maxy - miny

cap = cv2.VideoCapture(0)  # Use the correct camera index

# Set camera resolution to 1920x1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Create a window and set it to fullscreen
cv2.namedWindow('webcam', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('webcam', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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

        # Send to serial
        write_to_serial(serial_port, baud_rate, avg_coords)
    
    cv2.imshow('webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
