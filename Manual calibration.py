import cv2
from PIL import Image
import serial
import numpy as np

# Adjustable calibration variables
X_OFFSET = -2  # Adjust this offset to correct horizontal alignment
Y_OFFSET = -9  # Adjust this offset to correct vertical alignment
X_SCALE = 0.4257  # Initial scale factor for X, adjust as needed
Y_SCALE = 0.4556  # Initial scale factor for Y, adjust as needed

def get_limits(color, sensitivity=30):
    """Get HSV color limits with sensitivity for color detection."""
    c = np.uint8([[color]])  # BGR values
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV) 

    hue = hsvC[0][0][0]  # Get the hue value

    lowerLimit = np.array([max(0, hue - sensitivity), 100, 100], dtype=np.uint8)
    upperLimit = np.array([min(180, hue + sensitivity), 255, 255], dtype=np.uint8)

    return lowerLimit, upperLimit

def write_to_serial(port, baudrate, avg_coords):
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        gcode_command = f"G1 X{45 + (250 - (avg_coords[0] + X_OFFSET) * X_SCALE):.2f} Y{75 + (250 - (avg_coords[1] + Y_OFFSET) * Y_SCALE):.2f} F15000\n"
        if (250 - (avg_coords[0] + X_OFFSET) * X_SCALE) < 250 and 10 + (250 - (avg_coords[1] + Y_OFFSET) * Y_SCALE) < 250:
            ser.write(gcode_command.encode())
            ser.close()
            print(f"Sent to serial: {gcode_command.strip()}")
        else:
            print("Coordinates out of range, not sending command.")
    except Exception as e:
        print(f"An error occurred: {e}")

def average_coordinates(minx, miny, maxx, maxy):
    avg_x = (minx + maxx) / 2
    avg_y = (miny + maxy) / 2
    return avg_x, avg_y

serial_port = 'COM3'
baud_rate = 115200

# Define the color to track in BGR
target_color = [255, 0, 0]  # Blue

cap = cv2.VideoCapture(0)  # Use the correct camera index

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lowerLimit, upperLimit = get_limits(color=target_color)

    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)

        avg_coords = average_coordinates(x, y, x + w, y + h)
        print(f"Average coordinates: {avg_coords}")

        if avg_coords[1] > 180:  # Condition for sending to serial
            write_to_serial(serial_port, baud_rate, avg_coords)
    
    cv2.imshow('webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
