import cv2
import numpy as np
import serial
import math

# Transformation matrix coefficients
A = 0.1927  # Scale factor for X
B = 0.19549999999999998  # Scale factor for Y
C_X = -77  # Offset for X
C_Y = 29  # Offset for Y
#0.1825 for y
#0.1667 for Xr

#A = 0.1667  # Scale factor for X
#B = 0.1825  # Scale factor for Y

# Serial communication parameters
serial_port = 'COM3'
baud_rate = 115200

# Background Subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)



def write_to_serial(port, baudrate, avg_coords):
    
    new_y = 540 + (avg_coords[1]-540)* math.cos(math.degrees(1)) #to project bounding box onto a plane not 1 degree off

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

        # Check if the average x-coordinate is within the valid range
        if 303 <= avg_coords[0] <= 1802:
            # Send to serial
            write_to_serial(serial_port, baud_rate, avg_coords)
        else:
            print("Coordinates out of range, not sending to galvo.")
    
    cv2.imshow('webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
