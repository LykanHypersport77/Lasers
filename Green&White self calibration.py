import cv2
import numpy as np
import serial

# Adjustable calibration variables
X_OFFSET = 0  # Adjust this offset to correct horizontal alignment
Y_OFFSET = 0  # Adjust this offset to correct vertical alignment
X_SCALE = 0.4257  # Initial scale factor for X, adjust as needed
Y_SCALE = 0.4556  # Initial scale factor for Y, adjust as needed

# Tolerance for calibration in pixels
CALIBRATION_TOLERANCE = 5

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
        gcode_command = f"G1 X{45 + (250 - (avg_coords[0] + X_OFFSET) * X_SCALE):.2f} Y{75 + (250 - (avg_coords[1] + Y_OFFSET) * Y_SCALE):.2f} F15000\n" #the scaling for x and y axis
        ser.write(gcode_command.encode())
        ser.close()
        print(f"Sent to serial: {gcode_command.strip()}")
    except Exception as e:
        print(f"An error occurred: {e}")

def average_coordinates(minx, miny, maxx, maxy):
    avg_x = (minx + maxx) / 2
    avg_y = (miny + maxy) / 2
    return avg_x, avg_y

def detect_laser(frame, white_threshold=240, green_range=((50, 100, 100), (70, 255, 255)), min_area=10, max_area=500): # finding laser that is green
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create a mask for bright white regions
    lower_white = np.array([0, 0, white_threshold], dtype=np.uint8) #made this because the camera is so bad that is sees the laser as white sometimes
    upper_white = np.array([180, 30, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv_frame, lower_white, upper_white)

    # Create a mask for green regions
    lower_green = np.array(green_range[0], dtype=np.uint8)
    upper_green = np.array(green_range[1], dtype=np.uint8)
    green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)

    # Combine both masks
    combined_mask = cv2.bitwise_or(white_mask, green_mask)

    # Apply operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.erode(combined_mask, kernel, iterations=1)
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # I dont really know if this helps or not
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:  # Filter by area to distinguish the laser spot
            x, y, w, h = cv2.boundingRect(contour)
            return average_coordinates(x, y, x + w, y + h)
    return None

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

    # Apply operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Draw the bounding box in a different color (red), was originally green but that interfered with laser detection
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)

        avg_coords = average_coordinates(x, y, x + w, y + h)
        print(f"Average coordinates: {avg_coords}")

        # Send to serial
        write_to_serial(serial_port, baud_rate, avg_coords)
        
        # Detect laser spot
        laser_coords = detect_laser(frame) #need to somehow get this to stop detecting the desk
        if laser_coords:
            frame = cv2.circle(frame, (int(laser_coords[0]), int(laser_coords[1])), 10, (0, 255, 255), -1)  # Yellow circle for laser spot
            print(f"Laser coordinates: {laser_coords}")

            # Auto-calibration
            x_diff = laser_coords[0] - avg_coords[0] #still glitchy because of laser detection
            y_diff = laser_coords[1] - avg_coords[1]

            if abs(x_diff) > CALIBRATION_TOLERANCE or abs(y_diff) > CALIBRATION_TOLERANCE:
                X_OFFSET -= x_diff * 0.1  # Adjust scaling factor as needed
                Y_OFFSET -= y_diff * 0.1  # Adjust scaling factor as needed
                print(f"Calibration: X_OFFSET={X_OFFSET}, Y_OFFSET={Y_OFFSET}")
    
    cv2.imshow('webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
