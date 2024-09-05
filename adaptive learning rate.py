import os
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin")
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5") #ignore these, not necessary for the code. originally put for cuda testing
import cv2
import numpy as np
import serial
import math
import threading
import time

# Transformation matrix coefficients
A = 0.1927
B = 0.19549999999999998
C_X = -64
C_Y = 31  # use test52.py to configure depending on distance, set up for around 20 feet rn

# Serial communication parameters
galvo_serial_port = 'COM3'
arduino_serial_port = 'COM5'  # Serial port for the Arduino
baud_rate = 115200

'''DEBUG STATEMENTS'''
# def read_arduino_output(ser_arduino):
#     """Function to read and print the Arduino's output via serial."""
#     while True:
#         if ser_arduino.in_waiting > 0:
#             arduino_output = ser_arduino.readline().decode('utf-8', errors='ignore').strip()
#             print(f"Arduino says: {arduino_output}")

# CUDA-based Background Subtractor with adaptive learning rate
bg_subtractor = cv2.cuda.createBackgroundSubtractorMOG2(history=100, varThreshold=300, detectShadows=True) #setting to false can speed up the process, may give false positives. check line 103

# Parameters for adaptive learning rate
min_learning_rate = 1e-6
max_learning_rate = 1e-3
motion_threshold = 3000  # Threshold to determine if there's significant motion. can aafford a lower tolerance with a faster background reset
background_reset_interval = 10  # Reset background model every 10 seconds. This prrevents the background model from going stale and detecting everything as motion

def write_to_galvo(ser_galvo, avg_coords):
    new_y = 540 + (avg_coords[1] - 540) * math.cos(math.degrees(1))  # calculation for the 1 degree offset on the plane caused by camera being higher than the laser. easy transformation

    galvo_x = (A * avg_coords[0]) + C_X
    galvo_y = (B * new_y) + C_Y
    galvo_x = max(0, min(3000, galvo_x))  # set t0 3000 because the driver handles higher numbers by subtracting 200 (max number) could lowkey be like 400 and be fine
    galvo_y = max(0, min(3000, galvo_y))  # not sure if i can set the min as a negative

    gcode_command = f"G1 X{galvo_x:.3f} Y{galvo_y:.3f} F9000\n"
    ser_galvo.write(gcode_command.encode())
    print(f"Sent to serial: {gcode_command.strip()}")

def average_coordinates(minx, miny, maxx, maxy):
    avg_x = (minx + maxx) / 2
    avg_y = (miny + maxy) / 2
    return avg_x, avg_y

def calculate_learning_rate(contours):
    if contours:
        # Calculate the total area of all contours
        total_area = sum(cv2.contourArea(c) for c in contours)
        # Adjust learning rate based on the amount of motion
        if total_area > motion_threshold:
            return max_learning_rate
        else:
            return min_learning_rate
    else:
        return max_learning_rate  # Use maximum learning rate if no contours detected

def process_frame(frame, ser_galvo, ser_arduino, last_reset_time, fire_mode):
    global bg_subtractor
    frame_gpu = cv2.cuda_GpuMat()
    frame_gpu.upload(frame)  # processing the frame using CUDA data container

    # Apply background subtraction using CUDA
    fg_mask_gpu = bg_subtractor.apply(frame_gpu, learningRate=-1, stream=cv2.cuda.Stream_Null())  # Apply with default learning rate
    
    # Download fg_mask to CPU for further processing (necessary for contour operations)
    fg_mask = fg_mask_gpu.download()

    # Calculate the adaptive learning rate based on CPU-processed contours
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    learning_rate = calculate_learning_rate(contours)  # Calculate learning rate after downloading mask to CPU

    # Apply background subtraction again using the adaptive learning rate
    fg_mask_gpu = bg_subtractor.apply(frame_gpu, learningRate=learning_rate, stream=cv2.cuda.Stream_Null())

    # Download the final mask after applying adaptive learning rate
    fg_mask = fg_mask_gpu.download()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # contours and other morphology to clean up the frame
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        avg_coords = average_coordinates(x, y, x + w, y + h)
        print(f"Average coordinates: {avg_coords}")

        if 303 <= avg_coords[0] <= 1802:
            threading.Thread(target=write_to_galvo, args=(ser_galvo, avg_coords)).start()

            if fire_mode:
                # If fire mode is ON, send fire command and turn on the red light
                ser_arduino.write(b"tracking\n")
                print("Fire mode is ON, sending fire_tracking command")
            else:
                # If fire mode is OFF, just track and turn on the yellow light
                ser_arduino.write(b"fire_tracking\n")
                print("Fire mode is OFF, sending tracking command")
        else:
            print("Coordinates out of range, not sending to galvo.")
    else:
        # No contours detected, reset to idle mode and turn on the green light
        ser_arduino.write(b"idle\n")
        print("No drone detected, sending idle command")

    cv2.imshow('webcam', frame)

    # Reset background model if time interval has passed
    if time.time() - last_reset_time > background_reset_interval:
        bg_subtractor = cv2.cuda.createBackgroundSubtractorMOG2(history=100, varThreshold=300, detectShadows=True)
        print("Background model reset.")
        return time.time()  # Update last reset time
    return last_reset_time

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    ser_galvo = serial.Serial(galvo_serial_port, baud_rate, timeout=1)
    ser_arduino = serial.Serial(arduino_serial_port, baud_rate, timeout=1)
    
    '''DEBUG STATEMENTS'''
    # arduino_thread = threading.Thread(target=read_arduino_output, args=(ser_arduino,))
    # arduino_thread.daemon = True  # Daemonize the thread so it exits with the program
    # arduino_thread.start()

    last_reset_time = time.time()

    fire_mode = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Check if fire mode signal is received from Arduino
        if ser_arduino.in_waiting > 0:
            try:
                arduino_signal = ser_arduino.readline().decode('utf-8', errors='ignore').strip()
                print(f"Arduino signal: {arduino_signal}")
                
                if "Fire mode toggled: ON" in arduino_signal:
                    fire_mode = True
                elif "Fire mode toggled: OFF" in arduino_signal:
                    fire_mode = False
            except Exception as e:
                print(f"Error reading Arduino signal: {e}")
        last_reset_time = process_frame(frame, ser_galvo, ser_arduino, last_reset_time, fire_mode)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    ser_galvo.close()
    ser_arduino.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
