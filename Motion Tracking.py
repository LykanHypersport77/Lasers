import os
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin")
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5")
import cv2
import numpy as np
import serial
import math
import threading


# Transformation matrix coefficients
A = 0.1927
B = 0.19549999999999998
C_X = -64
C_Y = 31 #use test52.py to configure depending on distance, set up for aroun 15 feet rn

# Serial communication parameters
serial_port = 'COM3'
baud_rate = 115200

# CUDA-based Background Subtractor
bg_subtractor = cv2.cuda.createBackgroundSubtractorMOG2(history=100, varThreshold=100, detectShadows=True)

def write_to_serial(ser, avg_coords):
    new_y = 540 + (avg_coords[1] - 540) * math.cos(math.degrees(1)) #calculation for the 1 degree offset on the plane caused by camera being higher than the laser. easy transformation

    galvo_x = (A * avg_coords[0]) + C_X
    galvo_y = (B * new_y) + C_Y
    galvo_x = max(0, min(3000, galvo_x))#set t0 3000 because the driver handles higher numbers by subtracing 200 (max number)
    galvo_y = max(0, min(3000, galvo_y))

    gcode_command = f"G1 X{galvo_x:.3f} Y{galvo_y:.3f} F9000\n"
    ser.write(gcode_command.encode())
    print(f"Sent to serial: {gcode_command.strip()}")

def average_coordinates(minx, miny, maxx, maxy):
    avg_x = (minx + maxx) / 2
    avg_y = (miny + maxy) / 2
    return avg_x, avg_y

def process_frame(frame, ser):
    frame_gpu = cv2.cuda_GpuMat()
    frame_gpu.upload(frame) #processing the frame using CUDA data container
    
    fg_mask_gpu = bg_subtractor.apply(frame_gpu, learningRate=-1, stream=cv2.cuda.Stream_Null()) #idk why learning rate is a thing, -1 is auto, 0-1 is rest

    fg_mask = fg_mask_gpu.download()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) #countours and other morphology to clean up the frame
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
            threading.Thread(target=write_to_serial, args=(ser, avg_coords)).start() #these are the bounds that the galvo can shoot to. drone can be tracked outside that range but galvo will not shoot
        else:
            print("Coordinates out of range, not sending to galvo.")
    
    cv2.imshow('webcam', frame)

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    ser = serial.Serial(serial_port, baud_rate, timeout=1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        process_frame(frame, ser)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    ser.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()