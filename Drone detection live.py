import cv2
from ultralytics import YOLO
import torch
import time
import numpy as np

# Path to the trained model weights
weights_path = r"D:\dronetraining\runs\detect\train5\weights\best.pt"

# Initialize YOLO model with the trained weights
model = YOLO(weights_path)

# Set up the webcam or any other camera (0 is the default webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Set the resolution of the camera to 1920x1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Check if CUDA is available and use it
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Open the text file for logging processing times
with open('processing_times.txt', 'w') as log_file:

    # Function for optional preprocessing (add preprocessing steps if needed)
    def preprocess_frame(frame):
        # trains in rgb but opencv uses bgr
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame = frame / 255.0 #normalizing, remove if needed
        return frame

    # Loop to continuously get frames from the camera
    while True:
        ret, frame = cap.read()
        
        # If the frame was not retrieved, break the loop
        if not ret:
            print("Failed to grab frame")
            break

        # Optional preprocessing
        frame = preprocess_frame(frame)

        # Measure time taken for processing
        start_time = time.time()

        # Run YOLO model on the original frame
        results = model(frame)

        # Measure time taken for processing
        end_time = time.time()
        processing_time = end_time - start_time
        fps = 1 / processing_time

        # Log processing time and FPS to the text file
        log_file.write(f"Processing Time: {processing_time:.4f} seconds, FPS: {fps:.2f}\n")

        # Find the detection with the highest confidence
        best_detection = None
        best_confidence = 0.0
        
        for result in results:
            for detection in result.boxes:
                confidence = detection.conf[0].item()
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_detection = detection
        
        # Draw bounding box for the detection with the highest confidence
        if best_detection is not None:
            xmin, ymin, xmax, ymax = best_detection.xyxy[0].tolist()
            class_id = int(best_detection.cls[0])
            class_name = model.names[class_id]

            if best_confidence >= 0.3:
                # Draw the bounding box
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                
                # Draw the label
                label = f"{class_name} {best_confidence:.2f}"
                cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame with the predictions
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('YOLO Drone Detection', frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
