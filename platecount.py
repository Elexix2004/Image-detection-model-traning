#region hit box plate counting  
from ultralytics import YOLO
import cv2
import supervision as sv
import numpy as np
import csv
import os

# Load YOLOv8 model
model = YOLO("C:/root/v2/runs/detect/train2/weights/best.pt")

# Load video
video_path = "C:/Users/Desktop/pm.mkv"
cap = cv2.VideoCapture(video_path)

# Output CSV path
csv_path = "plate_counts.csv"

# CSV header
with open(csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Frame", "Track ID", "Timestamp (s)", "Center X", "Center Y", "Cumulative Plate Count"])

# Screen size for display
DISPLAY_SIZE = (1280, 720)

# ROI coordinates
ROI_TOP_LEFT = (200, 400)
ROI_BOTTOM_RIGHT = (1100, 700)

# Settings
frame_skip = 5
frame_id = 0
fps = cap.get(cv2.CAP_PROP_FPS)

# Tracking and Counting
tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()
already_counted_ids = set()
plate_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    if frame_id % frame_skip != 0:
        continue

    # Resize frame
    resized_input = cv2.resize(frame, DISPLAY_SIZE)

    # Inference
    results = model(resized_input, imgsz=960, conf=0.5)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.class_id == 2]  # Filter for "Plate"

    tracked = tracker.update_with_detections(detections)

    for i in range(len(tracked)):
        x1, y1, x2, y2 = tracked.xyxy[i]
        track_id = tracked.tracker_id[i]
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        if (ROI_TOP_LEFT[0] < center_x < ROI_BOTTOM_RIGHT[0] and
            ROI_TOP_LEFT[1] < center_y < ROI_BOTTOM_RIGHT[1]):

            if track_id not in already_counted_ids:
                already_counted_ids.add(track_id)
                plate_count += 1

                # Timestamp calculation
                timestamp = round(frame_id / fps, 2)

                # Log to CSV
                with open(csv_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([frame_id, track_id, timestamp, center_x, center_y, plate_count])

    # Draw annotations
    annotated_frame = box_annotator.annotate(resized_input.copy(), detections=tracked)
    cv2.rectangle(annotated_frame, ROI_TOP_LEFT, ROI_BOTTOM_RIGHT, (0, 255, 255), 2)
    cv2.putText(annotated_frame, f"Plates Counted: {plate_count}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

    # Show frame
    cv2.imshow("Plate Counter with CSV Logging", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
