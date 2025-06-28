#capture frames from a live feed at a time interval

import cv2
import os
from datetime import datetime, timedelta

# Live stream URL
stream_url = 'http://your_live_stream_link'

# Save directory
save_path = "C:/Users/YourName/Desktop/snapshots/"
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Cannot open stream")
    exit()

# Set the next capture time
next_capture = datetime.now()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        continue

    now = datetime.now()

    # If it's time to capture a new image
    if now >= next_capture:
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_path, f'snapshot_{timestamp}.jpg')
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")

        # Set time for next capture
        next_capture = now + timedelta(minutes=1)

cap.release()
