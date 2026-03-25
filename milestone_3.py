import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import random
from collections import deque
from ultralytics import YOLO



# -------------------------------
# ZONE CONFIG (re-use from milestone 1)
# -------------------------------
ZONE_FILE = "zones.json"


def random_color():
    return [random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255)]


def load_zones():
    if os.path.exists(ZONE_FILE):
        with open(ZONE_FILE, "r") as f:
            data = json.load(f)
            for z in data:
                if "color" not in z or not z["color"]:
                    z["color"] = random_color()
            return data
    else:
        # fallback: two example zones if JSON is missing
        print("No zones.json found, using default zones.")
        return [
            {
                "name": "Zone 1",
                "coords": [50, 150, 300, 450],
                "color": random_color(),
            },
            {
                "name": "Zone 2",
                "coords": [350, 150, 600, 450],
                "color": random_color(),
            },
        ]


zones = load_zones()

# -------------------------------
# MODEL & CAMERA
# -------------------------------
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

zone_counts = {z["name"]: 0 for z in zones}
entry_count = 0
exit_count = 0
line_y = 300

cv2.namedWindow("Milestone 3", cv2.WINDOW_NORMAL)
time_data = deque(maxlen=50)
total_data = deque(maxlen=50)
zone1_data = deque(maxlen=50)
zone2_data = deque(maxlen=50)

frame_number = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    total_count = 0
    frame_number += 1
    # reset counts at start of frame
    zone_counts = {z["name"]: 0 for z in zones}

    results = model(frame)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls == 0 and conf > 0.6:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1+x2)//2
                cy = (y1+y2)//2

                total_count += 1

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)

                # Zone Check (zone-based counting)
                for z in zones:
                    name = z["name"]
                    zx1, zy1, zx2, zy2 = z["coords"]
                    if zx1 < cx < zx2 and zy1 < cy < zy2:
                        zone_counts[name] += 1

    # Draw Zones
    for z in zones:
        name = z["name"]
        x1, y1, x2, y2 = z["coords"]
        color = tuple(z.get("color", random_color()))
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        cv2.putText(frame,f"{name}: {zone_counts[name]}",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,color,2)

    # Draw Entry Line
    cv2.line(frame,(0,line_y),(frame.shape[1],line_y),(0,255,255),2)

    # ---------------- DASHBOARD PANEL (zone-based) ----------------
    dashboard = np.zeros((frame.shape[0], 350, 3), dtype=np.uint8)
    dashboard[:] = (30, 30, 70)

    y_offset = 50

    cv2.putText(dashboard,"Live Dashboard",
                (50,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,(255,255,255),2)

    cv2.putText(dashboard,f"Total People: {total_count}",
                (20,y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,(0,255,0),2)

    y_offset += 40

    for name in zone_counts:
        cv2.putText(dashboard,
                    f"{name}: {zone_counts[name]}",
                    (20,y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,(255,255,0),2)
        y_offset += 35

    y_offset += 20
    cv2.putText(dashboard,f"Entry: {entry_count}",
                (20,y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,(255,255,255),2)

    y_offset += 40
    cv2.putText(dashboard,f"Exit: {exit_count}",
                (20,y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,(255,255,255),2)

    # Alert Example (zone-aware threshold)
    if total_count > 5:
        cv2.rectangle(dashboard,(10,frame.shape[0]-80),
                      (330,frame.shape[0]-30),
                      (0,0,255),-1)
        cv2.putText(dashboard,"⚠ Capacity Approaching Limit",
                    (20,frame.shape[0]-45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,(255,255,255),2)

    # Update analytics buffers (for graphs)
    time_data.append(frame_number)
    total_data.append(total_count)
    zone_names = list(zone_counts.keys())
    # handle up to 2 zones for simple chart
    if len(zone_names) > 0:
        zone1_data.append(zone_counts[zone_names[0]])
    if len(zone_names) > 1:
        zone2_data.append(zone_counts[zone_names[1]])

    # Combine Camera + Dashboard
    combined = np.hstack((frame, dashboard))

    cv2.imshow("Milestone 3", combined)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('g'):
        # show simple line chart of recent counts
        plt.figure(figsize=(8,5))
        plt.plot(time_data, total_data, label="Total People")
        if len(zone1_data) > 0:
            plt.plot(time_data, zone1_data, label=zone_names[0])
        if len(zone2_data) > 0 and len(zone_names) > 1:
            plt.plot(time_data, zone2_data, label=zone_names[1])
        plt.xlabel("Frame")
        plt.ylabel("People Count")
        plt.title("Live Population Analytics")
        plt.legend()
        plt.show()

cap.release()
cv2.destroyAllWindows()