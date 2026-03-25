import cv2
import numpy as np
import json
import os
import random
from ultralytics import YOLO

# -------------------------------
# CONFIG
# -------------------------------
ZONE_FILE = "zones.json"  # created in milestone_1.py

# -------------------------------
# LOAD YOLO MODEL
# -------------------------------
model = YOLO("yolov8n.pt")


# -------------------------------
# HELPERS FOR ZONES
# -------------------------------
def random_color():
    """Fallback color if zone has no color saved."""
    return [random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255)]


def load_zones():
    """Load zones created in milestone_1 from JSON file."""
    if os.path.exists(ZONE_FILE):
        with open(ZONE_FILE, "r") as f:
            data = json.load(f)
            # make sure each zone has color
            for z in data:
                if "color" not in z or not z["color"]:
                    z["color"] = random_color()
            return data
    else:
        print("No zones.json found. Please create zones in milestone_1.py first.")
        return []


zones = load_zones()

# -------------------------------
# START CAMERA
# -------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not opening!")
    exit()

cv2.namedWindow("Milestone 2", cv2.WINDOW_NORMAL)

# -------------------------------
# MAIN LOOP
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    total_count = 0
    zone_counts = [0 for _ in zones]

    # -------------------------------
    # DETECT PEOPLE
    # -------------------------------
    results = model(frame)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])

            # Class 0 = person
            if cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                total_count += 1

                # Draw person box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"Person {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)

                # Check zone membership (center point method)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                for i, zone in enumerate(zones):
                    zx1, zy1, zx2, zy2 = zone["coords"]
                    if zx1 < cx < zx2 and zy1 < cy < zy2:
                        zone_counts[i] += 1

    # -------------------------------
    # DRAW ZONES (from milestone_1)
    # -------------------------------
    for i, zone in enumerate(zones):
        x1, y1, x2, y2 = zone["coords"]
        color = tuple(zone.get("color", random_color()))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        label = zone.get("name", f"Zone {i+1}")
        cv2.putText(frame, f"{label}: {zone_counts[i]}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)

    # -------------------------------
    # TOP HEADER BAR
    # -------------------------------
    header = np.zeros((80, frame.shape[1], 3), dtype=np.uint8)
    header[:] = (40, 40, 120)

    cv2.putText(header,
                "Milestone 2: People Detection & Counting",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2)

    # Combine header + frame
    frame = np.vstack((header, frame))

    # -------------------------------
    # TOTAL COUNT BOX
    # -------------------------------
    cv2.rectangle(frame,
                  (frame.shape[1] - 220, 20),
                  (frame.shape[1] - 20, 70),
                  (0, 255, 0),
                  -1)

    cv2.putText(frame,
                f"Total: {total_count}",
                (frame.shape[1] - 200, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2)

    # -------------------------------
    # FOOTER BADGE
    # -------------------------------
    cv2.putText(frame,
                "Real-Time | AI Powered",
                (30, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2)

    cv2.imshow("Milestone 2", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()