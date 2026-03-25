import cv2
import numpy as np
import json
import os
import random

ZONE_FILE = "zones.json"

zones = []
drawing = False
start_point = None

# ----------------------------
# Generate Random Color
# ----------------------------
def random_color():
    return [random.randint(50,255),
            random.randint(50,255),
            random.randint(50,255)]

# ----------------------------
# Load Zones
# ----------------------------
def load_zones():
    if os.path.exists(ZONE_FILE):
        with open(ZONE_FILE, "r") as f:
            return json.load(f)
    return []

# ----------------------------
# Save Zones
# ----------------------------
def save_zones():
    with open(ZONE_FILE, "w") as f:
        json.dump(zones, f, indent=4)

# ----------------------------
# Mouse Draw Function
# ----------------------------
def draw_rectangle(event, x, y, flags, param):
    global drawing, start_point

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)

        zone = {
            "name": f"Zone {len(zones)+1}", 
            "coords": [start_point[0], start_point[1],
                       end_point[0], end_point[1]],
            "color": random_color(),
            "count": 0
        }

        zones.append(zone)
        # save zones to JSON every time a new one is created
        save_zones()

# ----------------------------
# MAIN
# ----------------------------
zones = load_zones()

cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Camera not opened!")
    exit()
fullscreen = False

cv2.namedWindow("Milestone 1",cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Milestone 1", draw_rectangle)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    header_height = 80

    # -------- HEADER --------
    header = np.zeros((header_height, frame.shape[1], 3), dtype=np.uint8)
    header[:] = (40, 40, 120)

    cv2.putText(header,
                "Milestone 1: Zone Management",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,255,255), 2)

    cv2.putText(header,
                "Q=Quit | D=Delete | R=Reset | S=Save | F=Fullscreen",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200,200,200), 2)

    frame = np.vstack((header, frame))

    # -------- DRAW ZONES --------
    for i, zone in enumerate(zones):
        x1, y1, x2, y2 = zone["coords"]

        y1 += header_height
        y2 += header_height
        if "color" not in zone:
            zone["color"] = random_color()
        color = tuple(zone["color"])

        cv2.rectangle(frame,
                      (x1, y1),
                      (x2, y2),
                      color, 2)

        label = zone.get("name", f"Zone {i+1}")
        cv2.putText(frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color, 2)

    cv2.imshow("Milestone 1", frame)

    key = cv2.waitKey(1) & 0xFF
    

    # Quit
    if key == ord('q'):
        break

    # Delete last zone
    elif key == ord('d'):
        if len(zones) > 0:
            zones.pop()
            save_zones()

    # Reset all zones
    elif key == ord('r'):
        zones.clear()
        if os.path.exists(ZONE_FILE):
            os.remove(ZONE_FILE)

    # Save zone screenshot (zones are already stored in zones.json)
    elif key == ord('s'):
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"zone_{timestamp}.png"
        cv2.imwrite(filename, frame)
        print("Zone Screenshot Saved:", filename)
    
    # Toggle Fullscreen
    elif key == ord('f'):
        fullscreen = not fullscreen
    if fullscreen:
        cv2.setWindowProperty("Milestone 1",
                              cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
    else:
        cv2.setWindowProperty("Milestone 1",
                              cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_NORMAL)

cap.release()
cv2.destroyAllWindows()