import cv2
import numpy as np
import csv
import datetime
import matplotlib.pyplot as plt
from collections import deque
from ultralytics import YOLO
model = YOLO("yolov8n.pt")

# -----------------------------
# LOAD YOLO MODEL
# -----------------------------
model = YOLO("yolov8n.pt")

# -----------------------------
# ZONES
# -----------------------------
zones = {
    "Entrance": (50,150,300,450),
    "Retail Area": (350,150,600,450)
}

zone_counts = {name:0 for name in zones}

# -----------------------------
# ENTRY / EXIT
# -----------------------------
entry_count = 0
exit_count = 0
line_y = 700
# store previous person positions (for simple line‑cross logic)
prev_centroids = []

# -----------------------------
# CROWD LIMIT
# -----------------------------
MAX_LIMIT = 5

# -----------------------------
# DATA STORAGE
# -----------------------------
csv_file = open("crowd_data.csv","a",newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Time","Zone","Entry","Exit","Total"])

# -----------------------------
# LOG FUNCTION
# -----------------------------
def log_event(msg):
    with open("system_log.txt","a") as f:
        f.write(f"{datetime.datetime.now()} : {msg}\n")

log_event("Camera Started")

# -----------------------------
# CAMERA
# -----------------------------
# CAMERA / VIDEO INPUT
# Put your video file name here (same folder as script)
VIDEO_PATH = r"C:\Users\Rohith\OneDrive\Desktop\infosys\counting video.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Camera not opening")
    exit()

cv2.namedWindow("Milestone 4", cv2.WINDOW_NORMAL)

# -----------------------------
# GRAPH DATA
# -----------------------------
time_data=[]
total_data=[]
entry_history = []
exit_history = []
inside_history = []
inside_window = deque(maxlen=100)

frame_id = 0

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    total_count = 0
    zone_counts = {name:0 for name in zones}
    back_side_count = 0   # people above the yellow line
    front_side_count = 0  # people below the yellow line

    # store current frame centroids for entry/exit estimation
    current_centroids = []

    # run YOLO with lower confidence threshold (better for distant/small people)
    results = model(frame, conf=0.3, imgsz=960)

    for r in results:
        for box in r.boxes:

            cls = int(box.cls[0])
            # Class 0 = person (COCO)
            if cls == 0:

                x1,y1,x2,y2 = map(int,box.xyxy[0])

                cx = (x1+x2)//2
                cy = (y1+y2)//2

                total_count += 1

                # keep track of this person's center
                current_centroids.append((cx,cy))

                # back / front side counting (relative to yellow line)
                if cy < line_y:
                    back_side_count += 1
                else:
                    front_side_count += 1

                # draw box
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)

                # zone check
                for name,(zx1,zy1,zx2,zy2) in zones.items():
                    if zx1 < cx < zx2 and zy1 < cy < zy2:
                        zone_counts[name]+=1

    # -----------------------------
    # SIMPLE ENTRY / EXIT COUNTING
    # -----------------------------
    # Match previous centroids to current ones by nearest neighbour
    used_curr = set()
    for (px,py) in prev_centroids:
        best_idx = -1
        best_dist = 999999
        for idx,(cx,cy) in enumerate(current_centroids):
            if idx in used_curr:
                continue
            dist = (px-cx)**2 + (py-cy)**2
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        # if we found a reasonably close match, check if it crossed the counting line
        if best_idx != -1 and best_dist < 200**2:
            used_curr.add(best_idx)
            cx,cy = current_centroids[best_idx]

            # moved from above line to below -> entry
            if py < line_y <= cy:
                entry_count += 1
                log_event("Entry detected")
            # moved from below line to above -> exit
            elif py > line_y >= cy:
                exit_count += 1
                log_event("Exit detected")

    # remember centroids for next frame
    prev_centroids = current_centroids[:]

    # -----------------------------
    # ENTRY / EXIT LINE (yellow boundary)
    # -----------------------------
    cv2.line(frame,(0,line_y),(frame.shape[1],line_y),(0,255,255),2)

    # -----------------------------
    # OVERCROWD ALERT
    # -----------------------------
    alert=False

    if total_count > MAX_LIMIT:
        alert=True

        cv2.putText(frame,
                    "WARNING : OVERCROWDING",
                    (200,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    3)

        filename=f"alert_{datetime.datetime.now().strftime('%H%M%S')}.png"
        cv2.imwrite(filename,frame)

        log_event("Overcrowding Alert Triggered")

    # -----------------------------
    # DASHBOARD
    # -----------------------------
    dashboard = np.zeros((frame.shape[0],300,3),dtype=np.uint8)
    dashboard[:]=(30,30,70)

    y=40

    cv2.putText(dashboard,"Live Statistics",
                (50,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,(255,255,255),2)

    cv2.putText(dashboard,f"Total: {total_count}",
                (20,y+40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,(0,255,0),2)

    y+=80

    cv2.putText(dashboard,f"Entry: {entry_count}",
                (20,y+20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,(255,255,255),2)

    cv2.putText(dashboard,f"Exit: {exit_count}",
                (20,y+60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,(255,255,255),2)

    # people currently inside/front side in this frame
    cv2.putText(dashboard,f"Inside (current): {front_side_count}",
                (20,y+100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,(0,255,255),2)
    # -----------------------------
    # SAVE DATA
    # -----------------------------
    current_time = datetime.datetime.now()
    # single row: overall stats (no per-zone)
    csv_writer.writerow([
        current_time,
        "All",
        entry_count,
        exit_count,
        total_count
    ])

    # -----------------------------
    # GRAPH DATA
    # -----------------------------
    time_data.append(frame_id)
    total_data.append(total_count)
    entry_history.append(entry_count)
    exit_history.append(exit_count)
    inside_history.append(max(entry_count - exit_count, 0))

    # -----------------------------
    # COMBINE SCREEN
    # -----------------------------
    combined = np.hstack((frame,dashboard))

    cv2.imshow("Milestone 4",combined)

    # -----------------------------
    # KEY CONTROLS
    # -----------------------------
    key=cv2.waitKey(1)&0xFF

    if key==ord('q'):
        log_event("System Shutdown")
        break

    elif key==ord('g'):
        # show graph based on in / out data
        plt.figure(figsize=(8,5))
        plt.plot(time_data, total_data, label="Total people")
        plt.plot(time_data, entry_history, label="Entry (cumulative)")
        plt.plot(time_data, exit_history, label="Exit (cumulative)")
        plt.xlabel("Frame")
        plt.ylabel("People")
        plt.title("Crowd analytics")
        plt.legend()
        plt.tight_layout()
        plt.show()

# -----------------------------
# CLOSE SYSTEM
# -----------------------------
csv_file.close()
cap.release()
cv2.destroyAllWindows()