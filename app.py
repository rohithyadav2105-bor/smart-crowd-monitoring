import cv2
import numpy as np
import datetime
import os
import time

from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


current_video_source = None
last_alert_time = 0


def log_event(msg):
    with open("system_log.txt", "a") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp} : {msg}\n")

log_event("System Started")


def init_stats():
    return {
        "total": 0,
        "entry": 0,
        "exit": 0,
        "inside": 0,
        "alert": False,
        "history": {
            "time": [],
            "total": [],
            "entry": [],
            "exit": [],
            "inside": []
        }
    }

stats = init_stats()

MAX_LIMIT = 5
line_y = 700
MAX_HISTORY = 50

model = YOLO("yolov8n.pt")


def update_history(time_str, total, entry, exit_count, inside_count):
    stats["history"]["time"].append(time_str)
    stats["history"]["total"].append(total)
    stats["history"]["entry"].append(entry)
    stats["history"]["exit"].append(exit_count)
    stats["history"]["inside"].append(inside_count)

    if len(stats["history"]["time"]) > MAX_HISTORY:
        stats["history"]["time"].pop(0)
        stats["history"]["total"].pop(0)
        stats["history"]["entry"].pop(0)
        stats["history"]["exit"].pop(0)
        stats["history"]["inside"].pop(0)

def generate_frames():
    global stats, current_video_source, last_alert_time

    while current_video_source is None:
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank_frame, "Upload a video to start analysis.",
                    (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (200, 200, 200), 2)

        _, buffer = cv2.imencode('.jpg', blank_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')
        time.sleep(1)

    log_event(f"Started processing video: {os.path.basename(current_video_source)}")

    cap = cv2.VideoCapture(current_video_source)
    prev_centroids = []

    while True:
        success, frame = cap.read()
        if not success:
            log_event("Video playback finished.")
            break

        current_centroids = []
        total_count = 0

        results = model(frame, conf=0.3, imgsz=960)

        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:  
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    total_count += 1
                    current_centroids.append((cx, cy))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        
        used_curr = set()
        for (px, py) in prev_centroids:
            best_idx, best_dist = -1, 999999

            for idx, (cx, cy) in enumerate(current_centroids):
                if idx in used_curr:
                    continue

                dist = (px - cx)**2 + (py - cy)**2
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            if best_idx != -1 and best_dist < 200**2:
                used_curr.add(best_idx)
                cx, cy = current_centroids[best_idx]

                if py < line_y <= cy:
                    stats["entry"] += 1
                    log_event("Entry detected")

                elif py > line_y >= cy:
                    stats["exit"] += 1
                    log_event("Exit detected")

        prev_centroids = current_centroids[:]

        
        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 255), 2)

        
        stats["total"] = total_count
        stats["inside"] = max(0, stats["entry"] - stats["exit"])
        stats["alert"] = stats["inside"] > MAX_LIMIT

        
        current_time_str = datetime.datetime.now().strftime("%H:%M:%S")
        update_history(current_time_str, total_count, stats["entry"], stats["exit"], stats["inside"])

    
        if stats["alert"]:
            cv2.putText(frame, "WARNING: OVERCROWDING",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 3)

            current_time_sec = time.time()
            if (current_time_sec - last_alert_time) > 10:
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = os.path.join(app.config['UPLOAD_FOLDER'], f"alert_{timestamp}.png")

                cv2.imwrite(filename, frame)
                last_alert_time = current_time_sec

                log_event(f"Overcrowding alert! ({stats['inside']} inside)")

        
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    return jsonify(stats)

@app.route('/alerts')
def get_alerts():
    files = os.listdir(app.config['UPLOAD_FOLDER'])

    alert_files = [f for f in files if f.startswith("alert_")]
    alert_files.sort(reverse=True)

    return jsonify(alert_files[:10])

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_video():
    global current_video_source, stats

    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    log_event(f"New file uploaded: {filename}")

    current_video_source = filepath
    stats = init_stats()

    return jsonify({"message": "File uploaded successfully"}), 200


if __name__ == '__main__':
    app.run(debug=True, threaded=True)