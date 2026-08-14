# 👥 Smart Crowd Monitoring System

An AI-based **People Counting and Crowd Monitoring System** using **YOLOv8, OpenCV, Python, and Flask**.

The system detects people from video, counts them, monitors different zones, tracks entry and exit, and provides a simple web dashboard for viewing crowd information.

---

## 🚀 Features

* 👤 **People Detection** – Detects people using YOLOv8.
* 🔢 **People Counting** – Counts the number of people detected.
* 🚪 **Entry & Exit Detection** – Tracks people entering and leaving.
* 📍 **Zone Monitoring** – Monitors people in different areas.
* ⚠️ **Overcrowding Alert** – Detects when the crowd exceeds the limit.
* 📊 **Dashboard** – Displays crowd information through a Flask web interface.
* 📝 **Data Logging** – Stores crowd and system information.
* 📈 **Crowd Analytics** – Stores data for further analysis.
* 🎥 **Video Monitoring** – Processes video for real-time detection.

---

## 🧠 How It Works

The system follows these basic steps:

```text
🎥 Video Input
      ↓
🤖 YOLOv8 Person Detection
      ↓
👤 Detect People
      ↓
🔢 Count People
      ↓
🚪 Entry / Exit Detection
      ↓
📍 Zone Monitoring
      ↓
⚠️ Crowd Limit Check
      ↓
📊 Flask Dashboard
      ↓
📝 Save Data & Logs
```

---

## 🛠️ Technologies Used

| Technology    | Purpose                    |
| ------------- | -------------------------- |
| 🐍 Python     | Main programming language  |
| 🤖 YOLOv8     | Person detection           |
| 👁️ OpenCV    | Video and image processing |
| 🌐 Flask      | Web dashboard              |
| 📊 Matplotlib | Data visualization         |
| 🔢 NumPy      | Numerical operations       |
| 📄 CSV        | Data storage               |
| 📋 JSON       | Zone configuration         |
| 🌐 HTML/CSS   | Dashboard interface        |

---

## 📂 Project Structure

```text
Smart-Crowd-Monitoring-System/
│
├── 📁 templates/
│   └── Dashboard HTML files
│
├── 📄 app.py
├── 📄 README.md
├── 📄 .gitignore
│
├── 📊 crowd_data.csv
├── 📝 system_log.txt
├── 📊 zone_log.csv
├── ⚙️ zones.json
│
└── 🤖 yolov8n.pt
```

> **Note:** The exact files in your repository may change as the project is updated.

---

## 🤖 YOLOv8

This project uses **YOLOv8** for detecting people in video frames.

YOLOv8 is a real-time object detection model that can identify objects and their locations in an image or video.

For this project, the main object of interest is:

```text
👤 Person
```

The detected person's bounding box is used to calculate their position and perform counting and zone monitoring.

---

## 👥 People Counting

The system detects people in each video frame.

For every detected person:

1. 👤 A person is detected.
2. 📦 A bounding box is created.
3. 📍 The center position is calculated.
4. 🔢 The person is included in the current count.
5. 📊 The count is displayed on the dashboard.

Example:

```text
Total People: 8
```

---

## 🚪 Entry & Exit Detection

The system uses a virtual counting line to detect movement.

### ➡️ Entry

When a person moves across the line in the entry direction:

```text
Entry Count +1
```

### ⬅️ Exit

When a person moves across the line in the opposite direction:

```text
Exit Count +1
```

This provides basic information about the movement of people.

---

## 📍 Zone Monitoring

The system supports different monitoring zones.

For example:

```text
📍 Entrance
📍 Retail Area
```

Zone information is stored in:

```text
zones.json
```

The system checks the position of detected people and determines which zone they are currently in.

This can be useful for monitoring crowd levels in different areas.

---

## ⚠️ Overcrowding Detection

A maximum crowd limit can be defined.

For example:

```text
Maximum Limit = 5
```

If the number of detected people becomes greater than the configured limit:

```text
⚠️ WARNING: OVERCROWDING
```

The system can record the event in the system log.

This feature can be useful for:

* 🏬 Shopping areas
* 🎓 College campuses
* 🚉 Railway stations
* ✈️ Airports
* 🎫 Events
* 🏢 Offices

---

## 🌐 Web Dashboard

The project uses **Flask** to provide a web-based dashboard.

The dashboard can display important crowd information such as:

```text
👥 Total People
🚪 Entry
⬅️ Exit
📍 Zone Information
⚠️ Crowd Status
📊 Crowd Data
```

The dashboard is located in the:

```text
templates/
```

folder.

---

## 📊 Data Storage

The system stores crowd information in CSV files.

### `crowd_data.csv`

Stores general crowd information.

Example:

```text
Time,Zone,Entry,Exit,Total
10:30,All,5,2,8
10:31,All,6,2,9
```

### `zone_log.csv`

Stores information related to monitored zones.

---

## 📝 System Logs

System events are stored in:

```text
system_log.txt
```

Example events:

```text
Camera Started
Entry detected
Exit detected
Overcrowding Alert Triggered
System Shutdown
```

These logs can help in checking what happened during system execution.

---

## ⚙️ Zone Configuration

Zone information is stored in:

```text
zones.json
```

This makes it easier to configure or modify the monitoring areas without changing the main program.

Example concept:

```text
Entrance
Retail Area
```

---

## 💻 Installation

### 1️⃣ Clone the Repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
```

### 2️⃣ Open the Project Folder

```bash
cd Smart-Crowd-Monitoring-System
```

### 3️⃣ Install Required Libraries

```bash
pip install ultralytics opencv-python flask numpy matplotlib
```

If you have a `requirements.txt` file, you can use:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

Run the Flask application:

```bash
python app.py
```

After the server starts, open the local address shown in the terminal.

Usually:

```text
http://127.0.0.1:5000
```

The Flask dashboard will then open in your browser.

---

## 📸 System Output

The system provides two main outputs:

### 🎥 Detection Output

The video displays:

* 👤 Detected people
* 📦 Bounding boxes
* 📍 Monitoring zones
* 🚪 Counting line
* ⚠️ Crowd warnings

### 🌐 Dashboard Output

The web dashboard displays:

* 👥 People count
* 🚪 Entry count
* ⬅️ Exit count
* 📍 Zone information
* ⚠️ Crowd status
* 📊 Crowd analytics

---

## 📈 Project Workflow

```text
🎥 Input Video
      ↓
🤖 YOLOv8
      ↓
👤 Person Detection
      ↓
📦 Bounding Boxes
      ↓
📍 Position Calculation
      ↓
🔢 People Counting
      ↓
🚪 Entry / Exit
      ↓
📍 Zone Monitoring
      ↓
⚠️ Overcrowding Check
      ↓
📝 Logging
      ↓
📊 Data Storage
      ↓
🌐 Flask Dashboard
```

---

## 🎯 Applications

This project can be used for:

* 🏬 Shopping malls
* 🛒 Retail stores
* 🎓 Educational institutions
* 🚉 Railway stations
* ✈️ Airports
* 🎫 Events
* 🏢 Offices
* 🏙️ Smart-city monitoring

---

## ✅ Advantages

* 🤖 Automated people detection
* 🔢 Automatic people counting
* 🚪 Entry and exit monitoring
* 📍 Zone-based monitoring
* ⚠️ Overcrowding detection
* 🌐 Web-based dashboard
* 📝 Automatic logging
* 📊 Data storage for analysis
* 💻 Easy to run and modify

---

## ⚠️ Limitations

The accuracy of the system can depend on:

* 📷 Camera angle
* 💡 Lighting conditions
* 👥 Number of people
* 🚶 People overlapping with each other
* 🎥 Video quality
* 🤖 YOLO detection accuracy

In highly crowded scenes, people may overlap, which can make detection and counting more difficult.

---

## 🔮 Future Improvements

The project can be improved by adding:

* 📹 Live CCTV/IP camera support
* 🎯 Advanced multi-object tracking
* 🔥 Crowd density heatmaps
* 📊 More advanced analytics
* 📱 Mobile notifications
* ☁️ Cloud deployment
* 🗄️ Database integration
* 📧 Real-time alerts
* 📈 Crowd prediction
* 📷 Multiple camera support

---

## 📌 Project Goal

The main goal of this project is to develop a simple AI-based system that can automatically monitor people and provide useful crowd information.

Instead of manually watching CCTV footage, the system uses **AI and computer vision** to automatically detect and count people.

---

## 🏁 Conclusion

The **Smart Crowd Monitoring System** demonstrates how AI and computer vision can be used for automated people counting and crowd monitoring.

Using **YOLOv8**, the system detects people from video and provides information about crowd size, entry, exit, and different monitoring zones.

The **Flask dashboard** provides a simple way to view the collected information, while CSV files and system logs help store and analyze the monitoring data.

The project can be further extended into a complete real-time crowd management solution using advanced tracking, live cameras, databases, notifications, and predictive analytics.

---

## 👨‍💻 Project Information

**Project Name:** Smart Crowd Monitoring System

**Domain:** Artificial Intelligence & Computer Vision

**Main Technologies:**

```text
Python
YOLOv8
OpenCV
Flask
NumPy
Matplotlib
HTML/CSS
CSV
JSON
```

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

## 📄 License

This project is developed for **educational and project purposes**.

Please check the applicable license requirements of the third-party libraries and models used in this project.
