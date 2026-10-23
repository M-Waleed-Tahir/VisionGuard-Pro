# 🛡️ VisionGuard Pro — Industrial Safety SOC Platform

**VisionGuard Pro** is an industrial-grade, real-time computer vision platform designed for hazardous workspace monitoring, PPE compliance enforcement, and automated safety audit logging. Powered by **YOLOv8**, **OpenCV**, and **Streamlit**, it provides Security Operations Centers (SOC) with real-time audio-visual hazard alerts and instant analytical insights.

---

## 🌟 Key Features

* **Multi-Input Surveillance Pipeline:** Supports high-throughput **Single Image Input**, fast **Video Stream Processing**, and real-time **Live Webcam Feed**.
* **Dynamic Model Engine Support:** Native support for loading custom-trained industrial PPE weights (`weights/ppe_yolov8n.pt`) with fallback to standard safety models.
* **Peak Concurrent Object Analytics:** Implements per-frame peak object counting logic to prevent frame-accumulation count inflation during video and webcam stream analysis.
* **Automated Audio-Visual Hazard Alerts:** Instant red alert banner flashing paired with HTML5 audio beep triggers upon detecting hazardous conditions or PPE non-compliance.
* **Noise Mitigation Pipeline:** Intelligent class filtering logic to ignore non-critical background object false-positives in industrial environments.
* **One-Click Audit CSV Export:** Automatically compiles structured session summary reports exportable as CSV files for compliance audits.

---

## 🛠️ Tech Stack

* **Core Language:** Python 3.10+
* **Computer Vision Engine:** OpenCV, Ultralytics YOLOv8
* **Frontend / SOC Interface:** Streamlit
* **Data Processing & Analytics:** Pandas, Plotly Express
* **Image Utilities:** PIL (Pillow), NumPy

---

## 📁 Repository Structure

```text
VisionGuard-Pro/
│
├── app.py                  # Main Streamlit SOC Application Engine
├── requirements.txt        # System Dependencies & Packages
├── README.md               # Documentation & Setup Guide
│
├── src/
│   ├── detector.py         # YOLO Processing Engine & Bounding Box Scaler
│   └── utils.py            # Peak Object Counting & Analytics Formatter
│
└── weights/
    └── ppe_yolov8n.pt      # Custom Industrial PPE Weights Directory
