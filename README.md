<div align="center">

# 🔬 Microorganism Detection System
### Real-Time AI Detection using YOLOv8 + Streamlit

An AI-powered real-time microorganism detection and analysis system built using **YOLOv8**, **Streamlit**, and **Computer Vision** technologies. Detects microorganisms from images and videos, draws bounding boxes, shows confidence scores, performs counting, and provides organism descriptions through an interactive web application.

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://microorgan-detection.streamlit.app/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Srinivas-pallapu/microorgan-detection)

</div>

---

## 📌 Project Overview

This system uses a custom-trained **YOLOv8 Object Detection Model**, trained on microorganism datasets, to identify and analyze microorganisms from microscope images and videos.

**Key Capabilities:**

| Feature | Description |
|---|---|
| ✅ Image Detection | Detects microorganisms in static images |
| ✅ Video Detection | Frame-by-frame video analysis |
| ✅ Bounding Box Visualization | Visual markers on detected organisms |
| ✅ Confidence Score Analysis | Per-detection confidence levels |
| ✅ Organism Counting | Automatic count of each organism type |
| ✅ Detection Dashboard | Interactive analytics view |
| ✅ Download Results | Export processed images/videos |
| ✅ Interactive UI | Built with Streamlit |

---

## 🧠 Detected Classes

- 🦠 Amoeba
- 🦠 Euglena
- 🦠 Rod Bacteria
- 🦠 Spherical Bacteria
- 🦠 Spiral Bacteria

---

## 🏗️ System Architecture

<div align="center">
  <img src="assets/project_diagram.png" alt="System Architecture" width="700"/>
</div>

---

## ⚙️ Process Flow
Input Image/Video
↓

Preprocessing
↓

YOLOv8 Detection
↓
Post-processing
↓
Bounding Boxes + Counts + Confidence Scores
↓
Visualization Dashboard
---

## 🚀 Features

### 🔍 Detection
- Detects multiple microorganisms simultaneously
- Real-time object detection using YOLOv8

### 📊 Visualization
- Detection count charts
- Interactive analysis dashboard

### 📖 Descriptions
- Provides information about detected microorganisms

### 🎥 Video Analysis
- Frame-wise microorganism detection
- Video annotation support

### ⬇️ Download Results
- Download processed image/video outputs

### ⚙️ Settings
- Adjustable confidence threshold
- Configurable detection parameters

---

## 🛠️ Tech Stack

| Technology | Usage |
|---|---|
| ![YOLOv8](https://img.shields.io/badge/YOLOv8-FF4500?style=flat-square) | Object Detection |
| ![Python](https://img.shields.io/badge/Python-3670A0?style=flat-square&logo=python&logoColor=ffdd54) | Backend Logic |
| ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) | Web Application |
| ![OpenCV](https://img.shields.io/badge/OpenCV-white.svg?style=flat-square&logo=opencv&logoColor=black) | Image/Video Processing |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) | Numerical Operations |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) | Data Handling |
| ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white) | Visualization |

---

## 📂 Project Structure
microorgan-detection/
│

├── app.py
├── requirements.txt
├── detected_video.mp4
├── runs/
│   └── detect/
│       └── microorganism_yolov8/
│           └── weights/
│               ├── best.pt
│               └── last.pt
│
├── datasets/
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
│
├── assets/
│   └── project_diagram.png
│
└── README.md
---

## 🧪 Model Training

The YOLOv8 model was trained on a custom microorganism dataset with annotated bounding boxes.

**Training Command:**

```bash
yolo detect train data=datasets/data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

---

## 📈 Model Performance

| Metric | Value |
|---|---|
| mAP50 | **80.3%** |
| Precision | **81.7%** |
| Recall | **71.7%** |

---

## ▶️ Run Locally

**1️⃣ Clone Repository**
```bash
git clone https://github.com/Srinivas-pallapu/microorgan-detection.git
```

**2️⃣ Open Project**
```bash
cd microorgan-detection
```

**3️⃣ Install Requirements**
```bash
pip install -r requirements.txt
```

**4️⃣ Run Streamlit App**
```bash
python -m streamlit run app.py
```

---

## 📸 Output Examples

**Image Detection**
- Bounding boxes
- Class labels
- Confidence scores

**Video Detection**
- Frame-wise detection
- Organism counting
- Annotated video output

---

## 🎯 Project Goal

To build an AI-powered intelligent microorganism analysis system capable of detecting, classifying, and analyzing microorganisms from microscope images and videos with high accuracy.

---

## ⭐ Future Improvements

- 🔄 Real-time tracking using ByteTrack
- 🔢 Unique microorganism counting
- 🔬 Live microscope integration
- 🧊 3D visualization
- ☁️ Cloud deployment optimization
- 🤖 AI-generated microorganism insights

---

## 👨‍💻 Author

**Srinivasa Rao Pallapu**  
B.Tech CSE (AI & ML) — VVIT University

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Srinivas-pallapu)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/srinivaspallapu9/)

</div>
