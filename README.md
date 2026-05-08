# 🔬 Microorganism Detection System using YOLOv8 + Streamlit

An AI-powered real-time microorganism detection and analysis system built using **YOLOv8**, **Streamlit**, and **Computer Vision** technologies.

The project detects microorganisms from images and videos, draws bounding boxes, shows confidence scores, performs counting, and provides organism descriptions through an interactive web application.

---

## 🌐 Live Demo

🚀 https://microorgan-detection.streamlit.app/

---

## 💻 GitHub Repository

🔗 https://github.com/Srinivas-pallapu/microorgan-detection

---

# 📌 Project Overview

This system uses a custom-trained **YOLOv8 Object Detection Model** trained on microorganism datasets to identify and analyze microorganisms from microscope images and videos.

The application supports:

✅ Image Detection  
✅ Video Detection  
✅ Bounding Box Visualization  
✅ Confidence Score Analysis  
✅ Organism Counting  
✅ Detection Dashboard  
✅ Download Processed Results  
✅ Interactive Streamlit Interface  

---

# 🧠 Detected Classes

- Amoeba
- Euglena
- Rod_bacteria
- Spherical_bacteria
- Spiral_bacteria

---

# 🏗️ System Architecture

![System Architecture](assets/project_diagram.png)

---

# ⚙️ Process Flow

```text
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
🚀 Features
🔍 Detection
Detects multiple microorganisms simultaneously
Real-time object detection using YOLOv8
📊 Visualization
Detection count charts
Interactive analysis dashboard
📖 Descriptions
Provides information about detected microorganisms
🎥 Video Analysis
Frame-wise microorganism detection
Video annotation support
⬇️ Download Results
Download processed image/video outputs
⚙️ Settings
Adjustable confidence threshold
Configurable detection parameters
🛠️ Tech Stack
Technology	Usage
YOLOv8	Object Detection
Python	Backend Logic
Streamlit	Web Application
OpenCV	Image/Video Processing
NumPy	Numerical Operations
Pandas	Data Handling
Plotly	Visualization
📂 Project Structure
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
🧪 Model Training

The YOLOv8 model was trained on a custom microorganism dataset with annotated bounding boxes.

Training Command
yolo detect train data=datasets/data.yaml model=yolov8n.pt epochs=50 imgsz=640
📈 Model Performance
Metric	Value
mAP50	80.3%
Precision	81.7%
Recall	71.7%
▶️ Run Locally
1️⃣ Clone Repository
git clone https://github.com/Srinivas-pallapu/microorgan-detection.git
2️⃣ Open Project
cd microorgan-detection
3️⃣ Install Requirements
pip install -r requirements.txt
4️⃣ Run Streamlit App
python -m streamlit run app.py
📸 Output Examples
Image Detection
Bounding boxes
Class labels
Confidence scores
Video Detection
Frame-wise detection
Organism counting
Annotated video output
🎯 Project Goal

To build an AI-powered intelligent microorganism analysis system capable of detecting, classifying, and analyzing microorganisms from microscope images and videos with high accuracy.

👨‍💻 Author
Srinivasa Rao Pallapu
B.Tech CSE (AI & ML)
VVIT University
Connect With Me
GitHub: https://github.com/Srinivas-pallapu
LinkedIn: https://www.linkedin.com/in/srinivaspallapu9/
⭐ Future Improvements
Real-time tracking using ByteTrack
Unique microorganism counting
Live microscope integration
3D visualization
Cloud deployment optimization
AI-generated microorganism insights
