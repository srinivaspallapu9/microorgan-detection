import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
from collections import Counter
import plotly.graph_objects as go
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Microorganism AI Lab",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# Fixed Detection Settings
# -----------------------------
CONF_THRESHOLD = 0.75
IOU_THRESHOLD = 0.45
MODEL_PATH = "runs/detect/microorganism_yolov8/weights/best.pt"

# -----------------------------
# Premium CSS
# -----------------------------
st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at top left, rgba(0,229,255,0.10), transparent 30%),
        radial-gradient(circle at bottom right, rgba(124,77,255,0.14), transparent 30%),
        linear-gradient(135deg, #050816 0%, #0b1023 50%, #050816 100%);
    color: #ffffff;
}

header[data-testid="stHeader"] {
    background: transparent;
}

section[data-testid="stSidebar"] {
    display: none;
}

.main-title {
    font-size: 4rem;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg, #00e5ff, #7c4dff, #00e5ff);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: glowFlow 6s linear infinite;
    margin-bottom: 0.5rem;
    letter-spacing: 2px;
}

.subtitle {
    text-align: center;
    color: #90caf9;
    font-size: 1.15rem;
    margin-bottom: 2rem;
    opacity: 0.95;
}

@keyframes glowFlow {
    0% { background-position: 0% center; }
    100% { background-position: 200% center; }
}

.hero-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.12);
    backdrop-filter: blur(20px);
    border-radius: 26px;
    padding: 1.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 0 40px rgba(0,229,255,0.08);
}

.feature-row {
    display: flex;
    justify-content: center;
    gap: 12px;
    flex-wrap: wrap;
}

.feature-pill {
    padding: 8px 16px;
    border-radius: 999px;
    background: rgba(0,229,255,0.08);
    border: 1px solid rgba(0,229,255,0.25);
    color: #b3f5ff;
    font-size: 0.85rem;
    font-weight: 600;
}

.card-container {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.12);
    backdrop-filter: blur(20px);
    border-radius: 22px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow:
        0 8px 32px rgba(0,0,0,0.3),
        0 0 25px rgba(0,229,255,0.08);
}

.stat-box {
    background: rgba(0,229,255,0.06);
    border: 1px solid rgba(0,229,255,0.18);
    border-radius: 18px;
    padding: 1.2rem;
    backdrop-filter: blur(14px);
    box-shadow: 0 0 25px rgba(0,229,255,0.08);
}

.ai-description {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 1.2rem;
    line-height: 1.8;
    color: #d0d7ff;
    backdrop-filter: blur(12px);
}

.stButton > button {
    width: 100%;
    border-radius: 16px;
    border: none;
    padding: 0.9rem 1rem;
    font-size: 1rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00e5ff, #7c4dff);
    color: white;
    box-shadow:
        0 0 20px rgba(0,229,255,0.25),
        0 0 40px rgba(124,77,255,0.18);
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: scale(1.03) translateY(-2px);
    box-shadow:
        0 0 30px rgba(0,229,255,0.45),
        0 0 60px rgba(124,77,255,0.35);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background: rgba(255,255,255,0.03);
    border-radius: 18px;
    padding: 10px;
    backdrop-filter: blur(12px);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 14px;
    padding: 12px 22px;
    color: #b0bec5;
    font-weight: 700;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #00e5ff, #7c4dff);
    color: white !important;
    box-shadow: 0 0 25px rgba(0,229,255,0.35);
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04);
    border: 2px dashed rgba(0,229,255,0.3);
    border-radius: 18px;
    padding: 1rem;
    backdrop-filter: blur(12px);
}

.footer {
    text-align: center;
    padding: 2rem;
    margin-top: 3rem;
    color: #90caf9;
    opacity: 0.85;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="main-title">🧬 MICROORGANISM AI LAB</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Next-Generation YOLOv8 Powered Detection & Real-Time Biological Intelligence Platform</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="hero-card">
    <div class="feature-row">
        <div class="feature-pill">🔬 Image Detection</div>
        <div class="feature-pill">🎥 Video Analysis</div>
        <div class="feature-pill">📷 Live Webcam</div>
        <div class="feature-pill">📊 Detection Analytics</div>
        <div class="feature-pill">🧬 3D Visualization</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Descriptions
# -----------------------------
organism_descriptions = {
    "Amoeba": "**Taxonomy:** Phylum Rhizopoda\n\n**Characteristics:** Unicellular protozoan with irregular shape. Moves via pseudopodia.\n\n**Habitat:** Freshwater environments and moist soil.",
    "Euglena": "**Taxonomy:** Phylum Euglenozoa\n\n**Characteristics:** Flagellated unicellular organism with plant and animal-like features.\n\n**Habitat:** Freshwater and nutrient-rich water.",
    "Rod_bacteria": "**Morphology:** Rod-shaped bacteria, also called bacilli.\n\n**Characteristics:** Cylindrical elongated cells.\n\n**Examples:** E. coli, Bacillus species.",
    "Spherical_bacteria": "**Morphology:** Spherical bacteria, also called cocci.\n\n**Characteristics:** Round-shaped cells found singly, in pairs, chains, or clusters.",
    "Spiral_bacteria": "**Morphology:** Spiral or helical bacteria.\n\n**Characteristics:** Spiral shape helps movement through viscous environments."
}

# -----------------------------
# Helper Functions
# -----------------------------
def create_3d_model(organism_name):
    if organism_name == "Spherical_bacteria":
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 30)
        x = 2 * np.outer(np.cos(u), np.sin(v))
        y = 2 * np.outer(np.sin(u), np.sin(v))
        z = 2 * np.outer(np.ones(np.size(u)), np.cos(v))
        fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, showscale=False)])
        title = "Spherical Bacteria - Cocci"

    elif organism_name == "Spiral_bacteria":
        t = np.linspace(0, 6 * np.pi, 120)
        x = 3 * np.cos(t)
        y = 3 * np.sin(t)
        z = 1.2 * t / (6 * np.pi)
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="markers"))
        title = "Spiral Bacteria - Spirillum"

    elif organism_name == "Rod_bacteria":
        x = np.linspace(-4, 4, 80)
        y = 0.5 * np.sin(x)
        z = 0.5 * np.cos(x)
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="markers"))
        title = "Rod Bacteria - Bacillus"

    else:
        points = np.random.randn(150, 3)
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers"
        ))
        title = organism_name.replace("_", " ")

    fig.update_layout(
        title=title,
        height=350,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    return fig


def count_objects(results):
    if results[0].boxes is None or len(results[0].boxes) == 0:
        return Counter()

    names = model.names
    class_ids = results[0].boxes.cls.tolist()
    detected = [names[int(cls_id)] for cls_id in class_ids]
    return Counter(detected)


def draw_count_text(frame, counts):
    y = 35

    if not counts:
        cv2.putText(frame, "No microorganisms detected", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    else:
        cv2.putText(frame, "DETECTED:", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 212, 255), 3)
        y += 40

        for cls_name, count in counts.items():
            cv2.putText(frame, f"{cls_name}: {count}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 100), 3)
            y += 40

    return frame


def show_image_analysis(counts, results):
    if not counts:
        st.markdown(
            '<div class="card-container">⚠️ No microorganisms detected in this image.</div>',
            unsafe_allow_html=True
        )
        return

    st.markdown("### 📊 Detection Count")

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = go.Figure(data=[
            go.Bar(
                x=list(counts.keys()),
                y=list(counts.values()),
                text=list(counts.values()),
                textposition="auto",
                marker=dict(color="#00e5ff")
            )
        ])
        fig.update_layout(
            xaxis_title="Organism Type",
            yaxis_title="Count",
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff")
        )
        st.plotly_chart(fig)

    with col2:
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        st.markdown(f"### {sum(counts.values())}")
        st.markdown("**Total Organisms**")
        st.markdown("---")
        st.markdown(f"### {len(counts)}")
        st.markdown("**Species Detected**")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 🔬 Detailed Organism Analysis")

    for organism, count in counts.items():
        with st.expander(f"▶ {organism.replace('_', ' ').title()} (×{count})"):
            col1, col2 = st.columns(2)

            with col1:
                desc = organism_descriptions.get(
                    organism,
                    "Detailed organism information not available."
                )
                st.markdown('<div class="ai-description">', unsafe_allow_html=True)
                st.markdown(desc)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.plotly_chart(create_3d_model(organism))

    st.markdown("### 📈 Detection Confidence")

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        st.write(f"**{model.names[cls_id]}** : `{conf:.2f}`")


def show_video_analysis(counts):
    if not counts:
        st.markdown(
            '<div class="card-container">⚠️ No microorganisms detected in this video.</div>',
            unsafe_allow_html=True
        )
        return

    st.markdown("### 📊 Frame-wise Detection Frequency")

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = go.Figure(data=[
            go.Bar(
                x=list(counts.keys()),
                y=list(counts.values()),
                text=list(counts.values()),
                textposition="auto",
                marker=dict(color="#7c4dff")
            )
        ])
        fig.update_layout(
            xaxis_title="Organism Type",
            yaxis_title="Frame-wise Frequency",
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff")
        )
        st.plotly_chart(fig)

    with col2:
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        st.markdown(f"### {sum(counts.values())}")
        st.markdown("**Total Frame Detections**")
        st.markdown("---")
        st.markdown(f"### {len(counts)}")
        st.markdown("**Species Detected**")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 🔬 Detailed Organism Analysis")

    for organism, count in counts.items():
        with st.expander(f"▶ {organism.replace('_', ' ').title()} (×{count})"):
            desc = organism_descriptions.get(
                organism,
                "Detailed organism information not available."
            )
            st.markdown('<div class="ai-description">', unsafe_allow_html=True)
            st.markdown(desc)
            st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs([
    "🖼️ Image Analysis",
    "🎥 Video Processing",
    "📷 Live Webcam"
])

# -----------------------------
# Image Tab
# -----------------------------
with tab1:
    st.markdown("### Upload Specimen")

    uploaded_image = st.file_uploader(
        "Choose a microscope image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="collapsed"
    )

    show_original = st.checkbox("Show original image", value=True)

    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image).convert("RGB")
        except Exception as e:
            st.error(f"Error loading image: {e}")
            st.stop()

        image_np = np.array(image)

        col1, col2 = st.columns(2)

        with col1:
            if show_original:
                st.markdown("**Original Image**")
                st.image(image)

        if st.button("🔍 Analyze Image"):
            with st.spinner("Running YOLOv8 detection..."):
                results = model(
                    image_np,
                    conf=CONF_THRESHOLD,
                    iou=IOU_THRESHOLD
                )

            annotated = results[0].plot()
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            counts = count_objects(results)

            with col2:
                st.markdown("**Detection Results**")
                st.image(annotated_rgb)

            detected_image_path = "detected_image.jpg"
            cv2.imwrite(
                detected_image_path,
                cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
            )

            # REPLACED IMAGE BLOCK
            with open(detected_image_path, "rb") as file:
                st.download_button(
        label="📥 Download Annotated Image",
        data=file,
        file_name="detected_image.jpg",
        mime="image/jpeg"
    )

            show_image_analysis(counts, results)

# -----------------------------
# Video Tab
# -----------------------------
with tab2:
    st.markdown("### Upload Video Sample")

    uploaded_video = st.file_uploader(
        "Choose a microscope video",
        type=["mp4", "avi", "mov"],
        label_visibility="collapsed"
    )

    if uploaded_video is not None:
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_input.write(uploaded_video.read())
        temp_input.close()

        st.video(temp_input.name)

        if st.button("🎬 Process Video"):
            cap = cv2.VideoCapture(temp_input.name)

            if not cap.isOpened():
                st.error("Could not open video file.")
                st.stop()

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            if fps == 0:
                fps = 25

            output_path = "detected_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            total_counts = Counter()
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            progress = st.progress(0)
            status = st.empty()

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                results = model(
                    frame,
                    conf=CONF_THRESHOLD,
                    iou=IOU_THRESHOLD
                )

                annotated_frame = results[0].plot()
                counts = count_objects(results)
                total_counts.update(counts)

                annotated_frame = draw_count_text(annotated_frame, counts)
                out.write(annotated_frame)

                frame_count += 1

                if total_frames > 0:
                    progress.progress(min(frame_count / total_frames, 1.0))

                status.text(f"Processing frame {frame_count}/{total_frames}")

            cap.release()
            out.release()

            progress.empty()
            status.empty()

            st.success("✅ Video processing complete!")
            st.video(output_path)

            # REPLACED VIDEO BLOCK
            with open(output_path, "rb") as file:
                st.download_button(
        label="📥 Download Processed Video",
        data=file,
        file_name="detected_video.mp4",
        mime="video/mp4"
    )

            show_video_analysis(total_counts)

# -----------------------------
# Webcam Tab
# -----------------------------
with tab3:
    st.markdown("### Live Microscope Feed")
    st.info("📹 Browser-based real-time detection. Click START and allow camera permission.")

    class YOLOVideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")

            results = model(
                img,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD
            )

            annotated_frame = results[0].plot()
            counts = count_objects(results)
            annotated_frame = draw_count_text(annotated_frame, counts)

            return av.VideoFrame.from_ndarray(
                annotated_frame,
                format="bgr24"
            )

    webrtc_streamer(
        key="microorganism-webcam",
        video_processor_factory=YOLOVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    <div class="footer">
    🚀 Advanced AI-Powered Microorganism Detection Laboratory<br>
    Built with YOLOv8 • Streamlit • Computer Vision • Real-Time Analysis
    </div>
    """,
    unsafe_allow_html=True
)