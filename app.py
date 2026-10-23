import os
import cv2
import tempfile
import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px

from src.detector import SafetyDetector
from src.utils import summarize_detections

# 1. Page Config
st.set_page_config(
    page_title="VisionGuard Pro | Industrial AI Surveillance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom Cyberpunk & SOC Theme CSS
st.markdown("""
    <style>
    .stApp { background-color: #0b0f19; color: #e2e8f0; }
    .main-title {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    div[data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 700 !important; color: #38bdf8 !important; }
    .stMetric { background: rgba(30, 41, 59, 0.7); padding: 1rem; border-radius: 12px; border: 1px solid rgba(56, 189, 248, 0.2); }
    section[data-testid="stSidebar"] { background-color: #111827 !important; }
    
    .alert-banner {
        background-color: #ef4444; color: white; padding: 12px;
        border-radius: 8px; font-weight: bold; text-align: center;
        font-size: 1.1rem; box-shadow: 0 0 20px rgba(239, 68, 68, 0.8); margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

def trigger_audio_alert():
    st.components.v1.html("""
        <script>
        var ctx = new (window.AudioContext || window.webkitAudioContext)();
        var osc = ctx.createOscillator();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(880, ctx.currentTime);
        osc.connect(ctx.destination);
        osc.start();
        osc.stop(ctx.currentTime + 0.3);
        </script>
    """, height=0)

def render_csv_download_button(df, filename="Surveillance_Analytics_Report.csv"):
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Analytics CSV Report",
        data=csv_data,
        file_name=filename,
        mime="text/csv",
        type="primary"
    )

# 3. Model Engine Loader
@st.cache_resource
def load_detector():
    return SafetyDetector(model_name="yolov8n.pt")

detector = load_detector()

# 4. Header
st.markdown('<div class="main-title">🛡️ VisionGuard Pro — Industrial Safety SOC</div>', unsafe_allow_html=True)
st.markdown("<p style='color: #94a3b8;'>Real-Time Computer Vision Platform for Industrial Safety & Hazard Monitoring</p>", unsafe_allow_html=True)

# 5. Sidebar Controls
st.sidebar.markdown("### ⚙️ System Controls")
confidence_threshold = st.sidebar.slider("YOLO Confidence Threshold", 0.10, 0.90, 0.45, 0.05)
input_source = st.sidebar.radio("Select Stream Source", ["📷 Single Image Input", "🎥 Video Stream Processing", "📹 Live Webcam Feed"])

st.sidebar.markdown("---")
enable_audio = st.sidebar.checkbox("🔊 Enable Audio Hazard Beep", value=True)

# Safe Engine Status Display
is_ppe_active = getattr(detector, 'is_custom_ppe', False)
if is_ppe_active:
    st.sidebar.success("✅ Engine: Custom Industrial PPE Model Loaded")
else:
    st.sidebar.info("ℹ️ Engine: Standard Safety Model Loaded")

# Session States Initialization
if 'webcam_history' not in st.session_state:
    st.session_state['webcam_history'] = []
if 'webcam_active' not in st.session_state:
    st.session_state['webcam_active'] = False

# 6. Mode 1: Image Processing
if input_source == "📷 Single Image Input":
    uploaded_image = st.file_uploader("Upload Target Frame...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        image_np = np.array(image)

        annotated_frame, detections = detector.process_frame(image_np, confidence=confidence_threshold)

        if len(detections) > 0:
            st.markdown('<div class="alert-banner">🚨 HAZARD / SAFETY ALERTS DETECTED</div>', unsafe_allow_html=True)
            if enable_audio:
                trigger_audio_alert()

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Objects Detected", len(detections))
        m2.metric("Unique Classes", len(set([d['class'] for d in detections])))
        m3.metric("System Status", "ALERTS ACTIVE" if len(detections) > 0 else "CLEAR")

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 📥 Original Input Frame")
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("##### 🎯 Bounding Box Output")
            st.image(annotated_frame, use_container_width=True)

        if detections:
            st.markdown("---")
            summary_df = summarize_detections(detections)
            col_chart, col_table = st.columns([3, 2])
            with col_chart:
                fig = px.bar(summary_df, x="Detected Item", y="Count", color="Detected Item", template="plotly_dark")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            with col_table:
                st.dataframe(summary_df, use_container_width=True)
                render_csv_download_button(summary_df, filename="Image_Detections_Report.csv")

# 7. Mode 2: Video Stream Processing (Fixed Error-Free Video Pipeline)
elif input_source == "🎥 Video Stream Processing":
    uploaded_video = st.file_uploader("Upload Video File...", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        if st.button("🚀 Start Fast Video Stream Analysis", type="primary"):
            # Safe temporary file handle
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            tfile.close()

            cap = cv2.VideoCapture(tfile.name)
            alert_placeholder = st.empty()
            
            col_left, col_center, col_right = st.columns([1, 4, 1])
            with col_center:
                stframe = st.empty()

            all_video_detections = []
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % 2 != 0:
                    continue

                frame_resized = cv2.resize(frame, (640, 360))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                
                annotated_frame, detections = detector.process_frame(frame_rgb, confidence=confidence_threshold)
                
                for d in detections:
                    d_item = d.copy()
                    d_item['frame_id'] = f"v_frame_{frame_count}"
                    all_video_detections.append(d_item)

                if len(detections) > 0:
                    alert_placeholder.markdown('<div class="alert-banner">🚨 LIVE HAZARD / SAFETY ALERTS DETECTED</div>', unsafe_allow_html=True)
                    if enable_audio and frame_count % 10 == 0:
                        trigger_audio_alert()
                else:
                    alert_placeholder.empty()

                stframe.image(annotated_frame, channels="RGB", use_container_width=True)

            cap.release()
            try:
                os.remove(tfile.name)
            except Exception:
                pass

            st.success("🎉 Video Stream Analysis Finished Successfully!")

            if all_video_detections:
                st.markdown("---")
                st.markdown("### 📊 Comprehensive Video Stream Summary Analytics")

                summary_df = summarize_detections(all_video_detections)

                m1, m2, m3 = st.columns(3)
                m1.metric("Peak Concurrent Objects Detected", summary_df['Count'].sum() if not summary_df.empty else 0)
                m2.metric("Unique Classes Encountered", len(summary_df))
                m3.metric("Highest Frequency Class", summary_df.iloc[0]['Detected Item'] if not summary_df.empty else "N/A")

                col_chart, col_table = st.columns([3, 2])
                with col_chart:
                    fig = px.bar(summary_df, x="Detected Item", y="Count", color="Detected Item", template="plotly_dark", title="Peak Concurrent Objects Per Class")
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                with col_table:
                    st.markdown("##### 📋 Peak Detection Analytics Log")
                    st.dataframe(summary_df, use_container_width=True)
                    render_csv_download_button(summary_df, filename="Video_Stream_Analytics_Report.csv")

# 8. Mode 3: Live Webcam Feed (Fixed Live Dashboard Session Rendering)
elif input_source == "📹 Live Webcam Feed":
    st.markdown("### 📹 Real-Time Live Webcam Stream")

    col_w1, col_w2, col_w3 = st.columns([2, 2, 2])
    
    with col_w1:
        if st.button("▶️ Start Camera Stream", type="primary"):
            st.session_state['webcam_active'] = True
            st.session_state['webcam_history'] = []

    with col_w2:
        if st.button("⏹️ Stop Stream & Show Summary"):
            st.session_state['webcam_active'] = False
            st.rerun()

    with col_w3:
        if st.button("🧹 Clear History"):
            st.session_state['webcam_history'] = []
            st.session_state['webcam_active'] = False
            st.rerun()

    if st.session_state['webcam_active']:
        cap = cv2.VideoCapture(0)
        alert_placeholder = st.empty()

        col_left, col_center, col_right = st.columns([1, 4, 1])
        with col_center:
            stframe = st.empty()

        frame_index = 0
        while cap.isOpened() and st.session_state['webcam_active']:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access system camera feed.")
                break

            frame_index += 1
            frame_resized = cv2.resize(frame, (640, 360))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            annotated_frame, detections = detector.process_frame(frame_rgb, confidence=confidence_threshold)
            
            for d in detections:
                d_entry = d.copy()
                d_entry['frame_id'] = f"w_frame_{frame_index}"
                st.session_state['webcam_history'].append(d_entry)

            if len(detections) > 0:
                alert_placeholder.markdown('<div class="alert-banner">🚨 LIVE HAZARD / DETECTED IN CAMERA FEED</div>', unsafe_allow_html=True)
                if enable_audio and frame_index % 15 == 0:
                    trigger_audio_alert()
            else:
                alert_placeholder.empty()

            stframe.image(annotated_frame, channels="RGB", use_container_width=True)

        cap.release()

    # Always render summary if webcam history exists and stream is stopped
    if not st.session_state['webcam_active'] and len(st.session_state['webcam_history']) > 0:
        st.markdown("---")
        st.markdown("### 📊 Live Webcam Session Summary Analytics")

        summary_df = summarize_detections(st.session_state['webcam_history'])

        m1, m2, m3 = st.columns(3)
        m1.metric("Peak Concurrent Objects Detected", summary_df['Count'].sum() if not summary_df.empty else 0)
        m2.metric("Unique Classes Encountered", len(summary_df))
        m3.metric("Highest Frequency Class", summary_df.iloc[0]['Detected Item'] if not summary_df.empty else "N/A")

        col_chart, col_table = st.columns([3, 2])
        with col_chart:
            fig = px.bar(summary_df, x="Detected Item", y="Count", color="Detected Item", template="plotly_dark", title="Peak Concurrent Objects in Frame")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col_table:
            st.markdown("##### 📋 Session Analytics Log")
            st.dataframe(summary_df, use_container_width=True)
            render_csv_download_button(summary_df, filename="Webcam_Session_Analytics_Report.csv")