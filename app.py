import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
from datetime import datetime
import time
import plotly.graph_objects as go
import plotly.express as px
import cv2

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SkyGuard Nexus", 
    page_icon="🛰️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CLEAN MODERN LIGHT THEME CSS ---
st.markdown("""
    <style>
    /* Global App Background */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }
    
    /* Clean Gradient Title */
    .nexus-title {
        background: linear-gradient(90deg, #1e3a8a 0%, #2563eb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 10px;
    }
    
    /* Sharp, Modern Buttons */
    .stButton>button {
        border-radius: 8px; 
        font-weight: 700; 
        background-color: #2563eb;
        color: white; 
        border: none; 
        padding: 0.6rem 2rem; 
        transition: all 0.2s ease;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2);
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        transform: translateY(-1px);
        box-shadow: 0 6px 12px rgba(37, 99, 235, 0.3);
    }

    /* Clean Alert Boxes */
    @keyframes pulse-light-red {
      0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
      70% { box-shadow: 0 0 0 15px rgba(239, 68, 68, 0); }
      100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    
    .alert-drone {
      background-color: #fef2f2; 
      border: 2px solid #ef4444; 
      padding: 20px; 
      border-radius: 12px; 
      text-align: center;
      font-size: 24px; 
      font-weight: 800; 
      color: #b91c1c;
      animation: pulse-light-red 2s infinite;
      margin-bottom: 20px;
    }
    
    .alert-bird {
      background-color: #f0fdf4; 
      border: 2px solid #10b981; 
      padding: 20px; 
      border-radius: 12px; 
      text-align: center;
      font-size: 24px; 
      font-weight: 800; 
      color: #047857;
      margin-bottom: 20px;
    }

    /* Modern White Cards with Soft Shadow */
    .modern-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    }
    
    /* Ensure tabs and text remain readable */
    .stTabs [data-baseweb="tab-list"] button {
        color: #334155;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #2563eb;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SESSION STATE ---
if 'logs' not in st.session_state:
    st.session_state['logs'] = []
if 'feedback_db' not in st.session_state:
    st.session_state['feedback_db'] = [] 
if 'last_scan' not in st.session_state:
    st.session_state['last_scan'] = None

# --- 4. MODEL INITIALIZATION ---
@st.cache_resource
def load_ai_model():
    return tf.keras.models.load_model('best_mobilenet_model.keras')

with st.spinner("Initializing neural processing unit..."):
    try:
        model = load_ai_model()
        sys_status = "ONLINE"
    except:
        st.error("SYSTEM ERROR: 'best_mobilenet_model.keras' not found in the directory.")
        sys_status = "OFFLINE"

# --- HELPER FUNCTIONS ---
def process_image(img):
    img = img.convert('RGB').resize((224, 224))
    return np.expand_dims(np.array(img) / 255.0, axis=0)

def generate_xray(img):
    """Applies Canny Edge detection to visualize AI feature extraction."""
    img_cv = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)
    edges = cv2.Canny(img_cv, 100, 200)
    return Image.fromarray(edges)

def render_gauge(confidence, target_type):
    color = "#ef4444" if target_type == "DRONE" else "#10b981"
    fig = go.Figure(go.Indicator(
        mode = "gauge+number", value = confidence,
        number = {'suffix': "%", 'font': {'color': '#0f172a', 'size': 40, 'weight': 'bold'}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#cbd5e1"},
            'bar': {'color': color}, 
            'bgcolor': "#f1f5f9",
            'borderwidth': 0,
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=250, margin=dict(l=10, r=10, t=10, b=10))
    return fig

# --- 5. SIDEBAR NAVIGATION ---
with st.sidebar:
    # Use a clean icon
    st.image("https://cdn-icons-png.flaticon.com/512/9183/9183017.png", width=60) 
    st.markdown("<h2 style='color: #0f172a; margin-top: 0;'>SkyGuard Nexus</h2>", unsafe_allow_html=True)
    
    st.write("---")
    mode = st.radio("Main Navigation", ["🎯 Precision Scanner", "📂 Batch Processing", "📈 Command Center"])
    st.write("---")
    
    # Modern card styling for sidebar info
    st.markdown(f"""
    <div style="background-color: #ffffff; border: 1px solid #e2e8f0; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
        <h4 style="margin-top:0px; color:#475569; font-size: 14px;">SYSTEM METRICS</h4>
        <b>Status:</b> <span style="color:#10b981; font-weight: bold;">{sys_status}</span><br>
        <b>Model:</b> MobileNetV2<br>
        <b>Accuracy:</b> 98.14%<br>
        <hr style="border-color: #f1f5f9; margin: 10px 0;">
        <span style="font-size: 12px; color: #64748b;">
        Developed by:<br>
        <b style="color: #0f172a; font-size: 22px;">Yashraj Pillay</b>
        </span>
    </div>
    """, unsafe_allow_html=True)

if sys_status == "OFFLINE":
    st.stop()

# --- 6. MODULE 1: PRECISION SCANNER ---
if mode == "🎯 Precision Scanner":
    st.markdown("<h1 class='nexus-title'>Precision Target Scanner</h1>", unsafe_allow_html=True)
    st.write("Upload surveillance imagery to extract metadata, view AI edge-detection, and run deep learning inference.")
    
    col1, col2 = st.columns([1.2, 1], gap="large")
    
    with col1:
        st.markdown("<div class='modern-card'>", unsafe_allow_html=True)
        upload = st.file_uploader("Upload Image File", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
        
        if upload:
            raw_img = Image.open(upload)
            
            xray_mode = st.toggle("🔍 Enable AI Vision X-Ray (Edge Detection)")
            if xray_mode:
                st.image(generate_xray(raw_img), caption="CNN Pre-Processing View (Canny Edge Filters)", use_container_width=True)
            else:
                st.image(raw_img, caption="Raw Input Feed", use_container_width=True)
            
            with st.expander("📊 View Image Metadata"):
                meta_cols = st.columns(3)
                meta_cols[0].metric("Format", raw_img.format if raw_img.format else "Unknown")
                meta_cols[1].metric("Resolution", f"{raw_img.width}x{raw_img.height}")
                meta_cols[2].metric("Color Mode", raw_img.mode)
                
        st.markdown("</div>", unsafe_allow_html=True)
            
    with col2:
        if upload:
            if st.button("Analyze Target", use_container_width=True):
                with st.spinner("Processing neural pathways..."):
                    time.sleep(0.3) 
                    processed = process_image(raw_img)
                    raw_pred = model.predict(processed)[0][0]
                    
                    if raw_pred >= 0.5:
                        target, conf = "DRONE", float(raw_pred * 100)
                    else:
                        target, conf = "BIRD", float((1 - raw_pred) * 100)
                        
                    st.session_state['last_scan'] = {
                        'filename': upload.name, 'target': target, 'conf': conf, 
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.session_state['logs'].insert(0, st.session_state['last_scan'])
        
        if st.session_state['last_scan']:
            res = st.session_state['last_scan']
            st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
            
            if res['target'] == "DRONE":
                st.markdown(f'<div class="alert-drone">🚨 THREAT ACQUIRED: DRONE</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-bird">✔️ SAFE: BIOLOGICAL ENTITY (BIRD)</div>', unsafe_allow_html=True)
                
            st.plotly_chart(render_gauge(res['conf'], res['target']), use_container_width=True)
            
            st.markdown("### Model Retraining Feedback")
            st.write("Did the AI misclassify this image? Flag it for future MLOps retraining.")
            if st.button("🚩 Flag as Incorrect Prediction"):
                st.session_state['feedback_db'].append(res)
                st.toast("Feedback logged in database. Thank you.", icon="✅")

# --- 7. MODULE 2: BATCH PROCESSING ---
elif mode == "📂 Batch Processing":
    st.markdown("<h1 class='nexus-title'>Batch Processing</h1>", unsafe_allow_html=True)
    st.write("Drag and drop multiple files to process entire folders at once.")
    
    uploads = st.file_uploader("Select multiple files", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    
    if uploads:
        if st.button(f"Initialize Batch Scan ({len(uploads)} files)", use_container_width=True):
            batch_results = []
            drones, birds = 0, 0
            
            p_bar = st.progress(0)
            
            for i, file in enumerate(uploads):
                img = Image.open(file)
                pred = model.predict(process_image(img), verbose=0)[0][0]
                
                if pred >= 0.5:
                    drones += 1
                    target, conf = "DRONE", float(pred * 100)
                else:
                    birds += 1
                    target, conf = "BIRD", float((1 - pred) * 100)
                
                log_entry = {"filename": file.name, "target": target, "conf": conf, "timestamp": datetime.now().strftime("%H:%M:%S")}
                batch_results.append(log_entry)
                st.session_state['logs'].insert(0, log_entry)
                
                p_bar.progress((i + 1) / len(uploads))
            
            st.success("Batch Operations Complete.")
            
            st.markdown("<div class='modern-card'>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Scanned", len(uploads))
            c2.metric("Drones Intercepted", drones)
            c3.metric("Birds Verified", birds)
            st.markdown("</div><br>", unsafe_allow_html=True)
            
            df_batch = pd.DataFrame(batch_results)
            df_batch.columns = ["Filename", "Detection", "Confidence (%)", "Timestamp"]
            st.dataframe(df_batch, use_container_width=True)

# --- 8. MODULE 3: COMMAND CENTER ---
elif mode == "📈 Command Center":
    st.markdown("<h1 class='nexus-title'>Command Center Dashboard</h1>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📊 Global Metrics", "🚩 MLOps Feedback Database"])
    
    with tab1:
        if st.session_state['logs']:
            df_logs = pd.DataFrame(st.session_state['logs'])
            df_logs.columns = ["File", "Classification", "Confidence", "Timestamp"]
            
            c1, c2 = st.columns([1, 1], gap="large")
            with c1:
                st.markdown("<div class='modern-card'>", unsafe_allow_html=True)
                st.write("#### Target Distribution")
                pie = px.pie(df_logs, names='Classification', color='Classification', 
                             color_discrete_map={'DRONE': '#ef4444', 'BIRD': '#10b981'}, hole=0.4)
                # Adjusted for light theme
                pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#0f172a", margin=dict(t=20, b=20, l=0, r=0))
                st.plotly_chart(pie, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with c2:
                st.markdown("<div class='modern-card'>", unsafe_allow_html=True)
                st.write("#### Master System Log")
                st.dataframe(df_logs, height=270, use_container_width=True)
                
                csv = df_logs.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Export Master CSV", data=csv, file_name="SkyGuard_Logs.csv", mime="text/csv", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("System logs are currently empty. Run single or batch scans to generate analytics.")
            
    with tab2:
        st.markdown("<div class='modern-card'>", unsafe_allow_html=True)
        st.write("### Human-in-the-Loop Corrections")
        st.write("This database stores images flagged by operators as incorrect. Export this data to refine and retrain the deep learning model.")
        if st.session_state['feedback_db']:
            df_flags = pd.DataFrame(st.session_state['feedback_db'])
            st.dataframe(df_flags, use_container_width=True)
        else:
            st.success("No incorrect predictions have been flagged by operators.")
        st.markdown("</div>", unsafe_allow_html=True)