import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import gc  
import time 
import os

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="SilverRoad Bozuk Yol Tespiti", layout="centered")

# --- CSS STİLLERİ (Görüntü Ortalama ve Butonlar) ---
st.markdown(
    """
    <style>
    div[data-testid="stMainBlock"] img {
        max-height: 60vh !important;
        object-fit: contain !important;
        width: auto !important;
    }
    div[data-testid="stImage"] {
        display: flex !important;
        justify-content: center !important;
        width: 100% !important;
    }
    div.stButton > button {
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- SIDEBAR ---
try:
    st.sidebar.image("silveroad.png", use_container_width=True)
except:
    st.sidebar.write("SilverRoad AI")

st.sidebar.header("Ayarlar")

# --- MODEL SEÇİMİ ---
model_secenekleri = {
    "YOLO12n (Hızlı)": "bestn.pt",
    "YOLO12s (Dengeli)": "bests.pt",
    "YOLO12m (Güçlü)": "bestm.pt"
}

secilen_model_ismi = st.sidebar.selectbox(
    "Model Seçimi",
    options=list(model_secenekleri.keys()),
    index=0 
)

model_path = model_secenekleri[secilen_model_ismi]

# --- PARAMETRELER ---
confidence = st.sidebar.slider("Güven Eşiği (Confidence)", 0.0, 1.0, 0.25)
skip_frames = st.sidebar.slider("İşleme Hızı (Skip Frame)", 1, 30, 5)

# --- BAŞLIK ---
st.title("🛣️ SilverRoad Bozuk Yol Tespiti")
st.caption(f"Aktif Model: **{secilen_model_ismi}**")

# --- MODEL YÜKLEME ---
@st.cache_resource
def load_model(path):
    try:
        model = YOLO(path)
        if not model.model.names:
             model.model.names = {0: "Catlak", 1: "Cukur", 2: "Kasis"}
        return model
    except Exception as e:
        st.error(f"Model yüklenemedi: {path} - Hata: {e}")
        return None

model = load_model(model_path)

if model is None:
    st.error(f"⚠️ HATA: **{model_path}** dosyası yüklenemedi!")
    st.stop()

# --- SESSION STATE ---
if 'is_running' not in st.session_state:
    st.session_state['is_running'] = False

# --- DOSYA YÜKLEME ---
uploaded_file = st.file_uploader("Video Yükle", type=['mp4', 'avi', 'mov'])

if uploaded_file and model:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    # Orijinal Video Bilgileri
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 24 
    
    process_width = 480 
    aspect_ratio = orig_height / orig_width
    process_height = int(process_width * aspect_ratio)

    col1, col2 = st.columns([1, 1])
    start_button = col1.button("▶️ Analizi Başlat", type="primary")
    stop_placeholder = col2.empty()
    st_frame = st.empty()

    if start_button:
        st.session_state['is_running'] = True

    if st.session_state['is_running']:
        if stop_placeholder.button("❌ Durdur / Sıfırla", type="secondary"):
            st.session_state['is_running'] = False
            cap.release()
            st.rerun()

        # Çıktı Videosu Ayarları
        output_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(output_temp.name, fourcc, fps, (process_width, process_height))
        
        frame_count = 0
        last_result = None
        
        display_every = 3

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                frame_resized = cv2.resize(frame, (process_width, process_height))

                if frame_count % skip_frames == 0 or last_result is None:
                    results = model(frame_resized, conf=confidence, verbose=False)
                    last_result = results[0]
                
                if last_result:
                    annotated_frame = last_result.plot(img=frame_resized)
                else:
                    annotated_frame = frame_resized

                out.write(annotated_frame)
                
                if frame_count % display_every == 0:
                    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    st_frame.image(frame_rgb, channels="RGB")
                    time.sleep(0.01)

                if frame_count % 100 == 0:
                    gc.collect()

        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")
        
        finally:
            cap.release()
            out.release()
            gc.collect()
        
        st.success("Analiz Tamamlandı!")
        
        if os.path.exists(output_temp.name):
            with open(output_temp.name, 'rb') as f:
                st.download_button('📥 İşlenmiş Videoyu İndir', f, file_name='SilverRoad_Output.mp4')
            
        st.session_state['is_running'] = False

