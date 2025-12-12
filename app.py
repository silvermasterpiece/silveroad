import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="SilverRoad AI", layout="wide")

# --- CSS İLE GÖRÜNTÜYÜ ORTALAMA VE SIĞDIRMA ---
st.markdown(
    """
    <style>
    /* 1. Görüntünün kendisi için kurallar */
    div[data-testid="stMainBlock"] img {
        max-height: 70vh !important;  /* Yükseklik sınırı */
        object-fit: contain !important; /* Görüntüyü bozma */
        width: auto !important; /* Genişlik serbest */
    }

    /* 2. Görüntüyü tutan kapsayıcıyı (container) ortala */
    div[data-testid="stImage"] {
        display: flex !important;
        justify-content: center !important; /* Yatayda ortala */
        width: 100% !important; /* Kapsayıcı tam genişlikte olsun */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- SIDEBAR ---
try:
    st.sidebar.image("silveroad.png", use_container_width=True)
except:
    st.sidebar.write("SilverRoad AI Logo")

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
    index=1 
)

model_path = model_secenekleri[secilen_model_ismi]

# --- DİĞER AYARLAR ---
confidence = st.sidebar.slider("Güven Eşiği", 0.0, 1.0, 0.40)
skip_frames = st.sidebar.slider("Hız (Skip Frame)", 1, 30, 10)

# --- BAŞLIK ---
st.title("🛣️ SilverRoad AI")
st.caption(f"Aktif Model: **{secilen_model_ismi}**")

# --- MODEL YÜKLEME ---
@st.cache_resource
def load_model(path):
    try:
        model = YOLO(path)
        model.model.names = {0: "Catlak", 1: "Cukur", 2: "Kasis"}
        return model
    except Exception as e:
        st.error(f"Model yüklenemedi! '{path}' dosyası klasörde bulunamadı.")
        return None

model = load_model(model_path)

# --- DOSYA YÜKLEME ---
uploaded_file = st.file_uploader("Video Yükle", type=['mp4', 'avi', 'mov'])

if uploaded_file and model:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Görüntü Alanı
    st_frame = st.empty()
    
    # Kontrol Butonları
    btn_col1, btn_col2 = st.columns([1, 10])
    start_button = btn_col1.button("Başlat")
    stop_placeholder = btn_col2.empty()

    if start_button:
        st.session_state['stop'] = False
        stop_button = stop_placeholder.button("Durdur")
        
        output_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(output_temp.name, fourcc, fps, (width, height))
        
        frame_count = 0
        last_result = None

        while cap.isOpened():
            if st.session_state.get('stop'):
                break

            if stop_placeholder.button("Durdur", key=f"stop_{frame_count}"):
                 st.session_state['stop'] = True
                 break

            ret, frame = cap.read()
            if not ret: break
            
            frame_count += 1
            
            # Skip Frame ve Tahmin
            if frame_count % skip_frames == 0 or last_result is None:
                results = model(frame, conf=confidence, verbose=False)
                last_result = results[0]
            
            # Çizim (Sadece kutuları çiziyoruz, sayaç hesaplamıyoruz)
            if last_result:
                annotated_frame = last_result.plot(img=frame)
            else:
                annotated_frame = frame

            out.write(annotated_frame)
            
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st_frame.image(frame_rgb, channels="RGB") 

        cap.release()
        out.release()
        stop_placeholder.empty()

        if st.session_state.get('stop'):
            st.warning("Durduruldu.")
        else:
            st.success("İşlem Bitti!")
        
        with open(output_temp.name, 'rb') as f:
            st.download_button('İndir', f, file_name='SilverRoad_Output.mp4')
