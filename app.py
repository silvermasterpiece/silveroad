import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="SilverRoad Bozuk Yol Tespiti", layout="centered")

# --- CSS İLE GÖRÜNTÜ VE BUTON DÜZENLEMELERİ ---
st.markdown(
    """
    <style>
    /* Görüntü ayarları */
    div[data-testid="stMainBlock"] img {
        max-height: 70vh !important;
        object-fit: contain !important;
        width: auto !important;
    }
    div[data-testid="stImage"] {
        display: flex !important;
        justify-content: center !important;
        width: 100% !important;
    }
    /* Butonları biraz daha belirgin yapalım */
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
confidence = st.sidebar.slider("Güven Eşiği (Confidence) ", 0.0, 1.0, 0.25)
skip_frames = st.sidebar.slider("Hız (Skip Frame)", 1, 30, 5)

# --- BAŞLIK ---
st.title("🛣️ SilverRoad Bozuk Yol Tespiti")
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

# --- SESSION STATE (DURUM KONTROLÜ) ---
if 'is_running' not in st.session_state:
    st.session_state['is_running'] = False

# --- DOSYA YÜKLEME ---
uploaded_file = st.file_uploader("Video Yükle", type=['mp4', 'avi', 'mov'])

if uploaded_file and model:
    # Geçici dosya oluşturma
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Butonlar için kolonlar
    col1, col2 = st.columns([1, 1])
    
    # BAŞLAT BUTONU
    start_button = col1.button("▶️ Analizi Başlat", type="primary")
    
    # ÇIKIŞ BUTONU (Placeholder)
    stop_placeholder = col2.empty()

    # Görüntü Alanı
    st_frame = st.empty()

    # Başlat'a basıldıysa durumu güncelle
    if start_button:
        st.session_state['is_running'] = True

    # Eğer analiz çalışıyorsa
    if st.session_state['is_running']:
        # Çıkış butonunu aktif et
        if stop_placeholder.button("❌ Videoyu Kapat / Sıfırla", type="secondary"):
            st.session_state['is_running'] = False
            cap.release()
            st.rerun()  # Sayfayı yenileyerek başa döner

        # Video Kaydı için hazırlık
        output_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(output_temp.name, fourcc, fps, (width, height))
        
        frame_count = 0
        last_result = None

        while cap.isOpened():
            # Kullanıcı "Videoyu Kapat" derse döngüyü kırmak için kontrol gerekebilir
            # Ancak Streamlit yapısında yukarıdaki buton kontrolü döngüden hemen önce olduğu için
            # döngü içindeyken butona basıldığında script baştan çalışır ve is_running False olur.
            
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip Frame ve Tahmin
            if frame_count % skip_frames == 0 or last_result is None:
                results = model(frame, conf=confidence, verbose=False)
                last_result = results[0]
            
            # Çizim
            if last_result:
                annotated_frame = last_result.plot(img=frame)
            else:
                annotated_frame = frame

            out.write(annotated_frame)
            
            # Görüntüyü ekrana bas
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st_frame.image(frame_rgb, channels="RGB") 
        
        # Döngü bittiğinde (Video sonu)
        cap.release()
        out.release()
        
        st.success("Analiz Tamamlandı!")
        
        # İndirme Butonu
        with open(output_temp.name, 'rb') as f:
            st.download_button('📥 İşlenmiş Videoyu İndir', f, file_name='SilverRoad_Output.mp4')
            
        # İşlem bitince is_running'i kapatabiliriz ki tekrar başlamasın
        st.session_state['is_running'] = False
