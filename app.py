import streamlit as st
import cv2
import tempfile
import gc
import time # ZAMANLAYICI EKLENDİ
from ultralytics import YOLO
import os

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="SilverRoad AI", layout="centered")

# --- CSS (Görüntü Ortalama) ---
st.markdown(
    """
    <style>
    div[data-testid="stImage"] {
        display: flex;
        justify-content: center;
    }
    div[data-testid="stImage"] img {
        max-height: 400px; /* Yükseklik sınırı koyduk ki tarayıcı yorulmasın */
        width: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🛣️ SilverRoad Bozuk Yol Tespiti")

# --- MODEL YÜKLEME ---
@st.cache_resource
def load_model():
    # Hata almamak için manuel path yerine doğrudan dosya adını dene
    # Eğer github'da dosya 'bestn.pt' ise:
    try:
        model = YOLO("bestn.pt") 
        return model
    except Exception as e:
        return None

model = load_model()

if not model:
    st.warning("⚠️ Model dosyası (bestn.pt) yüklenemedi. Lütfen dosyanın GitHub'da olduğundan emin olun.")
    st.stop()

# --- VİDEO YÜKLEME ---
uploaded_file = st.file_uploader("Video Yükle", type=['mp4', 'avi', 'mov'])

if uploaded_file:
    # Geçici dosya işlemleri
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    # Video bilgileri
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30 # Hata önleyici
    
    # Butonlar
    col1, col2 = st.columns(2)
    start_btn = col1.button("▶️ Başlat", type="primary")
    stop_btn = col2.button("⏹️ Durdur")
    
    st_frame = st.empty()
    
    if start_btn:
        # --- OPTİMİZASYON AYARLARI ---
        process_width = 480  # Çözünürlüğü düşük tutuyoruz (Hız için)
        skip_frames = 5      # Tahmin atlama (Model her karede çalışmasın)
        display_every = 3    # Ekran yenileme (Tarayıcı her karede yenilenmesin - KRİTİK AYAR)
        
        frame_count = 0
        last_result = None
        
        while cap.isOpened():
            # Stop butonuna basılırsa döngüyü kır
            # Not: Streamlit'te döngü içindeyken butonu algılamak zordur, 
            # ancak tarayıcı yenilenirse durur.
            
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # 1. Boyutlandırma (RAM Tasarrufu)
            h, w = frame.shape[:2]
            aspect = h / w
            new_h = int(process_width * aspect)
            frame_resized = cv2.resize(frame, (process_width, new_h))
            
            # 2. Model Tahmini (Her 5 karede bir)
            if frame_count % skip_frames == 0 or last_result is None:
                results = model(frame_resized, verbose=False, conf=0.25)
                last_result = results[0]
            
            # 3. Çizim
            if last_result:
                annotated_frame = last_result.plot()
            else:
                annotated_frame = frame_resized
                
            # 4. Ekrana Basma (Log Hatasını Önleyen Kısım)
            # Her kareyi değil, sadece her 3. kareyi ekrana basıyoruz.
            if frame_count % display_every == 0:
                # BGR -> RGB Dönüşümü
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                st_frame.image(frame_rgb, use_container_width=True) # use_column_width yerine bu yeni komut
                
                # MİNİK BEKLEME: Tarayıcının resmi indirmesine fırsat ver
                time.sleep(0.01) 
            
            # 5. RAM Temizliği (Nadir yap)
            if frame_count % 100 == 0:
                gc.collect()
        
        cap.release()
        st.success("Video tamamlandı.")
