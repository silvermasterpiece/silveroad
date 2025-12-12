import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import time

# --- 1. SAYFA AYARLARI ---
st.set_page_config(page_title="SilverRoad AI", page_icon="🛣️", layout="wide")

# --- 2. CSS İLE GÖRÜNTÜYÜ ORTALAMA VE STİL ---
st.markdown(
    """
    <style>
    /* Görüntü yüksekliğini sınırla ve ortala */
    div[data-testid="stMainBlock"] img {
        max-height: 600px !important;
        object-fit: contain !important;
        width: auto !important;
        border-radius: 10px;
    }
    
    /* Görüntü kapsayıcısını ortala */
    div[data-testid="stImage"] {
        display: flex !important;
        justify-content: center !important;
        width: 100% !important;
    }
    
    /* Metrik kutularını biraz süsle */
    div[data-testid="stMetricValue"] {
        font-size: 2rem !important;
        color: #FF4B4B;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- 3. SIDEBAR AYARLARI ---
# Eğer bir logon varsa burayı açabilirsin:
# st.sidebar.image("silveroad.png", use_container_width=True)

st.sidebar.header("⚙️ Model Ayarları")
confidence = st.sidebar.slider("Güven Eşiği (Confidence)", 0.0, 1.0, 0.40, help="Modelin ne kadar emin olduğunda çizim yapacağını belirler.")
skip_frames = st.sidebar.slider("Tahmin Sıklığı (Skip Frame)", 1, 30, 5, help="Her N karede bir tahmin yapar. Değer artarsa hız artar, hassasiyet düşer.")

# Model yolu (Dosyanın proje klasöründe olduğundan emin ol)
model_path = 'bests.pt'

# --- 4. BAŞLIK ---
st.title("🛣️ SilverRoad AI - Yol Kusur Tespit Sistemi")
st.markdown("Yol yüzeyindeki **Çatlak, Çukur ve Kasisleri** yapay zeka ile tespit edin.")
st.divider()

# --- 5. MODEL YÜKLEME FONKSİYONU ---
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"⚠️ Hata: '{path}' dosyası bulunamadı! Lütfen model dosyasını proje klasörüne yükleyin.")
        return None
    try:
        model = YOLO(path)
        # Sınıf isimlerini tanımla (Modelin eğitimine göre değişebilir)
        model.model.names = {0: "Catlak", 1: "Cukur", 2: "Kasis"}
        return model
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {e}")
        return None

model = load_model(model_path)

# --- 6. METRİK ALANI (ANLIK) ---
col1, col2, col3 = st.columns(3)
k1 = col1.metric("Anlık Çatlak", 0)
k2 = col2.metric("Anlık Çukur", 0)
k3 = col3.metric("Anlık Kasis", 0)

# --- 7. DOSYA YÜKLEME VE İŞLEME ---
uploaded_file = st.file_uploader("Analiz edilecek videoyu yükleyin", type=['mp4', 'avi', 'mov'])

if uploaded_file and model:
    # Geçici dosya oluştur (OpenCV'nin okuyabilmesi için)
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    # Video özelliklerini al
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Arayüz elemanları
    st_frame = st.empty() # Görüntü yer tutucusu
    progress_bar = st.progress(0) # İlerleme çubuğu
    status_text = st.empty() # Durum metni
    
    start_button = st.button("▶️ Analizi Başlat", type="primary")

    if start_button:
        # Çıktı dosyası için hazırlık
        output_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        # mp4v codec genel uyumluluk için iyidir, h264 varsa o daha iyi sonuç verir
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(output_temp.name, fourcc, fps, (width, height))
        
        frame_count = 0
        last_result = None
        
        status_text.info("Video işleniyor... Lütfen bekleyiniz.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # --- MODEL TAHMİNİ ---
            # skip_frames mantığı: Her karede model çalıştırma, öncekini kullan (Performans için)
            if frame_count % skip_frames == 0 or last_result is None:
                results = model(frame, conf=confidence, verbose=False)
                last_result = results[0]
            
            # --- ÇİZİM VE GÜNCELLEME ---
            if last_result:
                # Kutuları çiz
                annotated_frame = last_result.plot(img=frame)
                
                # Metrikleri güncelle (Sadece her 5 karede bir güncelle ki UI titremesin)
                if frame_count % 5 == 0:
                    cls_list = last_result.boxes.cls.cpu().numpy()
                    k1.metric("Anlık Çatlak", int((cls_list == 0).sum()))
                    k2.metric("Anlık Çukur", int((cls_list == 1).sum()))
                    k3.metric("Anlık Kasis", int((cls_list == 2).sum()))
            else:
                annotated_frame = frame

            # Videoya yaz
            out.write(annotated_frame)
            
            # --- UI GÜNCELLEMELERİ ---
            # Progress Bar Güncelle
            if total_frames > 0 and frame_count % 10 == 0:
                progress_bar.progress(min(frame_count / total_frames, 1.0))
            
            # Ekrana Görüntü Bas (Her 3 karede bir basarak tarayıcıyı rahatlat)
            if frame_count % 3 == 0:
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                st_frame.image(frame_rgb, channels="RGB")

        # --- KAYNAKLARI SERBEST BIRAK ---
        cap.release()
        out.release()
        progress_bar.progress(100)
        status_text.success("✅ İşlem Tamamlandı!")
        
        # --- İNDİRME BUTONU ---
        # Dosyayı okuyup indirme butonuna ver
        with open(output_temp.name, 'rb') as f:
            st.download_button(
                label="📥 İşlenmiş Videoyu İndir",
                data=f,
                file_name='SilverRoad_Output.mp4',
                mime='video/mp4'
            )
            
        # Temizlik: Geçici girdi dosyasını sil (Çıktı dosyası indirme bitene kadar silinmemeli)
        os.unlink(tfile.name)

elif not uploaded_file:
    st.info("Lütfen başlamak için bir video dosyası yükleyin.")
