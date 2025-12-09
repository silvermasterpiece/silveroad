import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import time

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Yol Hasar Tespiti", page_icon="🛣️", layout="wide")

st.title("🛣️ Yol Hasar Tespiti - AI Analizi")
st.markdown("""
Bu uygulama, yoldaki **Çatlak, Çukur ve Kasisleri** tespit eder.
Video işleme performansını artırmak için **Akıllı Kare Atlama (Smart Frame Skipping)** teknolojisi kullanılır.
""")

# --- YAN MENÜ (AYARLAR) ---
st.sidebar.header("⚙️ Ayarlar")

# Model Yükleme (Cache kullanarak her defasında tekrar yüklenmesini engelliyoruz)
@st.cache_resource
def load_model():
    # 'bests.pt' dosyasının bu script ile aynı klasörde olduğundan emin olun
    return YOLO('bests.pt')

try:
    model = load_model()
    st.sidebar.success("Model başarıyla yüklendi!")
except Exception as e:
    st.sidebar.error(f"Model yüklenemedi! 'bests.pt' dosyasını kontrol edin. Hata: {e}")

# Kullanıcı Ayarları
skip_frames = st.sidebar.slider("Hızlandırma (Kare Atlama)", min_value=1, max_value=10, value=3, help="1=Her kareyi işle (Yavaş), 3=Her 3 karede bir işle (Hızlı)")
conf_threshold = st.sidebar.slider("Güven Eşiği (Confidence)", min_value=0.1, max_value=1.0, value=0.40)

# Sınıf ve Renk Ayarları
turkish_labels = {0: "Catlak", 1: "Cukur", 2: "Kasis"}
class_colors = {
    0: (0, 255, 255),  # Sarı (OpenCV BGR Formatı)
    1: (0, 0, 255),    # Kırmızı
    2: (255, 0, 0)     # Mavi
}

# --- VİDEO YÜKLEME ---
uploaded_file = st.file_uploader("Analiz edilecek videoyu yükleyin", type=['mp4', 'avi', 'mov'])

# --- ANALİZ FONKSİYONU ---
def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    
    # Video Bilgileri
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Video Kaydedici (MP4V codeci genelde uyumludur, ancak web'de izlemek için h264 gerekebilir. 
    # Şimdilik indirme amaçlı standart mp4v kullanıyoruz)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Görselleştirme için Streamlit yer tutucusu
    st_frame = st.empty()
    progress_bar = st.progress(0)
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    last_results = [] # Tahmin saklama hafızası

    stop_button = st.button("İşlemi Durdur")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # --- AKILLI KARE ATLAMA ---
        if frame_count % skip_frames == 0 or len(last_results) == 0:
            # conf parametresi ile güven eşiğini modele iletiyoruz
            results = model(frame, conf=conf_threshold, verbose=False)
            last_results = results
        
        # --- ÇİZİM ---
        if last_results:
            for result in last_results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0]
                    
                    color = class_colors.get(cls, (255, 255, 255))
                    label_text = turkish_labels.get(cls, "Bilinmeyen")
                    label = f'{label_text} %{int(conf * 100)}'

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    c2 = int(x1) + t_size[0], int(y1) - t_size[1] - 3
                    cv2.rectangle(frame, (int(x1), int(y1)), c2, color, -1, cv2.LINE_AA)
                    cv2.putText(frame, label, (int(x1), int(y1) - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 1. Kayıt (BGR formatında)
        out.write(frame)
        
        # 2. Ekranda Gösterme (Streamlit RGB ister, OpenCV BGR verir -> Dönüştürmeliyiz)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_frame.image(frame_rgb, channels="RGB", use_column_width=True)

        # İlerleme çubuğunu güncelle
        if total_frames > 0:
            progress_bar.progress(min(frame_count / total_frames, 1.0))

        if stop_button:
            st.warning("İşlem kullanıcı tarafından durduruldu.")
            break

    cap.release()
    out.release()

# --- ANA AKIŞ ---
if uploaded_file is not None:
    # Geçici dosya oluştur (Streamlit dosyayı RAM'de tutar, OpenCV dosya yolu ister)
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    # Çıktı için geçici dosya yolu
    output_video_path = os.path.join(tempfile.gettempdir(), "islenmis_video.mp4")

    col1, col2 = st.columns(2)
    with col1:
        st.info("Video yüklendi. İşlemeye hazırsanız butona basın.")
    
    if st.button("🚀 Analizi Başlat"):
        with st.spinner('Video işleniyor... Bu işlem video uzunluğuna göre zaman alabilir.'):
            process_video(tfile.name, output_video_path)
        
        st.success("İşlem Tamamlandı!")
        
        # İndirme Butonu
        with open(output_video_path, "rb") as file:
            btn = st.download_button(
                label="📥 İşlenmiş Videoyu İndir",
                data=file,
                file_name="tespit_sonucu.mp4",
                mime="video/mp4"
            )