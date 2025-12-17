graph TD
    %% Genel Stil Tanımları
    classDef user fill:#f9f,stroke:#333,stroke-width:2px,color:black;
    classDef system fill:#e1f5fe,stroke:#0277bd,stroke-width:2px,color:black;
    classDef process fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,color:black;
    classDef storage fill:#e0e0e0,stroke:#616161,stroke-width:2px,stroke-dasharray: 5 5,color:black;

    %% Düğümler (Nodes)
    User([👤 Kullanıcı]) -->|1. Video Yükle| UI[🖥️ Streamlit Arayüzü]
    
    subgraph Streamlit_Cloud_Sunucusu [☁️ Streamlit Cloud Sunucusu]
        direction TB
        UI -->|2. Veri Transferi| TempIn[(📂 TempFile Yazma\nInput.mp4)]
        TempIn -->|3. Okuma| CV[⚙️ OpenCV VideoCapture]
        
        subgraph Video_Isleme_Dongusu [🔄 Video İşleme Döngüsü]
            direction TB
            CV -->|4. Kare Oku & Resize| Pre{Kare Atlama?}
            Pre -- Hayır (İşle) --> YOLO[🧠 YOLOv12 Tahmini\n(Inference)]
            Pre -- Evet (Atla) --> Cache[⚡ Önceki Sonuç]
            YOLO -->|Sonuç| Draw[🖌️ Çizim & Annotasyon]
            Cache -->|Sonuç| Draw
        end
        
        Draw -->|5. Kare Gönder| Display[📺 Streamlit Görüntüleme]
        Draw -->|6. Yazma| TempOut[(📂 TempFile Yazma\nOutput.mp4)]
    end

    Display -->|7. Canlı İzleme| User
    TempOut -->|8. İndirilebilir Dosya| UI
    UI -->|9. Videoyu İndir| User

    %% Sınıf Atamaları
    class User user;
    class UI,Display system;
    class CV,YOLO,Draw,Pre,Cache process;
    class TempIn,TempOut storage;
