<div align="center">
  <img src="silveroad.jpg" alt="silveRoAd Logo" width="300" />

  <h1>silveRoAd</h1>

  <p>
    <strong>AI-Powered Road Defect Detection System</strong>
  </p>

  <p>
    <a href="https://www.python.org/">
      <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version">
    </a>
    <a href="https://streamlit.io/">
      <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B.svg" alt="Streamlit">
    </a>
    <a href="https://github.com/ultralytics/ultralytics">
      <img src="https://img.shields.io/badge/YOLO-v12-green.svg" alt="YOLOv12">
    </a>
    <a href="https://opensource.org/licenses/MIT">
      <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
    </a>
  </p>
</div>

---

## 📖 Overview

**silveRoAd** is an advanced computer vision project designed to automatically detect and classify road defects such as potholes, cracks, and surface damages. Utilizing the power of **YOLOv12** (You Only Look Once) and **Streamlit**, this application processes vehicle camera footage in real-time to assist municipalities and road maintenance teams in identifying infrastructure issues efficiently.

## ✨ Key Features

* **Real-time Detection:** Process live video feeds or recorded dashcam footage instantly.
* **High Accuracy:** Trained on a custom dataset using the state-of-the-art **YOLOv12** architecture.
* **Interactive Dashboard:** User-friendly web interface built with **Streamlit** for easy visualization.
* **Defect Classification:** Capable of distinguishing between various types of road damage (e.g., Pothole, Crack, Rutting).
* **Location Mapping:** (Optional) Logs the GPS coordinates of detected defects for maintenance planning.

## 🛠️ Tech Stack

* **Language:** Python
* **Core AI Model:** YOLOv12 (Ultralytics)
* **Interface:** Streamlit
* **Image Processing:** OpenCV
* **Data Handling:** Pandas, NumPy

## 📂 Project Structure

```bash
silveRoAd/
├── assets/             # Images and logo files
├── data/               # Sample videos or test images
├── models/             # Trained YOLOv12 weights (.pt files)
├── src/                # Source code for detection logic
├── app.py              # Main Streamlit application entry point
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
