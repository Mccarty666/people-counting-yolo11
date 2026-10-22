# 🚶‍♂️ Real-Time People Counting & Tracking (Deep Learning)

Proyek ini adalah implementasi sistem *People Counting* menggunakan arsitektur **YOLO11-Small** dan algoritma Multi-Object Tracking (**ByteTrack** & **BoT-SORT**) untuk mendeteksi dan menghitung pejalan kaki yang melintasi area tertentu secara *real-time*.

Dibuat sebagai Tugas Proyek Akhir Mata Kuliah Deep Learning.

## 👤 Author
* **Nama:** Riandra Fajar Putra Pratama
* **NIM:** 2024071007
* **Institusi:** Universitas Pembangunan Jaya (UPJ)

## 🛠️ Tech Stack & Spesifikasi
* **Detektor:** YOLO11-Small (`best.pt` - Fine-tuned)
* **Tracker:** ByteTrack dan BoT-SORT
* **Metrik Evaluasi (Colab):** mAP50: 75.9% | Precision: 88.7%
* **Dataset:** MOT17 (Temporal Splitting ke 21 video klip)
* **Antarmuka:** Streamlit Dashboard

## 🚀 Cara Menjalankan Aplikasi Lokal
1. Clone repositori ini:
   ```bash
   git clone [https://github.com/Mccarty666/people-counting-yolo11.git](https://github.com/Mccarty666/people-counting-yolo11.git)
