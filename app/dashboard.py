import streamlit as st
import tempfile
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.track_and_count import process_and_yield_video

st.set_page_config(page_title="People Counting Analytics", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 600; color: #1E3A8A; margin-bottom: 5px;}
    .sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 25px;}
    </style>
    <div class="main-header">Sistem Analitik People Counting</div>
    <div class="sub-header">Evaluasi Kepadatan dan Pelacakan Multi-Objek secara Real-Time</div>
    <hr>
""", unsafe_allow_html=True)

col_config, col_display = st.columns([1, 2.5])

with col_config:
    st.subheader("Konfigurasi Sistem")
    model_path = st.text_input("Direktori Bobot Model:", value="models/best.pt")
    tracker_choice = st.selectbox("Algoritma Asosiasi (Tracker):", ["BoT-SORT", "ByteTrack"])
    tracker_file = "botsort.yaml" if tracker_choice == "BoT-SORT" else "bytetrack.yaml"
    
    st.markdown("<br>**Konfigurasi Area Deteksi**", unsafe_allow_html=True)
    line_orientation = st.radio("Orientasi Garis Pengawasan:", ["Horizontal", "Vertikal"])
    line_orientation_val = "horizontal" if line_orientation == "Horizontal" else "vertical"
    line_position = st.slider("Posisi Garis (%)", min_value=10, max_value=90, value=50, step=5)
    line_position_ratio = line_position / 100.0

    st.markdown("<br>**Input Data Uji**", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Pilih Berkas Video (.mp4, .avi)", type=['mp4', 'avi'])

with col_display:
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        
        if st.button("Jalankan Pemrosesan Analitik", type="primary", use_container_width=True):
            output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            
            st.markdown("### Live Analysis Preview")
            
            m1, m2, m3 = st.columns(3)
            metric_in = m1.empty()
            metric_out = m2.empty()
            metric_total = m3.empty()
            
            video_placeholder = st.empty()

            for frame_rgb, count_in, count_out in process_and_yield_video(
                video_path=tfile.name, 
                output_path=output_video_path, 
                model_path=model_path, 
                tracker_type=tracker_file,
                line_orientation=line_orientation_val,
                line_position_ratio=line_position_ratio
            ):
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                metric_in.metric(label="Total Masuk (Arah 1)", value=count_in)
                metric_out.metric(label="Total Keluar (Arah 2)", value=count_out)
                metric_total.metric(label="Total Volume Kepadatan", value=count_in + count_out)
            
            st.success("Analisis video selesai!")

            with open(output_video_path, 'rb') as v_file:
                st.download_button(
                    label="Unduh Rekaman Hasil Evaluasi",
                    data=v_file,
                    file_name=f"hasil_evaluasi_{tracker_choice}.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
    else:
        st.info("Silakan tentukan konfigurasi dan unggah video pada panel sebelah kiri untuk memulai pengujian.")