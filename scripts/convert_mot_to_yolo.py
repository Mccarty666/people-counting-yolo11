import os
import pandas as pd

# Definisikan path ke dataset mentah (sesuaikan dengan direktori ekstraksi Anda)
RAW_TRAIN_DIR = 'data/raw/MOT17/train'
OUTPUT_YOLO_DIR = 'data/processed/yolo_format'

# Dalam MOT17, class_id pedestrian = 1. Di YOLO, kita set class index ke 0 (person).
MOT17_PEDESTRIAN_CLASS = 1 
YOLO_CLASS_INDEX = 0

def convert_mot17_to_yolo():
    # Pastikan folder output untuk YOLO sudah ada
    os.makedirs(os.path.join(OUTPUT_YOLO_DIR, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_YOLO_DIR, 'train', 'labels'), exist_ok=True)

    # Loop semua sekuens di folder train
    for seq_name in os.listdir(RAW_TRAIN_DIR):
        # Gunakan sekuens FRCNN saja untuk menghindari redundansi data/video yang sama
        if not seq_name.endswith('-FRCNN'):
            continue
        
        print(f"Memproses sekuens: {seq_name}")
        seq_dir = os.path.join(RAW_TRAIN_DIR, seq_name)
        gt_path = os.path.join(seq_dir, 'gt', 'gt.txt')
        
        # Ambil resolusi gambar dari file seqinfo.ini untuk normalisasi bounding box YOLO
        seqinfo_path = os.path.join(seq_dir, 'seqinfo.ini')
        img_width, img_height = 1920, 1080 # Nilai default
        if os.path.exists(seqinfo_path):
            with open(seqinfo_path, 'r') as f:
                for line in f:
                    if line.startswith('imWidth'):
                        img_width = float(line.split('=')[1].strip())
                    elif line.startswith('imHeight'):
                        img_height = float(line.split('=')[1].strip())
                        break

        # Baca file gt.txt (ground truth) bawaan MOT17
        columns = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'class', 'vis']
        df = pd.read_csv(gt_path, names=columns)
        
        # Filter hanya pedestrian yang aktif/valid
        df = df[(df['class'] == MOT17_PEDESTRIAN_CLASS) & (df['conf'] == 1)]
        
        # Kelompokkan data per frame gambar
        grouped = df.groupby('frame')
        
        for frame_id, group in grouped:
            # Buat nama unik untuk output label YOLO
            new_label_name = f"{seq_name}_{frame_id:06d}.txt"
            label_path = os.path.join(OUTPUT_YOLO_DIR, 'train', 'labels', new_label_name)
            
            with open(label_path, 'w') as f_out:
                for _, row in group.iterrows():
                    # Hitung koordinat format YOLO (titik tengah X, titik tengah Y, lebar, tinggi) dinormalisasi 0-1
                    x_center = (row['bb_left'] + (row['bb_width'] / 2)) / img_width
                    y_center = (row['bb_top'] + (row['bb_height'] / 2)) / img_height
                    w = row['bb_width'] / img_width
                    h = row['bb_height'] / img_height
                    
                    # Pastikan nilai tidak keluar dari batas 0-1
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    w = max(0, min(1, w))
                    h = max(0, min(1, h))
                    
                    # Tulis ke dalam file .txt
                    f_out.write(f"{YOLO_CLASS_INDEX} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
                    
if __name__ == "__main__":
    convert_mot17_to_yolo()
    print("Konversi label MOT17 ke format YOLO selesai.")