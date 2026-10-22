import os
import shutil
import pandas as pd
import random

RAW_TRAIN_DIR = 'data/raw/MOT17/train'
OUTPUT_YOLO_DIR = 'data/processed/yolo_format'
MOT17_PEDESTRIAN_CLASS = 1 
YOLO_CLASS_INDEX = 0

def create_yolo_dirs():
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_YOLO_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_YOLO_DIR, split, 'labels'), exist_ok=True)

def process_and_split_dataset():
    create_yolo_dirs()

    sequences = [d for d in os.listdir(RAW_TRAIN_DIR) if d.endswith('-FRCNN')]
    
    sub_sequences = []

    for seq_name in sequences:
        seq_dir = os.path.join(RAW_TRAIN_DIR, seq_name)
        seqinfo_path = os.path.join(seq_dir, 'seqinfo.ini')
        
        seq_length, img_width, img_height = 0, 1920, 1080
        with open(seqinfo_path, 'r') as f:
            for line in f:
                if line.startswith('seqLength'):
                    seq_length = int(line.split('=')[1].strip())
                elif line.startswith('imWidth'):
                    img_width = float(line.split('=')[1].strip())
                elif line.startswith('imHeight'):
                    img_height = float(line.split('=')[1].strip())

        chunk_size = seq_length // 3
        sub_sequences.append({'name': f"{seq_name}_part1", 'orig': seq_name, 'start': 1, 'end': chunk_size, 'w': img_width, 'h': img_height})
        sub_sequences.append({'name': f"{seq_name}_part2", 'orig': seq_name, 'start': chunk_size + 1, 'end': chunk_size * 2, 'w': img_width, 'h': img_height})
        sub_sequences.append({'name': f"{seq_name}_part3", 'orig': seq_name, 'start': (chunk_size * 2) + 1, 'end': seq_length, 'w': img_width, 'h': img_height})

    random.seed(42)
    random.shuffle(sub_sequences)

    train_clips = sub_sequences[:15]
    val_clips = sub_sequences[15:19]
    test_clips = sub_sequences[19:]
    
    allocations = {'train': train_clips, 'val': val_clips, 'test': test_clips}

    for split_name, clips in allocations.items():
        print(f"\nMemproses {len(clips)} klip untuk {split_name.upper()} set...")
        for clip in clips:
            print(f" -> Memproses klip: {clip['name']} (Frame {clip['start']} s/d {clip['end']})")
            
            orig_seq_dir = os.path.join(RAW_TRAIN_DIR, clip['orig'])
            gt_path = os.path.join(orig_seq_dir, 'gt', 'gt.txt')
            img_dir = os.path.join(orig_seq_dir, 'img1')

            columns = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'class', 'vis']
            df = pd.read_csv(gt_path, names=columns)
            df = df[(df['class'] == MOT17_PEDESTRIAN_CLASS) & (df['conf'] == 1)]

            df_clip = df[(df['frame'] >= clip['start']) & (df['frame'] <= clip['end'])]
            grouped = df_clip.groupby('frame')

            for frame_id in range(clip['start'], clip['end'] + 1):
                img_name = f"{frame_id:06d}.jpg"
                new_base_name = f"{clip['name']}_{frame_id:06d}"

                src_img = os.path.join(img_dir, img_name)
                dst_img = os.path.join(OUTPUT_YOLO_DIR, split_name, 'images', f"{new_base_name}.jpg")
                if os.path.exists(src_img):
                    shutil.copy(src_img, dst_img)

                label_path = os.path.join(OUTPUT_YOLO_DIR, split_name, 'labels', f"{new_base_name}.txt")
                with open(label_path, 'w') as f_out:
                    if frame_id in grouped.groups:
                        group = grouped.get_group(frame_id)
                        for _, row in group.iterrows():
                            x_center = (row['bb_left'] + (row['bb_width'] / 2)) / clip['w']
                            y_center = (row['bb_top'] + (row['bb_height'] / 2)) / clip['h']
                            w = row['bb_width'] / clip['w']
                            h = row['bb_height'] / clip['h']
                            
                            x_center = max(0, min(1, x_center))
                            y_center = max(0, min(1, y_center))
                            w = max(0, min(1, w))
                            h = max(0, min(1, h))
                            
                            f_out.write(f"{YOLO_CLASS_INDEX} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

if __name__ == "__main__":
    print("Memulai proses Temporal Splitting & YOLO Conversion...")
    process_and_split_dataset()
    print("\nSUKSES! Dataset siap digunakan untuk training YOLO.")