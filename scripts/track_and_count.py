import cv2
import os
from ultralytics import YOLO
from collections import defaultdict

def process_and_yield_video(video_path, output_path, model_path, tracker_type='botsort.yaml', line_orientation='horizontal', line_position_ratio=0.5):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    if line_orientation == 'horizontal':
        line_pos = int(h * line_position_ratio)
        pt1, pt2 = (0, line_pos), (w, line_pos)
    else: # vertical
        line_pos = int(w * line_position_ratio)
        pt1, pt2 = (line_pos, 0), (line_pos, h)

    offset = 15
    track_history = defaultdict(list)
    count_in = 0
    count_out = 0
    counted_ids = set()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        results = model.track(frame, tracker=tracker_type, persist=True, classes=[0], verbose=False)
        cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                curr_coord = cy if line_orientation == 'horizontal' else cx
                track_history[track_id].append(curr_coord)
                
                if len(track_history[track_id]) > 10:
                    track_history[track_id].pop(0)

                if track_id not in counted_ids and len(track_history[track_id]) >= 2:
                    prev_coord = track_history[track_id][-2]
                    
                    if (line_pos - offset) < curr_coord < (line_pos + offset):
                        if prev_coord < line_pos and curr_coord >= line_pos:
                            count_in += 1
                            counted_ids.add(track_id)
                            cv2.line(frame, pt1, pt2, (0, 255, 0), 4)
                        elif prev_coord > line_pos and curr_coord <= line_pos:
                            count_out += 1
                            counted_ids.add(track_id)
                            cv2.line(frame, pt1, pt2, (0, 0, 255), 4)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

        cv2.rectangle(frame, (10, 10), (320, 90), (0, 0, 0), -1)
        cv2.putText(frame, f"Masuk: {count_in} | Keluar: {count_out}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(frame)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        yield frame_rgb, count_in, count_out

    cap.release()
    out.release()