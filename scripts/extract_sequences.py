import os
import json
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Cấu hình đường dẫn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(BASE_DIR, 'dataset', 'kyhieudong', 'WLASL', 'WLASL_v0.3.json')
VIDEOS_DIR = os.path.join(BASE_DIR, 'dataset', 'kyhieudong', 'WLASL', 'videos')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'hand_landmarker.task')
OUTPUT_DIR = os.path.join(BASE_DIR, 'models')

# Tham số xử lý
NUM_CLASSES = 100  # WLASL-100
SEQUENCE_LENGTH = 30  # Số frame mỗi chuỗi
NUM_LANDMARKS = 21
NUM_FEATURES = NUM_LANDMARKS * 3

def process_one_video(video_id, model_path):
    """Xử lý 1 video duy nhất và trả về landmarks"""
    video_path = os.path.join(VIDEOS_DIR, f"{video_id}.mp4")
    if not os.path.exists(video_path):
        return None

    # Khởi tạo landmarker bên trong mỗi process
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1
    )
    
    video_landmarks = []
    
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            detection_result = landmarker.detect(mp_image)
            
            if detection_result.hand_landmarks:
                hand = detection_result.hand_landmarks[0]
                landmarks = []
                wrist = hand[0]
                mcp_middle = hand[9] # Gốc ngón giữa
                
                # Tính hand_size để chuẩn hóa tỉ lệ (Scale Invariance)
                hand_size = np.sqrt((wrist.x - mcp_middle.x)**2 + 
                                    (wrist.y - mcp_middle.y)**2 + 
                                    (wrist.z - mcp_middle.z)**2) + 1e-6
                
                for lm in hand:
                    # Chuẩn hóa theo cổ tay VÀ kích thước bàn tay
                    landmarks.extend([(lm.x - wrist.x)/hand_size, 
                                      (lm.y - wrist.y)/hand_size, 
                                      (lm.z - wrist.z)/hand_size])
                video_landmarks.append(landmarks)
            else:
                video_landmarks.append([0.0] * NUM_FEATURES)
                
        cap.release()
    
    if not video_landmarks:
        return None
        
    video_landmarks = np.array(video_landmarks)
    num_frames = len(video_landmarks)
    
    # Lấy mẫu/Padding
    if num_frames >= SEQUENCE_LENGTH:
        indices = np.linspace(0, num_frames - 1, SEQUENCE_LENGTH, dtype=int)
        video_landmarks = video_landmarks[indices]
    else:
        padding = np.zeros((SEQUENCE_LENGTH - num_frames, NUM_FEATURES))
        if len(video_landmarks) > 0:
            video_landmarks = np.vstack([video_landmarks, padding])
        else:
            video_landmarks = padding
            
    return video_landmarks

def process_class(args):
    """Hàm xử lý cho một lớp (word)"""
    class_idx, gloss, instances, model_path = args
    results = []
    for inst in instances:
        video_id = inst['video_id']
        landmarks = process_one_video(video_id, model_path)
        if landmarks is not None:
            results.append((landmarks, class_idx))
    return gloss, results

def main():
    print(f"Bắt đầu trích xuất đa luồng (multiprocessing) cho {NUM_CLASSES} lớp...", flush=True)
    
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    
    # Chuẩn bị danh sách tham số cho các tiến trình
    tasks = []
    labels_map = {}
    for i in range(min(NUM_CLASSES, len(data))):
        gloss = data[i]['gloss']
        labels_map[i] = gloss
        tasks.append((i, gloss, data[i]['instances'], MODEL_PATH))
    
    x_data = []
    y_data = []
    
    # Sử dụng ProcessPoolExecutor để chạy song song
    # Tự động chọn số worker phù hợp với số core (12)
    start_time = time.time()
    with ProcessPoolExecutor() as executor:
        future_to_gloss = {executor.submit(process_class, task): task[1] for task in tasks}
        
        for i, future in enumerate(as_completed(future_to_gloss)):
            gloss, results = future.result()
            for landmarks, class_idx in results:
                x_data.append(landmarks)
                y_data.append(class_idx)
            
            if (i + 1) % 5 == 0 or (i + 1) == NUM_CLASSES:
                print(f"Đã xử lý xong {i+1}/{NUM_CLASSES} lớp. Tên lớp vừa xong: {gloss}", flush=True)
    
    if not x_data:
        print("Không tìm thấy dữ liệu video phù hợp!", flush=True)
        return
        
    # Lưu kết quả
    print("Đang lưu dữ liệu...", flush=True)
    X = np.array(x_data, dtype=np.float32)
    y = np.array(y_data, dtype=np.int32)
    
    np.save(os.path.join(OUTPUT_DIR, 'dynamic_X.npy'), X)
    np.save(os.path.join(OUTPUT_DIR, 'dynamic_y.npy'), y)
    
    with open(os.path.join(OUTPUT_DIR, 'dynamic_labels.json'), 'w') as f:
        json.dump(labels_map, f)
        
    end_time = time.time()
    print(f"Hoàn thành! Đã lưu {len(X)} mẫu vào thư mục models/", flush=True)
    print(f"Thời gian thực hiện: {end_time - start_time:.2f} giây", flush=True)
    print(f"Shape: X={X.shape}, y={y.shape}", flush=True)

if __name__ == "__main__":
    main()
