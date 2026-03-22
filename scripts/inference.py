import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tensorflow as tf
import numpy as np
import pickle
import os
import time

# Đường dẫn đến thư mục models
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))

# 1. Cấu hình MediaPipe Tasks
model_task_path = os.path.join(base_dir, 'hand_landmarker.task')
if not os.path.exists(model_task_path):
    print(f"LỖI: Không tìm thấy file {model_task_path}")
    exit()

# Load MLP model và các bộ tiền xử lý
mlp_model = tf.keras.models.load_model(os.path.join(base_dir, 'asl_static_mp.h5'))
with open(os.path.join(base_dir, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)
with open(os.path.join(base_dir, 'label_encoder.pkl'), 'rb') as f:
    label_encoder = pickle.load(f)

# Cấu hình Hand Landmarker với chế độ VIDEO
base_options = python.BaseOptions(model_asset_path=model_task_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.VIDEO
)

# 3. Sử dụng context manager (with) để đảm bảo giải phóng tài nguyên khi thoát
with vision.HandLandmarker.create_from_options(options) as detector:
    cap = cv2.VideoCapture(0)
    print("Mô hình đã sẵn sàng. Đang khởi động webcam... Nhấn 'q' để thoát.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Chuyển sang RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Tạo MediaPipe Image
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        except Exception as e:
            print(f"Lỗi tạo mp.Image: {e}")
            break
        
        # Lấy timestamp tính bằng miligiây
        timestamp_ms = int(time.time() * 1000)
        
        # Nhận diện ở chế độ Video
        try:
            print("Đang detect...", end="\r")
            detection_result = detector.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            print(f"\nLỗi khi detect: {e}")
            break
        
        if detection_result.hand_landmarks:
            print(f"Tìm thấy tay!                       ", end="\r")
            for hand_landmarks in detection_result.hand_landmarks:
                # 21 landmarks * 3 (x, y, z)
                coords = []
                for lm in hand_landmarks:
                    coords.extend([lm.x, lm.y, lm.z])
                
                # Dự đoán bằng MLP
                # Cần đảm bảo coords là array 2D (1, 63)
                input_data = scaler.transform([coords])
                prediction = mlp_model.predict(input_data, verbose=0)
                
                idx = np.argmax(prediction)
                conf = np.max(prediction) * 100
                label = label_encoder.inverse_transform([idx])[0]
                
                # Hiển thị
                cv2.putText(frame, f"{label} ({conf:.1f}%)", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Vẽ xương tay cơ bản
                for lm in hand_landmarks:
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

        cv2.imshow('Nhan dien ASL (MediaPipe v2)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
