import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tensorflow as tf
import numpy as np
import pickle
import os
import json
import time

# Cấu hình đường dẫn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# 1. Load Model Tĩnh (MLP)
STATIC_MODEL_PATH = os.path.join(MODELS_DIR, 'asl_static_mp.h5')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, 'label_encoder.pkl')

mlp_model = tf.keras.models.load_model(STATIC_MODEL_PATH)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)
with open(LABEL_ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

# 2. Load Model Động (LSTM)
DYNAMIC_MODEL_PATH = os.path.join(MODELS_DIR, 'asl_dynamic_lstm.h5')
DYNAMIC_LABELS_PATH = os.path.join(MODELS_DIR, 'dynamic_labels.json')
DYNAMIC_SCALER_PATH = os.path.join(MODELS_DIR, 'dynamic_scaler.pkl')

lstm_model = tf.keras.models.load_model(DYNAMIC_MODEL_PATH)
with open(DYNAMIC_LABELS_PATH, 'r') as f:
    dynamic_labels = json.load(f)
with open(DYNAMIC_SCALER_PATH, 'rb') as f:
    dynamic_scaler = pickle.load(f)

# 3. Cấu hình MediaPipe Task
HAND_LANDMARKER_TASK = os.path.join(MODELS_DIR, 'hand_landmarker.task')

# State variables
sequence_buffer = []
prediction_history = []  # Lưu lịch sử dự đoán để làm mượt
HISTORY_LENGTH = 8      # Số lượng dự đoán để lấy trung bình
SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 50.0 # Ngưỡng tự tin (%) - Tăng lại một chút
current_mode = "STATIC" # STATIC hoặc DYNAMIC
last_prediction = "None"
confidence = 0

def draw_landmarks(frame, hand_landmarks):
    h, w, _ = frame.shape
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

def main():
    global current_mode, sequence_buffer, last_prediction, confidence, prediction_history
    
    base_options = python.BaseOptions(model_asset_path=HAND_LANDMARKER_TASK)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=vision.RunningMode.VIDEO
    )
    
    cap = cv2.VideoCapture(0)
    
    with vision.HandLandmarker.create_from_options(options) as detector:
        print(f"Hệ thống sẵn sàng! Chế độ hiện tại: {current_mode}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(time.time() * 1000)
            
            detection_result = detector.detect_for_video(mp_image, timestamp_ms)
            
            if detection_result.hand_landmarks:
                hand = detection_result.hand_landmarks[0]
                draw_landmarks(frame, hand)
                
                if current_mode == "STATIC":
                    # STATIC: Dùng tọa độ tuyệt đối (giống lúc train trong collect_data.py)
                    coords = []
                    for lm in hand:
                        coords.extend([lm.x, lm.y, lm.z])
                    
                    input_data = scaler.transform([coords])
                    pred = mlp_model.predict(input_data, verbose=0)
                    idx = np.argmax(pred)
                    conf = np.max(pred) * 100
                    
                    if conf > 75: 
                        last_prediction = label_encoder.inverse_transform([idx])[0]
                        confidence = conf
                else:
                    # DYNAMIC: Dùng tọa độ tương đối + Hand-size scaling (Cách 3)
                    wrist = hand[0]
                    mcp_middle = hand[9]
                    hand_size = np.sqrt((wrist.x - mcp_middle.x)**2 + 
                                        (wrist.y - mcp_middle.y)**2 + 
                                        (wrist.z - mcp_middle.z)**2) + 1e-6
                    
                    coords = []
                    for lm in hand:
                        coords.extend([(lm.x - wrist.x)/hand_size, 
                                      (lm.y - wrist.y)/hand_size, 
                                      (lm.z - wrist.z)/hand_size])
                    
                    sequence_buffer.append(coords)
                    if len(sequence_buffer) > SEQUENCE_LENGTH:
                        sequence_buffer.pop(0)
                        
                    if len(sequence_buffer) == SEQUENCE_LENGTH:
                        seq_arr = np.array(sequence_buffer) 
                        seq_scaled = dynamic_scaler.transform(seq_arr)
                        input_seq = np.expand_dims(seq_scaled, axis=0) 
                        
                        pred = lstm_model.predict(input_seq, verbose=0)
                        idx = np.argmax(pred)
                        conf = np.max(pred) * 100
                        
                        if conf > 20: 
                            raw_label = dynamic_labels[str(idx)]
                            print(f"DEBUG: {raw_label} ({conf:.1f}%)", end='\r')

                        if conf > CONFIDENCE_THRESHOLD:
                            prediction_history.append(idx)
                            if len(prediction_history) > HISTORY_LENGTH:
                                prediction_history.pop(0)
                            
                            from collections import Counter
                            most_common_idx = Counter(prediction_history).most_common(1)[0][0]
                            last_prediction = dynamic_labels[str(most_common_idx)]
                            confidence = conf
                        else:
                            confidence *= 0.95
            else:
                if current_mode == "DYNAMIC":
                    confidence *= 0.8
                    if confidence < 5: 
                        last_prediction = "None"
                        sequence_buffer = []

            # UI
            color = (0, 255, 0) if current_mode == "STATIC" else (255, 255, 0)
            cv2.putText(frame, f"MODE: {current_mode} (m:toggle|q:quit)", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if last_prediction != "None":
                cv2.putText(frame, f"PRED: {last_prediction}", (10, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
                cv2.putText(frame, f"CONF: {confidence:.1f}%", (10, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

            cv2.imshow('ASL Unified Recognition', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('m'):
                current_mode = "DYNAMIC" if current_mode == "STATIC" else "STATIC"
                sequence_buffer = []
                prediction_history = []
                last_prediction = "None"
                confidence = 0
                print(f"\nChuyển sang: {current_mode}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
