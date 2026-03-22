import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import numpy as np

# Cấu hình MediaPipe Tasks
# Chỉnh lại đường dẫn vì script nằm trong thư mục scripts/
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'hand_landmarker.task')
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

DATA_DIR = r'e:\nahndien_ngonngu_kyhieu\dataset\kyhieutinh\asl_alphabet_train\asl_alphabet_train'
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '..', 'models', 'landmarks_mp.csv')

data = []
labels = []

print("Bắt đầu trích xuất landmarks...")

for label in sorted(os.listdir(DATA_DIR)):
    folder_path = os.path.join(DATA_DIR, label)
    if not os.path.isdir(folder_path):
        continue
    
    print(f"Đang xử lý: {label}")
    count = 0
    for img_name in os.listdir(folder_path):
        if count >= 50: break
        
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        detection_result = detector.detect(mp_image)
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                landmark_list = []
                for lm in hand_landmarks:
                    landmark_list.append(lm.x)
                    landmark_list.append(lm.y)
                    landmark_list.append(lm.z)
                data.append(landmark_list)
                labels.append(label)
                count += 1

df = pd.DataFrame(data)
df['label'] = labels
df.to_csv(OUTPUT_FILE, index=False)
print(f"Xong! Đã lưu vào {OUTPUT_FILE}")
