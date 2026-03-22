import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os

# Đường dẫn tương đối
base_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
csv_path = os.path.join(base_dir, 'landmarks_mp.csv')

# 1. Tải dữ liệu
df = pd.read_csv(csv_path)
X = df.drop('label', axis=1).values
y = df['label'].values

# 2. Tiền xử lý
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Mô hình MLP
model = models.Sequential([
    layers.Input(shape=(63,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

import matplotlib.pyplot as plt

print("Bắt đầu huấn luyện MLP...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# 4. Vẽ và lưu đồ thị huấn luyện
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.title('Static Model Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('Static Model Loss')
plt.legend()

plt.savefig(os.path.join(base_dir, 'static_training_plot.png'))
print("Đã lưu đồ thị vào models/static_training_plot.png")

# 5. Lưu mô hình và các đối tượng hỗ trợ
model.save(os.path.join(base_dir, 'asl_static_mp.h5'))
with open(os.path.join(base_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
with open(os.path.join(base_dir, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(label_encoder, f)

print("Đã lưu mô hình và các files vào thư mục models/")
