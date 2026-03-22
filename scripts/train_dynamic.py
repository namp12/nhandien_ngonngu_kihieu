import numpy as np
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle

# Cấu hình đường dẫn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
X_PATH = os.path.join(MODELS_DIR, 'dynamic_X.npy')
y_PATH = os.path.join(MODELS_DIR, 'dynamic_y.npy')

def augment_data(X, y):
    """Tạo thêm dữ liệu mô phỏng để tránh overfitting (Cách 2)"""
    X_aug = []
    y_aug = []
    print(f"Đang thực hiện Augmentation (Gốc: {len(X)} mẫu)...")
    
    for i in range(len(X)):
        # 1. Gốc
        X_aug.append(X[i])
        y_aug.append(y[i])
        
        # 2. Thêm nhiễu Gauss nhẹ (Noise)
        noise = np.random.normal(0, 0.005, X[i].shape)
        X_aug.append(X[i] + noise)
        y_aug.append(y[i])
        
        # 3. Scaling (Phóng to/thu nhỏ bàn tay nhẹ)
        scale = np.random.uniform(0.9, 1.1)
        X_aug.append(X[i] * scale)
        y_aug.append(y[i])
        
        # 4. Tịnh tiến (Shift)
        shift = np.random.uniform(-0.1, 0.1, (1, 63))
        X_aug.append(X[i] + shift)
        y_aug.append(y[i])

    return np.array(X_aug), np.array(y_aug)

def train_dynamic_model():
    print("Đang tải dữ liệu...")
    X = np.load(X_PATH).astype(np.float32) # (N, 30, 63)
    y = np.load(y_PATH)
    
    # Thực hiện Augmentation TRƯỚC khi scaling toàn cục
    X, y = augment_data(X, y)
    
    num_samples, seq_len, num_features = X.shape
    num_classes = len(np.unique(y))
    
    # 1. Scaling: Reshape to (N*30, 63), scale, then reshape back
    print("Đang chuẩn hóa dữ liệu với StandardScaler...")
    scaler = StandardScaler()
    X_reshaped = X.reshape(-1, num_features)
    X_scaled = scaler.fit_transform(X_reshaped)
    X = X_scaled.reshape(num_samples, seq_len, num_features)
    
    # Lưu scaler để dùng lúc inference
    with open(os.path.join(MODELS_DIR, 'dynamic_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    y_cat = to_categorical(y, num_classes=num_classes)
    
    print(f"Dataset sau khi Augmentation: {X.shape}, Classes: {num_classes}")
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.1, random_state=42)
    
    # 2. Xây dựng Model với Dropout cực mạnh
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(seq_len, num_features)),
        Dropout(0.5), # Tăng dropout để tránh overfitting
        BatchNormalization(),
        
        LSTM(256, return_sequences=False),
        Dropout(0.5),
        BatchNormalization(),
        
        Dense(256, activation='relu'), # Tăng thêm layer Dense
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("Đang huấn luyện (có thể hơi lâu vì dữ liệu đã gấp 4 lần)...")
    history = model.fit(
        X_train, y_train,
        epochs=200, # Tăng epoch
        batch_size=64, # Tăng batch size
        validation_data=(X_test, y_test),
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True)]
    )
    
    # Lưu model
    model_save_path = os.path.join(MODELS_DIR, 'asl_dynamic_lstm.h5')
    model.save(model_save_path)
    print(f"Đã lưu model tại: {model_save_path}")
    
    # Vẽ đồ thị
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.legend(); plt.title('Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend(); plt.title('Loss')
    
    plt.savefig(os.path.join(MODELS_DIR, 'dynamic_training_plot.png'))
    print("Đã lưu đồ thị quá trình huấn luyện.")

if __name__ == "__main__":
    train_dynamic_model()
