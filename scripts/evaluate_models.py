import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import json

# Đường dẫn
BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

def evaluate_static():
    print("Đang đánh giá Static Model...")
    csv_path = os.path.join(BASE_DIR, 'landmarks_mp.csv')
    df = pd.read_csv(csv_path)
    X = df.drop('label', axis=1).values
    y = df['label'].values
    
    with open(os.path.join(BASE_DIR, 'label_encoder.pkl'), 'rb') as f:
        le = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    y_encoded = le.transform(y)
    X_scaled = scaler.transform(X)
    
    # Split tương tự lúc train
    _, X_test, _, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
    
    model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'asl_static_mp.h5'))
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix - Static Model (Alphabet)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(BASE_DIR, 'static_confusion_matrix.png'))
    print("Đã lưu static_confusion_matrix.png")

def evaluate_dynamic():
    print("Đang đánh giá Dynamic Model...")
    X = np.load(os.path.join(BASE_DIR, 'dynamic_X.npy'))
    y = np.load(os.path.join(BASE_DIR, 'dynamic_y.npy'))
    
    with open(os.path.join(BASE_DIR, 'dynamic_labels.json'), 'r') as f:
        labels_dict = json.load(f)
    label_names = [labels_dict[str(i)] for i in range(len(labels_dict))]
    
    with open(os.path.join(BASE_DIR, 'dynamic_scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
        
    # Scale X (reshape to 2D first as it was trained)
    N, T, F = X.shape
    X_reshaped = X.reshape(-1, F)
    X_scaled = scaler.transform(X_reshaped).reshape(N, T, F)
    
    _, X_test, _, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'asl_dynamic_lstm.h5'))
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix - Dynamic Model (Actions)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(BASE_DIR, 'dynamic_confusion_matrix.png'))
    print("Đã lưu dynamic_confusion_matrix.png")

if __name__ == "__main__":
    evaluate_static()
    evaluate_dynamic()
    print("Hoàn tất đánh giá!")
