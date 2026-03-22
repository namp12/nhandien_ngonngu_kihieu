import pandas as pd
import numpy as np
import os
import json

BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

def static_stats():
    print("--- THỐNG KÊ STATIC DATASET ---")
    csv_path = os.path.join(BASE_DIR, 'landmarks_mp.csv')
    if not os.path.exists(csv_path):
        print("Không tìm thấy landmarks_mp.csv")
        return
    
    df = pd.read_csv(csv_path)
    total = len(df)
    counts = df['label'].value_counts()
    
    print(f"Tổng số mẫu Static: {total}")
    print("Phân bổ theo lớp (Alphabet):")
    print(counts)
    print("\n")

def dynamic_stats():
    print("--- THỐNG KÊ DYNAMIC DATASET ---")
    y_path = os.path.join(BASE_DIR, 'dynamic_y.npy')
    labels_path = os.path.join(BASE_DIR, 'dynamic_labels.json')
    
    if not os.path.exists(y_path) or not os.path.exists(labels_path):
        print("Không tìm thấy dữ liệu Dynamic")
        return
    
    y = np.load(y_path)
    with open(labels_path, 'r') as f:
        labels_dict = json.load(f)
    
    total = len(y)
    unique, counts = np.unique(y, return_counts=True)
    
    print(f"Tổng số chuỗi (Sequences) Dynamic: {total}")
    print("Phân bổ theo hành động (WLASL-100):")
    for idx, count in zip(unique, counts):
        label_name = labels_dict[str(idx)]
        print(f"{label_name}: {count} mẫu")

if __name__ == "__main__":
    static_stats()
    dynamic_stats()
