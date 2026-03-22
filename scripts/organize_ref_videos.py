import os
import json
import shutil

# Cấu hình đường dẫn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(BASE_DIR, 'dataset', 'kyhieudong', 'WLASL', 'WLASL_v0.3.json')
VIDEOS_DIR = os.path.join(BASE_DIR, 'dataset', 'kyhieudong', 'WLASL', 'videos')
LABELS_PATH = os.path.join(BASE_DIR, 'models', 'dynamic_labels.json')
REF_DIR = os.path.join(BASE_DIR, 'reference_videos')

if not os.path.exists(REF_DIR):
    os.makedirs(REF_DIR)

def organize_ref_videos():
    print("Đang tìm video ví dụ cho 100 hành động...", flush=True)
    
    with open(LABELS_PATH, 'r') as f:
        labels_map = json.load(f)
    
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
        
    # Tạo mapping từ gloss sang instances để tra cứu nhanh
    gloss_to_instances = {item['gloss']: item['instances'] for item in data}
    
    count = 0
    for idx_str, gloss in labels_map.items():
        idx = int(idx_str)
        if gloss in gloss_to_instances:
            instances = gloss_to_instances[gloss]
            
            # Tìm video đầu tiên tồn tại trong folder videos
            found = False
            for inst in instances:
                video_id = inst['video_id']
                src_path = os.path.join(VIDEOS_DIR, f"{video_id}.mp4")
                
                if os.path.exists(src_path):
                    # Copy và đổi tên thành '00_book.mp4'
                    dst_name = f"{idx:02d}_{gloss}.mp4"
                    dst_path = os.path.join(REF_DIR, dst_name)
                    
                    shutil.copy2(src_path, dst_path)
                    print(f"  [{idx:02d}] Đã lưu video mẫu cho: {gloss}", flush=True)
                    count += 1
                    found = True
                    break
            
            if not found:
                print(f"  [!!] CẢNH BÁO: Không tìm thấy video mẫu cho: {gloss}", flush=True)
                
    print(f"\nHoàn thành! Đã lưu {count} video vào thư mục: {REF_DIR}", flush=True)

if __name__ == "__main__":
    organize_ref_videos()
