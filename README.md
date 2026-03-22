# 🤟 Sign Language Recognition (ASL) - MediaPipe & LSTM 🚀

Hệ thống nhận diện Ngôn ngữ ký hiệu Mỹ (ASL) tích hợp, hỗ trợ cả **Bảng chữ cái (Tĩnh)** và **Hành động/Từ vựng (Động)** thời gian thực qua Webcam.

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.0-green)](https://mediapipe.dev/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)](https://tensorflow.org/)

---

## 🏛️ Kiến trúc Hệ thống (2-Stage Pipeline)

Dự án sử dụng kiến trúc phân tầng để tối ưu hiệu suất và độ chính xác:

1.  **Stage 1 - Trích xuất đặc trưng (MediaPipe):** Chuyển đổi hình ảnh Webcam thành 21 điểm mốc (landmarks) tọa độ 3D. Điều này giúp loại bỏ nhiễu từ môi trường và màu da.
2.  **Stage 2 - Phân loại (Neural Networks):**
    *   **Static Mode (MLP):** Sử dụng mạng MLP để phân loại tổ hợp 63 tọa độ thành 26 chữ cái.
    *   **Dynamic Mode (LSTM):** Sử dụng mạng LSTM để phân tích chuỗi 30 khung hình liên tiếp, nhận diện các hành động phức tạp (WLASL-100).

---

## 📊 Kết quả huấn luyện & Đánh giá

Hệ thống đạt độ chính xác cao nhờ quy trình chuẩn hóa dữ liệu chặt chẽ:
*   **Accuracy:** ~95% cho bảng chữ cái và độ tin cậy cao cho các từ vựng động.
*   **Visuals:** Xem đồ thị huấn luyện (`static_training_plot.png`) và Ma trận nhầm lẫn (`static_confusion_matrix.png`) trong thư mục `models/`.

---

## 📂 Danh mục Scripts (`scripts/`)

- `asl_unified_recognition.py`: **Script chính** chạy ứng dụng nhận diện hợp nhất.
- `collect_data.py`: Trích xuất landmarks từ ảnh để tạo dataset CSV.
- `train_model.py`: Huấn luyện mô hình MLP (Static).
- `train_dynamic.py`: Huấn luyện mô hình LSTM (Dynamic).
- `evaluate_models.py`: Tạo Ma trận nhầm lẫn (Confusion Matrix) và báo cáo chỉ số.
- `dataset_stats.py`: Thống kê số lượng mẫu và phân bổ các lớp dữ liệu.

---

## ⚙️ Hướng dẫn Cài đặt & Sử dụng (Chi tiết VSCode)

Để hệ thống hoạt động ổn định nhất trên Windows, bạn nên sử dụng **Visual Studio Code (VSCode)** và làm theo các bước sau:

### 1. Khởi tạo môi trường (Lần đầu tiên)
Mở Terminal trong VSCode (`Ctrl + Shift + ` `) và chạy:
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Cách chạy ứng dụng
Có 2 cách để khởi động hệ thống nhận diện:

**Cách A: Dùng Terminal (Ổn định nhất)**
1. Chạy lệnh kích hoạt: `.\venv\Scripts\activate`
2. Chạy script: `python scripts/asl_unified_recognition.py`

**Cách B: Dùng nút Run (Hình tam giác)**
*   Bạn cần chọn đúng **Interpreter**: Nhấn `Ctrl + Shift + P` -> Gõ `Select Interpreter` -> Chọn dòng có chữ `('venv': venv)`.
*   Sau đó nhấn nút **Run** ở góc trên bên phải.

### ⌨️ Phím tắt điều khiển:
*   **Phím 's':** Chuyển đổi giữa chế độ **STATIC** (Chữ cái) và **DYNAMIC** (Hành động).
*   **Phím 'q':** Thoát ứng dụng hoàn toàn.

---

## 📸 Xử lý lỗi Camera (Troubleshooting)

Nếu bạn gặp lỗi "Không thể mở Camera" hoặc "Mất kết nối":
1.  **Tắt các ứng dụng khác:** Đảm bảo Zalo, Zoom, Teams hoặc Trình duyệt đang không chiếm dụng Camera.
2.  **Chờ khởi động:** Hệ thống có cơ chế *Warm-up* 3 giây, hãy kiên nhẫn đợi đèn báo Camera sáng lên.
3.  **Đổi Index:** Nếu máy có nhiều camera (Cam rời, Cam ảo), hãy mở file `scripts/asl_unified_recognition.py` và đổi `cv2.VideoCapture(0, cv2.CAP_DSHOW)` thành `(1, ...)` hoặc `(2, ...)`.

---

## 💡 Lưu ý Tối ưu
*   **Chuẩn hóa:** Hệ thống sử dụng *Hand-size Normalization* để đảm bảo nhận diện đúng ở mọi khoảng cách.
*   **Lọc nhiễu:** Sử dụng bộ lọc *Counter-based Smoothing* để kết quả hiển thị ổn định, không bị nhảy chữ.

---
*Phát triển bởi namp12*