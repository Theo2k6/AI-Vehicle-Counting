# Đề 19: Đếm phương tiện qua vạch (AI & Ứng dụng)

Hệ thống sử dụng mô hình **YOLOv8** (phát hiện) kết hợp thuật toán **ByteTrack** (theo dõi đa đối tượng) để đếm số lượng xe đi qua một vạch ảo trong video giao thông.

## 1. Chức năng chính
- Phát hiện và theo dõi các loại phương tiện: Ô tô, Xe máy, Xe buýt, Xe tải.
- Đếm số lượng xe đi qua vạch theo 2 chiều (UP/DOWN).
- Xuất video kết quả có vẽ bounding box, ID và số đếm.
- Đánh giá độ chính xác của thuật toán theo dõi (Tracking) thông qua chỉ số **MOTA** trên bộ dữ liệu chuẩn **MOT17**.

## 2. Môi trường thực nghiệm
- **OS:** WSL2 (Ubuntu 22.04) / Windows 11
- **Python:** 3.11
- **GPU:** NVIDIA GeForce RTX 4050
- **Thư viện chính:** ultralytics (YOLOv8), supervision (ByteTrack), motmetrics, opencv-python.

## 3. Hướng dẫn cài đặt
Yêu cầu máy tính đã cài đặt Python 3.11+ và Git.

**Bước 1: Clone dự án**
`git clone https://github.com/Theo2k6/AI-Vehicle-Counting.git`
`cd AI-Vehicle-Counting`

**Bước 2: Tạo và kích hoạt môi trường ảo**
- *Linux/macOS/WSL:* `python3 -m venv .venv` và `source .venv/bin/activate`
- *Windows CMD:* `python -m venv .venv` và `.venv\Scripts\activate.bat`

**Bước 3: Cài đặt thư viện**
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
`pip install ultralytics supervision motmetrics pandas opencv-python lap`

## 4. Hướng dẫn chuẩn bị Dữ liệu
Để chạy được code, bạn cần chuẩn bị 2 bộ dữ liệu sau và đặt đúng vào thư mục `data/` ở thư mục gốc.

### 4.1. Video giao thông thực tế (Để đếm xe)
1. Tải video từ link Google Drive (do nhóm cung cấp).
2. Đổi tên thành `traffic_720p.mp4` và copy vào thư mục `data/`.
   - *Lưu ý:* Nếu muốn dùng video khác, chạy lệnh với cờ `--video` ở Mục 5.

### 4.2. Bộ dữ liệu MOT17 (Để tính điểm MOTA)
1. Tải file `MOT17.zip` từ `https://motchallenge.net/data/MOT17.zip` và giải nén.
2. Copy thư mục `MOT17` vào thư mục `data/`.
3. **Lưu ý đường dẫn:** Đảm bảo file đáp án `gt.txt` nằm chính xác tại: `data/MOT17/MOT17/train/MOT17-04-FRCNN/gt/gt.txt`

## 5. Cách chạy chương trình

### 5.1. Đếm xe trên video giao thông
Mở Terminal và chạy:
`python count_vehicles.py`
*(Kết quả video và file CSV sẽ tự động lưu vào thư mục `output/`).*

### 5.2. Đánh giá điểm MOTA trên MOT17
Chạy lần lượt 2 lệnh sau:
`python export_mot_format.py`
`python evaluate_mota.py`
*(Điểm MOTA được in ra màn hình và lưu tại `output/mota_report.txt`).*

## 6. Kết quả đạt được
- **Chỉ số MOTA trên MOT17:** `0.5326` (Đạt yêu cầu đề bài >= 0.50).
- **Kết quả đếm thực tế (video 2 phút):** 114 xe hơi, 6 xe buýt, 18 xe tải.

## 7. Phân tích lỗi điển hình
- **Che khuất:** Xe tải lớn che khuất xe con khiến YOLO bỏ sót (Misses).
- **Vật thể nhỏ ở xa:** Xe ở làn đường xa bị thu nhỏ khi resize 640x640 khiến AI khó nhận diện.
- **Đổi ID (ID Switch):** ByteTrack mất dấu xe khi bị che khuất quá lâu, phải cấp ID mới.
