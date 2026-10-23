# Dự án mô phỏng giọng nói con người (Voice Cloning)
**Môn học:** Nhập môn trí tuệ nhân tạo - ĐH Cần Thơ (CTU)

## Thành viên nhóm
- Thành viên 1: Trưởng nhóm & AI Engine: Sơn Thái KIệt - B2408916
- Thành viên 2: Xử lý Dữ liệu & Âm thanh: Nguyễn Quốc Vinh - B2408938
- Thành viên 3: Lập trình Giao diện (UI): Trần Trung Kiên - B2408915
- Thành viên 4: Kiểm thử & Đóng gói: Phan Ngọc Thế Hy - B2408909

## 📋 Yêu cầu hệ thống

* **Hệ điều hành:** Windows 10 / 11
* **Python:** Phiên bản `3.10.x` (khuyến nghị `3.10.11`)
* **Git:** Đã cài đặt trên máy

---

## 🚀 Hướng dẫn Cài đặt & Chạy cho Thành viên

```powershell
# 1. Clone dự án về máy
git clone <URL_REPOSITORY_CUA_NHOM>
cd voice-cloning-project

# 2. Tạo và kích hoạt môi trường ảo
python -m venv venv
.\venv\Scripts\Activate.ps1
# (Nếu gặp lỗi Execution Policy, hãy chạy: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process rồi kích hoạt lại)

# 3. Cài đặt thư viện phụ thuộc
pip install --upgrade pip
pip install -r requirements.txt

# 4. Thêm weights mô hình RVC (do Thành viên 2 bàn giao)
# Copy 2 file giong_mau.pth và giong_mau.index vào thư mục weights/

# 5. Kiểm thử Core AI Engine (Thành viên 1 & 4)
python src/ai_engine.py
# (Kết quả xuất tại outputs/test_rvc_vi.mp3)

# 6. Khởi chạy Giao diện Web UI Gradio (Thành viên 3)
python src/app.py
# (Truy cập trên trình duyệt tại địa chỉ: [http://127.0.0.1:7860](http://127.0.0.1:7860))