# Dự án mô phỏng giọng nói con người (Voice Cloning)
**Môn học:** Nhập môn trí tuệ nhân tạo - ĐH Cần Thơ (CTU)

---

## Thành viên nhóm & Phân công nhiệm vụ

- **Thành viên 1:** **Sơn Thái Kiệt** - B2408916  
  *Vai trò:* Trưởng nhóm & AI Engine (`src/ai_engine.py` - Tích hợp mô hình Zero-Shot F5-TTS tiếng Việt, quản lý quy trình lưu và tổng hợp giọng mẫu).
- **Thành viên 2:** **Nguyễn Quốc Vinh** - B2408938  
  *Vai trò:* Xử lý Dữ liệu & Tối ưu mô hình (Chuẩn hóa dữ liệu giọng mẫu đầu vào, tinh chỉnh tham số suy luận F5-TTS tối ưu tốc độ trên CPU/GPU).
- **Thành viên 3:** **Trần Trung Kiên** - B2408915  
  *Vai trò:* Lập trình Giao diện Web UI (`src/app.py` - Xây dựng giao diện tương tác Text-to-Speech, thu âm giọng mẫu, xác nhận ghi đè và tự động đồng bộ danh sách giọng trên Gradio).
- **Thành viên 4:** **Phan Ngọc Thế Hy** - B2408909  
  *Vai trò:* Tiền xử lý âm thanh & Kiểm thử hệ thống & Làm sildes (`src/audio_processor.py` - Lọc nhiễu, chuẩn hóa âm thanh tiếng Việt; `tests/test_engine.py` - Test suite tự động hóa toàn luồng).

---

## Yêu cầu hệ thống

* **Hệ điều hành:** Windows 10 / 11 (64-bit)
* **Python:** Phiên bản khuyến nghị `3.10.11`
* **Phần cứng:** Hỗ trợ chạy trên cả **CPU** (đã tối ưu `nfe_step=16` và giới hạn mẫu 6 giây) lẫn **NVIDIA GPU** (CUDA 12.1 để tăng tốc suy luận).
* **Git:** Đã cài đặt trên máy.

---

## Cấu trúc thư mục dự án (Project Structure)

```text
voice-cloning-ctu/
├── outputs/                      # Chứa file âm thanh kết quả được sinh ra
├── saved_voices/                 # Lưu trữ giọng mẫu cá nhân (.wav) & văn bản tham chiếu (.txt)
├── src/                          # Thư mục mã nguồn chính của dự án
│   ├── __init__.py               # Package marker
│   ├── ai_engine.py              # [Thành viên 1] Core AI Engine (Zero-Shot F5-TTS Vietnamese)
│   ├── app.py                    # [Thành viên 3] Giao diện Web tương tác (Gradio)
│   └── audio_processor.py        # [Thành viên 4] Tiền xử lý âm thanh & lọc nhiễu chuẩn ngữ âm tiếng Việt
├── tests/                        # Thư mục kiểm thử tự động
│   └── test_engine.py            # [Thành viên 4] Test Suite toàn luồng (Unit & E2E Tests)
├── .gitignore                    # Danh sách loại trừ file nhị phân / cache khi commit Git
├── README.md                     # Tài liệu hướng dẫn cài đặt & vận hành dự án
└── requirements.txt              # Danh sách các thư viện phụ thuộc của dự án

## Hướng dẫn cài đặt & Chạy hệ thống

### 1. Clone dự án và tạo môi trường ảo

```powershell
# Clone mã nguồn về máy
git clone <URL_REPOSITORY_CUA_NHOM>
cd voice-cloning-ctu

# Tạo môi trường ảo Python 3.10
python -m venv venv

# Kích hoạt môi trường ảo
.\venv\Scripts\Activate.ps1
# (Nếu gặp lỗi ExecutionPolicy, chạy: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process rồi kích hoạt lại)
```

### 2. Cài đặt các thư viện phụ thuộc

```powershell
python -m pip install --upgrade pip

# Cài đặt toàn bộ thư viện cần thiết
pip install -r requirements.txt
```

> [!NOTE]
> Đối với máy chạy Windows có GPU NVIDIA: Khuyến nghị cài đặt bản PyTorch hỗ trợ CUDA 12.1 và fairseq pre-built wheel:
> ```powershell
> pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
> pip install https://huggingface.co/Jmica/rvc/resolve/01b388e059df1218a5a7b48b91305b2e06fed030/fairseq-0.12.2-cp310-cp310-win_amd64.whl
> ```

### 3. Khởi chạy Giao diện Web UI Gradio (Thành viên 3)
```powershell
python src/app.py
```
Truy cập giao diện Web trên trình duyệt tại địa chỉ: **[http://127.0.0.1:7860](http://127.0.0.1:7860)**.

---

## Tiền xử lý âm thanh (Audio Preprocessing - `audio_processor.py`)

Module `src/audio_processor.py` (do **Thành viên 4** phụ trách) đóng vai trò là "chốt chặn chất lượng âm thanh đầu vào" cho toàn bộ pipeline, được xây dựng dựa trên kiến trúc xử lý của **Applio** kết hợp tối ưu đặc thù ngữ âm tiếng Việt từ **Vietnamese-RVC-Project**.

### Pipeline 5 bước xử lý:
1. **Load & Chuẩn hóa định dạng (Format Normalization)**:
   - Đọc các định dạng âm thanh thông dụng (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`).
   - Tự động convert Stereo thành Mono (trung bình cộng các kênh) và chuyển định dạng mảng sang `float32` trong khoảng `[-1.0, 1.0]`.
   - Xuất song song 2 tần số lấy mẫu: **40,000Hz / 48,000Hz** (chuẩn Vocoder RVC/Applio) và **16,000Hz** (chuẩn HuBERT ContentVec / RMVPE).
2. **Lọc tần số siêu trầm (High-pass Filter - HPF)**:
   - Áp dụng bộ lọc số Butterworth bậc 5 với tần số cắt $f_c = 48\text{ Hz}$ (chuẩn pipeline Applio).
   - Triệt tiêu hoàn toàn tiếng gió mic, rung động cơ học và tiếng ù nguồn điện (hum/rumble 50Hz) mà không làm suy hao thanh vị âm F0 của tiếng Việt.
3. **Khử nhiễu bảo toàn thanh điệu tiếng Việt (Tone-Preserving Noise Reduction)**:
   - Tích hợp spectral gating thông qua `noisereduce` với thông số tối ưu `prop_decrease=0.75` (không dùng mức 1.0 quá gắt).
   - Giữ nguyên toàn vẹn dải tần số cơ bản $F0$ (80 – 400 Hz) vốn là dải quyết định 6 thanh điệu tiếng Việt (ngang, huyền, sắc, hỏi, ngã, nặng) và bảo toàn các phụ âm bật hơi, âm ma sát tần số cao (s, x, ch, tr).
4. **Cắt khoảng lặng thông minh (Silence Trimming & Slicing)**:
   - Dựa trên cơ chế slicer ngưỡng năng lượng `top_db=40 dB`.
   - Giữ lại một khoảng đệm an toàn `pad_ms=100 ms` ở hai đầu mỗi phân đoạn phát âm, chống hiện tượng "nuốt âm" hoặc cộc âm cuối rất hay gặp trong tiếng Việt.
   - Hỗ trợ hàm `process_to_chunks()` tự động chia nhỏ file âm thanh dài thành các đoạn $\le 10$ giây để tránh tràn VRAM GPU khi inference.
5. **Chuẩn hóa âm lượng chống vỡ tiếng (Anti-clipping Loudness Normalization)**:
   - Chuẩn hóa Peak đạt mức tiêu chuẩn $-1.0\text{ dBFS}$ kết hợp cơ chế giới hạn biên độ chống clipping âm học.
6. **Xử lý ngoại lệ toàn diện (Fault Tolerance)**:
   - Kiểm tra và bắt lỗi tự động đối với file không tồn tại, file rỗng 0 bytes, định dạng hỏng, hoặc file chỉ chứa hoàn toàn khoảng lặng/tiếng ồn nền.

### Hướng dẫn sử dụng `AudioProcessor` trong mã nguồn

```python
from pathlib import Path
from audio_processor import AudioProcessor, AudioProcessorConfig

# 1. Khởi tạo cấu hình tiền xử lý (có thể tinh chỉnh theo yêu cầu)
config = AudioProcessorConfig(
    target_sr=40000,          # 40kHz chuẩn RVC v2
    hubert_sr=16000,          # 16kHz chuẩn HuBERT
    highpass_cutoff=48.0,     # Cắt tần số thấp dưới 48Hz
    noise_reduction=True,     # Bật lọc nhiễu spectral gating
    noise_prop_decrease=0.75, # Bảo toàn thanh điệu tiếng Việt
    silence_top_db=40.0,      # Ngưỡng phát hiện khoảng lặng (dB)
    silence_pad_ms=100.0,     # Đệm 100ms chống cụt đuôi âm
    target_peak_db=-1.0       # Chuẩn hóa âm lượng đỉnh
)

processor = AudioProcessor(config=config)

# 2. Xử lý một file âm thanh bất kỳ
# Trả về: audio_target (40kHz float32) và audio_hubert (16kHz float32)
audio_40k, audio_16k = processor.process("data/samples/sample.wav")

# 3. (Tùy chọn) Lưu ra file kết quả sau xử lý
processor.process_and_save("data/samples/sample.wav", "outputs/sample_clean.wav")

# 4. (Tùy chọn cho file dài) Chia thành các chunk nhỏ ≤ 10s
chunks = processor.process_to_chunks("long_audio.wav", max_duration_s=10.0)
```

### Lệnh chạy tiền xử lý độc lập từ CLI

```powershell
# Tiền xử lý file âm thanh bất kỳ từ terminal
python src/audio_processor.py data/samples/sample.wav outputs/sample_preprocessed.wav
```

---

## Kiểm thử tự động (Automated Testing - `test_engine.py`)

Module `tests/test_engine.py` (do **Thành viên 4** phụ trách) cung cấp bộ kiểm thử tự động toàn diện chuẩn CI/CD cho dự án, cho phép xác thực độ tin cậy từ cấp độ Unit (từng module) đến Integration (toàn luồng suy luận E2E) mà **không phụ thuộc vào dữ liệu bên ngoài**.

### Các kịch bản kiểm thử được bao phủ:

1. **Bộ giả lập âm thanh (Mock Audio Fixture)**:
   - Tự động sinh sóng hình sin $F0 = 180\text{ Hz}$ (mô phỏng giọng nói nam tiếng Việt) + nhiễu trắng ngẫu nhiên + khoảng lặng 2 đầu.
   - Cho phép toàn bộ test suite chạy độc lập, khép kín trong thư mục tạm (`tempfile`).
2. **Nhóm 1: Unit Test cho `AudioProcessor` (10 tests)**:
   - Kiểm tra định dạng đầu ra chuẩn `float32`, Mono 1 chiều, biên độ nghiêm ngặt trong $[-1.0, 1.0]$.
   - Kiểm tra độ chính xác của tần số lấy mẫu ($40,000\text{ Hz}$ và $16,000\text{ Hz}$).
   - Kiểm tra khả năng suy hao tần số thấp của bộ lọc Butterworth HPF 48Hz.
   - Kiểm tra tính năng cắt khoảng lặng làm giảm thời lượng đúng kỳ vọng.
   - Bắt các trường hợp ngoại lệ: File không tồn tại (`FileNotFoundError`), file rỗng 0 bytes (`ValueError`), file im lặng hoàn toàn (`ValueError`).
3. **Nhóm 2: Integration Test toàn luồng & Hiệu năng (6 tests)**:
   - Luồng tích hợp xuyên suốt: `Input Audio` $\to$ `AudioProcessor` $\to$ `Mock RVC Inference` $\to$ `Output WAV/MP3`.
   - Kiểm thử chuyển đổi tín hiệu đa kênh (Stereo $\to$ Mono).
   - Kiểm thử cơ chế cắt nhỏ file âm thanh dài (`process_to_chunks`).
   - Giám sát độ trễ (Latency Budget $\le 60\text{s}$).
   - Giám sát rò rỉ bộ nhớ CPU (`tracemalloc` theo dõi heap allocation qua $N=5$ lần lặp, delta $\le 50\text{ MB}$).
4. **Nhóm 3: Smoke Test Môi trường & Trọng số (8 tests)**:
   - Kiểm tra sự tồn tại và tính hợp lệ của trọng số `weights/giong_mau.pth` và `weights/giong_mau.index`.
   - Kiểm tra khả năng import và tương thích của các thư viện cốt lõi (`torch`, `fairseq`, `rvc_python`, `edge_tts`, `scipy`, `soundfile`, `numpy`).
   - Kiểm tra trạng thái tăng tốc phần cứng CUDA GPU.
5. **Nhóm 4: Kiểm thử chuyên biệt cho Ngữ âm Tiếng Việt (2 tests)**:
   - Kiểm thử bảo toàn pha tín hiệu sau khử nhiễu (chống đảo pha âm học).
   - Kiểm thử bảo toàn tần số cơ bản F0 tiếng Việt dải $80 - 400\text{ Hz}$ (đạt hệ số tương quan Pearson $> 0.70$ và năng lượng chuẩn hóa $> 20\%$).

### Hướng dẫn chạy kiểm thử trên Terminal

```powershell
# Kích hoạt venv (nếu chưa kích hoạt)
.\venv\Scripts\Activate.ps1

# Cách 1: Chạy toàn bộ test suite với báo cáo chi tiết từng bước (Khuyến nghị)
pytest tests/test_engine.py -v -s

# Cách 2: Chạy trực tiếp qua unittest runner tích hợp sẵn
python tests/test_engine.py
```

### Kết quả kiểm thử thực tế (Benchmark Data)

```text
============================= test session starts =============================
platform win32 -- Python 3.10.11, pytest-9.2.0, pluggy-1.7.0
rootdir: D:\voice-cloning-ctu
collected 26 items

tests/test_engine.py::TestAudioProcessorUnit::test_duration_reduced_after_trim PASSED
tests/test_engine.py::TestAudioProcessorUnit::test_hpf_removes_low_frequency PASSED
tests/test_engine.py::TestAudioProcessorUnit::test_output_amplitude_range PASSED
tests/test_engine.py::TestAudioProcessorUnit::test_output_dtype_float32 PASSED
tests/test_engine.py::TestAudioProcessorUnit::test_output_is_mono PASSED
tests/test_engine.py::TestAudioProcessorUnit::test_output_sample_rate_16k PASSED
tests/test_engine.py::TestAudioProcessorUnit::test_output_sample_rate_40k PASSED
tests/test_engine.py::TestAudioProcessorUnit::test_raises_on_empty_file PASSED
tests/test_engine.py::TestAudioProcessorUnit::test_raises_on_missing_file PASSED
tests/test_engine.py::TestAudioProcessorUnit::test_raises_on_silent_file PASSED
tests/test_engine.py::TestEndToEndPipeline::test_full_pipeline_with_mock_rvc PASSED
tests/test_engine.py::TestEndToEndPipeline::test_integration_with_ai_engine_mock PASSED
tests/test_engine.py::TestEndToEndPipeline::test_no_significant_memory_leak PASSED
tests/test_engine.py::TestEndToEndPipeline::test_pipeline_latency_budget PASSED
tests/test_engine.py::TestEndToEndPipeline::test_process_to_chunks_long_audio PASSED
tests/test_engine.py::TestEndToEndPipeline::test_stereo_input_converted_to_mono PASSED
tests/test_engine.py::TestSystemDependencies::test_edge_tts_import PASSED
tests/test_engine.py::TestSystemDependencies::test_numpy_version PASSED
tests/test_engine.py::TestSystemDependencies::test_rvc_python_import PASSED
tests/test_engine.py::TestSystemDependencies::test_scipy_import PASSED
tests/test_engine.py::TestSystemDependencies::test_soundfile_import PASSED
tests/test_engine.py::TestSystemDependencies::test_torch_cuda_available PASSED
tests/test_engine.py::TestSystemDependencies::test_weights_index_exists PASSED
tests/test_engine.py::TestSystemDependencies::test_weights_pth_exists PASSED
tests/test_engine.py::TestNoiseReduction::test_noise_reduction_does_not_flip_sign PASSED
tests/test_engine.py::TestNoiseReduction::test_noise_reduction_preserves_f0_frequency PASSED

============================= 26 passed in 12.20s =============================
```

- **Tỷ lệ Pass:** **26/26 Tests PASSED (100%)**
- **Thời gian xử lý trung bình (Latency trên mẫu giọng thật `models/sample.wav`):**
  - Khi không bật khử nhiễu: **~30.99 ms**
  - Khi bật khử nhiễu bảo toàn thanh điệu: **~195.33 ms** (Đáp ứng hoàn toàn ngưỡng thời gian thực)
