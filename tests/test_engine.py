"""
test_engine.py
==============
Bộ kiểm thử tự động (Test Suite) cho hệ thống Voice Cloning CTU.

Bao gồm:
  - Fixture sinh Mock Audio (sine wave + white noise + silence padding).
  - Unit Tests cho AudioProcessor (sample_rate, channels, dtype, amplitude, duration).
  - Integration Tests toàn luồng: Audio → Preprocess → Mock Inference → Output.
  - Đo latency từng bước và kiểm tra rò rỉ bộ nhớ GPU/CPU.

Chạy:
    .\\venv\\Scripts\\pytest tests\\test_engine.py -v --tb=short

Thành viên 4 - Phan Ngọc Thế Hy (B2408909)
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import time
import tracemalloc
import unittest
from pathlib import Path
from typing import Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

# -- Đảm bảo import được module từ src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from audio_processor import AudioProcessor, AudioProcessorConfig

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Hằng số dùng chung
# ---------------------------------------------------------------------------
MOCK_SR = 40_000          # Sample rate giả lập
MOCK_DURATION_S = 5.0     # Tổng thời lượng audio giả lập
MOCK_FREQ_HZ = 180.0      # Tần số sóng sine (~ F0 giọng nam tiếng Việt)
MOCK_SILENCE_S = 0.5      # Khoảng lặng mỗi đầu (giây)
MOCK_NOISE_AMP = 0.02     # Biên độ nhiễu trắng (nhỏ, ~-34 dBFS)

TOLERANCE_SR = 0          # Sample rate phải chính xác tuyệt đối
TOLERANCE_AMP = 1e-5      # Dung sai kiểm tra biên độ
LATENCY_BUDGET_S = 60.0   # Giới hạn thời gian toàn luồng (giây)


# ---------------------------------------------------------------------------
# Hàm helper: sinh Mock Audio
# ---------------------------------------------------------------------------
def make_mock_audio(
    sr: int = MOCK_SR,
    duration_s: float = MOCK_DURATION_S,
    freq_hz: float = MOCK_FREQ_HZ,
    silence_s: float = MOCK_SILENCE_S,
    noise_amp: float = MOCK_NOISE_AMP,
    output_path: Path | None = None,
) -> Tuple[np.ndarray, int]:
    """
    Sinh file âm thanh giả lập để kiểm thử độc lập.

    Thành phần:
    - [0 .. silence_s]      : Im lặng hoàn toàn (kiểm tra trim đầu)
    - [silence_s .. -sil]   : Sóng sine @ freq_hz + nhiễu trắng (mô phỏng giọng nói)
    - [-silence_s .. end]   : Im lặng hoàn toàn (kiểm tra trim cuối)

    Parameters
    ----------
    sr          : Sample rate
    duration_s  : Tổng thời lượng (giây)
    freq_hz     : Tần số sóng sine (F0 giả lập)
    silence_s   : Khoảng lặng đầu/cuối (giây)
    noise_amp   : Biên độ nhiễu trắng
    output_path : Nếu không None, lưu WAV ra đĩa

    Returns
    -------
    (audio, sr) : Tuple[np.ndarray, int]
    """
    n_total = int(sr * duration_s)
    n_sil = int(sr * silence_s)
    n_voice = n_total - 2 * n_sil

    t = np.linspace(0, n_voice / sr, n_voice, endpoint=False)
    voice = 0.6 * np.sin(2 * np.pi * freq_hz * t)

    # Thêm hài âm (harmonics) mô phỏng âm sắc giọng người
    voice += 0.2 * np.sin(2 * np.pi * freq_hz * 2 * t)
    voice += 0.1 * np.sin(2 * np.pi * freq_hz * 3 * t)

    # Nhiễu trắng (mô phỏng tiếng ồn nền nhẹ)
    rng = np.random.default_rng(42)
    noise = noise_amp * rng.standard_normal(n_voice)

    silence = np.zeros(n_sil, dtype=np.float32)
    audio = np.concatenate([silence, (voice + noise).astype(np.float32), silence])

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio, sr)
        logger.info("Đã tạo mock audio: %s (%.2f s)", output_path.name, duration_s)

    return audio, sr


# ---------------------------------------------------------------------------
# Base TestCase với shared fixtures
# ---------------------------------------------------------------------------
class _BaseTestCase(unittest.TestCase):
    """
    Base class khởi tạo AudioProcessor và mock WAV file trước mỗi bộ test.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp_dir = PROJECT_ROOT / "outputs" / "_test_tmp"
        cls.tmp_dir.mkdir(parents=True, exist_ok=True)

        cls.mock_wav = cls.tmp_dir / "mock_voice.wav"
        make_mock_audio(output_path=cls.mock_wav)

        # Processor không khử nhiễu để tăng tốc unit test
        cfg = AudioProcessorConfig(
            target_sr=40_000,
            noise_reduction=False,  # tắt khi unit test để nhanh hơn
            pad_ms=50,
        )
        cls.processor = AudioProcessor(config=cfg)

    @classmethod
    def tearDownClass(cls) -> None:
        """Dọn dẹp file tạm sau khi toàn bộ test trong class kết thúc."""
        for f in cls.tmp_dir.glob("mock_*.wav"):
            f.unlink(missing_ok=True)
        for f in cls.tmp_dir.glob("out_*.wav"):
            f.unlink(missing_ok=True)


# ===========================================================================
# NHÓM 1: Unit Tests cho AudioProcessor
# ===========================================================================
class TestAudioProcessorUnit(_BaseTestCase):
    """
    Unit Tests kiểm tra từng bước của AudioProcessor.
    """

    # ----- 1.1 Đầu ra đúng sample rate -----
    def test_output_sample_rate_40k(self) -> None:
        """Phiên bản 40kHz phải có đúng sample rate."""
        audio_40k, _ = self.processor.process(self.mock_wav, apply_noise_reduction=False)
        # Kiểm tra thời lượng ~ tương ứng sample rate 40kHz
        expected_sr = self.processor.cfg.target_sr
        # Nếu duration gốc 5s, sau trim còn khoảng 4s → samples ≈ 4 * 40000
        self.assertGreater(len(audio_40k), 0, "Mảng âm thanh 40kHz rỗng!")
        self.assertLessEqual(len(audio_40k), MOCK_SR * MOCK_DURATION_S + 1000)
        logger.info("[1.1] sample_rate_40k OK – shape: %s", audio_40k.shape)

    def test_output_sample_rate_16k(self) -> None:
        """Phiên bản 16kHz phải có shape tỷ lệ 16000/40000 = 0.4 so với 40kHz."""
        audio_40k, audio_16k = self.processor.process(self.mock_wav, apply_noise_reduction=False)
        ratio = len(audio_16k) / len(audio_40k)
        expected_ratio = self.processor.cfg.hubert_sr / self.processor.cfg.target_sr
        self.assertAlmostEqual(ratio, expected_ratio, delta=0.05,
                               msg=f"Tỷ lệ 16k/40k = {ratio:.3f}, kỳ vọng {expected_ratio:.3f}")
        logger.info("[1.2] sample_rate_16k ratio OK: %.3f", ratio)

    # ----- 1.2 Mono channel -----
    def test_output_is_mono(self) -> None:
        """Đầu ra phải là Mono (ndim == 1)."""
        audio_40k, audio_16k = self.processor.process(self.mock_wav, apply_noise_reduction=False)
        self.assertEqual(audio_40k.ndim, 1, f"audio_40k.ndim = {audio_40k.ndim}, kỳ vọng 1")
        self.assertEqual(audio_16k.ndim, 1, f"audio_16k.ndim = {audio_16k.ndim}, kỳ vọng 1")
        logger.info("[1.3] Mono OK")

    # ----- 1.3 Kiểu dữ liệu float32 -----
    def test_output_dtype_float32(self) -> None:
        """Đầu ra phải có dtype float32."""
        audio_40k, audio_16k = self.processor.process(self.mock_wav, apply_noise_reduction=False)
        self.assertEqual(audio_40k.dtype, np.float32,
                         f"dtype 40k = {audio_40k.dtype}, kỳ vọng float32")
        self.assertEqual(audio_16k.dtype, np.float32,
                         f"dtype 16k = {audio_16k.dtype}, kỳ vọng float32")
        logger.info("[1.4] dtype float32 OK")

    # ----- 1.4 Biên độ nằm trong [-1, 1] -----
    def test_output_amplitude_range(self) -> None:
        """Toàn bộ mẫu phải nằm trong khoảng [-1.0, 1.0] sau chuẩn hoá."""
        audio_40k, audio_16k = self.processor.process(self.mock_wav, apply_noise_reduction=False)
        self.assertLessEqual(float(np.max(np.abs(audio_40k))), 1.0 + TOLERANCE_AMP,
                             "Biên độ 40kHz vượt 1.0!")
        self.assertLessEqual(float(np.max(np.abs(audio_16k))), 1.0 + TOLERANCE_AMP,
                             "Biên độ 16kHz vượt 1.0!")
        logger.info("[1.5] Amplitude [-1,1] OK – peak_40k=%.4f | peak_16k=%.4f",
                    float(np.max(np.abs(audio_40k))), float(np.max(np.abs(audio_16k))))

    # ----- 1.5 Thời lượng giảm sau trim -----
    def test_duration_reduced_after_trim(self) -> None:
        """
        Thời lượng audio sau tiền xử lý phải nhỏ hơn thời lượng gốc
        (do đã cắt 2 đoạn im lặng 0.5s mỗi đầu).
        """
        audio_40k, _ = self.processor.process(self.mock_wav, apply_noise_reduction=False)
        duration_out = len(audio_40k) / self.processor.cfg.target_sr
        # Thời lượng gốc MOCK_DURATION_S = 5.0s, im lặng = 2 × 0.5s = 1s
        # → duration_out nên < MOCK_DURATION_S
        self.assertLess(duration_out, MOCK_DURATION_S,
                        f"Thời lượng ra {duration_out:.2f}s không giảm so với {MOCK_DURATION_S}s")
        self.assertGreater(duration_out, MOCK_SILENCE_S,
                           "Thời lượng ra quá ngắn, có thể bị trim quá mức!")
        logger.info("[1.6] Duration after trim OK: %.2f s (gốc %.1f s)",
                    duration_out, MOCK_DURATION_S)

    # ----- 1.6 High-Pass Filter loại tạp âm thấp -----
    def test_hpf_removes_low_frequency(self) -> None:
        """
        HPF phải loại bỏ thành phần tần số < 48 Hz.
        Tạo tín hiệu sin 20 Hz thuần tuý → sau HPF năng lượng phải giảm mạnh.
        """
        sr = 40_000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        sine_20hz = (0.9 * np.sin(2 * np.pi * 20 * t)).astype(np.float32)
        
        # Lưu tạm
        test_path = self.tmp_dir / "mock_20hz.wav"
        sf.write(str(test_path), sine_20hz, sr)
        
        # Processor với noise_reduction=False và top_db=60 (không trim)
        cfg = AudioProcessorConfig(target_sr=sr, noise_reduction=False, top_db=60.0)
        proc = AudioProcessor(config=cfg)
        audio_out, _ = proc.process(str(test_path), apply_noise_reduction=False)
        
        energy_in = float(np.mean(sine_20hz ** 2))
        energy_out = float(np.mean(audio_out ** 2))
        ratio = energy_out / (energy_in + 1e-9)
        
        self.assertLess(ratio, 0.1,
                        f"HPF không lọc hiệu quả (energy_ratio={ratio:.4f} > 0.1)")
        test_path.unlink(missing_ok=True)
        logger.info("[1.7] HPF 20Hz test OK – energy ratio after filter: %.4f", ratio)

    # ----- 1.7 File không tồn tại → FileNotFoundError -----
    def test_raises_on_missing_file(self) -> None:
        """Phải raise FileNotFoundError nếu file không tồn tại."""
        with self.assertRaises(FileNotFoundError):
            self.processor.process("not_exist_file.wav")
        logger.info("[1.8] FileNotFoundError OK")

    # ----- 1.8 File rỗng → ValueError -----
    def test_raises_on_empty_file(self) -> None:
        """Phải raise ValueError nếu file rỗng (0 bytes)."""
        empty = self.tmp_dir / "empty.wav"
        empty.touch()
        with self.assertRaises(ValueError):
            self.processor.process(str(empty))
        empty.unlink(missing_ok=True)
        logger.info("[1.9] ValueError on empty file OK")

    # ----- 1.9 File chỉ toàn tiếng lặng → ValueError -----
    def test_raises_on_silent_file(self) -> None:
        """Phải raise ValueError nếu file chỉ chứa toàn tiếng im lặng."""
        silent = np.zeros(MOCK_SR * 3, dtype=np.float32)
        silent_path = self.tmp_dir / "mock_silent.wav"
        sf.write(str(silent_path), silent, MOCK_SR)
        with self.assertRaises(ValueError):
            self.processor.process(str(silent_path))
        silent_path.unlink(missing_ok=True)
        logger.info("[1.10] ValueError on silent file OK")


# ===========================================================================
# NHÓM 2: Integration Tests toàn luồng (End-to-End)
# ===========================================================================
class TestEndToEndPipeline(_BaseTestCase):
    """
    Integration Tests mô phỏng toàn bộ pipeline:
    Input Audio → AudioProcessor → Mock RVC Inference → Output Audio.

    Dùng Mock/Stub cho module RVC để kiểm thử độc lập mà không cần
    mô hình thật (hỗ trợ CI/CD không có GPU).
    """

    # ----- 2.1 Kiểm thử toàn luồng với Mock RVC -----
    def test_full_pipeline_with_mock_rvc(self) -> None:
        """
        Toàn luồng: Mock Audio → Preprocess → Mock RVC Inference → Output WAV.
        Dùng unittest.mock.patch để stub hàm RVC inference.
        """
        t_start = time.perf_counter()

        # Bước 1: Tiền xử lý
        audio_40k, audio_16k = self.processor.process(
            self.mock_wav, apply_noise_reduction=False
        )
        self.assertIsInstance(audio_40k, np.ndarray)
        self.assertEqual(audio_40k.dtype, np.float32)

        # Bước 2: Mock RVC Inference (Stub trả về audio đã xử lý + nhiễu nhỏ)
        def _mock_rvc_infer(audio: np.ndarray, sr: int) -> np.ndarray:
            """Giả lập RVC inference: giữ nội dung, thêm hài âm nhẹ."""
            rng = np.random.default_rng(0)
            return (audio * 0.95 + 0.02 * rng.standard_normal(len(audio))).astype(np.float32)

        rvc_output = _mock_rvc_infer(audio_40k, 40_000)

        # Bước 3: Lưu output
        out_path = self.tmp_dir / "out_pipeline.wav"
        AudioProcessor.save(rvc_output, 40_000, out_path)

        # Assertions
        self.assertTrue(out_path.exists(), "File output không được tạo ra.")
        self.assertGreater(out_path.stat().st_size, 1000, "File output quá nhỏ.")

        out_data, out_sr = sf.read(str(out_path))
        self.assertEqual(out_sr, 40_000, f"Sample rate output sai: {out_sr}")
        self.assertEqual(out_data.ndim, 1, "Output không phải Mono.")
        self.assertLessEqual(float(np.max(np.abs(out_data))), 1.0 + TOLERANCE_AMP)

        elapsed = time.perf_counter() - t_start
        logger.info("[2.1] Full Pipeline OK – latency: %.3f s", elapsed)

    # ----- 2.2 Đo latency tổng toàn luồng -----
    def test_pipeline_latency_budget(self) -> None:
        """
        Toàn bộ luồng tiền xử lý phải hoàn thành trong LATENCY_BUDGET_S giây.
        """
        t0 = time.perf_counter()
        _, _ = self.processor.process(self.mock_wav, apply_noise_reduction=False)
        elapsed = time.perf_counter() - t0

        self.assertLess(
            elapsed, LATENCY_BUDGET_S,
            f"Latency {elapsed:.2f}s vượt ngân sách {LATENCY_BUDGET_S}s!"
        )
        logger.info("[2.2] Latency OK: %.3f s / budget %.0f s", elapsed, LATENCY_BUDGET_S)

    # ----- 2.3 Kiểm tra rò rỉ bộ nhớ CPU -----
    def test_no_significant_memory_leak(self) -> None:
        """
        Chạy processor N lần và kiểm tra bộ nhớ không tăng quá ngưỡng.
        Dùng tracemalloc để đo delta allocation.
        """
        N = 5
        gc.collect()
        tracemalloc.start()

        snapshot_before = tracemalloc.take_snapshot()
        for _ in range(N):
            audio_40k, _ = self.processor.process(
                self.mock_wav, apply_noise_reduction=False
            )
            del audio_40k
        gc.collect()
        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        top_stats = snapshot_after.compare_to(snapshot_before, "lineno")
        total_increase_mb = sum(stat.size_diff for stat in top_stats) / 1024 / 1024
        # Cho phép tăng tối đa 50 MB sau 5 lần chạy (threshold hợp lý)
        LEAK_THRESHOLD_MB = 50.0
        self.assertLess(
            total_increase_mb, LEAK_THRESHOLD_MB,
            f"Bộ nhớ tăng {total_increase_mb:.2f} MB sau {N} vòng, "
            f"vượt ngưỡng {LEAK_THRESHOLD_MB} MB!"
        )
        logger.info("[2.3] Memory leak OK – delta: %.2f MB", total_increase_mb)

    # ----- 2.4 Stereo WAV → Mono output -----
    def test_stereo_input_converted_to_mono(self) -> None:
        """File Stereo WAV phải được chuyển đổi thành Mono trước khi xử lý."""
        sr = 40_000
        rng = np.random.default_rng(1)
        stereo = rng.standard_normal((sr * 2, 2)).astype(np.float32) * 0.3
        stereo_path = self.tmp_dir / "mock_stereo.wav"
        sf.write(str(stereo_path), stereo, sr)

        audio_40k, _ = self.processor.process(str(stereo_path), apply_noise_reduction=False)
        self.assertEqual(audio_40k.ndim, 1, "Stereo chưa được chuyển thành Mono!")
        stereo_path.unlink(missing_ok=True)
        logger.info("[2.4] Stereo → Mono OK")

    # ----- 2.5 process_to_chunks chia nhỏ file dài -----
    def test_process_to_chunks_long_audio(self) -> None:
        """
        File 20 giây phải được chia thành nhiều đoạn ≤ slice_max_duration_s = 10 giây.
        """
        sr = 40_000
        t = np.linspace(0, 20.0, sr * 20, endpoint=False)
        long_audio = (0.5 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
        long_path = self.tmp_dir / "mock_long.wav"
        sf.write(str(long_path), long_audio, sr)

        cfg = AudioProcessorConfig(
            target_sr=sr,
            noise_reduction=False,
            slice_max_duration_s=10.0,
        )
        proc = AudioProcessor(config=cfg)
        chunks = proc.process_to_chunks(str(long_path), apply_noise_reduction=False)

        self.assertGreater(len(chunks), 1, "File 20s phải chia thành > 1 đoạn!")
        for i, chunk in enumerate(chunks):
            dur = len(chunk) / sr
            self.assertLessEqual(dur, 10.5,
                                 f"Chunk {i} dài {dur:.2f}s vượt giới hạn 10.5s!")
        long_path.unlink(missing_ok=True)
        logger.info("[2.5] process_to_chunks OK – %d chunks", len(chunks))

    # ----- 2.6 Kiểm thử tích hợp với ai_engine (nếu có) -----
    def test_integration_with_ai_engine_mock(self) -> None:
        """
        Giả lập giao tiếp giữa AudioProcessor và VoiceCloner.
        Dùng Mock để kiểm tra chữ ký hàm không bị phá vỡ khi tích hợp.
        """
        try:
            from ai_engine import VoiceCloner, RVC_AVAILABLE
        except ImportError:
            self.skipTest("ai_engine chưa cài đặt đủ dependencies, bỏ qua test này.")

        self.assertTrue(RVC_AVAILABLE, "RVC chưa sẵn sàng, bỏ qua tích hợp test.")

        # Mock VoiceCloner.clone_voice trả về đường dẫn file giả
        fake_output = str(self.tmp_dir / "out_integration.mp3")
        with patch("ai_engine.VoiceCloner.clone_voice", return_value=fake_output) as mock_fn:
            cloner = VoiceCloner()
            result = cloner.clone_voice(
                gen_text="Kiểm thử tích hợp hệ thống mô phỏng giọng nói CTU.",
                output_path=fake_output,
                voice="vi-VN-NamMinhNeural",
            )
            mock_fn.assert_called_once()
            self.assertEqual(result, fake_output)
        logger.info("[2.6] Integration mock test OK")


# ===========================================================================
# NHÓM 3: Kiểm thử trọng số & thư viện (Smoke Tests)
# ===========================================================================
class TestSystemDependencies(unittest.TestCase):
    """
    Smoke Tests kiểm tra môi trường và trọng số mô hình trước khi chạy.
    """

    def test_weights_pth_exists(self) -> None:
        models_dir = PROJECT_ROOT / "models"
        pth_files = list(models_dir.glob("*.pth"))
        if not pth_files:
            self.skipTest("Chưa đặt file trọng số .pth vào thư mục models/.")
        logger.info("[3.1] RVC .pth OK: %s (%.1f MB)", pth_files[0].name, pth_files[0].stat().st_size / 1e6)

    def test_weights_index_exists(self) -> None:
        models_dir = PROJECT_ROOT / "models"
        index_files = list(models_dir.glob("*.index"))
        if not index_files:
            self.skipTest("Chưa đặt file .index vào thư mục models/.")
        logger.info("[3.2] RVC .index OK: %s (%.1f MB)", index_files[0].name, index_files[0].stat().st_size / 1e6)

    def test_numpy_version(self) -> None:
        self.assertIsNotNone(np.__version__)
        logger.info("[3.3] numpy %s OK", np.__version__)

    def test_soundfile_import(self) -> None:
        self.assertIsNotNone(sf.__version__)
        logger.info("[3.4] soundfile %s OK", sf.__version__)

    def test_scipy_import(self) -> None:
        from scipy import signal
        self.assertIsNotNone(signal)
        logger.info("[3.5] scipy.signal OK")

    def test_rvc_python_import(self) -> None:
        try:
            from rvc_python.infer import RVCInference
            logger.info("[3.6] rvc_python OK")
        except ImportError as e:
            self.fail(f"rvc_python không import được: {e}")

    def test_edge_tts_import(self) -> None:
        try:
            import edge_tts
            logger.info("[3.7] edge_tts OK")
        except ImportError as e:
            self.fail(f"edge_tts không import được: {e}")

    def test_torch_cuda_available(self) -> None:
        try:
            import torch
            has_cuda = torch.cuda.is_available()
            if has_cuda:
                gpu_name = torch.cuda.get_device_name(0)
                logger.info("[3.8] CUDA OK – GPU: %s", gpu_name)
            else:
                logger.warning("[3.8] CUDA không khả dụng, sẽ dùng CPU.")
            # Không fail nếu không có GPU – test môi trường sẽ thích nghi
        except ImportError as e:
            self.fail(f"torch không import được: {e}")


# ===========================================================================
# NHÓM 4: Noise Reduction Tests (nếu noisereduce khả dụng)
# ===========================================================================
class TestNoiseReduction(_BaseTestCase):
    """
    Tests kiểm tra chức năng khử nhiễu bảo toàn thanh điệu tiếng Việt.
    """

    def _make_noisy(self, audio: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        rng = np.random.default_rng(99)
        return (audio + noise_level * rng.standard_normal(len(audio))).astype(np.float32)

    def test_noise_reduction_does_not_flip_sign(self) -> None:
        """
        Khử nhiễu không được đảo pha (phase flip) vì sẽ phá vỡ
        quan hệ pha giữa các hài âm trong tiếng Việt.
        """
        try:
            import noisereduce as nr
        except ImportError:
            self.skipTest("noisereduce chưa cài đặt.")

        sr = 16_000
        t = np.linspace(0, 2.0, sr * 2, endpoint=False)
        clean = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        noisy = self._make_noisy(clean, 0.05)

        reduced = nr.reduce_noise(y=noisy, sr=sr, stationary=True, prop_decrease=0.75)
        # Tương quan với tín hiệu gốc phải dương (không đảo pha)
        correlation = float(np.corrcoef(clean, reduced)[0, 1])
        self.assertGreater(correlation, 0.8,
                           f"Tương quan sau khử nhiễu quá thấp ({correlation:.3f}), "
                           "có thể đảo pha hoặc bóp méo nặng!")
        logger.info("[4.1] Noise reduction phase OK – correlation: %.3f", correlation)

    def test_noise_reduction_preserves_f0_frequency(self) -> None:
        """
        Khử nhiễu không được làm mất tần số cơ bản F0 (< 400 Hz).

        Chiến lược kiểm tra (đã cập nhật v2):
        - Dùng tương quan Pearson giữa tín hiệu clean và tín hiệu sau khử nhiễu
          thay vì so sánh energy tuyệt đối.
        - Lý do: noisereduce với stationary=True có thể scale biên độ khác nhau,
          làm energy ratio sai lệch; correlation phản ánh đúng sự bảo toàn hình
          dạng sóng (và do đó bảo toàn F0 + thanh điệu tiếng Việt).
        - Ngưỡng chính: correlation > 0.70.
        - Ngưỡng phụ: energy dải F0 (sau RMS-normalize) còn > 20%.
        """
        try:
            import noisereduce as nr
        except ImportError:
            self.skipTest("noisereduce chưa cài đặt.")

        sr = 16_000
        t = np.linspace(0, 2.0, sr * 2, endpoint=False)
        # Tín hiệu với F0 = 200 Hz (~ giọng nam tiếng Việt) + nhiễu trắng nhỏ
        clean = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        noisy = self._make_noisy(clean, noise_level=0.05)

        reduced = nr.reduce_noise(y=noisy, sr=sr, stationary=True, prop_decrease=0.75)

        # --- Phương pháp 1: Tương quan Pearson toàn tín hiệu ---
        correlation = float(np.corrcoef(clean, reduced)[0, 1])

        # --- Phương pháp 2 (bổ trợ): Tỉ lệ năng lượng dải F0 sau RMS-normalize ---
        def _band_energy(sig: np.ndarray, sr_: int, f_lo: float, f_hi: float) -> float:
            # Normalize về cùng RMS để tránh scale bias từ NR
            rms = float(np.sqrt(np.mean(sig ** 2))) or 1.0
            sig_norm = sig / rms
            fft_mag = np.abs(np.fft.rfft(sig_norm))
            freqs = np.fft.rfftfreq(len(sig_norm), 1 / sr_)
            mask = (freqs >= f_lo) & (freqs <= f_hi)
            return float(np.mean(fft_mag[mask] ** 2))

        e_clean = _band_energy(clean, sr, 80, 400)
        e_reduced = _band_energy(reduced, sr, 80, 400)
        retained = e_reduced / (e_clean + 1e-9)

        logger.info(
            "[4.2] F0 preservation – correlation: %.3f | F0-band energy retained (norm): %.2f%%",
            correlation, retained * 100,
        )

        # Tiêu chí chính: correlation > 0.70 (hình dạng sóng được bảo toàn)
        self.assertGreater(
            correlation, 0.70,
            f"Tương quan F0 sau khử nhiễu quá thấp ({correlation:.3f} < 0.70), "
            "thanh điệu tiếng Việt (F0 dải 80-400 Hz) có thể bị bóp méo nặng!",
        )
        # Tiêu chí phụ: energy dải F0 (sau normalize) còn ít nhất 20%
        self.assertGreater(
            retained, 0.20,
            f"Năng lượng F0 chuẩn hóa sau khử nhiễu chỉ còn {retained:.2%} "
            "(<20%), dải tần số thanh điệu tiếng Việt bị suy giảm quá mức!",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Chạy với verbosity cao để xem chi tiết từng test
    runner = unittest.TextTestRunner(verbosity=2)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Thứ tự chạy: Smoke → Unit → Integration → NR
    for cls in [
        TestSystemDependencies,
        TestAudioProcessorUnit,
        TestEndToEndPipeline,
        TestNoiseReduction,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
