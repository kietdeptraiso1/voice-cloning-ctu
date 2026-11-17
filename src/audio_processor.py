"""
audio_processor.py
==================
Module tiền xử lý âm thanh (Audio Preprocessing) cho hệ thống Voice Cloning CTU.

Kiến trúc tham khảo:
- Applio (IAHispano): pipeline preprocess.py với Butterworth HPF + Slicer2.
- Vietnamese-RVC-Project: bảo toàn thanh điệu và phụ âm đặc thù tiếng Việt.

Thành viên 4 - Phan Ngọc Thế Hy (B2408909)
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy import signal as scipy_signal

# Optional heavy-weight deps
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    warnings.warn(
        "[AudioProcessor] 'librosa' chưa được cài đặt. Một số tính năng sẽ bị hạn chế.",
        ImportWarning,
        stacklevel=2,
    )

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    warnings.warn(
        "[AudioProcessor] 'noisereduce' chưa được cài đặt. Bước khử nhiễu sẽ bị bỏ qua.",
        ImportWarning,
        stacklevel=2,
    )

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ---------------------------------------------------------------------------
# Dataclass cấu hình
# ---------------------------------------------------------------------------
@dataclass
class AudioProcessorConfig:
    """
    Toàn bộ tham số tuỳ chỉnh cho pipeline tiền xử lý âm thanh.

    Attributes
    ----------
    target_sr : int
        Tần số lấy mẫu đầu ra (40 000 Hz ~ chuẩn RVC v2 Applio).
        Dùng 16 000 Hz nếu chỉ cần đầu vào cho HuBERT / RMVPE content encoder.
    hubert_sr : int
        Tần số lấy mẫu cho content encoder (HuBERT), cố định 16 000 Hz.
    hpf_cutoff_hz : float
        Tần số cắt bộ lọc thông cao (Butterworth, chuẩn Applio = 48 Hz).
        Loại bỏ tiếng ù tần số thấp và DC offset mà không ảnh hưởng đến
        tần số cơ bản giọng nói (khoảng 80–300 Hz cho giọng nam tiếng Việt).
    hpf_order : int
        Bậc bộ lọc Butterworth (5 ~ chuẩn Applio).
    noise_reduction : bool
        Bật / tắt khử nhiễu tổng thể.
    noise_prop_decrease : float
        Mức giảm nhiễu (0 – 1). Khuyến nghị ≤ 0.75 cho tiếng Việt để tránh
        bóp méo đường cong F0 thanh điệu và mất phụ âm gió (s, x, ch).
    noise_stationary : bool
        True = nhiễu tĩnh (tiếng quạt, hiss); False = nhiễu động (đám đông).
    top_db : float
        Ngưỡng (dB) phát hiện khoảng lặng khi cắt đầu/cuối file.
    pad_ms : int
        Khoảng đệm (ms) giữ lại mỗi phía sau khi cắt lặng để tránh cụt âm
        cuối nguyên âm tiếng Việt (đặc biệt thanh nặng).
    silence_threshold_db : float
        Ngưỡng (dB) nhận dạng file chỉ toàn tiếng im lặng (kiểm tra lỗi).
    peak_target_dbfs : float
        Mức đỉnh biên độ mục tiêu sau chuẩn hoá (dBFS). Mặc định −1 dBFS.
    rms_target_dbfs : float
        Mức RMS mục tiêu (dBFS). Nếu None, chỉ dùng peak normalisation.
    anti_clip_headroom_db : float
        Khoảng bảo vệ chống xén clipping (dBFS).
    slice_max_duration_s : float
        Độ dài tối đa (giây) mỗi đoạn khi chia nhỏ file dài.
    slice_hop_overlap_s : float
        Khoảng chồng lấp (giây) giữa các đoạn tránh cắt giữa từ.
    """

    target_sr: int = 40_000
    hubert_sr: int = 16_000
    hpf_cutoff_hz: float = 48.0
    hpf_order: int = 5
    noise_reduction: bool = True
    noise_prop_decrease: float = 0.75
    noise_stationary: bool = True
    top_db: float = 40.0
    pad_ms: int = 100
    silence_threshold_db: float = -60.0
    peak_target_dbfs: float = -1.0
    rms_target_dbfs: Optional[float] = None  # None = chỉ peak norm
    anti_clip_headroom_db: float = 1.0
    slice_max_duration_s: float = 10.0
    slice_hop_overlap_s: float = 0.1
    supported_formats: Tuple[str, ...] = field(
        default_factory=lambda: (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")
    )


# ---------------------------------------------------------------------------
# Helper: Silence-aware slicer (theo phong cách slicer2 của Applio)
# ---------------------------------------------------------------------------
class _SilenceSlicer:
    """
    Cắt nhỏ mảng audio dựa trên ngưỡng năng lượng RMS.
    Tham khảo kiến trúc slicer2.py của Applio.
    """

    def __init__(
        self,
        sr: int,
        threshold_db: float = -40.0,
        min_length_ms: int = 800,
        min_interval_ms: int = 400,
        hop_size_ms: int = 10,
        max_sil_kept_ms: int = 150,
    ) -> None:
        self.sr = sr
        self.threshold = 10 ** (threshold_db / 20.0)
        self.min_length = int(sr * min_length_ms / 1000)
        self.min_interval = int(sr * min_interval_ms / 1000)
        self.hop_size = int(sr * hop_size_ms / 1000)
        self.max_sil_kept = int(sr * max_sil_kept_ms / 1000)

    def slice(self, audio: np.ndarray) -> List[np.ndarray]:
        """
        Phân đoạn mảng audio thành các chunks, loại bỏ khoảng lặng dài.

        Parameters
        ----------
        audio : np.ndarray  shape (N,)  dtype float32

        Returns
        -------
        List[np.ndarray]  Danh sách các đoạn âm thanh đã cắt.
        """
        if len(audio) < self.min_length:
            return [audio]

        rms_frames: List[float] = []
        for i in range(0, len(audio), self.hop_size):
            chunk = audio[i : i + self.hop_size]
            rms_frames.append(float(np.sqrt(np.mean(chunk ** 2))))

        # Phát hiện vùng im lặng (RMS < threshold)
        is_sil = np.array(rms_frames) < self.threshold

        # Tìm biên cắt
        boundaries = []
        in_sil = False
        sil_start = 0
        for i, sil in enumerate(is_sil):
            if sil and not in_sil:
                sil_start = i
                in_sil = True
            elif not sil and in_sil:
                sil_len = (i - sil_start) * self.hop_size
                if sil_len >= self.min_interval:
                    cut_start = sil_start * self.hop_size + min(
                        self.max_sil_kept, sil_len // 2
                    )
                    cut_end = i * self.hop_size - min(
                        self.max_sil_kept, sil_len // 2
                    )
                    boundaries.append((cut_start, cut_end))
                in_sil = False

        if not boundaries:
            return [audio]

        chunks: List[np.ndarray] = []
        prev = 0
        for start, end in boundaries:
            seg = audio[prev:start]
            if len(seg) >= self.min_length:
                chunks.append(seg)
            prev = end
        tail = audio[prev:]
        if len(tail) >= self.min_length:
            chunks.append(tail)

        return chunks if chunks else [audio]


# ---------------------------------------------------------------------------
# Class chính: AudioProcessor
# ---------------------------------------------------------------------------
class AudioProcessor:
    """
    Pipeline tiền xử lý âm thanh Production-grade cho hệ thống Voice Cloning.

    Kiến trúc dựa trên Applio preprocessing và tối ưu hoá cho tiếng Việt:
    - Butterworth High-Pass Filter 48 Hz (loại tạp âm thấp, giữ F0 giọng nói)
    - Noise reduction bảo toàn thanh điệu (prop_decrease ≤ 0.75)
    - Silence trimming + padding 100 ms chống cụt âm tiếng Việt
    - Peak & RMS Normalisation với anti-clipping
    - Xuất cả hai phiên bản: 40 kHz (RVC/vocoder) + 16 kHz (HuBERT)

    Parameters
    ----------
    config : AudioProcessorConfig | None
        Cấu hình tùy chỉnh. Nếu None sẽ dùng giá trị mặc định.

    Example
    -------
    >>> processor = AudioProcessor()
    >>> audio40k, audio16k = processor.process("input.wav")
    >>> processor.save(audio40k, 40000, "output_40k.wav")
    """

    def __init__(self, config: Optional[AudioProcessorConfig] = None) -> None:
        self.cfg = config or AudioProcessorConfig()

        # Thiết kế bộ lọc Butterworth HPF một lần (chuẩn Applio)
        self._bh, self._ah = scipy_signal.butter(
            N=self.cfg.hpf_order,
            Wn=self.cfg.hpf_cutoff_hz,
            btype="high",
            fs=self.cfg.target_sr,
        )
        # Bộ lọc HPF cho 16 kHz
        self._bh16, self._ah16 = scipy_signal.butter(
            N=self.cfg.hpf_order,
            Wn=self.cfg.hpf_cutoff_hz,
            btype="high",
            fs=self.cfg.hubert_sr,
        )
        # Slicer dùng chung
        self._slicer = _SilenceSlicer(
            sr=self.cfg.target_sr,
            threshold_db=-self.cfg.top_db,
            min_length_ms=800,
            min_interval_ms=400,
            hop_size_ms=10,
            max_sil_kept_ms=self.cfg.pad_ms,
        )
        logger.info(
            "AudioProcessor khởi tạo xong. target_sr=%d | HPF=%.0f Hz | "
            "noise_reduce=%s | top_db=%.0f",
            self.cfg.target_sr,
            self.cfg.hpf_cutoff_hz,
            self.cfg.noise_reduction,
            self.cfg.top_db,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        input_path: str | Path,
        apply_noise_reduction: Optional[bool] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Chạy toàn bộ pipeline tiền xử lý cho một tệp âm thanh.

        Parameters
        ----------
        input_path : str | Path
            Đường dẫn file đầu vào (.wav / .mp3 / .flac …).
        apply_noise_reduction : bool | None
            Ghi đè cài đặt noise_reduction trong config (tiện cho test).

        Returns
        -------
        (audio_40k, audio_16k) : Tuple[np.ndarray, np.ndarray]
            audio_40k – float32 mono, 40 000 Hz, biên độ [−1, 1]
            audio_16k – float32 mono, 16 000 Hz, biên độ [−1, 1]

        Raises
        ------
        FileNotFoundError   Nếu file không tồn tại.
        ValueError          Nếu file rỗng, chỉ toàn tiếng im lặng,
                            hoặc định dạng không hỗ trợ.
        """
        t0 = time.perf_counter()
        input_path = Path(input_path)
        self._validate_path(input_path)

        # 1. Nạp & chuyển thành Mono float32 ở target_sr
        audio = self._load(input_path)

        # 2. High-Pass Filter (loại tạp âm thấp < 48 Hz)
        audio = self._apply_hpf(audio, bh=self._bh, ah=self._ah)

        # 3. Khử nhiễu (bảo toàn thanh điệu tiếng Việt)
        do_nr = self.cfg.noise_reduction if apply_noise_reduction is None else apply_noise_reduction
        if do_nr:
            audio = self._reduce_noise(audio, sr=self.cfg.target_sr)

        # 4. Cắt khoảng lặng đầu/cuối + thêm padding
        audio = self._trim_silence(audio)

        # 5. Kiểm tra file chỉ toàn tiếng lặng sau xử lý
        self._validate_audio_content(audio)

        # 6. Chuẩn hóa âm lượng (Peak ± RMS, chống clipping)
        audio = self._normalize(audio)

        # 7. Tạo phiên bản 16 kHz cho HuBERT encoder
        audio_16k = self._resample(audio, self.cfg.target_sr, self.cfg.hubert_sr)
        audio_16k = self._apply_hpf(audio_16k, bh=self._bh16, ah=self._ah16)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Xử lý xong '%s': %.2f giây | out_shape=%s | 16k_shape=%s",
            input_path.name, elapsed, audio.shape, audio_16k.shape,
        )
        return audio.astype(np.float32), audio_16k.astype(np.float32)

    def process_to_chunks(
        self,
        input_path: str | Path,
        apply_noise_reduction: Optional[bool] = None,
    ) -> List[np.ndarray]:
        """
        Tiền xử lý và chia nhỏ file dài thành các đoạn ≤ slice_max_duration_s.
        Mỗi đoạn ở định dạng 40 kHz, float32, Mono.

        Returns
        -------
        List[np.ndarray]  Danh sách các đoạn âm thanh.
        """
        audio, _ = self.process(input_path, apply_noise_reduction)
        chunks = self._slicer.slice(audio)

        max_len = int(self.cfg.target_sr * self.cfg.slice_max_duration_s)
        overlap = int(self.cfg.target_sr * self.cfg.slice_hop_overlap_s)
        result: List[np.ndarray] = []
        for chunk in chunks:
            if len(chunk) <= max_len:
                result.append(chunk)
            else:
                # Cắt thêm nếu chunk vẫn quá dài
                for start in range(0, len(chunk), max_len - overlap):
                    sub = chunk[start : start + max_len]
                    if len(sub) > self.cfg.target_sr * 0.3:  # > 300 ms
                        result.append(sub)
        logger.info("Chia thành %d đoạn từ '%s'", len(result), Path(input_path).name)
        return result

    @staticmethod
    def save(
        audio: np.ndarray,
        sr: int,
        output_path: str | Path,
        subtype: str = "PCM_16",
    ) -> None:
        """
        Lưu mảng numpy thành file WAV.

        Parameters
        ----------
        audio       : np.ndarray  float32 mono
        sr          : int         Sample rate
        output_path : str | Path  Đường dẫn lưu
        subtype     : str         Codec WAV (PCM_16 / PCM_24 / FLOAT)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio, sr, subtype=subtype)
        logger.info("Đã lưu: %s (%.2f s)", output_path.name, len(audio) / sr)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_path(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file âm thanh: {path}")
        if path.suffix.lower() not in self.cfg.supported_formats:
            raise ValueError(
                f"Định dạng '{path.suffix}' không được hỗ trợ. "
                f"Hỗ trợ: {self.cfg.supported_formats}"
            )
        if path.stat().st_size == 0:
            raise ValueError(f"File rỗng (0 bytes): {path}")

    def _load(self, path: Path) -> np.ndarray:
        """
        Nạp file âm thanh, chuyển về Mono float32 và resample về target_sr.
        """
        if path.suffix.lower() == ".wav":
            try:
                audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
                # Chuyển Stereo → Mono
                if audio.ndim == 2:
                    audio = audio.mean(axis=1)
                if sr != self.cfg.target_sr:
                    audio = self._resample(audio, sr, self.cfg.target_sr)
                return audio
            except Exception:
                pass  # Thử librosa nếu soundfile thất bại

        if not LIBROSA_AVAILABLE:
            raise ImportError(
                "Cần cài 'librosa' để nạp định dạng không phải WAV: "
                f"pip install librosa"
            )
        audio, _ = librosa.load(str(path), sr=self.cfg.target_sr, mono=True, dtype=np.float32)
        return audio

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return audio
        if LIBROSA_AVAILABLE:
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        # Fallback: scipy polyphase resampling
        gcd = np.gcd(orig_sr, target_sr)
        up, down = target_sr // gcd, orig_sr // gcd
        return scipy_signal.resample_poly(audio, up, down).astype(np.float32)

    def _apply_hpf(self, audio: np.ndarray, bh: np.ndarray, ah: np.ndarray) -> np.ndarray:
        """
        Áp dụng Butterworth High-Pass Filter (chuẩn Applio, 48 Hz).
        Loại bỏ DC offset và tạp âm ù thấp mà không ảnh hưởng đến
        tần số cơ bản giọng nói tiếng Việt (F0 ~ 80-300 Hz).
        """
        return scipy_signal.filtfilt(bh, ah, audio).astype(np.float32)

    def _reduce_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Giảm nhiễu tĩnh bằng spectral gating (noisereduce).

        **Tối ưu cho tiếng Việt:**
        - prop_decrease ≤ 0.75 → tránh làm phẳng đường cong F0 thanh điệu.
        - Không dùng aggressive reduction để bảo toàn phụ âm gió (s, x, ch).
        - Nếu noisereduce chưa cài, bỏ qua bước này và log cảnh báo.
        """
        if not NOISEREDUCE_AVAILABLE:
            logger.warning(
                "Bỏ qua khử nhiễu: thư viện 'noisereduce' chưa được cài đặt."
            )
            return audio
        try:
            reduced = nr.reduce_noise(
                y=audio,
                sr=sr,
                stationary=self.cfg.noise_stationary,
                prop_decrease=self.cfg.noise_prop_decrease,
            )
            return reduced.astype(np.float32)
        except Exception as exc:
            logger.warning("Khử nhiễu thất bại (tiếp tục không khử): %s", exc)
            return audio

    def _trim_silence(self, audio: np.ndarray) -> np.ndarray:
        """
        Cắt khoảng im lặng đầu/cuối và giữ lại pad_ms ở mỗi đầu.
        Dùng thuật toán năng lượng khung (frame energy), không dùng librosa.trim
        để tránh phụ thuộc bắt buộc.
        """
        pad_samples = int(self.cfg.target_sr * self.cfg.pad_ms / 1000)
        frame_len = int(self.cfg.target_sr * 0.01)   # 10 ms frames
        threshold_amp = 10 ** (-(self.cfg.top_db) / 20.0)

        # Tìm frame đầu tiên có năng lượng > ngưỡng
        start_frame = 0
        for i in range(0, len(audio) - frame_len, frame_len):
            if np.sqrt(np.mean(audio[i : i + frame_len] ** 2)) > threshold_amp:
                start_frame = i
                break

        # Tìm frame cuối cùng có năng lượng > ngưỡng
        end_frame = len(audio)
        for i in range(len(audio) - frame_len, start_frame, -frame_len):
            if np.sqrt(np.mean(audio[i : i + frame_len] ** 2)) > threshold_amp:
                end_frame = i + frame_len
                break

        start = max(0, start_frame - pad_samples)
        end = min(len(audio), end_frame + pad_samples)
        trimmed = audio[start:end]

        if len(trimmed) == 0:
            raise ValueError(
                "File âm thanh chỉ chứa toàn tiếng im lặng sau khi trim. "
                "Hãy kiểm tra file đầu vào."
            )
        return trimmed

    def _validate_audio_content(self, audio: np.ndarray) -> None:
        """Kiểm tra file không phải toàn tiếng lặng hoặc quá ngắn."""
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms == 0:
            raise ValueError("Audio chỉ chứa tiếng im lặng (RMS = 0).")
        rms_db = 20 * np.log10(rms + 1e-9)
        if rms_db < self.cfg.silence_threshold_db:
            raise ValueError(
                f"Audio quá yếu (RMS = {rms_db:.1f} dBFS < "
                f"{self.cfg.silence_threshold_db} dBFS). "
                "Có thể file chỉ chứa nhiễu hoặc bị lỗi."
            )

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """
        Chuẩn hóa âm lượng:
        1. RMS normalisation nếu rms_target_dbfs được đặt.
        2. Peak normalisation về peak_target_dbfs.
        3. Anti-clipping: giảm thêm nếu đỉnh vượt ngưỡng.
        """
        # Bước 1: RMS (tuỳ chọn)
        if self.cfg.rms_target_dbfs is not None:
            rms = float(np.sqrt(np.mean(audio ** 2)) + 1e-9)
            target_rms = 10 ** (self.cfg.rms_target_dbfs / 20.0)
            audio = audio * (target_rms / rms)

        # Bước 2: Peak
        peak = float(np.max(np.abs(audio)) + 1e-9)
        target_peak = 10 ** (self.cfg.peak_target_dbfs / 20.0)
        audio = audio * (target_peak / peak)

        # Bước 3: Chống clipping
        headroom = 10 ** (-self.cfg.anti_clip_headroom_db / 20.0)
        final_peak = float(np.max(np.abs(audio)))
        if final_peak > headroom:
            audio = audio * (headroom / final_peak)

        return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
def _cli() -> None:
    """
    Chạy tiền xử lý thử nghiệm từ dòng lệnh:
        python src/audio_processor.py <input_file> [output_dir]
    """
    import sys

    if len(sys.argv) < 2:
        print("Cách dùng: python audio_processor.py <input.wav> [output_dir]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs/preprocessed")
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = AudioProcessor()
    audio_40k, audio_16k = processor.process(input_file)
    stem = Path(input_file).stem

    out_40k = output_dir / f"{stem}_40k.wav"
    out_16k = output_dir / f"{stem}_16k.wav"
    processor.save(audio_40k, 40_000, out_40k)
    processor.save(audio_16k, 16_000, out_16k)

    print(f"✅ Xuất thành công:")
    print(f"   40kHz (RVC): {out_40k}")
    print(f"   16kHz (HuBERT): {out_16k}")


if __name__ == "__main__":
    _cli()
