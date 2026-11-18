import os
import re
import soundfile as sf
import torch
import torchaudio

# Vá lỗi libtorchcodec trên Windows bằng cách chuyển hướng torchaudio sang soundfile
def _patched_torchaudio_load(filepath, *args, **kwargs):
    data, sr = sf.read(str(filepath), dtype="float32", always_2d=True)
    tensor = torch.from_numpy(data.T)
    return tensor, sr

def _patched_torchaudio_save(filepath, src, sample_rate, *args, **kwargs):
    data = src.detach().cpu().numpy()
    if data.ndim == 2:
        data = data.T
    sf.write(str(filepath), data, sample_rate)

torchaudio.load = _patched_torchaudio_load
torchaudio.save = _patched_torchaudio_save

# Kiểm tra thư viện Zero-Shot TTS (F5-TTS)
F5_AVAILABLE = False
try:
    from f5_tts.api import F5TTS
    from huggingface_hub import hf_hub_download
    F5_AVAILABLE = True
except ImportError:
    F5_AVAILABLE = False

try:
    from .audio_processor import AudioProcessor
except ImportError:
    from audio_processor import AudioProcessor


class VoiceCloner:
    def __init__(self, device: str = None):
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.processor = AudioProcessor()
        self.f5_instance = None
        print(f"[AI Engine] Khởi tạo VoiceCloner (Zero-Shot F5-TTS) | Thiết bị: {self.device}")

    def _get_f5(self):
        """Nạp mô hình Zero-Shot F5-TTS tiếng Việt (ViVoice)."""
        if not F5_AVAILABLE:
            return None
        if self.f5_instance is not None:
            return self.f5_instance

        print("[AI Engine] Đang nạp mô hình Zero-Shot F5-TTS Vietnamese...")
        repo_id = "hynt/F5-TTS-Vietnamese-ViVoice"
        ckpt_file = hf_hub_download(repo_id=repo_id, filename="model_last.pt")
        vocab_file = hf_hub_download(repo_id=repo_id, filename="config.json")
        self.f5_instance = F5TTS(
            model="F5TTS_Base",
            ckpt_file=ckpt_file,
            vocab_file=vocab_file,
            device=self.device,
        )
        return self.f5_instance

    def preprocess_and_save_voice(self, raw_audio_path: str, voice_name: str, save_dir: str = "saved_voices") -> str:
        """Tiền xử lý file thu âm giọng thật và lưu vào kho giọng mẫu (tối đa 6 giây)."""
        os.makedirs(save_dir, exist_ok=True)
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", voice_name.strip()) or "custom_voice"
        out_path = os.path.abspath(os.path.join(save_dir, f"{safe_name}.wav"))

        audio_40k, _ = self.processor.process(raw_audio_path)
        max_samples = self.processor.cfg.target_sr * 6
        if len(audio_40k) > max_samples:
            audio_40k = audio_40k[:max_samples]

        self.processor.save(audio_40k, self.processor.cfg.target_sr, out_path)
        return out_path

    def clone_from_recorded_voice(
        self,
        gen_text: str,
        ref_audio_path: str,
        ref_text: str = "",
        output_path: str = "outputs/zeroshot_out.wav",
    ) -> str:
        """Đọc văn bản bằng chính giọng thu âm mẫu qua Zero-Shot F5-TTS."""
        f5 = self._get_f5()
        if f5 is None:
            raise RuntimeError("Thư viện f5-tts chưa được cài đặt hoặc chưa sẵn sàng.")

        abs_output = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(abs_output), exist_ok=True)
        temp_f5_wav = os.path.abspath("outputs/temp_f5_raw.wav")

        try:
            f5.infer(
                ref_file=ref_audio_path,
                ref_text=ref_text.strip(),
                gen_text=gen_text.strip(),
                file_wave=temp_f5_wav,
                nfe_step=16,
                remove_silence=False,
            )
            audio_40k, _ = self.processor.process(temp_f5_wav)
            self.processor.save(audio_40k, self.processor.cfg.target_sr, abs_output)
            return abs_output
        finally:
            if os.path.exists(temp_f5_wav):
                os.remove(temp_f5_wav)