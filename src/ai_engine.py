import os
import asyncio
import edge_tts
import torch
import soundfile as sf
import numpy as np

# Fix lỗi WeightsUnpickler error trên PyTorch phiên bản mới
try:
    import fairseq.data.dictionary
    torch.serialization.add_safe_globals([fairseq.data.dictionary.Dictionary])
except Exception:
    pass

# Kiểm tra thư viện rvc_python
RVC_AVAILABLE = False
try:
    from rvc_python.infer import RVCInference
    RVC_AVAILABLE = True
except ImportError:
    RVC_AVAILABLE = False


class VoiceCloner:
    def __init__(self, default_voice: str = "vi-VN-NamMinhNeural", device: str = None):
        self.default_voice = default_voice
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.rvc_instance = None
        self.loaded_model_path = None
        print(f"[AI Engine] Khởi tạo VoiceCloner. Thiết bị: {self.device} | Giọng TTS nền: {self.default_voice}")

    def _get_rvc(self, rvc_pth: str, rvc_index: str = None):
        """Khởi tạo và lưu cache phiên làm việc RVC để tối ưu tốc độ suy luận."""
        if not RVC_AVAILABLE or not os.path.exists(rvc_pth):
            return None

        index_to_use = rvc_index if (rvc_index and os.path.exists(rvc_index)) else ""
        cache_key = f"{rvc_pth}_{index_to_use}"

        if self.rvc_instance is not None and self.loaded_model_path == cache_key:
            return self.rvc_instance

        print(f"[AI Engine] Nạp trọng số RVC: {rvc_pth} (Index: {index_to_use or 'None'})...")
        rvc = RVCInference(device=self.device)
        rvc.load_model(rvc_pth, version="v2", index_path=index_to_use)
        
        # Áp dụng thông số tinh chỉnh tối ưu hóa cho tiếng Việt
        rvc.set_params(
            f0method="rmvpe",       # Thuật toán RMVPE bám sát thanh điệu tiếng Việt tốt nhất
            index_rate=0.70,        # Cân bằng nhận diện âm sắc đặc trưng và chống rè kim loại
            filter_radius=3,        # Lọc trung vị làm mượt bước nhảy cao độ
            protect=0.35,           # Bảo vệ nguyên vẹn phụ âm vô thanh và nhịp thở
            rms_mix_rate=0.80       # Cân bằng cường độ năng lượng âm thanh
        )
        self.rvc_instance = rvc
        self.loaded_model_path = cache_key
        return self.rvc_instance

    def _generate_edge_tts_sync(self, text: str, output_path: str, voice: str) -> bool:
        """Sinh âm thanh Edge-TTS chạy đồng bộ an toàn trên Windows."""
        async def _async_generate():
            selected_voice = voice if voice else self.default_voice
            communicate = edge_tts.Communicate(text=text, voice=selected_voice)
            await communicate.save(output_path)

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_async_generate())
            loop.close()
            return True
        except Exception as e:
            print(f"[Lỗi Edge-TTS] Thất bại khi tạo âm thanh: {e}")
            return False

    def clone_voice(
        self, 
        gen_text: str, 
        output_path: str = "outputs/cloned_voice.mp3", 
        voice: str = "vi-VN-NamMinhNeural",
        rvc_pth: str = "weights/giong_mau.pth",
        rvc_index: str = "weights/giong_mau.index"
    ) -> str:
        """
        Quy trình mô phỏng giọng nói hoàn chỉnh:
        1. Sinh âm thanh nền chuẩn ngữ điệu tiếng Việt từ Edge-TTS.
        2. Chuyển đổi âm sắc (Voice Conversion) sang chất giọng mục tiêu thông qua RVC.
        3. Xuất file kết quả theo định dạng mong muốn (.wav hoặc .mp3).
        """
        abs_output = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(abs_output), exist_ok=True)
        
        temp_edge = os.path.abspath("outputs/temp_edge_tts.mp3")
        temp_rvc_wav = os.path.abspath("outputs/temp_rvc_out.wav")

        try:
            # 1. Sinh âm thanh ngữ điệu từ Edge-TTS
            print(f"[AI Engine] Bước 1: Sinh âm thanh ngữ điệu từ Edge-TTS ({voice})...")
            success = self._generate_edge_tts_sync(gen_text, temp_edge, voice)
            
            if not success or not os.path.exists(temp_edge):
                print("[Lỗi AI Engine] Không thể tạo file tạm từ Edge-TTS.")
                return None

            # 2. Bọc giọng RVC nếu có file .pth
            rvc = self._get_rvc(rvc_pth, rvc_index)
            if rvc is not None:
                print(f"[AI Engine] Bước 2: Chuyển đổi âm sắc bằng RVC ({rvc_pth})...")
                rvc.infer_file(temp_edge, temp_rvc_wav)
                
                # Chuyển đổi sang định dạng đích nếu yêu cầu file .mp3
                if abs_output.lower().endswith(".mp3"):
                    import av
                    inp = av.open(temp_rvc_wav, "r")
                    out = av.open(abs_output, "w")
                    ostream = out.add_stream("mp3")
                    for frame in inp.decode(audio=0):
                        for packet in ostream.encode(frame):
                            out.mux(packet)
                    for packet in ostream.encode(None):
                        out.mux(packet)
                    out.close()
                    inp.close()
                else:
                    # Nếu yêu cầu .wav thì sao chép hoặc đổi tên
                    if os.path.exists(abs_output):
                        os.remove(abs_output)
                    os.rename(temp_rvc_wav, abs_output)

                # Dọn dẹp tệp tạm
                for tmp in [temp_edge, temp_rvc_wav]:
                    if os.path.exists(tmp):
                        os.remove(tmp)

                print(f"[AI Engine] Xuất tệp thành phẩm RVC thành công: {abs_output}")
                return abs_output

            # 3. Fallback: Xuất file Edge-TTS gốc nếu không dùng RVC
            print(f"[AI Engine] Fallback: Xuất tệp âm thanh trực tiếp từ Edge-TTS: {abs_output}")
            self._generate_edge_tts_sync(gen_text, abs_output, voice)
            if os.path.exists(temp_edge):
                os.remove(temp_edge)

            return abs_output

        except Exception as e:
            print(f"[Lỗi AI Engine] Xử lý thất bại: {e}")
            return None


if __name__ == "__main__":
    test_script = "Ít người được giúp đỡ, vì ai thì cũng đều bận cơm áo gạo tiền. Họ chỉ thả cho bạn một like về bài viết."

    cloner = VoiceCloner()
    output_file = cloner.clone_voice(
        gen_text=test_script,
        output_path="outputs/test_rvc_vi.mp3",
        voice="vi-VN-NamMinhNeural",
        rvc_pth="weights/giong_mau.pth",
        rvc_index="weights/giong_mau.index"
    )
    print(f"[Kết quả] File hoàn chỉnh đã tạo tại: {output_file}")
