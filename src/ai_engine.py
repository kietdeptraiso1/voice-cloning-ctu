import os
import asyncio
import edge_tts
import torch

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
    def __init__(self, default_voice: str = "vi-VN-NamMinhNeural"):
        self.default_voice = default_voice
        print(f"[AI Engine] Khởi tạo thành công. Giọng mặc định: {self.default_voice}")

    def _generate_edge_tts_sync(self, text: str, output_path: str, voice: str) -> bool:
        """Sinh âm thanh Edge-TTS chạy đồng bộ an toàn trên Windows"""
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
        
        abs_output = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(abs_output), exist_ok=True)
        temp_edge_wav = os.path.abspath("outputs/temp_edge.wav")

        try:
            # 1. Sinh âm thanh ngữ điệu từ Edge-TTS
            print(f"[AI Engine] Bước 1: Sinh âm thanh ngữ điệu từ Edge-TTS ({voice})...")
            success = self._generate_edge_tts_sync(gen_text, temp_edge_wav, voice)
            
            if not success or not os.path.exists(temp_edge_wav):
                print("[Lỗi AI Engine] Không thể tạo file tạm từ Edge-TTS.")
                return None

            # 2. Bọc giọng RVC nếu có file .pth
            if RVC_AVAILABLE and os.path.exists(rvc_pth):
                print(f"[AI Engine] Bước 2: Chuyển đổi âm sắc sang giọng RVC ({rvc_pth})...")
                rvc = RVCInference(device="cpu")
                rvc.load_model(rvc_pth)
                
                try:
                    if os.path.exists(rvc_index):
                        rvc.infer_file(temp_edge_wav, abs_output, rvc_index)
                    else:
                        rvc.infer_file(temp_edge_wav, abs_output)
                except TypeError:
                    rvc.infer_file(input_path=temp_edge_wav, output_path=abs_output)
                
                if os.path.exists(temp_edge_wav):
                    os.remove(temp_edge_wav)

                print(f"[AI Engine] Xuất file MP3 RVC thành công: {abs_output}")
                return abs_output

            # 3. Fallback: Xuất file Edge-TTS gốc nếu không dùng RVC
            print(f"[AI Engine] Xuất file MP3 từ Edge-TTS gốc: {abs_output}")
            self._generate_edge_tts_sync(gen_text, abs_output, voice)
            
            if os.path.exists(temp_edge_wav):
                os.remove(temp_edge_wav)

            return abs_output

        except Exception as e:
            print(f"[Lỗi AI Engine] Xử lý thất bại: {e}")
            return None


if __name__ == "__main__":
    target_text = "Thành viên 1 đã hoàn thiện kết nối Edge-TTS và RVC hoàn hảo."

    cloner = VoiceCloner()
    cloner.clone_voice(
        gen_text=target_text,
        output_path="outputs/test_rvc_vi.mp3",
        voice="vi-VN-NamMinhNeural",
        rvc_pth="weights/giong_mau.pth",
        rvc_index="weights/giong_mau.index"
    )