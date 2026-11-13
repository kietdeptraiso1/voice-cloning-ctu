import gradio as gr
from ai_engine import RVC_AVAILABLE, VoiceCloner
import time
from pathlib import Path
from uuid import uuid4

cloner = VoiceCloner()
RVC_MODEL_PATH = Path("weights/giong_mau.pth")
RVC_READY = RVC_AVAILABLE and RVC_MODEL_PATH.is_file()

def generate_voice(text, voice):
    if not text or not text.strip():
        return None, "⚠️ Hãy nhập nội dung trước khi bấm Tạo giọng nói."

    for attempt in range(2):
        output_path = Path("outputs") / f"cloned_voice_{uuid4().hex}.mp3"
        audio_path = None

        try:
            audio_path = cloner.clone_voice(
                gen_text=text.strip(),
                output_path=str(output_path),
                voice=voice,
                rvc_pth="weights/giong_mau.pth",
                rvc_index="weights/giong_mau.index",
            )
        except Exception as error:
            print(f"[UI] Lần thử {attempt + 1} thất bại: {error}")

        if audio_path:
            audio_file = Path(audio_path)
            if audio_file.is_file() and audio_file.stat().st_size > 0:
                if RVC_READY:
                    return str(audio_file), "✅ Đã chuyển giọng bằng mô hình RVC."
                return str(audio_file), (
                    "✅ Đã tạo âm thanh Edge-TTS. "
                    "Hiện chưa có model RVC nên đây chưa phải giọng clone của nhóm."
                )

        if attempt == 0:
            time.sleep(3)

    return None, (
        "⚠️ Edge-TTS chưa trả âm thanh sau hai lần thử. "
        "Hãy thử lại sau; nếu tiếp tục lỗi, báo thành viên phụ trách AI Engine."
    )


with gr.Blocks(title="Mô phỏng giọng nói | CTU") as demo:
    gr.Markdown(
        """
        # 🎙️ Mô phỏng giọng nói
        Chọn một trong hai cách để tạo âm thanh.
        """
    )

    if RVC_READY:
        gr.Markdown("✅ **Chế độ RVC sẵn sàng — có thể chuyển sang giọng mẫu.**")
    else:
        gr.Markdown(
            "ℹ️ **Hiện hệ thống đang ở chế độ Edge-TTS.** "
            "Cần model và thư viện RVC để chuyển sang giọng mẫu."
        )

    with gr.Tabs():
        with gr.Tab("📝 Văn bản thành giọng AI"):
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(
                        label="Nội dung cần đọc",
                        placeholder="Nhập nội dung cần chuyển thành giọng nói...",
                        lines=6,
                        max_lines=12,
                    )

                    voice_input = gr.Dropdown(
                        choices=["vi-VN-NamMinhNeural"],
                        value="vi-VN-NamMinhNeural",
                        label="Giọng đọc",
                    )

                    generate_button = gr.Button(
                        "Tạo giọng nói",
                        variant="primary",
                    )

                with gr.Column():
                    audio_output = gr.Audio(
                        label="Âm thanh kết quả",
                        interactive=False,
                    )

                    status_output = gr.Markdown("Trạng thái: Sẵn sàng.")

            generate_button.click(
                fn=generate_voice,
                inputs=[text_input, voice_input],
                outputs=[audio_output, status_output],
            )

        with gr.Tab("🎤 Giọng thật thành giọng AI"):
            gr.Markdown(
                "Tải file giọng thật lên hoặc thu âm trực tiếp. "
                "Phần chuyển giọng sẽ được nối với AI Engine sau."
            )

            with gr.Row():
                with gr.Column():
                    recorded_audio = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        label="Tải lên hoặc thu âm giọng thật",
                    )

                    convert_audio_button = gr.Button(
                        "Chuyển sang giọng AI",
                        interactive=False,
                    )

                with gr.Column():
                    converted_audio_output = gr.Audio(
                        label="Âm thanh sau chuyển đổi",
                        interactive=False,
                    )

                    gr.Markdown(
                        "Trạng thái: Đang chờ phần xử lý âm thanh từ AI Engine."
                    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)