import re
import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr
from ai_engine import F5_AVAILABLE, VoiceCloner

SAVED_VOICES_DIR = PROJECT_ROOT / "saved_voices"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SAVED_VOICES_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

cloner = VoiceCloner()


def sanitize_voice_name(voice_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", voice_name.strip()) or "custom_voice"


def list_saved_voices():
    files = sorted(SAVED_VOICES_DIR.glob("*.wav"))
    return [f.name for f in files]


def refresh_saved_voices():
    choices = list_saved_voices()
    return gr.update(choices=choices, value=choices[0] if choices else None)


def _execute_save_voice(raw_audio: str, safe_name: str, ref_text: str, overwritten: bool = False):
    saved_path = cloner.preprocess_and_save_voice(raw_audio, safe_name, str(SAVED_VOICES_DIR))
    txt_path = Path(saved_path).with_suffix(".txt")
    txt_path.write_text(ref_text.strip() if ref_text else "", encoding="utf-8")
    choices = list_saved_voices()
    saved_filename = Path(saved_path).name
    action_text = "ghi đè và lưu" if overwritten else "lọc nhiễu & lưu"
    return (
        gr.update(choices=choices, value=saved_filename),
        f"✅ Đã {action_text} giọng mẫu: **{saved_filename}** (tối ưu 6 giây).",
        gr.update(visible=False),
        "",
    )


def save_custom_voice(raw_audio, voice_name, ref_text):
    if not raw_audio:
        return (
            gr.update(),
            "⚠️ Vui lòng thu âm hoặc tải lên file giọng nói trước.",
            gr.update(visible=False),
            "",
        )

    if voice_name and voice_name.strip():
        safe_name = sanitize_voice_name(voice_name)
    else:
        idx = len(list_saved_voices()) + 1
        while (SAVED_VOICES_DIR / f"giong_mau_{idx}.wav").exists():
            idx += 1
        safe_name = f"giong_mau_{idx}"

    target_wav = SAVED_VOICES_DIR / f"{safe_name}.wav"
    if target_wav.exists():
        warning_text = (
            f"### ⚠️ Tên giọng mẫu đã tồn tại\n"
            f"File **`{safe_name}.wav`** đã có sẵn trong kho giọng mẫu.\n\n"
            f"Bạn có muốn **ghi đè** lên file cũ không, hay muốn giữ lại file cũ và **đặt tên khác**?"
        )
        return (
            gr.update(),
            "⏳ Đang chờ xác nhận ghi đè...",
            gr.update(visible=True),
            warning_text,
        )

    try:
        return _execute_save_voice(raw_audio, safe_name, ref_text, overwritten=False)
    except Exception as e:
        return gr.update(), f"❌ Lỗi xử lý giọng mẫu: {e}", gr.update(visible=False), ""


def confirm_overwrite_voice(raw_audio, voice_name, ref_text):
    if not raw_audio:
        return (
            gr.update(),
            "⚠️ Vui lòng thu âm hoặc tải lên file giọng nói trước.",
            gr.update(visible=False),
            "",
        )
    safe_name = sanitize_voice_name(voice_name) if (voice_name and voice_name.strip()) else "custom_voice"
    try:
        return _execute_save_voice(raw_audio, safe_name, ref_text, overwritten=True)
    except Exception as e:
        return gr.update(), f"❌ Lỗi khi ghi đè giọng mẫu: {e}", gr.update(visible=False), ""


def cancel_overwrite_voice():
    return (
        gr.update(),
        "ℹ️ Đã hủy ghi đè. Vui lòng nhập tên khác vào ô **Tên giọng mẫu** bên trái rồi bấm lưu lại.",
        gr.update(visible=False),
        "",
    )


def generate_voice(text, custom_voice_file):
    if not text or not text.strip():
        return None, "⚠️ Hãy nhập nội dung văn bản trước khi bấm Tạo giọng nói."
    if not F5_AVAILABLE:
        return None, "⚠️ Chưa cài đặt thư viện `f5-tts` trong môi trường ảo."
    if not custom_voice_file:
        return None, "⚠️ Chưa chọn giọng mẫu cá nhân. Hãy thu âm và lưu giọng ở Tab 2 trước."

    ref_wav = SAVED_VOICES_DIR / custom_voice_file
    if not ref_wav.is_file():
        return None, f"⚠️ Không tìm thấy file giọng mẫu `{custom_voice_file}`."

    ref_txt_file = ref_wav.with_suffix(".txt")
    ref_text = ref_txt_file.read_text(encoding="utf-8").strip() if ref_txt_file.is_file() else ""
    output_path = OUTPUTS_DIR / f"cloned_{uuid4().hex[:8]}.wav"

    try:
        res = cloner.clone_from_recorded_voice(
            gen_text=text.strip(),
            ref_audio_path=str(ref_wav),
            ref_text=ref_text,
            output_path=str(output_path),
        )
        return res, f"✅ Đã tạo giọng nói theo mẫu **{custom_voice_file}**."
    except Exception as e:
        return None, f"❌ Lỗi Zero-Shot TTS: {e}"


with gr.Blocks(title="Mô phỏng giọng nói | CTU") as demo:
    gr.Markdown("# 🎙️ Hệ thống Mô phỏng & Nhân bản Giọng nói Tiếng Việt")
    gr.Markdown(
        f"- **Zero-Shot TTS (F5-TTS):** {'✅ Sẵn sàng' if F5_AVAILABLE else '⚠️ Chưa cài f5-tts'}"
    )

    with gr.Tabs():
        with gr.Tab("📝 Văn bản thành giọng AI"):
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(
                        label="Nội dung cần đọc",
                        placeholder="Nhập đoạn văn tiếng Việt cần chuyển thành giọng nói...",
                        lines=5,
                    )
                    custom_voice_dropdown = gr.Dropdown(
                        choices=list_saved_voices(),
                        value=list_saved_voices()[0] if list_saved_voices() else None,
                        label="Chọn giọng thu âm cá nhân đã lưu",
                    )
                    generate_button = gr.Button("Tạo giọng nói", variant="primary")

                with gr.Column():
                    audio_output = gr.Audio(label="Âm thanh kết quả", interactive=False)
                    status_output = gr.Markdown("Trạng thái: Sẵn sàng.")

            generate_button.click(
                fn=generate_voice,
                inputs=[text_input, custom_voice_dropdown],
                outputs=[audio_output, status_output],
            )

        with gr.Tab("🎤 Thu âm & Lưu giọng mẫu"):
            with gr.Row():
                with gr.Column():
                    recorded_audio = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        label="Thu âm trực tiếp (5–6 giây) hoặc tải file giọng mẫu",
                    )
                    voice_name_input = gr.Textbox(
                        label="Tên giọng mẫu (để lưu dùng cho Zero-Shot)",
                        placeholder="Ví dụ: giong_kiet",
                    )
                    ref_text_input = gr.Textbox(
                        label="Nội dung câu bạn vừa nói (giúp tăng tốc & đọc chuẩn dấu)",
                        placeholder="Gõ chính xác câu bạn đã đọc trong bản thu âm...",
                        lines=2,
                    )
                    save_voice_button = gr.Button("💾 Tiền xử lý & Lưu làm giọng mẫu", variant="primary")

                with gr.Column():
                    tab2_status = gr.Markdown("Trạng thái: Sẵn sàng nhận giọng thu âm.")

                    with gr.Group(visible=False) as overwrite_panel:
                        overwrite_msg = gr.Markdown("")
                        with gr.Row():
                            confirm_overwrite_btn = gr.Button("✅ Có, ghi đè file cũ", variant="primary")
                            cancel_overwrite_btn = gr.Button("✏️ Không, đặt tên khác", variant="secondary")

            save_voice_button.click(
                fn=save_custom_voice,
                inputs=[recorded_audio, voice_name_input, ref_text_input],
                outputs=[custom_voice_dropdown, tab2_status, overwrite_panel, overwrite_msg],
            )
            confirm_overwrite_btn.click(
                fn=confirm_overwrite_voice,
                inputs=[recorded_audio, voice_name_input, ref_text_input],
                outputs=[custom_voice_dropdown, tab2_status, overwrite_panel, overwrite_msg],
            )
            cancel_overwrite_btn.click(
                fn=cancel_overwrite_voice,
                inputs=None,
                outputs=[custom_voice_dropdown, tab2_status, overwrite_panel, overwrite_msg],
            )

    # Tự động cập nhật danh sách giọng mẫu mỗi khi tải lại trang (F5)
    demo.load(
        fn=refresh_saved_voices,
        inputs=None,
        outputs=[custom_voice_dropdown],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)