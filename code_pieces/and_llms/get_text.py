from faster_whisper import WhisperModel

root_dir = r"D:\ddesktop\26-0518-具身智能ppt\sucai"

video_audio = f"{root_dir}/audio.wav"

model = WhisperModel(
    "large-v3",
    device="cpu",          # 没有显卡就改成 "cpu"
    compute_type="int8"  # CPU 可改成 "int8"
)

segments, info = model.transcribe(
    video_audio,
    language="zh",
    task="transcribe",
    beam_size=5,
    vad_filter=True,
    initial_prompt="以下是普通话和少量英文混合的字幕内容，请输出简体中文，保留必要英文术语，添加合适标点。"
)

with open(f"{root_dir}/subtitle.txt", "w", encoding="utf-8") as f_txt, \
     open(f"{root_dir}/subtitle.srt", "w", encoding="utf-8") as f_srt:

    for i, seg in enumerate(segments, start=1):
        text = seg.text.strip()
        print(f"[{seg.start:.2f} -> {seg.end:.2f}] {text}")

        f_txt.write(text + "\n")

        def fmt(t):
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = int(t % 60)
            ms = int((t - int(t)) * 1000)
            return f"{h:02}:{m:02}:{s:02},{ms:03}"

        f_srt.write(f"{i}\n")
        f_srt.write(f"{fmt(seg.start)} --> {fmt(seg.end)}\n")
        f_srt.write(text + "\n\n")