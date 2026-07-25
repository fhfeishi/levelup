import os
import tqdm
import numpy as np
import imageio
from moviepy.editor import VideoFileClip
from PIL import Image, ImageDraw, ImageFont

def add_text_watermark(frame, text, position=(10, 10), font_path=None, font_size=24, color=(255, 255, 255, 128)):
    """
    在图片帧上添加文字水印。
    :param frame: PIL Image对象
    :param text: 水印文字
    :param position: 水印位置 (x, y)
    :param font_path: 字体文件路径
    :param font_size: 字体大小
    :param color: 字体颜色 (R, G, B, A)
    :return: 添加水印后的PIL Image对象
    """
    draw = ImageDraw.Draw(frame)
    if font_path and os.path.exists(font_path):
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return frame

def video_to_gif_with_watermark(input_video_path, output_gif_path, text, position=(750, 10),
                                font_path=None, font_size=24, color=(255, 255, 255, 128),
                                resize_width=None, resize_height=None, fps=10):
    """
    使用moviepy从视频中提取帧，使用Pillow给每一帧添加水印，然后使用imageio保存为高质量GIF。
    如果已安装imageio的libimagequant支持，则会得到更好的色彩量化效果。
    """
    # 加载视频
    clip = VideoFileClip(input_video_path)
    total_frames = int(clip.duration * fps)

    # 根据需要调整视频大小
    if resize_width or resize_height:
        clip = clip.resize(width=resize_width, height=resize_height)

    frames_array = []
    # 使用tqdm显示进度条
    for i, frame in enumerate(tqdm.tqdm(clip.iter_frames(fps=fps, dtype="uint8"), total=total_frames, desc="正在转换")):
        # frame是一个numpy数组 (h, w, 3)
        img = Image.fromarray(frame)
        # img.save(f"a/{i}.jpg")

        # 添加水印
        img = add_text_watermark(img, text, position, font_path, font_size, color)
        # img.save(f"a/{i}_sy.jpg")
        # 转回numpy数组以便imageio处理
        arr = np.array(img.convert("RGB"))
        frames_array.append(arr)

    # 使用imageio将帧列表保存为GIF
    # 如已安装libimagequant支持，可指定 quantizer='libimagequant' 获得更好结果
    imageio.mimsave(
        output_gif_path,
        frames_array,
        format='GIF',
        fps=fps,
        quantizer='libimagequant'  # 若未安装libimagequant支持，可去掉此参数
    )

    print(f"GIF已保存到 {output_gif_path}")


if __name__ == "__main__":
    # 示例
    input_video = "videos/e.mp4"
    output_gif = "videos/e_output3.gif"
    watermark_text = "WHU-MVR"
    watermark_position = (540, 320)
    font_file = "simhei.ttf"
    font_size = 24
    text_color = (255, 255, 0, 250)
    resize_width = 640
    # resize_height = 250
    fps = 8

    video_to_gif_with_watermark(
        input_video_path=input_video,
        output_gif_path=output_gif,
        text=watermark_text,
        position=watermark_position,
        font_path=font_file,
        font_size=font_size,
        color=text_color,
        resize_width=resize_width,
        fps=fps
    )
