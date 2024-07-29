# 滑动窗口裁剪图片  允许重叠
from PIL import Image, ImageFont, ImageDraw, ImageOps
import random 

class ImageBlock:
    def __init__(self, block, row_index, col_index):
        self.block = block
        self.row_index = row_index
        self.col_index = col_index

class pil_img:
    def __init__(self):
        # 输入图片的高度宽度
        self.width = None
        self.height = None
        # 缩放之后，图片的高度、宽度
        self.scaled_height = None
        self.scaled_width = None
        # 滑窗大小
        self.crop_width = None
        self.crop_height = None
        # 滑窗移动步长：宽度方向 高度方向
        self.stride_width = None
        self.stride_height = None
        # 对原图下方补零长度 右方补零长度
        self.padding_bottom = 0
        self.padding_right = 0
        self.padding_top = 0
        self.padding_left = 0
        # 滑窗的移动步数宽度/高度方向 + 1、滑窗截图的行/列数 + 1
        self.cols_num = None
        self.rows_num = None

    def image_resize_fixed(self, image, target_size=(3000, 2400), padding=None, plot=None):
            """  固定纵横比， 然后缩放， 如果padding就在短边方向上padding"""
            
            width, height = image.size
            base_ratio = width / height

            self.width = width
            self.height = height

            target_width, target_height = target_size
            target_aspect_ratio = target_width / target_height

            if base_ratio > target_aspect_ratio:
                # 情况1：将原始图像的宽度等比缩放到 target_width，然后在高度方向上补零到 target_height
                scaled_width = target_width
                scaled_height = int(height * (target_width / width))

                resized_image = image.resize((scaled_width, scaled_height), Image.ANTIALIAS)

                if padding is not None:
                    canvas = Image.new("RGB", (target_width, target_height), (0, 0, 0))
                    left = 0
                    top = 0
                    canvas.paste(resized_image, (left, top))
                else:
                    canvas = Image.new("RGB", (scaled_width, scaled_height), (0, 0, 0))
                    canvas.paste(resized_image)
            else:
                scaled_height = target_height
                scaled_width = int(width * (target_height / height))
                resized_image = image.resize((scaled_width, scaled_height), Image.ANTIALIAS)

                if padding is not None:
                    canvas = Image.new("RGB", (target_width, target_height), (0, 0, 0))
                    left = 0
                    top = 0
                    canvas.paste(resized_image, (left, top))
                else:
                    canvas = Image.new("RGB", (scaled_width, scaled_height), (0, 0, 0))
                    canvas.paste(resized_image)

            if plot is not None:
                canvas.show()

            return canvas

    def image_resize(self, image, target_size=(3000, 2400), plot=None, stamp=None):
        """ resize到target-size, 尽可能保证纵横比 """
        
        width, height = image.size

        if stamp:
            print("(input_width, input_height):", (width, height))

        self.width = width
        self.height = height

        target_width, target_height = target_size
        target_aspect_ratio = target_width / target_height

        if width / height > target_aspect_ratio:
            scaled_width = target_width
            scaled_height = int(height * (target_width / width))
        else:
            scaled_height = target_height
            scaled_width = int(width * (target_height / height))

        self.scaled_height = scaled_height
        self.scaled_width = scaled_width

        if stamp:
            print("(scaled_width, scaled_height):", (scaled_width, scaled_height))

        resized_image = image.resize((scaled_width, scaled_height), Image.ANTIALIAS)

        if plot is not None:
            resized_image.show()

        return resized_image

    def paddedImageCrop_base(self, image, crop_size=(640, 640), overlap=(0, 0), draw_cropRectangle=False):
            width, height = image.size
            self.crop_width, self.crop_height = crop_size

            overlap_width, overlap_height = overlap
            stride_height = self.crop_height - overlap_height
            stride_width = self.crop_width - overlap_width
            self.stride_height = stride_height
            self.stride_width = stride_width

            height_length = height - self.crop_height
            if height_length % stride_height != 0:
                num_rows = int(height_length // stride_height) + 1
                self.rows_num = num_rows + 1
            else:
                num_rows = int(height_length / stride_height)
                self.rows_num = num_rows + 1
            width_length = width - self.crop_width
            if width_length % stride_width != 0:
                num_cols = int(width_length // stride_width) + 1
                self.cols_num = num_cols + 1
            else:
                num_cols = int(width_length // stride_width)
                self.cols_num = num_cols + 1
                        
            padding_height = num_rows * stride_height + self.crop_height - height
            padding_width = num_cols * stride_width + self.crop_width - width
            padding_height = int(padding_height) + (int(padding_height) % 2)
            padding_width = int(padding_width) + (int(padding_width) % 2)
            
            self.padding_right = padding_width
            self.padding_bottom = padding_height

            # border=(left, top, right, down)
            # 右边和下面补0
            padded_image = ImageOps.expand(image, border=(
                0, 0, int(padding_width), int(padding_height)), fill=(0, 0, 0))

            # 新建pillow画布 1------  全显示的分割框，     ---取出每个crop
            draw_img = padded_image.copy()
            draw_a = ImageDraw.Draw(draw_img)
            font = ImageFont.truetype("arial.ttf", 34)
            title = "crop-rectangle --a"
            title_font = ImageFont.truetype("arial.ttf", 40)
            title_width, title_height = draw_a.textsize(title, font=title_font)
            title_position = (int((width - title_width) / 2), 20)
            draw_a.text(title_position, title, font=title_font, fill=(255, 255, 255))

            # 新建pillow画布 2------  每个crop依次写入画布2，     ---依次写入取出的每个crop
            canvas_height = self.crop_height + stride_height * num_rows
            canvas_width = self.crop_width + stride_width * num_cols
            canvas = Image.new("RGB", (canvas_width, canvas_height), (0, 0, 0))
            draw_b = ImageDraw.Draw(canvas)
            title = "crops write in onebyone  --b"
            title_font = ImageFont.truetype("arial.ttf", 40)
            title_width, title_height = draw_b.textsize(title, font=title_font)
            title_position = (int((width - title_width) / 2), 20)
            draw_b.text(title_position, title, font=title_font, fill=(255, 255, 255))

            # # 还是记住相对于在padded_image的位置，然后在这上面画预测框
            # 切割图像--所有的分割的预选框，有重叠overlap，
            image_blocks = []
            for r in range(num_rows + 1):
                for c in range(num_cols + 1):
                    # 计算当前块的起始和结束位置
                    start_row = r * stride_height
                    end_row = start_row + self.crop_height
                    start_col = c * stride_width
                    end_col = start_col + self.crop_width
                    # 提取当前块
                    block = padded_image.crop((start_col, start_row, end_col, end_row))
                    
                    # may no use
                    if block.size[0] == self.crop_width and block.size[1] == self.crop_height:
                        image_block = ImageBlock(block, r, c)
                        image_blocks.append(image_block)
                    else:
                        print(block.size)
                        continue
                    
                    if draw_cropRectangle:
                        # 生成随机颜色
                        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                        wid_a = random.randint(2, 21)
                        
                        # canvas-a
                        # 绘制边界框
                        draw_a.rectangle([(start_col, start_row), (end_col, end_row)], outline=color, width=wid_a)
                        # 标记编号
                        block_number = f"Block-bounding {r + 1}{c + 1}"
                        draw_a.text((start_col + 10, start_row + 10), block_number, font=font, fill=color)
                        
                        # canvas-b
                        canvas.paste(block, (start_col, start_row))
                        # # # 绘制边界框 ---这里要用到end_row,end_col
                        draw_b.rectangle([(start_col, start_row), (end_col, end_row)], outline=color, width=wid_a)
                        # 标记编号
                        block_number = f"Block {r + 1}{c + 1}"
                        draw_b.text((start_col + 10, start_row + 10), block_number, font=font, fill=color)

                        # # #crop一块一块往画布上写入，后写入的会覆盖之前的 
                        # canvas.show()
            
            if draw_cropRectangle:
                # ## 显示全部的crop-rectangle
                draw_img.show()

            # # ## 显示全部的crop-rectangle
            # draw_img.show()

            # # # # 有覆盖的框, crop依次写入之后
            # canvas.show()
            
            return image_blocks, padded_image

    def paddedImageCrop(self, image, crop_size=(640,640), overlap=(0, 0)):
            width, height = image.size
            self.crop_width, self.crop_height = crop_size

            overlap_width, overlap_height = overlap
            stride_height = self.crop_height - overlap_height
            stride_width = self.crop_width - overlap_width
            self.stride_height = stride_height
            self.stride_width = stride_width

            height_length = height - self.crop_height
            if height_length % stride_height != 0:
                num_rows = int(height_length // stride_height) + 1
                self.rows_num = num_rows + 1
            else:
                num_rows = int(height_length / stride_height)
                self.rows_num = num_rows + 1
            width_length = width - self.crop_width
            if width_length % stride_width != 0:
                num_cols = int(width_length // stride_width) + 1
                self.cols_num = num_cols + 1
            else:
                num_cols = int(width_length // stride_width)
                self.cols_num = num_cols + 1

            padding_height = num_rows * stride_height + self.crop_height - height
            padding_width = num_cols * stride_width + self.crop_width - width
            padding_height = int(padding_height) + (int(padding_height) % 2)
            padding_width = int(padding_width) + (int(padding_width) % 2)
            
            self.padding_right = padding_width
            self.padding_bottom = padding_height

            # border=(left, top, right, down)
            # 右边和下面补0
            padded_image = ImageOps.expand(image, border=(
                0, 0, int(padding_width), int(padding_height)), fill=(0, 0, 0))

            image_blocks = []
            for r in range(num_rows + 1):
                for c in range(num_cols + 1):
                    start_row = r * stride_height
                    end_row = start_row + self.crop_height
                    start_col = c * stride_width
                    end_col = start_col + self.crop_width

                    block = padded_image.crop((start_col, start_row, end_col, end_row))
                    if block.size[0] == self.crop_width and block.size[1] == self.crop_height:
                        image_block = ImageBlock(block, r, c)
                        image_blocks.append(image_block)
                    else:
                        print(block.size)
                        continue                   

            return image_blocks, padded_image
       
       
if __name__ == '__main__':
    img_path = r"D:\chyCodespace\repo\origin\CV_PROJECTION\U2NET_SS\raw_data\rgbpngs\train\image(17).png"
    pil_img = pil_img()
    image = Image.open(img_path)
    pil_img.paddedImageCrop_base(image, crop_size=(640,640), overlap=(0, 0), draw_cropRectangle=True)
    # img_blocks = pil_img.paddedImageCrop(image, crop_size=(640,640), overlap=(200, 200))