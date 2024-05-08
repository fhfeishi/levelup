# ----------------------------------------------------#
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
# ----------------------------------------------------#
import time
from PIL import Image, ImageFont, ImageDraw, ImageOps
import cv2
import numpy as np
import random
from deeplab import DeeplabV3

from tqdm import tqdm

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
        self.crop_size = None
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

    def image_resize2(self, image, target_size=(3000, 2400), padding=None, plot=None):
            print(type(image))
            # 获取原始图像的宽度和高度
            width, height = image.size
            base_ratio = width / height

            # # ***shared params*** # # width
            self.width = width
            # # ***shared params*** # # height
            self.height = height

            # 计算目标纵横比
            target_width, target_height = target_size
            target_aspect_ratio = target_width / target_height

            # 判断采用的情况
            if base_ratio > target_aspect_ratio:
                # 情况1：将原始图像的宽度等比缩放到 target_width，然后在高度方向上补零到 target_height

                # 计算等比缩放后的宽度和高度
                scaled_width = target_width
                scaled_height = int(height * (target_width / width))
                # print("-----------------------")
                # print("scaled_width:", scaled_width)
                # print("scaled_height:", scaled_height)
                # print("-----------------------")

                # 进行等比缩放
                resized_image = image.resize((scaled_width, scaled_height), Image.ANTIALIAS)

                if padding is not None:
                    # 创建目标尺寸的空白画布
                    canvas = Image.new("RGB", (target_width, target_height), (0, 0, 0))
                    # 计算补零的位置
                    left = 0
                    top = 0
                    # 将等比缩放后的图像粘贴到画布上
                    canvas.paste(resized_image, (left, top))
                else:
                    # 创建目标尺寸的空白画布
                    canvas = Image.new("RGB", (scaled_width, scaled_height), (0, 0, 0))
                    canvas.paste(resized_image)
            else:
                # 计算等比缩放后的宽度和高度
                scaled_height = target_height
                scaled_width = int(width * (target_height / height))
                # print("-----------------------")
                # print("scaled_width:", scaled_width)
                # print("scaled_height:", scaled_height)
                # print("-----------------------")
                # 进行等比缩放
                resized_image = image.resize((scaled_width, scaled_height), Image.ANTIALIAS)

                if padding is not None:
                    # 创建目标尺寸的空白画布
                    canvas = Image.new("RGB", (target_width, target_height), (0, 0, 0))
                    # 计算补零的位置
                    left = 0
                    top = 0
                    # 将等比缩放后的图像粘贴到画布上
                    canvas.paste(resized_image, (left, top))
                else:
                    # 创建目标尺寸的空白画布
                    canvas = Image.new("RGB", (scaled_width, scaled_height), (0, 0, 0))
                    canvas.paste(resized_image)

            if plot is not None:
                # show
                canvas.show()
            # 返回最终结果
            return canvas

    def image_resize(self, image, target_size=(3000, 2400), plot=None, stamp=None):

        # 获取原始图像的宽度和高度
        width, height = image.size

        if stamp:
            print("(input_width, input_height):", (width, height))

        # #***shared params***# # width
        self.width = width
        # #***shared params***# # height
        self.height = height

        # 计算目标纵横比
        target_width, target_height = target_size
        target_aspect_ratio = target_width / target_height

        # 计算等比缩放后的宽度和高度
        if width / height > target_aspect_ratio:
            scaled_width = target_width
            scaled_height = int(height * (target_width / width))
        else:
            scaled_height = target_height
            scaled_width = int(width * (target_height / height))

        # #***shared params***# # scaled_height
        self.scaled_height = scaled_height
        # #***shared params***# # scaled_width
        self.scaled_width = scaled_width

        if stamp:
            print("(scaled_width, scaled_height):", (scaled_width, scaled_height))

        # 进行等比缩放
        resized_image = image.resize((scaled_width, scaled_height), Image.ANTIALIAS)

        if plot is not None:
            # show
            resized_image.show()
        # 返回最终结果
        return resized_image

    def padded_image_crop(self, image, crop_size=640, overlap=(0, 0), stamp=False, plot=False, all_box_show=False,
                               coverred_box_show=False, show_onebyone=False, padded_img_show=None):

            width, height = image.size
            # print("input_image_size:", image.size)

            # # ***shared params*** # # crop_size
            self.crop_size = crop_size

            overlap_width, overlap_height = overlap
            stride_height = crop_size - overlap_height
            stride_width = crop_size - overlap_width
            # # ***shared params*** # # stride_height
            self.stride_height = stride_height
            # # ***shared params*** # # stride_width
            self.stride_width = stride_width

            # 设计padding，让其再滑动一步  最后的blocks的行和列总数还得在这基础上+1
            # 方法a
            height_length = height - crop_size
            if height_length % stride_height != 0:
                # print("1")
                num_rows = int(height_length // stride_height) + 1
                # # ***shared params*** # # rows_num
                self.rows_num = num_rows + 1
            else:
                # print("11")
                num_rows = int(height_length / stride_height)
                # # ***shared params*** # # rows_num
                self.rows_num = num_rows + 1
            width_length = width - crop_size
            if width_length % stride_width != 0:
                # print("2")
                num_cols = int(width_length // stride_width) + 1
                # # ***shared params*** # # cols_num
                self.cols_num = num_cols + 1
            else:
                # print("22")
                num_cols = int(width_length // stride_width)
                # # ***shared params*** # # cols_num
                self.cols_num = num_cols + 1
            if stamp:
                print("-----------------------")
                # print("num_rows:", num_rows)
                # print("num_cols:", num_cols)
                # # concat根据框框的位置---i行j列---拼接，可能没用吧
                print("rows_num:", self.rows_num)
                print("cols_num:", self.cols_num)
                print("-----------------------")
            #
            # 方法b  ---好像是稳定补零的一种方式
            # num_rows = (height - overlap_height) // stride_height
            # num_cols = (width - overlap_width) // stride_width
            # g_cols_num = num_cols + 1
            # g_rows_num = num_rows + 1
            # print("-----------------------")
            # print("num_rows:", num_rows)
            # print("num_cols:", num_cols)
            # print("g_rows_num:", g_rows_num)
            # print("g_cols_num:", g_cols_num)
            # print("-----------------------")

            padding_height = num_rows * stride_height + crop_size - height
            padding_width = num_cols * stride_width + crop_size - width

            # 就是说如果这个padding_height padding_width不是偶数，就+1，然后去掉这个多余的
            # 通常来说并不会有奇数
            padding_height = int(padding_height) + (int(padding_height) % 2)
            padding_width = int(padding_width) + (int(padding_width) % 2)
            if stamp:
                print("-----------------------")
                print("padding_height:", padding_height)
                print("padding_width:", padding_width)
                print("-----------------------")
            # # ***shared params*** # # padding_right
            self.padding_right = padding_width
            # # ***shared params*** # # padding_bottom
            self.padding_bottom = padding_height

            #  bord=(left, top, right, down)
            # a 上下左右补0
            # padded_image = ImageOps.expand(image, border=(
            #     int(padding_width * 0.5), int(padding_height * 0.5),
            #     int(padding_width * 0.5), int(padding_height * 0.5)),
            #                                fill=(0, 0, 0))
            # b 右边和下面补0
            padded_image = ImageOps.expand(image, border=(
                0, 0, int(padding_width), int(padding_height)), fill=(0, 0, 0))
            if padded_img_show is not None:
                padded_image.show()  # ok
            # print("padded_image.shape", padded_image.size)  # ok
            # padded_image_width, padded_image_height = padded_image.size

            # print("------------")
            # row_number = (padded_image_height - crop_size) / stride_height
            # col_number = (padded_image_width - crop_size) / stride_width
            # print("stride_width:", stride_width)
            # print("stride_height:", stride_height)
            # print("row_number看看是否是整数:", row_number)
            # print("col_number看看是否是整数:", col_number)
            # print("------------")

            # 新建pillow画布 1------  全显示的分割框
            draw_img = padded_image.copy()
            draw_a = ImageDraw.Draw(draw_img)
            font = ImageFont.truetype("arial.ttf", 34)
            # title
            title = "overlap box a"
            title_font = ImageFont.truetype("arial.ttf", 40)
            title_width, title_height = draw_a.textsize(title, font=title_font)
            title_position = (int((width - title_width) / 2), 20)
            draw_a.text(title_position, title, font=title_font, fill=(255, 255, 255))

            # 新建pillow画布 2------  有覆盖的
            canvas_height = crop_size + stride_height * num_rows
            canvas_width = crop_size + stride_width * num_cols
            canvas = Image.new("RGB", (canvas_width, canvas_height), (0, 0, 0))
            draw_b = ImageDraw.Draw(canvas)
            # title
            title = "overlap box b"
            title_font = ImageFont.truetype("arial.ttf", 40)
            title_width, title_height = draw_b.textsize(title, font=title_font)
            title_position = (int((width - title_width) / 2), 20)
            draw_b.text(title_position, title, font=title_font, fill=(255, 255, 255))

            # print("---------------------")
            # print("canvas_height:", canvas_height)
            # print("canvas_width:", canvas_width)
            # print("---------------------")

            # # 还是记住相对于在padded_image的位置，然后在这上面画预测框
            # 切割图像--所有的分割的预选框，有重叠overlap，
            image_blocks = []
            for r in range(num_rows + 1):
                for c in range(num_cols + 1):

                    # 计算当前块的起始和结束位置
                    start_row = r * stride_height
                    end_row = start_row + crop_size
                    start_col = c * stride_width
                    end_col = start_col + crop_size

                    # 提取当前块
                    block = padded_image.crop((start_col, start_row, end_col, end_row))
                    if block.size[0] == crop_size and block.size[1] == crop_size:
                        # 创建 ImageBlock 对象并存储
                        image_block = ImageBlock(block, r, c)
                        image_blocks.append(image_block)
                    else:
                        print(block.size)
                        continue

                    # 生成随机颜色
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                    wid_a = random.randint(5, 21)
                    # 绘制边界框
                    draw_a.rectangle([(start_col, start_row), (end_col, end_row)], outline=color, width=wid_a)

                    # 标记编号
                    block_number = f"Block-bounding {r + 1}{c + 1}"
                    draw_a.text((start_col + 10, start_row + 10), block_number, font=font, fill=color)

                    # 画块的边框和编号
                    if plot:
                        # 遍历每个图像块并在画布上进行拼接、编号和边界框标记
                        for i, block in enumerate(image_blocks):
                            # 生成随机颜色
                            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                            # 计算当前图像块在画布上的位置
                            row = i // (num_cols + 1)
                            col = i % (num_cols + 1)
                            start_row = row * stride_height
                            end_row = start_row + crop_size
                            start_col = col * stride_width
                            end_col = start_col + crop_size

                            # block
                            block = image_block.block

                            # 在画布上拼接图像块
                            boxuu = (start_col, start_row, end_col, end_row)
                            canvas.paste(block, (start_col, start_col))
                            # canvas.paste(block, boxuu)

                            # # # 绘制边界框 ---这里要用到end_row,end_col
                            # draw_b.rectangle([(start_col, start_row), (end_col, end_row)], outline=color, width=6)

                            # 标记编号
                            block_number = f"Block {row + 1}{col + 1}"
                            draw_b.text((start_col + 10, start_row + 10), block_number, font=font, fill=color)

                            if show_onebyone:
                                # 显示拼接后的图像
                                # 新建画布，然后一块一块往画布上写入，后写入的会覆盖之前的
                                canvas.show()

                    # # crop
                    # if crop is not None:
                    #
                    #     return None
                    #
                    # # cat_back
                    # if cat_back is not None:
                    #
                    #     return None
            if stamp:
                print("box_num", len(image_blocks))

            # # 显示拼接后的图像

            if plot and all_box_show:
                # 全部的overlap框
                # print("a")
                draw_img.show()
            elif plot and coverred_box_show:
                # print("b")
                # 有覆盖的框
                canvas.show()
            else:
                pass

            return image_blocks, padded_image

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # -------------------------------------------------------------------------#
    #   如果想要修改对应种类的颜色，到__init__函数里修改self.colors即可
    # -------------------------------------------------------------------------#
    deeplab = DeeplabV3()
    # ----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'               表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #   'export_onnx'       表示将模型导出为onnx，需要pytorch1.7.1以上。
    # ----------------------------------------------------------------------------------------------------------#
    # mode = "dir_predict"
    mode = "predict"
    # -------------------------------------------------------------------------#
    #   count               指定了是否进行目标的像素点计数（即面积）与比例计算
    #   name_classes        区分的种类，和json_to_dataset里面的一样，用于打印种类和数量
    #
    #   count、name_classes仅在mode='predict'时有效
    # -------------------------------------------------------------------------#
    count = False
    name_classes    = ["background", "jyz", "baodian"]
    # -------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    #
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    # -------------------------------------------------------------------------#
    dir_origin_path = "img/test428a"
    dir_save_path = "img/result428a"
    # ------------------------------------------------------------------------------------------#
    # 使用pillow 实现滑动窗口
    pillow_crop_silde = True
    # 将有重叠的窗口 去重叠然后拼接， True就是简单的直接拼接   --不好看
    cat_pillow_image_blocks = False
    # 去掉padding黑边
    show_on_resized_image = True
    # image_block大小、重叠
    crop_size = 640
    target_size = (3000, 2250)
    overlap = (200, 200)


    # ----------------------------------------------------------------------------------------------------------#
    #   video_path          用于指定视频的路径，当video_path=0时表示检测摄像头
    #                       想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path     表示视频保存的路径，当video_save_path=""时表示不保存
    #                       想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps           用于保存的视频的fps
    #
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    # ----------------------------------------------------------------------------------------------------------#
    video_path = 0
    video_save_path = ""
    video_fps = 25.0
    # ----------------------------------------------------------------------------------------------------------#
    #   test_interval       用于指定测量fps的时候，图片检测的次数。理论上test_interval越大，fps越准确。
    #   fps_image_path      用于指定测试的fps图片
    #
    #   test_interval和fps_image_path仅在mode='fps'有效
    # ----------------------------------------------------------------------------------------------------------#
    test_interval = 100
    fps_image_path = "img/street.jpg"
    # -------------------------------------------------------------------------#
    #   simplify            使用Simplify onnx
    #   onnx_save_path      指定了onnx的保存路径
    # -------------------------------------------------------------------------#
    simplify = True
    onnx_save_path = "model_data/models.onnx"

    if mode == "predict":

        # 实例化
        pil_img = pil_img()

        '''
        predict.py有几个注意点
        1、该代码无法直接进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
        具体流程可以参考get_miou_prediction.py，在get_miou_prediction.py即实现了遍历。
        2、如果想要保存，利用r_image.save("img.jpg")即可保存。
        3、如果想要原图和分割图不混合，可以把blend参数设置成False。
        4、如果想根据mask获取对应的区域，可以参考detect_image函数中，利用预测结果绘图的部分，判断每一个像素点的种类，然后根据种类获取对应的部分。
        seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))
        for c in range(self.num_classes):
            seg_img[:, :, 0] += ((pr == c)*( self.colors[c][0] )).astype('uint8')
            seg_img[:, :, 1] += ((pr == c)*( self.colors[c][1] )).astype('uint8')
            seg_img[:, :, 2] += ((pr == c)*( self.colors[c][2] )).astype('uint8')
        '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                crops, padded_image = pil_img.padded_image_crop(image, crop_size, overlap, plot=False)
                width = pil_img.width
                height = pil_img.height
                stride_width = pil_img.stride_width
                stride_height = pil_img.stride_height
                rows_num = pil_img.rows_num
                cols_num = pil_img.cols_num
                padding_bottom = pil_img.padding_bottom
                padding_right = pil_img.padding_right

                # 以padded_image作为画布
                result_image = padded_image.copy()
                draw = ImageDraw.Draw(result_image)

                # 遍历 crops
                for i, crop in enumerate(crops):
                    # 取出存入时候block的行标列标
                    # 利用了ImageBlock 这个class
                    row_index = crop.row_index
                    col_index = crop.col_index
                    crop = crop.block  # crop - pillow-Image
                    r_image = deeplab.detect_image(crop, count=count, name_classes=name_classes)

                    # 拼接就可以  因为这里是分割，对每个像素点进行分类
                    # 计算该crop在padded_image中的位置
                    x_start = col_index * stride_width
                    y_start = row_index * stride_height

                    # 粘贴r_image到result_image上对应的位置
                    result_image.paste(r_image, (x_start, y_start))

                # 去掉padding 黑边
                if show_on_resized_image:
                    # 去掉原画布（padded_img的right bottom补零黑边）
                    #                          .crop((left, top, right, bottom))
                    paddedimg_width, paddedimg_height = padded_image.size
                    if paddedimg_width is not None:
                        result_image = result_image.crop((0, 0, paddedimg_width-padding_right, paddedimg_height-padding_bottom))
                    result_image.show()
                else:
                    # 不去掉
                    result_image.show()




    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        # 实例化
        pil_img = pil_img()

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)

                crops, padded_image = pil_img.padded_image_crop(image, crop_size, overlap, plot=False)
                width = pil_img.width
                height = pil_img.height
                stride_width = pil_img.stride_width
                stride_height = pil_img.stride_height
                rows_num = pil_img.rows_num
                cols_num = pil_img.cols_num
                padding_bottom = pil_img.padding_bottom
                padding_right = pil_img.padding_right

                # 以padded_image作为画布
                result_image = padded_image.copy()
                draw = ImageDraw.Draw(result_image)

                # 遍历 crops
                for i, crop in enumerate(crops):
                    # 取出存入时候block的行标列标
                    # 利用了ImageBlock 这个class
                    row_index = crop.row_index
                    col_index = crop.col_index
                    crop = crop.block  # crop - pillow-Image
                    r_image = deeplab.detect_image(crop, count=count, name_classes=name_classes)

                    # 拼接就可以  因为这里是分割，对每个像素点进行分类
                    # 计算该crop在padded_image中的位置
                    x_start = col_index * stride_width
                    y_start = row_index * stride_height

                    # 粘贴r_image到result_image上对应的位置
                    result_image.paste(r_image, (x_start, y_start))

                # 去掉padding 黑边
                if show_on_resized_image:
                    # 去掉原画布（padded_img的right bottom补零黑边）
                    #                          .crop((left, top, right, bottom))
                    paddedimg_width, paddedimg_height = padded_image.size
                    if paddedimg_width is not None:
                        result_image = result_image.crop(
                            (0, 0, paddedimg_width - padding_right, paddedimg_height - padding_bottom))
                    # result_image.show()
                else:
                    # 不去掉
                    # result_image.show()
                    pass

                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                result_image.save(os.path.join(dir_save_path, img_name))



    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        while (True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame = np.array(deeplab.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break
        print("Video Detection Done!")
        capture.release()
        if video_save_path != "":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = deeplab.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

    elif mode == "export_onnx":
        deeplab.convert_to_onnx(simplify, onnx_save_path)

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
