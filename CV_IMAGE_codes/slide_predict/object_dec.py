import time
import os
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageOps
import io
import matplotlib.pyplot as plt
import random
import xml.etree.cElementTree as ET
from xml.dom.minidom import Document
from tqdm import tqdm



# 一般设置 input_size: [640, 640]
# 所以在Dataset里面可以设置一个  随机起点(x, y) -- ([0, img-height - 640], [0, img-width - 640])
#   single_class:   threshold:   in-x_bbox > 60% -x_bbox
#   multi_class:  ?

# 一般来说，检测模型输出的结果就直接将predict-box话在上面了, 输出是带predict-box的输入图片
# 但是我只需要这个这个predict-bbox的 x1 y1 x2 y2, 然后加上 crop_row_idx * stride_h crop_col_idx*stride_w
# 所以肯定是要更改这个 object_dec  返回值的逻辑的。

# 同时因为是检测，而且是有重叠的滑动窗口检测， predict_box的合并也是需要的

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

    # 锁定纵横比的缩放策略，不足的话右边、下面补零
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

    # jpg_image_path.jpg   annotations_path.json/xml/txt
    # clip_image
    # 先待定吧
    # def pil_clip_img(self, oriname, image_folder, label_xml_folder, image_crop_folder, lablel_xml_folder_new):
    def pil_clip_img(self, image, oriname, label_xml_folder, image_crop_folder, lablel_xml_folder_new):
        # ------------------------------------------------------------#
        #   这里就是resized padding之后的paddedimg （pillow.Image）
        # 读入原始图像  input_jpg -> resized_jpg -> padded_jpg（因为在右边、下边补零，对坐标没有影响）
        # 保存原图的大小
        # 可以resize也可以不resize，看情况而定  # 这里整理一下，直至与测试的策略相近，最好是完全一样
        # ------------------------------------------------------------#
        # # from_name = os.path.join(image_path, oriname + '.png')
        # from_name = os.path.join(image_folder, oriname + '.jpg')
        # # #          dir_path + basename + .jpg ---- basename: img=img_xml
        # # # print(from_name)
        # img = Image.open(from_name)
        # # print("img")
        # # print(img)
        # h_ori, w_ori = img.size
        # img = img.resize((2048, 1536))
        # h, w = img.size
        h_ori = self.height
        w_ori = self.width
        h = self.scaled_height
        w = self.scaled_width

        # ------------------------------------------------------------#
        # 输入.xml文件
        # 创建存放坐标四个值和类别的列表
        # ------------------------------------------------------------#
        xml_name = os.path.join(label_xml_folder, oriname + '.xml')
        xml_ori = ET.parse(xml_name).getroot()
        res = np.empty((0, 5))

        # ------------------------------------------------------------#
        # 找到每个.xml文件中的bbox
        # lower().strip()转化小写移除字符串头尾空格
        # vstack() 水平堆叠
        # ------------------------------------------------------------#
        for obj in xml_ori.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                cur_pt = int(cur_pt * h / h_ori) if i % 2 == 1 else int(cur_pt * w / w_ori)
                bndbox.append(cur_pt)

            bndbox.append(name)
            res = np.vstack((res, bndbox))
        print('*' * 5, res)

        # -------------------------------------------------------------#
        # 开始剪切 + 写入标签信息
        # -------------------------------------------------------------#
        i = 0
        win_size = self.crop_size  # 分块的大小
        stride = self.stride_width  # 重叠的大小
        for r in range(0, h - win_size, stride):
            for c in range(0, w - win_size, stride):
                flag = np.zeros([1, len(res)])

                youwu = False
                xiefou = True

                tmp = image.crop((c, r, c + win_size, r + win_size))
                for re in range(res.shape[0]):
                    xmin, ymin, xmax, ymax, label = res[re]
                    # ------------------------------------------------#
                    # 判断bb是否在当前剪切的区域内
                    # ------------------------------------------------#
                    if int(xmin) >= c and int(xmax) <= c + win_size and int(ymin) >= r and int(ymax) <= r + win_size:
                        flag[0][re] = 1
                        youwu = True
                    elif int(xmin) < c or int(xmax) > c + win_size or int(ymin) < r or int(ymax) > r + win_size:
                        pass
                    else:
                        xiefou = False
                        break

                # 如果物体被分割了，则忽略不写入,
                # 这个直接就给过滤了可能会造成一些影响----------
                if xiefou:
                    # 有物体则写入xml文件
                    if youwu:
                        # ---------------------------------------------------#
                        # 创建.xml文件 + 写入bb
                        # ---------------------------------------------------#
                        doc = Document()

                        width, height, channel = str(win_size), str(win_size), str(3)

                        annotation = doc.createElement('annotation')
                        doc.appendChild(annotation)

                        size_chartu = doc.createElement('size')
                        annotation.appendChild(size_chartu)

                        width1 = doc.createElement('width')
                        width1_text = doc.createTextNode(width)
                        width1.appendChild(width1_text)
                        size_chartu.appendChild(width1)

                        height1 = doc.createElement('height')
                        height1_text = doc.createTextNode(height)
                        height1.appendChild(height1_text)
                        size_chartu.appendChild(height1)

                        channel1 = doc.createElement('channel')
                        channel1_text = doc.createTextNode(channel)
                        channel1.appendChild(channel1_text)
                        size_chartu.appendChild(channel1)

                        for re in range(res.shape[0]):

                            xmin, ymin, xmax, ymax, label = res[re]

                            xmin = int(xmin)
                            ymin = int(ymin)
                            xmax = int(xmax)
                            ymax = int(ymax)

                            if flag[0][re] == 1:

                                xmin = str(xmin - c)
                                ymin = str(ymin - r)
                                xmax = str(xmax - c)
                                ymax = str(ymax - r)

                                object_charu = doc.createElement('object')
                                annotation.appendChild(object_charu)

                                name_charu = doc.createElement('name')
                                name_charu_text = doc.createTextNode(label)
                                name_charu.appendChild(name_charu_text)
                                object_charu.appendChild(name_charu)

                                dif = doc.createElement('difficult')
                                dif_text = doc.createTextNode('0')
                                dif.appendChild(dif_text)
                                object_charu.appendChild(dif)

                                bndbox = doc.createElement('bndbox')
                                object_charu.appendChild(bndbox)

                                xmin1 = doc.createElement('xmin')
                                xmin_text = doc.createTextNode(xmin)
                                xmin1.appendChild(xmin_text)
                                bndbox.appendChild(xmin1)

                                ymin1 = doc.createElement('ymin')
                                ymin_text = doc.createTextNode(ymin)
                                ymin1.appendChild(ymin_text)
                                bndbox.appendChild(ymin1)

                                xmax1 = doc.createElement('xmax')
                                xmax_text = doc.createTextNode(xmax)
                                xmax1.appendChild(xmax_text)
                                bndbox.appendChild(xmax1)

                                ymax1 = doc.createElement('ymax')
                                ymax_text = doc.createTextNode(ymax)
                                ymax1.appendChild(ymax_text)
                                bndbox.appendChild(ymax1)
                            else:
                                continue
                        xml_name = oriname + '_%03d.xml' % (i)
                        to_xml_name = os.path.join(lablel_xml_folder_new, xml_name)
                        with open(to_xml_name, 'wb+') as f:
                            f.write(doc.toprettyxml(indent="\t", encoding='utf-8'))
                        # name = '%02d_%02d_%02d_.bmp' % (number, int(r/win_size), int(c/win_size))
                        img_name = oriname + '_%03d.jpg' % (i)
                        to_name = os.path.join(image_crop_folder, img_name)
                        i = i + 1
                        tmp.save(to_name)


    def update_xml(self, original_xml, new_xml_dir):
        # Parse original XML
        tree = ET.parse(original_xml)
        root = tree.getroot()

        scaled_width = self.scaled_width
        scaled_height = self.scaled_height
        padding_top = self.padding_top
        padding_bottom = self.padding_bottom
        padding_left = self.padding_left
        padding_right = self.padding_right

        # Get original image size
        original_size = root.find('size')
        original_width = int(original_size.find('width').text)
        original_height = int(original_size.find('height').text)

        # 更新xml中的图像的高度宽度
        original_size.find('width').text = str(scaled_width + padding_right)
        original_size.find('height').text = str(scaled_height + padding_bottom)

        # Resize and pad bounding boxes
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            # Resize coordinates
            width_ratio = scaled_width / original_width
            height_ratio = scaled_height / original_height
            xmin = int(xmin * width_ratio)
            ymin = int(ymin * height_ratio)
            xmax = int(xmax * width_ratio)
            ymax = int(ymax * height_ratio)

            # # 右边和下边补零，好像是不需要更新坐标的，只需要将orignal_image  --匹配-- resized_image
            # # # Pad coordinates
            # xmin += padding_left
            # ymin += padding_top
            # xmax += padding_right
            # ymax += padding_bottom

            # Update XML with new bounding box coordinates
            bbox.find('xmin').text = str(xmin)
            bbox.find('ymin').text = str(ymin)
            bbox.find('xmax').text = str(xmax)
            bbox.find('ymax').text = str(ymax)

        # Save new XML
        xml_name = os.path.splitext(os.path.basename(original_xml))[0]
        new_xml_name = f'{xml_name}.xml'
        new_xml_path = os.path.join(new_xml_dir, new_xml_name)
        if not os.path.exists(new_xml_dir):
            os.makedirs(new_xml_dir)
        tree.write(new_xml_path)
        print(f"Generated XML: {new_xml_path}")

    def pilplot_xml_paddedimg(self, image, xml_file, box_save_dir=None):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # # 构建图像路径
        image_file = os.path.splitext(os.path.basename(xml_file))[0] + '.jpg'
        # image_path = os.path.join(image_folder, image_file)
        # if not os.path.isfile(image_path):
        #     print(f"Error: Corresponding image file not found for {xml_file}")
        #     return

        # image = Image.open(image)
        draw = ImageDraw.Draw(image)

        # 解析每个对象的标注信息并在图像上绘制标注框
        for obj in root.findall('object'):
            class_name_element = obj.find('name')
            if class_name_element is None:
                print(f"Error: 'name' element not found in {xml_file}")
                continue
            class_name = class_name_element.text

            bbox = obj.find('bndbox')
            if bbox is None:
                print(f"Error: 'bndbox' element not found in {xml_file}")
                continue

            xmin_element = bbox.find('xmin')
            ymin_element = bbox.find('ymin')
            xmax_element = bbox.find('xmax')
            ymax_element = bbox.find('ymax')

            if xmin_element is None or ymin_element is None or xmax_element is None or ymax_element is None:
                print(f"Error: Invalid bounding box coordinates in {xml_file}")
                continue

            xmin = int(xmin_element.text)
            ymin = int(ymin_element.text)
            xmax = int(xmax_element.text)
            ymax = int(ymax_element.text)

            # 绘制标注框
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=(0, 255, 0), width=2)
            draw.text((xmin, ymin - 10), class_name, fill=(0, 255, 0), font=None)  # 如果需要在图上标注类别

        if box_save_dir is None:
            # 显示图像
            image.show()
        else:
            save_path = os.path.join(box_save_dir, f"annotated_{image_file}")
            image.save(save_path)
            print(f"Annotated image saved at: {save_path}")



if __name__ == "__main__":

    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    #实例化class
    yolo = YOLO()
    pil_img = pil_img()

    # mode
    mode = "predict"

    # crop-image: pillow or cv2
    pillow_crop_slide = True
    # cv2_crop_slide = False

    #
    # 图块拼接输出，当然还是画出来  ---不美观
    cat_pillow_img_blocks = False

    # resize -> 640*640 detect_out
    resize_detect = True

    # show in resized_image
    show_on_resized_image = True
    # # show on padded_image
    # show_on_resized_image = False

    # dir_origin_path = "D:/chyCodespace/repo/yolov8-pytorch/testinfo/test_img111/"
    # dir_save_path = "D:/chyCodespace/repo/yolov8-pytorch/testinfo/result111a/"
    # # 创建目标文件夹（如果dir_save_path不存在）
    # if not os.path.exists(dir_save_path):
    #     os.makedirs(dir_save_path)

    # image_block大小、重叠
    crop_size = 640
    target_size = (3000, 2400)
    overlap = (200, 400)

    if mode == "predict":
        '''
        1、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。
        2、如果想要获得预测框的坐标，可以进入yolo.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
        3、如果想要利用预测框截取下目标，可以进入yolo.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
        在原图上利用矩阵的方式进行截取。
        4、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入yolo.detect_image函数，在绘图部分对predicted_class进行判断，
        比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
        '''
        while True:
            img = input('Input image filename:')
            # # -------  shared_paras class
            # if pillow_crop_slide:
            #     try:
            #         image = Image.open(img)
            #         # resized_image = pil_img_resize(image)
            #         resized_image = pilimg_resize(image, target_size)
            #         # width, height = image.size
            #         # image = cv2.imread(img)
            #         # hright, width, _ = image.shape
            #         # print("image.size:", image.size)   # (x, x)
            #         # #error  print("image.shape:", image.shape)   # error
            #     except:
            #         print('Open Error! Try again!')
            #         continue
            #
            #     #### -----传参class  shared_params
            #     # 实例化shared_params(), 然后传入pilimg_padded_crop更新参数，
            #     # 最后调用这个实例，参数就可以传出来了
            #     set_shared_params = shared_params()
            #     crops, padded_image = pilimg_padded_crop(resized_image, crop_size, set_shared_params=set_shared_params, overlap=(128, 128), plot=True, all_box_show=True)
            #     # 获取共享参数 shared params
            #     width = set_shared_params.width
            #     height = set_shared_params.height
            #     stride_width = set_shared_params.stride_width
            #     stride_height = set_shared_params.stride_height
            #     rows_num = set_shared_params.rows_num
            #     cols_num = set_shared_params.cols_num
            #     padding_bottom = set_shared_params.padding_bottom
            #     padding_right = set_shared_params.padding_right
            #
            #     print("pillow crop start")
            #     # 新建一个画布
            #     # # # result_image = Image.new("RGB", padded_image.size)
            #     # # # draw = ImageDraw.Draw(result_image)
            #     # 以padded_image作为画布
            #     result_image = padded_image.copy()
            #     draw = ImageDraw.Draw(result_image)
            #
            #     # 字体与边框厚度
            #     font = ImageFont.truetype(font='model_data/simhei.ttf',
            #                               size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
            #     thickness = int(max((image.size[0] + image.size[1]) // np.mean(yolo.input_shape), 1))
            #
            #     # 遍历crop，获取对应crop的top_label, top_conf, top_boxes
            #     for i, crop in enumerate(crops):
            #         # 取出存入时候block的行标列标
            #         # 利用了ImageBlock 这个class
            #         row_index = crop.row_index
            #         col_index = crop.col_index
            #         crop = crop.block  # crop - pillow-Image
            #
            #         detect_out = yolo.detect_image(crop, crop=True)  # detect是一个图片本身没什么用
            #         results_out = yolo.results
            #         if detect_out is None:
            #             continue
            #         else:
            #             top_label = np.array(results_out[0][:, 5], dtype='int32')
            #             top_conf = results_out[0][:, 4]
            #             top_boxes = results_out[0][:, :4]
            #
            #             for i, c in list(enumerate(top_label)):
            #                 predicted_class = yolo.class_names[int(c)]
            #                 box = top_boxes[i]
            #                 score = top_conf[i]
            #
            #                 top, left, bottom, right = box
            #
            #                 top = max(0, np.floor(top).astype('int32')) + row_index*stride_height
            #                 left = max(0, np.floor(left).astype('int32')) + col_index*stride_width
            #                 bottom = min(image.size[1], np.floor(bottom).astype('int32')) + row_index*stride_height
            #                 right = min(image.size[0], np.floor(right).astype('int32')) + col_index*stride_width
            #
            #                 label = '{} {:.2f}'.format(predicted_class, score)
            #                 # pillow 8.2.0
            #                 # 这里又重新设置了
            #                 # draw = ImageDraw.Draw(result_image)
            #                 label_size = draw.textsize(label, font)
            #                 label = label.encode('utf-8')
            #
            #                 print(label, top, left, bottom, right)
            #
            #                 if top - label_size[1] >= 0:
            #                     text_origin = np.array([left, top - label_size[1]])
            #                 else:
            #                     text_origin = np.array([left, top + 1])
            #
            #                 for i in range(thickness):
            #                     draw.rectangle([left + i, top + i, right - i, bottom - i], outline=yolo.colors[c])
            #                     # print(yolo.colors[c])
            #
            #                 draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=yolo.colors[c])
            #                 print("draw ranctangle")
            #                 draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            #                 print("draw text")
            #                 print(row_index, col_index)
            #     if show_on_resized_image:
            #         # 去掉原画布（padded_img的right bottom补零黑边）
            #         #                          .crop((left, top, right, bottom))
            #         paddedimg_width, paddedimg_height = padded_image.size
            #         result_image = result_image.crop((0, 0, paddedimg_width-padding_right, paddedimg_height-padding_bottom))
            #         result_image.show()
            #     else:    # show on padded_img
            #         result_image.show()

            # ----- pil_img class
            if pillow_crop_slide:
                try:
                    image = Image.open(img)
                    # resized_image = pil_img_resize(image)
                    resized_image = pil_img.image_resize(image, target_size)
                    # width, height = image.size
                    # image = cv2.imread(img)
                    # hright, width, _ = image.shape
                    # print("image.size:", image.size)   # (x, x)
                    # #error  print("image.shape:", image.shape)   # error
                except:
                    print('Open Error! Try again!')
                    continue
                crops, padded_image = pil_img.padded_image_crop(resized_image, crop_size, overlap, plot=False)
                width = pil_img.width
                height = pil_img.height
                stride_width = pil_img.stride_width
                stride_height = pil_img.stride_height
                rows_num = pil_img.rows_num
                cols_num = pil_img.cols_num
                padding_bottom = pil_img.padding_bottom
                padding_right = pil_img.padding_right
                print("padding_bottom:", padding_bottom)

                print("pillow crop start")
                # 新建一个画布
                # # # result_image = Image.new("RGB", padded_image.size)
                # # # draw = ImageDraw.Draw(result_image)
                # 以padded_image作为画布
                result_image = padded_image.copy()
                draw = ImageDraw.Draw(result_image)

                # 字体与边框厚度
                font = ImageFont.truetype(font='model_data/simhei.ttf',
                                          size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
                thickness = int(max((image.size[0] + image.size[1]) // np.mean(yolo.input_shape), 1))

                # 遍历crop，获取对应crop的top_label, top_conf, top_boxes
                results_record = []
                for i, crop in enumerate(crops):
                    # 取出存入时候block的行标列标
                    # 利用了ImageBlock 这个class
                    row_index = crop.row_index
                    col_index = crop.col_index
                    crop = crop.block  # crop - pillow-Image

                    detect_out = yolo.detect_image(crop, crop=True)  # detect_out是一个图片本身没什么用
                    results_out = yolo.results
                    if detect_out is None:
                        continue
                    else:
                        top_label = np.array(results_out[0][:, 5], dtype='int32')
                        top_conf = results_out[0][:, 4]
                        top_boxes = results_out[0][:, :4]

                        for i, c in list(enumerate(top_label)):
                            predicted_class = yolo.class_names[int(c)]
                            box = top_boxes[i]
                            score = top_conf[i]

                            # top, left, bottom, right = update_crops_box(box, row_index, col_index, stride_height, stride_width, (width, height))

                            top, left, bottom, right = box

                            top = max(0, np.floor(top).astype('int32')) + row_index*stride_height
                            left = max(0, np.floor(left).astype('int32')) + col_index*stride_width
                            bottom = min(image.size[1], np.floor(bottom).astype('int32')) + row_index*stride_height
                            right = min(image.size[0], np.floor(right).astype('int32')) + col_index*stride_width

                            label = '{} {:.2f}'.format(predicted_class, score)
                            # pillow 8.2.0
                            # 这里又重新设置了
                            # draw = ImageDraw.Draw(result_image)
                            label_size = draw.textsize(label, font)
                            label = label.encode('utf-8')

                            print(label, top, left, bottom, right)

                            # result_record
                            result_record = {
                                            'label': predicted_class,
                                            'score': score,
                                            'box': [left, top, right, bottom],
                                            'color': yolo.colors[c],
                                            'font': font
                                            }
                            results_record.append(result_record)

                            if top - label_size[1] >= 0:
                                text_origin = np.array([left, top - label_size[1]])
                            else:
                                text_origin = np.array([left, top + 1])

                            for i in range(thickness):
                                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=yolo.colors[c])
                                # print(yolo.colors[c])

                            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=yolo.colors[c])
                            print("draw ranctangle")
                            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
                            print("draw text")
                            print(row_index, col_index)
                if show_on_resized_image:
                    # 去掉原画布（padded_img的right bottom补零黑边）
                    #                          .crop((left, top, right, bottom))
                    paddedimg_width, paddedimg_height = padded_image.size
                    if paddedimg_width is not None:
                        result_image = result_image.crop((0, 0, paddedimg_width-padding_right, paddedimg_height-padding_bottom))
                    result_image.show()
                else:    # show on padded_img
                    result_image.show()

                #
                # 绘制合并后的预测框
                draw = ImageDraw.Draw(padded_image)
                font = ImageFont.truetype(font='model_data/simhei.ttf', size=30)  # 字体大小可能需要调整

                # 应用NMS
                nms_results = nms_with_merging(results_record, iou_threshold=0.4, merge_threshold=0.5)

                # 遍历合并后的预测结果并绘制边界框和标签
                for result in nms_results:
                    label = result['label']
                    score = result['score']
                    box = result['box']

                    color = result['color']
                    font = result['font']

                    draw.rectangle(box, outline=color, width=16)
                    # 绘制带背景的文本标签
                    label_with_score = "{} {:.2f}".format(label, score)
                    text_size = draw.textsize(label_with_score, font=font)
                    text_origin = (box[0] + 2, box[1] - text_size[1] - 4)
                    draw.rectangle([text_origin, (text_origin[0] + text_size[0], text_origin[1] + text_size[1])],
                                   fill=color)
                    draw.text(text_origin, label_with_score, fill='white', font=font)  # 使用白色字体

                # 显示或保存修改后的图像
                padded_image.show()


            #
            #
            #     # img_blocks = []
            #     # for crop in crops:
            #     #     r_image = yolo.detect_image(crop, crop=False, count=False)
            #     #     # r_image.show()
            #     #     img_blocks.append(r_image)
            #     # if cat_pillow_img_blocks:
            #     #     cat_pilimg_block(img_blocks)

            # 好像还不行吧
            # if cv2_crop_slide:
            #     try:
            #         # image = Image.open(img)
            #         # width, height = image.size
            #         image = cv2.imread(img)
            #         hright, width, _ = image.shape
            #         # print("image.size:", image.size)   # (x, x)
            #         # #error  print("image.shape:", image.shape)   # error
            #     except:
            #         print('Open Error! Try again!')
            #         continue
            #
            #     crops = crop_image_cv2(image, crop_size, overlap)
            #     print("cv2 crop start")
            #     for crop in crops:
            #         r_image = yolo.detect_image(crop, crop=False, count=False)
            #         r_image.show()

            if resize_detect:
                try:
                    image = Image.open(img)
                    width, height = image.size
                except:
                    print('Open Error! Try again!')
                    continue
                r_image = yolo.detect_image(image, crop=False, count=False)
                print("----------直接resize到640输出-----")
                if r_image is not None:
                    r_image.show()

            # else:
            #     # crop - predict - merge
            #     crops = crop_image_pil(image, crop_size, overlap)
            #     print("crop start")
            #     results = []
            #     for crop in crops:
            #         imgs, result = yolo.detect_image(crop, crop=False, count=False, save=True)
            #         results.append(result)
            #         print("result:", results)
            #         # print("crop done")
            #     print("merge start")
            #     merged_result = merge_results(results, crop_size, crop_size, overlap)
            #     print("merge out")



























