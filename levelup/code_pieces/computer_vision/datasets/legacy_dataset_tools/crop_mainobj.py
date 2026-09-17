import os
from PIL import Image
import numpy as np
from tqdm import tqdm


def getSquaredCropBox(xmin, xmax, ymin, ymax, w, h):
    top_rest = ymin
    left_rest = xmin
    bot_rest = h - ymax
    right_rest = w - xmax

    # 可能无法把目标裁出来一个方块
    w_box = xmax - xmin  # box w
    h_box = ymax - ymin  # box h
    h_reset = top_rest + bot_rest  # rest w
    w_rest = right_rest + left_rest  # rest h
    if (w_box-h_box)>h_reset or (h_box-w_box)>w_rest:
        return xmin, xmax, ymin, ymax

    if h_box > w_box:
        # 宽度方向
        expand_x_ = h_box - w_box
        xmin_s = max(0, xmin - (expand_x_ // 2))   # xmin_s 可能需要更小一点
        xmax_s = xmax + (expand_x_ - (xmin - xmin_s))
        if xmax_s > w:
            xmax_s = min(w, xmax + (expand_x_ // 2))
            xmin_s = xmin - (expand_x_-(xmax_s-xmax))
        xmin, xmax = xmin_s, xmax_s

    else:
        # ymax-ymin <= xmax-xmin
        # 高度方向
        expand_y_ = w_box - h_box
        ymin_s = max(0, ymin - (expand_y_ // 2))  # ymin_s 可能需要更小一点
        ymax_s = ymax + (expand_y_ - (ymin-ymin_s))
        if ymax_s > h:
            ymax_s = min(h, ymax + (expand_y_ //2))
            ymin_s = ymin - (expand_y_-(ymax_s-ymax))
        ymin, ymax = ymin_s,ymax_s

    return xmin, xmax, ymin, ymax

def getExpandCropBox(xmin, xmax, ymin, ymax, w, h, expand=(2, 320)):
    """
    exoand=(2,320)  扩展 2[0,320] = [320, 640]
    """
    top_rest = ymin
    left_rest = xmin
    bot_rest = h - ymax
    right_rest = w - xmax

    expand_xn_ = ((((xmax - xmin) // expand[1]) + expand[0]) * expand[1]) - (xmax - xmin)
    expand_yn_ = ((((ymax - ymin) // expand[1]) + expand[0]) * expand[1]) - (ymax - ymin)

    # 还是要考虑无法expand的情况
    if expand_xn_ > (left_rest + right_rest):
        # print("没有成功crop, obj is too big, image no enough margin --expand --x")
        pass
    else:
        x1 = max(0, xmin - (expand_xn_ // 2))
        x2 = xmax + (expand_xn_ - (xmin - x1))
        if x2 > w:
            x2 = min(w, xmax + (expand_xn_ // 2))
            # x1 = max(0, xmin - (expand_xn_ - (x2 - xmax)))  # 虽说这个计算式也可能<0,那可就是说明图片不够宽 图片不够宽的逻辑if已经处理过了
            x1 = xmin - (expand_xn_ - (x2 - xmax))
        xmin, xmax = x1, x2

    if expand_yn_ > (top_rest + bot_rest):
        # print("没有成功crop, obj is too big, image no enough margin --expand --y")
        pass
    else:
        y1 = max(0, ymin - (expand_yn_ // 2))
        y2 = ymax + (expand_yn_ - (ymin - y1))
        if y2 > h:
            y2 = min(h, ymax + (expand_yn_ //2))
            # y1 = max(0, ymin - (expand_yn_ - (y2-ymax)))  # 虽说这个计算式也可能<0,那可就是说明图片不够高 图片不够高的逻辑if已经处理过了
            y1 = ymin - (expand_yn_ - (y2-ymax))
        ymin, ymax = y1, y2

    return xmin, xmax, ymin, ymax



def crop_masks(jpg_dir, png_dir, save_dir):
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 遍历PNG掩码目录以找到所有掩码文件
    for mask_file in tqdm(os.listdir(png_dir)):
        if mask_file.endswith('.png'):
            mask_path = os.path.join(png_dir, mask_file)
            image_file = mask_file.replace('.png', '.jpg')
            image_path = os.path.join(jpg_dir, image_file)
            
            if not os.path.exists(image_path):
                print(f"Image file {image_path} not found.")
                continue

            # 读取图像和掩码
            image = Image.open(image_path)
            mask = Image.open(mask_path).convert('RGB') # 不rgb len(where[0])可能会报错？？
            mask_array = np.array(mask)  # h w c
            
            # # 寻找掩码中的目标边界框   gray label
            # where = np.where(mask_array)
            # if where[0].size == 0 or where[1].size == 0:
            #     print(f"No target found in mask {mask_path}.")
            #     continue
            # ymin, xmin = np.min(where[0]), np.min(where[1])
            # ymax, xmax = np.max(where[0]), np.max(where[1])

            # 假设目标不是黑色   rgb mask
            # 如果掩码是彩色的，此处需要修改以适应实际颜色通道条件
            target = np.any(mask_array != [0, 0, 0], axis=-1)  # 检查是否非黑色

            where = np.where(target)
            # if where[0].size == 0 or where[1].size == 0:
            if len(where[0]) == 0 or len(where[1]) == 0:  # 确保where不为空
                print(f"No target found in mask {mask_path}.")
                continue
            ymin, xmin = np.min(where[0]), np.min(where[1])
            ymax, xmax = np.max(where[0]), np.max(where[1])

            w, h = image.size
            # print(f"start  Cropped and saved {image_file} and {mask_file}")

            # square bbox
            x1,x2,y1,y2 = getSquaredCropBox(xmin, xmax, ymin, ymax, w, h)

            # expand bbox
            x1, x2, y1, y2 = getExpandCropBox(x1, x2, y1, y2, w, h)

            # 裁剪图像和掩码
            cropped_image = image.crop((x1, y1, x2, y2))
            cropped_mask = mask.crop((x1, y1, x2, y2))

            # 保存裁剪后的图像和掩码
            cropped_image.save(os.path.join(save_dir, image_file))
            cropped_mask.save(os.path.join(save_dir, mask_file))

            print(f"Cropped and saved {image_file} and {mask_file}")


# 调用函数
jpg_dir = r'D:\ddesktop\xianlan_measure\bianping_dataset\jpgs'
mask_dir = r'D:\ddesktop\xianlan_measure\bianping_dataset\pngs'
output_dir = r'D:\ddesktop\xianlan_measure\codespace\deeplabv3p_flat\datasets\before'
crop_masks(jpg_dir, mask_dir, output_dir)