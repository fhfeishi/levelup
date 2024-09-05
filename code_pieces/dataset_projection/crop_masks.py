import os
from PIL import Image
import numpy as np

from tqdm import tqdm 
def crop_masks(jpg_dir, png_dir, save_dir, expand=640, squareCrop=True):
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
            
            # # 寻找掩码中的目标边界框
            # where = np.where(mask_array)
            # if where[0].size == 0 or where[1].size == 0:
            #     print(f"No target found in mask {mask_path}.")
            #     continue
            # ymin, xmin = np.min(where[0]), np.min(where[1])
            # ymax, xmax = np.max(where[0]), np.max(where[1])
            
            # 假设目标不是黑色
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

            # square crop  ->  square_crop_x1, square_c1rop_y1, square_crop_x2, square_crop_y2
            top_rest = xmin
            left_rest = ymin
            bot_rest = h - ymax
            right_rest = w - xmax 
            if squareCrop:
                if ymax-ymin > xmax-xmin:
                    # 宽度方向
                    expand_x_ = (ymax-ymin) - (xmax-xmin)
                    if left_rest + right_rest < expand_x_:
                        print(f"{mask_file} 没有成功crop, obj is too big, image no enough margin --square --x")
                        continue
                    else:
                        xmin_s = max(0, xmin-(expand_x_//2))
                        xmax_s = min(w, xmax+(expand_x_-(xmin-xmin_s)))
                        xmin = xmin_s
                        xmax = xmax_s
                else:
                    # ymax-ymin <= xmax-xmin
                    # 高度方向
                    expand_y_ = (xmax-xmin) - (ymax-ymin)
                    if top_rest + bot_rest < expand_y_:
                        print(f"{mask_file} 没有成功crop, obj is too big, image no enough margin --square --y")
                        continue
                    else:
                        ymin_s = max(0, ymin-(expand_y_//2))
                        ymax_s = min(h, ymax+(expand_y_-(ymin-ymin_s)))
                        ymin = ymin_s
                        ymax = ymax_s
            
            # expand128 ->   final_crop_x1, final_c1rop_y1, final_crop_x2, final_crop_y2  
            top_rest_n = xmin
            left_rest_n = ymin
            bot_rest_n = h - ymax
            right_rest_n = w - xmax 
            expand_xn_ = ((((xmax-xmin)//expand)+1) * expand) - (xmax-xmin)
            expand_yn_ = ((((ymax-ymin)//expand)+1) * expand) - (ymax-ymin)
            assert expand_xn_ == expand_yn_, f"square not work, why?"
            
            if expand_xn_ > (left_rest_n+right_rest_n):
                print(f"{mask_file} 没有成功crop, obj is too big, image no enough margin --expand --x")
                continue
            else:
                x1 = max(0, xmin-(expand_xn_//2))
                x2 = min(w, xmax+(expand_xn_-(xmin-x1)))
                
            if expand_yn_ > (top_rest_n+bot_rest_n):
                print(f"{mask_file} 没有成功crop, obj is too big, image no enough margin --expand --y")
                continue
            else:
                y1 = max(0, ymin-(expand_yn_//2))
                y2 = min(h, ymax+(expand_yn_-(ymin-y1)))
            
            # 裁剪图像和掩码
            cropped_image = image.crop((x1, y1, x2, y2))
            cropped_mask = mask.crop((x1, y1, x2, y2))

            # 保存裁剪后的图像和掩码
            cropped_image.save(os.path.join(save_dir, image_file))
            cropped_mask.save(os.path.join(save_dir, mask_file))

            print(f"Cropped and saved {image_file} and /{mask_file}")


# 调用函数
png_dir = r'D:\ddesktop\xianpian\datadata\jpg-layer3'
mask_dir = r'D:\ddesktop\xianpian\datadata\layer3-png'
output_dir = r'D:\ddesktop\xianpian\codespace\deeplabv3p_triple\datasets\before'
crop_masks(png_dir, mask_dir, output_dir)