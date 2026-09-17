from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from glob import glob


def apply_mask_to_others(mask_path, image_path, save_dir, target_x=2354, target_y=768):
    mask = Image.open(mask_path).convert('L')
    mask_array = np.array(mask)

    image = Image.open(image_path).convert('L')
    image_array = np.array(image)

    mask_coords = np.column_stack(np.where(mask_array == 255))

    result_image_array = image_array.copy()
    target_gray_value = result_image_array[target_y, target_x]

    for coord in mask_coords:
        mask_y, mask_x = coord

        # 确保目标位置在shape_image范围内
        if 0 <= mask_x < result_image_array.shape[1] and 0 <= mask_y < result_image_array.shape[0]:
            result_image_array[mask_y, mask_x] = target_gray_value

    # 保存处理后的图片
    result_image = Image.fromarray(result_image_array)

    save_fname = os.path.basename(image_path)
    output_path = f"{save_dir}/{save_fname}"
    result_image.save(output_path)


if __name__ == '__main__':
    
    # batch 
    
    # image_dir = glob(r"E:\喷码ok\2.5D-线扫-侧壁套膜OK-50\*")
    # # D:\Ddesktop\ppt\work\missing-2.5D-30NG-0524
    # mask_paths = glob(r"D:\Ddesktop\ppt\work\missing-2.5D-30NG-0518\*\*_Mask.png")

    mask_paths = glob(r"D:\Ddesktop\ppt\work\missing-2.5D-30NG-0524\*\*_Mask.png")
    
    image_paths = glob(r"E:\喷码ok\2.5D-线扫-侧壁套膜OK-50\*\*.png")
    
    for mask_path in mask_paths:
        save_dir = os.path.dirname(mask_path)
        assert os.path.exists(save_dir), f"{save_dir} not exist !"
        
        mask_name = os.path.basename(mask_path)
        mask_coo = mask_name.split('--------_')[1].rsplit('_', 1)[0] if '--------_' in mask_name else mask_name.rsplit('_', 1)[0]

        for image_path in image_paths:
            parent_folder = os.path.basename(os.path.dirname(image_path))
            if parent_folder.isdigit() and 46 > int(parent_folder) > 30 and int(parent_folder) != 41:
                image_name = os.path.basename(image_path)
                image_coo = image_name.split('--------_')[1].rsplit('_', 1)[0] if '--------_' in image_name else image_name.rsplit('_', 1)[0]

                if mask_coo == image_coo:
                    print(mask_coo)
                    if "Normal" not in image_name.split('.')[0]:
                        apply_mask_to_others(mask_path, image_path, save_dir)
                else:
                    print('ggggg')


    # single 
    # mask_path = r"D:\Ddesktop\ppt\work\missing-2.5D-30NG-0518\IMG_9E710037_2024-04-10_14-30-14\IMG_9E710037_2024-04-10_14-30-14_Mask.png"
    # image_dir = r"E:\喷码ok\2.5D-线扫-侧壁套膜OK-50\8"
    # save_dir = r"D:\Ddesktop\ppt\work\missing-2.5D-30NG-0518\IMG_9E710037_2024-04-10_14-30-14"
    # for name in tqdm(os.listdir(image_dir)):
        # if name.endswith('.png'):
        #     # # get target folder name:
        #     # if '--------_' in name:
        #     #     common_part = name.split('--------_')[1].rsplit('_', 1)[0]
        #     # else:
        #     #     # 获取文件名的公共部分（去掉最后一个下划线后的部分）
        #     #     common_part = name.rsplit('_', 1)[0]
        #     if "Normal" in name.split('.')[0]:
        #         pass
        #     else:
        #         image_path = fr"{image_dir}/{name}"
        #         apply_mask_to_others(mask_path, image_path, save_dir)











