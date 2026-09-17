# 把一套图片和掩码整理到一起，同时汇总所有有标注的图片，不再分日期

import os
import shutil
from glob import glob
from tqdm import tqdm
from PIL import Image, ImageOps

def parse_bbox_log(bbox_log_file):
    bbox_info = {}
    with open(bbox_log_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(', ')
            filename = parts[0]
            left, top, right, bottom = map(int, parts[1:])
            bbox_info[filename] = (left, top, right, bottom)
    return bbox_info

def pad_to_size(image, left, top, target_width, target_height):
    width, height = image.size
    if image.mode == 'L':  # For grayscale images
        result = Image.new(image.mode, (target_width, target_height), 0)
    else:  # For RGB or other modes
        result = Image.new(image.mode, (target_width, target_height), (0, 0, 0))
    # left = (target_width - width) // 2
    # top = (target_height - height) // 2
    result.paste(image, (left, top))
    return result

def copy_files_with_mid_and_log(normal_files, label_path, dest_path, log_file, has_not):
    # 打开txt文件用于写入
    with open(log_file, 'a') as log:
        # 获取总共所有的 *_Normal.png 文件
        # normal_files = glob(os.path.join(root, '*/*/*_Normal.png'))

        # 遍历每个 *_Normal.png 文件
        for normal_file in tqdm(normal_files):
            filename = os.path.basename(normal_file)

            # 查找 label_path 中的已标注同名文件
            label_file = os.path.join(label_path, filename)
            if os.path.exists(label_file):

                # 写入文件名到log文件
                log.write(filename + '\n')

                # 提取 'IMG_--------_' 到 '_Normal.png' 之间的字符串
                if '--------_' in filename:
                    mid = filename.split('--------_')[1].split('_Normal.png')[0]
                else:
                    mid = filename.split('_Normal.png')[0]

                # 创建目标文件夹
                target_dir = os.path.join(dest_path, mid)
                os.makedirs(target_dir, exist_ok=True)

                # 查找所有与 *_Normal.png 同名的文件，并复制到目标文件夹
                pattern = normal_file.replace('Normal', '*')
                related_files = glob(pattern)
                for file in related_files:
                    shutil.copy(file, target_dir)

                # 改名后复制label文件到目标文件夹
                new_label_name = filename.replace('Normal', 'Mask')
                new_label_path = os.path.join(target_dir, new_label_name)
                shutil.copy(label_file, new_label_path)


def copy_files_with_mid_and_log_crop(normal_files, label_path, dest_path, log_file, has_not, bbox_info):
    # 打开txt文件用于写入
    with open(log_file, 'a') as log:
        # 遍历每个 *_Normal.png 文件
        for normal_file in tqdm(normal_files):
            filename = os.path.basename(normal_file)

            # 查找 label_path 中的已标注同名文件
            label_file = os.path.join(label_path, filename)
            if os.path.exists(label_file):

                # 写入文件名到log文件
                log.write(filename + '\n')

                # 提取 'IMG_--------_' 到 '_Normal.png' 之间的字符串
                if '--------_' in filename:
                    mid = filename.split('--------_')[1].split('_Normal.png')[0]
                else:
                    mid = filename.split('_Normal.png')[0]

                # 创建目标文件夹
                target_dir = os.path.join(dest_path, mid)
                os.makedirs(target_dir, exist_ok=True)

                # 查找所有与 *_Normal.png 同名的文件，并复制到目标文件夹
                pattern = normal_file.replace('Normal', '*')
                related_files = glob(pattern)
                for file in related_files:
                    shutil.copy(file, target_dir)

                # 改名后复制并处理label文件到目标文件夹
                new_label_name = filename.replace('Normal', 'Mask')
                new_label_path = os.path.join(target_dir, new_label_name)

                # try:
                with Image.open(label_file) as img:
                    if new_label_name.replace('Mask','Normal') in bbox_info.keys():
                        left, top, right, bottom = bbox_info[new_label_name.replace('Mask','Normal')]
                        # cropped_img = img.crop((left, top, right, bottom))
                        padded_img = pad_to_size(img, left, top, 2432, 2040)
                        padded_img.save(new_label_path)
                    else:
                        padded_img = pad_to_size(img, 0,0,2432, 2040)
                        padded_img.save(new_label_path)
                # except Exception as e:
                #     print(f"Error processing {label_file}: {e}")


def read_log_file(log_file):
    with open(log_file, 'r') as file:
        lines = file.readlines()
    return set(line.strip() for line in lines)


def copy_files_with_mid_and_log_new(normal_files, label_path, dest_path, log_file, has_not, log_set):
    # 打开txt文件用于写入
    with open(log_file, 'a') as log:
        # 遍历每个 *_Normal.png 文件
        for normal_file in tqdm(normal_files):
            filename = os.path.basename(normal_file)

            # 仅处理在log_set中的文件
            if filename not in log_set:
                continue

            # 查找 label_path 中的已标注同名文件
            label_file = os.path.join(label_path, filename)
            if os.path.exists(label_file):

                # 写入文件名到log文件
                log.write(filename + '\n')

                # 提取 'IMG_--------_' 到 '_Normal.png' 之间的字符串
                if '--------_' in filename:
                    mid = filename.split('--------_')[1].split('_Normal.png')[0]
                else:
                    mid = filename.split('_Normal.png')[0]

                # 创建目标文件夹
                target_dir = os.path.join(dest_path, mid)
                os.makedirs(target_dir, exist_ok=True)

                # 查找所有与 *_Normal.png 同名的文件，并复制到目标文件夹
                pattern = normal_file.replace('Normal', '*')
                related_files = glob(pattern)
                for file in related_files:
                    shutil.copy(file, target_dir)

                # 改名后复制label文件到目标文件夹
                new_label_name = filename.replace('Normal', 'Mask')
                new_label_path = os.path.join(target_dir, new_label_name)
                shutil.copy(label_file, new_label_path)


def check_not_exist(normal_files, log_file, has_not):
    # normal_files = glob(os.path.join(root, '*/*/*_Normal.png'))
    normal_files = [os.path.basename(fff) for fff in normal_files]
    exist = []
    with open(log_file, 'r') as file:
        for line in file:
            if line != "\n":
                exist.append(line.strip())
    for filename in tqdm(normal_files):
        if filename not in exist:
            if '--------_' in filename:
                mid = filename.split('--------_')[1].split('_Normal.png')[0]
            else:
                mid = filename.split('_Normal.png')[0]

            # 创建目标文件夹
            target_dir = os.path.join(has_not, mid)
            os.makedirs(target_dir, exist_ok=True)

            # 查找所有与 *_Normal.png 同名的文件，并复制到目标文件夹
            # pattern =
            related_files = glob(f"{root_root}/*/*/{filename}".replace('Normal', '*'))\
                            +glob(f"{root_root}/*/{filename}".replace('Normal', '*'))
            for file in related_files:
                shutil.copy(file, target_dir)

name = "头部套膜破损"
# name = "喷码不良"
# name = "底部钢壳破损"
# root_root = r"F:\xiamen_已分类\cri_钢壳划痕_头_25d"
# root_root = r"F:\xiamen_已分类\min_套膜凸点_侧_25d"
# root_root = r"F:\xiamen_已分类\min_套膜划痕_侧_25d"
# root_root = r"F:\xiamen_当前未标注\头部套膜污渍25D未标注50"
root_root = r"F:\xiamen_已分类\cri_套膜破损_头_25d"
# 示例使用
# root = glob(fr'{root_root}/*/*_Normal.png')
root = glob(fr'{root_root}/*/*/*_Normal.png') + glob(fr'{root_root}/*/*_Normal.png')
label_path = r'F:\xiamen_未整理\0518-待拷走\0518标注\0518标注\0517-2.5D端面-头部缺陷normal图-100\0517-2.5D端面-头部污渍-normal-50张\0517-2.5D端面-头部污渍-normal\头部污渍_labels'
# label_path = r'F:\xiamen_已标注\5601项目训练数据\0511-2.5D-侧壁-套膜凸点-146\2_labels'
# label_path = r'F:\xiamen_已标注\5601项目训练数据\0517-底端钢壳划痕_25d_2000x2000_100-已解析-96\新项目_labels'
# label_path = r'F:\xiamen_已标注\5601项目训练数据\0517-侧面套膜破损_25d_2867x1948_146-标注134\新项目_labels'
# label_path = r'F:\xiamen_已标注\5601项目训练数据\0515-头部套膜破损2-已解析-34\新项目_labels'
dest_path = fr'F:\xiamen_最终整理\{name}-2.5D-0NG-0520'
# dest_path = fr'F:\xiamen_最终整理\{name}-2.5D-50NG-0517-'
os.makedirs(dest_path,exist_ok=True)
hasnot = fr'F:\xiamen_当前未标注/{name}25D未标注-0520'
log_file = f'F:/xiamen_最终整理/头部套膜破损-2.5D-54NG-0517/头部破损25D已标注.txt'
# log_file = f'{dest_path}/{name}25D已标注-0520.txt'

# bbox_log_file = r'../bbox/盖帽划痕_crop_log.txt'  # 替换为您的bbox日志文件路径
#
# bbox_info = parse_bbox_log(bbox_log_file)


# 从记录文件读取已经存在的文件名
# log_set = read_log_file("../侧面凸点25D已标注.txt")
# copy_files_with_mid_and_log_new(root, label_path, dest_path, log_file, hasnot,log_set)
# copy_files_with_mid_and_log(root, label_path, dest_path, log_file, hasnot)

# copy_files_with_mid_and_log_crop(root, label_path, dest_path, log_file, hasnot, bbox_info)

# # copy_files_with_mid_and_log(glob(r'F:\xiamen_已分类\maj_缺口_头_25d/*/*/*_Shape1.png')
# #                             + glob(r'F:\xiamen_已分类\maj_缺口_头_25d/*/*_Shape1.png'),
# #                             r"F:\xiamen_已标注\5601项目训练数据\0511-2.5D-端面-盖帽缺口-shapes-8\新项目_labels",
# #                             dest_path, log_file, hasnot)

check_not_exist(root,log_file,hasnot)
