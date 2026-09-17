import os
import shutil
from glob import glob

def get_target_dirname(file_extension):
    """根据文件后缀返回对应的目标目录名"""
    extension_to_dirname = {
        "bin": "顶部划痕-3D-bin图",
        "tif": "顶部划痕-3D-tif图",
        "other": "顶部划痕-3D-亮度图"  # 其他文件类型
    }
    return extension_to_dirname.get(file_extension, "顶部划痕-3D-亮度图")



def merge_subdir_img(data_root):
    """
    data_root/target_dirs/other-subdirs/files
    move target-file to target_dir
    """
    for root, dirs, files in os.walk(data_root):
        for file in files:
            file_extension = file.split('.')[-1]
            target_dirname = get_target_dirname(file_extension)
            target_dir = os.path.join(data_root, target_dirname)

            source_path = os.path.join(root, file)
            target_path = os.path.join(target_dir, file)
            print(f"Moving {source_path} to {target_path}")
            # shutil.copyfile(source_path, target_path)
            shutil.move(source_path, target_path)



if __name__ == "__main__":

    data_root = r"E:\xiamen_已分类\cri_划痕_头_3d\0514-顶部划痕-3D"

    target_dirname_list = ["顶部划痕-3D-bin图", "顶部划痕-3D-tif图", "顶部划痕-3D-亮度图"]

    merge_subdir_img(data_root)


