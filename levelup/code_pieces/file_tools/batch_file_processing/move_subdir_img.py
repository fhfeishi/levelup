import os
import shutil
from tqdm import tqdm


def move_files_and_clean_empty_dirs(root_path):
    # Walk through the directory structure
    for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
        for filename in tqdm(filenames):
            if filename.endswith(('.bin', '.tif', '.bmp')):
                current_file_path = os.path.join(dirpath, filename)
                
                # Check if the file is already in the correct structure
                if dirpath.count(os.sep) == root_path.count(os.sep) + 2:
                    print(f"Already in correct structure: {current_file_path}")
                    continue
                
                new_dir_path = os.path.dirname(os.path.dirname(current_file_path))
                new_file_path = os.path.join(new_dir_path, filename)

                if os.path.isfile(new_file_path):
                    new_dir_path = os.path.dirname(new_file_path)
                    new_fname = os.path.basename(new_file_path).replace('.', f"_{zz}.")
                    # new_file_path = os.path.join(new_dir_path, new_fname)
                    new_file_path = f"{new_dir_path}/{new_fname}"
                    print("----------=-=-=-=-",new_file_path)
                    zz += 1

                # # Move the file
                shutil.move(current_file_path, new_file_path)
                print(f"Moved: {current_file_path} to {new_file_path}")

        # After moving files, remove empty directories
        if not os.listdir(dirpath):
            os.rmdir(dirpath)
            print(f"Removed empty directory: {dirpath}")

if __name__ == "__main__":
    root_path = r"E:\xiamen_已分类\cri_凹坑_底_3d"
    zz = 1
    move_files_and_clean_empty_dirs(root_path)
    
    # 自动
    # root_path = r"E:\xiamen_已分类"
    # for dir in os.listdir(root_path):
    #     if "_3d" in str(dir):
    #         print(dir)
    #         root_path = f"{root_path}/{dir}"
    #         move_files_and_clean_empty_dirs(root_path)
