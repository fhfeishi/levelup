import os
import shutil
from tqdm import tqdm 

def read_txtLine(txt_path, encoding='utf-8'):
    basename_set = set()
    with open(txt_path, 'r', encoding=encoding) as f:
        for line in tqdm(f):
            # 去掉行末的换行符，并添加到set中
            basename = line.strip()
            basename_set.add(basename)
    return basename_set

def write2txt(name_set, txt_path, encoding='utf-8'):
    with open (txt_path, 'w', encoding=encoding) as f:
        for name in tqdm(name_set):
            f.write(name + '\n')

def move_nameSetFiles_normal(name_set, img_folder, target_folder):
    os.makedirs(target_folder, exist_ok=True)
    for name in tqdm(name_set):
        img_path = f"{img_folder}/{name}.png"
        assert os.path.isfile(img_path), "img path do not a file!!!"
        target_path = f"{target_folder}/{name}.png"
        shutil.copyfile(img_path, target_path)
        print(f"move {img_path} to {target_path} done!")


if __name__ == '__main__':

    txt_root = r"F:\aaaaaa-xiamen2024-ok\顶盖缺口25Dtxt"
    # 头部ok.txt
    # no.txt
    # 头部钢壳ok.txt
    # 头部套膜ok.txt
    ok_txt_file = "头部ok.txt"
    ok_txt_path = f"{txt_root}/{ok_txt_file}"

    no_txt_file = "no.txt"
    no_txt_path = f"{txt_root}/{no_txt_file}"

    ke_txt_file = "头部钢壳ok.txt"
    ke_txt_path = f"{txt_root}/{ke_txt_file}"

    mo_txt_file = "头部套膜ok.txt"
    mo_txt_path = f"{txt_root}/{mo_txt_file}"

    ke_set = read_txtLine(ke_txt_path)
    mo_set = read_txtLine(mo_txt_path)
    
    no_set = read_txtLine(no_txt_path)

    img_root = r"F:\aaaaaa-xiamen2024-ok"
    img_dir = "顶盖缺口-ok-25D-normal"
    img_folder = f"{img_root}/{img_dir}"

    all_name_set = {name.split('.')[0] for name in os.listdir(img_folder) if name.endswith('.png')}

    ok_set = all_name_set - ke_set - mo_set - no_set
    print(len(ok_set))
    write2txt(ok_set, ok_txt_path)

    
    ok_dir = "头部ok"
    target_folder = f"{img_root}/{ok_dir}"
    move_nameSetFiles_normal(ok_set, img_folder, target_folder)

    ke_ok_dir = "keok"
    ke_ok_folder =  f"{img_root}/{ke_ok_dir}"
    move_nameSetFiles_normal(ke_set, img_folder, ke_ok_folder)





