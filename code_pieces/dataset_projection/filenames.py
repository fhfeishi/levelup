# 文件名中的空格、 中文、 

import os
from tqdm import tqdm

# 去掉空格\中文
# filename.replace(' ', '')
def del_kongge(dir_path):
    # 获取目录中的所有文件和文件夹名
    names = os.listdir(dir_path)
    # 使用 tqdm 来显示进度条
    for name in tqdm(names):
        # 新文件名去除空格
        new_name = name.replace(' ', '')
        # 如果新旧文件名不同，则重命名
        if new_name != name:
            # 构建完整的旧文件路径和新文件路径
            old_path = os.path.join(dir_path, name)
            new_path = os.path.join(dir_path, new_name)
            # 重命名操作
            os.rename(old_path, new_path)
            print(f'rename {old_path} to {new_path}')

def cn2eng(dir_path, cn='巡检图片', eng='image'):
    names = [name for name in os.listdir(dir_path) if cn in name]
    for name in tqdm(names):
        new_name = name.replace(cn, eng) 
        if new_name != name:
            old_path = os.path.join(dir_path, name)
            new_path = os.path.join(dir_path, new_name)
            os.rename(old_path, new_path)
            print(f'rename {old_path} to {new_path}')


def JPG2jpg(dir_path):
    names = [name for name in os.listdir(dir_path) if name.endswith('.JPG')]
    for name in tqdm(names):
        new_name = name.replace('.JPG', '.jpg') 
        if new_name != name:
            old_path = os.path.join(dir_path, name)
            new_path = os.path.join(dir_path, new_name)
            os.rename(old_path, new_path)
            print(f'rename {old_path} to {new_path}')

# 重新命名并编号


if __name__  == '__main__':
    # dir_path = r"F:\bianse\normaldata\conservator_normal"
    # dir_path = r"F:\bianse\normaldata\xml_normal"
    # dir_path = r'F:\bianse\dataset\jpg'
    dir_path = r'F:\bianse\dataset\xml'
    # del_kongge(dir_path)
    # JPG2jpg(dir_path)
    # cn2eng(dir_path, cn='巡检图片', eng='image')
    cn2eng(dir_path, cn='散热器', eng='sanreqi')
    # cn2eng(dir_path, cn='主变压器', eng='image')
    # cn2eng(dir_path, cn='主变间隔', eng='image')
    # cn2eng(dir_path, cn='相变低套管', eng='image')
    # cn2eng(dir_path, cn='瓦斯继电器（主变本体）', eng='imagewasijidianqi')
    # cn2eng(dir_path, cn='相变高套管', eng='image')

    
    


