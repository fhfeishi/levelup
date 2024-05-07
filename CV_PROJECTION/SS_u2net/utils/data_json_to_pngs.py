# labelme  .json  -->  dataset:pngs
# pip install labelme==3.16.7     # 仅为了使用内置的数据转换脚本

import base64
import os
import json
import numpy as np
import PIL.Image as Image
import labelme.utils as utils
from tqdm import tqdm 

def get_ss_classes(json_folder, stamp=True):
    classes = set()
    for file in tqdm(os.listdir(json_folder)):
        path = os.path.join(json_folder, file)
        if os.path.isfile(path) and path.endswith('.json'):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if "shapes" in data:
                    for shape in data["shapes"]:
                        classes.add(shape['label'])
    if stamp:
        print(list(classes))
    return list(classes)

def json_to_jpgs_pngs(jsons_folder, classes, jpgs_folder, pngs_folder):
    
    count = os.listdir(jsons_folder)

    for i in tqdm(range(0, len(count))):
        path = os.path.join(jsons_folder, count[i])
        if os.path.isfile(path) and path.endswith('.json'):
            data = json.load(open(path, encoding='utf-8'))
            
            if data['imageData']:
                # 这个判定理论上来说是一定通过的
                imageData = data['imageData'] # base64编码的整个jpg image
            else:
                # 理论上来说这句代码不会运行，这个path有中文、空格都是可以的
                jpgPath = os.path.join(jpgs_folder, count[i].replace(".jon", ".jpg"))
                with open(jpgPath, "rb") as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode("utf-8")

            jpg = utils.img_b64_to_arr(imageData)
            # jpg numpy.ndarray

            label_name_to_value = {'background': 0}
            for shape in data['shapes']:
                label_name = shape['label']
                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)   # 字典的长度现在就是1  从0开始的
                    label_name_to_value[label_name] = label_value

            # label_values must be dense
            label_values, label_names = [], []
            for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
                label_values.append(lv)
                label_names.append(ln)
            assert label_values == list(range(len(label_values)))

            lbl = utils.shapes_to_label(jpg.shape, data['shapes'], label_name_to_value)
            # ndarray: (h, w)   value: 0, 1, 2  

            Image.fromarray(jpg).save(os.path.join(jpgs_folder, count[i].split(".")[0]+'.jpg'))

            new = np.zeros([np.shape(jpg)[0], np.shape(jpg)[1]])
            # ndarray (h, w)

            for name in label_names:
                index_json = label_names.index(name)   # list.index(name) --->int: 0, 1, 2
                index_all = classes.index(name)  # list.index(name) -->int: 0, 1, 2
                new = new + index_all*(np.array(lbl == index_json)) # ndarray + ndarray

            # 将标签如转换为png图片, lblsave(filename, lbl) 接收两个参数：filename 是预期输出文件的名字，lbl 是一个标签图的NumPy数组
            utils.lblsave(os.path.join(pngs_folder, count[i].split(".")[0] + '.png'), new)

            print('Saved ' + count[i].split(".")[0] + '.jpg and ' + count[i].split(".")[0] + '.png')

    print("classes: label:", label_name_to_value)
    # return label_name_to_value

# 现在得到了 jpgs pngs
# 现在划分数据集
"""
/raw_data/   
    +----/jpgs/ 
    +----/pngs/
    +----/jsons/
/daatset/
    +---/train_data/
            +--------/jpgs/
            +--------/pngs/
    +---/test_data/
            +--------/jpgs/
            +--------/pngs/
"""
def split_data(jpgs_folder, pngs_folder, tr_jpgs_folder, tr_pngs_folder, te_jpgs_folder, te_pngs_folder, tr_data_num):
    import random
    import shutil
    data_names_list = [name.split('.')[0] for name in os.listdir(jpgs_folder) if name.endswith('.jpg')]
    random.shuffle(data_names_list)
    train_names_list = data_names_list[:tr_data_num]
    test_names_list = data_names_list[tr_data_num:]
    
    # shutil.copyfile()  --> get train jpgs pngs
    for name in tqdm(train_names_list):
        jpg_path = os.path.join(jpgs_folder, name+'.jpg')
        assert os.path.isfile(jpg_path), f"train {name} jpg not exists"
        jpg_path_new = os.path.join(tr_jpgs_folder, name+'.jpg')
        shutil.copyfile(jpg_path, jpg_path_new)

        png_path = os.path.join(pngs_folder, name+'.png')
        assert os.path.isfile(jpg_path), f"train {name} png not exists"
        png_path_new = os.path.join(tr_pngs_folder, name+'.png')
        shutil.copyfile(png_path, png_path_new)

        print('Saved ' + name+ '.jpg -----and---- ' + name + '.png')


    # shutils.copyfile()  -- get test jpgs pngs
    for name in tqdm(test_names_list):
        jpg_path = os.path.join(jpgs_folder, name+'.jpg')
        assert os.path.isfile(jpg_path), f"test {name} jpg not exists"
        jpg_path_new = os.path.join(te_jpgs_folder, name+'.jpg')
        shutil.copyfile(jpg_path, jpg_path_new)

        png_path = os.path.join(pngs_folder, name+'.png')
        assert os.path.isfile(jpg_path), f"test {name} png not exists"
        png_path_new = os.path.join(te_pngs_folder, name+'.png')
        shutil.copyfile(png_path, png_path_new)

        print('Saved ' + name+ '.jpg -----and---- ' + name + '.png')


if __name__ == '__main__':
    jsons_folder = r"CV_PROJECTION\SOD_u2net\raw_data\json"
    jpgs_folder = r"CV_PROJECTION\SOD_u2net\raw_data\jpgs"
    pngs_folder = r"CV_PROJECTION\SOD_u2net\raw_data\pngs"
    classes = get_ss_classes(jsons_folder)
    # classes.append("background")
    # print(classes) # ['jyz', 'baodian'] # ["jyz", "baodian", "background"]
    # classes的先后顺序很重要， 第一个得是 "background" --黑色 (0,0,0)
    classes.insert(0, "background")
    # classes = ["background", "jyz", "baodian"]   # "_background_" 不知道为啥这样命名，但是好像没有影响

    json_to_jpgs_pngs(jsons_folder, classes, jpgs_folder, pngs_folder)  # 这里也是用了调色板的，只是是默认的而调色板palette  0-255类的一个调色模板

    # split data --get--> dataset
    tr_jpgs_foler = r"CV_PROJECTION\SOD_u2net\dataset\train_data\jpgs"
    os.makedirs(tr_jpgs_foler, exist_ok=True)
    tr_pngs_folder = r"CV_PROJECTION\SOD_u2net\dataset\train_data\pngs"
    os.makedirs(tr_pngs_folder, exist_ok=True)
    te_jpgs_folder = r"CV_PROJECTION\SOD_u2net\dataset\test_data\jpgs"
    os.makedirs(te_jpgs_folder, exist_ok=True)
    te_pngs_folder = r"CV_PROJECTION\SOD_u2net\dataset\test_data\pngs"
    os.makedirs(te_pngs_folder, exist_ok=True)

    tr_data_num = 400
    split_data(jpgs_folder, pngs_folder, tr_jpgs_foler, tr_pngs_folder, te_jpgs_folder, te_pngs_folder, tr_data_num)







