# labelme语义分割标注json转png
import base64
import json
import os
import numpy as np
from PIL import Image
from labelme import utils   # labelme==3.16.7

if __name__ == '__main__':
    jpgs_path = r"path\to\jpgs"
    jsons_path = r"path\to\jsons"
    pngs_path = r"path\to\pngs"  # save_folder

    classes = ["_background_", "jyz", "baodian"]

    count = os.listdir(jsons_path)
    for i in range(0, len(count)):
        path = os.path.join(jsons_path, count[i])

        if os.path.isfile(path) and path.endswith(".json"):
            data = json.load(open(path), encoding="utf-8")

            if data['imageData']:
                imageData = data['imageData']  # 这就是用base64编码的原jpg数据
            else:
                # 理论上来说这个path有中文、有空格是都可以的, 这句一般不运行， if data['imageData'] 判定肯定通过的。
                jpgPath = os.path.join(jpgs_path, count[i].replace(".jon", ".jpg"))
                with open(jpgPath, "rb") as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode("utf-8")

            jpg = utils.img_b64_to_arr(imageData)
            # jpg numpy.ndarray

            label_name_to_value = {'_background_': 0}
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

            lbl = utils.shapes_to_label(jpg.shape, data['shapes'], label_values)
            # ndarray: (h, w)  0, 1, 2  # ndarray:(2250,3000)

            Image.fromarray(jpg).save(os.path.join(jpgs_path, count[i].split(".")[0]+'.jpg'))

            new = np.zeros([np.shape(jpg)[0], np.shape(jpg)[1]])
            # ndarray (2250, 3000)

            for name in label_names:
                index_json = label_names.index(name)   # list.index(name) --->int: 0, 1, 2
                index_all = classes.index(name)  # list.index(name) -->int: 0, 1, 2
                new = new + index_all*(np.array(lbl == index_json)) # ndarray + ndarray

            # 将标签如转换为png图片, lblsave(filename, lbl) 接收两个参数：filename 是预期输出文件的名字，lbl 是一个标签图的NumPy数组
            utils.lblsave(os.path.join(pngs_path, count[i].split(".")[0] + '.png'), new)

            print('Saved ' + count[i].split(".")[0] + '.jpg and ' + count[i].split(".")[0] + '.png')





