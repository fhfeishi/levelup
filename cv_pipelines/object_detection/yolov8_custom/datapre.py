import os, random, shutil
import xml.etree.ElementTree as ET
from xml.dom import minidom

import torch.nn as nn

nn.ReLU()

def get_classes(cls_path):
    with open(cls_path, 'r', encoding='utf-8') as f:
        class_names = f.readlines()
        class_names = [x.strip() for x in class_names]
        return class_names, len(class_names)

# 根据bbox-cls  writeLines: data_filename
def get_BboxClsTxt(data_root, target_dir, classes_path,
                   data_struct={'image': 'images', "label": 'Annotations'}):

    xml_dir = os.path.join(data_root, data_struct['label'])

    def get_xml_objs(xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        objects = root.findall('object')
        return objects

    classes, _ = get_classes(classes_path)
    dict_cls2fname = {cls: set() for cls in classes}

    for file in os.listdir(xml_dir):
        if file.endswith('.xml'):
            xmlpath = os.path.join(xml_dir, file)
            objects = get_xml_objs(xmlpath)

            filename_ = file.split('.')[0]

            for obj in objects:
                name_tag = obj.find('name')
                if name_tag is not None:
                    class_name = name_tag.text.strip()
                    if class_name in classes:
                        dict_cls2fname[class_name].add(filename_)

    # write to taregt_dir/class_name.txt
    for cls in classes:
        txt_filename = f"{cls}.txt"
        txt_path = os.path.join(target_dir, txt_filename)
        with open(txt_path, 'w', encoding='utf-8') as f:
            for data_fname in dict_cls2fname[cls]:
                f.write(f"{data_fname}\n")

# reduplicated txtline
def unique_txtlines(txtpath, output_path=None):
    if output_path is None:
        output_path = txtpath

    unique_lines = []
    seen = set()

    with open(txtpath, 'r', encoding='utf-8') as infile:
        for line in infile:
            stripped_line = line.strip()
            if stripped_line and stripped_line not in seen:
                unique_lines.append(stripped_line)
                seen.add(stripped_line)

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for line in unique_lines:
            outfile.write(f"{line}\n")

    # 直接转集合也会自动去重
    return set(unique_lines)

def get_lines(target_dir, nameset, txtname):
    testline_path = os.path.join(target_dir, txtname)
    with open(testline_path, 'w', encoding='utf-8') as f:
        for name in nameset:
            f.write(f"{name}/n")

# 现在得到了每个bbox类别的 data_lines.txt   ---当然一个data可能包含多个
# testlines.txt
# -> set()
def sample_txtlines(txtpath, sample_size):
    lines = []
    with open(txtpath, 'r', encoding='utf-8') as infile:
        for line in infile:
            lines.append(line.strip())
    random.shuffle(lines)
    return set(lines[:sample_size])

def get_testline(target_dir, nameset):
    testline_path = os.path.join(target_dir, 'testline.txt')
    with open(testline_path, 'w', encoding='utf-8') as f:
        for name in nameset:
            f.write(f"{name}\n")

def get_line(target_dir, nameset, txtname):
    txtpth = os.path.join(target_dir, txtname)
    with open(txtpth, 'w', encoding='utf-8') as f:
        for name in nameset:
            f.write(f"{name}\n")


# 得到目标数据集，  test上指标要够
def get_test(test_line, image_dir, xml_dir, target_root, 
             struct_ = {'train': 'trainval/train', 'val': 'trainval/val', 'test': 'test'}):
    
    image_suffix = '.jpg'
    xml_suffix = '.xml'
    
    test_imgdir = os.path.join(target_root, f"{struct_['test']}/images")
    test_lbldir = os.path.join(target_root, f"{struct_['test']}/labels")
    os.makedirs(test_imgdir, exist_ok=True)
    os.makedirs(test_lbldir, exist_ok=True)
    
    with open(test_line, 'r', encoding='utf-8') as infile:
        for line in infile:
            name = line.strip()
            img_path = os.path.join(image_dir, name+image_suffix)
            lbl_path = os.path.join(xml_dir, name+xml_suffix)
            if os.path.exists(img_path) and os.path.exists(lbl_path):
                img_path_save = os.path.join(test_imgdir, name+image_suffix)
                shutil.copy2(img_path, img_path_save)
                lbl_path_save = os.path.join(test_lbldir, name+xml_suffix)
                shutil.copy2(lbl_path, lbl_path_save)
            else:
                print(f"{name=}")
    
def get_dataline(datadir, target_dir):
    dataline_path = os.path.join(target_dir, 'dataline.txt')
    nameset = set()
    with open(dataline_path, 'w', encoding='utf-8') as infile:
        for f in os.listdir(datadir):
            if f.endswith('.xml', '.jpg'):
                name_ = os.path.splitext(f)[0]
                nameset.add(name_)  
                infile.write(f"{name_}\n")
        return nameset
    
# 余下的数据分到trainval里面
def get_trainvalline():
    pass 
def get_trainval():
    pass 

def rename_BboxCls_inplace(xml_dir, save_dir=None, rename_dict={"呼吸器-硅胶变色": "hxq_bianse",
                                                    "呼吸器-硅胶疑似变色": "hxq_yisi",
                                                    "带铁壳的呼吸器-疑似变色": "tk_yisi",
                                                    "呼吸器-硅胶正常": "hxq_zhengchang",
                                                    "带铁壳的呼吸器-变色": "tk_bianse",
                                                    "带铁壳的呼吸器-正常": "tk_zhengchang",
                                                    }):
      
    if save_dir is None:
        save_dir = xml_dir
    else:
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        save_dir = save_dir
    # 获取所有 xml 文件名（不带后缀）
    xml_list = [f.split('.')[0] for f in os.listdir(xml_dir) if f.endswith('.xml')]

    for file in xml_list:
        # 解析 xml 文件
        file_path = os.path.join(xml_dir, f"{file}.xml")
        tree = ET.parse(file_path)
        root = tree.getroot()

        # 遍历 xml 中的所有标签，修改对应类别名称
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name in rename_dict:
                obj.find('name').text = rename_dict[name]

        # 使用 minidom 格式化 XML
        xml_str = ET.tostring(root, encoding="utf-8").decode("utf-8")
        pretty_xml_str = minidom.parseString(xml_str).toprettyxml(indent="  ")

        # 去掉多余的空白行
        lines = pretty_xml_str.splitlines()
        lines = [line for line in lines if line.strip()]  # 移除空行

        # 合并并写入文件
        pretty_xml_str = "\n".join(lines)

        # 保存修改后的 xml 到新目录
        save_path = os.path.join(save_dir, f"{file}.xml")
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(pretty_xml_str)



def get_test_set(nametxt, image_root, label_root, taregt_root, 
                 struct_={'image': 'images', 'label': 'labels'}):
    datalist = []
    with open(nametxt, 'r', encoding='utf-8') as infile:
        for line in infile:
            name = line.strip()
            datalist.append(name)
    image_dist = os.path.join(taregt_root, struct_['image'])
    label_dist = os.path.join(taregt_root, struct_['label'])
    os.makedirs(image_dist, exist_ok=True)
    os.makedirs(label_dist, exist_ok=True)
    for name in datalist:
        image_src = os.path.join(image_root, name+'.jpg')
        # label_src = os.path.join(label_root, name+'.xml')
        label_src = os.path.join(label_root, name+'.png')
        if os.path.exists(image_src) and os.path.exists(label_src):
            image_save = os.path.join(image_dist, name+'.jpg')
            # label_save = os.path.join(label_dist, name+'.xml')
            label_save = os.path.join(label_dist, name+'.png')
            shutil.move(image_src, image_save)
            shutil.move(label_src, label_save)
        else:
            print("error filename")
            
def get_trainval_set(nametxt, image_root, label_root, taregt_root, 
                 struct_={'image': 'images', 'label': 'labels'}):
    datalist = []
    with open(nametxt, 'r', encoding='utf-8') as infile:
        for line in infile:
            name = line.strip()
            datalist.append(name)
    image_dist = os.path.join(taregt_root, struct_['image'])
    label_dist = os.path.join(taregt_root, struct_['label'])
    os.makedirs(image_dist, exist_ok=True)
    os.makedirs(label_dist, exist_ok=True)
    for name in datalist:
        image_src = os.path.join(image_root, name+'.jpg')
        label_src = os.path.join(label_root, name+'.png') # png
        if os.path.exists(image_src) and os.path.exists(label_src):
            image_save = os.path.join(image_dist, name+'.jpg')
            label_save = os.path.join(label_dist, name+'.png') # png xml
            shutil.copy2(image_src, image_save)
            shutil.copy2(label_src, label_save)
        else:
            print('error filename')


if __name__ == '__main__':

    # # # --- get BboxCls.txt --- #
    # data_ = r"F:\luoshuan\zyanshou\code_projection\yolov8_luoshuan\VOCdevkit\VOC2007"
    # classes_path = r"F:\luoshuan\zyanshou\code_projection\yolov8_luoshuan\model_data\luoshuan_classes.txt"
    # target_d = r"F:\luoshuan\zyanshou\dataset"
    # get_BboxClsTxt(data_, target_dir, classes_path)

    # # # ---- get test line ----- #
    # def get_testlines(teseline_dir):

    #     cls_a = f"{testline_d}/class_a.txt"  
    #     set1 = sample_txtlines(cls_a, 0)
    #     get_line(testline_d, set1, 'set1.txt')

    #     cls_b = f"{testline_d}/class_b.txt" 
    #     set2 = sample_txtlines(cls_b, 0)
    #     get_line(testline_d, set2, 'set2.txt')

    #     cls_c = f"{testline_d}/class_c.txt" 
    #     set3 = sample_txtlines(cls_c, 10)
    #     get_line(testline_d, set3, 'set3.txt')

    #     cls_d = f"{testline_d}/class_d.txt" 
    #     set4 = sample_txtlines(cls_d, 50)
    #     get_line(testline_d, set4, 'set4.txt')

    #     cls_e = f"{testline_d}/class_e.txt" 
    #     set5 = sample_txtlines(cls_e, 50)
    #     get_line(testline_d, set5, 'set5.txt')

    #     cls_f = f"{testline_d}/class_f.txt" 
    #     set6 = sample_txtlines(cls_f, 40)
    #     get_line(testline_d, set6, 'set6.txt')

    #     nameset = set1.union(set2, set3, set4, set5, set6)
    #     teseline_path = os.path.join(teseline_dir, 'testline.txt')
    #     with open(teseline_path, 'w', encoding='utf-8') as infile:
    #         for name in nameset:
    #             infile.write(f"{name}\n")
    
    # testline_d = r"F:\luoshuan\zyanshou\dataset"

    # get_testlines(testline_d)

    # target_root = r"F:\luoshuan\zyanshou\dataset"
    # test_line = r"F:\luoshuan\zyanshou\dataset\testline.txt"
    # img_d = r"F:\luoshuan\zyanshou\code_projection\yolov8_luoshuan\VOCdevkit\VOC2007\JPEGImages"
    # lbl_d = r"F:\luoshuan\zyanshou\code_projection\yolov8_luoshuan\VOCdevkit\VOC2007\Annotations"
    
    # get_test(test_line=test_line, image_dir=img_d, xml_dir=lbl_d, target_root=target_root)

    # ---------------------------------------------# 
    # ## get Zdataset: test,  trai、val
    data_root = r"F:\jueyuanzi\zyanshou\code_projection\jueyuanzi\VOCdevkit\VOC2007"
    zdata = r"F:\jueyuanzi\zyanshou\dataset"
    txtpath = os.path.join(zdata, 'test_set.txt') 
    imaged = os.path.join(data_root, "JPEGImages")
    xmld = os.path.join(data_root, "Annotations")
    pngd = os.path.join(data_root, "SegmentationClass")
    test_root = os.path.join(zdata, 'test')  

    # # get_test_set(txtpath, imaged, xmld, test_root)
    # get_test_set(txtpath, imaged, pngd, test_root)
    
    # -- get train 
    traintxt = os.path.join(data_root, 'ImageSets/Segmentation/train.txt')   # Segmentation # Main
    tr_root =  os.path.join(zdata, 'trainval/train') 
    # get_trainval_set(traintxt, imaged, xmld, tr_root)
    get_trainval_set(traintxt, imaged, pngd, tr_root)
    # -- get val
    valtxt = os.path.join(data_root, 'ImageSets/Segmentation/val.txt')    # Segmentation # Main
    val_root = os.path.join(zdata, 'trainval/val') 
    # get_trainval_set(valtxt, imaged, xmld, val_root)
    get_trainval_set(valtxt, imaged, pngd, val_root)