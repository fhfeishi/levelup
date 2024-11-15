# 数据集预处理，
# normal：fileName、dataMatch、dataType、dataSampleMove、dataInfo 
  
# od dataset：datasetCrop、labelProcess、getODDetaset

import os, shutil, random
from collections import defaultdict
import xml.etree.ElementTree as ET 
import xml.dom.minidom
from PIL import Image
import uuid

def norm_filename(image_dir, label_dir, img_suffix='.jpg', lbl_suffix='.xml', rename_mode = None, image_save=None, label_save=None) -> None:
    
    if image_save is not None:
        os.makedirs(image_save, exist_ok=True)
    if label_save is not None:
        os.makedirs(label_save, exist_ok=True)
    
    im_name = {f.split('.')[0] for f in os.listdir(image_dir) if f.endswith(img_suffix)}
    lb_name = {f.split('.')[0] for f in os.listdir(label_dir) if f.endswith(lbl_suffix)}
    assert (im_name-lb_name) is {}, 'data not match!'

    ends_ = 0
    for name in list(im_name):
        
        image_path = os.path.join(image_dir, name+img_suffix)
        label_path = os.path.join(label_dir, name+lbl_suffix) 
        
        # rename
        if rename_mode == 'uuid':
            name_new = str(uuid.uuid4()) 
        # string + number
        elif rename_mode == 'dirname':
            name_new = f'{image_dir}_{ends_}'
        elif rename_mode is None:
            # rename inplace: filerename
            pass
        else:
            name_new = f'{rename_mode}_{ends_}'
                  
        if image_save is None:
            image_path_save = os.path.join(image_dir, name_new+img_suffix)
            # rename - inplace
            image = Image.open(image_path).convert('RGB')
            image.save(image_path_save)
            if image_path != image_path_save:
                os.remove(image_path)
        else:
            # copy rename to newdir
            image_path_save = os.path.join(image_save, name_new+img_suffix)
            shutil.copy(image_path, image_path_save)
        
        if label_save is None:
            label_path_save = norm_xml(label_path)    
            # rename
        else:
            label_path_save = os.path.join(label_save, name_new+lbl_suffix)
            # copy rename to newdir
            shutil.copyfile(label_path, label_path_save)
               
        ends_ += 1

def norm_xml(xml_lbl_path):
    # norm in place
    
    return xml_lbl_path   



def txt2xmlLBL(xml_file_path):
    # norm txt to norm xml
    if xml_file_path.endswith('.xml'):
        
        pass


def xml2txtLBL(txt_lbl_path):
    # norm xml to norm txt
    if txt_lbl_path.endswith('.txt'):
        
        pass



class getODDataset():
    def __init__(self, rawDatasetStruct=None, targetDatasetStruct=None):
        
        if rawDatasetStruct:
            self.raw_xml_dir = rawDatasetStruct["image"]
            self.raw_jpg_dir = rawDatasetStruct["label"]
        
        if targetDatasetStruct:
            self.target_xml_dir = targetDatasetStruct["image"]
            self.target_jpg_dir = targetDatasetStruct["label"]
    
    def xmls_info(self, xml_dir, dataset_txt=None):
        
        class_count = defaultdict(int)
        resolution_count = defaultdict(int)
        bbox_size_count = defaultdict(int)

        for xml_file in os.listdir(xml_dir):
            if xml_file.endswith('.xml'):
                path = os.path.join(xml_dir, xml_file)
                tree = ET.parse(path)
                root = tree.getroot()
                width = root.find('size/width').text
                height = root.find('size/height').text
                resolution = f'{width}x{height}'
                resolution_count[resolution] += 1

                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    class_count[class_name] += 1
                    bndbox = obj.find('bndbox')
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)
                    bbox_width = xmax - xmin
                    bbox_height = ymax - ymin
                    bbox_size = f'{bbox_width}x{bbox_height}'
                    bbox_size_count[bbox_size] += 1

        print("类别及其数量:")
        for cls, count in class_count.items():
            print(f'{cls}: {count}')
        print("\n分辨率及其图像数量:")
        for res, count in resolution_count.items():
            # print(f'{res}: {count}')
            continue
        print("\n标注框尺寸及其数量:")
        for size, count in bbox_size_count.items():
            # print(f'{size}: {count}')
            continue

        if dataset_txt is not None:
            with open(dataset_txt, 'w', encoding='utf-8') as file:
                file.write("类别及其数量:\n")
                for cls, count in class_count.items():
                    file.write(f'{cls}: {count}\n')
                file.write("\n分辨率及其图像数量:\n")
                for res, count in resolution_count.items():
                    file.write(f'{res}: {count}\n')
                file.write("\n标注框尺寸及其数量:\n")
                for size, count in bbox_size_count.items():
                    file.write(f'{size}: {count}\n')
    
    def del_xmlLabel(self, xml_dir, save_label_list=None, del_label_list=None, save_dir=None):
        
        if save_label_list is None and del_label_list is None:
            print("未指定任何标签进行保留或删除。")
            return
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for xm in os.listdir(xml_dir):
            if xm.endswith('.xml'):
                xml_path = os.path.join(xml_dir, xm)
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                # modified = False
                modified = 1
                
                for obj in root.findall('object'):
                    label = obj.find('name').text
                    remove = False
                    
                    if save_label_list is not None and label not in save_label_list:
                        remove = True
                    if del_label_list is not None and label in del_label_list:
                        remove = True
                    
                    if remove:
                        root.remove(obj)
                        modified = True
            
                if len(root.findall('object')) == 0:
                    continue 
            
                if modified:
                    if save_dir is not None:
                        save_path = os.path.join(save_dir, xm)
                    else:
                        save_path = xml_path
                    
                    try:
                        rough_string = ET.tostring(root, 'utf-8')
                        reparsed = xml.dom.minidom.parseString(rough_string)
                        pretty_xml = reparsed.toprettyxml(indent="  ")
                        # 去除多余的空行
                        pretty_xml = '\n'.join([line for line in pretty_xml.split('\n') if line.strip()])

                        with open(save_path, 'w', encoding='utf-8') as f:
                            f.write(pretty_xml)
                    except Exception as e:
                        print(f"保存XML文件时出错：{save_path}，错误信息：{e}")
      
    def norm_ImageSuffix(self, image_dir, save_dir, target_suffix='.jpg'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif']
            
        for im in os.listdir(image_dir):
            ext = os.path.splitext(im)[1].lower()
            if ext in supported_formats: # JPG jpg
                im_path = os.path.join(image_dir, im)
                im_path_save = os.path.join(save_dir, im.split('.')[0]+target_suffix)

                try:
                    with Image.open(im_path) as img:
                        img = img.convert('RGB')
                        img.save(im_path_save)
                except Exception as e:
                    print(f"处理图片时出错：{im_path}，错误信息：{e}")
      
    def del_not_label_data(self, image_dir, xml_dir):
        
        deleted_xml_files = []
        deleted_image_files = []
        total_xml_files = 0
        
        for xml_file in os.listdir(xml_dir):
            # xml is empty
            if xml_file.endswith('.xml'):
                total_xml_files += 1
                xml_path = os.path.join(xml_dir, xml_file)
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    objects = root.findall('object')
                    if not objects:
                        os.remove(xml_path)
                        deleted_xml_files.append(xml_file)
                        
                        base_name = os.path.splitext(xml_file)[0]
                        img_extensions = '.jpg'
                        image_deleted = False
                        imp_file = base_name + img_extensions
                        img_path = os.path.join(img_dir, imp_file)
                        if os.path.isfile(img_path):
                            os.remove(img_path)
                            deleted_image_files.append(imp_file)
                            image_deleted = True
                            
                except ET.ParseError:
                    # also can remove the errorXml and -jpg
                    print(xml_file)

        # image no xml-label-file, remove ipg
        jpg_nameset = [f.split('.')[0] for f in os.listdir(image_dir)]
        xml_nameset = [f.split('.')[0] for f in os.listdir(xml_dir)]
        for im in jpg_nameset:
            if im not in xml_nameset:
                im_path = os.path.join(image_dir, im + '.jpg')
                os.remove(im_path)
        
    def check_odDataset(self, image_dir, xml_dir, stamp_=True, delete_=False, move_root = None):
        img_list = [f.split('.')[0] for f in os.listdir(image_dir) if f.endswith('.jpg')]
        lbl_list = [f.split('.')[0] for f in os.listdir(xml_dir) if f.endswith('.xml')]
        
        unmatched_images = set(img_list) - set(lbl_list)
        unmatched_labels = set(lbl_list) - set(img_list)
        
        if stamp_:
            if unmatched_images or unmatched_labels:
                print("Unmatched images:", unmatched_images)
                print("Unmatched labels:", unmatched_labels)
            else:
                print("All images and labels match.")
            
        if delete_:
            # delete  jpg which is no xml
            for im in list(unmatched_images):
                imp = os.path.join(image_dir, im+'.jpg')
                if os.path.isfile(imp):
                    os.remove(imp)
                else:
                    print(f"cannnot remove notExists file {imp}")
            # delete xml which is no jpg
            for xm in list(unmatched_labels):
                xmp = os.path.join(xml_dir, xm+'.xml')
                if os.path.exists(xmp):
                    os.remove(xmp)
                else:
                    print(f"cannnot remove notExists file {xmp}")
            
        if move_root is not None:
            assert isinstance(move_root, str), f"{move_root} 得是一个输出文件夹"
            if not os.path.exists(move_root):
                os.makedirs(move_root, exist_ok=True)
            
            img_dir = os.path.join(move_root, "images")
            lbl_dir = os.path.join(move_root, "labels")
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)
            
            for name in img_list:
                if name in lbl_list:
                    impth = os.path.join(image_dir, name+'.jpg')
                    impth_move = os.path.join(img_dir, name+'.jpg')
                    if os.path.isfile(impth):
                        shutil.move(impth, impth_move)
                        xmpth = os.path.join(xml_dir, name+'.xml')
                        xmpth_move = os.path.join(lbl_dir, name+'.xml')
                        if os.path.isfile(xmpth):
                            shutil.move(xmpth, xmpth_move)
                        else:
                            print(f"error xml file {name}.xml")
                    else:
                        print(f"error image file {name}.jpg")
        
        return unmatched_images, unmatched_labels    
       
            
# ss dataset：datasetCrop、maskProcess、getSSDataset、
class ssDataset():
    pass


if __name__ == '__main__':
    
    xml_dir = r"F:\shenlouyou\data\sly_wurenji_data"
    
    data_ = getODDataset()
    # data_.xmls_info(xml_dir, "sly_wurenji_data_datainfo.txt")
    
    # # data_.del_xmlLabel(xml_dir, ["新油", "旧油"], save_dir=r"F:\shenlouyou\data\rawdataset\xmls")
    # save_dir=r"F:\shenlouyou\data\rawdataset\xmls"
    # data_.xmls_info(save_dir, "rawdataset_info.txt")

    img_dir = r"F:\shenlouyou\data\sly_wurenji_data"
    savedir = r"F:\shenlouyou\data\rawdataset\jpgs"
    data_.norm_ImageSuffix(img_dir, savedir)
    
    
    
    