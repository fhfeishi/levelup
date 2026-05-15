# mindspore 标准格式数据集
import os, tqdm, cv2
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore.mindrecord import FileWriter

class zvocdatasetz():
    def __init__(self, tgt_imshape=None, ds_name=None, tsk='detection', 
                 label_list=None, num_classes=None, annotation_lines_dir=None, 
                 set_name="train", transforms=None, subdir="VOC2012"):

        self.tgt_imshape = tgt_imshape if tgt_imshape is not None else [224, 224]
        self.set_name = set_name.strip().lower()
        self.tsk     = tsk.strip().lower()
        assert self.tsk in ['detection', 'segmentation'], f"task {tsk} is not supported."
        assert self.set_name in ['train', 'val', 'test'], f"set_name {self.set_name} is not supported."
        if self.tsk == 'detection' and label_list is None:raise ValueError("label_list must be provided for detection task")
        if self.tsk == 'segmentation' and num_classes is None:raise ValueError("num_classes must be provided for segmentation task")

        self.ds_name = ds_name if ds_name is not None else "VOCDevkit"
        self.srcd = self.ds_name
        if not os.path.isdir(self.srcd): raise ValueError("Dataset not found at {}".format(self.srcd))
        self.subd = subdir
        self.src_root_struct = {
            "img_dir" : f"{self.subd}/JPEGImages",
            # "sets_dir": f"{self.subd}/{self.task[0].upper()+self.task[1:]}",
            "sets_dir": f"{self.subd}/ImageSets/{self.tsk.capitalize()}",
            "lbl_dir" : f"{self.subd}/{'Annotations' if self.tsk == 'detection' else 'SegmentationClass'}",
        }
        self.imd   = os.path.join(self.srcd, self.src_root_struct['img_dir'])
        self.setsd = os.path.join(self.srcd, self.src_root_struct['sets_dir'])
        self.lbld  = os.path.join(self.srcd, self.src_root_struct['lbl_dir'])
        # zvocdatasetz structure
        #  self.ds_name  (= self.srcd)
        #      └─ self.subd
        #             ├─ self.setsd
        #             ├─ self.imgd
        #             └─ self.lbld

        # from dataset.txt get data_stems_list
        splitfile = os.path.join(self.setsd, f"{self.set_name}.txt")
        with open(splitfile, "r", encoding='utf-8') as f:
            self.stems = [lin.strip() for lin in f.readlines()]

        self.annolinesd = annotation_lines_dir
        if self.tsk == 'detection':
            self.labelt = label_list
            self.lbl_   = ".xml"
            if self.annolinesd:
                self.annolinesd = annotation_lines_dir
                if not os.path.isdir(self.annolinesd): raise ValueError(f"Annotation lines dir not found at {self.annolinesd}")
                self.annolines = self.parse_annolines()
            else:
                self.annolines = self.stems
        else:
            # segmentation task
            self.num_classes = num_classes
            self.lbl_ = ".png"
            self.annolines = self.stems

        self.transforms = transforms
        self.stemslen = len(self.stems)

    def __len__(self):
        return self.stemslen

    def __getitem__(self, index):
        if self.tsk == 'detection':
            if self.annolinesd:
                image_path, label = self.parse_annolines(index=index)
                image = Image.open(image_path).convert('RGB')
            else:
                image = Image.open(os.path.join(self.imd, self.annolines[index] + '.jpg')).convert('RGB')
                # np.array([x,y,x,y,clsid], ...)
                label = self.get_xml_annoslst(xml_path=os.path.join(self.lbld, self.annolines[index] + self.lbl_),
                                           label_list=self.labelt,
                                           output_label_type='xyxy')
            if self.transforms is not None:
                image, label = self.transforms(image, label, self.tsk, )
        else:
            image = Image.open(os.path.join(self.imd, self.annolines[index] + '.jpg')).convert('RGB')
            label = Image.open(os.path.join(self.lbld, self.annolines[index] + self.lbl_))

            if self.transforms is not None:
                image, label = self.transforms(image, label, self.tsk, self)

        return np.array(image).astype(np.uint8), np.array(label).astype(np.uint8)

    def parse_annolines(self, index=None):
        if self.set_name != 'test':
            f = open(f"{self.annolinesd}/{self.set_name}_lines.txt", "r", encoding='utf-8')
            if f is None: raise ValueError(f"Annotation lines file {self.annolinesd}/{self.set_name}_lines.txt not found.")
            else:
                impath_lst, annolines_list= [x.strip().split(" ", 1) for x in f.readlines()]
                if index  is None:
                    return impath_lst, annolines_list
                else:
                    return impath_lst[index], annolines_list[index]

    def _get_samples_path_lst(self, index):
        if self.tsk == 'detection':
            if self.annolinesd:
                image_path, label = self.parse_annolines(index=index)
            else:
                image_path = os.path.join(self.imd, self.annolines[index] + self.lbl_)
                # np.array([x,y,x,y,clsid], ...)
                label = self.get_xml_annoslst(xml_path=os.path.join(self.lbld, self.annolines[index] + self.lbl_),
                                               label_list=self.labelt,
                                               output_label_type='xyxy')
            return [image_path, label]
        else:
            image_path = os.path.join(self.imd, self.annolines[index] + '.jpg')
            label_path = os.path.join(self.lbld, self.annolines[index] + self.lbl_)
            # print(f"{[image_path].append(label_path) = } ")  # None
            return [image_path, label_path]

    def get_mindrecord_dataset(self, mrds_name, num_shards=1, maxdatalength=1000, ds_index=None, shuffle=True):
        mrdsave_dir = os.path.join(mrds_name, f"{self.subd}/{self.set_name}")
        os.makedirs(mrdsave_dir, exist_ok=True)
        indexs_list = list(range(self.stemslen))
        if shuffle:
            np.random.shuffle(indexs_list)
        print('creating mindrecord dataset...')
        # in-params filename: path/to/mindrecord.mindrecord
        mindrecord_savep = os.path.normpath(os.path.join(mrdsave_dir, 'mindrecord.mindrecord'))

        # 定义schema
        schema = {"file_name": {"type": "string"},
                  "label"    : {"type": "bytes" if self.tsk == 'segmentation' else "int32"},
                  "data"     : {"type": "bytes"},}

        # 初始化Writer
        writer = FileWriter(file_name=mindrecord_savep, shard_num=num_shards, overwrite=True)
        writer.add_schema(schema, f"tsk:{self.tsk if self.tsk == 'detection' else 'semantic segmentation'} mindrecord dataset")
        # # 索引字段（加速数据检索）  # int32/float/string
        # indexes = ["file_name", "tsk"]  if ds_index is None  else ds_index
        # writer.add_index(indexes)  # 加入设置的索引字段

        # 构建数据样本
        datas = []
        counts = 0
        for l in tqdm.tqdm(indexs_list):
            [impath_list, labels_list] = self._get_samples_path_lst(index=l)
            # 读取图像数据
            if   isinstance(impath_list, str)   :  im_ = impath_list
            elif isinstance(impath_list, list):  im_ = str(impath_list)
            else:
                print(f"{type(impath_list) = } 为什么不是  str(impath) or [str(impath] 呢？")
                raise ValueError(f"{impath_list = }  is nusupport")
            with open(im_, 'rb') as f: img_bytes = f.read()

            # 读取标签数据
            if isinstance(labels_list, str) and self.tsk == 'segmentation':
                label_data = labels_list  # msk
            elif (isinstance(impath_list, list) and len(labels_list[0]) == 5 and self.tsk == 'detection'):
                with open(labels_list, 'rb') as f:  label_data = f.read()
            else:
                print(f"{labels_list[0] = } 为什么不是 xyxyc or cxywh ?")
                raise ValueError(f"{labels_list[0] = }  is nusupport")

            # 写入字段 align -> schema
            sample_ = {
                # "file_name": impath_list.split('/')[-1]
                "file_name": os.path.basename(im_),
                "data": img_bytes,
                "label": label_data,}
            datas.append(sample_)
            counts += 1
            if counts % maxdatalength == 0:
                writer.write_raw_data(datas)
                print('number of samples written:', counts)
                datas = []
        # # 写入剩余数据
        # if datas:
        #     writer.write_raw_data(datas)
        writer.commit()
        print('number of samples written:', counts)
        print('Create Mindrecord Done.')

    @staticmethod
    def get_random_data(tgt_shape, image: np.ndarray, label: np.ndarray, task='detection', set='train', num_classes=None):
        pass
        return image, label

    @staticmethod
    def resize_sample(stem: str, image: np.array, label: np.array):
        resize_dict = {"data_stem": stem, "pad": True, "scale": "scale",
                       "pading": ["pad_left", "pad_right", "pad_top", "pad_bottom", "pad_value"]}
        return image, label, resize_dict

    @staticmethod
    def get_xml_annoslst(xml_path, label_list, output_label_type="xyxy"):
        """解析XML标注文件"""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        annos_list = []
        for obj in root.iter('object'):
            # 获取类别名称
            cls_name = obj.find('name').text
            # 获取边界框坐标（相对坐标）
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            if output_label_type == 'yolo':
                # np.array([cls_id, xc_n, yc_n, w_n, y_n],...)
                xc_n = (xmin + xmax) / 2 / width
                yc_n = (ymin + ymax) / 2 / height
                w_n = (xmax - xmin) / width
                h_n = (ymax - ymin) / height
                cls_id = label_list.index(cls_name)
                annos_list.append([cls_id, round(xc_n, 6), round(yc_n, 6), round(w_n, ), round(h_n, 6)])
            else:
                # np.array([x, y, x, y, cls_id], ...)
                cls_id = label_list.index(cls_name)
                annos_list.append([xmin, ymin, xmax, ymax, cls_id])
        return annos_list

if __name__ == '__main__':
    """
    # data.mindrecord 数据集  ds-name.mindrecord
    
    ## 索引 
    索引的核心作用:
    1. 加速数据查询
     - 场景示例：当需要根据特定字段（如file_name或task）筛选数据时
     - 原理：索引会预先建立字段值的快速查找结构（如B+树），避免全量扫描
     - 性能提升：查询耗时从 O(n) 降低到 O(log n) 甚至 O(1)
    2. 优化分布式训练
     - 场景示例：多卡训练时按文件名快速定位分片数据
     - 原理：索引帮助快速定位数据在文件中的物理位置
     - 性能提升：减少数据分片时的I/O开销
    3. 支持高效数据过滤
     - 场景示例：在混合数据集中筛选特定任务类型（如只读取检测任务）
     - 原理：通过tsk字段索引快速过滤数据
     - 性能提升：过滤操作速度提升10倍以上
    
    索引的典型应用场景：
     - 场景1：按文件名快速定位样本
        - # 无索引时：全量扫描
            dataset = ds.MindDataset("data.mindrecord")
            sample = dataset.filter(lambda x: x["file_name"] == "0001.jpg")
        - # 有索引时：直接定位
            # （底层自动使用索引加速）
     - 场景2：按任务类型批量加载数据
        # 仅加载检测任务数据
        dataset = ds.MindDataset("data.mindrecord")
        detection_data = dataset.filter(lambda x: x["tsk"] == "detection")
     - 场景3：联合条件查询
        # 查找检测任务中标签为"person"的样本
        dataset.filter(lambda x: (x["tsk"] == "detection") & (x["label_meta"] == 0))
    
    ## 索引的底层实现原理
    MindRecord采用 列式存储 + 索引结构 的设计：
        data.mindrecord
        ├── data/          # 实际数据块
        ├── index/         # 索引数据
        │   ├── file_name.idx
        │   └── tsk.idx
        └── schema.json    # 元数据描述
        
        
    ## 索引使用注意事项
        字段类型限制：仅支持 基本类型（int32/float/string）
        存储开销：索引会增加约 5-15% 的存储空间
        写入性能：添加索引会略微降低数据写入速度
    """

    train_ds  = zvocdatasetz(
        ds_name='jyz_voc',
        tsk = 'segmentation',
        num_classes=3,
        set_name='train',
        subdir='VOC2012',
    )
    train_ds.get_mindrecord_dataset(mrds_name="mrdataset")

    val_ds = zvocdatasetz(
        ds_name='jyz_voc',
        tsk='segmentation',
        num_classes=3,
        set_name='val',
        subdir='VOC2012',
    )
    val_ds.get_mindrecord_dataset(mrds_name="mrdataset")

    test_ds = zvocdatasetz(
        ds_name='jyz_voc',
        tsk='segmentation',
        num_classes=3,
        set_name='test',
        subdir='VOC2012',
    )
    test_ds.get_mindrecord_dataset(mrds_name="mrdataset")
