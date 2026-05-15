import os, random, tqdm, shutil
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from typing import Union

# 请在在工程主目录下使用这个脚本，
# 否则,
# 需要指定工程目录 self.project_dir,  比如 path/to/your/project_dir 而不是默认的

# 数据集还没移动到指定目录 rawds_root 就不能是None
# 数据集已经移动到指定目录了 rawds_root  可以设置为None

class convert2voc_base:
    def __init__(self, tsk, bbox_list=None, seg_labels=None,rawds_root=None, project_dir=None,
                 new_training_logs_dir=None,
                 mode=0, trainval_percent=.9, train_percent=.9,
                 target_dir=None, dataset_name=None, subdir: Union[str, bool]='VOC2012'):

        self.tsk = str(tsk).strip().lower()
        assert self.tsk in ['detection', 'segmentation'], f"{tsk} must in ['detection', 'segmentation']"
        self.annomode = 2 if tsk == "segmentation" else mode
        self.subd = subdir  # subdir 为None False 就没有这个subdir了
        self.rawds_root = rawds_root  # raw_dataset_dir
        self.tgtd = target_dir
        self.ds_name = 'VOCDevkit' if dataset_name is None else str(dataset_name)  # target_dir
        self.project_dir = os.getcwd() if project_dir is None else  project_dir # 工程路径
        # 训练记录的目录，可用来存放训练用的train_lines.txt、val_lines.txt
        self.linesd = f"{self.project_dir}/train_logs"

        self.src_struct = {"root_dir": f"{self.rawds_root}",
                           "img_dir": 'jpgs',
                           "label_dir": "pngs"}
        self.tgt_struct = {"root_dir": f"{self.tgtd}/{self.ds_name}/{self.subd}" if self.subd else f"{self.tgtd}/{self.ds_name}",
                           "img_dir": "JPEGImages",
                           "xml_dir": "Annotations",
                           "png_dir": "SegmentationClass",
                           "det_set_dir": "ImageSets/Detection",
                           "seg_set_dir": "ImageSets/Segmentation"}

        self.tv_p = trainval_percent  # = trainval_ds / test_ds
        self.tr_p = train_percent  # = train_ds / val_ds
        if rawds_root is not None:
            print("dataset move start.")
        self.imgd = os.path.join(self.tgt_struct["root_dir"], self.tgt_struct['img_dir'])
        os.makedirs(self.imgd, exist_ok=True)
        # copy files
        if rawds_root is not None:
            self._copy_files(os.path.join(rawds_root, self.src_struct["img_dir"]), self.imgd)
        if tsk == "detection":
            #  mkdir f"{self.project_dir}/train_logs"
            if new_training_logs_dir is True or None: os.makedirs(self.linesd, exist_ok=True)
            elif new_training_logs_dir is False: pass
            else:
                self.linesd = f"{self.project_dir}/{new_training_logs_dir}"
                os.makedirs(self.linesd, exist_ok=True)
            self.lbld = os.path.join(self.tgt_struct["root_dir"], self.tgt_struct["xml_dir"])
            self.setsd = os.path.join(self.tgt_struct["root_dir"], self.tgt_struct["det_set_dir"])
            self.bbox_list = bbox_list
        else:
            # tsk: segmentation
            self.lbld = os.path.join(self.tgt_struct["root_dir"], self.tgt_struct["png_dir"])
            self.setsd = os.path.join(self.tgt_struct["root_dir"], self.tgt_struct["seg_set_dir"])
            # 不需要bbox_list
        os.makedirs(self.lbld, exist_ok=True)
        os.makedirs(self.setsd, exist_ok=True)
        # copy files
        if rawds_root is not None:
            self._copy_files(os.path.join(rawds_root, self.src_struct["label_dir"]), self.lbld, img=False)
        self.sets = ['train', 'val', 'test'] if self.tv_p != 1 else ['train', 'val']
        if rawds_root is not None:
            print("dataset move done.")

    def labels_process(self):
        if self.annomode == 0 or self.annomode == 1:
            self.split_data()

        if self.annomode == 0 or self.annomode == 2:
            if self.tsk == 'detection':
                print(f"Generate {self.linesd}/train_lines.txt {self.linesd}/val_lines.txt for det train start.")
                self.gen_det_trainlines()
            print(f"Generate {self.linesd}/train_lines.txt {self.linesd}/val_lines.txt for det train done.")


    def split_data(self):
        print(f"Start split data to xx.txt in {self.setsd}")
        if self.tsk == 'detection':
            labels = [x for x in os.listdir(self.lbld) if x.endswith('.xml')]
        else:
            # task segmentation
            labels = [x for x in os.listdir(self.lbld) if x.endswith('.png')]
        data_length = len(labels)
        data_index = list(range(data_length))  # [0, 1, 2, ...]
        tv = int(data_length * self.tv_p)
        tr = int(tv * self.tr_p)
        trainval_index = random.sample(data_index, tv)
        train_index = random.sample(trainval_index, tr)
        print("train_set and val_set size", tv)
        print("train_set size", tr)
        print("test_set size:", data_length - tv)
        ftrainval = open(os.path.join(self.setsd, 'trainval.txt'), 'w')
        ftest = open(os.path.join(self.setsd, 'test.txt'), 'w')
        ftrain = open(os.path.join(self.setsd, 'train.txt'), 'w')
        fval = open(os.path.join(self.setsd, 'val.txt'), 'w')
        for i in data_index:
            name = labels[i][:-4] + '\n'
            if i in trainval_index:
                ftrainval.write(name)
                if i in train_index:
                    ftrain.write(name)
                else:
                    fval.write(name)
            else:
                ftest.write(name)
        ftrainval.close()
        ftrain.close()
        fval.close()
        ftest.close()

        if self.tsk == 'segmentation':
            print("Check semantic segmentation datasets format, this may take a while.")
            print("检查数据集格式是否符合要求，这可能需要一段时间。")
            seg_lbl_nums = np.zeros([256], np.int64)  # 
            for i in tqdm.tqdm(data_index):
                name = labels[i]
                mskpath = os.path.join(self.lbld, name)
                if not os.path.exists(mskpath):
                    raise ValueError(f"未检测到标签图片{mskpath},请查看具体路径下文件是否存在以及后缀是否为png。")
                msk = np.array(Image.open(mskpath), np.int64)
                if len(np.shape(msk)) > 2:
                    if seg_lbl_nums == 2: # _background_ target
                        print(f"标签图片{name}的shape为{str(np.shape(msk))},不属于灰度图或者八位彩图,",
                              "由于分类对象只有背景和目标,如果msak只有两种颜色的话就能用(需要记住此时要处理一下标签！！！)",
                              "！！！如果mask不只两种颜色，请一定要先处理好mask！！！！")
                    else:
                        print(f"标签图片{name}的shape为{str(np.shape(msk))},不属于灰度图或者八位彩图,请仔细检查数据集格式。")
                        print("标签图片需要为灰度图或者八位彩图,标签的每个像素点的值就是这个像素点所属的种类。")
                seg_lbl_nums += np.bincount(np.reshape(msk, [-1]), minlength=256)
                if seg_lbl_nums[255] > 0 and seg_lbl_nums[0] > 0 and np.sum(seg_lbl_nums[1:255]) == 0:
                    print("检测到标签中像素点的值仅包含0与255,数据格式有误。")
                    print("二分类问题需要将标签修改为背景的像素点值为0,目标的像素点值为1。")
                elif seg_lbl_nums[0] > 0 and np.sum(seg_lbl_nums[1:]) == 0:
                    print("检测到标签中仅仅包含背景像素点,数据格式有误,请仔细检查数据集格式。")

            print("打印像素点的值与数量。")
            print('-' * 37)
            print("| %15s | %15s |" % ("label", "pixel nums"))
            print('-' * 37)
            for i in range(256):
                if seg_lbl_nums[i] > 0:
                    print("| %15s | %15s |" % (str(i), str(seg_lbl_nums[i])))
                    print('-' * 37)

            if seg_lbl_nums[255] > 0 and seg_lbl_nums[0] > 0 and np.sum(seg_lbl_nums[1:255]) == 0:
                print("检测到标签中像素点的值仅包含0与255，数据格式有误。")
                print("二分类问题需要将标签修改为背景的像素点值为0，目标的像素点值为1。")
            elif seg_lbl_nums[0] > 0 and np.sum(seg_lbl_nums[1:]) == 0:
                print("检测到标签中仅仅包含背景像素点，数据格式有误，请仔细检查数据集格式。")

        print(f"Generate xx.txt in {self.setsd} done.")

    def gen_det_trainlines(self):
        ims_cls_logs = [[0] * len(self.bbox_list)] * len(self.sets)  # 每个类别的标注框存在于多少张图片中
        lbls_cls_logs = [[0] * len(self.bbox_list)] * len(self.sets)  # 每个类别有多少个标注框
        for i, s in enumerate(self.sets):
            imgids = open(os.path.join(self.setsd, f"{s}.txt"), "r", encoding='utf-8').read().strip().split()
            if s == "test":  # 训练过程不需要生成test_lines.txt
                linefile = None
            else:
                # 'train' 'val'
                linefile = open(os.path.join(self.linesd, f"{s}_lines.txt"), "w", encoding='utf-8')

            for imid in tqdm.tqdm(imgids):
                # relative path
                impath = f"{os.path.abspath(os.path.join(self.imgd, imid + '.jpg'))}"
                # # abslute path
                # impath = f"{os.path.join(self.imgd, imid+'.jpg')}"
                if linefile is not None:
                    linefile.write(impath)
                imsize = Image.open(impath).convert('RGB').size
                ims_cls_log, lbls_cls_log = self.det_convert_anno(imid, imsize, linefile)
                ims_cls_logs[i] = [ims_cls_log[j] + ims_cls_logs[i][j] for j in range(len(ims_cls_log))]
                lbls_cls_logs[i] = [lbls_cls_log[j] + lbls_cls_logs[i][j] for j in range(len(lbls_cls_log))]
            if linefile is not None:
                linefile.close()

            print()
            print(f"{s} dataset每个标签类别包含的图像数目")
            self.print_table(self.bbox_list, ims_cls_logs[i], ['class_name', 'img_nums'])
            print()
            print(f"{s} dataset包含的每个类别的bbox标注框的数目")
            self.print_table(self.bbox_list, ims_cls_logs[i], ['class_name', 'bbox_nums'])

    def det_convert_anno(self, image_id, imsize, infile, label_out='xyxy'):
        # 处理一张图片的标注
        assert label_out in ['xyxy', 'yolo'], f"label_out must in ['xyxy', 'yolo']"
        xmlfile = os.path.join(self.lbld, f"{image_id}.xml")
        tree = ET.parse(xmlfile)
        root = tree.getroot()
        cls_id_log = []  # 统计 cls_id
        for obj in root.iter('object'):
            difficult = 0
            if obj.find('difficult') != None:
                difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in self.bbox_list or int(difficult) == 1:
                print(f"{image_id} label cls-wrong or difficule is 1. passed this ?")
                continue
            cls_id = self.bbox_list.index(cls)
            cls_id_log.append(cls_id)
            xmlbox = obj.find('bndbox')
            xmin, ymin, xmax, ymax = (int(float(xmlbox.find('xmin').text)),
                                      int(float(xmlbox.find('ymin').text)),
                                      int(float(xmlbox.find('xmax').text)),
                                      int(float(xmlbox.find('ymax').text)),)
            if label_out == 'xyxy':
                line_ = f" {xmin},{ymin},{xmax},{ymax},{str(cls_id)}"
                # line_ = " " + ','.join[str(xmin),str(ymin),str(xmax),str(ymax)]+','+str(cls_id)
            else:
                # yolo
                xc_n = (xmin + xmax) / 2 / imsize[0]
                yc_n = (ymin + ymax) / 2 / imsize[1]
                w_n = (xmax - xmin) / imsize[0]
                h_n = (ymax - ymin) / imsize[1]
                line_ = f" {str(cls_id)},{round(xc_n, 6)},{round(yc_n, 6)},{round(w_n, 6)},{round(h_n, 6)}"
            if infile is not None:
                infile.write(str(line_))
        infile.write('\n') # 行末转义字符
        ims_cls_log = [' '] * len(self.bbox_list)  # 每个个类别的标注框存在于多少张图片中
        lbls_cls_log = [0] * len(self.bbox_list)  # 每个类别有多少个标注框
        cls_idx = list(range(len(self.bbox_list)))
        for i in cls_id_log:
            if i in cls_idx:
                ims_cls_log[i] = 1
                lbls_cls_log[i] += 1

        for i, k in enumerate(ims_cls_log):
            if isinstance(k, int):
                continue
            else:
                ims_cls_log[i] = 0

        return ims_cls_log, lbls_cls_log

    def _copy_files(self, src_dir, target_dir, img=True, copy=True):
        if img:
            suffix = ".jpg"
        else:
            suffix = ".xml" if self.tsk == "detection" else ".png"
        for s in tqdm.tqdm(os.listdir(src_dir)):
            if s.endswith(suffix):
                if copy:
                    shutil.copyfile(os.path.join(src_dir, s), os.path.join(target_dir, s))
                else:
                    # move
                    shutil.move(os.path.join(src_dir, s), os.path.join(target_dir, s))
            else:
                print(s)
                print("数据后缀名不对？")

    def print_table(self, list_str_1, list_str_2, table_title, horizontal=True):

        assert len(list_str_1) == len(list_str_2), "len(list_str_1) != len(list_str_2)"
        # normal
        list_str_1 = [str(x) for x in (list_str_1)]
        list_str_2 = [str(x) for x in (list_str_2)]

        if horizontal:
            # 计算列宽
            label_width = max(len(l) for l in list_str_1 + [table_title[0]])
            num_width = max(len(n) for n in list_str_2 + [table_title[1]])

            # 构建表格
            header = f"| {table_title[0]:^{label_width}} | {table_title[1]:^{num_width}} |"
            separator = f"+{'-' * (label_width + 2)}+{'-' * (num_width + 2)}+"
            rows = []
            for l, n in zip(list_str_1, list_str_2):
                rows.append(f"| {l:<{label_width}} | {n:>{num_width}} |")

            # 打印表格
            print(header)
            print(separator)
            print('\n'.join(rows))

        else:
            # 计算每列宽度
            col_widths = [max(len(list_str_1[i]), len(list_str_2[i])) for i in range(len(list_str_2))]

            # 构建表头
            header = "|".join([f" {label:^{w}} " for label, w in zip(list_str_1, col_widths)])
            separator = "+".join(["-" * (w + 2) for w in col_widths])

            # 构建数值行
            values = "|".join([f" {n:^{w}} " for n, w in zip(list_str_2, col_widths)])

            # 打印表格
            print(f"|{header}|")
            print(f"|{separator}|")
            print(f"|{values}|")



if __name__ == '__main__':
    random.seed(0)
    tgtd = "dataset"
    dsname = "flatds"
    rawdsd = None
    voc = convert2voc_base(tsk="seGmenTatiOn",
                           bbox_list=None,
                           rawds_root=rawdsd,
                           target_dir=tgtd,
                           subdir=False,
                           dataset_name=dsname)

    # voc.labels_process()


