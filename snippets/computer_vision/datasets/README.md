# Dataset snippets

数据集脚本按“检查 -> 转换 -> 划分 -> 训练读取”组织。

1. 检查数据：
   `dataset_inspection/` 统计图片、JSON、XML、JPG。
2. 转换标注：
   `label_conversion/` 和 `legacy_dataset_tools/` 处理 YOLO txt、VOC XML、PNG mask。
3. 清洗/投影：
   `projection_tools/` 处理文件名、坏 XML、LabelMe JSON、类别数据集抽取。
4. 裁剪检测数据：
   `detection_cropping/` 根据 bbox 或中心点裁剪。
5. 接入训练：
   `pytorch_dataset_templates/` 提供 Dataset 写法参考。

下一步如果继续精简，可以把重复的 `txt2xml_normed.py`、`txtlabel_to_xmllabel.py` 合并成一个带 argparse 的工具。
