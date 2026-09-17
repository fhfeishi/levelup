# Computer vision snippets

## 快速定位

| 路径 | 用途 |
| --- | --- |
| `image_basics/` | OpenCV/Pillow 读图、缩放、裁剪基础。 |
| `datasets/dataset_inspection/` | 图片、JSON、XML、JPG 数据统计。 |
| `datasets/label_conversion/` | YOLO/VOC/XML 标签转换。 |
| `datasets/projection_tools/` | 数据集清洗、缩放、裁剪、LabelMe 转 mask。 |
| `datasets/pytorch_dataset_templates/` | PyTorch Dataset 模板。 |
| `datasets/detection_cropping/` | 目标检测数据裁剪、bbox 处理。 |
| `face/` | face mesh、dlib、mediapipe、人脸裁剪和数据清理。 |
| `video/` | gif、m4s/mp4、视频预处理。 |
| `3d_data_preprocessing/` | tif、点云、3D 缺陷检测预处理。 |
| `slide_prediction/` | 大图切片、检测/分割推理片段。 |

这些脚本大多是任务型工具，真正纳入工程前建议补 argparse、日志、输入检查和单元测试。
