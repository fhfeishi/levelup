# Object detection

目标检测目录现在分两层：`yolov8_custom/` 是可跑工程，`reference_notes/` 是 RCNN/Fast R-CNN/Faster R-CNN/YOLO 系列笔记。

## YOLOv8 训练流水线

1. 准备 VOC 数据结构：
   `VOCdevkit/VOC2007/JPEGImages/` 放图片，`Annotations/` 放 XML，`ImageSets/Main/` 放划分文件。
2. 准备类别文件：
   从 `yolov8_custom/model_data/my_classes.example.txt` 复制出 `my_classes.txt` 并填写真实类别。
3. 检查配置：
   在 `yolov8_custom/cfg.yaml` 里确认 `input_shape`、`phi`、batch size、epoch、学习率和 `classes_file`。
4. 生成训练索引：
   运行 `python voc_annotation.py`，得到 `2007_train.txt` 和 `2007_val.txt`。
5. 训练：
   运行 `python train.py`，权重与日志输出到 `logs/`，该目录默认忽略。
6. 评估：
   运行 `python get_map.py`，输出 mAP 评估结果。
7. 推理：
   使用 `inference.py` 或 `infer_by_torch.py`，在 `cfg.yaml` 的 `infer_cfg.model_path` 指向本地权重。
8. 部署：
   训练完成后进入 `../deployment/` 做 ONNX/Paddle/API 等导出。

## 轻量化说明

历史权重、训练日志和本地 class 文件已经移到 `archive/local_ignored/model_artifacts/`，不会进入 git。需要复用旧权重时，再从本机归档复制回 `yolov8_custom/logs/` 或 `model_data/`。
