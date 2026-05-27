# Salient object detection

这是显著目标检测方向的 U2Net 工程，目录名已经从历史拼写 `SlientObjectDetedtion` 收正为 `salient_object_detection`。

## U2Net SOD 流水线

1. 原始数据：
   `u2net_pytorch/raw_data/jpgs/` 放 RGB 图片，`raw_data/pngs/` 放同名二值/灰度显著图。
2. 数据划分：
   运行 `u2net_pytorch/utils/rawdata_to_dataset.py`，按 9:1 拆到 `dataset/train_data` 和 `dataset/test_data`。
3. Dataset：
   `data_dataset.py` 和 `utils/data_datasetb.py` 保留了读取图片/mask 的模板。
4. Transform：
   `utils/data_transforms.py` 放图像归一化、缩放、张量转换等逻辑。
5. Model：
   `src/model.py` 是 U2Net 网络结构。
6. Train：
   `train.py` 目前是待补入口，建议补 BCE/IoU loss、多尺度 side-output loss 和可视化日志。
7. Export：
   Paddle 推理模型 artifact 已移到 `archive/local_ignored/model_artifacts/`，保留 `x2paddle_code.py` 作为转换参考。

显著目标检测与语义分割很像，但输出通常是前景显著概率图，评估时更关注 MAE、F-measure、S-measure、E-measure。
