# CV pipelines

这里放能串成工程的 CV 工作流，不再和零散脚本混在一起。

## 工作流地图

| 任务 | 路径 | 当前状态 |
| --- | --- | --- |
| 目标检测 | `object_detection/` | YOLOv8 自定义训练工程 + 检测算法笔记。 |
| 语义分割 | `semantic_segmentation/` | U2Net 数据工具、模型结构、SMP 实验代码。 |
| 显著目标检测 | `salient_object_detection/` | U2Net 数据拆分、模型、Paddle 转换记录。 |
| 部署 | `deployment/` | ONNX、Paddle、模型评估、C++ demo、Gradio 快速原型。 |

## 通用流水线

1. 定义任务：检测、分割、显著目标检测或部署。
2. 准备数据：统一 raw data、train/test data、标注格式。
3. 固化配置：类别、输入尺寸、增强、训练轮次、输出目录。
4. 训练模型：只提交训练代码和轻量配置，权重放本地 artifact。
5. 评估结果：mAP、IoU、Dice、可视化或业务指标。
6. 推理验证：单图、批量、视频或接口化验证。
7. 导出部署：Torch -> ONNX/Paddle/OpenVINO/TensorRT/API。

训练输出和模型文件默认不提交，统一由 `.gitignore` 拦住。
