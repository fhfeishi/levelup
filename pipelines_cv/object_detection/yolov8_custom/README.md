# YOLOv8 custom project

这是一个偏教学和手写理解的 YOLOv8 PyTorch 工程，保留了模型结构、训练循环、VOC 标注转换、mAP 评估和推理脚本。

## 文件职责

| 文件/目录 | 作用 |
| --- | --- |
| `cfg.yaml` | 数据、模型、训练、推理、mAP 的统一配置。 |
| `voc_annotation.py` | VOC XML -> 训练索引 txt。 |
| `train.py` | 训练入口，读取 `cfg.yaml`。 |
| `get_map.py` | mAP 评估入口。 |
| `inference.py` / `infer_by_torch.py` | 推理脚本。 |
| `nets/` | backbone、YOLO body、loss、EMA、LR scheduler。 |
| `utils/` | dataloader、bbox、fit loop、callback、mAP 工具。 |
| `VOCdevkit/` | 只保留目录占位，真实数据不提交。 |

## 最小运行顺序

```powershell
cd cv_pipelines/object_detection/yolov8_custom
Copy-Item model_data/my_classes.example.txt model_data/my_classes.txt
python voc_annotation.py
python train.py
python get_map.py
python inference.py
```

如果要加载旧权重，把本地 `.pth` 放回 `logs/` 或 `model_data/`，再改 `cfg.yaml` 里的 `train_cfg.model_path` 或 `infer_cfg.model_path`。

## 约定

- `logs/`、`weights/`、`*.pth`、生成的 txt 索引都不提交。
- `load_cfg()` 已改成按工程目录读取 `cfg.yaml`，从别的目录调用脚本也不会找错配置。
- 新增实验先改 `cfg.yaml`，不要把超参数散落在脚本里。
