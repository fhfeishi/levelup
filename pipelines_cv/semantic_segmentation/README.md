# Semantic segmentation

这里保留语义分割相关工程：`pytorch_u2net/` 是 U2Net 数据与模型骨架，`smp_experiments/` 是 segmentation-models-pytorch 方向的实验代码。

## U2Net 分割流水线

1. 放原始数据：
   图片放 `pytorch_u2net/raw_data/jpgs/`，mask 放 `pytorch_u2net/raw_data/pngs/` 或从 LabelMe JSON 转换。
2. 检查标注：
   用 `utils/dataset_check_json.py`、`dataset_get_png_palette.py` 检查 JSON、PNG palette 和类别。
3. 标注转换：
   LabelMe JSON 用 `dataset_json_to_pngs.py` 转 PNG；RGB mask 用 `dataset_rgb_to_pngs.py` 转类别 mask。
4. 组织训练集：
   目标结构是 `dataset/train_data/jpgs|pngs` 和 `dataset/test_data/jpgs|pngs`。
5. 读取数据：
   `utils/my_dataset.py` 按图片名匹配同名 mask，返回训练样本。
6. 训练模型：
   `src/model.py` 已保留 U2Net 结构，`train.py` 目前是待补入口，建议从 dataloader、loss、metric、fit loop 四块补齐。
7. 评估与可视化：
   先用小样本 overfit，再补 IoU/Dice、单图可视化和批量导出。

真实图片、mask 和训练输出默认不提交。
