# Deployment

部署目录用于把训练好的模型从“能跑脚本”推进到“能给别人用”。

## 路线

1. 固定输入输出：
   明确输入尺寸、颜色空间、归一化、batch 维度、输出 tensor 含义。
2. 导出中间格式：
   PyTorch 优先导出 ONNX；Paddle 工程保留 Paddle Inference 或 X2Paddle 转换。
3. 校验数值：
   在 `notebooks/model_eval/` 对比原框架输出和部署格式输出，先小样本再批量。
4. 包装服务：
   快速演示用 Gradio，工程服务优先 FastAPI，批处理任务可直接 CLI。
5. 性能优化：
   CPU 看 OpenVINO/ONNX Runtime，GPU 看 TensorRT/ONNX Runtime CUDA。
6. 交付：
   写清环境、模型来源、输入输出 schema、版本号和回滚方式。

## 目录

| 路径 | 内容 |
| --- | --- |
| `notebooks/onnx/` | ONNX 学习与导出记录。 |
| `notebooks/paddle/` | Paddle 推理和转换记录。 |
| `notebooks/model_eval/` | 模型导出前后的结果校验。 |
| `cpp_demo/` | C++ demo。 |
| `gradio_quickstart/` | 快速原型入口。 |

权重、ONNX、engine、Paddle inference model 等 artifact 不提交。
