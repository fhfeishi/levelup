# levelup repo

这个仓库现在按用途拆成四块：代码片段、CV 工程流水线、开发知识积累、资源索引。以前的 `ZZ*`、`pipeline_*`、`code_pieces` 已经收拢到更稳定的目录里。

## 目录

| 路径 | 内容 |
| --- | --- |
| `snippets/` | 平时积累的可复用脚本，按任务场景分类。 |
| `cv_pipelines/` | CV 工程工作流：目标检测、语义分割、显著目标检测、部署。 |
| `knowledge/` | Python、前端、算法、深度学习、DevOps 等学习笔记。 |
| `resources/` | 资源链接、资料清单。 |
| `archive/local_ignored/` | 本机保留的大文件、权重、wheel、外部仓库、scratch，不进入 git。 |

## 推荐入口

- 目标检测 YOLOv8：`cv_pipelines/object_detection/README.md`
- 自定义 YOLOv8 工程：`cv_pipelines/object_detection/yolov8_custom/README.md`
- 语义分割/U2Net：`cv_pipelines/semantic_segmentation/README.md`
- 显著目标检测/U2Net：`cv_pipelines/salient_object_detection/README.md`
- 部署路线：`cv_pipelines/deployment/README.md`
- 代码片段索引：`snippets/README.md`
- 开发知识片段：`knowledge/知识片段.md`
- 低优先级知识归档：`knowledge/低优先级归档.md`

## 仓库规则

- 权重、日志、数据集、推理模型、wheel、`.env` 不进 git。
- 需要保留的本地大文件放在 `archive/local_ignored/`。
- 可复现工程保留代码、配置、README、目录占位和示例 class 文件。
- 新增片段优先放到 `snippets/<domain>/<task>/`，新增学习笔记优先放到 `knowledge/<topic>/`。
