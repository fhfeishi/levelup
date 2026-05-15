# Snippets

这里存“以后还会复制、改造、嵌进工程”的代码片段。原则是按任务场景找，不按当时随手起的文件夹名找。

## 分类

| 路径 | 内容 |
| --- | --- |
| `computer_vision/` | 图像、数据集、检测裁剪、分割 mask、视频、3D、face pipeline。 |
| `file_tools/` | 批量文件处理、PDF 合并/解析。 |
| `python/` | 爬虫、环境检查、反射/参数、设计模式。 |
| `ai/` | MNIST 多框架、HuggingFace、RAG 片段。 |
| `shell/` | WSL 和 bash 配置脚本。 |
| `visualization/` | 可视化和小玩具。 |

## 使用约定

- 新片段先放到最接近的任务目录。
- 文件名优先使用动词 + 对象，例如 `convert_jpg.py`、`merge_xmls.py`。
- 如果脚本依赖特定目录结构，在同级 README 里写输入、输出和示例命令。
- 大文件、样例图片、视频、wheel 不放这里，放 `archive/local_ignored/`。
