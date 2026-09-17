# RAG 检索知识与代码片段

这个目录聚焦 RAG 里的 retrieval 部分：关键词搜索、语义搜索、混合检索、chunking、检索评估。目标是先有可解释 baseline，再逐步叠加 embedding、reranker 和工程化索引。

## 快速选择

| 场景 | 优先方案 | 原因 |
| --- | --- | --- |
| 错误码、函数名、字段名、专有术语 | BM25 / 倒排索引 | 精确词匹配更可靠 |
| 同义表达、自然语言问答、跨语言 | 语义搜索 | embedding 能捕捉语义相近 |
| 真实生产 RAG | 混合检索 | BM25 保精确，向量保召回 |
| 排名前 20 里有答案但顺序差 | reranker | 对候选 chunk 做更精细排序 |
| 回答引用不稳定 | 调 chunking 和 metadata | chunk 是 RAG 的上下文边界 |

## 文件索引

| 文件 | 内容 |
| --- | --- |
| `keyword_search.py` | 纯 Python 倒排索引，理解关键词检索的最小实现。 |
| `tf_idf.py` | 纯 Python 实现 TF-IDF + cosine similarity。 |
| `bm25.py` | 纯 Python 实现 BM25 检索公式。 |
| `semantic_search.py` | 使用 sentence-transformers 生成 embedding 并做向量检索。 |
| `hybrid_search.py` | 使用 Reciprocal Rank Fusion 合并 BM25 与向量召回结果。 |
| `chunking.py` | 一个递归文本切分示例，保留 overlap 和 chunk metadata。 |
| `evaluation.py` | `hit@k`、`MRR@k`、`precision@k` 等检索评估指标。 |
| `keyword_search.ipynb` | 关键词检索 notebook 入口。 |
| `semantic_search.ipynb` | 语义检索 notebook 入口。 |

## RAG 检索主链路

```text
documents
  -> load / parse
  -> clean / normalize
  -> chunk
  -> index: BM25 index + vector index
  -> retrieve: lexical top_k + vector top_k
  -> fuse / rerank
  -> build context
  -> generate answer with citations
```

## 关键词搜索

关键词搜索关注“字面匹配”：

- 倒排索引：token -> doc_ids，是搜索引擎最基础的数据结构。
- TF-IDF：词越稀有越重要，但长文档和词频饱和处理较弱。
- BM25：TF-IDF 的强 baseline，考虑词频饱和与文档长度归一化。

适合：

- 错误码：`CUDA out of memory`、`HTTP 429`
- 代码符号：函数名、类名、配置键
- 法条、合同条款、产品型号、专有名词
- 用户查询中关键词非常明确的场景

短板：

- 用户用了同义词会漏召回，例如“出差费用” vs “差旅报销”。
- 拼写错误、缩写、跨语言表达需要额外处理。
- 对 chunk 的语义整体相似度理解较弱。

## 语义搜索

语义搜索把文本映射成向量，通过 cosine similarity、dot product 或 ANN 索引找近邻。

适合：

- 自然语言问答
- 同义表达召回
- 用户不知道准确术语
- 跨语言或表达方式变化大的资料库

短板：

- 对错误码、函数名、罕见实体不一定敏感。
- embedding 模型会影响召回边界。
- 需要向量库或 ANN 索引来支撑大规模数据。

## 混合检索

真实 RAG 常用混合检索：

```text
query
  -> BM25 top 20
  -> vector top 20
  -> RRF / weighted score fusion
  -> reranker top 5
```

推荐先用 RRF，因为它只依赖 rank，不依赖不同检索器的分数尺度。等评估集稳定后，再考虑加权融合或训练排序模型。

## Chunking 经验

chunk 不是简单切文本，它决定了“能召回什么上下文”。

- 通用文档：先试 `500-1000 tokens`，`10%-20% overlap`。
- Markdown：优先按标题、段落、列表切。
- 代码：优先按文件、类、函数、方法切。
- 表格：保留表头、单位、字段解释。
- 法律/合同：保留条款编号和父级标题。

常见问题：

- chunk 太大：召回噪声多，LLM 容易引用无关内容。
- chunk 太小：标题、定义、表格字段、上下文关系丢失。
- overlap 太大：索引膨胀，重复上下文挤占 prompt。

## 最小评估集

不要只看单次 demo。建议手写一个小 golden set：

```python
queries = [
    "出差费用怎么报销",
    "HTTP 429 是什么问题",
    "向量检索和 BM25 怎么结合",
]

expected_doc_ids = [
    {"travel-policy"},
    {"api-rate-limit"},
    {"hybrid-search"},
]
```

先看：

- `hit@k`：top-k 里有没有正确文档。
- `MRR@k`：正确文档排得靠不靠前。
- `precision@k`：top-k 中相关内容比例。

调参顺序建议：

1. 固定测试集。
2. 比较 BM25、vector、hybrid。
3. 调 chunk_size、overlap、top_k。
4. 增加 reranker。
5. 最后再优化 prompt 和答案格式。

## 依赖

关键词、TF-IDF、BM25、chunking、evaluation 示例都是纯 Python。语义检索需要按需安装：

```bash
pip install sentence-transformers
```

如果数据量上来，建议把纯 Python 示例替换成成熟组件：Elasticsearch/OpenSearch、Lucene、Tantivy、FAISS、Milvus、Qdrant、Chroma 等。
