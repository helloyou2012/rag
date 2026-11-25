## 导读：RAG 全景与本书约定

### 一、RAG 的完整流水线

一个生产级 RAG 系统通常可以拆解为"离线"和"在线"两条链路：

```
【离线：索引构建】
原始文档 → 解析/清洗 → 分块(Chunking) → [上下文增强] → 嵌入(Embedding) → 向量库 / 倒排索引
                                         └→ 元数据抽取 ─────────────────────↗

【在线：查询与生成】
用户问题 → [查询理解：改写/分解/路由/HyDE] → 召回(向量+关键词+过滤)
        → [重排序] → [上下文压缩/组装] → LLM 生成 → [自检/纠错] → 答案
```

方括号内的环节是"可选增强"，也就是本书 20 种技术各自落点所在。

### 二、20 种技术在流水线上的位置

| 流水线阶段 | 对应技术（章节） |
|---|---|
| 索引阶段 | 语义分块(2)、父子分块(10)、上下文检索(12)、GraphRAG 索引(14)、多模态索引(19) |
| 查询阶段（查询理解） | 查询改写(5)、多查询(6)、HyDE(7)、查询分解(8)、查询路由(9)、自适应(18) |
| 召回阶段 | 朴素 RAG(1)、混合检索(3)、元数据过滤(11)、GraphRAG(14) |
| 排序与上下文阶段 | 重排序(4)、上下文压缩(13) |
| 控制与反馈 | Agentic(15)、Corrective(16)、Self-RAG(17)、Adaptive(18) |
| 质量保障 | RAG 评估(20) |

### 三、本书的统一结构

每章都遵循同一模板，方便你快速查阅：

1. **一句话概括**：与信息图右侧说明对应。
2. **要解决的问题**：没有它会怎样。
3. **工作原理与流程**：对应图中的流程箭头。
4. **实现要点与代码**：以 Python 伪代码为主。
5. **优点与代价**：成本、延迟、复杂度。
6. **适用场景与不适用场景**。
7. **常见陷阱与调优建议**。
8. **与其他技术的组合**。
9. **本章小结**。

### 四、代码约定

为避免绑定具体框架，书中代码使用以下"抽象接口"，你可以替换成 LangChain、LlamaIndex、Haystack 或自研实现：

```python
from typing import Any

def embed(texts: list[str]) -> list[list[float]]:
    """调用嵌入模型，返回向量列表。"""

def llm(prompt: str, **kwargs) -> str:
    """调用大语言模型，返回文本。"""

class VectorStore:
    def add(self, ids, vectors, texts, metadatas): ...
    def search(self, vector, k: int = 5, filter: dict | None = None) -> list[dict]:
        """返回 [{'id':..., 'text':..., 'score':..., 'metadata':{...}}, ...]"""

class KeywordIndex:   # 例如 BM25 / Elasticsearch / OpenSearch
    def search(self, query: str, k: int = 5) -> list[dict]: ...
```
