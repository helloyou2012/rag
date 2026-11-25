# 20 种 RAG 技术：AI 工程师必修手册（分册版）

> 依据 Prasad Rao（@kprasadrao）的《20 RAG Techniques Every AI Engineer Should Know》信息图展开。每章包含：原理 → 实现 → 取舍 → 陷阱 → **案例与实战**。

## 如何使用本书

- 新手：`前言` → `导读` → 第 1–4 章 → 运行 `labs/lab01_mini_rag.py`。
- 有经验者：直接看 `附录 A 决策树` 与 `附录 C 故障排查`，再回到对应章节。
- 动手：每章末尾的「案例与实战」都有**步骤与验收标准**；综合项目见 `projects/`。

## 目录

### 开篇

- [前言](chapters/00a-前言.md)
- [导读：RAG 全景与约定](chapters/00b-导读-RAG全景与约定.md)

### 第一部分　检索基础

- [本部分导读](chapters/part1-检索基础-导读.md)
- [第 1 章　Naive RAG（朴素 RAG）](chapters/ch01-naive-rag.md)
- [第 2 章　Semantic Chunking（语义分块）](chapters/ch02-semantic-chunking.md)
- [第 3 章　Hybrid Search（混合检索）](chapters/ch03-hybrid-search.md)
- [第 4 章　Reranking（重排序）](chapters/ch04-reranking.md)

### 第二部分　查询转换

- [本部分导读](chapters/part2-查询转换-导读.md)
- [第 5 章　Query Rewriting（查询改写）](chapters/ch05-query-rewriting.md)
- [第 6 章　Multi-Query Retrieval（多查询检索）](chapters/ch06-multi-query.md)
- [第 7 章　HyDE（假设性文档嵌入）](chapters/ch07-hyde.md)
- [第 8 章　Query Decomposition（查询分解）](chapters/ch08-query-decomposition.md)
- [第 9 章　Query Routing（查询路由）](chapters/ch09-query-routing.md)

### 第三部分　索引与上下文

- [本部分导读](chapters/part3-索引与上下文-导读.md)
- [第 10 章　Parent-Child Chunking（父子分块）](chapters/ch10-parent-child-chunking.md)
- [第 11 章　Metadata Filtering（元数据过滤）](chapters/ch11-metadata-filtering.md)
- [第 12 章　Contextual Retrieval（上下文检索）](chapters/ch12-contextual-retrieval.md)
- [第 13 章　Context Compression（上下文压缩）](chapters/ch13-context-compression.md)

### 第四部分　高级架构

- [本部分导读](chapters/part4-高级架构-导读.md)
- [第 14 章　GraphRAG（图检索增强）](chapters/ch14-graphrag.md)
- [第 15 章　Agentic RAG（智能体 RAG）](chapters/ch15-agentic-rag.md)
- [第 16 章　Corrective RAG（纠错 RAG）](chapters/ch16-corrective-rag.md)
- [第 17 章　Self-RAG（自反思 RAG）](chapters/ch17-self-rag.md)
- [第 18 章　Adaptive RAG（自适应 RAG）](chapters/ch18-adaptive-rag.md)

### 第五部分　多模态与评估

- [本部分导读](chapters/part5-多模态与评估-导读.md)
- [第 19 章　Multimodal RAG（多模态 RAG）](chapters/ch19-multimodal-rag.md)
- [第 20 章　RAG Evaluation（RAG 评估）](chapters/ch20-rag-evaluation.md)

### 综合实战项目

- [项目 1：企业知识库问答（四周计划）](projects/project1-企业知识库问答.md)
- [项目 2：财报 / 合同分析助手（高精度型）](projects/project2-财报与合同分析助手.md)
- [项目 3：研究型助手（Adaptive + Agent + GraphRAG）](projects/project3-研究型Agent与GraphRAG.md)

### 实验代码

- `labs/lab01_mini_rag.py`：**零依赖（仅 numpy）可直接运行**。对比向量 / BM25 / 混合 RRF / 重排序 / 元数据过滤 / 父子分块，并输出 Recall@k、MRR。
- `labs/lab02_llm_pipeline.py`：接入真实 LLM 的改写、多查询、评分回退骨架（需要 API Key；**未做真实 API 端到端验证**，请先小样本试跑）。
- `labs/eval_set_template.jsonl`：评估集模板。

### 附录

- [附录 A　技术选型决策树](chapters/appendix-A.md)
- [附录 B　演进路线与组合方案](chapters/appendix-B.md)
- [附录 C　故障排查手册](chapters/appendix-C.md)
- [附录 D　术语表](chapters/appendix-D.md)
- [结语](chapters/appendix-Z-结语.md)

## 快速开始

```bash
cd labs
pip install numpy
python lab01_mini_rag.py
```

实验 1 的参考输出（玩具嵌入，仅演示流程；15 个问题的样本很小，差异在噪声范围内）：

```
配置                              R@1    R@3    R@5     MRR
① 仅向量（Naive RAG）             0.80   0.80   0.93   0.844
② 仅 BM25                     0.67   0.93   1.00   0.794
③ 混合 RRF                     0.80   0.80   1.00   0.847
④ 混合 + 重排序                   0.73   0.80   1.00   0.813
⑤ 元数据过滤 + 混合 + 重排序           0.73   0.93   1.00   0.824
```

> 读法提示：这个结果恰好说明——**玩具重排序没有提升 R@1**。结论不是“重排序没用”，而是“**换成真实重排序模型并在自己的评估集上验证之前，不要下结论**”。

## 关于案例与数据

书中“案例”均为依据常见项目情形编写的**虚构示例**，其中的数字仅作示意，不代表真实项目或公开基准；请以你自己的评估数据为准。
