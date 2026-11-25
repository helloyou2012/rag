## 附录 D　术语表

| 术语 | 解释 |
|---|---|
| **RAG** | Retrieval-Augmented Generation，检索增强生成：先检索外部知识，再让 LLM 基于其生成答案 |
| **Chunk（块）** | 文档被切分后的检索单位 |
| **Embedding（嵌入）** | 把文本/图像映射为稠密向量，语义相近者距离更近 |
| **Vector DB（向量数据库）** | 存储向量并支持近似最近邻（ANN）检索的数据库 |
| **ANN** | Approximate Nearest Neighbor，近似最近邻搜索（如 HNSW、IVF） |
| **Top-k** | 取相似度/得分最高的 k 个结果 |
| **BM25** | 经典的基于词频与逆文档频率的关键词检索排序函数 |
| **Sparse / Dense** | 稀疏表示（词级，如 BM25/SPLADE）/ 稠密表示（嵌入向量） |
| **RRF** | Reciprocal Rank Fusion，倒数排名融合，仅依赖名次合并多路结果 |
| **Bi-Encoder** | 双塔编码器：查询与文档独立编码后比较，速度快 |
| **Cross-Encoder** | 交叉编码器：查询与文档拼接联合编码，精度高、速度慢，常用于重排序 |
| **Reranker** | 重排序模型 |
| **ColBERT / 延迟交互** | 为每个 token 保留向量并在检索时做细粒度匹配的方法，介于双塔与交叉编码之间 |
| **HyDE** | Hypothetical Document Embeddings，用生成的假设答案做检索 |
| **Query Rewriting** | 将原始问题改写成更适合检索的查询 |
| **Step-back Prompting** | 先生成更抽象的上位问题 |
| **Multi-hop（多跳）** | 需要多步推理、逐步检索才能得出答案的问题 |
| **Router（路由器）** | 决定查询交给哪个数据源/策略的组件 |
| **Parent-Child Chunking** | 小块（子）用于检索，大块（父）用于生成 |
| **Metadata（元数据）** | 描述文档/块的结构化属性（时间、来源、权限等） |
| **Self-Query** | 让 LLM 从自然语言中解析出语义查询与元数据过滤条件 |
| **Contextual Retrieval** | 为每个块预置文档级背景说明后再索引 |
| **Prompt Caching** | 缓存提示词的固定前缀以降低重复调用的成本 |
| **Context Compression** | 缩减检索内容，仅保留与问题相关部分 |
| **Lost in the Middle** | 模型对长上下文中间位置的信息利用较差的现象 |
| **GraphRAG** | 基于知识图谱与社区摘要的检索增强 |
| **Community Detection** | 图中的社区发现算法（如 Leiden、Louvain） |
| **Agent / Tool Use** | 由 LLM 规划并调用外部工具的系统 / 函数调用能力 |
| **ReAct** | Reason + Act：推理与行动交替的 Agent 范式 |
| **CRAG** | Corrective RAG，对检索结果评估并纠正 |
| **Self-RAG** | 通过反思标记让模型自主决定检索并批判输出 |
| **Adaptive RAG** | 依据问题复杂度选择检索策略 |
| **Vision LLM / VLM** | 能够理解图像输入的多模态大模型 |
| **CLIP / SigLIP** | 图文联合嵌入模型 |
| **ColPali** | 基于视觉语言模型对文档页面图像做多向量检索的方法 |
| **Faithfulness** | 忠实度：答案被上下文支持的程度 |
| **Answer Relevancy** | 答案与问题的相关性 |
| **Context Precision / Recall** | 上下文精确率 / 召回率 |
| **Recall@k / MRR / nDCG** | 经典检索评估指标 |
| **LLM-as-a-Judge** | 用 LLM 作为评审自动打分 |
| **RAGAS** | 开源的 RAG 评估框架 |
| **Hallucination（幻觉）** | 模型生成无依据或错误的内容 |
| **Grounding（有据性）** | 答案基于可验证的资料来源 |
