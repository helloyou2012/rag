## 第 14 章　GraphRAG（图检索增强）

> **一句话概括**：在实体和关系上做检索，而不只是在文本块上检索。

### 14.1 要解决的问题

基于块的向量检索擅长回答"**局部的、事实性的**"问题（"某条款是什么？"），但在以下问题上表现很差：

| 问题类型 | 示例 | 为什么向量检索不行 |
|---|---|---|
| **全局性/总结性** | "这批 500 份文档的主要主题是什么？" | 答案分散在全库，没有哪个块"包含全部" |
| **关系型/多跳** | "A 的合作方的投资人还投了哪些公司？" | 需要沿关系链遍历，块之间没有显式连接 |
| **跨文档聚合** | "有哪些人在多个事件中同时出现？" | 需要实体消歧与聚合 |

向量检索把世界看成"一堆互不相连的文本块"，丢失了**实体之间的结构**。

### 14.2 工作原理

对应图中流程：`Docs → Entity Extraction → Knowledge Graph → Community Summaries → LLM`

以微软研究院提出的 GraphRAG（2024）为代表，流程分为**索引**与**查询**两部分。

**索引阶段**

1. **文本分块**：把语料切成适合抽取的块。
2. **实体与关系抽取**：用 LLM 从每个块中抽取
   - **实体**（人、组织、地点、事件、概念……）及其描述；
   - **关系**（A—收购→B，A—任职于→C）及其描述、强度。
3. **构建知识图谱**：合并重复实体（实体消歧/对齐），形成图（节点 = 实体，边 = 关系），并保留到原文块的链接（可溯源）。
4. **社区检测（Community Detection）**：使用图聚类算法（如 Leiden）把图划分为**层级化的社区**——紧密相连的实体群。
5. **社区摘要**：让 LLM 为每个社区生成一份报告式摘要（自底向上，大社区的摘要可基于其子社区的摘要）。

**查询阶段（两种主要模式）**

| 模式 | 适用问题 | 流程 |
|---|---|---|
| **Local Search（局部）** | 围绕具体实体的问题 | 识别问题中的实体 → 在图中找到该实体及其邻居、关系、相关块与社区报告 → 组装上下文 → 生成 |
| **Global Search（全局）** | 概括、主题、趋势类问题 | 把各社区摘要分批交给 LLM 各自回答（map）→ 汇总各部分答案（reduce）→ 生成最终回答 |

> 其他图检索方案还包括：基于 Neo4j 等图数据库的 Text2Cypher / 图遍历、LightRAG、HippoRAG（受记忆机理启发，结合个性化 PageRank）等。思路相通：**把"关系"作为一等公民**。

### 14.3 实现要点与代码

```python
import json
import networkx as nx

EXTRACT_PROMPT = """从文本中抽取实体与关系，只输出 JSON：
{{"entities": [{{"name": "...", "type": "...", "desc": "..."}}],
  "relations": [{{"src": "...", "dst": "...", "type": "...", "desc": "..."}}]}}
- 实体名使用文本中最完整、规范的写法；同一实体保持一致写法。
- 只抽取文本明确陈述的内容，不要推测。

文本：
{chunk}"""

def build_graph(chunks: list[dict]) -> nx.Graph:
    G = nx.Graph()
    for ch in chunks:
        data = json.loads(llm(EXTRACT_PROMPT.format(chunk=ch["text"]), temperature=0))
        for e in data["entities"]:
            n = e["name"].strip()
            if n not in G:
                G.add_node(n, type=e["type"], descs=[], chunks=set())
            G.nodes[n]["descs"].append(e["desc"])
            G.nodes[n]["chunks"].add(ch["id"])
        for r in data["relations"]:
            G.add_edge(r["src"], r["dst"], type=r["type"], desc=r["desc"], chunk=ch["id"])
    return G

def detect_communities(G: nx.Graph):
    # 生产建议使用 Leiden（如 graspologic / igraph / leidenalg）；此处用 networkx 自带的 Louvain 示意
    return nx.community.louvain_communities(G, seed=42)

def summarize_communities(G, communities):
    reports = []
    for cid, nodes in enumerate(communities):
        sub = G.subgraph(nodes)
        facts = "\n".join(
            [f"实体 {n}: {'; '.join(G.nodes[n]['descs'][:2])}" for n in sub.nodes] +
            [f"关系 {u}-{v}: {d['desc']}" for u, v, d in sub.edges(data=True)]
        )
        report = llm(f"请基于以下事实，写一份这个实体群的主题报告（主题、关键实体、关键关系、要点）：\n{facts}")
        reports.append({"id": cid, "nodes": list(nodes), "report": report})
    return reports

# ---- 查询：Local ----
def local_search(question, G, entity_index, chunk_store, k_entities=5, hops=1):
    seed = entity_index.search(embed([question])[0], k=k_entities)       # 实体向量索引
    nodes = set(h["text"] for h in seed)
    for _ in range(hops):
        for n in list(nodes):
            nodes.update(G.neighbors(n))
    sub = G.subgraph(nodes)
    facts = [f"{u} -[{d['type']}]-> {v}: {d['desc']}" for u, v, d in sub.edges(data=True)]
    chunk_ids = set().union(*(G.nodes[n]["chunks"] for n in sub.nodes))
    ctx = "\n".join(facts) + "\n\n" + "\n".join(chunk_store[c] for c in list(chunk_ids)[:5])
    return llm(PROMPT.format(context=ctx, question=question))

# ---- 查询：Global (map-reduce) ----
def global_search(question, reports, batch=5):
    partials = []
    for i in range(0, len(reports), batch):
        part = "\n\n".join(r["report"] for r in reports[i:i + batch])
        partials.append(llm(f"根据以下社区报告，针对问题给出要点（无关则回答 NONE）：\n问题：{question}\n{part}"))
    merged = "\n".join(p for p in partials if p.strip() != "NONE")
    return llm(f"综合下列要点，回答问题：{question}\n{merged}")
```

**工程要点**

| 主题 | 建议 |
|---|---|
| 抽取质量 | 提示词中加入少量**领域示例**与**实体类型白名单**；多轮"补漏抽取"（gleaning）可提高召回 |
| 实体对齐 | 同名异写（"微软/Microsoft/MS"）需要合并，常用字符串规则 + 嵌入相似度 + LLM 判定 |
| 图存储 | 小规模：内存 NetworkX；中大型：Neo4j / NebulaGraph / 图数据库；或关系型/Parquet 存储 |
| 可溯源 | 每个实体、关系都保留指向原块的引用 |
| 增量更新 | 全量重建成本高，需设计增量抽取与社区重算策略 |
| 与向量检索结合 | 给实体、关系描述、社区摘要都建向量索引，用"向量找入口，图做扩展" |

### 14.4 优点与代价

**优点**

- **能回答全局性与关系型问题**，这是纯向量 RAG 的盲区。
- 通过关系遍历完成**多跳推理**。
- 结构化、可解释（可以展示实体与关系路径）。
- 减少"信息分散在多个块"带来的遗漏。

**代价**

- **索引成本极高**：需要对全部文本跑 LLM 抽取（以及社区摘要），在大语料上开销可能是普通 RAG 索引的数十倍甚至更多。
- **抽取质量决定上限**：漏抽、错抽、实体未对齐都会拖垮效果。
- **工程复杂度**高：图存储、增量更新、社区重算。
- **对简单事实问题不一定比向量 RAG 好**，反而更慢更贵。
- 全局搜索的 map-reduce 延迟较高。

### 14.5 适用场景

- 需要"全局洞察"：情报分析、舆情、尽职调查、科研文献综述、案件卷宗。
- **实体与关系丰富**的领域：企业关系、供应链、生物医药（基因—疾病—药物）、金融风控、人物关系。
- 多跳推理问题占比高的业务。

**不推荐**：规模小、问题以"查事实"为主的知识库；预算有限、无法承担抽取成本的团队。

### 14.6 常见陷阱

1. **一上来就全库建图**：先在**小样本**上验证抽取质量与收益，再放大。
2. **不做实体对齐**：图是碎片化的，遍历毫无意义。
3. **图太稠密**：把所有共现都建边，产生噪声；对关系设置类型与权重阈值。
4. **只用图、不用向量**：实践中**图 + 向量的混合**更稳健。
5. **忽略更新机制**：知识库一更新就要全量重建，运维难以为继。
6. **没有评估收益**：用"全局问题集"与"多跳问题集"专项评估。

### 14.7 与其他技术的组合

- **+ 查询路由（9）**：全局问题 → Global Search；局部事实 → 向量/Local；关系问题 → 图遍历。
- **+ Agentic RAG（15）**：Agent 把"图查询"当作一个工具。
- **+ 元数据过滤（11）**：对子图/社区按时间、来源过滤。

### 14.8 本章小结

GraphRAG 把"文档集合"变成"知识网络"。它不是替代向量 RAG，而是为**全局与关系型问题**补上一块拼图——并为此付出显著的索引成本。

## 14.9 案例与实战

> 说明：案例为虚构示例，数字仅作示意。

### 案例：尽调资料库的“全局问题”

**背景**：某并购尽调资料库有 300 份文档（合同、访谈、公告），分析师常问：
- “这些文档里反复出现的主要风险主题是什么？”（全局）
- “与 A 公司有关联交易的所有主体有哪些？”（关系）

**向量 RAG 的表现**：对第一个问题只能给出零散片段；第二个问题漏掉间接关联方。

**GraphRAG 做法**
1. LLM 抽取实体（公司、人物、合同）与关系（持股、签署、任职），实体对齐；
2. 建图并做社区划分，为每个社区生成报告；
3. 全局问题走 Global Search（map-reduce 汇总社区报告）；关系问题走 Local Search + 图遍历。

**结果（示意）**：全局类问题的人工评分 2.5 → 4.0；索引成本约为向量 RAG 的 20 倍，因此只对“高价值项目库”启用，且先小样本验证。

### 实战 14：50 份文档的迷你 GraphRAG

**步骤**
1. 选 30–50 篇短文（新闻、公司公告或你自己的文档）。
2. 用第 14 章的 `EXTRACT_PROMPT` 抽取实体与关系，输出到 JSON；抽检 30 条，计算抽取精确率。
3. 实现实体对齐（字符串规则 + 嵌入相似度阈值），统计合并前后实体数。
4. 用 NetworkX 建图，Louvain/Leiden 划分社区，生成社区报告。
5. 准备两类问题各 8 个：**全局总结类**、**多跳关系类**；对比“向量 RAG / GraphRAG”的人工评分。
6. 估算索引成本（LLM 调用次数 × 平均 token）。

**验收标准**
- [ ] 抽取精确率与实体对齐前后对比。
- [ ] 两类问题的对比结论，并说明在**简单事实类**问题上是否仍应走向量检索。
- [ ] 成本估算与“是否值得”的判断。

**提醒**：评估时同时保留一组“单文档事实问题”，确认 GraphRAG 没有让它们变差。
