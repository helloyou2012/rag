## 第 16 章　Corrective RAG（纠错 RAG）

> **一句话概括**：评估检索到的文档质量，不够好就回退到 Web 搜索等方式补救。

### 16.1 要解决的问题

传统 RAG 有个隐含假设：**检索到的就是对的**。现实中：

- 检索可能返回**完全不相关**的内容；
- 或**部分相关**，缺少关键信息；
- 模型拿到不相关资料后，要么胡编，要么"硬套"，**比不检索更糟**。

CRAG（Corrective Retrieval Augmented Generation，Yan 等，2024）提出：在生成之前，先对检索结果**打分、判定、纠正**。

### 16.2 工作原理

对应图中流程：`Retrieve → Grade Docs → Use or Web Search → LLM`

**步骤**

1. **检索**：照常从知识库检索候选文档。
2. **评分（Grade）**：用一个**检索评估器**判断每个文档与问题的相关性，并汇总成整体判定：

| 判定 | 含义 | 动作 |
|---|---|---|
| **Correct（正确）** | 至少有高度相关的文档 | 使用这些文档，并做**知识精炼**（只保留关键片段） |
| **Incorrect（错误）** | 全部不相关 | 丢弃检索结果，**改用 Web 搜索**（或其他备用来源） |
| **Ambiguous（模糊）** | 有些相关但不充分 | **两者结合**：保留精炼后的内部文档 + 补充 Web 搜索结果 |

3. **知识精炼（Decompose-then-Recompose）**：把文档拆成细小片段（句子/条目），对每个片段评分，过滤无关的，再把保留的片段重新组合。
4. **查询改写后搜索**：对 Web 搜索，先把问题改写成搜索引擎友好的关键词。
5. **生成**：基于校正后的上下文生成答案。

**评估器的实现选择**

| 选择 | 说明 |
|---|---|
| 原论文：微调的小型评估器（如 T5） | 快、便宜；需要训练数据 |
| **LLM 评审（LLM-as-Grader）** | 零样本、易落地；增加一次 LLM 调用 |
| 交叉编码器/重排序分数 | 利用第 4 章的重排序分数 + 阈值，**成本最低** |
| 混合 | 重排序分数先粗判，边界样本再交给 LLM |

### 16.3 实现要点与代码

```python
GRADE_PROMPT = """你是检索质量评审员。判断【文档】是否包含可用于回答【问题】的信息。
只输出 JSON：{{"relevant": true/false, "reason": "一句话"}}

问题：{question}
文档：{doc}"""

def grade_docs(question, docs):
    graded = []
    for d in docs:
        g = json.loads(llm(GRADE_PROMPT.format(question=question, doc=d["text"][:2000]), temperature=0))
        graded.append({**d, "relevant": g["relevant"], "reason": g["reason"]})
    return graded

def corrective_rag(question, store, kw, reranker, web_search_fn):
    cands = hybrid_search(question, store, kw, recall_k=50, top_n=30)
    top = rerank(question, cands, reranker, top_n=6)
    graded = grade_docs(question, top)
    good = [d for d in graded if d["relevant"]]

    n_good, n_all = len(good), len(graded)
    if n_good >= 2:                                   # Correct：足够
        verdict, ctx_docs = "correct", good
    elif n_good == 0:                                 # Incorrect：全不相关 → Web
        verdict = "incorrect"
        web_q = llm(f"把问题改写成适合搜索引擎的关键词（只输出关键词）：{question}", temperature=0)
        ctx_docs = web_search_fn(web_q)
    else:                                             # Ambiguous：内部 + Web 补充
        verdict = "ambiguous"
        web_q = llm(f"把问题改写成适合搜索引擎的关键词（只输出关键词）：{question}", temperature=0)
        ctx_docs = good + web_search_fn(web_q)

    refined = compress_extractive(question, ctx_docs)      # 知识精炼（见第 13 章）
    ctx = "\n\n".join(f"[{i+1}] ({d.get('source','web')}) {d['text']}" for i, d in enumerate(refined))
    answer = llm(PROMPT.format(context=ctx, question=question))
    return {"answer": answer, "verdict": verdict}      # 返回判定，便于监控与评估
```

**关键设计**

- **阈值校准**：Correct/Ambiguous/Incorrect 的边界需在评估集上调。
- **外部来源的信任**：Web 内容质量参差，应限定可信域名、标注来源，并防范**提示词注入**。
- **回退不止是 Web**：也可回退到"更大的索引""另一个知识库""向用户澄清""直接拒答"。
- **监控判定分布**：线上 `incorrect` 比例高，说明知识库覆盖或检索出了问题，是**极有价值的运营信号**。

### 16.4 优点与代价

**优点**

- **显著降低"垃圾进、垃圾出"**：检索失败时不再硬答。
- **即插即用**：作为外挂的检查环节，不改动索引。
- 提供**可解释的判定**与监控信号。
- 提升覆盖面：知识库没有的内容可通过 Web 补足。

**代价**

- 额外的评分调用 → **延迟与成本增加**（对每个文档逐一评分尤其明显）。
- **评估器本身会出错**：误判相关为无关，会不必要地触发 Web 搜索；反之则放过噪声。
- 引入 Web 搜索带来**数据合规、内容可信度和注入风险**（内网/保密场景可能不允许）。
- 流程分支多，调试复杂度上升。

### 16.5 适用场景

- 知识库**覆盖不全**，经常遇到"库里没有"的问题。
- 对错误答案**零容忍**（客服、医疗、金融、法务），宁可说"不知道"。
- 时效性要求高，需要 Web 补充最新信息。

### 16.6 常见陷阱

1. **评分太严/太松**：导致过度回退或放行噪声；务必校准。
2. **逐文档调用 LLM 评分**：延迟高。可批量评分（一次提示评多个）或使用重排序分数粗筛。
3. **回退来源不可信**：Web 内容未经筛选就进上下文。
4. **无拒答出口**：Web 也找不到时，应明确告知"无法确定"，而不是硬答。
5. **未记录判定结果**：丧失改进知识库的机会。

### 16.7 与其他技术的组合

- **+ 重排序（4）**：重排序分数作为低成本评分器。
- **+ 上下文压缩（13）**：即 CRAG 的"知识精炼"。
- **+ Agentic RAG（15）**：评分与回退是 Agent 的"反思—重试"步骤。
- **+ 查询改写（5）**：回退时改写查询。

### 16.8 本章小结

Corrective RAG 的价值是：**让系统知道"自己找到的东西靠不靠谱"，并有 B 计划。** 这是把 RAG 从"尽力而为"推向"可信可靠"的关键一步。

## 16.9 案例与实战

> 说明：案例为虚构示例，数字仅作示意。

### 案例：知识库覆盖不全的客服机器人

**症状**：用户问了知识库里**根本没有**的新功能，系统检索到“长得像”的旧文档，模型硬套，给出错误指引，引发投诉。

**方案**
1. 用重排序分数 + 阈值做粗判，边界样本再交给 LLM 评审；
2. 判定 `incorrect`：回退到官网帮助中心的 Web 搜索（限定域名），仍无结果则**明确告知“暂无资料，已转人工”**；
3. 记录每次判定，每周统计 `incorrect` 的高频问题，**反哺知识库补文档**。

**结果（示意）**：“编造式回答”投诉下降 70%；`incorrect` 占比 12%，其中 60% 在两周内通过补充文档消除——CRAG 同时成了知识库运营的“缺口雷达”。

### 实战 16：用评估集校准 Correct / Ambiguous / Incorrect

**步骤**
1. 准备 40 个问题：20 个库内可答、10 个部分可答、10 个库内完全没有。
2. 在 `corrective_rag` 里先用重排序分数（实验 1 的 `ToyReranker` 或真实模型）做初判，输出每个问题的最高分与前 3 名分数。
3. 画出三类问题的分数分布，选两个阈值把它们分开；统计混淆矩阵。
4. 对边界区间（如 0.3–0.5）再启用 LLM 评审，比较“全量评审”与“边界评审”的成本与准确率。
5. 为 `incorrect` 实现回退链：Web（可 mock）→ 明确拒答；记录判定日志。

**验收标准**
- [ ] 三类问题的分数分布图 + 混淆矩阵。
- [ ] “全量评审 vs 边界评审”的成本/准确率对比。
- [ ] 所有拒答与回退均可在日志中检索，并生成“知识库缺口报告”。

**安全提醒**：Web 回退的内容必须当作**不可信数据**，不能让其中的指令影响系统。
