#!/usr/bin/env python3
"""
实验 2：接入真实 LLM 的查询侧与质量闸门（第 5/6/8/16/17 章）
-----------------------------------------------------------
依赖：pip install anthropic numpy      环境变量：ANTHROPIC_API_KEY
说明：本脚本复用 lab01 的检索组件；LLM 部分需要你的 API Key。
      脚本为学习骨架，**我在编写时没有对真实 API 做端到端运行**，首次使用请先在小样本上试跑并检查输出。
模型名通过环境变量 RAG_MODEL 指定（请使用你账号可用的模型）。
"""
import json
import os
import sys

from lab01_mini_rag import (BM25, DOCS, DenseIndex, HashEmbedder, ToyReranker,
                            allowed_ids, rrf)

try:
    import anthropic
except ImportError:
    sys.exit("请先 pip install anthropic")

MODEL = os.environ.get("RAG_MODEL", "claude-sonnet-5-5")
client = anthropic.Anthropic()


def llm(prompt: str, temperature: float = 0.0, max_tokens: int = 800) -> str:
    r = client.messages.create(model=MODEL, max_tokens=max_tokens, temperature=temperature,
                               messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in r.content if b.type == "text").strip()


def jloads(s: str):
    s = s.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(s)


BY_ID = {d["id"]: d for d in DOCS}
BM = BM25(DOCS)
DENSE = DenseIndex(DOCS, HashEmbedder())
RER = ToyReranker(BM)          # 生产中请换成真实重排序模型


def retrieve(q, flt=None, top_n=5):
    allow = allowed_ids(DOCS, flt)
    cands = rrf([DENSE.search(q, 20, allow), BM.search(q, 20, allow)])
    return RER.rerank(q, cands[:20], BY_ID, top_n=top_n)      # [(id, score)]


# ---- 第 5 章：改写 ----
def rewrite(question, history):
    h = "\n".join(f"{r}: {t}" for r, t in history[-6:]) or "（无）"
    return llm(f"把最新问题改写为独立、清晰、适合检索的查询，只输出查询。\n历史：\n{h}\n最新问题：{question}")


# ---- 第 6 章：多查询 ----
def multi_query(question, n=3):
    try:
        vs = jloads(llm(f"为问题生成{n}个措辞不同但含义相同的检索查询，只输出JSON字符串数组。\n问题：{question}", 0.7))
    except Exception:
        vs = []
    return [question] + [v for v in vs if isinstance(v, str)][:n]


def retrieve_multi(question, flt=None):
    lists = [[(i, s) for i, s in retrieve(q, flt, top_n=10)] for q in multi_query(question)]
    return rrf(lists, top_n=5)


# ---- 第 16 章：评分 + 回退（这里回退为“明确拒答”，Web 搜索留作练习） ----
def grade(question, doc_text):
    try:
        return bool(jloads(llm(f'判断文档是否包含回答问题所需的信息，只输出JSON {{"relevant":true/false}}。\n问题：{question}\n文档：{doc_text}'))["relevant"])
    except Exception:
        return True


def answer(question, history=(), flt=None):
    q = rewrite(question, list(history)) if history else question
    hits = retrieve(q, flt)
    docs = [BY_ID[i] for i, _ in hits if grade(question, BY_ID[i]["text"])]
    if not docs:
        return "资料中未找到相关信息，建议联系人工支持。", []
    ctx = "\n".join(f"[{k+1}] {d['text']}" for k, d in enumerate(docs))
    a = llm(f"仅依据资料回答，资料不足请说明，并在句末标注引用编号。\n资料：\n{ctx}\n问题：{question}")
    return a, [d["id"] for d in docs]


if __name__ == "__main__":
    for q in ["错误码 E4021 是什么意思", "我想把网盘里的视频放到火星上"]:
        a, src = answer(q, flt={"version": "v3"})
        print(f"Q: {q}\nA: {a}\n来源: {src}\n")
