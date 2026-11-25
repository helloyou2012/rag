#!/usr/bin/env python3
"""
实验 1：零依赖（仅 numpy）的迷你 RAG 检索实验台
------------------------------------------------
目的：亲手跑通并对比  向量检索 / BM25 / 混合检索(RRF) / 重排序 / 元数据过滤 / 父子分块，
      并用 Recall@k、MRR 量化差异（对应书中第 1、3、4、10、11、20 章）。

说明（务必阅读）：
  * 这里的“向量”是用字符 n-gram 哈希出来的玩具嵌入，不是真正的语义嵌入模型；
    它的作用是让你**无需下载模型/调用 API** 就能跑通流程。
  * 因此它几乎只能捕捉“字面重叠”，对同义改写没有真实嵌入模型强。
    把 `HashEmbedder` 换成真实嵌入模型后，再跑同样的评估，才能得到有参考价值的结论。
  * 语料与问题均为虚构的示例数据。

运行：python lab01_mini_rag.py
"""
import math
import re
from collections import Counter, defaultdict

import numpy as np

# ----------------------------------------------------------------------------
# 1. 示例语料（虚构产品“云帆网盘”）。parent 表示所属章节（用于父子分块演示）
# ----------------------------------------------------------------------------
CORPUS = [
    ("d01", "P1", "v3", "云帆网盘 v3 支持单文件最大 100GB 上传，超过限制时会返回错误码 E4021，请使用分片上传接口。"),
    ("d02", "P1", "v2", "云帆网盘 v2 单文件最大 5GB，超过限制返回错误码 E4021，请压缩或拆分文件。"),
    ("d03", "P2", "v3", "分享链接默认有效期为 7 天，企业版管理员可在后台将有效期延长至 90 天。"),
    ("d04", "P2", "v3", "分享链接可设置提取码，连续输错 5 次提取码后链接将被锁定 30 分钟。"),
    ("d05", "P3", "v3", "回收站中的文件保留 30 天，到期后自动永久删除，无法恢复。"),
    ("d06", "P3", "v3", "企业版管理员可在回收站设置中将保留期延长至 180 天。"),
    ("d07", "P4", "v3", "忘记登录密码时，点击登录页的“忘记密码”，通过绑定手机号获取验证码重置。"),
    ("d08", "P4", "v3", "账号连续登录失败 10 次会被临时冻结 1 小时，错误码为 E1007。"),
    ("d09", "P5", "v3", "同步客户端支持 Windows 10 及以上、macOS 12 及以上，以及主流 Linux 发行版。"),
    ("d10", "P5", "v3", "同步冲突时，客户端会保留两份文件，并在较新的文件名后追加“冲突副本”字样。"),
    ("d11", "P6", "v3", "个人免费版提供 15GB 存储空间，专业版 2TB，每月 19 元。"),
    ("d12", "P6", "v3", "企业版按席位计费，每席位每月 39 元，最少 5 个席位，提供审计日志与单点登录。"),
    ("d13", "P7", "v3", "数据传输使用 TLS 1.3 加密，静态数据使用 AES-256 加密存储。"),
    ("d14", "P7", "v3", "企业版支持客户自管密钥（BYOK），密钥可存放于第三方密钥管理服务。"),
    ("d15", "P8", "v3", "开放 API 的调用频率限制为每个应用每分钟 600 次，超出后返回错误码 E4290。"),
    ("d16", "P8", "v3", "API 鉴权使用 OAuth 2.0，访问令牌有效期为 2 小时，可用刷新令牌续期。"),
    ("d17", "P9", "v3", "文件版本历史默认保留最近 50 个版本，专业版与企业版保留 200 个版本。"),
    ("d18", "P9", "v3", "误删或误改文件后，可在版本历史中选择任意版本进行还原。"),
    ("d19", "P10", "v3", "离线访问需要先在移动端将文件标记为“可离线使用”，之后无网络时也能打开。"),
    ("d20", "P10", "v3", "移动端缓存占用空间可在设置中清理，清理缓存不会删除云端文件。"),
]
DOCS = [dict(id=i, parent=p, version=v, text=t) for i, p, v, t in CORPUS]

# 评估集： (问题, 正确块 id 集合, 元数据过滤)
EVAL = [
    ("错误码 E4021 是什么意思", {"d01"}, {"version": "v3"}),
    ("上传大文件被拒绝怎么办", {"d01"}, {"version": "v3"}),
    ("分享链接的有效期能改吗", {"d03"}, {"version": "v3"}),
    ("别人乱猜提取码会怎样", {"d04"}, {"version": "v3"}),
    ("删掉的文件还能找回吗", {"d05"}, {"version": "v3"}),
    ("回收站保留期最长能设成多久", {"d06"}, {"version": "v3"}),
    ("密码忘了怎么办", {"d07"}, {"version": "v3"}),
    ("E1007", {"d08"}, {"version": "v3"}),
    ("macOS 支持哪个版本", {"d09"}, {"version": "v3"}),
    ("企业版多少钱", {"d12"}, {"version": "v3"}),
    ("数据加密方式", {"d13"}, {"version": "v3"}),
    ("API 一分钟能调多少次", {"d15"}, {"version": "v3"}),
    ("E4290", {"d15"}, {"version": "v3"}),
    ("改错了文件怎么恢复到之前的样子", {"d18"}, {"version": "v3"}),
    ("离线怎么用文件", {"d19"}, {"version": "v3"}),
]


# ----------------------------------------------------------------------------
# 2. 分词：中文用字符 bigram（+ 单字），英文数字用整词（小写）
# ----------------------------------------------------------------------------
_STOP = set("的了是在和与及或吗呢吧啊怎么什么")


def tokenize(text: str) -> list[str]:
    text = text.lower()
    toks: list[str] = []
    for m in re.finditer(r"[a-z0-9\.]+|[\u4e00-\u9fff]+", text):
        s = m.group()
        if re.match(r"[a-z0-9]", s):
            toks.append(s)
        else:
            chars = [c for c in s if c not in _STOP]
            toks.extend(chars[i] + chars[i + 1] for i in range(len(chars) - 1))
    return toks


# ----------------------------------------------------------------------------
# 3. BM25（稀疏检索）
# ----------------------------------------------------------------------------
class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.tf = [Counter(tokenize(d["text"])) for d in docs]
        self.len = np.array([sum(c.values()) for c in self.tf], dtype=float)
        self.avgdl = self.len.mean()
        df = Counter(t for c in self.tf for t in c)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query, k=10, allow=None):
        q = tokenize(query)
        scores = []
        for i, c in enumerate(self.tf):
            if allow is not None and self.docs[i]["id"] not in allow:
                continue
            s = 0.0
            for t in q:
                if t in c:
                    f = c[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avgdl))
            if s > 0:
                scores.append((self.docs[i]["id"], s))
        return sorted(scores, key=lambda x: -x[1])[:k]


# ----------------------------------------------------------------------------
# 4. 玩具“稠密向量”：字符 n-gram 哈希嵌入。换成真实嵌入模型时，只需替换这个类。
# ----------------------------------------------------------------------------
class HashEmbedder:
    def __init__(self, dim=512):
        self.dim = dim

    def _vec(self, text):
        v = np.zeros(self.dim)
        chars = [c for c in text.lower() if not c.isspace() and c not in _STOP]
        grams = chars + [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
        for g in grams:
            h = hash_str(g)
            v[h % self.dim] += 1.0 if (h >> 20) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode(self, texts):
        return np.vstack([self._vec(t) for t in texts])


def hash_str(s: str) -> int:  # 稳定哈希（不依赖 PYTHONHASHSEED）
    h = 1469598103934665603
    for ch in s.encode("utf-8"):
        h = ((h ^ ch) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


class DenseIndex:
    def __init__(self, docs, embedder):
        self.docs, self.emb = docs, embedder
        self.mat = embedder.encode([d["text"] for d in docs])

    def search(self, query, k=10, allow=None):
        sims = self.mat @ self.emb.encode([query])[0]
        order = np.argsort(-sims)
        out = []
        for i in order:
            if allow is not None and self.docs[i]["id"] not in allow:
                continue
            out.append((self.docs[i]["id"], float(sims[i])))
            if len(out) >= k:
                break
        return out


# ----------------------------------------------------------------------------
# 5. RRF 融合（第 3 章）与 玩具重排序（第 4 章）
# ----------------------------------------------------------------------------
def rrf(lists, k=60, top_n=20):
    sc = defaultdict(float)
    for lst in lists:
        for rank, (doc_id, _) in enumerate(lst, start=1):
            sc[doc_id] += 1.0 / (k + rank)
    return sorted(sc.items(), key=lambda x: -x[1])[:top_n]


class ToyReranker:
    """用“逐词 IDF 加权覆盖率 + 连续短语奖励”模拟交叉编码器的细粒度交互。
    真实场景请换成 bge-reranker / Cohere Rerank 等模型。"""

    def __init__(self, bm25: BM25):
        self.idf = bm25.idf

    def score(self, query, text):
        qt, dt = tokenize(query), set(tokenize(text))
        if not qt:
            return 0.0
        tot = sum(self.idf.get(t, 1.0) for t in qt)
        hit = sum(self.idf.get(t, 1.0) for t in qt if t in dt)
        cover = hit / tot
        # 连续两个 query 词在文本中也相邻出现，视为短语命中，给小奖励
        phrase = sum(1 for a, b in zip(qt, qt[1:]) if a in dt and b in dt) / max(1, len(qt) - 1)
        return cover + 0.2 * phrase

    def rerank(self, query, cands, by_id, top_n=5):
        sc = [(i, self.score(query, by_id[i]["text"])) for i, _ in cands]
        return sorted(sc, key=lambda x: -x[1])[:top_n]


# ----------------------------------------------------------------------------
# 6. 元数据过滤（第 11 章）与 父子分块（第 10 章）
# ----------------------------------------------------------------------------
def allowed_ids(docs, flt):
    if not flt:
        return None
    return {d["id"] for d in docs if all(d.get(k) == v for k, v in flt.items())}


def parent_expand(top_ids, by_id, docs, flt=None, max_parents=2):
    """命中子块 → 取回其父章节（同一 parent 下所有块），按命中顺序去重。
    注意：展开父块时也必须应用元数据过滤，否则会把被过滤掉的旧版本内容带回上下文！"""
    allow = allowed_ids(docs, flt)
    seen, out = [], []
    for i in top_ids:
        p = by_id[i]["parent"]
        if p not in seen:
            seen.append(p)
            out.append((p, [d["text"] for d in docs
                            if d["parent"] == p and (allow is None or d["id"] in allow)]))
        if len(out) >= max_parents:
            break
    return out


# ----------------------------------------------------------------------------
# 7. 评估（第 20 章）
# ----------------------------------------------------------------------------
def evaluate(name, retrieve_fn, ks=(1, 3, 5)):
    hits = {k: 0 for k in ks}
    rr = 0.0
    misses = []
    for q, rel, flt in EVAL:
        ranked = [i for i, _ in retrieve_fn(q, flt)]
        for k in ks:
            hits[k] += bool(rel & set(ranked[:k]))
        pos = next((r for r, i in enumerate(ranked, 1) if i in rel), None)
        rr += 1.0 / pos if pos else 0.0
        if not pos or pos > 1:
            misses.append((q, pos))
    n = len(EVAL)
    row = {f"R@{k}": hits[k] / n for k in ks}
    row["MRR"] = rr / n
    return name, row, misses


def main():
    by_id = {d["id"]: d for d in DOCS}
    bm25 = BM25(DOCS)
    dense = DenseIndex(DOCS, HashEmbedder())
    rer = ToyReranker(bm25)

    def r_dense(q, flt):  return dense.search(q, 20)
    def r_bm25(q, flt):   return bm25.search(q, 20)
    def r_hybrid(q, flt): return rrf([dense.search(q, 20), bm25.search(q, 20)])
    def r_hybrid_rerank(q, flt):
        return rer.rerank(q, r_hybrid(q, flt)[:20], by_id, top_n=5)
    def r_full(q, flt):   # 元数据预过滤 + 混合 + 重排序
        allow = allowed_ids(DOCS, flt)
        c = rrf([dense.search(q, 20, allow), bm25.search(q, 20, allow)])
        return rer.rerank(q, c[:20], by_id, top_n=5)

    configs = [
        ("① 仅向量（Naive RAG）", r_dense),
        ("② 仅 BM25", r_bm25),
        ("③ 混合 RRF", r_hybrid),
        ("④ 混合 + 重排序", r_hybrid_rerank),
        ("⑤ 元数据过滤 + 混合 + 重排序", r_full),
    ]
    print(f"{'配置':<28}{'R@1':>7}{'R@3':>7}{'R@5':>7}{'MRR':>8}")
    print("-" * 58)
    last_misses = None
    for name, fn in configs:
        n, row, misses = evaluate(name, fn)
        print(f"{n:<26}{row['R@1']:>7.2f}{row['R@3']:>7.2f}{row['R@5']:>7.2f}{row['MRR']:>8.3f}")
        last_misses = misses
    print("\n提示：仅 15 个问题，1 题 ≈ 6.7 个百分点，差异在噪声范围内；真实项目请用 100+ 条评估集。")
    print("\n[⑤ 仍未排第 1 的问题]")
    for q, pos in last_misses:
        print(f"  - {q}  → 正确块名次：{pos}")

    # 父子分块演示
    q = "错误码 E4021 是什么意思"
    top = [i for i, _ in r_full(q, {"version": "v3"})][:1]
    print(f"\n[父子分块演示] 问题：{q}\n  命中子块：{top}")
    for p, texts in parent_expand(top, by_id, DOCS, {"version": "v3"}):
        print(f"  返回父块 {p}（共 {len(texts)} 个子块）：")
        for t in texts:
            print("   ·", t)


if __name__ == "__main__":
    main()
