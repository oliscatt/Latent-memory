#!/usr/bin/env python3
"""
粗筛+重排序两阶段对比实验（设计笔记"待验证的关键问题"第 2 条，任务卡
"粗筛重排序两阶段"）。

待验证的假设原文："粗筛+重排序（rerank）两阶段可能比一次到位的搜索更准，值得
对比测试"——措辞是假设不是结论，所以本文件产出两样：
  1. 零依赖参考 reranker：lexical_rerank——字符 n-gram（n=2/3/4）覆盖率加权和，
     长 n-gram 权重高。粗筛的 bigram 层完全不看词序，"阳台的薄荷茶"和散落的
     "阳台…薄荷…"碎片在它眼里差不多；trigram/4-gram 命中就是短语级一致的词序
     信号。这比 bigram 贵，只对粗筛出的 topM 候选算——这正是两阶段存在的理由。
     真正的重排序该用 cross-encoder（query+候选一起进模型），这是它的零依赖
     代理，retrieve(reranker=...) 同一个槽将来直接换。
  2. A/B 评估工具：一阶段 vs 两阶段跑同一份 queries.json（格式同 chunking 实验），
     输出 hit@1/hit@k 对照表。真实语料上的结论本地跑了才算数，本仓库不含真实
     语料，selftest 只带合成用例。

**假设已被证伪（2026.08.01，真实私人语料 602 块 / 214 条自动构造的查询-答案对）**：

    一阶段（无重排）   hit@1=0.682  recall@5=0.939  MRR=0.789
    两阶段（本文件的 lexical_rerank）hit@1=0.640  recall@5=0.888  MRR=0.747

**每一项都更差**，不是持平也不是有得有失。所以"粗筛+重排序可能更准"这条挂了两天的
假设，对**我们这个零依赖 lexical reranker** 而言答案是否定的，不该再当成待办挂着，
也不该向用户推荐开启。

原因不难理解：`lexical_rerank` 用的是字符 n-gram 覆盖率，跟粗筛的 BM25 是同一族
信号（这跟"零依赖向量路跟 BM25 高度重合"是同一个病）。它不但没带来独立判断，还把
粗筛里 RRF 融合出来的多路共识（图谱路，以及 2026.08.24 之前权重第四路的贡献）
压成了单一的词面覆盖率分——
**重排序把已经融合好的信息又降维回去了**。

**保留这个可插拔槽，不保留这个默认建议**：`retrieve(reranker=...)` 的槽本身是对的，
将来换 cross-encoder（query+候选一起进模型，是真正独立的信号）从同一个口进来。
下次评估新 reranker 时，判据不是"比一阶段好"这么笼统，而是**它带来的信号是否独立于
粗筛已有的那几路**——不独立的重排序必然是降维。
现行主检索的 weight 已退出排序；这里保留的是当时实验的采集条件，不代表当前仍有权重第四路。

零依赖，stdlib only。
用法：
  python rerank_experiment.py --selftest
  python rerank_experiment.py --corpus <md目录> --queries <queries.json> [--embed] [--topM 20]
queries.json：[{"query": "...", "answer": "语料里逐字存在的子串"}, ...]
"""

import argparse
import json
import re
from pathlib import Path

# 同目录模块，import 不触发其 CLI
from memory_retrieval import MemoryIndex, load_corpus  # noqa: F401

# n-gram 权重：越长的命中越说明词序/短语一致，给的分越高。经验值，待真实评估再调
NGRAM_WEIGHTS = ((2, 1.0), (3, 2.0), (4, 4.0))


def _ngrams(text, n):
    s = re.sub(r"[^\w一-鿿]", "", text.lower())
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def lexical_rerank(query, texts):
    """零依赖词面精排：Σ w_n × (query 的 n-gram 命中率)，n=2/3/4。
    query 太短出不了某档 n-gram 时该档记 0，不崩；全部为 0 时 retrieve 的
    同分规则会保持粗筛序，精排"没意见"就不乱序。"""
    q_grams = [(w, _ngrams(query, n)) for n, w in NGRAM_WEIGHTS]
    scores = []
    for t in texts:
        s = 0.0
        for w, qg in q_grams:
            if qg:
                s += w * len(qg & _ngrams(t, len(next(iter(qg))))) / len(qg)
        scores.append(s)
    return scores


def evaluate(index, queries, k=3, reranker=None, coarse_topM=20):
    """hit@1/hit@k：answer 子串出现在检回 chunk 里算命中（同 chunking 实验口径）。
    评估结束把权重恢复原状——retrieve 的用进废退副作用会污染下一轮对比，
    一阶段/两阶段必须在同一权重起点上跑才公平。"""
    w0 = list(index.weights)
    hit1 = hitk = 0
    for q in queries:
        r = index.retrieve(q["query"], topN=k, reranker=reranker, coarse_topM=coarse_topM)
        texts = [x["text"] for x in r]
        if texts and q["answer"] in texts[0]:
            hit1 += 1
        if any(q["answer"] in t for t in texts):
            hitk += 1
    index.weights[:] = w0
    return hit1 / len(queries), hitk / len(queries)


def run(corpus_dir, queries_path, k=3, coarse_topM=20, embed=False):
    index = load_corpus(corpus_dir, embed=embed)
    queries = json.loads(Path(queries_path).read_text(encoding="utf-8"))
    b1, bk = evaluate(index, queries, k)
    r1, rk = evaluate(index, queries, k, reranker=lexical_rerank, coarse_topM=coarse_topM)
    print(f"索引 {len(index.chunks)} 块，{len(queries)} 条查询，粗筛 topM={coarse_topM}"
          + ("，真embedding" if embed else "，零依赖"))
    print(f"| 模式 | hit@1 | hit@{k} |")
    print("|---|---|---|")
    print(f"| 一阶段（三路RRF） | {b1:.0%} | {bk:.0%} |")
    print(f"| 两阶段（+词面精排） | {r1:.0%} | {rk:.0%} |")


# ---------- selftest（合成语料，全部虚构） ----------

# 粗筛盲区构造：0 号块散落着 query 的全部 bigram 且反复出现（bigram 层眼里它最像
# query），1 号块是真正的短语级命中。索引顺序故意让散落块在前。
_SYNTH = [
    "## 杂记\n阳台台的的薄薄荷荷茶。薄荷好养，阳台向东，薄荷要多晒。",   # 0 散落碎片
    "## 一句话\n阳台的薄荷茶配一本旧小说，是夏天最好的时刻。",           # 1 短语命中
    "## 修灯\n换了走廊的灯泡。",                                        # 2 无关
    "## 买伞\n梅雨季前买了折叠伞。",                                    # 3 无关
]
_QUERY = "阳台的薄荷茶"


def _build():
    idx = MemoryIndex()
    for t in _SYNTH:
        idx.add(t)
    return idx.build()


def _selftest():
    # 1. 精排分数：短语级命中三档 n-gram 全中（1+2+4=7），散落碎片只有 bigram 档
    s_scatter, s_phrase = lexical_rerank(_QUERY, [_SYNTH[0], _SYNTH[1]])
    assert abs(s_phrase - 7.0) < 1e-9 and abs(s_scatter - 1.0) < 1e-9, \
        f"短语命中该三档全中、散落只中 bigram 档：{s_phrase} vs {s_scatter}"

    # 2. 粗筛盲区实测：一阶段把散落碎片块排在短语命中块前面（bigram 不看词序，
    #    这不是 bug 是粗筛的结构性盲区——正是重排序阶段存在的理由）
    base = _build().retrieve(_QUERY, topN=2)
    assert base[0]["id"] == 0 and base[1]["id"] == 1, \
        f"一阶段该先返回散落块（粗筛盲区），实际 {[x['id'] for x in base]}"

    # 3.【变异靶心：长 n-gram 权重】两阶段把短语命中捞回第一
    two = _build().retrieve(_QUERY, topN=2, reranker=lexical_rerank, coarse_topM=4)
    assert two[0]["id"] == 1, f"精排该把短语命中捞回第一，实际 {[x['id'] for x in two]}"

    # 4. A/B 评估工具：同一份查询，一阶段 hit@1=0、两阶段 hit@1=1；评估不留权重副作用
    idx = _build()
    queries = [{"query": _QUERY, "answer": "阳台的薄荷茶配一本旧小说"}]
    b1, bk = evaluate(idx, queries, k=2)
    r1, rk = evaluate(idx, queries, k=2, reranker=lexical_rerank, coarse_topM=4)
    assert (b1, r1) == (0.0, 1.0), f"对照该是 0% vs 100%：{b1} vs {r1}"
    assert bk == rk == 1.0, "hit@2 两边都该命中"
    assert idx.weights == [1.0] * 4, "评估结束权重该恢复原状，不留用进废退副作用"

    # 5. 兜底：query 太短出不了 n-gram → 全 0 分不崩；配 retrieve 时保持粗筛序
    assert lexical_rerank("茶", [_SYNTH[0], _SYNTH[1]]) == [0.0, 0.0]
    short = _build().retrieve("茶", topN=2, reranker=lexical_rerank, coarse_topM=4)
    assert [x["id"] for x in short] == [x["id"] for x in _build().retrieve("茶", topN=2)], \
        "精排全 0 分＝没意见，保持粗筛序"

    print("selftest ok（5项断言：精排分数 / 粗筛盲区实测 / 两阶段捞回 / A/B工具 / 兜底）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--corpus", help="md 语料目录")
    ap.add_argument("--queries", help="queries.json")
    ap.add_argument("--topM", type=int, default=20)
    ap.add_argument("--embed", action="store_true", help="粗筛向量层用真 embedding（需 fastembed）")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    elif args.corpus and args.queries:
        run(args.corpus, args.queries, coarse_topM=args.topM, embed=args.embed)
    else:
        ap.print_help()
