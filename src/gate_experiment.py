#!/usr/bin/env python3
"""词面候选闸的**判据对照实验**——判据可切换，一条命令出表。

## 这个文件为什么存在

任务卡「区分性token离开bigram层」定的取数方式（**方案三，贡献者自己提的**）：
**新判据做出来 → 发给对方 → 对方在自己机器上跑 → 只回数字。**

⚠ **不许向任何人索取语料。** 别人的语料是他们和自己 AI 的对话，性质跟我们自己的
timeline 一样，"要过来"这个动作本身就是错的，加一句"不会外传"也不改变它。
所以这个文件的全部设计目标只有一个：**让别人在自己机器上跑得起来，只回数字给我们。**
它因此不内置任何语料、不内置任何题目（同 `absent_probe_harness.py` 的纪律）。

## 怎么跑

    # 全部判据跑一遍，出对照表
    python3 gate_experiment.py --corpus <你的语料目录> --probes <你的编造题.json>

    # 只跑一个判据
    python3 gate_experiment.py --corpus <...> --probes <...> --gate span4

    # 只跑合成回归集那把尺子（不需要任何语料，零配置）
    python3 gate_experiment.py

    # 自检（不需要语料）
    python3 gate_experiment.py --selftest

`--probes` 是 JSON：`[["编造的问题", ["区分词1", "区分词2"]], ...]`。
**区分词表是护栏的输入，不是装饰**：计分前会先验证这些词在你语料全文里一个都
不出现，出现即作废——护栏来自 `absent_probe_harness.py`，本文件**不另抄一份**，
也不绕过它（任务卡：不许绕过护栏）。

## 输出里有什么、没有什么

有：每个判据在 absent / present 两侧的聚合数字。
没有：任何语料正文。跑别人语料时 `absent` 侧默认静音（`--verbose` 才打印逐题），
因为逐题诊断会带出候选块的开头几十字。

## 判据都是什么

`bigram` 是现状（字符 bigram 的 df 闸），其余都是"离开 bigram 层"的候选：
`spanN` 把证据单元换成**查询与块共享的变长子串**（≥N 字），`*_k2` 额外要求
**共享至少两段**互不相同的证据，`jieba*` 是分词路线（要装 jieba，装不上会自动跳过
并说明——零依赖档仍是默认档，任何依赖路线都必须能不装它照常跑）。

**这个文件不改任何实现**，它只是把候选判据临时挂到 `MemoryIndex.lexical_admit`
上量一遍。量完就还原。判据选哪个是数据说了算，不是这个文件说了算。
"""
import argparse

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# stdout 的 UTF-8 包装交给 absent_probe_harness 那一处（它 import 时就做了）。
# **这里不能再包一层**：两层 TextIOWrapper 套同一个 buffer，先被回收的那个会把
# buffer 关掉，后面所有 print 全炸（第一版就是这么红的）。

from absent_probe_harness import run_probes                      # noqa: E402
from memory_retrieval import MemoryIndex, load_corpus, tokenize  # noqa: E402

SPAN_MAX = 8   # 只为省算力：更长的子串必是更短子串的超集，到 8 已经覆盖完


def _norm(text):
    """跟 tokenize 同一套清洗口径（清标点/空白、转小写），但不切 bigram。"""
    return re.sub(r"[^\w一-鿿]", "", text.lower())


def _restore(query_tokens):
    """现在的 `lexical_admit` 收的是**字符 bigram 列表**（`retrieve` 里
    `tokenize(query)` 传进来的），而候选判据要在整串上找变长子串。

    bigram 是逐字滑窗、两两重叠的，所以原串能无损还原：首个 token + 其余每个的末字。
    **这样做是为了不碰实现**——任务卡这一单明令"先做判定实验，不许直接改实现"，
    改 `retrieve` 的调用形态就已经是改实现了。判据将来真要落地时，
    该做的是把 `lexical_admit` 的入参改成原串，而不是永远留着这个还原步骤。
    """
    if isinstance(query_tokens, str):
        return _norm(query_tokens)
    toks = list(query_tokens)
    if not toks:
        return ""
    return toks[0] + "".join(t[-1] for t in toks[1:])


# ---------------- 候选判据 ----------------
#
# 每个工厂返回一个可以顶替 `MemoryIndex.lexical_admit` 的函数：query → set(块下标)。
# 现状那条直接用类里原来的实现，**不重写一份**——重写等于给对照组也埋一次改写风险。

def _units_span(lo):
    """长度 ≥lo 的共享子串当证据单元。

    ⚠ **短查询会降级，这是判据的一部分，不是实现细节**：查询本身短于 lo 字时
    下界取查询全长（`min(lo, len(s))`），也就是"整句原样共享"。所以 `span4`
    这个名字在短查询上名不副实——它那时实际是 `span≥len(查询)`，比标称宽松，
    而且恰好宽松在短查询这一类上。

    保留降级而不是直接判空手，理由是第一轮量到过的一个真实后果：「保险丝」
    「望远镜」这类三字专名直接当查询是**最该命中**的形态，不降级的话它们一段
    证据都产不出来，会被判成空手——那是把最该命中的查询误杀掉。
    代价是表里 spanN 各行在短查询上不完全可比，跑真实语料时要看查询长度分布
    （index 摘要句都长，这条基本碰不到；换成真人短问句就会碰到）。"""
    def f(_self, s):
        l, h = min(lo, len(s)), min(SPAN_MAX, len(s))
        return {s[i:i + n] for n in range(l, h + 1) for i in range(len(s) - n + 1)}
    return f


def _units_jieba(nouns_only):
    def f(_self, s):
        import jieba
        if nouns_only:
            import jieba.posseg as pseg
            return {w for w, flag in pseg.cut(s) if len(w) >= 2 and flag.startswith("n")}
        return {w for w in jieba.cut(s) if len(w) >= 2}
    return f


def make_gate(units, k=1):
    """units：查询 → 候选证据单元集合；k：一个块要共享几段证据才有候选资格。

    df 上限**问闸本人**（`_distinctive_df()`），不抄一份会漂移的判据——
    同 `absent_probe_harness` 那条纪律。这一维本轮实测几乎不动指标，
    换层才是有效的那一维。"""
    def gate(self, query):
        max_df = self._distinctive_df()
        s = _restore(query)
        if not s:
            return set()
        idx = _prep(self)
        seen = defaultdict(int)
        for u in units(self, s):
            hit = idx._gx_cache.get(u)
            if hit is None:
                # 单元 → 含它的块集合，**按 index 缓存**：同一个单元在几百条查询
                # 里会反复出现，不缓存的话这张表要跑半小时，没人会真去跑它
                hit = {i for i, c in enumerate(idx._gx_chunks) if u in c}
                idx._gx_cache[u] = hit
            if 0 < len(hit) <= max_df:
                for i in hit:
                    seen[i] += 1
        return {i for i, n in seen.items() if n >= k}
    return gate


def build_gates():
    """→ {名字: 判据或 None}。None＝这条路线在本机跑不了（缺依赖），如实留个坑位。"""
    gates = {
        "bigram(现状)": None,          # None 在这里是特例：见 apply_gate
        # span2 **不等于** bigram(现状)：它是长度 2~8 的全部子串，证据单元集合比
        # 纯字符 bigram 大一截。留着这一行是为了让两者能真的逐格对照——
        # 第一轮的表把 span≥2 标成"＝现状层"，那个等号是错的。
        "span2": make_gate(_units_span(2)),
        "span3": make_gate(_units_span(3)),
        "span4": make_gate(_units_span(4)),
        "span2_k2": make_gate(_units_span(2), k=2),
        "span3_k2": make_gate(_units_span(3), k=2),
        "span4_k2": make_gate(_units_span(4), k=2),
    }
    try:
        import jieba  # noqa: F401
        import jieba.posseg  # noqa: F401
        gates["jieba"] = make_gate(_units_jieba(False))
        gates["jieba_k2"] = make_gate(_units_jieba(False), k=2)
        gates["jieba_名词"] = make_gate(_units_jieba(True))
    except ImportError:
        gates["jieba(未安装,跳过)"] = "missing"
    return gates


_ORIG_ADMIT = MemoryIndex.lexical_admit


class apply_gate:
    """with 块内把判据换成 gate，出块还原。名字 'bigram(现状)' 表示不换。"""

    def __init__(self, gate):
        self.gate = gate

    def __enter__(self):
        if self.gate is not None:
            MemoryIndex.lexical_admit = self.gate
        return self

    def __exit__(self, *exc):
        MemoryIndex.lexical_admit = _ORIG_ADMIT
        return False


def _prep(idx):
    """判据要按整串子串匹配，预先存一份归一化正文（每个 index 只算一次）。

    **必须是惰性的**：合成回归集那把尺子在自己内部建 index，本文件碰不到它，
    只能等判据第一次被调用时就地补上（第一版忘了，跑合成尺子当场 AttributeError）。"""
    if not hasattr(idx, "_gx_chunks"):
        idx._gx_chunks = [_norm(c) for c in idx.chunks]
        idx._gx_cache = {}
    return idx


# ---------------- present 侧的两把尺子 ----------------

def build_pairs(idx):
    """零标注 present 对：index 摘要句当查询、**同窗口的 timeline 块**当答案。

    靠的是语料自带的结构（同一场会话被写了两遍），所以不需要任何人工标注、
    也不需要读语料正文。语料没有 index/timeline 分层时返回空——**那时就如实说
    这把尺子在你的语料上不适用**，不硬凑一个假的出来。"""
    by_window = defaultdict(list)
    for i, m in enumerate(idx.meta):
        if m.get("window") is not None and m.get("layer") == "timeline":
            by_window[m["window"]].append(i)
    pairs = []
    for i, m in enumerate(idx.meta):
        if m.get("layer") != "index" or m.get("window") is None:
            continue
        gold = set(by_window.get(m["window"], []))
        if not gold:
            continue
        for s in re.split(r"[。！？\n]", idx.chunks[i]):
            s = s.strip()
            if len(s) >= 12:
                pairs.append((s[:60], gold, i))
    return pairs


def score_pairs(idx, pairs, topN=5):
    """→ (误杀率, hit@1, recall@5)。误杀＝这条查询一条候选都没拿到。"""
    if not pairs:
        return None
    weights0 = list(idx.weights)
    killed = h1 = r5 = 0
    for q, gold, src in pairs:
        res = [r for r in idx.retrieve(q, topN=topN + 1) if r["id"] != src][:topN]
        if not res:
            killed += 1
            continue
        h1 += res[0]["id"] in gold
        r5 += any(r["id"] in gold for r in res)
    idx.weights = weights0          # 用进废退是副作用，量完必须还原，否则下一个判据不公平
    n = len(pairs)
    return killed / n, h1 / n, r5 / n


def lcs_len(a, b, cap=40):
    """最长公共子串长度（够用即可）。"""
    for n in range(min(len(a), cap), 0, -1):
        for i in range(len(a) - n + 1):
            if a[i:i + n] in b:
                return n
    return 0


def _pctl(vals, k):
    vals = sorted(vals)
    n = len(vals)
    le = lambda x: sum(1 for v in vals if v <= x) / n
    return {"n": n, "≤2": le(2), "≤3": le(3), "≤4": le(4), "中位": vals[n // 2]}


def span_profile(pairs, idx):
    """**这把尺子自己合不合用**：查询与它的正确答案块最长共享多少字。

    上一轮判定实验最要紧的发现就出在这里：我们的两把 present 尺子分别落在待定
    阈值的两侧（真实语料中位共享十几字，合成集手写查询 100% ≤4 字），于是
    "长度门槛"这一维怎么调都像在两个极端之间跳。**先看这张分布，再看误杀率**
    ——分布告诉你这把尺子测不测得动你要判的那件事。

    ⚠ **三种口径一起给，因为 max 会被 gold 集大小顶高**（2026.08.02 验收指出）：
    `max` 对每条 present 对取"与所有 gold 块里共享最长的那个"，而它在 gold 数量上
    **单调递增**——gold 越多，最大共享字数天然越大。真实语料一个 window 平均十几个
    timeline 块，合成集手写题的 gold 少得多，直接比 max 等于把"语料性质"和
    "gold 集大小的算术后果"混在一格里。所以同时给：

      max    —— 老口径，留着可比
      中位   —— gold 内取中位数，**对 gold 集大小不敏感**，这是排掉混杂的那一格
      单个   —— 每条 present 对只取一个 gold（窗口内下标最小的那块），
                gold 数强行拉平到 1

    三种口径都还在同一个量级，这个发现才真的立住。"""
    if not pairs:
        return None
    mx, med, one = [], [], []
    for q, gold, _ in pairs:
        qn = _norm(q)
        ls = sorted(lcs_len(qn, _norm(idx.chunks[j])) for j in gold)
        mx.append(ls[-1])
        med.append(ls[len(ls) // 2])
        one.append(lcs_len(qn, _norm(idx.chunks[min(gold)])))
    return {"max": _pctl(mx, 4), "中位": _pctl(med, 4), "单个": _pctl(one, 4),
            "gold均值": sum(len(g) for _, g, _ in pairs) / len(pairs)}


def synth_scores():
    """第二把尺子：合成回归集（零配置，任何人都能跑）。"""
    try:
        import regression_set as RS
    except ImportError:
        return None
    s = RS.score(RS.build_index(scale="established"))
    g = lambda k, f: s.get(k, {}).get(f, float("nan"))
    return {"literal": g("literal", "recall"), "改述": g("paraphrase", "recall"),
            "linked": g("linked", "recall"), "感受": g("感受回忆", "recall"),
            "absent日常": g("absent_日常", "correct_empty")}


def _has_synth():
    """合成尺子在不在——**只 import，不跑分**。
    原来这里写的是 `synth_scores() is not None`，打一次表头就把全量合成集
    白跑一遍，跟专门加缓存提速的目的正好相反。"""
    try:
        import regression_set  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------- 主流程 ----------------

def run(corpus=None, probes=None, only=None, verbose=False, embed=False):
    gates = build_gates()
    if only:
        gates = {k: v for k, v in gates.items() if k == only or k.startswith(only)}
        if not gates:
            sys.exit(f"没有这个判据：{only}")

    chunks, idx0, pairs = None, None, []
    if corpus:
        idx0 = _prep(load_corpus(corpus, embed=embed))
        chunks = list(idx0.chunks)
        pairs = build_pairs(idx0)
        print(f"【语料】{len(chunks)} 块；零标注 present 对 {len(pairs)} 条"
              + ("" if pairs else "（没有 index/timeline 分层，这把尺子在你的语料上不适用）"))
        prof = span_profile(pairs, idx0)
        if prof:
            print(f"【尺子体检】present 对与正确答案块共享多少字"
                  f"（每条 present 对平均 {prof['gold均值']:.1f} 个 gold 块）")
            for kind in ("max", "中位", "单个"):
                d = prof[kind]
                print(f"   {kind:<4} 中位 {d['中位']:>3} 字   "
                      f"≤2 字 {d['≤2']:>6.1%}   ≤3 字 {d['≤3']:>6.1%}   ≤4 字 {d['≤4']:>6.1%}")
            print("   ↑ 三种口径一起看：max 会被 gold 集大小顶高，"
                  "「中位」「单个」两行才是排掉这个混杂之后的读数。")
            print("   这几行决定下面的 present 数字怎么读：全是长共享的尺子测不出"
                  "长度门槛的代价，全是短共享的尺子只会说不行。")
    probe_list = []
    if probes:
        probe_list = [(q, list(w)) for q, w in json.loads(Path(probes).read_text(encoding="utf-8"))]
        print(f"【编造题】{len(probe_list)} 道（护栏会在计分前逐题验证区分词）")
    print()

    head = f"{'判据':<16}"
    if probe_list and chunks:
        head += f"{'absent正确空手':>14}{'作废':>6}"
    if pairs:
        head += f"{'present误杀':>12}{'hit@1':>8}{'recall@5':>10}"
    if _has_synth():
        head += f"{'合literal':>10}{'合改述':>8}{'合linked':>9}{'合感受':>8}{'合absent':>9}"
    print(head)
    print("-" * 110)

    for name, gate in gates.items():
        if gate == "missing":
            print(f"{name:<16}（本机没装 jieba，这条路线跳过——零依赖档是默认档，"
                  f"依赖路线缺席不影响其余结论）")
            continue
        row = f"{name:<16}"
        with apply_gate(gate):
            if probe_list and chunks:
                ok, tested, skipped, _ = run_probes(chunks, probe_list,
                                                    embed=embed, verbose=verbose)
                row += f"{f'{ok}/{tested}':>14}{skipped:>6}"
            if pairs:
                # 复用同一个 index：单元缓存跨判据是共用的（它只取决于语料本身，
                # 不取决于判据），重建一次等于把缓存扔了
                k, h1, r5 = score_pairs(idx0, pairs)
                row += f"{k:>12.1%}{h1:>8.3f}{r5:>10.3f}"
            syn = synth_scores()
            if syn is not None:
                row += (f"{syn['literal']:>10.2f}{syn['改述']:>8.2f}{syn['linked']:>9.2f}"
                        f"{syn['感受']:>8.2f}{syn['absent日常']:>9.2f}")
        print(row, flush=True)

    print("\n把这张表（只有数字）回给我们就够了——语料、题目都留在你自己机器上。")


def selftest():
    """验的是本文件的机械部分，不需要语料：判据切换真的生效并且**真的会还原**、
    k≥2 真的比 k≥1 严、护栏是从 harness 走的不是另抄一份。"""
    chunks = ["周三下午去学了陶艺，拉坯拉坏了三个，老师说手太急。",
              "昨天猫把水杯打翻了，键盘遭殃，擦了半天。",
              "楼下那家螺蛳粉店换了老板，味道淡了不少。"]
    idx = MemoryIndex()
    for c in chunks:
        idx.add(c, {})
    idx.build()
    _prep(idx)

    # 1. 判据切换生效，且**出块必须还原**——还原漏了会污染后面每一个判据的数
    before = MemoryIndex.lexical_admit
    with apply_gate(make_gate(_units_span(4))):
        assert MemoryIndex.lexical_admit is not before, "判据没被换上去"
    assert MemoryIndex.lexical_admit is before, \
        "出块没还原——这一条漏了，后面所有判据量的都是上一个判据"

    # 2. span 越长越严，k 越大越严（方向性，不是具体数值）
    with apply_gate(make_gate(_units_span(2))):
        wide = MemoryIndex.lexical_admit(idx, "楼下那家螺蛳粉店")
    with apply_gate(make_gate(_units_span(4))):
        narrow = MemoryIndex.lexical_admit(idx, "楼下那家螺蛳粉店")
    with apply_gate(make_gate(_units_span(2), k=2)):
        k2 = MemoryIndex.lexical_admit(idx, "楼下那家螺蛳粉店")
    assert wide, "对照组是空集——三者全空时下面那条包含关系恒成立，等于没测"
    assert narrow <= wide and k2 <= wide, "更严的判据放行的块不该更多"

    # 3. 护栏是**走 harness 的那一份**：区分词在语料里出现过的题必须作废。
    #    这条不是重测 harness，是钉住"本文件没有绕过它另抄一份宽松的"。
    ok, tested, skipped, _ = run_probes(
        chunks, [("上次陶艺课我拉坏了几个坯", ["陶艺", "坯"])], verbose=False)
    assert (skipped, tested) == (1, 0), "护栏没生效——本文件绕过了 harness"

    # 4. 语料没有分层时 present 那把尺子如实缺席，不硬凑
    assert build_pairs(idx) == [], \
        "没有 index/timeline 分层却造出了 present 对——那是编的，不是量的"

    # 5.【隐私：默认输出一个字语料正文都不许有】这个文件的全部设计目标是"让别人
    #    在自己机器上跑、只回数字给我们"，那么"回给我们的东西不夹带正文"就不能
    #    只写在 docstring 里靠自觉——`absent_probe_harness.run_probes` 的 LEAK
    #    分支会打印命中块开头几十字，而**它的 verbose 默认是 True**：这里但凡
    #    漏传一次参数，别人回给我们的表里就夹着 TA 的私人对话。
    #    所以这条走**真正会泄漏的那条路**：造一份临时语料 + 一道必然 LEAK 的探针，
    #    按默认参数跑一遍 run()，捕获全部 stdout，断言语料正文一个字都不在里面。
    #    **变异：把 run() 的 verbose 默认改成 True，这条立刻红。**
    import contextlib
    import io as _io
    import json as _json
    import tempfile as _tf
    _P = Path
    with _tf.TemporaryDirectory() as _td:
        _d = _P(_td) / "corpus"
        (_d / "timeline").mkdir(parents=True)
        (_d / "index").mkdir(parents=True)
        (_d / "timeline" / "w01_2026-06-17.md").write_text(
            "## 陶艺课\n" + chunks[0] + "\n", encoding="utf-8")
        (_d / "index" / "w01.md").write_text(
            "## 第1窗摘要\n" + chunks[2] + "\n", encoding="utf-8")
        _pf = _P(_td) / "probes.json"
        # 这道题必须**真的命中**才测得到泄漏：区分词（单簧管）语料里没有，护栏放行；
        # 但句子其余部分（陶艺/拉坏）在语料里，于是一定 LEAK → 走那条打印正文的分支。
        # ⚠ 别把它换成一道会 EMPTY 的题——那样这条断言就是空转的（写的时候错过一次）
        _pf.write_text(_json.dumps([["陶艺课上我那把单簧管拉坏了吗", ["单簧管"]]],
                                   ensure_ascii=False), encoding="utf-8")
        _buf = _io.StringIO()
        with contextlib.redirect_stdout(_buf):
            run(corpus=str(_d), probes=str(_pf))
        _out = _buf.getvalue()
    for c in chunks:
        assert c[:10] not in _out, f"默认输出里出现了语料正文：{c[:10]}"

    print("selftest ok（5项断言：判据换得上且还原得回 / 更严的判据不放行更多 / "
          "护栏走 harness 没另抄 / 尺子不适用时如实缺席 / 默认输出不含语料正文）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", help="md 语料目录（不传就只跑合成回归集那把尺子）")
    ap.add_argument("--probes", help="编造题 JSON：[[问题, [区分词...]], ...]")
    ap.add_argument("--gate", help="只跑这一个判据")
    ap.add_argument("--verbose", action="store_true",
                    help="打印逐题诊断（会带出候选块开头几十字，跑别人语料时别开）")
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    run(corpus=a.corpus, probes=a.probes, only=a.gate, verbose=a.verbose, embed=a.embed)


if __name__ == "__main__":
    main()
