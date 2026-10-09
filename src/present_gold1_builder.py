#!/usr/bin/env python3
"""present 侧 **gold=1** 尺子的构造器与护栏——只含方法，不含任何语料和题目。

## 这个文件为什么存在

任务卡「区分性token离开bigram层」的结论是「方向对，但**还不能改实现**——不是没找到
候选，是**没有一把尺子判得了**」。卡在哪，任务卡「present尺子gold1」说得很准：

- `gate_experiment.build_pairs()` 那把**零标注**尺子（index 摘要句当查询、同窗
  timeline 块当答案）每条查询平均对 **15.3 个** gold 块。只要其中**任意一块**恰好
  跟查询共享一段 ≥4 字，这条查询就既不算误杀、也照样进 recall@5。
  **它的宽松来自「每题给十几次机会」，不是语料性质**——所以它测不出长度门槛
  会杀掉什么。
- 合成回归集那把是 gold=1，测得出，但只有 206 块合成语料、手写查询 100% 共享
  ≤4 字，**它只会说不行**。

缺的东西因此说得很死：**落在真实语料上、每条查询只对一个正确答案块（gold=1）的
present 集**。本文件就是造这把尺子的工具。

## 为什么不能零标注构造（别再试一次）

零标注那条路已经量死了：那 320 条是「同一个人把同一场会话写了两遍」，中位共享
**13 个字**，≤4 字的只有 0.9%。**构造出来的必然共享长字面。**
所以这一单的成本认在这儿：**得有人真读语料写问题**，本文件只负责把「我觉得只有
一块」变成**可复现的机械判定**。

## gold=1 是构造出来的性质，不是标注出来的态度

每道题带一张**锚词表**（写题人认为「只有那一块讲了这件事」的凭据），计分前逐题验：

1. 锚词少于 2 个 → 作废。单个锚词钉不住一个块，"唯一"是碰巧不是构造。
2. 含**全部**锚词的块数 ≠ 1 → 作废（0 个＝锚点根本不存在，写错了；
   ≥2 个＝这件事被讲过不止一次，gold 不唯一）。
3. **同题近邻**：除 gold 外，还有别的块含了**一半以上**锚词 → 作废。
   这一条抓的是任务卡里那句「若有第二个块也讲同一件事，这条作废」——
   真实语料的 index 摘要层天然会把 timeline 里的事再讲一遍，**这正是老尺子
   gold 冗余的来源**，不拦住它，造出来的还是那把老尺子。

**护栏的价值就在于它抓写题的人自己**（上一单护栏拦下 5 道，外部反馈里那份报告
也被自己的护栏拦下 1 道——那是两份报告里最值得信的地方）。所以**作废要计数、要分原因、
要进报告，不许静默丢**。

## 隐私

同 `probe_guard.py` / `gate_experiment.py` 的纪律，并加严一处：
**题面本身也不打印**。absent 那边的题是编造的，打印无妨；这里的题是**真读语料
写出来的**，题面等价于语料内容。默认输出只有「第几题、哪一类、作废原因」，
足够写题人回去改，也不会把内容带进任何报告或 commit。
`--verbose` 才打印题面，**跑别人语料时不许开**。

## 怎么跑

    # 只跑护栏，出作废报告 + 尺子体检
    python3 present_gold1_builder.py --corpus <语料目录> --queries <你的题.json>

    # 护栏 + 用这把 gold=1 尺子重跑判据主表
    python3 present_gold1_builder.py --corpus <...> --queries <...> --table

    # 自检（不需要语料）
    python3 present_gold1_builder.py --selftest

`--queries` 是 JSON：`[{"style": "literal", "q": "问题", "anchors": ["锚词1", "锚词2"]}, ...]`
`style` 四类：`literal`（原话）/ `改述` / `linked`（跨块关联）/ `感受`（抽象归纳）。

## 这个文件不改任何实现

判据一行不碰：主表直接 import `gate_experiment` 的 `build_gates()` / `apply_gate`，
**不在本文件重写一份**。重写等于给对照组也埋一次改写风险，而且下一轮就对不上了。
"""
import argparse
import io
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# ⚠ **这里不能自己包一层 TextIOWrapper**：两个 wrapper 套同一个 buffer，先被回收的
# 那个把 buffer 关掉，后面所有 print 全炸（gate_experiment 的注释里写过同一件事）。
# stdout 改 UTF-8 交给 probe_guard（被 gate_experiment 带进来，import 时 reconfigure）。
# gate_experiment 也要顶层 import：自检里有一条要在 `redirect_stdout` 里跑整个 run()，
# 那时 `sys.stdout` 是 StringIO，改编码的那一步得发生在重定向之前。
import gate_experiment as GX                          # noqa: E402
from memory_retrieval import MemoryIndex, load_corpus  # noqa: E402

STYLES = ("literal", "改述", "linked", "感受")
MIN_PER_STYLE = 15          # 任务卡：四类各不少于 15 条
MIN_ANCHORS = 2             # 锚词少于这个数钉不住一个块

# 作废原因（进报告的那几档，别改字面——报告和验收都按它对）
R_少锚词 = "锚词<2"
R_无 = "gold=0"
R_多 = "gold≥2"
R_近邻 = "同题近邻"


def screen(chunks, items):
    """逐题过护栏 → (kept, dropped)。

    kept 每项：(style, q, gold下标)；dropped 每项：(序号, style, 原因, 附加数)。

    **顺序是护栏的一部分**：全部作废判定都在检索之前跑完（同
    `probe_guard` 那条纪律）——先检索再验证，等于允许一道无效题的结果
    影响判断，哪怕最后没计分，看的人已经看过那个数了。"""
    kept, dropped = [], []
    for n, it in enumerate(items, 1):
        style, q, anchors = it["style"], it["q"], list(it["anchors"])
        if len(anchors) < MIN_ANCHORS:
            dropped.append((n, style, R_少锚词, len(anchors)))
            continue
        # 含全部锚词的块 = 这道题声称的 gold
        full = [i for i, c in enumerate(chunks) if all(a in c for a in anchors)]
        if not full:
            dropped.append((n, style, R_无, 0))
            continue
        if len(full) > 1:
            dropped.append((n, style, R_多, len(full)))
            continue
        gold = full[0]
        # 同题近邻：除 gold 外还有块含一半以上锚词 → 这件事很可能被讲过第二遍
        half = math.ceil(len(anchors) / 2)
        near = sum(1 for i, c in enumerate(chunks)
                   if i != gold and sum(a in c for a in anchors) >= half)
        if near:
            dropped.append((n, style, R_近邻, near))
            continue
        kept.append((style, q, gold))
    return kept, dropped


def report_screen(kept, dropped, total, verbose=False, items=None):
    """作废必须进报告、必须分原因、必须能对上账。"""
    print(f"【护栏】收题 {total} 道 → 留用 {len(kept)}，作废 {len(dropped)}"
          f"（作废率 {len(dropped) / total:.1%}）" if total else "【护栏】收题 0 道")
    if dropped:
        for reason, cnt in Counter(d[2] for d in dropped).most_common():
            print(f"   作废·{reason:<8} {cnt} 道")
        print("   逐题（只给序号/类别/原因，题面等价于语料内容，默认不打印）：")
        for n, style, reason, extra in dropped:
            line = f"   #{n:<3} [{style:<7}] {reason}（{extra}）"
            if verbose and items:
                line += f"  {items[n - 1]['q']}"
            print(line)
    by = Counter(s for s, _, _ in kept)
    print("【四类分布】" + "  ".join(
        f"{s} {by.get(s, 0)}" + ("" if by.get(s, 0) >= MIN_PER_STYLE else "⚠不足15")
        for s in STYLES))
    return by


def score(idx, kept, topN=5):
    """→ (误杀率, hit@1, recall@5)。误杀＝这条查询一条候选都没拿到。

    口径跟 `gate_experiment.score_pairs` 逐字对齐，**只有 gold 集大小不同**
    （那里是同窗全部 timeline 块，这里恒为 1）——两把尺子要能逐格对照，
    差异必须只剩这一处。"""
    if not kept:
        return None
    weights0 = list(idx.weights)
    killed = h1 = r5 = 0
    for _, q, gold in kept:
        res = idx.retrieve(q, topN=topN)
        if not res:
            killed += 1
            continue
        h1 += res[0]["id"] == gold
        r5 += any(r["id"] == gold for r in res)
    idx.weights = weights0      # 用进废退是副作用，量完必须还原，否则下一个判据不公平
    n = len(kept)
    return killed / n, h1 / n, r5 / n


def score_by_style(idx, kept, topN=5):
    """分类别再给一份——四类里 `改述`／`感受` 才是这一单的重点，混在一起会被
    `literal` 抬上去看不见。"""
    out = {}
    for s in STYLES:
        sub = [k for k in kept if k[0] == s]
        out[s] = score(idx, sub, topN=topN) if sub else None
    return out


def profile(idx, kept):
    """**这把尺子自己合不合用**：查询与它那唯一一个 gold 块最长共享多少字。

    老尺子的三口径（max/中位/单个）在这里退化成一格——gold 恒为 1，
    「gold 集大小顶高 max」这个混杂**从构造上就不存在了**，这正是这把尺子的意义。
    共享字数分布直接读：全是长共享 → 又是一把测不出长度门槛的尺子，白做。"""
    if not kept:
        return None
    ls = [GX.lcs_len(GX._norm(q), GX._norm(idx.chunks[g])) for _, q, g in kept]
    ls.sort()
    n = len(ls)
    le = lambda x: sum(1 for v in ls if v <= x) / n
    return {"n": n, "中位": ls[n // 2], "≤2": le(2), "≤3": le(3), "≤4": le(4)}


def run_table(idx, kept, only=None):
    """用这把 gold=1 尺子重跑判据主表。

    ⚠ **判据代码零改动**：`build_gates()` / `apply_gate` 原样 import 自
    `gate_experiment`，本文件不新增、不修改、不重写任何一个判据。"""
    gates = GX.build_gates()
    if only:
        gates = {k: v for k, v in gates.items() if k == only or k.startswith(only)}
        if not gates:
            sys.exit(f"没有这个判据：{only}")
    print(f"\n【主表·gold=1 尺子】{len(kept)} 条 present 查询，每条只对 1 个正确答案块")
    head = (f"{'判据':<16}{'误杀':>8}{'hit@1':>8}{'recall@5':>10}"
            f"{'literal':>9}{'改述':>7}{'linked':>8}{'感受':>7}")
    print(head)
    print("-" * 76)
    for name, gate in gates.items():
        if gate == "missing":
            print(f"{name:<16}（本机没装 jieba，这条路线跳过）")
            continue
        with GX.apply_gate(gate):
            k, h1, r5 = score(idx, kept)
            per = score_by_style(idx, kept)
        row = f"{name:<16}{k:>8.1%}{h1:>8.3f}{r5:>10.3f}"
        for s in STYLES:
            row += f"{per[s][2]:>9.2f}" if per[s] else f"{'—':>9}"
        print(row, flush=True)
    print("\n（四类列是各类的 recall@5；gold 恒为 1，所以这几个数跟合成集那四类同口径。）")


def privacy_scan(corpus, targets, n=12, embed=False):
    """**提交前的隐私自检**：语料正文有没有漏进留痕/报告/commit。

    任务卡验收判据最后一条要求"提交前跑一次全文扫描确认"。人眼看一遍不算扫描——
    这一单读了 791 块真实语料才写得出题，正文渗进文档的方式可以是转述里顺手抄的
    半句，不一定是整段粘贴。所以按 **{n} 字滑窗**做集合相交：语料的每个 {n}-gram
    与目标文件的每个 {n}-gram，交集必须为空。

    n 取 12：短于这个长度的重合会被常用短语和技术名词刷屏（"任务卡""回归集"
    这种词两边都有，且本来就该有），长于这个长度又会漏掉抄半句的情况。

    ⚠ **只有"纯汉字 12 连"算正文级命中**，这一条是第一版跑完才补上的，
    补的理由是第一版把自己判红了：这个仓库跟那份语料**本来就共享标识符**——
    语料里记的就是这个项目每天在干什么，`memory_retrieval`、`EMBED_HIT_FLOOR`、
    `commit message` 这些词两边都有、而且**本来就该有**。把它们算成泄漏，
    扫描就成了一份没人会真去跑的红报告，那比不扫还糟。
    所以分两档报：**正文级（纯汉字 12 连）必须为 0**，标识符级只报数不拦。
    **代价说清楚**：这一刀放过了英文原句的泄漏。目前这份语料是中文的，
    这个口子是知情的取舍，不是没想到；换成英文语料的人要把这一条改回全字符。

    → (正文级命中列表, 标识符级重合数)；**命中片段本身不打印**
    （打印它等于在扫描报告里再泄一次），只给"哪个文件、命中几段"。
    ⚠ **方向判不出来，这是这把扫描的第二个已知限制**：它只会说"两边有一段一样的
    话"，说不出是语料抄进了仓库、还是仓库里的工程结论被日记引用了过去。
    实测就撞上了后者——`changelog.md` / `后续计划.md` 里被判红的那几段，
    是这个项目自己的工程句子被那份日记记了一遍。所以这把扫描**该扫的是这一次
    新增的文字**（`--privacy-scan-diff`，只看 diff 里的 `+` 行），不是整仓库存量：
    存量里的重合得人去看方向，机器判不了。"""
    prose = lambda g: not any(c.isascii() and c.isalnum() for c in g)
    idx = load_corpus(corpus, embed=embed)
    grams, ids = set(), set()
    for c in idx.chunks:
        s = GX._norm(c)
        for i in range(len(s) - n + 1):
            g = s[i:i + n]
            (grams if prose(g) else ids).add(g)
    hits, id_hits = [], 0
    for t in targets:
        p = Path(t)
        if not p.is_file():
            continue
        try:
            s = GX._norm(p.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, OSError):
            continue
        c = sum(1 for i in range(len(s) - n + 1) if s[i:i + n] in grams)
        id_hits += sum(1 for i in range(len(s) - n + 1) if s[i:i + n] in ids)
        if c:
            hits.append((str(p), c))
    return hits, id_hits


def load_items(path):
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    for it in items:
        if it.get("style") not in STYLES:
            sys.exit(f"style 只能是 {STYLES} 之一，收到：{it.get('style')!r}")
    return items


def run(corpus, queries, table=False, only=None, verbose=False, embed=False):
    items = load_items(queries)
    idx = load_corpus(corpus, embed=embed)
    print(f"【语料】{len(idx.chunks)} 块")
    kept, dropped = screen(idx.chunks, items)
    report_screen(kept, dropped, len(items), verbose=verbose, items=items)
    p = profile(idx, kept)
    if p:
        print(f"【尺子体检】查询与它唯一 gold 块最长共享：中位 {p['中位']} 字   "
              f"≤2 字 {p['≤2']:.1%}   ≤3 字 {p['≤3']:.1%}   ≤4 字 {p['≤4']:.1%}")
        print("   ↑ 老尺子那三口径在这里退化成一格：gold 恒为 1，"
              "「max 被 gold 集大小顶高」这个混杂从构造上就没有了。")
    if table:
        run_table(idx, kept, only=only)
    print("\n把这几张表（只有数字）回给我们就够了——语料、题目都留在你自己机器上。")


# ---------------- 自检 ----------------

def _tiny_corpus():
    """三块正文里，「陶艺」这件事只在第 0 块讲过（gold=1 该成立的样子）。"""
    return ["周三下午去学了陶艺，拉坯拉坏了三个，老师说手太急。",
            "昨天猫把水杯打翻了，键盘遭殃，擦了半天。",
            "楼下那家螺蛳粉店换了老板，味道淡了不少。"]


def selftest():
    chunks = _tiny_corpus()

    # 1.【变异靶心】gold 唯一性检查**真的会作废重复命中的查询**。
    #    把 screen() 里 `len(full) > 1` 那一段去掉，这条立刻红——而那正是这把
    #    尺子唯一不能省的一步：不拦住它，造出来的还是老那把 gold 冗余的尺子。
    dup = chunks + ["上周又去了一次陶艺教室，还是拉坯，还是拉坏了三个。"]
    kept, dropped = screen(dup, [{"style": "literal", "q": "陶艺课拉坏了几个坯",
                                  "anchors": ["陶艺", "拉坯"]}])
    assert (kept, [d[2] for d in dropped]) == ([], [R_多]), \
        "两个块都讲了同一件事，这道题必须作废——gold 唯一性检查没生效"

    # 2. 反向也要守：护栏不能拧成"什么都作废"，干净题必须留用且 gold 指对块
    kept, dropped = screen(chunks, [{"style": "literal", "q": "陶艺课拉坏了几个坯",
                                     "anchors": ["陶艺", "拉坯"]}])
    assert (len(kept), len(dropped)) == (1, 0) and kept[0][2] == 0, \
        f"干净题必须留用、gold 必须是第 0 块，实际 {kept=} {dropped=}"

    # 3. 同题近邻：另一个块含了一半以上锚词就作废（真实语料里 index 摘要层
    #    天天这么干，**那正是老尺子 gold 冗余的来源**）
    near = chunks + ["那天陶艺课后来又去了一次，没细说。"]
    kept, dropped = screen(near, [{"style": "literal", "q": "陶艺课拉坏了几个坯",
                                   "anchors": ["陶艺", "拉坯"]}])
    assert (kept, [d[2] for d in dropped]) == ([], [R_近邻]), \
        "有第二个块也在讲这件事，这道题必须作废"

    # 4. 锚词不足、锚点不存在，各自有自己的原因，不混成一个"反正没过"
    kept, dropped = screen(chunks, [
        {"style": "literal", "q": "a", "anchors": ["陶艺"]},
        {"style": "改述", "q": "b", "anchors": ["单簧管", "目镜"]}])
    assert [d[2] for d in dropped] == [R_少锚词, R_无] and not kept, \
        "作废原因必须分档——只说作废，写题人不知道该改哪儿"

    # 5.【作废数进报告，不静默丢】账要对得上：留用 + 作废 = 收题；
    #    且报告里逐条原因的计数之和等于作废总数。
    items = [{"style": "literal", "q": "陶艺课拉坏了几个坯", "anchors": ["陶艺", "拉坯"]},
             {"style": "改述", "q": "b", "anchors": ["单簧管", "目镜"]},
             {"style": "linked", "q": "c", "anchors": ["猫"]}]
    kept, dropped = screen(chunks, items)
    assert len(kept) + len(dropped) == len(items), "有题既没留用也没作废——被静默丢了"
    buf = io.StringIO()
    import contextlib
    with contextlib.redirect_stdout(buf):
        report_screen(kept, dropped, len(items))
    out = buf.getvalue()
    assert f"作废 {len(dropped)}" in out, "作废总数没进报告"
    assert sum(int(l.split()[-2]) for l in out.splitlines() if "作废·" in l) == len(dropped), \
        "报告里分原因的计数之和对不上作废总数"

    # 6.【隐私：默认输出一个字题面、一个字正文都不许有】题面是真读语料写出来的，
    #    等价于语料内容。这条走真正会打印的那条路：造临时语料 + 一道必作废的题
    #    （作废那条是唯一会逐题打印的分支），按默认参数跑一遍 run()，
    #    捕获全部 stdout，断言正文与题面都不在里面。
    #    **变异：把 run() 的 verbose 默认改成 True，这条立刻红。**
    import tempfile
    q_text = "螺蛳粉店那次换老板之后味道怎么样"
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "corpus"
        (d / "timeline").mkdir(parents=True)
        (d / "timeline" / "w01_2026-06-17.md").write_text(
            "## 陶艺课\n" + chunks[0] + "\n" + chunks[2] + "\n", encoding="utf-8")
        qf = Path(td) / "q.json"
        qf.write_text(json.dumps(
            [{"style": "literal", "q": q_text, "anchors": ["单簧管", "目镜"]}],
            ensure_ascii=False), encoding="utf-8")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run(str(d), str(qf))
        out = buf.getvalue()
    assert q_text not in out, "默认输出里出现了题面——题面等价于语料内容"
    for c in chunks:
        assert c[:10] not in out, f"默认输出里出现了语料正文：{c[:10]}"

    # 7.【判据零改动】主表用的判据必须是 gate_experiment 里那一份，本文件不另抄；
    #    并且跑完必须还原（还原漏了，后面每一个判据量的都是上一个判据）。
    before = MemoryIndex.lexical_admit
    idx = MemoryIndex()
    for c in chunks:
        idx.add(c, {})
    idx.build()
    kept, _ = screen(chunks, [{"style": "literal", "q": "陶艺课拉坏了几个坯",
                               "anchors": ["陶艺", "拉坯"]}])
    with contextlib.redirect_stdout(io.StringIO()):
        run_table(idx, kept, only="span4")
    assert MemoryIndex.lexical_admit is before, \
        "主表跑完没还原判据——后面所有数都会是上一个判据的"
    assert set(GX.build_gates()) & {"span4", "span2"}, \
        "判据必须来自 gate_experiment.build_gates，本文件不许另抄一份"

    # 8.【隐私自检本身要能抓到东西】一个抄了半句语料的文件必须被抓出来，
    #    一个只有聚合数字的文件必须放行。**变异：把 n 调到 40，第一条立刻绿不了**
    #    ——那正是"扫描跑过了"和"扫描真的在扫"的区别。
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "corpus"
        (d / "timeline").mkdir(parents=True)
        (d / "timeline" / "w01_2026-06-17.md").write_text(
            "## 陶艺课\n" + chunks[0] + "\n", encoding="utf-8")
        dirty = Path(td) / "脏.md"
        dirty.write_text("这一单的结论是：" + chunks[0][2:20] + "，所以不改实现。",
                         encoding="utf-8")
        clean = Path(td) / "净.md"
        clean.write_text("误杀 7.6%，hit@1 0.494，recall@5 0.557。", encoding="utf-8")
        assert [f for f, _ in privacy_scan(str(d), [str(dirty)])[0]] == [str(dirty)], \
            "抄了半句语料的文件没被抓出来——这条扫描是空转的"
        assert privacy_scan(str(d), [str(clean)])[0] == [], \
            "只有聚合数字的文件被误报了——扫描拧成这样没人会真去跑它"
        # 标识符不算正文：这一条钉住上面那句"两边本来就该有"不是随口说的
        ident = Path(td) / "标识符.md"
        ident.write_text("跑 `memory_retrieval.load_corpus` 就行。", encoding="utf-8")
        (d / "timeline" / "w02.md").write_text(
            "改了 memory_retrieval.load_corpus 这个函数。", encoding="utf-8")
        assert privacy_scan(str(d), [str(ident)])[0] == [], \
            "共享标识符被判成了泄漏——扫描会红成没人跑"

    print("selftest ok（8项断言：gold 唯一性真的作废重复命中 / 干净题照常留用 / "
          "同题近邻作废 / 作废原因分档 / 作废数进报告且账对得上 / "
          "默认输出无题面无正文 / 判据取自 gate_experiment 且跑完还原 / 隐私自检抓得到抄半句、放得过标识符）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", help="md 语料目录")
    ap.add_argument("--queries", help="gold=1 题 JSON：[{style,q,anchors}, ...]")
    ap.add_argument("--table", action="store_true", help="护栏之后再重跑判据主表")
    ap.add_argument("--gate", help="主表只跑这一个判据")
    ap.add_argument("--verbose", action="store_true",
                    help="打印作废题的题面（题面等价于语料内容，跑别人语料时别开）")
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--privacy-scan", nargs="+", metavar="文件",
                    help="隐私自检：这些文件里有没有语料正文（纯汉字12连相交）")
    ap.add_argument("--privacy-scan-diff", action="store_true",
                    help="提交前隐私自检：从 stdin 读 diff，只扫新增的 + 行")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if a.privacy_scan_diff:
        if not a.corpus:
            sys.exit("隐私自检要 --corpus")
        import tempfile as _tf
        added = "\n".join(l[1:] for l in sys.stdin.read().splitlines()
                          if l.startswith("+") and not l.startswith("+++"))
        with _tf.TemporaryDirectory() as td:
            f = Path(td) / "added.txt"
            f.write_text(added, encoding="utf-8")
            hits, id_hits = privacy_scan(a.corpus, [str(f)], embed=a.embed)
        if hits:
            sys.exit(f"隐私自检不通过：本次新增的文字里有 {hits[0][1]} 段语料正文")
        print(f"隐私自检通过：本次新增 {len(added)} 字，正文级命中 0 段"
              f"（纯汉字 12 连）；标识符级重合 {id_hits} 段——那是项目自己的"
              f"文件名与常数名，两边本来就该有，不算泄漏。")
        return
    if a.privacy_scan:
        if not a.corpus:
            sys.exit("隐私自检要 --corpus")
        hits, id_hits = privacy_scan(a.corpus, a.privacy_scan, embed=a.embed)
        if hits:
            for f, c in hits:
                print(f"⚠ 正文级命中 {c} 段：{f}")
            sys.exit(f"隐私自检不通过：{len(hits)} 个文件里有语料正文")
        print(f"隐私自检通过：{len(a.privacy_scan)} 个文件，正文级命中 0 段"
              f"（纯汉字 12 连）；标识符级重合 {id_hits} 段——那是项目自己的"
              f"文件名与常数名，两边本来就该有，不算泄漏。")
        return
    if not (a.corpus and a.queries):
        sys.exit("要 --corpus 和 --queries（或者 --selftest）")
    run(a.corpus, a.queries, table=a.table, only=a.gate,
        verbose=a.verbose, embed=a.embed)


if __name__ == "__main__":
    main()
