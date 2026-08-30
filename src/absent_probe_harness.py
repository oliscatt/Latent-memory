#!/usr/bin/env python3
"""absent 探针协议（护栏版）——只含方法，不含任何语料和题目。

**出处：外部贡献者 `lu7899112-source`（2026.08.02）**，原样收录其协议与诊断设计。
报告人在自己 1144 条中文语料上用这套东西量出了 `lexical_admit` 的 `absent` 正确空手率
**0/9**，并且**它抓过出题人自己一次**（一道编造题里的"科目二"在那份语料里真的存在，
被护栏拦下作废）——那一道是这份报告最值得信的地方，也是这套协议的全部价值所在。

我们自己的合成 absent 集缺的正是这一层：题是自己编的，"语料里确实没有"从来没被
验证过，只是被相信过。

**2026.08.03：这件事问到答案了，下面这段留作教训、结论已更新。**
维护者确认 PR #1／#2／#4 是同一个人、且维护者本人认识她——**代词是「她」，
维护者给的，不是推断的**。本文件与其它文档里已有的中性写法（「报告人」「对方」
「贡献者」，约 24 处）**不用回头批量改**，它们不算错；新写的直接用「她」即可。
仓库里只写公开身份（GitHub 账号与她自己的署名「星迟 & Ember」），**真名不入库**。

⚠ 原先那条「别顺手把『报告人』『对方』改成『他』或『她』」的依据是：PR #1 的全部
往返里判不出（对方三次发言都是中文技术陈述、第一人称「我」，我方回复全程「你」），
而那条线程 2026.08.02 已经结束、维护者定不再打扰，所以**以为不会再有答案**——
**答案不在 PR 文本里，在维护者那儿，这就是当时没想到的那一步。**
仓库历史里两种都出现过：08.01 采纳 PR 时写「他」，08.02 收录本文件时写「她」，
两轮各猜一次且互相矛盾——**至少有一处指错了一个真实的人**。
**教训照旧作数**：判不出只是多问一句，判错了没有任何人会知道。

协议三条（原文保留）：
1. 每道编造题必须带一张「区分词表」：这道题里真正承载事实的内容词。
2. 计分前先验证：区分词在语料全文里**一个都不出现**才计分；出现任何一个 → SKIP，
   这道题作废（"我以为语料里没有"不算数，护栏抓的就是出题人自己）。
3. 正确空手 = retrieve() 返回空列表。返回任何候选都算漏（不看候选"像不像"，
   像不像是人判的，协议只认空手）。

附带：开闸词诊断——每道题打印是哪些 token 以什么 df 通过了词面闸的区分性判据，
漏检时先看这张表再谈修法（**bigram 边界碎片** vs **真稀有词撞车**，是两种不同的病）。

**⚠ 我们相对原版改了一处，是对方自己点名的隐患**：原版把 `cap = max(3, N // 5)`
照抄进诊断，并附注"你们那边改了判据，这行要跟着改，不然表就不准了"。抄一个会漂移的
判据正是这个项目反复交学费的形状，所以这里改成**直接问闸本人**——
`idx._distinctive_df()`。判据以后怎么改，诊断表都跟着动，不可能对不上。
`df` 也不再自己重扫一遍语料，直接读 `idx._bm25.df`——**诊断必须看闸看到的那份数，
不是一份"应该一样"的数**。

用法（要自己接两处）：
  1. `load_chunks()` 接上你的语料加载；
  2. `ABSENT_PROBES` 换成你自己造的题（下面两道是示例占位，**不是真题**）。
  python3 absent_probe_harness.py

  自检（不需要语料，验的是护栏本身）：
  python3 absent_probe_harness.py --selftest
"""
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from memory_retrieval import MemoryIndex, tokenize  # noqa: E402


# ── 换成你自己的语料加载 ──
def load_chunks():
    """返回语料块正文列表。示例：JSONL 每行 {"content": ...}"""
    path = Path(__file__).resolve().parent.parent / "corpus.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(l)["content"] for l in f]


# ── 换成你自己造的题：(编造问题, [区分词, ...]) ──
# 区分词 = 这道题里承载"编造事实"的内容词。**示例占位，不是真题**：
ABSENT_PROBES = [
    ("我那台天文望远镜的目镜后来配齐了吗", ["天文望远镜", "目镜"]),
    ("上次陶艺课我拉坏了几个坯", ["陶艺", "坯"]),
]


def screen_probes(chunks, probes):
    """检索前一次性过护栏 → (留用, 作废)。

    留用每项为 ``(原序号, 问题, 区分词)``；作废每项为
    ``(原序号, 问题, 命中的区分词)``。单独导出这一层，是让别的正式量具复用同一把
    护栏，而不是各自在循环里抄一份看起来相同、以后会漂移的实现。
    """
    alltext = "\n".join(chunks)
    kept, dropped = [], []
    for n, (q, words) in enumerate(probes, 1):
        leaked = [w for w in words if w in alltext]
        target = dropped if leaked else kept
        target.append((n, q, leaked if leaked else list(words)))
    return kept, dropped


def run_probes(chunks, probes, embed=False, verbose=True, index=None):
    """跑一轮探针，返回 (正确空手数, 计分数, 被护栏拦下数, 明细)。

    明细每项：(状态, 问题, 附加信息)，状态 ∈ {"SKIP", "EMPTY", "LEAK"}。
    `index` 供判据实验传入已接好外部表的索引；不给时维持原来的建库路径。"""
    kept, dropped = screen_probes(chunks, probes)

    idx = index
    if idx is None:
        idx = MemoryIndex(embed=embed)
        for c in chunks:
            idx.add(c, {})
        idx.build()

    cap = idx._distinctive_df()        # 问闸本人，不抄它的判据
    dfs = idx._bm25.df                 # 读闸看到的那份 df，不自己重扫

    tested, ok, skipped = len(kept), 0, len(dropped)
    detail = []
    # 护栏必须在第一次 retrieve 前整批完成；不让无效题先影响权重、再在循环里作废。
    # 输出顺序仍按原题序，避免导出独立入口时顺手改坏旧调用方看到的 detail 顺序。
    dropped_by_id = {n: leaked for n, _q, leaked in dropped}
    for n, (q, _words) in enumerate(probes, 1):
        leaked = dropped_by_id.get(n)
        if leaked is not None:
            detail.append(("SKIP", q, leaked))
            if verbose:
                print(f"SKIP(语料里其实有: {','.join(leaked)})", q)
            continue
        res = idx.retrieve(q, topN=5)
        if not res:
            ok += 1
            detail.append(("EMPTY", q, None))
            if verbose:
                print("EMPTY✓", q)
        else:
            opener = sorted((dfs.get(t, 0), t) for t in set(tokenize(q))
                            if 0 < dfs.get(t, 0) <= cap)
            detail.append(("LEAK", q, opener))
            if verbose:
                print(f"LEAK({len(res)}条, 榜首:{res[0]['text'][:30]}…)", q)
                print("   开闸词:",
                      ", ".join(f"{t}(df={d})" for d, t in opener[:8]) or "无")
    return ok, tested, skipped, detail


def selftest():
    """验的是护栏本身，不需要真实语料。

    ⚠ **变异靶心**：把 `leaked` 那一段去掉（即不验证区分词是否真的不在语料里），
    第 1 项必红——那正是这套协议唯一不能省的一步。"""
    chunks = [
        "周三下午去学了陶艺，拉坯拉坏了三个，老师说手太急。",
        "昨天猫把水杯打翻了，键盘遭殃，擦了半天。",
        "楼下那家螺蛳粉店换了老板，味道淡了不少。",
    ]
    # 1. 护栏抓住"我以为语料里没有"——「陶艺」其实在第一块里
    kept, dropped = screen_probes(
        chunks, [("上次陶艺课我拉坏了几个坯", ["陶艺", "坯"])])
    assert not kept and dropped[0][0] == 1 and "陶艺" in dropped[0][2], \
        "独立护栏入口必须保留原序号并说出命中的区分词"
    ok, tested, skipped, detail = run_probes(
        chunks, [("上次陶艺课我拉坏了几个坯", ["陶艺", "坯"])], verbose=False)
    assert (skipped, tested) == (1, 0), \
        f"区分词在语料里出现过，这道题必须作废、不进分母，实际 {skipped=} {tested=}"
    assert detail[0][0] == "SKIP" and "陶艺" in detail[0][2], \
        "SKIP 必须说出是哪个词漏的——只说作废，出题人不知道该改哪儿"

    # 2. 干净的题照常计分（反向也要守：护栏不能拧成"什么都作废"）
    ok, tested, skipped, detail = run_probes(
        chunks, [("我那台天文望远镜的目镜配齐了吗", ["天文望远镜", "目镜"])],
        verbose=False)
    assert (skipped, tested) == (0, 1), \
        f"区分词一个都不在语料里，这道题必须计分，实际 {skipped=} {tested=}"

    # 3. 漏检时必须给出开闸词诊断，否则这份报告只能得出"松紧"这一种结论
    if detail[0][0] == "LEAK":
        assert detail[0][2] is not None, "LEAK 必须附开闸词表"

    # 4. 判据不许再被抄一份：诊断用的上限必须来自闸自己
    idx = MemoryIndex(embed=False)
    for c in chunks:
        idx.add(c, {})
    idx.build()
    assert idx._distinctive_df() == max(3, idx._bm25.N // 5), \
        "诊断上限必须等于闸的判据本身（这条不是重复实现，是防止两边悄悄分岔）"

    print("selftest ok（4项断言：护栏抓住出题人 / 说出漏的是哪个词 / "
          "干净题照常计分 / 诊断上限取自闸本人不另抄一份）")


def main():
    if "--selftest" in sys.argv:
        selftest()
        return
    chunks = load_chunks()
    ok, tested, skipped, _ = run_probes(chunks, ABSENT_PROBES)
    print(f"\n[absent] 正确空手 {ok}/{tested}（SKIP {skipped} 道被护栏拦下）")


if __name__ == "__main__":
    main()
