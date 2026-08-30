#!/usr/bin/env python3
"""present 探针协议（gold=1）——absent_probe_harness 的姊妹篇：只含方法，不含语料和题目。

**出处：外部贡献者「星迟 & 烬」（公开仓库 PR #2，2026.08.02 合并为 `69ee4c0`）**，
原样收录其协议与实现。**下文的「我」「我这边」指的是贡献者，不是本项目**——
那几处第一人称是原话，保留不改，署出处正是为了让它有主语。

动机（2026.08.02 span2_k2 复测那轮）：零标注 present 尺子要求语料带 index/timeline
分层，换一份真实语料就可能直接失效（在我的 1144 条上「对 0 条」，present 侧一格都量
不出来）。gold=1 的手写题不依赖语料结构，谁都能在自己的语料上量 present——代价是
题要人写，所以护栏必须跟 absent 侧一样严。

协议三条（与 absent 侧对偶）：
1. 每道题带一句「金标原话」（gold）：答案块里真实存在的一段原文，不是你记忆里的大意。
2. 计分前先验证：gold **必须在语料全文里逐字存在**，不存在 → SKIP 作废——
   absent 护栏抓的是「我以为语料里没有」，present 护栏抓的是「我以为语料里有」。
   两边抓的都是出题人自己。
3. 命中 = gold 出现在 retrieve() 返回的 topN 文本里。空手、或端回一堆不含 gold 的
   候选，都算 miss（不看候选「像不像」，像不像是人判的，协议只认逐字）。

判据 A/B：--gate 借 gate_experiment.apply_gate 换闸重跑**同一份题**，两列并排。
present 的代价具体掉在哪道题上直接可见——我这边 span2_k2 掉的那 1/20 恰好是改述
形态的题（语料里换了说法，凑不出两段独立字面证据），跟合成集改述列的信号对得上。
逐题对照就是为了这种「代价藏在哪一格」的问题存在的。

用法（要自己接两处）：
  1. load_chunks() 接上你的语料加载；
  2. PRESENT_PROBES 换成你自己写的题（下面两道是示例占位，**不是真题**）。
  python3 present_probe_harness.py [--gate span2_k2] [--embed] [--topn 5]
"""
import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from memory_retrieval import MemoryIndex  # noqa: E402
from gate_experiment import build_gates, apply_gate  # noqa: E402


# ── 换成你自己的语料加载 ──
def load_chunks() -> list[str]:
    """返回语料块正文列表。示例：JSONL 每行 {"content": ...}"""
    path = Path(__file__).parent / "corpus.jsonl"
    return [json.loads(l)["content"] for l in open(path, encoding="utf-8")]


# ── 换成你自己写的题：(问题, 金标原话)。示例占位，不是真题 ──
PRESENT_PROBES = [
    ("我那台旧望远镜最后修好了吗", "目镜找配件配齐了"),
    ("陶艺课那次我拉坏了几个坯", "拉坏了三个坯"),
]


def screen(chunks, probes):
    """护栏：gold 逐字不在语料里的题作废。返回 (留用, 作废明细)。

    ⚠ **本项目收录时把这段从 main() 里提出来了**（原版护栏写在 main 内）。
    理由是本文件 docstring 自己写的那句「护栏必须跟 absent 侧一样严」——
    而 absent 侧的护栏在 `run_probes()` 函数内部、绕不过去，原版这边只要有人
    import 本模块直接调 `run_arm()`，护栏就整个被跳过了。提成函数之后
    selftest 才测得到它（第 1、2 项断言），行为与原版逐字一致。"""
    alltext = "\n".join(chunks)
    kept, dropped = [], []
    for q, gold in probes:
        if gold and gold in alltext:
            kept.append((q, gold))
        else:
            dropped.append(q)
    return kept, dropped


def run_arm(idx, probes, gate, topn):
    """一条判据臂：返回 {问题: True/False}。gate=None 表示现状不换闸。"""
    results = {}
    with apply_gate(gate):
        for q, gold in probes:
            res = idx.retrieve(q, topN=topn)
            results[q] = gold in "\n".join(r["text"] for r in res)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", help="要 A/B 的判据名（见 gate_experiment.build_gates）")
    ap.add_argument("--embed", action="store_true", help="用真 embedding（默认零依赖档）")
    ap.add_argument("--topn", type=int, default=5)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    chunks = load_chunks()

    # 护栏：gold 逐字不在语料里的题作废，不计分（实现见 screen()）
    probes, dropped = screen(chunks, PRESENT_PROBES)
    skipped = len(dropped)
    for q in dropped:
        print(f"SKIP(gold 在语料里逐字找不到) {q}")
    if not probes:
        print("没有可计分的题——先按协议第 1 条把 gold 换成语料里真实存在的原话。")
        return

    idx = MemoryIndex(embed=args.embed)
    for c in chunks:
        idx.add(c, {})
    idx.build()

    base = run_arm(idx, probes, None, args.topn)
    arm = None
    if args.gate:
        gates = build_gates()
        if args.gate not in gates:
            print(f"未知判据 {args.gate}；可选：{', '.join(sorted(gates))}")
            return
        arm = run_arm(idx, probes, gates[args.gate], args.topn)

    head = "现状"
    print(f"\n{'题目':<24} {head:>4}" + (f" {args.gate:>10}" if arm else ""))
    for q, _ in probes:
        row = f"{q[:22]:<24} {'HIT' if base[q] else 'MISS':>4}"
        if arm:
            row += f" {'HIT' if arm[q] else 'MISS':>10}"
        print(row)
    n = len(probes)
    total = f"\n[present gold=1] 现状 {sum(base.values())}/{n}"
    if arm:
        total += f" | {args.gate} {sum(arm.values())}/{n}"
    total += f"（SKIP {skipped} 道被护栏拦下）"
    print(total)


def selftest():
    """验的是协议本身，不需要真实语料。合成语料全部虚构。

    ⚠ **变异靶心**：把 screen() 里 `gold in alltext` 那条去掉，第 1 项必红——
    那是这套协议唯一不能省的一步（present 侧抓的是「我以为语料里有」）。"""
    chunks = [
        "周三下午去学了陶艺，拉坯拉坏了三个，老师说手太急。",
        "楼下那家螺蛳粉店换了老板，味道淡了不少。",
    ]

    # 1. 护栏抓住出题人：gold 在语料里逐字找不到 → 作废，不进分母
    kept, dropped = screen(chunks, [("陶艺课拉坏了几个", "拉坏了四个坯")])
    assert (kept, dropped) == ([], ["陶艺课拉坏了几个"]), \
        "gold 逐字不在语料里，这道题必须作废——护栏没生效"

    # 2. 反向也要守：gold 逐字存在的题必须留用（护栏不能拧成什么都作废）
    kept, dropped = screen(chunks, [("陶艺课拉坏了几个", "拉坏了三个")])
    assert (len(kept), dropped) == (1, []), "gold 逐字存在的题必须计分"

    idx = MemoryIndex(embed=False)
    for c in chunks:
        idx.add(c, {})
    idx.build()

    # 3.【协议核心】命中判定只认逐字，不认「像不像」——换个说法的 gold 必须算 miss，
    #    哪怕检索把那一块原原本本端回来了。这条是本协议区别于人判的全部所在。
    hit = run_arm(idx, [("陶艺课拉坏了几个坯", "拉坏了三个")], None, 5)
    near = run_arm(idx, [("陶艺课拉坏了几个坯", "弄坏了三个")], None, 5)
    assert hit["陶艺课拉坏了几个坯"] is True, "gold 逐字在返回文本里，该判 HIT"
    assert near["陶艺课拉坏了几个坯"] is False, \
        "近似说法被判成了命中——协议只认逐字，认了「像不像」这把尺子就没有意义了"

    # 4. 判据切换真的生效、且出块必须还原（借 gate_experiment 的闸，不另抄一份）
    before = MemoryIndex.lexical_admit
    gates = build_gates()
    name = "span4" if "span4" in gates else sorted(gates)[0]
    with apply_gate(gates[name]):
        assert MemoryIndex.lexical_admit is not before, "判据没被换上去"
    assert MemoryIndex.lexical_admit is before, \
        "出块没还原——漏了这一步，后面每一臂量的都是上一个判据"

    print("selftest ok（4项断言：护栏抓住出题人 / 干净题照常计分 / "
          "命中只认逐字不认像不像 / 判据换得上且还原得回）")


if __name__ == "__main__":
    main()
