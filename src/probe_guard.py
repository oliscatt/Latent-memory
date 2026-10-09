#!/usr/bin/env python3
"""探针护栏与命中门槛标定——只含方法，不含任何语料、题目和 key。

## 护栏防的是什么

探针题是人写的。写 absent 题的人相信"这件事语料里没有"，写 present 题的人相信
"答案就在那一块"。这两种相信不先验证就计分，量出来的就不只是检索，还掺着写题人的
记性：absent 题问到了语料里其实有的事，检索端回东西本是对的，却被记成"漏"；present
题的答案块其实有好几块，或者一块都不存在，命中和误杀就都失去了意义。

所以护栏是计分的**前一步**：整批题先过完护栏、作废的题记账，才轮到第一次检索。
先检索再作废，哪怕最后没计分，看的人也已经看过那个数了。

## absent 侧（本文件）

每道题带一张**事实词表**：这道题之所以是"编造"，落在哪几个词上（"上个月搬去海边住"
里是「海边」，"那台咖啡机修好了吗"里是「咖啡机」）。

1. **词表为空 → 未验证**。没有词就没有东西可验，"没查出问题"不等于"验过没问题"——
   这类题单独计数，不进计分分母，也不算通过护栏。整句都是功能词的编造题（"那次我们说好的
   事后来怎么样了"）就落在这一档，它们恰是最真实的威胁，所以更要如实标成"没验过"。
2. **词表里任何一个词在任何一块里出现 → 作废**，记下是哪个词、出现在几块（只记块数，
   不带正文）。比对前两边做同一套规整：NFKC（全角半角统一）、转小写、去掉全部空白——
   块里写成「科目 二」、跨了行、用了全角字母，照样算出现。逐块比，不把全库拼成一串，
   免得一个词被两块的首尾凑出来。
3. **通过的题才计分**：`retrieve()` 返回空列表＝正确空手；返回任何东西都算漏，不看候选
   "像不像"。每道题检索前把用进废退的权重恢复成同一份，题序不影响结果。
   漏了的题附一张**闸认下的区分性 token**（直接问 `MemoryIndex._distinctive_tokens`，
   判据以后怎么改，这张表跟着变），用来分辨是跨词边界的碎片开了闸，还是真撞上了稀有词。

**输出里不带语料正文**：哪怕 `verbose`，打印的也只是题面（编造题本来就不是语料）、
状态、返回条数和区分性 token。

## present 侧

gold=1 的 present 题走 `present_gold1_builder.screen()`：锚词至少两个、含全部锚词的块恰好
一块、没有同题近邻。那是本项目自己的 present 尺子，这里不另写一份。

## 命中门槛标定（`--floors`）

云端档换了 embedding 模型，命中门槛要用户自己量——`embedding_provider` 的标定表只收我们
自己量过的模型（目前只有 bge-small-zh-v1.5 的 0.45）。量法沿用我们定 0.45 时的做法：
**看两组余弦分布**，不反复跑整条检索。

- present：每道 gold=1 题，查询向量与它那唯一 gold 块的余弦；
- absent：每道通过护栏的编造题，查询向量与全库各块余弦的最大值。

每个候选门槛报两个比例：**present 被挡**（gold 余弦 ≤ 门槛，这道题的答案块进不了向量路）
与 **absent 放行**（最高余弦 > 门槛，向量路会给这道编造题递至少一块）。两个分布多半重叠，
没有干净间隙就只有权衡点：我们给 bge-small-zh-v1.5 定 0.45 时，接受的是真实查询被挡 2.8%。
挑好之后填环境变量 `MEMORY_EMBED_HIT_FLOOR`。门槛只管向量路；词面路照样会放行，
所以 absent 放行比例为 0 也不等于"库里没有就会空手"。

## 用法

    # 只过护栏（不检索），看作废与未验证各几道
    python probe_guard.py --corpus <语料目录> --absent <编造题.json> [--present <gold1题.json>] --screen-only

    # 过护栏后计分（零依赖档；--embed 走 MEMORY_EMBED_* 配好的 provider）
    python probe_guard.py --corpus <...> --absent <...> [--embed]

    # 门槛标定：要 --embed、要 present 题
    python probe_guard.py --corpus <...> --absent <...> --present <...> --embed --floors 0.30:0.70:0.02

    python probe_guard.py --selftest

`--absent` 是 JSON：`[["编造的问题", ["事实词1", "事实词2"]], ...]`；
`--present` 是 `present_gold1_builder` 的题格式：`[{"style": …, "q": …, "anchors": […]}, ...]`。
"""
import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):     # Windows 控制台默认 GBK；reconfigure 是原地改，叠几次都安全
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from memory_retrieval import MemoryIndex, load_corpus, tokenize  # noqa: E402

_SPACE = re.compile(r"\s+")


def _fold(text):
    """比对用的规整：NFKC、转小写、去空白。只用于判"出没出现"，不改任何人手上的文本。"""
    return _SPACE.sub("", unicodedata.normalize("NFKC", text).casefold())


def _hits(folded_chunks, terms):
    hits = {}
    for term in terms:
        t = _fold(term)
        n = sum(1 for c in folded_chunks if t in c)
        if n:
            hits[term] = n
    return hits


def term_hits(chunks, terms):
    """每个事实词出现在几块里 → {词: 块数}，只列出现过的。"""
    return _hits([_fold(c) for c in chunks], [t for t in terms if _fold(t)])


def screen_absent(chunks, probes):
    """整批过 absent 护栏 → (kept, voided, unverified)，序号从 1 起、按原题序。

    kept / unverified 每项 ``(序号, 问题, 词表)``；voided 每项 ``(序号, 问题, {词: 块数})``。
    这一步不碰检索，必须在第一次 retrieve() 之前跑完。"""
    folded = [_fold(c) for c in chunks]
    kept, voided, unverified = [], [], []
    for n, (q, terms) in enumerate(probes, 1):
        terms = [t for t in terms if _fold(t)]
        hits = _hits(folded, terms)
        if not terms:
            unverified.append((n, q, terms))
        elif hits:
            voided.append((n, q, hits))
        else:
            kept.append((n, q, terms))
    return kept, voided, unverified


def run_absent(idx, probes, verbose=False):
    """先整批过护栏，再逐题检索计分 → dict。

    ``correct``／``scored``：正确空手数／计分题数；``voided``／``unverified``：两档不计分的题数；
    ``rows``：按原题序，每项 ``{"n", "status", "returned", "evidence"}``，status 取
    EMPTY（正确空手）／LEAK（漏）／VOID（作废）／UNVERIFIED（词表为空）。"""
    kept, voided, unverified = screen_absent(idx.chunks, probes)
    rows = {n: {"n": n, "status": "VOID", "returned": None, "evidence": hits}
            for n, _q, hits in voided}
    rows.update({n: {"n": n, "status": "UNVERIFIED", "returned": None, "evidence": None}
                 for n, _q, _t in unverified})
    weights0 = list(idx.weights)
    correct = 0
    for n, q, _terms in kept:
        idx.weights[:] = weights0
        res = idx.retrieve(q, topN=5)
        if res:
            df = idx._bm25.df
            evidence = sorted((df.get(t, 0), t) for t in idx._distinctive_tokens(tokenize(q)))
            rows[n] = {"n": n, "status": "LEAK", "returned": len(res), "evidence": evidence}
        else:
            correct += 1
            rows[n] = {"n": n, "status": "EMPTY", "returned": 0, "evidence": None}
    idx.weights[:] = weights0
    out = [rows[n] for n in sorted(rows)]
    if verbose:
        qs = dict(enumerate((q for q, _ in probes), 1))
        for r in out:
            extra = ""
            if r["status"] == "LEAK":
                extra = f" 返回 {r['returned']} 条；闸认下的区分性 token：" + (
                    "、".join(f"{t}(df={d})" for d, t in r["evidence"][:8]) or "无")
            elif r["status"] == "VOID":
                extra = " 语料里其实有：" + "、".join(f"{t}（{k} 块）" for t, k in r["evidence"].items())
            print(f"#{r['n']:<3} {r['status']:<10} {qs[r['n']]}{extra}")
    return {"correct": correct, "scored": len(kept), "voided": len(voided),
            "unverified": len(unverified), "rows": out}


def floor_table(idx, present_kept, absent_kept, floors):
    """按两组余弦分布给每个候选门槛报 (门槛, present 被挡数, absent 放行数)。

    present_kept：`present_gold1_builder.screen()` 的留用项 (style, q, gold 下标)；
    absent_kept：`screen_absent()` 的留用项。只在真 embedding 档有意义。"""
    if not idx.embed:
        raise ValueError("门槛标定要真 embedding（--embed）：零依赖档的 bigram 余弦是另一套标度")
    qs = [q for _s, q, _g in present_kept] + [q for _n, q, _t in absent_kept]
    idx.prime_query_vectors(qs)          # 查询向量一次批量算完，之后全是本地点积
    gold_cos = [idx._vector_scores(q)[g] for _s, q, g in present_kept]
    absent_max = [max(idx._vector_scores(q), default=0.0) for _n, q, _t in absent_kept]
    idx.clear_primed_query_vectors()
    return [(f, sum(c <= f for c in gold_cos), sum(c > f for c in absent_max)) for f in floors]


def _parse_floors(spec):
    lo, hi, step = (float(x) for x in spec.split(":"))
    n = int(round((hi - lo) / step))
    return [round(lo + i * step, 4) for i in range(n + 1)]


def run(corpus, absent_path, present_path=None, embed=False, screen_only=False,
        floors=None, verbose=False):
    idx = load_corpus(corpus, embed=embed)
    probes = [(q, list(t)) for q, t in json.loads(Path(absent_path).read_text(encoding="utf-8"))]
    kept, voided, unverified = screen_absent(idx.chunks, probes)
    print(f"【absent 护栏】{len(probes)} 道 → 计分 {len(kept)}，作废 {len(voided)}，"
          f"未验证（词表为空）{len(unverified)}")
    for n, _q, hits in voided:
        print(f"   #{n:<3} 作废：" + "、".join(f"{t}（{k} 块）" for t, k in hits.items()))
    present_kept = []
    if present_path:
        import present_gold1_builder as PG   # 放在这里：PG → gate_experiment → 本文件，顶层 import 会成环
        items = json.loads(Path(present_path).read_text(encoding="utf-8"))
        present_kept, dropped = PG.screen(idx.chunks, items)
        PG.report_screen(present_kept, dropped, len(items))
    if screen_only:
        return
    res = run_absent(idx, probes, verbose=verbose)
    print(f"\n[absent] 正确空手 {res['correct']}/{res['scored']}"
          f"（作废 {res['voided']}、未验证 {res['unverified']}，均不进分母）")
    if floors:
        if not present_kept:
            raise SystemExit("门槛标定要 --present（gold=1 题），只看 absent 一侧选不出门槛")
        print(f"\n【门槛标定】{idx.provider.id}；present {len(present_kept)} 道、absent {len(kept)} 道")
        print(f"{'门槛':>6}  {'present 被挡':>12}  {'absent 放行':>11}")
        for f, blocked, passed in floor_table(idx, present_kept, kept, floors):
            print(f"{f:>6.2f}  {blocked:>5}/{len(present_kept):<6}  {passed:>4}/{len(kept)}")
        print("挑好之后：MEMORY_EMBED_HIT_FLOOR=<门槛>（门槛只管向量路，词面路照样会放行）")


def selftest():
    """不需要语料，夹具全部虚构。⚠ 每条断言后面写着它的变异靶心。"""
    chunks = ["周三下午去学了陶艺，拉坯拉坏了三个，老师说手太急。",
              "昨天猫把水杯打翻了，键盘遭殃，擦了半天。",
              "楼下那家牛肉面馆换了老板，味道淡了不少。科目 二\n考完了。"]
    probes = [("上次陶艺课我拉坏了几个坯", ["陶艺", "坯"]),      # 1 词在语料里 → 作废
              ("我那台天文望远镜的目镜配齐了吗", ["天文望远镜"]),  # 2 干净 → 计分
              ("那次我们说好的事后来怎么样了", []),               # 3 没有词 → 未验证
              ("科目二后来考过了吗", ["科目二"]),                  # 4 块里写成「科目 二」+换行 → 作废
              ("ＡＢＣ那家店还开着吗", ["abc"])]                   # 5 全角对半角，语料里没有 → 计分

    kept, voided, unverified = screen_absent(chunks, probes)
    # 1. 作废要说出是哪个词、几块。变异：去掉 hits 判断（所有题都留用）→ 红
    assert [v[0] for v in voided] == [1, 4] and voided[0][2] == {"陶艺": 1, "坯": 1}, voided
    # 2. 规整：「科目 二」跨空白换行照样算出现。变异：_fold 不去空白 → 第 4 题留用 → 红
    assert voided[1][2] == {"科目二": 1}
    # 3. 词表为空不算验过。变异：空词表落进 kept → 红
    assert [u[0] for u in unverified] == [3] and [k[0] for k in kept] == [2, 5]
    # 4. NFKC：全角词表能拦住半角语料
    assert term_hits(["店名是 ABC 小馆"], ["ＡＢＣ"]) == {"ＡＢＣ": 1}
    # 5. 逐块比，不让两块首尾拼出一个词
    assert term_hits(["今天下雨", "伞丢了"], ["雨伞"]) == {}

    idx = MemoryIndex(embed=False)
    for c in chunks:
        idx.add(c, {})
    idx.build()
    w0 = list(idx.weights)
    res = run_absent(idx, probes)
    # 6. 分母只有通过护栏的题；作废、未验证分别记账，五道一道不丢
    assert (res["scored"], res["voided"], res["unverified"]) == (2, 2, 1)
    assert [r["status"] for r in res["rows"]][:1] == ["VOID"] and len(res["rows"]) == 5
    assert res["rows"][2]["status"] == "UNVERIFIED"
    # 7. 漏了的题必须附闸认下的 token；证据取自闸本人
    leak_q = "楼下牛肉面馆换老板以后我去吃过几次"
    r = run_absent(idx, [(leak_q, ["吃过几次"])])
    assert r["rows"][0]["status"] == "LEAK", "夹具前提：这道题要真的漏，下面那条才测得到东西"
    assert {t for _d, t in r["rows"][0]["evidence"]} == idx._distinctive_tokens(tokenize(leak_q))
    # 8. 权重不留副作用（每题前恢复、量完恢复）
    assert idx.weights == w0, "run_absent 改动了用进废退权重，下一轮量的就不是同一个起点"
    # 9. 护栏在检索之前：作废题一次 retrieve 都不许触发。变异：先检索后作废 → 红
    calls = []
    orig = idx.retrieve
    idx.retrieve = lambda q, topN=5: calls.append(q) or orig(q, topN=topN)
    run_absent(idx, probes)
    del idx.retrieve
    assert set(calls) == {probes[1][0], probes[4][0]}, f"作废或未验证的题被检索了：{calls}"

    # 10. 默认不打印、verbose 也不带语料正文
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_absent(idx, probes + [(leak_q, ["吃过几次"])], verbose=True)
    out = buf.getvalue()
    assert "LEAK" in out and not any(c[:8] in out for c in chunks), "verbose 输出夹带了语料正文"

    # 11. 门槛表：用一个按关键字出向量的桩 provider，算得出就核得准
    class Stub:
        id, model = "stub:floor", "stub"

        def hit_floor(self):
            return None

        def embed(self, texts, is_query=False):     # provider 的契约是交回单位向量
            from embedding_provider import normalize
            axes = ("陶艺", "猫", "面馆")
            return [normalize([1.0 if a in t else 0.1 for a in axes]) for t in texts]

    sidx = MemoryIndex(embed=True, provider=Stub())
    for c in chunks:
        sidx.add(c, {})
    sidx.build()
    table = floor_table(sidx, [("literal", "陶艺课", 0)], [(1, "猫和面馆", ["x"])], [0.5, 0.95])
    #   gold 余弦＝1.0 不被挡；absent 最高余弦≈0.78（「猫」「面馆」两轴都亮）：0.5 放行、0.95 不放行
    assert table == [(0.5, 0, 1), (0.95, 0, 0)], table
    assert _parse_floors("0.30:0.40:0.05") == [0.3, 0.35, 0.4]

    print("selftest ok（11 项：作废说出词与块数 / 空白换行照样算出现 / 词表为空记未验证 / "
          "NFKC 全半角 / 逐块比不跨块 / 分母只含通过护栏的题 / 漏题附闸认下的 token / "
          "权重不留副作用 / 护栏先于检索 / 输出不带语料正文 / 门槛表按两组余弦分布计数）")


def main():
    ap = argparse.ArgumentParser(description="探针护栏与命中门槛标定")
    ap.add_argument("--corpus", help="md 语料目录")
    ap.add_argument("--absent", help='编造题 JSON：[["问题", ["事实词", ...]], ...]')
    ap.add_argument("--present", help="gold=1 题 JSON（present_gold1_builder 的格式）")
    ap.add_argument("--embed", action="store_true", help="用 MEMORY_EMBED_* 配好的 provider")
    ap.add_argument("--screen-only", action="store_true", help="只过护栏，不检索")
    ap.add_argument("--floors", help="门槛标定的扫描范围 起:止:步长，如 0.30:0.70:0.02")
    ap.add_argument("--verbose", action="store_true", help="逐题打印（只有题面与 token，不带正文）")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not (a.corpus and a.absent):
        ap.error("要 --corpus 和 --absent（或 --selftest）")
    run(a.corpus, a.absent, a.present, embed=a.embed, screen_only=a.screen_only,
        floors=_parse_floors(a.floors) if a.floors else None, verbose=a.verbose)


if __name__ == "__main__":
    main()
