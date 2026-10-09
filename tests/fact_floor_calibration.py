"""事实模式的噪音下限（LATENT_PASSIVE_FACT_FLOOR）按向量模型标定。

默认 0.42 只在 voyage-3.5 上量过；换了向量模型，余弦的整体高低会变（有的模型不相关的也普遍在 0.6 以上），
要在自己的事实库上重新量。两步：

1. score：对一组句子以 floor=0 跑事实模式，每句列出前 N 条事实的 id 与分数。准入、排除规则与线上同一判据
   （低信息碎话不捞；写入日不早于今天的、来源块已撤回／被取代／不在语料里的不递），只是不设下限、不看冷却。
2. 人工标注：在输出里给每句前 2 条各加一个 "相关": true／false（对着句子文件和事实库按 id 查原文）；
   拿不准的写 null，不进表、只报条数。
3. sweep：读标好的文件，按一组下限扫一遍，打出表：留下几条、其中相关几条、准确率、相关保留、不相关挡掉、
   有东西浮上来的句子。只看每句前 2 条，线上也只递这 2 条（同块兄弟行不在表里）。
   加 --margins 改扫前两名分差（线上 ready 响应里的 fact_margin）：一句的分差够了前 2 条都留、不够都不留，
   可再用 --floor 叠在某个下限之上。表头另有一行 AUC：相关排在不相关前面的概率，按分数、按分差各一个。
   加 --align-to <参照分数文件> [--ref-floor 0.42] 先按浮出比例对齐再比（换模型、改了分数分布时用）：算参照配置在
   参照下限上“有东西浮上来的句子”占比，在本文件扫的各档里挑占比最接近的那档（平手取大），表后单独报那一档。
   对齐只看分数、不看标注，参照文件可以是没标的 score 输出；只用于扫下限。

句子取自历史聊天时，行首写提问日加一个制表符（`2026-09-01<TAB>句子`）：写入日不早于那天的事实不算，
跟那句话当天线上能递的一致；不写就按今天算，事后从这段对话里提出来的事实会混进来。

只读：事实库、语料、`LATENT_PASSIVE_FACT_STATE_DIR` 里的向量缓存与冷却一律不写，可以直接指向线上那份，
也可以拷一份出来跑；缓存里没有的事实向量只在内存里现算（会调向量服务，按条计费的服务注意用量）。
**不打印事实正文和原句**：句子按行号、事实按 id 出现。向量提供方与服务同一套参数
（--embed-provider 或 MEMORY_EMBED_* 环境变量），换模型标定就换这里。

用法：
  python tests/fact_floor_calibration.py score --facts <事实库> --sentences <句子文件，一行一句> \\
      [--corpus <语料>] [--top 2] [--state-dir <LATENT_PASSIVE_FACT_STATE_DIR>] > 分数.json
  （在 分数.json 里标 "相关"）
  python tests/fact_floor_calibration.py sweep 分数.json [--floors 0.40 0.42 0.45 …]
  python tests/fact_floor_calibration.py sweep 分数.json --margins [0.05 0.10 …] [--floor 0.42]
  python tests/fact_floor_calibration.py sweep 新分数.json --align-to 参照分数.json [--ref-floor 0.42] [--floors …]
  python tests/fact_floor_calibration.py --selftest
"""
import argparse
import hashlib
import json
import os
import re
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from passive_facts import FACT_FLOOR, FACT_TOP, FactIndex, FactVectorCache      # noqa: E402

GRID = [0.0] + [round(0.30 + 0.02 * k, 2) for k in range(31)]
MARGIN_GRID = [round(0.002 * k, 3) for k in range(31)]   # voyage-3.5 上前两名分差几乎都在 0.06 以下
_DATED = re.compile(r"(\d{4}-\d{2}-\d{2})\t(.*)")


class _ReadOnlyCache(FactVectorCache):
    def save(self):
        """零写入：缓存里缺的事实向量只在内存里现算，不落盘。"""


def score(facts, sentences, *, provider, corpus=None, state_dir=None, top=FACT_TOP):
    """sentences＝[(行号, 句子[, 提问日])]，没给提问日按今天。
    → {"条件": …, "句子": [{"句": 行号, "准入": 原因码, "前N": [{"id", "分数"}]}]}"""
    import passive_recall as pr
    from memory_retrieval import load_corpus
    index = None
    if corpus:
        index = load_corpus(corpus)                     # 不给向量档：只切块，不写缓存
        for name, load in ((".retractions.json", index.load_retractions),
                           (".supersessions.json", index.load_supersessions)):
            if (Path(corpus) / name).exists():
                load(Path(corpus) / name)
    cache = _ReadOnlyCache(str(Path(state_dir) / "facts"), provider.id) if state_dir else None
    facts_index = FactIndex(facts, provider=provider, cache=cache)
    service = pr.PassiveRecallService(index, fact_index=facts_index)   # 不给冷却：不看、不登记
    today, live = service._today(), service._live_blocks()

    def exclude_before(day):
        # 与 PassiveRecallService._fact_candidate 的 exclude 同一判据：写入日不早于那天的、来源块不现行的不递。
        def exclude(row):
            written = row["meta"].get("written") or row["meta"].get("local_date") or ""
            return not written or written >= day or not service._fact_source_live(row, live)
        return exclude

    out = []
    for n, text, *day in sentences:
        exclude = exclude_before(day[0] if day else today)
        # ponytail: 不读 short_trigger_terms 元数据——事实模式只在「整句是低信息词又被登记成许可短句」时才差这一步
        admitted, reason, _licensed = pr._input_gate(text, short_terms=(), index=index, require_topic=False)
        ranked = facts_index.ranked(text, exclude=exclude, top=top) if admitted else []
        out.append({"句": n, "准入": reason, "前N": [
            {"id": row["meta"].get("heading") or pr._chunk_key(row["text"]), "分数": round(s, 6)}
            for row, s in ranked]})
    return {"条件": {"向量提供方": provider.id, "事实条数": len(facts_index.rows), "句子数": len(sentences),
                     "每句列": top, "今天": "按行首提问日" if any(len(s) > 2 for s in sentences) else today,
                     "来源块现行检查": "开" if live is not None else "关（没给 --corpus）"},
            "句子": out}


def _auc(pairs):
    """[(打分, 相关)] → 相关排在不相关前面的概率，平手算一半；缺一边时 None。"""
    pos = [x for x, rel in pairs if rel]
    neg = [x for x, rel in pairs if not rel]
    if not pos or not neg:
        return None
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def sweep(data, floors=GRID, margins=None, floor=0.0):
    """→ (样本说明, [每档一行])。只看每句前 FACT_TOP 条；没标的报错，不当成不相关；标 null 的（拿不准）不进表。
    给了 margins 就扫前两名分差：一句的分差 ≥ m 时它的前 2 条（再过 floor）都留、否则都不留；只有 1 条的句子分差算无穷大。"""
    sentences = data["句子"]
    items = []
    for s in sentences:
        top = s["前N"]
        margin = round(top[0]["分数"] - top[1]["分数"], 6) if len(top) > 1 else float("inf")   # 不 round 时 0.3-0.2 < 0.1
        items += [(s["句"], item, margin) for item in top[:FACT_TOP]]
    missing = sum("相关" not in item or not isinstance(item["相关"], (bool, type(None))) for _n, item, _m in items)
    if missing:
        raise SystemExit(f"每句前 {FACT_TOP} 条里还有 {missing} 条没标 \"相关\": true／false（拿不准写 null）")
    unsure = sum(item["相关"] is None for _n, item, _m in items)
    items = [x for x in items if x[1]["相关"] is not None]
    relevant = sum(item["相关"] for _n, item, _m in items)
    irrelevant = len(items) - relevant
    gates = [(m, lambda item, margin, m=m: item["分数"] >= floor and margin >= m) for m in margins] \
        if margins is not None else [(f, lambda item, margin, f=f: item["分数"] >= f) for f in floors]
    rows = []
    for value, keep in gates:
        kept = [(n, item) for n, item, margin in items if keep(item, margin)]
        hit = sum(item["相关"] for _n, item in kept)
        rows.append({"floor": value, "留下": len(kept), "其中相关": hit,
                     "相关保留": (hit, relevant), "不相关挡掉": (irrelevant - (len(kept) - hit), irrelevant),
                     "有东西浮上来的句子": (len({n for n, _item in kept}), len(sentences))})
    auc = [_auc([(key(item, margin), item["相关"]) for _n, item, margin in items])
           for key in (lambda item, _m: item["分数"], lambda _item, margin: margin)]
    fmt = lambda x: "—" if x is None else f"{x:.3f}"
    note = (f"句子 {len(sentences)}，标注 {len(items)} 条（相关 {relevant}、不相关 {irrelevant}"
            + (f"；拿不准 {unsure} 条不进表" if unsure else "") + f"）；只看每句前 {FACT_TOP} 条，同块兄弟行不在表里"
            + (f"；按前两名分差扫，叠在下限 {floor:.2f} 之上" if margins is not None else "")
            + f"\nAUC（相关排在不相关前面的概率）：按分数 {fmt(auc[0])}，按分差 {fmt(auc[1])}")
    return note, rows


def _surfaced(data, floor):
    """有东西浮上来的句子：前 FACT_TOP 条里有一条 ≥ floor。只看分数、不看标注。→ (句数, 总句数)"""
    rows = data["句子"]
    return sum(any(x["分数"] >= floor for x in s["前N"][:FACT_TOP]) for s in rows), len(rows)


def align(data, ref, ref_floor, floors=GRID):
    """按浮出比例对齐：在 floors 里挑本文件浮出比例与参照在 ref_floor 上最接近的那档，平手取大
    （同《事实模式分差按后十名均值比较》第六节 #48 的取法）。→ (参照浮出, 那档, 本文件在那档的浮出)"""
    want = Fraction(*_surfaced(ref, ref_floor))
    best = min(floors, key=lambda f: (abs(Fraction(*_surfaced(data, f)) - want), -f))
    return _surfaced(ref, ref_floor), best, _surfaced(data, best)


def render(data, floors=GRID, margins=None, floor=0.0, ref=None, ref_floor=FACT_FLOOR):
    note, rows = sweep(data, floors, margins, floor)
    pct = lambda a, b: f"{a / b:.0%}" if b else "—"
    fl = lambda x: f"{x:.2f}" if round(x, 2) == x else f"{x:g}"    # 0.495 这类细档不被四舍五入成 0.50
    head = [f"| {'floor' if margins is None else '分差≥'} | 留下 | 其中相关 | 准确率 | 相关保留 | 不相关挡掉 | 有东西浮上来的句子 |",
            "|---|---|---|---|---|---|---|"]

    def line(r):
        (kh, rel), (blk, irr), (ss, sn) = r["相关保留"], r["不相关挡掉"], r["有东西浮上来的句子"]
        return (f"| {fl(r['floor']) if margins is None else format(r['floor'], '.3f')} | {r['留下']} | {r['其中相关']} "
                f"| {pct(r['其中相关'], r['留下'])} | {kh}/{rel}（{pct(kh, rel)}） | {blk}/{irr}（{pct(blk, irr)}） | {ss}/{sn} |")
    lines = [note, ""] + head + [line(r) for r in rows]
    if ref is not None:
        (rs, rn), best, (bs, bn) = align(data, ref, ref_floor, floors)
        lines += ["", f"按浮出比例对齐（平手取大）：参照在下限 {fl(ref_floor)} 浮出 {rs}/{rn}（{pct(rs, rn)}），"
                      f"本文件最接近的是 {fl(best)}：{bs}/{bn}（{pct(bs, bn)}）", *head,
                  line(next(r for r in rows if r["floor"] == best))]
    return "\n".join(lines)


def main(argv=None, provider=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("score", help="以 floor=0 跑事实模式，列出每句前 N 条的 id 与分数")
    sc.add_argument("--facts", required=True, help="事实库目录或单个 facts.jsonl（服务的 LATENT_PASSIVE_FACTS）")
    sc.add_argument("--sentences", required=True,
                    help="句子文件，UTF-8，一行一句，空行跳过；输出按行号认句子。行首可写“提问日<TAB>”，按那天排除写入日")
    sc.add_argument("--corpus", help="服务的 --corpus；给了才核对来源块是否现行（与线上同一判据）")
    sc.add_argument("--top", type=int, default=FACT_TOP, help=f"每句列几条，默认 {FACT_TOP}（线上只递前 {FACT_TOP} 条）")
    sc.add_argument("--state-dir", default=os.environ.get("LATENT_PASSIVE_FACT_STATE_DIR")
                    or str(Path.home() / ".cache" / "latent-passive-facts"),
                    help="读这里的事实向量缓存（只读）；默认同服务")
    sc.add_argument("--embed-provider", help="同服务的 --embed-provider；不给就看 MEMORY_EMBED_* 环境变量")
    sw = sub.add_parser("sweep", help="读标好的 score 输出，按下限扫一遍")
    sw.add_argument("labelled", help="score 的输出，每句前 2 条已标 \"相关\": true／false")
    sw.add_argument("--floors", type=float, nargs="+", default=GRID, help="要扫的下限，默认 0 与 0.30～0.90 每 0.02 一档")
    sw.add_argument("--margins", type=float, nargs="*", help="改扫前两名分差；不跟值时 0～0.06 每 0.002 一档")
    sw.add_argument("--floor", type=float, default=0.0, help="扫分差时叠在哪个下限之上，默认 0（只看分差）")
    sw.add_argument("--align-to", help="参照配置的 score 输出（可以没标）：在本文件扫的各档里挑浮出比例与参照最接近的那档"
                                       "（平手取大），表后单独报")
    sw.add_argument("--ref-floor", type=float, default=FACT_FLOOR,
                    help=f"参照配置的下限，默认 {FACT_FLOOR}（同服务的 LATENT_PASSIVE_FACT_FLOOR）")
    args = ap.parse_args(argv)

    if args.cmd == "sweep":
        if args.align_to and args.margins is not None:
            ap.error("--align-to 只用于扫下限，不和 --margins 一起用")
        margins = None if args.margins is None else (args.margins or MARGIN_GRID)
        read = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
        print(render(read(args.labelled), args.floors, margins, args.floor,
                     read(args.align_to) if args.align_to else None, args.ref_floor))
        return
    if provider is None:
        from embedding_provider import resolve_provider
        provider = resolve_provider(args.embed_provider)
    lines = Path(args.sentences).read_text(encoding="utf-8").splitlines()
    sentences = [(n, *reversed(m.groups())) if (m := _DATED.fullmatch(line.strip())) else (n, line.strip())
                 for n, line in enumerate(lines, 1) if line.strip()]
    out = score(args.facts, sentences, provider=provider, corpus=args.corpus,
                state_dir=args.state_dir, top=args.top)
    print(json.dumps(out, ensure_ascii=False, indent=1))


def selftest():
    """自检判据 1a～1c、2d 与按浮出比例对齐。夹具全是虚构的；向量用字符袋假向量，不联网。"""
    import contextlib
    import io
    import tempfile
    import time
    from memory_retrieval import load_corpus
    from passive_facts import FactCooldown, _dot, _unit

    class CharBag:
        """字符袋假向量：共享的字越多越近。"""
        id = "selftest-char-bag"

        def embed(self, texts, is_query=False):
            out = []
            for text in texts:
                vec = [0.0] * 64
                for ch in text:
                    vec[ord(ch) % 64] += 1.0
                out.append(vec)
            return out

    def run(*argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(list(argv), provider=CharBag())
        return buf.getvalue()

    def digest(*roots):
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for root in roots for p in sorted(Path(root).rglob("*")) if p.is_file()}

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        corpus = td / "语料"
        (corpus / "timeline").mkdir(parents=True)
        (corpus / "timeline" / "2026-01-02.md").write_text(
            "## 2026-01-02\n\n周六在楼下面馆吃了一碗牛肉面，汤太咸。\n\n"
            "## 2026-01-03\n\n把阳台的绿萝换了一个蓝色的新花盆。\n", encoding="utf-8")
        live = [m["record_id"] for m in load_corpus(str(corpus)).meta]
        facts = [("fa", live[0], "2026-01-02", "周六她在楼下面馆吃了一碗牛肉面，说汤太咸。"),
                 ("fb", live[0], "2026-01-02", "她说下次去那家面馆要让老板少放盐。"),
                 ("fc", live[1], "2026-01-03", "她把阳台的绿萝换了一个蓝色的新花盆。"),
                 ("fd", live[1], "2026-01-03", "她给猫买了一个会发光的球，猫不理那个球。"),
                 ("fe", "0000gone0000", "2026-01-04", "她给猫买的发光球被猫拍到了床底下。"),  # 来源块不在语料：不递
                 ("ff", live[1], "2999-01-01", "她说猫明天会去玩那个发光的球。")]         # 写入日不早于今天：不递
        fact_dir = td / "事实库"
        (fact_dir / "全量-2026-01-05").mkdir(parents=True)
        (fact_dir / "全量-2026-01-05" / "facts.jsonl").write_text("".join(
            json.dumps({"id": i, "block": b, "fact": t, "written": w, "tag": "life", "kind": "event"},
                       ensure_ascii=False) + "\n" for i, b, w, t in facts), encoding="utf-8")
        sentences = ["那家面馆的牛肉面汤太咸了", "猫今天又不理那个发光的球", "嗯嗯", "量子色动力学的渐近自由"]
        sent_path = td / "句子.txt"
        # 第 5 句与第 2 句同文，只是行首带提问日 01-03：写在 01-03 的 fc、fd 不算。
        sent_path.write_text("\n".join(sentences) + "\n2026-01-03\t" + sentences[1] + "\n", encoding="utf-8")
        # 预置只含一条事实的向量缓存和一份冷却：工具要读缓存、补算缺的那几条，但都不许落盘。
        state = td / "state"
        state.mkdir()
        cache = FactVectorCache(str(state / "facts"), CharBag.id)
        cache.put(facts[0][3], _unit(CharBag().embed([facts[0][3]])[0]))
        cache.save()
        FactCooldown(str(state / "cooldown.json")).mark(["0123456789abcdef"], time.time())
        before = digest(fact_dir, state, corpus)

        args = ("score", "--facts", str(fact_dir), "--sentences", str(sent_path),
                "--corpus", str(corpus), "--state-dir", str(state))
        first, second = run(*args), run(*args)
        # 1a：前 2 条的 id 与分数；不出正文与原句；两次逐位相同。
        assert first == second, "1a：同一输入跑两次结果不逐位相同"
        assert not any(t in first for *_x, t in facts) and not any(s in first for s in sentences), \
            "1a：输出里出现了事实正文或原句"
        rows = json.loads(first)["句子"]
        assert [r["句"] for r in rows] == [1, 2, 3, 4, 5], rows
        assert {x["id"] for x in rows[4]["前N"]} == {"fa", "fb"} and json.loads(first)["条件"]["今天"] == "按行首提问日", \
            "2d：行首带提问日的句子要按那天排除写入日"
        assert rows[2]["准入"] == "low_information" and rows[2]["前N"] == [], "1a：低信息碎话线上不捞，这里也不该有分数"
        assert all(len(r["前N"]) == 2 for r in rows if r is not rows[2]), "1a：每句要列前 2 条"
        assert rows[0]["前N"][0]["id"] == "fa" and rows[1]["前N"][0]["id"] == "fd", rows
        assert not {x["id"] for r in rows for x in r["前N"]} & {"fe", "ff"}, \
            "1a：来源块不现行、写入日不早于今天的事实线上不递，这里也不该列"
        assert min(x["分数"] for x in rows[3]["前N"]) < FACT_FLOOR, "1a：floor=0，低于线上下限的也要列出"
        bag = CharBag()
        expect = _dot(_unit(bag.embed([sentences[0]])[0]), _unit(bag.embed([facts[0][3]])[0]))
        assert abs(rows[0]["前N"][0]["分数"] - expect) < 1e-5, "1a：分数不是原句与事实的余弦"
        # 1c：零写入——事实库、状态目录（向量缓存与冷却）、语料的文件与哈希都不变。
        assert digest(fact_dir, state, corpus) == before, "1c：跑完有文件被写了（缓存落盘或冷却被登记）"

        # 1b：虚构标注手算一档（floor=0.44）。只看每句前 2 条：
        #   留下 0.70✓ 0.50✗ 0.45✓ 0.44✗ = 4 条，其中相关 2，准确率 50%；
        #   相关共 2 条全留 → 2/2；不相关共 4 条（0.50 0.44 0.30 0.20），挡掉 0.30 0.20 → 2/4；
        #   有东西浮上来的句子：第 1、2 句 → 2/4（第 3 句被准入挡掉也算进分母）。
        labelled = {"句子": [
            {"句": 1, "准入": "fact_mode_signal", "前N": [{"id": "x1", "分数": 0.70, "相关": True},
                                                       {"id": "x2", "分数": 0.50, "相关": False}]},
            {"句": 2, "准入": "fact_mode_signal", "前N": [{"id": "x3", "分数": 0.45, "相关": True},
                                                       {"id": "x4", "分数": 0.44, "相关": False}]},
            {"句": 3, "准入": "low_information", "前N": []},
            {"句": 4, "准入": "fact_mode_signal", "前N": [{"id": "x5", "分数": 0.30, "相关": False},
                                                       {"id": "x6", "分数": 0.20, "相关": False},
                                                       {"id": "x7", "分数": 0.10}]}]}   # 第 3 条线上不递，不用标
        lab_path = td / "标注.json"
        lab_path.write_text(json.dumps(labelled, ensure_ascii=False), encoding="utf-8")
        table = run("sweep", str(lab_path), "--floors", "0.44")
        assert "| 0.44 | 4 | 2 | 50% | 2/2（100%） | 2/4（50%） | 2/4 |" in table, table
        # 2d：分差＝每句第 1 名减第 2 名：第 1 句 0.20、第 2 句 0.01、第 4 句 0.10。分差 ≥ 0.10 留第 1、4 句的前 2 条：
        #   留下 x1✓ x2✗ x5✗ x6✗ = 4、相关 1、准确率 25%；相关保留 1/2；不相关 4 条只挡掉 x4 → 1/4；浮上来的句子 2/4。
        #   叠在下限 0.42 上：只剩 x1✓ x2✗ → 2 条，相关保留 1/2，不相关挡掉 3/4，句子 1/4。
        margin_table = run("sweep", str(lab_path), "--margins", "0.10")
        assert "| 0.100 | 4 | 1 | 25% | 1/2（50%） | 1/4（25%） | 2/4 |" in margin_table, margin_table
        assert "| 0.100 | 2 | 1 | 50% | 1/2（50%） | 3/4（75%） | 1/4 |" in \
            run("sweep", str(lab_path), "--margins", "0.10", "--floor", "0.42")
        # AUC：按分数，相关 0.70、0.45 对不相关 0.50、0.44、0.30、0.20 赢 4＋3＝7/8；
        #      按分差，相关 0.20、0.01 对不相关 0.20、0.01、0.10、0.10 得 3.5＋0.5＝4/8（平手算一半）。
        assert "按分数 0.875，按分差 0.500" in table, table
        # 拿不准（null）不进表、报条数：x4 记 null，floor=0.44 留 x1 x2 x3 → 3 条，不相关 3 条挡掉 x5 x6 → 2/3。
        labelled["句子"][1]["前N"][1]["相关"] = None
        lab_path.write_text(json.dumps(labelled, ensure_ascii=False), encoding="utf-8")
        unsure = run("sweep", str(lab_path), "--floors", "0.44")
        assert "拿不准 1 条" in unsure and "| 0.44 | 3 | 2 | 67% | 2/2（100%） | 2/3（67%） | 2/4 |" in unsure, unsure
        del labelled["句子"][1]["前N"][1]["相关"]
        lab_path.write_text(json.dumps(labelled, ensure_ascii=False), encoding="utf-8")
        try:
            run("sweep", str(lab_path))
        except SystemExit as e:
            assert "1 条" in str(e.code), e.code
        else:
            raise AssertionError("1b：前 2 条里有没标的，应当报错而不是当成不相关")

        # 对齐（《标定工具按浮出比例对齐比较》第三节，先于实现写定）：参照的分数整体加 δ 当成“新配置”，
        #   在新配置扫的各档里挑浮出比例与参照最接近的那档，必须正好是参照下限 ＋ δ。参照只看每句前 2 条：
        #   第 1 句 0.70✓ 0.60✗、第 2 句 0.50✓ 0.40✗、第 3 句 0.45✗ 0.30✗、第 4 句 0.20✗ 0.10✗；
        #   参照下限 0.45 浮出第 1～3 句 → 3/4。新配置扫 0.30～0.80 每 0.01 一档，浮出 3/4 的是 0.30 到 0.45＋δ
        #   一整段：平手取大才落在 0.45＋δ，取小落在 0.30。
        #   参照在 0.45 的准确率 2/4；新配置准确率 2/4 的是 0.41＋δ～0.45＋δ 与 0.51＋δ～0.60＋δ 两段，
        #   改看准确率对齐会落在 0.60＋δ（平手取大）或 0.41＋δ（取小），都不是 0.45＋δ。
        ref = {"句子": [{"句": n, "准入": "fact_mode_signal",
                         "前N": [{"id": f"r{n}{k}", "分数": s, "相关": rel} for k, (s, rel) in enumerate(top)]}
                        for n, top in enumerate([[(0.70, True), (0.60, False)], [(0.50, True), (0.40, False)],
                                                 [(0.45, False), (0.30, False)], [(0.20, False), (0.10, False)]], 1)]}
        ref_path, new_path = td / "参照.json", td / "新配置.json"
        ref_path.write_text(json.dumps(ref, ensure_ascii=False), encoding="utf-8")
        grid = [f"{0.30 + 0.01 * k:.2f}" for k in range(51)]
        for delta in (0.07, -0.05):
            shifted = json.loads(json.dumps(ref))
            for s in shifted["句子"]:
                for item in s["前N"]:
                    item["分数"] = round(item["分数"] + delta, 6)
            new_path.write_text(json.dumps(shifted, ensure_ascii=False), encoding="utf-8")
            cmd = ("sweep", str(new_path), "--floors", *grid, "--align-to", str(ref_path), "--ref-floor", "0.45")
            out = run(*cmd)
            aligned, want = out[out.index("按浮出比例对齐"):], f"{0.45 + delta:.2f}"
            assert "参照在下限 0.45 浮出 3/4" in aligned and f"本文件最接近的是 {want}：3/4" in aligned, \
                f"对齐：δ={delta:+.2f} 应对齐到 {want}\n{aligned}"
            # 单独报那一档：留下 4 条（相关 2）、相关保留 2/2、不相关 6 条挡掉 4 条、浮出 3/4，与参照在 0.45 那行相同。
            assert f"| {want} | 4 | 2 | 50% | 2/2（100%） | 4/6（67%） | 3/4 |" in aligned, aligned
        # 参照只看分数：没标“相关”的 score 输出也能当参照。
        for s in ref["句子"]:
            for item in s["前N"]:
                del item["相关"]
        ref_path.write_text(json.dumps(ref, ensure_ascii=False), encoding="utf-8")
        assert f"本文件最接近的是 {want}：3/4" in run(*cmd), "对齐：参照文件没标也应当能对齐"
    print("selftest 事实下限标定：1a 1b 1c 2d 对齐 通过")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
