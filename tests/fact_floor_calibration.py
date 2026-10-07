"""事实模式的噪音下限（LATENT_PASSIVE_FACT_FLOOR）按向量模型标定。

默认 0.42 只在 voyage-3.5 上量过；换了向量模型，余弦的整体高低会变（有的模型不相关的也普遍在 0.6 以上），
要在自己的事实库上重新量。两步：

1. score：对一组句子以 floor=0 跑事实模式，每句列出前 N 条事实的 id 与分数。准入、排除规则与线上同一判据
   （低信息碎话不捞；写入日不早于今天的、来源块已撤回／被取代／不在语料里的不递），只是不设下限、不看冷却。
2. 人工标注：在输出里给每句前 2 条各加一个 "相关": true／false（对着句子文件和事实库按 id 查原文）。
3. sweep：读标好的文件，按一组下限扫一遍，打出表：留下几条、其中相关几条、准确率、相关保留、不相关挡掉、
   有东西浮上来的句子。只看每句前 2 条，线上也只递这 2 条（同块兄弟行不在表里）。

只读：事实库、语料、`LATENT_PASSIVE_FACT_STATE_DIR` 里的向量缓存与冷却一律不写，可以直接指向线上那份，
也可以拷一份出来跑；缓存里没有的事实向量只在内存里现算（会调向量服务，按条计费的服务注意用量）。
**不打印事实正文和原句**：句子按行号、事实按 id 出现。向量提供方与服务同一套参数
（--embed-provider 或 MEMORY_EMBED_* 环境变量），换模型标定就换这里。

用法：
  python tests/fact_floor_calibration.py score --facts <事实库> --sentences <句子文件，一行一句> \\
      [--corpus <语料>] [--top 2] [--state-dir <LATENT_PASSIVE_FACT_STATE_DIR>] > 分数.json
  （在 分数.json 里标 "相关"）
  python tests/fact_floor_calibration.py sweep 分数.json [--floors 0.40 0.42 0.45 …]
  python tests/fact_floor_calibration.py --selftest
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from passive_facts import FACT_TOP, FactIndex, FactVectorCache      # noqa: E402

GRID = [0.0] + [round(0.30 + 0.02 * k, 2) for k in range(31)]


class _ReadOnlyCache(FactVectorCache):
    def save(self):
        """零写入：缓存里缺的事实向量只在内存里现算，不落盘。"""


def score(facts, sentences, *, provider, corpus=None, state_dir=None, top=FACT_TOP):
    """sentences＝[(行号, 句子)]。→ {"条件": …, "句子": [{"句": 行号, "准入": 原因码, "前N": [{"id", "分数"}]}]}"""
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

    def exclude(row):
        # 与 PassiveRecallService._fact_candidate 的 exclude 同一判据：写入日不早于今天的、来源块不现行的不递。
        written = row["meta"].get("written") or row["meta"].get("local_date") or ""
        return not written or written >= today or not service._fact_source_live(row, live)

    out = []
    for n, text in sentences:
        # ponytail: 不读 short_trigger_terms 元数据——事实模式只在「整句是低信息词又被登记成许可短句」时才差这一步
        admitted, reason, _licensed = pr._input_gate(text, short_terms=(), index=index, require_topic=False)
        ranked = facts_index.ranked(text, exclude=exclude, top=top) if admitted else []
        out.append({"句": n, "准入": reason, "前N": [
            {"id": row["meta"].get("heading") or pr._chunk_key(row["text"]), "分数": round(s, 6)}
            for row, s in ranked]})
    return {"条件": {"向量提供方": provider.id, "事实条数": len(facts_index.rows), "句子数": len(sentences),
                     "每句列": top, "今天": today, "来源块现行检查": "开" if live is not None else "关（没给 --corpus）"},
            "句子": out}


def sweep(data, floors=GRID):
    """→ (样本说明, [每个下限一行])。只看每句前 FACT_TOP 条；没标的报错，不当成不相关。"""
    sentences = data["句子"]
    items = [(s["句"], item) for s in sentences for item in s["前N"][:FACT_TOP]]
    missing = sum(not isinstance(item.get("相关"), bool) for _n, item in items)
    if missing:
        raise SystemExit(f"每句前 {FACT_TOP} 条里还有 {missing} 条没标 \"相关\": true／false")
    relevant = sum(item["相关"] for _n, item in items)
    irrelevant = len(items) - relevant
    rows = []
    for floor in floors:
        kept = [(n, item) for n, item in items if item["分数"] >= floor]
        hit = sum(item["相关"] for _n, item in kept)
        rows.append({"floor": floor, "留下": len(kept), "其中相关": hit,
                     "相关保留": (hit, relevant), "不相关挡掉": (irrelevant - (len(kept) - hit), irrelevant),
                     "有东西浮上来的句子": (len({n for n, _item in kept}), len(sentences))})
    note = (f"句子 {len(sentences)}，标注 {len(items)} 条（相关 {relevant}、不相关 {irrelevant}）；"
            f"只看每句前 {FACT_TOP} 条，同块兄弟行不在表里")
    return note, rows


def render(data, floors=GRID):
    note, rows = sweep(data, floors)
    pct = lambda a, b: f"{a / b:.0%}" if b else "—"
    lines = [note, "", "| floor | 留下 | 其中相关 | 准确率 | 相关保留 | 不相关挡掉 | 有东西浮上来的句子 |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        (kh, rel), (blk, irr), (ss, sn) = r["相关保留"], r["不相关挡掉"], r["有东西浮上来的句子"]
        lines.append(f"| {r['floor']:.2f} | {r['留下']} | {r['其中相关']} | {pct(r['其中相关'], r['留下'])} "
                     f"| {kh}/{rel}（{pct(kh, rel)}） | {blk}/{irr}（{pct(blk, irr)}） | {ss}/{sn} |")
    return "\n".join(lines)


def main(argv=None, provider=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("score", help="以 floor=0 跑事实模式，列出每句前 N 条的 id 与分数")
    sc.add_argument("--facts", required=True, help="事实库目录或单个 facts.jsonl（服务的 LATENT_PASSIVE_FACTS）")
    sc.add_argument("--sentences", required=True, help="句子文件，UTF-8，一行一句，空行跳过；输出按行号认句子")
    sc.add_argument("--corpus", help="服务的 --corpus；给了才核对来源块是否现行（与线上同一判据）")
    sc.add_argument("--top", type=int, default=FACT_TOP, help=f"每句列几条，默认 {FACT_TOP}（线上只递前 {FACT_TOP} 条）")
    sc.add_argument("--state-dir", default=os.environ.get("LATENT_PASSIVE_FACT_STATE_DIR")
                    or str(Path.home() / ".cache" / "latent-passive-facts"),
                    help="读这里的事实向量缓存（只读）；默认同服务")
    sc.add_argument("--embed-provider", help="同服务的 --embed-provider；不给就看 MEMORY_EMBED_* 环境变量")
    sw = sub.add_parser("sweep", help="读标好的 score 输出，按下限扫一遍")
    sw.add_argument("labelled", help="score 的输出，每句前 2 条已标 \"相关\": true／false")
    sw.add_argument("--floors", type=float, nargs="+", default=GRID, help="要扫的下限，默认 0 与 0.30～0.90 每 0.02 一档")
    args = ap.parse_args(argv)

    if args.cmd == "sweep":
        print(render(json.loads(Path(args.labelled).read_text(encoding="utf-8")), args.floors))
        return
    if provider is None:
        from embedding_provider import resolve_provider
        provider = resolve_provider(args.embed_provider)
    lines = Path(args.sentences).read_text(encoding="utf-8").splitlines()
    sentences = [(n, line.strip()) for n, line in enumerate(lines, 1) if line.strip()]
    out = score(args.facts, sentences, provider=provider, corpus=args.corpus,
                state_dir=args.state_dir, top=args.top)
    print(json.dumps(out, ensure_ascii=False, indent=1))


def selftest():
    """自检判据 1a～1c。夹具全是虚构的；向量用字符袋假向量，不联网。"""
    import contextlib
    import io
    import tempfile
    import time
    from memory_retrieval import load_corpus
    from passive_facts import FACT_FLOOR, FactCooldown, _dot, _unit

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
        sent_path.write_text("\n".join(sentences) + "\n", encoding="utf-8")
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
        assert [r["句"] for r in rows] == [1, 2, 3, 4], rows
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
        del labelled["句子"][1]["前N"][1]["相关"]
        lab_path.write_text(json.dumps(labelled, ensure_ascii=False), encoding="utf-8")
        try:
            run("sweep", str(lab_path))
        except SystemExit as e:
            assert "1 条" in str(e.code), e.code
        else:
            raise AssertionError("1b：前 2 条里有没标的，应当报错而不是当成不相关")
    print("selftest 事实下限标定：1a 1b 1c 通过")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
