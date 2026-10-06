"""issue #45 第 2 步：给写入端旧记录提示定线（LATENT_SUPERSEDE_HINT_MIN＝T、LATENT_SUPERSEDE_HINT_LEAD＝M）。

只读：语料、块向量缓存、事实变迁账本、撤回账本；不调向量服务（缓存里没有向量的块跳过、计数）；不写任何文件；
**不打印任何正文**，只出计数、分位数和提示率表。规则：

- 反例（触发总体）：--population（写回目录）下的每条 timeline 记录，只和写入时间早于它、不是同一天的现行 timeline 记录比
  （同一天按记录自己的日期，与服务端 supersede_pool 同一个判断 same_record_day），
  记第一名分数 s1、领先量 s1−s2；提示与否按 memory_retrieval.pick_supersede_hint 判（一条或两条都算一次）。
- 网格 T∈{0.30, 0.35, …, 0.95}、M∈{0.02, 0.04, …, 0.20} 逐格算提示率。给了 --max-rate，就取提示率不超过它的格里
  提示率最高的那格；并列取 M 大的，再并列取 T 大的。整张表照样打出来。
- 正例：账本里新记录在写回目录里、两条都有向量的每一对，按选定的线看提示能不能列出旧的那条；一对都没有就是
  “未知，待正例”。
- 给了 --line T M，另按这条线列出会提示的每条：新记录与提示到的旧记录的 recordId、相似度（只有 recordId 和分数）。

用法（内存紧的机器上建议限内存跑，参数照服务的启动命令带）：
  python tests/supersede_hint_calibration.py --corpus <语料> --population <写回目录> \\
      --cache <语料>/.embed_cache.json [--max-rate 0.05] [--line 0.90 0.10]
"""
import argparse
import heapq
import json
import math
from array import array
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embedding_provider import text_key                                  # noqa: E402
from memory_retrieval import load_corpus, pick_supersede_hint, same_record_day, _dot      # noqa: E402

QS = (5, 10, 25, 50, 75, 90, 95, 99)
GRID_T = [round(0.30 + 0.05 * k, 2) for k in range(14)]
GRID_M = [round(0.02 * k, 2) for k in range(1, 11)]


def pct(values, q):
    """最近秩法：第 ceil(q% × n) 小的那个。"""
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)], 4)


def describe(values):
    if not values:
        return {"n": 0}
    return {"n": len(values), "min": round(min(values), 4), "max": round(max(values), 4),
            "分位": {f"P{q}": pct(values, q) for q in QS}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="服务的 --corpus")
    ap.add_argument("--population", required=True, help="模型写回的目录（服务的 --write-dir），触发总体")
    ap.add_argument("--cache", required=True, help="块向量缓存，一般是 <语料>/.embed_cache.json")
    ap.add_argument("--supersessions", help="默认 <语料>/.supersessions.json")
    ap.add_argument("--retractions", help="默认 <语料>/.retractions.json")
    ap.add_argument("--max-rate", type=float, help="提示频率上限（0～1），由部署方自己定；不给就只出表")
    ap.add_argument("--line", type=float, nargs=2, metavar=("T", "M"), help="按这条线列出提示明细（只有 recordId 和分数）")
    args = ap.parse_args(argv)

    corpus = Path(args.corpus)
    index = load_corpus(str(corpus))                      # 不给向量档：只切块、不算向量
    retractions = Path(args.retractions or corpus / ".retractions.json")
    ledger_path = Path(args.supersessions or corpus / ".supersessions.json")
    if retractions.exists():
        index.load_retractions(retractions)
    if ledger_path.exists():
        index.load_supersessions(ledger_path)

    raw = json.loads(Path(args.cache).read_text(encoding="utf-8"))
    cached = raw.get("vectors") or {}
    vecs = {}
    for i, chunk in enumerate(index.chunks):
        if index.meta[i].get("layer", "timeline") == "timeline":
            vec = cached.get(text_key(chunk))
            if vec is not None:
                vecs[i] = array("f", vec)
    provider, dim = raw.get("provider"), len(next(iter(vecs.values()))) if vecs else None
    del raw, cached

    records = json.loads(ledger_path.read_text(encoding="utf-8")).get("records", {}) \
        if ledger_path.exists() else {}
    superseded_at = {rid: e.get("superseded_at") or 0 for rid, e in records.items()
                     if e.get("status") == "superseded"}
    population_ids = {m.get("record_id") for m in load_corpus(args.population).meta
                      if m.get("layer", "timeline") == "timeline"}
    population = [i for i, m in enumerate(index.meta)
                  if m.get("layer", "timeline") == "timeline" and m.get("record_id") in population_ids]
    hidden = set(index.hidden_indices())
    rows, missing = {}, 0
    for p in population:
        if p not in vecs:
            missing += 1
            continue
        at, rid = index.meta[p].get("timestamp") or 0, index.meta[p].get("record_id")
        # 还原写入那一刻的候选池：比它早写的现行记录。后来才被取代的旧记录那时还是现行的，要算进来
        # （正例里的旧记录正是这种）；撤回与插件隐藏按现状算。同一天的不算（same_record_day，与服务端同一判断）。
        pool = [j for j in vecs if j != p and j not in index.retracted and j not in hidden
                and index.meta[j].get("record_id") != rid
                and (index.meta[j].get("timestamp") or 0) < at
                and superseded_at.get(index.meta[j].get("record_id"), at) >= at
                and not same_record_day(index.meta[j], index.meta[p])]
        rows[p] = heapq.nlargest(3, ((_dot(vecs[p], vecs[j]), j) for j in pool),
                                 key=lambda row: (row[0], -row[1]))

    table = {f"T={t:.2f}": {f"M={m:.2f}": round(sum(bool(pick_supersede_hint(top, t, m))
                                                    for top in rows.values()) / len(rows), 4)
                            for m in GRID_M} for t in GRID_T} if rows else {}
    s1 = [top[0][0] for top in rows.values() if top]
    lead = [top[0][0] - (top[1][0] if len(top) > 1 else 0.0) for top in rows.values() if top]

    by_id = {index.meta[p].get("record_id"): p for p in rows}
    pairs = [(old, by_id[e["superseded_by"]]) for old, e in records.items()
             if e.get("superseded_by") in by_id]
    out = {"条件": {"语料块": len(index.chunks), "有向量的 timeline 块": len(vecs), "向量提供方": provider,
                    "维度": dim, "触发总体": len(population), "触发总体缺向量": missing,
                    "撤回账本": "存在" if retractions.exists() else "不存在",
                    "取代账本": "存在" if ledger_path.exists() else "不存在", "账本里新记录在写回目录的新旧对": len(pairs)},
           "第一名分数 s1": describe(s1), "领先量 s1−s2": describe(lead), "提示率（T×M）": table}

    if args.max_rate is not None and rows:
        ok = [(rate, m, t) for t in GRID_T for m in GRID_M
              if (rate := table[f"T={t:.2f}"][f"M={m:.2f}"]) <= args.max_rate]
        if not ok:
            out["定线"] = {"结论": f"网格里没有提示率 ≤ {args.max_rate} 的格，换个上限再看"}
        else:
            rate, m, t = max(ok)
            hinted = [p for p, top in rows.items() if pick_supersede_hint(top, t, m)]
            same_day = sum(index.meta[rows[p][0][1]].get("local_date") == index.meta[p].get("local_date")
                           for p in hinted)
            caught = sum(any(index.meta[j].get("record_id") == old
                             for _s, j in pick_supersede_hint(rows[new], t, m)) for old, new in pairs)
            out["定线"] = {"T": t, "M": m, "提示率": rate, "提示次数": len(hinted),
                           "其中第一名与新记录同一天": same_day,
                           "接住率": f"{caught}/{len(pairs)}" if pairs else "未知，待正例"}
    if args.line and rows:
        t, m = args.line
        rid = lambda i: index.meta[i].get("record_id")
        hits = [{"新记录": rid(p), "提示": [{"recordId": rid(j), "相似度": round(s, 4)}
                                         for s, j in pick_supersede_hint(top, t, m)]}
                for p, top in rows.items() if pick_supersede_hint(top, t, m)]
        out["按给定的线"] = {"T": t, "M": m, "提示次数": len(hits), "明细": hits}
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
