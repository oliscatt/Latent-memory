#!/usr/bin/env python3
"""判定实验：区分性 token 该挂在哪一层（任务卡"区分性token离开bigram层"）。

**这一单的第一个产物是"哪条路可行"的数据，不是一个新常数**，所以这个文件
**不改任何实现**：候选判据以运行时补丁的形式挂上去（`_patch`），进程一退就没了。
`memory_retrieval.py` 一个字都不动——判据选定之后再动它，那是下一单的事。

背景（外部实测，2026.08.02，报告人 `lu7899112-source`，1144 条中文语料）：
现行词面闸 `lexical_admit` 的判据挂在**字符 bigram** 上，而 bigram 层会产出
**跨词边界碎片**（"看极光是哪天"→`光是`df=1、"摔伤的是哪条腿"→`伤的`df=1），
每道编造题都有 df=1 的碎片开闸，`absent` 正确空手 **0/9**。报告里那句话是病因本身：
**在 bigram 这一层，"内容词"这个概念本身就产不出来。**

六条候选判据（`--criterion`，可 `--all` 一次跑完）：
  bigram        现状基线（`lexical_admit` 原样）
  df3           把 df 上限收到 3——**不是候选方案**，是用来在数据上核对报告
                那句"收紧 N//5 也救不了"，卡里明写了拍新常数不算做完
  ngram         零依赖：共享的连续字符串长度 ≥ n（默认 3）且 df ≤ 闸的上限
  entity        只认实体表里的实体（图谱那路已有的槽，`.entities.json`）
  ngram+entity  两条紧判据取并集
  seg           分词后的内容词——**要引入分词依赖**，装不上时如实报"未测"，
                不静默降级（零依赖档是默认档，依赖代价必须说在明处）

用法：
  python3 distinctive_token_experiment.py --all            # 我们的合成语料
  python3 distinctive_token_experiment.py --selftest

**给外部复测的人（不需要把语料给任何人）**：
  python3 distinctive_token_experiment.py --all \\
      --corpus <你的语料目录或 jsonl> --probes <你的编造题.json> \\
      --entities <对应语料的 .entities.json>

  `--entities` 不给时，`entity` 与 `ngram+entity` 两行明确显示“未测”，不会把
  恒空放行集计成成绩；只跑零依赖判据时不需要实体表。实体表沿用产品既有格式，
  按内容哈希对号入座；与本轮语料零块匹配会直接报错。

  编造题文件是一个 JSON 数组，每项 `[问题, [区分词, ...]]`——区分词是这道题里
  承载"编造事实"的内容词，护栏会**先验证它们在你语料全文里一个都不出现**，
  出现即作废（`absent_probe_harness.py`，外部贡献者送的，本单不改它）。

  ⚠ **外部语料模式下只打印数字**：不打印任何语料正文、不打印命中片段——
  你回传给我们的那张表里不该夹带你和你的 AI 说过的话。这是刻意的，别改。
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from memory_retrieval import (  # noqa: E402
    MemoryIndex,
    entities_from_answer,
    load_corpus,
)
import absent_probe_harness as harness  # noqa: E402

CRITERIA = ("bigram", "df3", "ngram", "entity", "ngram+entity", "seg")
DEFAULT_N = 3


def _norm(text):
    """跟 tokenize 同一套归一（去标点空白、转小写），好让 n-gram 跟 bigram 可比。"""
    return re.sub(r"[^\w一-鿿]", "", text.lower())


# ---------- 我们自己造的编造题（走护栏，护栏不改） ----------
#
# ⚠ **区分词要写成"语料里可能出现的最小内容词"**：护栏做的是子串检查，
# 区分词写成「天文望远镜」这种长串时，语料里的「望远镜」拦不住它——出题人
# 等于绕过了自己的护栏。这是本单实测踩到的，记在这儿给下一个出题的人。
OUR_PROBES = [
    ("摔伤的是哪条腿", ["摔伤"]),
    ("上次陶艺课我拉坏了几个坯", ["陶艺", "坯"]),
    ("学会说什么了", ["学会说"]),
    ("猫在哪家医院绝育的", ["绝育"]),
    ("上个月我们搬去海边住的那阵子", ["海边"]),
    ("她说想学吉他那回后来买了吗", ["吉他"]),
    ("我们一起养的那只狗叫什么来着", ["养的那只狗"]),
    ("我那台单簧管修好了吗", ["单簧管"]),
    ("体检报告出来了没有", ["体检报告"]),
    ("看极光是哪天去的", ["极光"]),
]


# ---------- 候选判据（运行时补丁，不落进实现） ----------

_ORIG_ADMIT = MemoryIndex.lexical_admit
_ORIG_RETRIEVE = MemoryIndex.retrieve


def _remember_query(self, query, **kw):
    """`lexical_admit` 只收到 tokens，而 n-gram/实体两条判据要的是原始 query。
    包一层把 query 记在实例上——**只在实验里这么干**：真要选中这条判据，
    实现侧应该改成把 query 传下去，而不是留这么一个隐式状态。"""
    self._exp_query = query
    return _ORIG_RETRIEVE(self, query, **kw)


def _ngram_df(self, n):
    """按块统计 n-gram 的文档频次。缓存在实例上，别每次查询重算全库。"""
    key = f"_exp_ndf_{n}"
    if not hasattr(self, key):
        table = {}
        self._exp_norm_chunks = [_norm(c) for c in self.chunks]
        for c in self._exp_norm_chunks:
            for g in {c[i:i + n] for i in range(len(c) - n + 1)}:
                table[g] = table.get(g, 0) + 1
        setattr(self, key, table)
    return getattr(self, key)


def _admit_ngram(self, n):
    """共享的连续字符串长度 ≥ n、且 df ≤ 闸自己的上限 → 放行。

    直觉：跨词边界碎片是两个字的偶然拼接，扩到三个字之后偶然重合的概率掉一个
    量级——"光是""伤的"这类还在，但它们在别的块里同样是碎片，很难再撞上。
    **df 上限仍问闸本人**（`_distinctive_df`），不另抄一份常数（同护栏那条纪律）。"""
    q = _norm(getattr(self, "_exp_query", ""))
    cap = self._distinctive_df()
    table = _ngram_df(self, n)
    ev = [g for g in {q[i:i + n] for i in range(len(q) - n + 1)}
          if 0 < table.get(g, 0) <= cap]
    return {i for i, c in enumerate(self._exp_norm_chunks) if any(g in c for g in ev)}


def _admit_entity(self):
    """只认实体表：块的实体在查询里出现过才放行（图谱那路已有的 `.entities.json`）。

    实体表是**外部产物**（LLM 抽取，路线 C），不是从 bigram 推出来的——这正是
    报告建议的方向：让"区分性 token"这个概念长在一个真的能产出内容词的层上。"""
    q = _norm(getattr(self, "_exp_query", ""))
    out = set()
    for i, ents in getattr(self, "_entities", {}).items():
        if any(_norm(e) and _norm(e) in q for e in ents):
            out.add(i)
    return out


def _admit_df3(self, query_tokens):
    """bigram 判据 + 把 df 上限硬收到 3。**这不是候选方案**——卡里写死了"把
    N//5 改成 N//10 交差 = 没做"。它在这里只有一个用途：在我们自己的数据上
    核对报告那句"收到 df<=3 也拦不住"，免得我们只是相信它。"""
    dfs = self._bm25.df
    dist = {t for t in set(query_tokens) if 0 < dfs.get(t, 0) <= 3}
    if not dist:
        return set()
    return {i for i, tf in enumerate(self._bm25.tf) if any(t in tf for t in dist)}


def _seg_words(text):
    """分词后的内容词。装不上分词器就返回 None——调用方据此如实报"未测"。"""
    try:
        import jieba                                    # noqa: PLC0415
    except ImportError:
        return None
    return [w for w in jieba.lcut(text) if len(w) >= 2]


def _admit_seg(self):
    words = _seg_words(getattr(self, "_exp_query", ""))
    if words is None:
        raise RuntimeError("没装分词器")
    cap = self._distinctive_df()
    table = {}
    if not hasattr(self, "_exp_seg_chunks"):
        self._exp_seg_chunks = [set(_seg_words(c) or ()) for c in self.chunks]
    for ws in self._exp_seg_chunks:
        for w in ws:
            table[w] = table.get(w, 0) + 1
    ev = {w for w in words if 0 < table.get(w, 0) <= cap}
    return {i for i, ws in enumerate(self._exp_seg_chunks) if ws & ev}


def patch(criterion, n=DEFAULT_N):
    """把某条判据挂上去。**进程内补丁，不写盘、不改实现**——判据还没选定，
    现在往 `memory_retrieval.py` 里写任何一条，都是在拿实现赌一个还没验的结论。"""
    def admit(self, query_tokens):
        if criterion == "bigram":
            return _ORIG_ADMIT(self, query_tokens)
        if criterion == "df3":
            return _admit_df3(self, query_tokens)
        if criterion == "ngram":
            return _admit_ngram(self, n)
        if criterion == "entity":
            return _admit_entity(self)
        if criterion == "ngram+entity":
            return _admit_ngram(self, n) | _admit_entity(self)
        if criterion == "seg":
            return _admit_seg(self)
        raise ValueError(f"未知判据：{criterion}")
    MemoryIndex.lexical_admit = admit
    MemoryIndex.retrieve = _remember_query


def unpatch():
    MemoryIndex.lexical_admit = _ORIG_ADMIT
    MemoryIndex.retrieve = _ORIG_RETRIEVE


# ---------- 实体表：我们自己合成语料的"上界表" ----------
#
# ⚠ **这是上界，不是产品形态**：真实部署里实体表要花 LLM 调用抽（路线 C），
# 抽得多准是另一个变量。这里手工/机械地给一份"抽得很好"的表，是为了回答
# **"假设实体表是好的，这条路可行吗"**——可行才值得去谈抽取成本。
#
# 纪律：**只看语料正文标注，不看查询集**。填充块的实体直接取生成器自己的
# 词表（地点/话题/细节），那是机械导出的，没有我挑词的空间；前 22 块是人写的
# 合成语料，按正文里的名词标，标完就不再改（不许拿分数回头调表）。
_AUTHORED_ENTITIES = {
    "搬家第一天": ["厨房", "新住处"],
    "修理咖啡机": ["咖啡机", "加热管", "保险丝"],
    "山顶的约定": ["山顶", "望远镜", "土星", "奖金"],
    "种薄荷失败": ["薄荷", "花盆", "阳台"],
    "读城市与狗": ["城市与狗", "略萨", "小说"],
    "挑设备": ["望远镜", "脚架"],
    "一次远足": ["青云山", "桂花糕", "凉亭"],
    "加班到很晚": ["加班", "项目"],
    "学做咖喱": ["咖喱", "椰浆"],
    "阳台那晚": ["阳台", "脚架", "夜空"],
    "又坏了一次": ["咖啡机", "水泵"],
    "终于换掉": ["咖啡机"],
    "改种别的": ["多肉", "阳台"],
    "长得还行": ["多肉", "叶子"],
    "交付完了": ["项目", "交付"],
    "睡到中午": ["睡到中午"],
    "奖金": ["奖金", "望远镜"],
    "更正": ["奖金", "望远镜"],
}


def oracle_entities(idx):
    """给合成语料造一份"抽得很好"的实体表 → {块下标: {实体}}。"""
    import regression_set as RS                          # noqa: PLC0415
    out = {}
    for i, (chunk, meta) in enumerate(zip(idx.chunks, idx.meta)):
        head = (meta.get("heading") or "").strip()
        if head in _AUTHORED_ENTITIES:
            out[i] = set(_AUTHORED_ENTITIES[head])
        elif meta.get("heading") == "填充" or head.endswith(tuple(RS._TOPICS)):
            # 填充块：主语是「地点＋话题」，细节短语也是生成器给的——机械取回来，
            # 不是我挑的词
            ents = {p for p in RS._PLACES if p in chunk}
            ents |= {t for t in RS._TOPICS if t in chunk}
            ents |= {d for d in RS._DETAILS if d in chunk}
            if ents:
                out[i] = ents
    return out


# ---------- 指标 ----------

def gate_miss_rate(idx, cases):
    """**闸误杀率**：期望块被词面闸挡在候选外的比例。

    为什么要单独报、不能只看端到端 recall：零依赖档里 `lexical_admit` 同时管着
    BM25 与 bigram 向量两条路（见 retrieve 里那段注释），图谱路的种子也只能从
    放行的块里长出来——**闸一关，这个块基本就没有别的路能回来了**。
    端到端 recall 好看有可能只是因为别的查询捞回来了，误杀要在闸这一层看。"""
    missed = total = 0
    for _kind, query, expect in cases:
        if not expect:
            continue
        total += 1
        idx._exp_query = query
        from memory_retrieval import tokenize             # noqa: PLC0415
        admitted = idx.lexical_admit(tokenize(query))
        target = {i for i, c in enumerate(idx.chunks)
                  if all(f in c for f in expect)}
        if not (target & admitted):
            missed += 1
    return missed, total


def run_one(criterion, chunks, probes, cases, entities=None, n=DEFAULT_N,
            quiet=False):
    """跑一条判据 → 一行数字。语料正文一个字都不进返回值。"""
    if criterion in {"entity", "ngram+entity"} and not entities:
        return {"criterion": criterion, "skip_reason": "未提供实体表"}
    row = {"criterion": criterion}
    patch(criterion, n=n)
    try:
        # absent 侧走护栏（本单不改它）：每道题先验证区分词不在语料里才计分
        idx_probe = MemoryIndex()
        for c in chunks:
            idx_probe.add(c, {})
        idx_probe.build()
        if entities:
            idx_probe._entities = entities
            idx_probe.build()
        try:
            ok, tested, skipped, detail = harness.run_probes(
                chunks, probes, verbose=False, index=idx_probe)
        except RuntimeError as e:                        # seg 档没装分词器
            unpatch()
            return {"criterion": criterion, "skip_reason": str(e)}
        row.update(absent_ok=ok, absent_n=tested, absent_skip=skipped)
        row["leak_evidence"] = [(q, ev) for st, q, ev in detail if st == "LEAK"]

        if cases is not None:
            import regression_set as RS                   # noqa: PLC0415
            idx = RS.build_index(scale="established")
            if entities:
                idx._entities = entities
                idx.build()
            row["present"] = RS.score(idx, cases=cases)
            missed, total = gate_miss_rate(idx, cases)
            row["gate_missed"], row["gate_total"] = missed, total
    finally:
        unpatch()
    return row


def format_table(rows, quiet=False):
    out = []
    head = f"{'判据':<14}{'absent正确空手':<16}{'闸误杀':<10}"
    if not quiet:
        head += (f"{'literal':<9}{'paraphr':<9}{'linked':<9}{'感受回忆':<10}"
                     f"{'absent_日常':<12}")
    out.append(head)
    for r in rows:
        if "skip_reason" in r:
            out.append(f"{r['criterion']:<14}未测（{r['skip_reason']}）")
            continue
        absent = f"{r['absent_ok']}/{r['absent_n']}"
        if r.get("absent_skip"):
            absent += f"(弃{r['absent_skip']})"
        gate = (f"{r['gate_missed']}/{r['gate_total']}"
                if "gate_missed" in r else "-")
        line = f"{r['criterion']:<14}{absent:<16}{gate:<10}"
        if not quiet and "present" in r:
            g = lambda k: (f"{r['present'][k]['recall']:.2f}"      # noqa: E731
                           if k in r["present"] else "-")
            empty = (f"{r['present']['absent_日常']['correct_empty']:.2f}"
                     if "absent_日常" in r["present"] else "-")
            line += (f"{g('literal'):<9}{g('paraphrase'):<9}"
                     f"{g('linked'):<9}{g('感受回忆'):<10}{empty:<12}")
        out.append(line)
    return "\n".join(out)


def load_external(corpus_path):
    """外部语料 → 块正文列表。目录走 load_corpus，jsonl 走每行的 content 字段。"""
    p = Path(corpus_path)
    if p.is_dir():
        return list(load_corpus(p).chunks)
    return [json.loads(line)["content"] for line in
            p.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_external_entities(chunks, entities_path):
    """把产品既有的 .entities.json 按内容哈希接到本轮外部语料。"""
    path = Path(entities_path)
    if not path.exists():
        raise FileNotFoundError(f"实体表不存在：{path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            "实体表格式错误：必须是内容哈希到实体列表的 JSON 对象") from exc
    if not isinstance(raw, dict) or any(
            not isinstance(values, list)
            or any(not isinstance(value, str) for value in values)
            for values in raw.values()):
        raise ValueError("实体表格式错误：必须是内容哈希到实体列表的 JSON 对象")
    idx = MemoryIndex()
    for chunk in chunks:
        idx.add(chunk, {})
    idx.build()
    try:
        matched = idx.load_entities(path)
    except (AttributeError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "实体表格式错误：必须是内容哈希到实体列表的 JSON 对象") from exc
    if matched == 0:
        raise ValueError("实体表与当前外部语料零块匹配")
    return {i: set(values) for i, values in idx._entities.items()}


# ---------- selftest ----------

def _selftest():
    chunks = ["## 修咖啡机\n加热管不工作，拆开发现保险丝熔断。",
              "## 种薄荷\n四月阳台的薄荷死了，盆底积水烂了根。",
              "## 远足\n去青云山走了十二公里，半山腰买了桂花糕。"]

    # 1. 判据切换真的生效：同一句话在两条判据下放行集不同，且**补丁退出后复原**
    patch("bigram")
    idx = MemoryIndex()
    for c in chunks:
        idx.add(c, {})
    idx.build()
    #    「了根」是跨词边界碎片：查询里它跨在「摔伤了|根本」之间，语料里跨在
    #    「烂了|根」之间——两边都不是词，df=1，却足够开闸。报告说的就是这一类
    idx._exp_query = "他摔伤了根本走不动"
    from memory_retrieval import tokenize
    a = idx.lexical_admit(tokenize(idx._exp_query))
    patch("ngram")
    b = idx.lexical_admit(tokenize(idx._exp_query))
    unpatch()
    assert MemoryIndex.lexical_admit is _ORIG_ADMIT and \
        MemoryIndex.retrieve is _ORIG_RETRIEVE, \
        "补丁必须只活在实验里——退出后实现要原样复原"
    #    这句话跟这三块语料没有任何关系：bigram 层靠边界碎片放行，n≥3 关掉
    assert a and not b, f"边界碎片该在 bigram 下开闸、在 n≥3 下关掉，实际 {a=} {b=}"

    # 2.【本单靶心】护栏真的被调用了，且它照样能抓住我们自己
    #    ——出题人写了一道区分词其实在语料里的题，必须作废、不进分母
    patch("bigram")
    ok, tested, skipped, detail = harness.run_probes(
        chunks, [("上次陶艺课我拉坏了几个坯", ["陶艺"]),
                 ("那台咖啡机后来修好了吗", ["咖啡机"])], verbose=False)
    unpatch()
    assert skipped == 1 and tested == 1, \
        f"「咖啡机」在语料里，那道题必须被护栏拦下作废，实际 {skipped=} {tested=}"
    assert any(st == "SKIP" and "咖啡机" in ev for st, _q, ev in detail), \
        "护栏要说出是哪个词漏的"

    # 3. 实体判据只认实体表：表空 → 一个都不放行（**别静默退回 bigram**，
    #    那会把"这条路行不行"的实验变成"bigram 行不行"的实验）
    patch("entity")
    idx2 = MemoryIndex()
    for c in chunks:
        idx2.add(c, {})
    idx2.build()
    idx2._exp_query = "咖啡机修好了吗"
    assert idx2.lexical_admit(tokenize(idx2._exp_query)) == set(), \
        "没有实体表时实体判据放行集必须是空的，不许偷偷退回别的判据"
    idx2._entities = {0: {"咖啡机"}}
    assert idx2.lexical_admit(tokenize(idx2._exp_query)) == {0}, "有表时该认得出"
    unpatch()

    # 4. 实体相关判据没接实体表时必须明确报“未测”，不能把恒空放行集计成成绩。
    #    这条会抓住外部模式把 entities 固定成 None 后仍输出 entity 9/9 的平凡解。
    probes = [("我那台单簧管修好了吗", ["单簧管"])]
    for criterion in ("entity", "ngram+entity"):
        for empty_entities in (None, {}):
            row = run_one(
                criterion, chunks, probes, cases=None, entities=empty_entities)
            assert row == {"criterion": criterion, "skip_reason": "未提供实体表"}, \
                f"{criterion} 没有可用实体时必须标成未测，实际 {row}"

    # 5. 实体表不只要“载入成功”，还必须真进 absent 计分所用的那一个索引。
    #    语料里有“咖啡机”，编造事实词“电池／牌子”没有；接线正确时实体判据会
    #    因查询中的“咖啡机”开闸，所以这题应漏检 0/1，而不是恒空放行的 1/1。
    wired = run_one(
        "entity",
        chunks,
        [("咖啡机换的电池是什么牌子", ["电池", "牌子"])],
        cases=None,
        entities={0: {"咖啡机"}},
    )
    assert (wired["absent_ok"], wired["absent_n"]) == (0, 1), \
        f"实体表必须进入 absent 计分索引，实际 {wired}"

    # 6. 外部实体表沿用产品的内容哈希格式：匹配表真接入，错表必须吵着失败。
    #    这条会抓住“参数看似存在但没接线”和“拿错表仍算平凡解”两种坏法。
    entities_path = Path(__file__).with_name(".distinctive-token-selftest.entities.json")
    wrong_path = Path(__file__).with_name(".distinctive-token-selftest-wrong.entities.json")
    malformed_path = Path(__file__).with_name(
        ".distinctive-token-selftest-malformed.entities.json")
    malformed_values_path = Path(__file__).with_name(
        ".distinctive-token-selftest-malformed-values.entities.json")
    try:
        source = MemoryIndex()
        for c in chunks:
            source.add(c, {})
        valid = entities_from_answer(source, {"1": ["咖啡机"]})
        entities_path.write_text(json.dumps(valid, ensure_ascii=False), encoding="utf-8")
        loaded = load_external_entities(chunks, entities_path)
        assert loaded == {0: {"咖啡机"}}, f"实体表没有按内容哈希接回原块：{loaded}"

        wrong_path.write_text(
            json.dumps({"不匹配的内容哈希": ["咖啡机"]}, ensure_ascii=False),
            encoding="utf-8",
        )
        try:
            load_external_entities(chunks, wrong_path)
        except ValueError as exc:
            assert "零块匹配" in str(exc), f"错表错误信息没有指出根因：{exc}"
        else:
            raise AssertionError("与当前语料零块匹配的实体表必须失败")

        malformed_path.write_text("[]", encoding="utf-8")
        try:
            load_external_entities(chunks, malformed_path)
        except ValueError as exc:
            assert "格式错误" in str(exc), f"坏格式错误信息没有指出根因：{exc}"
        else:
            raise AssertionError("不是内容哈希映射的实体表必须失败")

        malformed_values_path.write_text(
            json.dumps({next(iter(valid)): "咖啡机"}, ensure_ascii=False),
            encoding="utf-8",
        )
        try:
            load_external_entities(chunks, malformed_values_path)
        except ValueError as exc:
            assert "格式错误" in str(exc), f"实体值不是列表时没有指出格式错误：{exc}"
        else:
            raise AssertionError("实体表的每个值都必须是实体字符串列表")
    finally:
        entities_path.unlink(missing_ok=True)
        wrong_path.unlink(missing_ok=True)
        malformed_path.unlink(missing_ok=True)
        malformed_values_path.unlink(missing_ok=True)

    # 7. 外部语料模式只回数字：报表里不许出现任何语料正文
    #    （回传的那张表不该夹带别人和 TA 的 AI 说过的话）
    rows = [run_one("bigram", chunks, [("我那台单簧管修好了吗", ["单簧管"])],
                    cases=None)]
    table = format_table(rows, quiet=True)
    for c in chunks:
        body = c.split("\n", 1)[1][:8]
        assert body not in table, f"外部模式的报表里出现了语料正文：{body}"
    assert "1/1" in table, f"数字该在：{table}"

    # 8. seg 档没装分词器时如实报"未测"，不静默降级成别的判据
    if _seg_words("测试一下") is None:
        r = run_one("seg", chunks, [("我那台单簧管修好了吗", ["单簧管"])], cases=None)
        assert r.get("skip_reason"), "装不上分词器要明确报未测，不许偷偷换判据"

    print("selftest ok（8项断言：判据切换生效且补丁可复原 / 护栏真被调用且抓住出题人 / "
          "实体判据不静默退回 / 空实体表不计分 / 实体表真进入 absent 计分索引 / "
          "外部实体表接线且错表拒绝 / 外部模式只回数字 / 缺依赖如实报未测）")


def main():
    ap = argparse.ArgumentParser(description="区分性 token 判据的判定实验（不改实现）")
    ap.add_argument("--criterion", choices=CRITERIA, action="append",
                    help="跑哪条判据，可重复；默认全部")
    ap.add_argument("--all", action="store_true", help="跑全部判据")
    ap.add_argument("-n", type=int, default=DEFAULT_N, help="ngram 档的 n，默认 3")
    ap.add_argument("--corpus", help="外部语料（目录或 jsonl）；不给则用回归集合成语料")
    ap.add_argument("--probes", help="外部编造题 json：[[问题, [区分词,...]], ...]")
    ap.add_argument("--entities", help="外部实体表（.entities.json，须与 --corpus 同用）")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return

    criteria = args.criterion or list(CRITERIA)
    if args.all:
        criteria = list(CRITERIA)

    if args.entities and not args.corpus:
        ap.error("--entities 只能与 --corpus 一起使用")

    external = bool(args.corpus)
    if external:
        chunks = load_external(args.corpus)
        cases = None                      # 别人的语料没有我们的期望集
        entities = None
        if args.entities:
            try:
                entities = load_external_entities(chunks, args.entities)
            except (OSError, ValueError) as exc:
                ap.error(f"实体表不可用：{exc}")
    else:
        import regression_set as RS
        idx = RS.build_index(scale="established")
        chunks = list(idx.chunks)
        cases = list(RS.CASES) + list(RS.HELDOUT_CASES)
        entities = oracle_entities(idx)
    probes = OUR_PROBES
    if args.probes:
        probes = [(q, list(w)) for q, w in
                  json.loads(Path(args.probes).read_text(encoding="utf-8"))]

    print(f"语料 {len(chunks)} 块 / 编造题 {len(probes)} 道"
          + ("（外部语料：只输出数字）" if external else "（我们的合成语料）"))
    rows = [run_one(c, chunks, probes, cases, entities=entities, n=args.n,
                    quiet=external) for c in criteria]
    print(format_table(rows, quiet=external))
    if not external:
        print("\n开闸证据（漏检时是什么放行的——碎片还是真词，两种病）：")
        for r in rows:
            if r.get("leak_evidence"):
                q, ev = r["leak_evidence"][0]
                shown = "、".join(f"{t}(df={d})" for d, t in (ev or [])[:4]) or "无"
                print(f"  {r['criterion']:<14}例：{q} ← {shown}")


if __name__ == "__main__":
    main()
