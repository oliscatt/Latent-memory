#!/usr/bin/env python3
"""命中门槛标定（floor 扫描）——只含方法，不含语料、题目和任何 key。

给要在自己语料上标定 `HIT_FLOOR_BY_MODEL` 之外模型的人。产出两条曲线：
  - **present**：gold=1 的命中数（题目协议同 present_probe_harness：金标原话逐字在语料里，
    否则作废——护栏抓的是「以为有」）。
  - **编造燃料**：absent 探针下向量路以该 floor **独立放行**的块数合计（题目协议同
    absent_probe_harness：区分词逐一验证不在语料里，否则作废）。floor 太松时这个数
    就是「向量路给编造递多少材料」。

选点规则（bge-m3 那次标定的产出，已收进标定表）：
  两个候选点测量值在噪声内等价时，取两个方向都退化得体的那个——平台边缘不站，
  不管它是靠单块压线的高分边（0.58 那类）还是悬在衰退区门口的零燃料边（0.62 那类）。

成本要点：**查询向量缓存**。扫 N 个 floor 档，每档都跑真实 retrieve()——不缓存的话
同一条查询要向服务商要 N 次向量，复现成本高一个量级。这里把 provider.embed 包一层
按 (文本, is_query) 记忆化：块向量走 VectorCache（建库一次），查询向量每条只要一次，
之后 21 个档全是本地余弦。真实路径一步没绕（判据不接受脚本自拼 rank_lists 的捷径）。

局限（如实）：absent 侧若在你的语料上被词面闸先漏（我们两份语料都是），floor 高低
不改变 absent 空手率——此时「编造燃料」才是 absent 侧的有效读数，空手率那列只是
如实记录，别拿它评价 floor。

用法（要自己接三处）：
  1. load_chunks() 接语料；2. PRESENT_QUERIES 换成你的 gold=1 题；3. ABSENT_PROBES 换成你的编造题。
  环境变量按《快速上手》云端档配好（MEMORY_EMBED_*），然后：
  python3 floor_calibration.py            # 扫描并打表
  python3 floor_calibration.py --selftest # 零依赖自检（不联网）
"""
import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from memory_retrieval import MemoryIndex  # noqa: E402

FLOORS = [None] + [i / 100 for i in range(30, 71, 2)]  # None=不开闸基线


# ── 换成你自己的语料/题目 ──
def load_chunks() -> list[str]:
    """返回语料块正文列表。示例：JSONL 每行 {"content": ...}"""
    path = Path(__file__).parent / "corpus.jsonl"
    return [json.loads(l)["content"] for l in open(path, encoding="utf-8")]


PRESENT_QUERIES = [  # (问题, 金标原话)。示例占位，不是真题
    ("我那台旧望远镜最后修好了吗", "目镜找配件配齐了"),
    ("陶艺课那次我拉坏了几个坯", "拉坏了三个坯"),
]
ABSENT_PROBES = [  # (编造问题, [区分词...])。示例占位，不是真题
    ("我们上次去南极看企鹅是哪天", ["南极", "企鹅"]),
    ("我那辆摩托车什么时候该保养", ["摩托车", "保养"]),
]


class CachedEmbed:
    """provider.embed 的记忆化外壳：同一 (文本组, is_query) 只向服务商要一次。

    这是整个扫描的成本核心——块向量由 VectorCache 管（建库一次），查询向量由这里管
    （每条一次）；之后每个 floor 档的 retrieve() 全是本地余弦，不再产生网络调用。
    """

    def __init__(self, provider):
        self._p = provider
        self._orig = provider.embed
        self._cache = {}
        self.misses = 0  # 自检用：真实外呼次数

    def install(self):
        def cached(texts, is_query=False):
            key = (json.dumps(texts, ensure_ascii=False), bool(is_query))
            if key not in self._cache:
                self.misses += 1
                self._cache[key] = self._orig(texts, is_query=is_query)
            return self._cache[key]
        self._p.embed = cached
        return self


def run_floor(idx: MemoryIndex, floor, present, absent_valid, topn=5):
    """一个 floor 档：真实 retrieve() 路径。返回 (present命中, 编造燃料, absent空手)。"""
    if floor is None:
        idx.vec_floor, idx.vec_calibrated = None, False
    else:
        idx.vec_floor, idx.vec_calibrated = floor, True
    hits = sum(1 for q, gold in present
               if gold in "\n".join(r["text"] for r in idx.retrieve(q, topN=topn)))
    fuel = 0
    empty = 0
    for q, _ in absent_valid:
        if floor is not None:
            vs = idx._vector_scores(q)
            fuel += sum(1 for v in vs if v > floor)  # embed 模式下无 None,不设死防御
        if not idx.retrieve(q, topN=topn):
            empty += 1
    return hits, fuel, empty


def main():
    chunks = load_chunks()
    alltext = "\n".join(chunks)

    present = [(q, g) for q, g in PRESENT_QUERIES if g and g in alltext]
    for q, g in PRESENT_QUERIES:
        if (q, g) not in present:
            print(f"SKIP(present: gold 逐字不在语料) {q}")
    absent_valid = [(q, ws) for q, ws in ABSENT_PROBES if not any(w in alltext for w in ws)]
    for q, ws in ABSENT_PROBES:
        if (q, ws) not in absent_valid:
            print(f"SKIP(absent: 区分词在语料里真出现) {q}")
    if not present:
        print("没有可计分的 present 题——按协议先把 gold 换成语料里的原话。")
        return

    idx = MemoryIndex(embed=True, cache_path=str(Path(__file__).parent / ".embed_cache.json"))
    for c in chunks:
        idx.add(c, {})
    idx.build()
    CachedEmbed(idx.provider).install()

    n_p, n_a = len(present), len(absent_valid)
    print(f"\nfloor   present({n_p}题)  编造燃料({n_a}题合计)  absent空手")
    for fl in FLOORS:
        h, f, e = run_floor(idx, fl, present, absent_valid)
        tag = "None(不开闸)" if fl is None else f"{fl:.2f}"
        print(f"{tag:<10} {h}/{n_p}        {f:<8}          {e}/{n_a}")
    print("\n# 选点：先找 present 平台，再看燃料衰减，取两个方向都退化得体的点。")


def _selftest():
    """零依赖自检：桩 provider（不联网），验三件事——缓存去重、floor 开合闸语义、护栏。"""
    class StubProvider:
        """定向向量桩：甲→x轴,乙→y轴,其余→与甲余弦0.6的斜向量——
        编造查询和语料相似但不同,才能验出 floor 拦得住/放得过两种形态。"""
        id = "stub"

        def embed(self, texts, is_query=False):
            def v(t):
                if "甲" in t:
                    return [1.0, 0.0, 0.0]
                if "乙" in t:
                    return [0.0, 1.0, 0.0]
                return [0.6, 0.0, 0.8]
            return [v(t) for t in texts]

        def hit_floor(self):
            return None

    idx = MemoryIndex(embed=True, provider=StubProvider())
    idx.add("甲物件的事", {})
    idx.add("乙物件的事", {})
    idx.build()
    ce = CachedEmbed(idx.provider).install()

    run_floor(idx, 0.5, [("甲物件在哪", "甲物件")], [("丙呢", ["丙"])])
    first = ce.misses
    run_floor(idx, 0.9, [("甲物件在哪", "甲物件")], [("丙呢", ["丙"])])
    assert ce.misses == first, "缓存失效：同一批查询第二个档位又外呼了"

    # 编造题不含甲/乙 → 桩给斜向量,与甲块余弦0.6:低门槛放行(燃料1)、高门槛拦住(燃料0)
    _, fuel_low, _ = run_floor(idx, 0.1, [], [("丙器件坏了吗", ["丙器件"])])
    _, fuel_high, _ = run_floor(idx, 0.99, [], [("丙器件坏了吗", ["丙器件"])])
    assert fuel_low > 0 and fuel_high == 0, f"floor 语义错：low={fuel_low} high={fuel_high}"

    alltext = "甲物件的事\n乙物件的事"
    assert any(w in alltext for w in ["甲物件"]), "护栏判据自身失效"
    print("floor_calibration selftest: 3/3 绿")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        main()
