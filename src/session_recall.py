#!/usr/bin/env python3
"""
换窗/压缩两个触发点的接线（任务卡"换窗压缩记忆召回"第 3 条）。

memory_retrieval.MemoryIndex.recall_recent 解决"没有 query 时召回什么"（时间新鲜度×
用进废退权重），本文件解决"什么时刻调它"，只做触发与格式化，不碰排序逻辑：

  触发点一·换新窗口：信号干净——新 session/新进程启动本身就是边界。宿主在 session
  启动逻辑里调一次 on_session_start()，把返回的文本块注入开场 context。

  触发点二·context 压缩：**近似触发，不是精确对齐压缩时刻**——Claude Code 的
  auto-compact 之类没有可订阅的"压缩刚发生"事件信号，退而求其次：累计对话字符量
  （中文语境下字符数≈token 数的粗代理）超过可配置阈值就触发一次召回、计数器清零
  重新累计。这个局限是结构性的，不要假装精确：触发点跟真实压缩时刻之间必有偏差，
  阈值该配得比宿主真实压缩阈值略低——宁可早触发（早注入的召回内容会被压缩摘要
  一起接住），不要晚触发（压缩后细节已丢，晚到的召回救不回压缩瞬间的断层）。

第二轮验收补的两道防线——recency×权重解决"召回什么"，这两道管"召回的东西会不会
被新窗口读歪"，缺一不可：

  时间标注（防时间感塌陷）：每条注入片段带发生日期，块头写明"是过去的事，不是
  正在发生"。纯按 recency 召回时，新窗口容易把"几天前已经和好的争吵"当成正在
  发生的事去接。注意召回层只能标时间："每段记录收尾交代这件事现在的状态"是
  CLAUDE.md"病灶迁移"那条语料写作纪律，得在写记录时遵守，这里替代不了。

  自查关卡（防退化窗口自圆其说）：注入块尾部带四项故障信号自查指令（复用
  degradation_protocol 的同一套信号）；换窗触发还支持宿主传入 SessionStatus 先过
  程序化关卡——有故障信号就不接入召回记忆，老实返回"可能在退化，请核实"。窗口
  真漂移时，召回机制本身没错，但旧记忆会被拿去往错的方向自圆其说，必须先拦。
  与 degradation_protocol v2 核心规则一致：只认故障信号，信号干净就正常放行。

零依赖，stdlib only。

用法：
  python session_recall.py --selftest
"""

import argparse
import re
from datetime import datetime

# 同目录模块，import 不触发各自的 CLI
from memory_retrieval import MemoryIndex, _chunk_key  # noqa: F401
from degradation_protocol import SessionStatus, degradation_signals  # noqa: F401
from time_context import TimeContext  # noqa: F401
from session_thread import ThreadStore, format_thread_block  # noqa: F401
from unresolved_state import (UnresolvedStoreError, format_block as format_unresolved_block,
                              source_record_ids)

# 压缩近似触发的默认阈值（字符）：取得明显低于常见 context 窗口的量级，宁可早触发。
# 经验值，接真实宿主时按其压缩阈值往下压一档配置。
DEFAULT_COMPACT_THRESHOLD_CHARS = 30_000
DEFAULT_TOPN = 3


SELF_CHECK_FOOTER = (
    "【自查】接上以上记忆前先过四项故障信号（degradation_protocol 同一套）：\n"
    "①时间感：上面各段的日期读对了吗——都是历史，不是正在发生；②硬事实：当前日期、"
    "自己在哪个窗口这类问题答得上来吗；③逻辑还连贯吗；④有没有卡在循环道歉里。\n"
    "任一项不对劲：不要拿召回内容自圆其说往下编，老实说“可能在退化，需要你核实”，"
    "按 degradation_protocol 分级处理。")


# 单条片段的字符上限（2026.07.31 真实语料实测后加）：块大小中位 352 但最长 4386，
# 一个长块就能吃掉半个注入；topN=5 时实测注入体积中位 4095、最大 7131 字符，每次
# 换窗和每次压缩触发都要付一遍。
# 选择"截断单条"而不是"少给几条"：召回块是广度层（最近发生过什么），条数比单条
# 完整度更值钱；被截掉的全文让模型用 latent_search 再查一次，标注写在截断处。
DEFAULT_MAX_ITEM_CHARS = 500
TRUNCATION_NOTE = "…（片段已截断，要全文就用 latent_search 再查一次）"


def _clip(text, limit):
    text = text.strip()
    if limit is None or len(text) <= limit:
        return text
    return text[:limit].rstrip() + TRUNCATION_NOTE


def _time_label(meta, time_context=None):
    """召回片段的时间标注——防时间感塌陷的第一道措施：注入的记忆必须带着"这是什么
    时候的事"，否则新窗口容易把旧片段读成正在发生。**没有真实日期依据的一律标
    「时间未知」**：缺时间戳的，以及只能退到 mtime 的。

    **`local_date` 优先于 epoch**（2026.08.04 跨时区事故，任务卡"写回时区与跨日
    归窗"）：`local_date` 是记忆所有者时区里的那个日历日，已经是结论；而把 epoch
    在这里格式化一次，格出来的是**读它的这个进程所在时区**的日期——UTC 的服务器上
    东八区凌晨那条就少一天。没有 `local_date` 的旧 meta 才退回按 epoch 格式化，
    这时用配置的 TimeContext，不用宿主本地时区。

    **mtime 档从"≈某日"改成「时间未知」（2026.08.03，跨模型实测逼出来的）**：
    原先 mtime 兜底会打 `≈2026.08.03` 这种标签，看着"只是不太精确"，实际是
    **假装知道**——mtime 在全新 clone／刚落盘的语料上就是"今天"，于是最没有时间
    依据的片段拿到了全库最新的日期，仲裁句照规则把它端成当下状态。
    实测（DS／`glm-5.2` 各 3 次独立会话，同一夹具 A/B）：**改前 0/6 答对当下状态、
    5/6 被无日期那条冒充现状**（另 1 次检索没命中、C 没出现，不计冒充）；
    **改后冒充现状 0/6**，答对 5/6——剩下那 1 次同样是 `latent_search` 没命中、
    诚实说“没找到”，不是冒充。
    排序一个字没动（那条仍排在召回块第一），**只把"假装知道"换成"如实说不知道"
    就够了**——这也是为什么没有采纳"让 mtime 档不参与新鲜度排序"那个更重的方案：
    真实语料里 mtime 占比可以很高（本项目那份加 window_sibling 之前 34/37），
    让它们整体退出新鲜度，等于把换窗召回要救的"刚被冲掉的最近上下文"一起关掉。"""
    ts = meta.get("timestamp")
    if ts is None or meta.get("timestamp_source") == "mtime":
        return "时间未知"
    day = meta.get("local_date")
    if day:
        return day.replace("-", ".")
    return (time_context or TimeContext.default()).label(ts)


def format_recall_block(results, max_item_chars=DEFAULT_MAX_ITEM_CHARS, time_context=None):
    """recall_recent 结果 → 可直接注入 context 的文本块，每条带发生日期。
    单条超 max_item_chars 就截断并标注（传 None 关闭截断）。
    空结果返回 None——没东西可召回就什么都不注入，不留空壳标题。"""
    if not results:
        return None
    # 仲裁句 2026.08.03 改口径（任务卡「新旧仲裁口径」）：旧句"同一件事以日期最新的
    # 片段为准"要求读的一方先判定"是不是同一件事"——"上月搬到 A / 这周搬到 B"算不算
    # 同一件事，全靠模型自由裁量。新口径来自外部反馈（PR #2 第 3 条回答）：按"同类
    # 相关"读，最近一条是当下状态、更早的读作历史——判定门槛更低，纠错场景（用户
    # 纠正过的记忆＝更新的同类记忆）同一条约定通吃。时间未知那半句是本项目补的：
    # 缺时间戳的块本来就标"时间未知"，仲裁规则却对它没说法，而那正是"旧事实冒充
    # 现状"最容易钻的一格——保守方向：读作历史背景，不当作当下状态。
    lines = ["【最近记忆召回】以下是历史记忆片段，各自标注发生日期——是过去的事，"
             "不是正在发生。同类相关的记忆里，以日期最近的一条为当下状态，"
             "更早的读作历史；标「时间未知」的片段只当历史背景，不当作当下状态："]
    for r in results:
        head = r["meta"].get("heading") or r["meta"].get("source", "")
        tag = "·".join(x for x in (_time_label(r["meta"], time_context), head) if x)
        lines.append(f"- [{tag}] {_clip(r['text'], max_item_chars)}")
    return "\n".join(lines)


class SessionRecall:
    """两个触发点的最小接线。宿主用法：
         sr = SessionRecall(index)
         opening = sr.on_session_start()      # 换窗触发：非 None 就注入开场 context
         for turn_text in 对话流:
             block = sr.on_turn(turn_text)    # 压缩近似触发：非 None 就注入
    只持有触发状态（累计字符数），召回排序全部委托给 index.recall_recent。"""

    def __init__(self, index, topN=DEFAULT_TOPN, half_life=None,
                 compact_threshold_chars=DEFAULT_COMPACT_THRESHOLD_CHARS,
                 thread_store=None, time_context=None, unresolved_store=None):
        self.index = index
        self.topN = topN
        self.half_life = half_life  # None → recall_recent 用自己的默认半衰期
        self.compact_threshold_chars = compact_threshold_chars
        # 可选的会话线索存储（session_thread.ThreadStore）。给了就在换窗时先注入
        # “上次聊到哪”，没给就跳过 thread；其余层照常给——不断线（规格 §5 三层）
        self.thread_store = thread_store
        self.unresolved_store = unresolved_store
        # 记忆所有者的时区（任务卡"写回时区与跨日归窗"）：只在 meta 没有 local_date
        # 的旧块上用得到——新块的日期在写入时就定死了，读的时候不再换算一次
        self.time_context = time_context
        self._chars_since_recall = 0

    def _recall(self, now=None, exclude_record_ids=()):
        results = self.index.recall_recent(
            topN=self.topN, half_life=self.half_life, now=now)
        if exclude_record_ids:
            results = [row for row in results if _chunk_key(row["text"])
                       not in exclude_record_ids]
        return format_recall_block(results, time_context=self.time_context)

    def on_session_start(self, now=None, status=None):
        """触发点一·换窗：session 启动时调一次。
        返回可注入开场 context 的文本块（无可召回内容时 None）。
        顺带把压缩计数器清零——新窗口的对话量从零累计。

        status: 可选的 degradation_protocol.SessionStatus。宿主启动时若能填出四项
        故障信号，先过程序化自查关卡：有信号就不接入召回记忆，返回"可能在退化，
        请核实"的提示——时间感塌陷的窗口拿着旧记忆只会往错的方向自圆其说，先拦。
        没给 status（宿主查不了）时照常注入，靠块尾自查指令让模型自检。
        与 degradation_protocol v2 核心规则一致：只认故障信号，信号干净正常放行，
        不做"疑神疑鬼"式拦截。

        三层注入（规格 §5）：当前未解决先给，thread 只作上个窗口历史快照，再给
        召回块（广度："最近发生过什么、什么曾经重要"）；缺哪层就跳过，不断线。
        退化关卡在三层之前——
        窗口真漂移时一个字都不注入。"""
        self._chars_since_recall = 0
        if status is not None:
            signals = degradation_signals(status)
            if signals:
                return ("【开场自查未通过】检测到" + "/".join(signals) +
                        "，本窗口可能在退化：先不接入召回记忆，请对方核实后再继续。")
        blocks, unresolved_ids = [], set()
        if self.unresolved_store is not None:
            try:
                unresolved, bad_lines = self.unresolved_store.read(allow_partial=True)
                unresolved_block = format_unresolved_block(unresolved, bad_lines)
                unresolved_ids = source_record_ids(unresolved)
            except UnresolvedStoreError as exc:
                unresolved_block = ("【当前未解决】⚠ 清单读取失败，未静默当成空清单："
                                    f"{exc}。上次会话与最近记忆仍继续召回。")
            if unresolved_block:
                blocks.append(unresolved_block)
        if self.thread_store is not None:
            # 时区跟召回块传同一个：两块贴在一起注入，日期口径不许分家
            thread_block = format_thread_block(self.thread_store.latest(), now=now,
                                               time_context=self.time_context)
            if thread_block:
                blocks.append(thread_block)
        recall_block = self._recall(now=now, exclude_record_ids=unresolved_ids)
        if recall_block:
            blocks.append(recall_block)
        if blocks:
            blocks.append(SELF_CHECK_FOOTER)
        return "\n\n".join(blocks) if blocks else None

    def on_turn(self, turn_text, now=None):
        """触发点二·压缩近似：每轮对话文本喂进来，累计字符量过阈值即触发并清零。

        再强调一次局限（docstring 顶部详述）：这不是"压缩刚发生"的精确信号，
        只是"对话长到差不多该压缩了"的粗估。返回召回文本块或 None（未过阈值）。"""
        self._chars_since_recall += len(turn_text)
        if self._chars_since_recall < self.compact_threshold_chars:
            return None
        self._chars_since_recall = 0
        block = self._recall(now=now)
        return block + "\n" + SELF_CHECK_FOOTER if block else None


# ---------- selftest（合成数据，全部虚构） ----------

def _selftest():
    DAY = 86400.0
    now = 1_800_000_000.0  # 固定时刻，不依赖真实时钟

    idx = MemoryIndex()
    idx.add("## 昨天\n昨天调好了咖啡机的研磨度。", {"heading": "昨天", "timestamp": now - DAY})
    idx.add("## 上周\n上周整理了阳台的花盆。", {"heading": "上周", "timestamp": now - 7 * DAY})
    idx.add("## 上月\n上月读完了一本小说。", {"heading": "上月", "timestamp": now - 30 * DAY})

    # 1. 换窗触发：启动即召回，最新的排最前，topN 截断生效
    sr = SessionRecall(idx, topN=2, compact_threshold_chars=100)
    block = sr.on_session_start(now=now)
    assert block is not None and "昨天" in block.splitlines()[1], "最新记忆排最前"
    assert "上月" not in block, "topN=2 截断"

    # 2. 压缩近似触发：累计没过阈值不触发
    assert sr.on_turn("x" * 40, now=now) is None
    assert sr.on_turn("x" * 40, now=now) is None

    # 3. 累计过阈值 → 触发一次
    compact_block = sr.on_turn("x" * 40, now=now)
    assert compact_block is not None and "昨天" in compact_block, "累计 120 ≥ 阈值 100，触发"

    # 4. 触发后计数器清零，不连发
    assert sr.on_turn("x" * 40, now=now) is None, "刚触发过，重新从零累计"

    # 5. 再次累计过阈值 → 可重复触发
    sr.on_turn("x" * 40, now=now)
    assert sr.on_turn("x" * 40, now=now) is not None, "第二轮累计过阈值，再次触发"

    # 6. 换窗把压缩计数器清零：窗口边界之前的累计不带进新窗口
    sr2 = SessionRecall(idx, topN=1, compact_threshold_chars=100)
    sr2.on_turn("x" * 90, now=now)
    sr2.on_session_start(now=now)
    assert sr2.on_turn("x" * 40, now=now) is None, "换窗后对话量从零累计"

    # 7. 空索引兜底：不崩，两个触发点都返回 None（不注入空壳标题也不注入自查指令）
    empty = SessionRecall(MemoryIndex(), compact_threshold_chars=10)
    assert empty.on_session_start(now=now) is None
    assert empty.on_turn("x" * 20, now=now) is None

    # 8.【防时间感塌陷·变异靶心】每条片段带发生日期，块头写明"不是正在发生"
    assert "不是正在发生" in block.splitlines()[0], "块头声明这是历史片段"
    assert re.search(r"- \[\d{4}\.\d{2}\.\d{2}·昨天\]", block), "片段标注发生日期"

    # 8b.【新旧仲裁·变异靶心】（2026.08.03 补，验收变异测试打出来的洞：仲裁那两句
    #     整段删掉，全仓 22 个自检照样绿——**这句话是"防旧事实冒充现状"的唯一防线，
    #     却没有任何断言守着**）。断言守的是"这句话还在、两半都在"；模型认不认是
    #     另一件事，那要真机实测（任务卡「新旧仲裁口径」第三节），两者不互相替代。
    head_line = block.splitlines()[0]
    assert "当下状态" in head_line and "读作历史" in head_line, \
        "块头要给出新旧仲裁口径：同类相关记忆里最近一条是当下状态、更早的读作历史"
    assert "时间未知" in head_line and "不当作当下状态" in head_line, \
        "缺时间戳那一格必须有说法——它是'旧事实冒充现状'最容易钻的一格"

    # 9.【不假装知道·变异靶心】没有真实日期依据的一律标「时间未知」：缺时间戳的，
    #    以及只能退到 mtime 的。**mtime 那档 2026.08.03 从 "≈某日" 改过来**，
    #    因为 mtime 在全新 clone／刚落盘的语料上就是"今天"，打一个日期出去等于
    #    让最没有时间依据的片段拿到全库最新的日期，仲裁句会照规则把它端成当下状态。
    #    实测 A/B（DS／glm-5.2 各 3 次独立会话）：改前冒充现状 5/6、答对 0/6；
    #    改后冒充现状 0/6、答对 5/6（各有 1 次检索没命中，不计冒充）。
    #    ⚠ 这条断言守的就是那次实测的结论，**别顺手改回带日期的形态**。
    idx9 = MemoryIndex()
    idx9.add("近似时间的事", {"heading": "近似", "timestamp": now - DAY, "timestamp_source": "mtime"})
    idx9.add("没有时间的事", {"heading": "没戳"})
    b9 = SessionRecall(idx9, topN=2).on_session_start(now=now)
    assert "[时间未知·近似]" in b9, "mtime 兜底的块必须标时间未知——打日期就是假装知道"
    assert not re.search(r"\[≈?\d{4}\.\d{2}\.\d{2}·近似\]", b9), \
        "mtime 那档不许出现任何日期形态的标签（带不带 ≈ 都不行）"
    assert "[时间未知·没戳]" in b9, "缺时间戳标时间未知，不崩"

    # 10. 自查指令跟着每次注入走：换窗和压缩触发的块尾都带
    assert block.endswith(SELF_CHECK_FOOTER) and compact_block.endswith(SELF_CHECK_FOOTER)

    # 11.【防退化窗口自圆其说·变异靶心】程序化自查关卡：
    #    信号干净 → 正常注入（不做"疑神疑鬼"式拦截，与 degradation_protocol v2 一致）
    b_clean = sr.on_session_start(now=now, status=SessionStatus())
    assert b_clean is not None and "昨天" in b_clean, "没有故障信号不拦截"
    #    有故障信号 → 不接入召回记忆，返回"请核实"，召回内容一个字都不出现
    b_broken = sr.on_session_start(now=now, status=SessionStatus(time_check_ok=False))
    assert "时间错乱" in b_broken and "核实" in b_broken, "带出检测到的信号并请求核实"
    assert "昨天" not in b_broken and SELF_CHECK_FOOTER not in b_broken, "退化窗口不注入召回记忆"

    # ---- 未解决／会话线索／最近召回三层注入（规格 §5）----
    from session_thread import ThreadStore, close_thread

    # 12. 没有未解决清单时仍是 thread 历史快照在 recent 前；旧 open_loops 不再冒充当前集合
    store = ThreadStore()
    store.append(close_thread(12, now - 2 * DAY, now - DAY, ("阳台的花",),
                              "花买好了，周末的事没定。", open_loops=("周末去哪还没定",)))
    sr_t = SessionRecall(idx, topN=2, thread_store=store)
    b12 = sr_t.on_session_start(now=now)
    assert b12.startswith("【上次会话】"), "thread 块该排在最前"
    assert b12.index("【上次会话】") < b12.index("【最近记忆召回】"), "thread 在召回之前"
    assert "周末去哪还没定" not in b12 and "历史快照" in b12
    assert b12.endswith(SELF_CHECK_FOOTER), "自查指令仍在末尾"

    # 13. 没有 thread_store 就只有召回块——不断线（退回原行为）
    b13 = SessionRecall(idx, topN=2).on_session_start(now=now)
    assert b13.startswith("【最近记忆召回】") and "【上次会话】" not in b13

    # 14. 空 store 同样退回召回块；索引空但有 thread 时，thread 也得能单独注入且带自查
    assert SessionRecall(idx, topN=2, thread_store=ThreadStore()
                         ).on_session_start(now=now).startswith("【最近记忆召回】")
    only_thread = SessionRecall(MemoryIndex(), thread_store=store).on_session_start(now=now)
    assert only_thread.startswith("【上次会话】") and only_thread.endswith(SELF_CHECK_FOOTER)

    # 15. 退化关卡优先于三层注入——窗口漂移时 thread 也不给
    b15 = sr_t.on_session_start(now=now, status=SessionStatus(time_check_ok=False))
    assert "【上次会话】" not in b15 and "核实" in b15, "退化窗口连 thread 都不该注入"

    # 16.【变异靶心：单条截断】长片段被截断并标注，短的原样不动
    long_idx = MemoryIndex()
    long_idx.add("长" * 2000, {"heading": "很长的一块", "timestamp": now - DAY})
    long_idx.add("短块", {"heading": "短的", "timestamp": now - DAY})
    b16 = format_recall_block(long_idx.recall_recent(topN=2, now=now))
    assert TRUNCATION_NOTE in b16 and "短块" in b16, "长片段该截断、短片段该原样保留"
    assert len(b16) < 1200, f"截断后整块该显著变小，实际 {len(b16)}"
    #    关掉截断时原样给全（旋钮真的可关）
    assert TRUNCATION_NOTE not in format_recall_block(
        long_idx.recall_recent(topN=2, now=now), max_item_chars=None)

    # 17.【跨时区标注·变异靶心】（2026.08.04 手机 Connector 真机事故，任务卡
    #    「写回时区与跨日归窗」）：标签上的日期是**记忆所有者时区里的那一天**，
    #    不是读它的这个进程所在时区的那一天。事故现场 UTC 的 VPS 上，东八区凌晨
    #    02:05 写下的记忆被标成 `[2026.08.03]`，而仲裁句要求"以日期最近的一条为
    #    当下状态"——日期错一天，模型对"现在是什么状态"的判断就跟着错。
    #    变异：把 `_time_label` 里的 local_date 那一支删掉 → 下面这批红。
    from datetime import timezone as _tz
    from time_context import tzdb_available
    east8 = "Asia/Shanghai" if tzdb_available() else "UTC+08:00"   # 采集条件见 time_context
    epoch18 = datetime(2026, 8, 3, 18, 5, 0, tzinfo=_tz.utc).timestamp()
    #    a) 有 local_date 就用它，一个字都不换算——它已经是结论
    assert _time_label({"timestamp": epoch18, "local_date": "2026-08-04",
                        "timestamp_source": "append"}) == "2026.08.04", \
        "写回时定死的自然日必须原样标出来，不许在读的时候按宿主时区重算一遍"
    #    b) 旧 meta（没有 local_date）才退回按 epoch 格式化，且要用配置的时区
    old_meta = {"timestamp": epoch18, "timestamp_source": "append"}
    assert _time_label(old_meta, TimeContext(east8)) == "2026.08.04"
    assert _time_label(old_meta, TimeContext("UTC")) == "2026.08.03", \
        "同一个 epoch 在不同时区就是不同的日历日——这个数是配置的函数，不是常量"
    #    c) mtime 那档不许被 local_date 救活：没有日期依据就是没有（第 9 项那条结论）
    assert _time_label({"timestamp": epoch18, "local_date": "2026-08-04",
                        "timestamp_source": "mtime"}) == "时间未知"

    # 18. 当前未解决在历史快照／recent 前；同一 recordId 的 recent 块退出，不做语义猜测
    import tempfile
    from pathlib import Path
    from unresolved_state import UnresolvedStore
    with tempfile.TemporaryDirectory() as td:
        unresolved_store = UnresolvedStore(Path(td) / "未解决.md")
        duplicate = "## 今天\n这块正文同时被标成未解决来源。"
        record_id = _chunk_key(duplicate)
        unresolved_store.apply([{"action": "open", "summary": "还差最终确认"}],
                               f"timeline/window_03.md#record={record_id}")
        idx18 = MemoryIndex()
        idx18.add(duplicate, {"heading": "今天", "timestamp": now})
        idx18.add("另一条最近记忆", {"heading": "另一条", "timestamp": now - DAY})
        b18 = SessionRecall(idx18, topN=2, thread_store=store,
                            unresolved_store=unresolved_store).on_session_start(now=now)
        assert b18.startswith("【当前未解决】") and "还差最终确认" in b18
        assert "这块正文同时被标成未解决来源" not in b18, "同 recordId 的 recent 块应退出"
        assert "另一条最近记忆" in b18 and b18.index("【上次会话】") < b18.index("【最近记忆召回】")

    print("selftest ok（18项断言：触发时机 / 时间标注防塌陷 / 跨时区标注 / 自查关卡 / "
          "threads 历史快照 / 未解决优先与确定性去重 / 单条截断）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        ap.print_help()
