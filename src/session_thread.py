#!/usr/bin/env python3
"""
会话线索 threads 参考实现（设计规格《人格md与记忆库规格》§5，任务卡"会话线索threads"）。

**一次会话 = 一条 thread**（B 类语义：上次聊到哪），不是"一件事的来龙去脉"。
跟未解决清单／recall_recent 的分工：
  未解决清单   当前仍没有结束的事——唯一事实源
  recall_recent  散块，"最近发生过什么、什么曾经重要"——广度
  thread         一个单元，"上次那条线聊到哪、结束时是什么状态"——历史快照
换窗时三层一起给：当前未解决优先，thread 与召回块随后；没有某层就跳过，不断线。
接线在 session_recall.SessionRecall。

**单独成文件，不进人格 md**（规格 §5）：人格 md 是不变量层，每次会话往里追加会
让它慢慢变成日志；而最高权重的结尾位置该留给最终约定，不是留给会话日志。

**收尾谁写**：
  模型主动收尾 → close_thread()，当下状态必填（病灶迁移那条纪律的产品化）
  写不到收尾 → auto_close_thread() 从最后几轮兜底生成，source="auto"，**注入时
               明确标注自动生成、未经确认**——跟时间戳 mtime 兜底同一个思路：
               成色标出来，不假装是模型确认过的判断
窗口极少优雅结束（压缩/掉线/直接关掉），所以兜底不是可选项，是主路径之一。

**确认分级**（规格 §7）：thread 收尾是过程记录，自动写、带来源标注、可回看可改，
不走用户确认关卡——只有动到人格 md 才需要用户确认。每次换窗都弹确认没人受得了。

**时间怎么写**（规格 §3.2）：thread 块是动态召回，目的就是传达"这是多久以前"，
所以用真实日期 + 距今天数；窗口号作为标识一并给出（静态 md 里才是只用窗口号）。

零依赖，stdlib only。
用法：
  python session_thread.py --selftest
"""

import argparse
import json
import time
from dataclasses import dataclass, asdict, field as dc_field
from datetime import datetime, timezone as _tz
from pathlib import Path

# 记忆所有者的时区（任务卡"写回时区与跨日归窗"）：注入块上的日期跟召回块同一个口径
from time_context import TimeContext

# 自动兜底时回看的对话轮数：经验值，跟检索层 +0.05/k=60 同一待遇，待真实评估再调
DEFAULT_TAIL_TURNS = 6
# 兜底生成的当下状态前缀——成色标注写死在内容里，下游不加工也不会丢
AUTO_STATE_PREFIX = "（自动生成，未经确认）"


class ThreadValidationError(Exception):
    pass


@dataclass
class Thread:
    """一条会话线索。字段各防一种失效：
      window        标识这是第几个窗（静态 md 侧的定位标签，规格 §3.2）
      started_at/ended_at  真实时间戳，注入时算"多久以前"用
      topics        聊了什么线——换窗后接话的抓手
      current_state 当下状态，病灶迁移那条纪律：不写就会被下个窗口读成正在发生
      open_loops    旧客户端兼容字段；不再作为当前未解决集合渲染
      source        model(主动收尾) / auto(兜底生成，未经确认)"""
    window: int
    started_at: float
    ended_at: float
    topics: tuple = ()
    current_state: str = ""
    open_loops: tuple = ()
    source: str = "model"

    def validate(self):
        if not isinstance(self.window, int) or isinstance(self.window, bool) or self.window < 1:
            raise ThreadValidationError(f"window 必须是正整数窗口号，收到 {self.window!r}")
        if self.ended_at < self.started_at:
            raise ThreadValidationError("ended_at 早于 started_at")
        if self.source not in ("model", "auto"):
            raise ThreadValidationError(f"未知 source：{self.source}")
        if not self.current_state.strip():
            # 主动收尾没写状态是纪律问题，兜底路径由 auto_close_thread 保证非空
            raise ThreadValidationError("thread 缺当下状态——不写会被下个窗口读成正在发生")
        return self


def close_thread(window, started_at, ended_at, topics, current_state, open_loops=()):
    """模型主动收尾。当下状态必填——这是病灶迁移纪律落到代码上的地方。"""
    return Thread(window=window, started_at=started_at, ended_at=ended_at,
                  topics=tuple(topics), current_state=current_state.strip(),
                  open_loops=tuple(open_loops), source="model").validate()


def auto_close_thread(window, started_at, turns, now=None, tail=DEFAULT_TAIL_TURNS):
    """兜底收尾：窗口没能优雅结束时，从最后几轮粗糙生成一条 thread。

    只做机械摘要（取最后几轮的首句当线索），不调模型、不猜状态——猜出来的状态
    比没有更危险。当下状态一律带 AUTO_STATE_PREFIX，成色写死在内容里。
    turns: [(speaker, text)] 或 [text]，空列表也不崩。"""
    now = time.time() if now is None else now
    picked = []
    for t in list(turns)[-tail:]:
        text = t[1] if isinstance(t, (tuple, list)) and len(t) >= 2 else str(t)
        first = text.strip().splitlines()[0].strip() if text.strip() else ""
        if first:
            picked.append(first[:40])
    state = AUTO_STATE_PREFIX + ("窗口未正常收尾，以上是中断前最后几轮的话头。"
                                 if picked else "窗口未正常收尾，且没有可回看的对话。")
    return Thread(window=window, started_at=started_at, ended_at=now,
                  topics=tuple(picked), current_state=state,
                  open_loops=(), source="auto").validate()


class ThreadStore:
    """JSONL 文件存取（一行一条 thread，追加写）。零依赖，也顺带让 thread 这一层
    先有了持久化——检索层的持久化仍是未决项（规格 §9）。path=None 时纯内存态。"""

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self._mem = []

    def append(self, thread: Thread):
        thread.validate()
        if self.path is None:
            self._mem.append(thread)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(thread), ensure_ascii=False) + "\n")
        return thread

    def all(self):
        if self.path is None:
            return list(self._mem)
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d["topics"], d["open_loops"] = tuple(d.get("topics", ())), tuple(d.get("open_loops", ()))
            out.append(Thread(**d))
        return out

    def latest(self):
        """最近一条——按窗口号取，不是按写入顺序：补写旧窗口的收尾不该顶掉新窗口。"""
        threads = self.all()
        return max(threads, key=lambda t: (t.window, t.ended_at)) if threads else None


def format_thread_block(thread: Thread, now=None, time_context=None):
    """thread → 可注入 context 的文本块。带真实日期与距今天数（规格 §3.2：动态召回
    的目的就是传达时间距离），来源成色明确标出。

    ⚠ **日期按记忆所有者的时区算，跟召回块同一个口径**（2026.08.04 跨时区事故，
    任务卡"写回时区与跨日归窗"）：这一块就贴在召回块上面一起注入，这里裸着
    `fromtimestamp` 的话，UTC 的服务器上东八区深夜那次会话会写成前一天——**同一屏
    里两个日期口径**比单独错更坏，模型会拿它们互相校准。距今天数仍按真实经过时间
    算（那是时间距离，不是日历日），不受时区影响。"""
    if thread is None:
        return None
    now = time.time() if now is None else now
    day = (time_context or TimeContext.default()).label(thread.ended_at)
    gap_days = max(0, int((now - thread.ended_at) // 86400))
    when = "今天" if gap_days == 0 else f"{gap_days} 天前"
    lines = [f"【上次会话】第 {thread.window} 个窗口，{day}（{when}）：",
             "以下是上个窗口结束时的历史快照，不代表其中事项目前仍未解决。"]
    if thread.topics:
        lines.append("- 聊了：" + "；".join(thread.topics))
    lines.append("- 当时状态：" + thread.current_state)
    # `open_loops` 旧字段继续读写，避免已有 threads.jsonl 失兼容；但当前未解决集合
    # 只认 `<corpus>/未解决.md`。这里再渲染会形成两本会互相漂移的账。
    if thread.source == "auto":
        lines.append("- 注意：这条收尾是窗口中断后自动生成的，未经确认，别当成结论。")
    return "\n".join(lines)


# ---------- selftest（合成数据，全部虚构） ----------

def _selftest():
    DAY = 86400.0
    now = 1_800_000_000.0
    t0 = now - 2 * DAY

    # 1.【变异靶心：validate 的当下状态必填】主动收尾缺状态即拒（病灶迁移纪律）
    try:
        close_thread(3, t0, t0 + 3600, ("阳台的花",), "   ")
        assert False, "缺当下状态该被拒绝"
    except ThreadValidationError:
        pass

    # 2. 窗口号必须是正整数——日期/0/布尔都不收
    for bad in ("2026-07-15", 0, -1, True):
        try:
            close_thread(bad, t0, t0 + 60, (), "聊完了")
            assert False, f"window={bad!r} 该被拒绝"
        except ThreadValidationError:
            pass

    # 3. 正常主动收尾
    th = close_thread(12, t0, t0 + 3600, ("阳台的花", "周末去哪"),
                      "花买好了，周末的事没定。", open_loops=("周末去哪还没定",))
    assert th.source == "model" and th.window == 12

    # 4.【变异靶心：auto 的成色标注】兜底收尾必须自带"未经确认"，且不猜状态
    auto = auto_close_thread(13, t0, [("林岸", "先这样吧"), ("星回", "好，等你回来接着说")], now=now)
    assert auto.source == "auto" and auto.current_state.startswith(AUTO_STATE_PREFIX)
    assert auto.topics == ("先这样吧", "好，等你回来接着说"), f"取最后几轮话头：{auto.topics}"
    #    tail 生效：只取最后几轮，不是全量
    long_auto = auto_close_thread(13, t0, [f"第{i}句" for i in range(20)], now=now, tail=3)
    assert long_auto.topics == ("第17句", "第18句", "第19句"), f"tail 截断：{long_auto.topics}"
    #    没有可回看的对话也不崩
    empty_auto = auto_close_thread(14, t0, [], now=now)
    assert empty_auto.topics == () and AUTO_STATE_PREFIX in empty_auto.current_state

    # 5.【变异靶心：format 里的日期与距今】注入块带真实日期和时间距离
    block = format_thread_block(th, now=now)
    #    距今天数向下取整（1.96 天说成 1 天前）——宁可少说，不夸大时间距离
    assert "第 12 个窗口" in block and "1 天前" in block, block
    assert datetime.fromtimestamp(th.ended_at).strftime("%Y.%m.%d") in block, "带真实日期"
    assert "花买好了" in block and "周末去哪还没定" not in block
    assert "历史快照" in block and "当时状态" in block
    assert "未经确认" not in block, "主动收尾的块不该带自动生成标注"
    #    auto 来源的块必须带标注
    assert "未经确认" in format_thread_block(auto, now=now)
    #    当天结束的显示"今天"
    assert "今天" in format_thread_block(
        close_thread(15, now - 600, now - 60, (), "刚聊完"), now=now)
    assert format_thread_block(None) is None
    #    【跨时区·变异靶心】（2026.08.04 事故，任务卡「写回时区与跨日归窗」）：
    #    这一块的日期也得按记忆所有者的时区算——它跟召回块贴在一起注入，口径分家
    #    比单独错更坏。把上面那行改回裸 fromtimestamp → 下面这两条在非东八区的机器上红。
    from time_context import tzdb_available
    east8 = "Asia/Shanghai" if tzdb_available() else "UTC+08:00"   # 采集条件见 time_context
    #    东八区 2026-08-04 02:05（＝UTC 08-03 18:05）结束的那次会话
    tz_end = datetime(2026, 8, 3, 18, 5, 0, tzinfo=_tz.utc).timestamp()
    tz_th = close_thread(16, tz_end - 3600, tz_end, ("凌晨聊的",), "聊完了")
    assert "2026.08.04" in format_thread_block(tz_th, now=tz_end + 60,
                                               time_context=TimeContext(east8)), \
        "上次会话的日期该按记忆所有者的时区算，不是按跑服务那台机器的"
    assert "2026.08.03" in format_thread_block(tz_th, now=tz_end + 60,
                                               time_context=TimeContext("UTC")), \
        "同一个 epoch 在不同时区就是不同的日历日——这个数是配置的函数，不是常量"

    # 6. 存取往返：内存态与文件态行为一致
    import tempfile
    for path in (None, "x"):
        with tempfile.TemporaryDirectory() as td:
            store = ThreadStore(None if path is None else Path(td, "sub", "threads.jsonl"))
            assert store.latest() is None, "空 store 该返回 None"
            store.append(th)
            store.append(auto)
            got = store.latest()
            assert got.window == 13 and got.source == "auto", "latest 取窗口号最大的那条"
            assert len(store.all()) == 2
            assert isinstance(store.all()[0].topics, tuple), "读回来 topics 仍是 tuple"

    # 7. latest 按窗口号取，不按写入顺序——补写旧窗口的收尾不该顶掉新窗口
    store2 = ThreadStore()
    store2.append(close_thread(20, t0, t0 + 60, (), "新窗口"))
    store2.append(close_thread(5, t0, t0 + 60, (), "补写的旧窗口"))
    assert store2.latest().window == 20, "补写旧窗口不该顶掉新窗口"

    print("selftest ok（8项断言：状态必填 / 窗口号 / 兜底标注 / 注入日期 / 注入日期按所有者时区 / 存取 / latest 语义）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        ap.print_help()
