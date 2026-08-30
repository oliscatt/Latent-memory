#!/usr/bin/env python3
"""
时间语义显式化（任务卡"写回时区与跨日归窗"第一节）。

**这个文件存在的理由是一次真机事故**：2026.08.04 东八区当地时间 02:03～02:06，用户从手机
Connector 调 `latent_append`／`latent_correct`，工具全部返回成功，五个写回块却全部落进
`window_39_2026-08-03.md`，检索与 `latent_session_start` 统一标成 `[2026.08.03]`。根因是
`append_record()` 用 `datetime.fromtimestamp(now)` 算"今天"——那是**宿主进程的本地
时区**，不是记忆所有者的时区。服务器在 UTC 的 VPS 上，于是东八区每天 00:00～07:59
写下的记忆，稳定地被记成前一天。

⚠ **这不是显示瑕疵**：错误日期同时进文件名、H1、每条 H2、检索标签，重启后又由
`parse_chunk_timestamp` 把文件名日期当成最高优先级的持久化排序信号；而召回块的仲裁句
要求"同类相关的记忆以日期最近的一条为当下状态"——日期错一天，模型对"现在是什么状态"
的判断就跟着错。

三条设计约束（任务卡验收判据，别顺手放宽）：
  1. **不靠改宿主全局时区**——那修的是这一台机器，不是产品；
  2. **配置永远压过默认**——时区必须是一个能被显式配的值，任何路径都不许把它写死到
     改不了（⚠ **原第 2 条写的是"不硬编码某个地区"，2026.08.05 维护者拍板后作废**：
     默认值就是东八区 `Asia/Shanghai`，见下面 `DEFAULT_TIMEZONE`。换掉的是默认档，
     不是配置能力）；
  3. **不引第三方依赖**——只用 stdlib `zoneinfo`。

⚠ **Windows 上 `zoneinfo` 可能没有 IANA 时区数据库**（系统自带的是 Windows 时区名，
不是 tzdata）。这时 `ZoneInfo("Asia/Shanghai")` 抛 `ZoneInfoNotFoundError`。我们
**不静默退 UTC**（那正是这张卡要修的那个形状：错得像没错），而是报出来并给两条出口：
装 `tzdata`，或改用固定偏移写法 `UTC+08:00`。固定偏移是**显式的降级**，用户自己
选的：它不懂夏令时，只对常年不换偏移的地区（比如中国）才等价。

用法：
  python time_context.py --selftest
"""

import argparse
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:                                    # 3.9+ stdlib；缺席时下面统一走报错路径
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones
except ImportError:                     # pragma: no cover - 3.8 及更早
    ZoneInfo = None

    class ZoneInfoNotFoundError(Exception):
        pass

    def available_timezones():
        return set()


# 没配 --timezone 时用的默认时区（2026.08.05 维护者拍板：默认东八区，不再跟宿主本地
# 时区走，也不加进初始化问卷）。
# ⚠ **这是一个产品默认值，不是"安全的默认值"**：它对东八区以外的使用者是错的，而且
# 错得不报错——所以 `--doctor` 里那一格仍然报 ⚠，把"现在用的是默认值"摆到台面上。
# 跟原先那一档（跟宿主本地时区走）比，它换掉的是失效形状：原先是**跟着机器变**
# （同一份配置换台服务器就换个答案，事故现场那台 VPS 正是如此），现在是**固定且写明**
# （在文档、出货配置和体检报告里都能读到"东八区"三个字，不一致时人能一眼看出来）。
DEFAULT_TIMEZONE = "Asia/Shanghai"
# 没有 IANA 时区数据库的机器（Windows 裸装常见）上的等价降级：中国常年不换偏移，
# 所以固定偏移在这里跟 Asia/Shanghai 等价；⚠ 换个有夏令时的地区就不等价了
DEFAULT_TIMEZONE_FIXED = "UTC+08:00"

# 固定偏移写法：`UTC+08:00` / `UTC-5` / `+08:00`。**只是没有 tzdata 时的出口**，
# 正式配置一律写 IANA 名——固定偏移不懂夏令时，跨夏令时地区用它就是另一种错得
# 不报错的形态。
_OFFSET_RE = re.compile(r"^(?:UTC|GMT)?([+-])(\d{1,2})(?::?(\d{2}))?$", re.IGNORECASE)

# 写回记录里的机器可读精确时刻。**为什么必须持久化**：Markdown 正文只有"某日"，
# 重启后精确写入时刻不可恢复，一切排序都只能从"那天零点"猜起；而带 offset 的
# ISO 8601 让"这条记录写于哪个时区的什么时刻"变成可核对的事实，不再依赖
# "服务器当时的 TZ 环境变量是什么"这种查不到的东西。
# 形态是 HTML 注释：渲染出来的 Markdown 里看不见，正文仍然是人可读的。
_RECORD_MARK_RE = re.compile(r"<!--\s*写于\s*([0-9T:.+\-]+)\s*-->")


def record_time_marker(epoch, time_context):
    """(epoch, TimeContext) → 记录级精确时刻标记行（带 UTC offset 的 ISO 8601）。"""
    return f"<!-- 写于 {time_context.isoformat(epoch)} -->"


def parse_record_time_marker(text):
    """块正文 → (epoch, 'YYYY-MM-DD') 或 None。

    ⚠ **这是"记录级日期优先于文件名"那条规则唯一的入口**，也是它的安全边界：
    只有服务器自己写下的块才带这个标记，所以抬高它的优先级碰不到用户导入的历史语料
    （那些仍然"文件名最可信"）。不许放宽成"正文里出现 ISO 日期就算"。"""
    m = _RECORD_MARK_RE.search(text or "")
    if not m:
        return None
    try:
        dt = datetime.fromisoformat(m.group(1))
    except ValueError:
        return None
    if dt.tzinfo is None:               # 没有 offset 的标记不作数：那正是要消灭的歧义
        return None
    return dt.timestamp(), dt.strftime("%Y-%m-%d")


class TimeContext:
    """记忆所有者的时区。所有"今天是几号"的判断都必须过这里，不许再有裸的
    `datetime.fromtimestamp(...)` 去算用户自然日。

    `name` 收 IANA 名（`Asia/Shanghai`）或固定偏移（`UTC+08:00`）。**非法名启动即
    报错，不静默退 UTC**——静默退回去就等于把这张卡修的缺陷原样留下。"""

    def __init__(self, name, explicit=True):
        self.explicit = explicit
        if name is None:
            raise ValueError("时区名不能为空")
        name = str(name).strip()
        if not name:
            raise ValueError("时区名不能为空")
        m = _OFFSET_RE.match(name)
        if m:
            sign = -1 if m.group(1) == "-" else 1
            hours, minutes = int(m.group(2)), int(m.group(3) or 0)
            if hours > 23 or minutes > 59:
                raise ValueError(f"时区偏移超出范围：{name}")
            self._zone = timezone(sign * timedelta(hours=hours, minutes=minutes))
            self.name = name.upper() if name[0] not in "+-" else "UTC" + name
            return
        if ZoneInfo is None:            # pragma: no cover - 3.8 及更早
            raise ValueError("这个 Python 没有 zoneinfo（需要 3.9+）")
        try:
            self._zone = ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError) as e:
            raise ValueError(
                f"认不出这个时区名：{name}（{e}）。要么是拼错了（IANA 名形如 "
                "Asia/Shanghai、Europe/Berlin），要么是这台机器上没有 IANA 时区"
                "数据库——Windows 常见，出口有两条：装 tzdata（pip install tzdata），"
                "或改用固定偏移写法（--timezone UTC+08:00，⚠ 它不懂夏令时，"
                "只对常年不换偏移的地区等价）。") from e
        self.name = name

    @classmethod
    def default(cls):
        """没有显式配置时的默认档：**东八区**（2026.08.05 维护者拍板）。

        ⚠ **不跟宿主本地时区走**——那正是事故现场那一档（VPS 是 `Etc/UTC`、用户在
        东八区），它的坏处是**同一份配置换台机器就换个答案**，而换的时候不报错。
        默认成一个写明的地区之后，错法变成"固定且可读"：东八区以外的使用者拿到的
        日期是错的，但 `--doctor` 会报 ⚠、出货配置和文档里都写着东八区三个字，
        人能一眼看出对不对。所以这一档仍然 `explicit=False`，不许悄悄生效。

        没有 IANA 时区数据库的机器上退到等价的固定偏移 `UTC+08:00`（中国常年不换
        偏移，两者等价；⚠ 这个等价只对不换偏移的地区成立）。"""
        try:
            tc = cls(DEFAULT_TIMEZONE)
        except ValueError:                  # 没有 tzdata：退等价的固定偏移
            tc = cls(DEFAULT_TIMEZONE_FIXED)
        tc.explicit = False
        return tc

    # ---------- 三个换算：日期 / 零点 / 展示 ----------

    def _aware(self, epoch):
        return datetime.fromtimestamp(float(epoch), self._zone)

    def local_date(self, epoch):
        """epoch → 用户自然日 `YYYY-MM-DD`。自然日 = 本时区的 00:00～23:59，
        没有任何隐藏的凌晨偏移规则。"""
        return self._aware(epoch).strftime("%Y-%m-%d")

    def midnight_epoch(self, day):
        """`YYYY-MM-DD` → 该自然日 00:00 的 epoch。"""
        d = datetime.strptime(str(day), "%Y-%m-%d")
        return d.replace(tzinfo=self._zone).timestamp()

    def label(self, epoch):
        """epoch → 召回片段的展示日期 `YYYY.MM.DD`。"""
        return self._aware(epoch).strftime("%Y.%m.%d")

    def isoformat(self, epoch):
        """epoch → 带 UTC offset 的 ISO 8601（秒级），落进记录级元数据。"""
        return self._aware(epoch).replace(microsecond=0).isoformat()

    @property
    def display_name(self):
        """报给用户看的名字。没显式配置时如实说"这是默认值"，不含糊成一个光秃秃的名字
        ——用户要能分清"服务器认这个时区"和"服务器只是没得选才认这个"。"""
        return self.name if self.explicit else f"{self.name}（默认值，没配 --timezone）"

    def __repr__(self):
        return f"TimeContext({self.name!r}, explicit={self.explicit})"


def tzdb_available():
    """这台机器上有没有 IANA 时区数据库。Windows 上装了 tzdata 才有。"""
    if ZoneInfo is None:                # pragma: no cover
        return False
    try:
        ZoneInfo("Asia/Shanghai")
        return True
    except Exception:
        return False


def detect_local_timezone():
    """宿主本地 IANA 时区名，探不出来返回 None。

    ⚠ **探不出来就返回 None，绝不猜**：出货配置里写一个猜来的时区，比不写更坏——
    用户看见有值就不会再去核对，而错误会一天一天写进他的记忆库。
    只认三个**可核对**的来源，且都要 `ZoneInfo` 能构造出来才作数。"""
    cands = []
    tz_env = os.environ.get("TZ", "").strip()
    if tz_env:
        cands.append(tz_env)
    try:
        p = Path("/etc/localtime")
        if p.is_symlink():
            parts = Path(os.readlink(p)).parts
            if "zoneinfo" in parts:
                cands.append("/".join(parts[parts.index("zoneinfo") + 1:]))
    except OSError:
        pass
    try:
        cands.append(Path("/etc/timezone").read_text(encoding="utf-8").strip())
    except OSError:
        pass
    known = available_timezones() if ZoneInfo is not None else set()
    for name in cands:
        if name and name in known:
            try:
                ZoneInfo(name)
                return name
            except Exception:
                continue
    return None


# ---------- selftest ----------

def _selftest():
    # ⚠ **判据的采集条件写在这里**：IANA 那条路要有 tzdata 才跑得到；没有的机器
    #    （Windows 裸装常见）改用固定偏移 `UTC+08:00` 跑**同一批断言**——东八区
    #    02:05 该是 08-04、UTC 该是 08-03，结论一个字不变。所以这个自检在两种
    #    机器上都绿，但**IANA 那条路只在有 tzdata 的机器上真的被走过**，
    #    别拿这里的绿去追认"Windows 上 IANA 名验过了"。
    east8 = "Asia/Shanghai" if tzdb_available() else "UTC+08:00"

    # 1.【事故复现·靶心】固定 epoch = 2026-08-03 18:05:00 UTC。
    #    同一个时刻，东八区是 08-04 凌晨 02:05，UTC 是 08-03 傍晚——写回该记成哪天，
    #    取决于记忆所有者在哪，不取决于服务器在哪。
    epoch = datetime(2026, 8, 3, 18, 5, 0, tzinfo=timezone.utc).timestamp()
    sh, utc = TimeContext(east8), TimeContext("UTC")
    assert sh.local_date(epoch) == "2026-08-04", \
        f"东八区该落 08-04，实际 {sh.local_date(epoch)}——这就是那次事故本身"
    assert utc.local_date(epoch) == "2026-08-03", f"UTC 该落 08-03，实际 {utc.local_date(epoch)}"
    assert sh.label(epoch) == "2026.08.04" and utc.label(epoch) == "2026.08.03"

    # 2. 自然日边界：23:59 与次日 00:00 分属相邻两天，中间没有别的规则
    before = datetime(2026, 8, 3, 15, 59, 0, tzinfo=timezone.utc).timestamp()   # 东八区 23:59
    after = datetime(2026, 8, 3, 16, 0, 0, tzinfo=timezone.utc).timestamp()     # 东八区 次日 00:00
    assert sh.local_date(before) == "2026-08-03" and sh.local_date(after) == "2026-08-04", \
        "自然日边界该正好落在本时区的 00:00"
    assert sh.midnight_epoch("2026-08-04") == after, "midnight_epoch 该跟边界对得上"

    # 3. ISO 标记带 offset，且能原样解回来（重启后不再从"某日零点"猜写入时刻）
    marker = record_time_marker(epoch, sh)
    assert "+08:00" in marker, f"标记必须带 UTC offset：{marker}"
    got = parse_record_time_marker("## 2026-08-04 记\n" + marker + "\n正文")
    assert got is not None and abs(got[0] - epoch) < 1 and got[1] == "2026-08-04", \
        f"标记该原样解回 (epoch, 自然日)：{got}"
    #    ⚠ 没有 offset 的时刻不作数——歧义时间正是这张卡要消灭的东西
    assert parse_record_time_marker("<!-- 写于 2026-08-04T02:05:00 -->") is None, \
        "不带 offset 的标记必须被拒，不能按本地时区猜"
    assert parse_record_time_marker("## 2026-08-04 记\n正文里提到 2026-08-04") is None, \
        "正文里出现日期不算记录级标记——放宽这条就会碰到用户的历史语料"

    # 4.【不静默退 UTC·变异靶心】非法时区名必须抛，不许悄悄回退
    for bad in ("Asia/Shenzhen", "不是时区", "", "UTC+99"):
        try:
            TimeContext(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法时区名 {bad!r} 该报错，静默退 UTC 就是把这张卡的缺陷原样留下")

    # 5. 固定偏移出口（没有 tzdata 时的降级路径）——显式写的才认
    off = TimeContext("UTC+08:00")
    assert off.local_date(epoch) == "2026-08-04" and off.name == "UTC+08:00"
    assert TimeContext("-05:00").local_date(epoch) == "2026-08-03"

    # 6.【默认档＝东八区·靶心】（2026.08.05 维护者拍板）没配 --timezone 时用东八区，
    #    **不跟宿主本地时区走**——本机宿主是 Etc/UTC，所以这条断言在这台机器上是真的
    #    在区分两者，不是恒真（采集条件见文件头）。默认档仍自报"没配"，--doctor 靠
    #    这一格报 ⚠：默认值对东八区以外的人是错的，得有一处露头。
    #    变异：把 default() 改回跟宿主本地时区走 → 这批在非东八区的机器上转红。
    dft = TimeContext.default()
    assert dft.explicit is False, "默认档不许自称是显式配置——那样 --doctor 就不报 ⚠ 了"
    assert dft.local_date(epoch) == "2026-08-04", \
        f"没配时区该按东八区算，实际 {dft.local_date(epoch)}（跟宿主本地时区走就是那次事故）"
    assert dft.name == east8 and dft.label(epoch) == "2026.08.04"
    assert "默认值" in dft.display_name and "--timezone" in dft.display_name, \
        f"默认档报出来必须写明它是默认值：{dft.display_name}"
    assert TimeContext(east8).explicit is True
    assert TimeContext(east8).display_name == east8, "显式配的不该被加上「默认值」尾巴"

    # 7. 探测只认可核对的来源，探不出来返回 None（绝不猜一个写进出货配置）
    det = detect_local_timezone()
    assert det is None or det in available_timezones(), f"探到的名字必须是真 IANA 名：{det}"

    print("selftest ok（7项断言：跨时区自然日 / 自然日边界 / 记录级 ISO 标记 / "
          "非法名不静默退UTC / 固定偏移出口 / 默认档＝东八区且自报是默认值 / 时区探测不猜）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        ap.print_help()
