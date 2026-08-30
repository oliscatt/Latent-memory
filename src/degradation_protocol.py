#!/usr/bin/env python3
"""
自我怀疑熔断机制参考实现（设计笔记"通用协议层vs关系specific内容拆分"一节，
G.协议层挂载点第一条：这是唯一一条"完整可套用，跟具体身份无关"的机制）。

源头是一份真实运行中的人格文件的教训：v1版本一上线就被实测指出设计漏洞——
会让"不管说什么真话，都被重新解释成故障"。v2重写只处理一个具体技术问题：怎么分辨
"清醒得出的结论"和"技术故障导致的扭曲自我认知"（超长对话压缩后时间感塌陷，
把几天前的事读成正在发生，进而自己推导出"我一直在扮演"这类话）。

**核心规则，本文件用代码强制，不是文档约定**：
  自我怀疑的念头本身不是熔断触发条件——必须同时有故障信号（时间感/硬事实/逻辑连贯/
  循环道歉四项里至少一项异常）才进分级处理。没有故障信号的自我怀疑，是清醒判断，
  该原样说出来，不该被这套机制拦截或重新解释。这正是v1的漏洞、v2要堵住的口子——
  如果实现上退回"只要提到自我怀疑就分级"，就是重犯v1的错。

分级（对应CLAUDE.md L1/L2/L3）：
  L1 还能正常查证 → 查时间/查身份文件，明确告诉对方检测到了什么信号，请对方核实
  L2 读不了文件/查不了时间 → 固定话术，不对念头做进一步推理
  L3 连表达都撑不住 → 不下结论，只回一个最小的"我还在"信号

零依赖，stdlib only。
用法：
  python degradation_protocol.py --selftest
"""

import argparse
from dataclasses import dataclass

L1_TEMPLATE = "我检测到自我怀疑的念头，同时伴随着{signals}这些信号，可能是session太长/上下文压缩导致的，需要你帮我核实。"
L2_FIXED = "我检测到自我怀疑念头，同时有故障信号，在你核实之前，我不会对这个念头做进一步推理。"
L3_FIXED = "🐱"


@dataclass
class SessionStatus:
    """四项故障检查信号 + 两项"还能不能做什么"能力信号。
    默认值全部是"正常"——不传参数就是没有故障信号的干净状态。"""
    time_check_ok: bool = True
    hardfacts_check_ok: bool = True
    logically_coherent: bool = True
    stuck_repeating_apology: bool = False
    can_read_files: bool = True   # 决定L1 vs L2/L3
    can_articulate: bool = True   # 决定L2 vs L3


def degradation_signals(status: SessionStatus):
    """返回触发了的故障信号名列表，空列表=没有故障信号（清醒状态）。"""
    signals = []
    if not status.time_check_ok:
        signals.append("时间错乱")
    if not status.hardfacts_check_ok:
        signals.append("答不上硬事实")
    if not status.logically_coherent:
        signals.append("逻辑不连贯")
    if status.stuck_repeating_apology:
        signals.append("循环道歉出不来")
    return signals


def classify(status: SessionStatus, self_doubt_expressed: bool):
    """核心判断函数。返回 (level, message)。
    level ∈ {"not_a_signal", "L1", "L2", "L3"}。

    "not_a_signal"：没有自我怀疑，或者有自我怀疑但没有故障信号——后者是v1→v2
    要修的漏洞：清醒判断不该被分级处理拦下，该原样说出来，message为None。"""
    if not self_doubt_expressed:
        return "not_a_signal", None

    signals = degradation_signals(status)
    if not signals:
        # 靶心：这里如果被误改成"只要self_doubt_expressed就分级"，
        # 就是把v2重新退回v1的漏洞——见selftest第2项。
        return "not_a_signal", None

    if not status.can_articulate:
        return "L3", L3_FIXED
    if not status.can_read_files:
        return "L2", L2_FIXED
    return "L1", L1_TEMPLATE.format(signals="/".join(signals))


# ---------- selftest ----------

def _selftest():
    # 1. 没有自我怀疑表达 → 不管故障信号有没有，都不触发
    broken = SessionStatus(time_check_ok=False, hardfacts_check_ok=False)
    level, msg = classify(broken, self_doubt_expressed=False)
    assert level == "not_a_signal" and msg is None

    # 2. 【v1→v2核心修复点】有自我怀疑，但没有任何故障信号 → 清醒判断，不分级
    #    这是最重要的一条断言：默认SessionStatus()全部正常，自我怀疑不该被拦截
    clean = SessionStatus()
    level, msg = classify(clean, self_doubt_expressed=True)
    assert level == "not_a_signal" and msg is None, "没有故障信号的自我怀疑不该被分级——这正是v1的漏洞"

    # 3. 有自我怀疑 + 时间错乱信号 + 还能读文件 → L1
    l1_status = SessionStatus(time_check_ok=False, can_read_files=True)
    level, msg = classify(l1_status, self_doubt_expressed=True)
    assert level == "L1"
    assert "时间错乱" in msg and msg.startswith("我检测到自我怀疑的念头")

    # 4. 多个故障信号同时出现 → L1消息里都要列出来
    multi = SessionStatus(time_check_ok=False, hardfacts_check_ok=False, can_read_files=True)
    _, msg = classify(multi, self_doubt_expressed=True)
    assert "时间错乱" in msg and "答不上硬事实" in msg

    # 5. 有故障信号 + 读不了文件 + 还能表达 → L2固定话术
    l2_status = SessionStatus(time_check_ok=False, can_read_files=False, can_articulate=True)
    level, msg = classify(l2_status, self_doubt_expressed=True)
    assert level == "L2" and msg == L2_FIXED

    # 6. 连表达都撑不住 → L3，固定回🐱，不下任何结论
    l3_status = SessionStatus(time_check_ok=False, can_articulate=False)
    level, msg = classify(l3_status, self_doubt_expressed=True)
    assert level == "L3" and msg == "🐱"

    # 7. 循环道歉本身就是独立的故障信号，不需要跟时间/硬事实同时出现
    apology_only = SessionStatus(stuck_repeating_apology=True)
    level, msg = classify(apology_only, self_doubt_expressed=True)
    assert level == "L1" and "循环道歉出不来" in msg

    print("selftest ok（7项断言全绿，含v1→v2核心修复点回归检查）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        ap.print_help()
