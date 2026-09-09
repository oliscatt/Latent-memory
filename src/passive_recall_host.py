#!/usr/bin/env python3
"""自动浮现 W0：参考宿主的生命周期、选择与工具协调适配层。

本文件不做检索、不判断相关性、不组装历史证据；这些属于 W1～W4。它只冻结宿主
和后续 ``latent_passive_recall`` 入口之间的请求／响应形状，并管理一次交付在请求、
历史、工具回路、取消和预算里的去向。默认关闭，没有明确模式时绝不调用服务端。
"""

from dataclasses import dataclass
import json
import time
from pathlib import Path

from passive_recall_observation import opaque_id


HIDDEN_TOOL = "latent_passive_recall"
CAPABILITY_VERSION = "reference-host-passive-w5-v1"
MODES = {"temporary", "retained"}
SINGLE_LIMIT = 300
ORDINARY_LIMIT = 2400
STATUS_LIMIT = 600
TOTAL_LIMIT = 3000
FINAL_STATUS_RESERVE = 150


class PassiveRecallConfigError(ValueError):
    """配置或宿主能力不允许启用自动浮现。"""


@dataclass(frozen=True)
class PassiveRecallConfig:
    enabled: bool = False
    mode: str | None = None


def load_config(path):
    """读取宿主侧选择；文件不存在等同默认关闭。"""
    path = Path(path)
    if not path.exists():
        return PassiveRecallConfig()
    raw = json.loads(path.read_text(encoding="utf-8"))
    value = raw.get("passive_recall", {})
    if not isinstance(value, dict):
        raise PassiveRecallConfigError("passive_recall 必须是对象")
    enabled = value.get("enabled", False)
    mode = value.get("mode")
    if not isinstance(enabled, bool):
        raise PassiveRecallConfigError("passive_recall.enabled 必须是布尔值")
    if not enabled:
        return PassiveRecallConfig(False, mode if mode in MODES else None)
    if mode not in MODES:
        raise PassiveRecallConfigError("启用自动浮现前必须明确选择 temporary 或 retained")
    return PassiveRecallConfig(True, mode)


def request_token_upper_bound(message):
    """参考宿主的保守 token 上界：新增消息 UTF-8 线长。

    这是 W0 的可复算计数器，不冒充目标模型的实际计费 token。对字节级分词器，任何
    token 至少承载一个输入字节，因此完整 JSON 消息的 UTF-8 字节数是保守上界。
    后续接入具体宿主时可换成目标模型的准确计数器，但不能换成字符数。
    """
    wire = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
    # 参考宿主的 messages 原本非空；插入一条消息还会新增一个数组逗号。
    return len(wire.encode("utf-8")) + 1


def _public_message(message):
    """剥掉参考宿主内部账本，API 只收到标准 role/content。"""
    return {key: value for key, value in message.items()
            if key in {"role", "content", "name", "tool_call_id", "tool_calls"}}


def _insert_after_latest_user(history, message):
    out = [_public_message(item) for item in history]
    at = next((i + 1 for i in range(len(out) - 1, -1, -1)
               if out[i].get("role") == "user"), len(out))
    out.insert(at, _public_message(message))
    return out


class PassiveRecallAdapter:
    """参考宿主的单会话交付协调器；服务端检索实现以回调注入。"""

    def __init__(self, config=None, fetch=None, timeout_seconds=0.25, clock=None,
                 can_remove_retained=True, token_counter=None,
                 token_counter_name="utf8-json-byte-upper-bound"):
        self.config = config or PassiveRecallConfig()
        self.fetch = fetch
        self.timeout_seconds = float(timeout_seconds)
        self.clock = clock or time.monotonic
        self.can_remove_retained = bool(can_remove_retained)
        self.token_counter = token_counter or request_token_upper_bound
        self.token_counter_name = token_counter_name
        self.turn = None
        self.seen_deliveries = set()
        self.delivery_ledger = {}
        self.ordinary_used = 0
        self.status_used = 0
        self.paused_reason = None

    @property
    def enabled(self):
        return self.config.enabled and self.paused_reason is None

    def begin_turn(self, history, *, session_id, turn_id, delivery_id, user_input,
                   scope=None, previous_anchors=None, context_evidence=None):
        """用户消息入 history 后、第一次模型请求前调用；同 delivery 幂等。"""
        self.turn = {"delivery_id": delivery_id, "message": None,
                     "state": "skipped", "history": history,
                     "session_id": session_id, "scope": scope or "general"}
        if not self.enabled:
            return "disabled" if not self.config.enabled else "paused"
        if delivery_id in self.seen_deliveries:
            self.turn["state"] = "duplicate"
            return "duplicate"
        if not history or history[-1].get("role") != "user" \
                or history[-1].get("content") != user_input:
            raise ValueError("自动浮现必须在用户消息入 history 后、首次模型请求前调度")
        if self.fetch is None:
            self.paused_reason = "宿主没有 latent_passive_recall 入口"
            return "paused"
        if previous_anchors is None:
            previous_anchors = self.previous_anchors()
        if context_evidence is None:
            context_evidence = self.context_evidence()
        request = {
            "userInput": user_input,
            "turn": {"sessionId": session_id, "turnId": turn_id,
                     "deliveryId": delivery_id},
            "scope": scope,
            "previousAnchors": list(previous_anchors),
            "contextEvidence": list(context_evidence),
            "capability": {"host": "reference-host",
                           "version": CAPABILITY_VERSION,
                           "mode": self.config.mode,
                           "tokenCounter": self.token_counter_name},
        }
        started = self.clock()
        try:
            response = self.fetch(request, self.timeout_seconds)
        except TimeoutError:
            self.turn["elapsed_ms"] = round((self.clock() - started) * 1000, 3)
            self.seen_deliveries.add(delivery_id)
            self.turn["state"] = "timeout"
            return "timeout"
        except Exception:
            self.turn["elapsed_ms"] = round((self.clock() - started) * 1000, 3)
            self.seen_deliveries.add(delivery_id)
            self.turn["state"] = "source_unavailable"
            return "source_unavailable"
        elapsed = self.clock() - started
        self.turn["elapsed_ms"] = round(elapsed * 1000, 3)
        self.seen_deliveries.add(delivery_id)
        if elapsed > self.timeout_seconds:
            self.turn["state"] = "timeout"
            return "timeout"
        if not isinstance(response, dict) or response.get("status") not in {
                "candidate", "ready", "empty"}:
            self.turn["state"] = "invalid_response"
            return "invalid_response"
        self.turn["response"] = {
            key: json.loads(json.dumps(response[key])) for key in (
                "status", "wireVersion", "policyVersion", "assemblyPolicyVersion",
                "reasonCodes", "records", "dependencies") if key in response
        }
        if response["status"] == "empty":
            self.turn["state"] = "empty"
            if response.get("notApplicableDeliveryIds"):
                self.observe_state({
                    "notApplicableDeliveryIds": response["notApplicableDeliveryIds"],
                    "statusNotice": response.get("statusNotice"),
                })
            return "empty"
        if response.get("deliveryId") != delivery_id:
            self.turn["state"] = "stale"
            return "stale"
        if response["status"] == "candidate":
            # W1 可先交付只读候选；没有 W4 组装文本时不得把原始结构硬塞进模型。
            self.turn.update(state="candidate", candidate=response)
            return "candidate"
        content = response.get("content")
        if not isinstance(content, str) or not content:
            self.turn["state"] = "invalid_response"
            return "invalid_response"
        message = {"role": "system", "content": content,
                   "_passive": {"deliveryId": delivery_id,
                                "assemblyVersion": response.get("assemblyVersion"),
                                "dependencies": response.get("dependencies", []),
                                "state": "active"}}
        cost = int(self.token_counter(_public_message(message)))
        if cost < 0:
            self.turn["state"] = "invalid_response"
            return "invalid_response"
        if cost > SINGLE_LIMIT:
            self.turn["state"] = "budget_exceeded"
            return "budget_exceeded"
        if self.config.mode == "retained" and self.ordinary_used + cost > ORDINARY_LIMIT:
            self.turn["state"] = "budget_exhausted"
            return "budget_exhausted"
        self.turn.update(message=message, state="ready", cost=cost)
        self.delivery_ledger[delivery_id] = {
            "sessionId": session_id, "turnId": turn_id, "scope": scope or "general",
            "assemblyVersion": response.get("assemblyVersion"),
            "dependencies": json.loads(json.dumps(response.get("dependencies", []))),
            "state": "prepared", "confirmed": False, "visibility": "unknown",
            "cost": cost,
        }
        if self.config.mode == "retained":
            history.append(message)
            self.ordinary_used += cost
        return "ready"

    def messages_for_request(self, history):
        """每次模型请求前取消息；临时层只写请求副本，不改 history。"""
        if not self.turn or self.turn.get("state") != "ready":
            return [_public_message(item) for item in history]
        ledger = self.delivery_ledger.get(self.turn["delivery_id"])
        if ledger is not None:
            ledger.update(state="active", confirmed=True, visibility="visible")
        if self.config.mode == "temporary":
            return _insert_after_latest_user(history, self.turn["message"])
        return [_public_message(item) for item in history]

    def previous_anchors(self):
        """只给服务端来源明确的少量账本锚点；不转发模型自由改写。"""
        recent_ids = [delivery_id for delivery_id, value in self.delivery_ledger.items()
                      if value.get("state") in {
                          "prepared", "active", "delivery_unknown"}][-3:]
        return [{"deliveryId": delivery_id,
                 **{key: self.delivery_ledger[delivery_id].get(key) for key in
                    ("assemblyVersion", "visibility", "turnId")}}
                for delivery_id in recent_ids]

    def context_evidence(self):
        """仅把确认仍可见的原始证据报为覆盖；自然回答复述不算。"""
        return [dependency for value in self.delivery_ledger.values()
                if value.get("state") == "active" and value.get("confirmed")
                and value.get("visibility") == "visible"
                for dependency in value.get("dependencies", [])]

    def observe_state(self, state):
        """接收工具回执或只读复核给出的覆盖／失效状态，不重新检索。"""
        if not isinstance(state, dict):
            return "unchanged"
        covered_ids = set(state.get("coveredDeliveryIds", []))
        invalid_ids = set(state.get("invalidatedDeliveryIds", []))
        not_applicable_ids = set(state.get("notApplicableDeliveryIds", []))
        targets = [delivery_id for delivery_id, value in self.delivery_ledger.items()
                   if value.get("state") in {"prepared", "active", "delivery_unknown"}
                   and delivery_id in covered_ids | invalid_ids | not_applicable_ids]
        if not targets:
            return "unchanged"
        for delivery_id in targets:
            self.delivery_ledger[delivery_id]["state"] = (
                "covered" if delivery_id in covered_ids else
                "invalidated" if delivery_id in invalid_ids else "not_applicable")
            self.delivery_ledger[delivery_id]["visibility"] = "stale"
        current_id = self.turn.get("delivery_id") if self.turn else None
        if current_id in targets:
            self.turn["state"] = self.delivery_ledger[current_id]["state"]
        outcome = ("invalidated" if invalid_ids.intersection(targets) else
                   "not_applicable" if not_applicable_ids.intersection(targets) else "covered")
        if self.config.mode == "temporary":
            return outcome
        if self.can_remove_retained:
            history = self.turn["history"] if self.turn else []
            history[:] = [message for message in history
                           if message.get("_passive", {}).get("deliveryId") not in targets]
            return outcome
        notice = state.get("statusNotice")
        if (invalid_ids | not_applicable_ids).intersection(targets) \
                and isinstance(notice, str) and notice:
            message = {"role": "system", "content": notice,
                       "_passive": {"deliveryIds": targets, "state": "status"}}
            cost = int(self.token_counter(_public_message(message)))
            remaining = STATUS_LIMIT - self.status_used
            if cost > FINAL_STATUS_RESERVE or cost > remaining - FINAL_STATUS_RESERVE:
                retirement = state.get("retirementNotice")
                if isinstance(retirement, str) and retirement:
                    retired_message = {"role": "system", "content": retirement,
                                       "_passive": {"state": "retired"}}
                    retired_cost = int(self.token_counter(_public_message(retired_message)))
                    if retired_cost <= FINAL_STATUS_RESERVE and retired_cost <= remaining:
                        self.turn["history"].append(retired_message)
                        self.status_used += retired_cost
                self.paused_reason = "状态预算不足，自动资料已退役或有效性未确认"
                for value in self.delivery_ledger.values():
                    if value.get("state") in {"prepared", "active", "delivery_unknown",
                                              "invalidated", "covered"}:
                        value["state"] = "retired"
                return "retired"
            self.turn["history"].append(message)
            self.status_used += cost
        return outcome

    def observe_tool_result(self, result):
        structured = result.get("structuredContent", {}) if isinstance(result, dict) else {}
        return self.observe_state(structured.get("passiveRecall", {}))

    def finish_turn(self):
        """回答完成、失败或取消都丢掉临时层；历史保留块留在 history。"""
        if self.turn and self.config.mode == "temporary":
            ledger = self.delivery_ledger.get(self.turn["delivery_id"])
            if ledger is not None and ledger.get("state") == "active":
                ledger["visibility"] = "clipped"
            self.turn["message"] = None
        self.turn = None

    cancel_turn = finish_turn

    def usage(self):
        return {"ordinary": self.ordinary_used, "status": self.status_used,
                "total": self.ordinary_used + self.status_used,
                "limits": {"ordinary": ORDINARY_LIMIT, "status": STATUS_LIMIT,
                           "total": TOTAL_LIMIT, "finalStatusReserve": FINAL_STATUS_RESERVE}}

    def observation(self, *, host="reference-host", host_version=CAPABILITY_VERSION):
        """返回不含输入／证据原文的 W5 观测事件；是否落盘由宿主显式决定。"""
        turn = self.turn or {}
        delivery_id = turn.get("delivery_id")
        ledger = self.delivery_ledger.get(delivery_id, {})
        state = turn.get("state", "idle")
        candidate = turn.get("candidate") or {}
        dependencies = ledger.get("dependencies") or candidate.get("dependencies") or []
        source_ids = list(dict.fromkeys(
            item.get("recordId") for item in dependencies
            if isinstance(item, dict) and item.get("recordId")))
        response = turn.get("response") or candidate
        reason_codes = response.get("reasonCodes", []) if isinstance(response, dict) else []
        queried = state not in {"idle", "skipped", "disabled", "paused", "duplicate"}
        retrieval_skips = {"low_information", "control_command", "quoted_or_code_only",
                           "scope_unknown", "source_unresolved"}
        retrieval_performed = queried and not retrieval_skips.intersection(reason_codes)
        return {
            "event": "turn",
            "host": host,
            "hostVersion": host_version,
            "mode": self.config.mode if self.config.enabled else "off",
            "scope": ledger.get("scope", turn.get("scope", "general")),
            "sessionHash": opaque_id(ledger.get("sessionId", turn.get("session_id"))),
            "deliveryHash": opaque_id(delivery_id),
            "state": state,
            "prefilter": queried,
            "retrieved": retrieval_performed,
            "injected": bool(ledger.get("confirmed")),
            "reasonCodes": reason_codes,
            "sourceIds": source_ids,
            "dependencyCoverage": "complete" if source_ids else "none",
            "visibility": ledger.get("visibility", "unknown"),
            "elapsedMs": turn.get("elapsed_ms"),
            "tokenCounter": self.token_counter_name,
            "incrementalTokens": ledger.get("cost"),
            "ordinaryUsed": self.ordinary_used,
            "statusUsed": self.status_used,
            "policyVersion": response.get("policyVersion") if isinstance(response, dict) else None,
            "assemblyPolicyVersion": response.get("assemblyPolicyVersion")
            if isinstance(response, dict) else None,
            "wireVersion": response.get("wireVersion") if isinstance(response, dict) else None,
            "failurePoint": state if state in {
                "timeout", "source_unavailable", "invalid_response", "stale",
                "budget_exceeded", "budget_exhausted"} else None,
        }


def _selftest():
    """W0 离线夹具：只测宿主契约，不把假候选计成检索或自然效果。"""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "passive-recall.json"
        assert load_config(path) == PassiveRecallConfig(), "缺配置必须默认关闭"
        path.write_text('{"passive_recall":{"enabled":true}}', encoding="utf-8")
        try:
            load_config(path)
            raise AssertionError("只开 enabled、未选模式必须拒绝")
        except PassiveRecallConfigError:
            pass
        path.write_text('{"passive_recall":{"enabled":true,"mode":"temporary"}}',
                        encoding="utf-8")
        assert load_config(path) == PassiveRecallConfig(True, "temporary")

    calls = []

    def ready(request, timeout_seconds):
        calls.append(request)
        return {"status": "ready", "deliveryId": request["turn"]["deliveryId"],
                "assemblyVersion": "a1", "dependencies": [{"recordId": "r1"}],
                "records": [{"recordId": "r1", "ranges": [[0, 8]],
                             "source": "synthetic.md"}],
                "content": "〔W0 合成原料〕"}

    # L1／T1：用户消息之后才调；临时层进入请求副本，同轮工具回路复用，绝不进 history。
    history = [{"role": "user", "content": "纸箱又挡门了"}]
    temporary = PassiveRecallAdapter(PassiveRecallConfig(True, "temporary"), ready)
    assert temporary.begin_turn(history, session_id="s", turn_id="t1", delivery_id="d1",
                                user_input="纸箱又挡门了") == "ready"
    first = temporary.messages_for_request(history)
    second = temporary.messages_for_request(history + [
        {"role": "assistant", "content": None, "tool_calls": []},
        {"role": "tool", "tool_call_id": "c1", "content": "真实工具结果"}])
    assert len(calls) == 1 and any("W0 合成原料" in (m.get("content") or "") for m in first)
    assert any("W0 合成原料" in (m.get("content") or "") for m in second), \
        "同轮后续请求应复用，不得重新检索"
    assert "W0 合成原料" not in json.dumps(history, ensure_ascii=False), \
        "temporary 不能原地修改正常 history"
    for derived in (list(history), json.loads(json.dumps(history, ensure_ascii=False))):
        assert "W0 合成原料" not in json.dumps(derived, ensure_ascii=False), \
            "保存／恢复／分叉输入不得带临时原料"

    # S1～S3／T2：部分或无关工具不清空；完整覆盖、撤回在下一请求移除且不重检索。
    assert temporary.observe_state({"coveredDeliveryIds": []}) == "unchanged"
    assert temporary.observe_state({"coveredDeliveryIds": ["d1"]}) == "covered"
    assert "W0 合成原料" not in json.dumps(
        temporary.messages_for_request(history), ensure_ascii=False)
    assert len(calls) == 1
    temporary.finish_turn()
    assert "W0 合成原料" not in json.dumps(
        temporary.messages_for_request(history), ensure_ascii=False)

    # W1 可先返回结构化候选；W4 未组装时只留在宿主状态，不得直接注入模型。
    candidate_only = PassiveRecallAdapter(
        PassiveRecallConfig(True, "temporary"),
        lambda request, timeout: {
            "status": "candidate", "deliveryId": request["turn"]["deliveryId"],
            "assemblyVersion": "a0", "dependencies": [{"recordId": "r0"}],
            "records": [{"recordId": "r0", "ranges": [[0, 3]], "source": "s.md"}]})
    hc = [{"role": "user", "content": "候选"}]
    assert candidate_only.begin_turn(hc, session_id="s", turn_id="tc", delivery_id="dc",
                                     user_input="候选") == "candidate"
    assert candidate_only.messages_for_request(hc) == hc

    # L2：同 delivery 幂等；超时和迟到结果丢弃，不能串进下一轮。
    ticks = iter([0.0, 1.0])
    slow = PassiveRecallAdapter(PassiveRecallConfig(True, "temporary"), ready,
                                timeout_seconds=0.2, clock=lambda: next(ticks))
    h2 = [{"role": "user", "content": "迟到夹具"}]
    assert slow.begin_turn(h2, session_id="s", turn_id="t2", delivery_id="d2",
                           user_input="迟到夹具") == "timeout"
    assert "W0 合成原料" not in json.dumps(slow.messages_for_request(h2), ensure_ascii=False)
    assert slow.begin_turn(h2, session_id="s", turn_id="t2", delivery_id="d2",
                           user_input="迟到夹具") == "duplicate"
    timed_out = PassiveRecallAdapter(
        PassiveRecallConfig(True, "temporary"),
        lambda request, timeout: (_ for _ in ()).throw(TimeoutError("夹具超时")))
    h2b = [{"role": "user", "content": "传输超时"}]
    assert timed_out.begin_turn(h2b, session_id="s", turn_id="t2b", delivery_id="d2b",
                                user_input="传输超时") == "timeout"

    # H1／H2：retained 只追加一次；失效时旧块可留，但下一请求前有明确状态通知。
    retained_history = [{"role": "user", "content": "保留模式"}]
    retained = PassiveRecallAdapter(PassiveRecallConfig(True, "retained"), ready,
                                    can_remove_retained=False)
    assert retained.begin_turn(retained_history, session_id="s", turn_id="t3",
                               delivery_id="d3", user_input="保留模式") == "ready"
    assert sum("W0 合成原料" in (m.get("content") or "")
               for m in retained_history) == 1
    state = {"invalidatedDeliveryIds": ["d3"], "statusNotice": "〔状态〕旧资料已失效。",
             "retirementNotice": "〔状态〕全部自动资料停止作为有效依据。"}
    assert retained.observe_state(state) == "invalidated"
    assert "旧资料已失效" in json.dumps(retained_history, ensure_ascii=False)
    assert retained.usage()["ordinary"] > 0 and retained.usage()["status"] > 0
    removable_history = [{"role": "user", "content": "可移除模式"}]
    removable = PassiveRecallAdapter(PassiveRecallConfig(True, "retained"), ready)
    assert removable.begin_turn(removable_history, session_id="s", turn_id="t3b",
                                delivery_id="d3b", user_input="可移除模式") == "ready"
    assert removable.observe_state({"invalidatedDeliveryIds": ["d3b"]}) == "invalidated"
    assert "W0 合成原料" not in json.dumps(removable_history, ensure_ascii=False), \
        "能控制 history 的宿主应直接移除失效资料"

    # L3／H3／H4：关闭不调入口；单条上限按实际消息封装上界计算；额度不互借。
    disabled = PassiveRecallAdapter(PassiveRecallConfig(),
                                    lambda request, timeout: (_ for _ in ()).throw(
                                        AssertionError("关闭状态不应检索")))
    assert disabled.begin_turn([{"role": "user", "content": "关闭"}], session_id="s",
                               turn_id="t4", delivery_id="d4", user_input="关闭") == "disabled"

    def oversized(request, timeout_seconds):
        return {"status": "ready", "deliveryId": request["turn"]["deliveryId"],
                "content": "长" * 301}

    budget = PassiveRecallAdapter(PassiveRecallConfig(True, "retained"), oversized)
    hb = [{"role": "user", "content": "预算"}]
    assert budget.begin_turn(hb, session_id="s", turn_id="t5", delivery_id="d5",
                             user_input="预算") == "budget_exceeded"
    assert len(hb) == 1, "超预算资料不能先追加再回滚"
    assert retained.usage()["total"] <= TOTAL_LIMIT
    print("selftest 通过：W0 L1-L3／T1-T2／H1-H4 宿主契约离线夹具")


if __name__ == "__main__":
    _selftest()
