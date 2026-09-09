#!/usr/bin/env python3
"""自动浮现 W5：不保存用户原文的观测记录。

记录只接受本文件列出的结构化字段。用户输入、模型回答、完整历史证据与凭证即使由
调用方误传也不会落盘；逐例效果原文须写入所有者明确指定的本地评测结果，不进工作仓。
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


SCHEMA_VERSION = "passive-observation-w5-v1"
_FIELDS = {
    "event", "host", "hostVersion", "mode", "scope", "sessionHash",
    "deliveryHash", "state", "prefilter", "retrieved", "injected",
    "reasonCodes", "sourceIds", "dependencyCoverage", "visibility",
    "elapsedMs", "tokenCounter", "incrementalTokens", "ordinaryUsed",
    "statusUsed", "policyVersion", "assemblyPolicyVersion", "wireVersion",
    "failurePoint", "reviewRequired", "shadowStrictState", "shadowWideState",
    "model", "base", "requestInputTokens", "cachedInputTokens",
    "historyAddedMessages", "historyAddedChars",
}
_FORBIDDEN = {"userInput", "input", "prompt", "content", "answer", "history",
              "messages", "apiKey", "authorization"}


def opaque_id(value):
    """把会话／交付标识变成不可逆的短标识，避免日志串回原始会话名。"""
    if value is None:
        return None
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


def safe_base(value):
    """只保留 API base 的 scheme／host／path，丢掉用户信息、查询串与片段。"""
    if value is None:
        return None
    parts = urlsplit(str(value))
    host = parts.hostname or ""
    if parts.port:
        host += f":{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def sanitize_observation(event):
    """按白名单生成可落盘事件；未知字段与原文类字段一律拒绝。"""
    if not isinstance(event, dict):
        raise TypeError("观测事件必须是对象")
    forbidden = _FORBIDDEN.intersection(event)
    if forbidden:
        raise ValueError(f"观测事件不得包含原文类字段：{'／'.join(sorted(forbidden))}")
    unknown = set(event) - _FIELDS
    if unknown:
        raise ValueError(f"观测事件包含未登记字段：{'／'.join(sorted(unknown))}")
    clean = {key: value for key, value in event.items() if value is not None}
    if "base" in clean:
        clean["base"] = safe_base(clean["base"])
    clean["schemaVersion"] = SCHEMA_VERSION
    clean["observedAt"] = datetime.now(timezone.utc).isoformat()
    return clean


class JsonlObservationRecorder:
    """显式给路径才写入；每次一行，便于中断后保留已完成样本。"""

    def __init__(self, path):
        self.path = Path(path)

    def __call__(self, event):
        clean = sanitize_observation(event)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(clean, ensure_ascii=False, sort_keys=True) + "\n")
        return clean


def _selftest():
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        recorder = JsonlObservationRecorder(Path(td) / "observations.jsonl")
        row = recorder({"event": "turn", "host": "fixture", "mode": "temporary",
                        "sessionHash": opaque_id("真实会话名"), "state": "empty",
                        "reasonCodes": ["low_information"],
                        "base": "https://user:secret@example.invalid/v1?key=secret"})
        assert row["schemaVersion"] == SCHEMA_VERSION and "observedAt" in row
        saved = json.loads(recorder.path.read_text(encoding="utf-8"))
        assert "真实会话名" not in recorder.path.read_text(encoding="utf-8")
        assert "secret" not in recorder.path.read_text(encoding="utf-8")
        assert saved["reasonCodes"] == ["low_information"]
        try:
            recorder({"event": "turn", "content": "不应落盘的原文"})
            raise AssertionError("原文字段必须被拒绝")
        except ValueError:
            pass
    print("W5 observation selftest 通过：字段白名单、会话散列、默认不收原文")


if __name__ == "__main__":
    _selftest()
