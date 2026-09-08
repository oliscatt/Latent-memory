"""自动浮现 W2：可选证据范围与触发元数据的校验、版本和原子写计划。

本文件只管理辅助 sidecar。正文和撤回账本始终是权威来源；sidecar 缺失、损坏或
来源失配时不得改写正文，也不得把失效记录重新变成可用记录。
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path


FILENAME = ".passive.json"
KINDS = {"meme", "event", "fact", "unknown"}
RANGE_TYPES = {"formation", "support", "revision"}
_PUNCTUATION = str.maketrans({
    "“": '"', "”": '"', "„": '"', "‟": '"', "‘": "'", "’": "'",
    "：": ":", "；": ";", "，": ",", "。": ".", "！": "!", "？": "?",
})


def _digest(value):
    wire = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(wire).hexdigest()


def source_signature(text):
    """供写回层与读路径复算正文来源签名。"""
    return _digest(text)


def _canonical(value):
    chars, positions = [], []
    for original_pos, ch in enumerate(value):
        for normalized in unicodedata.normalize("NFKC", ch).translate(_PUNCTUATION):
            if normalized.isspace():
                continue
            chars.append(normalized)
            positions.append(original_pos)
    return "".join(chars), positions


def locate_quote(text, quote, label="证据", *, require_unique=True):
    """把模型给的 quote 映射回正文 Unicode 码点范围，不接受同义改写或歧义。"""
    if not isinstance(quote, str) or not quote.strip():
        raise ValueError(f"{label} quote 不能为空")
    needle, _ = _canonical(quote.strip())
    if len(needle) < 2:
        raise ValueError(f"{label} quote 太短；请摘至少两个内容字符的连续原文")
    haystack, positions = _canonical(text)
    starts, cursor = [], 0
    while True:
        pos = haystack.find(needle, cursor)
        if pos < 0:
            break
        starts.append(pos)
        cursor = pos + 1
    if not starts:
        raise ValueError(f"{label}无法映射回原文：{quote}")
    if require_unique and len(starts) != 1:
        raise ValueError(f"{label}在原文中出现 {len(starts)} 次，必须补足前后文使范围唯一")
    pos = starts[0]
    start, end = positions[pos], positions[pos + len(needle) - 1] + 1
    excerpt = text[start:end]
    return {"start": start, "end": end, "quote": excerpt,
            "signature": _digest(excerpt)}


def _string_list(value, name, *, allow_empty=True):
    if value is None:
        return []
    if not isinstance(value, list) or (not allow_empty and not value):
        qualifier = "非空" if not allow_empty else ""
        raise ValueError(f"passive.{name} 必须是{qualifier}字符串数组")
    result = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"passive.{name} 不能含空值")
        clean = item.strip()
        if clean not in result:
            result.append(clean)
    return result


def validate_passive(value, *, record_id, record_text, record_lookup,
                     allowed_scopes=("general",), prior=None, known_records=()):
    """校验一条 passive 对象并生成可复算 revision；不读写磁盘。"""
    if not isinstance(value, dict):
        raise ValueError("passive 必须是对象")
    unknown = set(value) - {"trigger_terms", "short_trigger_terms", "kind", "scope",
                            "episode_id", "context_refs", "source_ranges"}
    if unknown:
        raise ValueError("passive 含未知字段：" + "、".join(sorted(unknown)))
    trigger_terms = _string_list(value.get("trigger_terms"), "trigger_terms")
    short_terms = _string_list(value.get("short_trigger_terms"), "short_trigger_terms")
    if any(term not in trigger_terms for term in short_terms):
        raise ValueError("passive.short_trigger_terms 必须是 trigger_terms 的子集")
    trigger_ranges = []
    for term in trigger_terms:
        trigger_ranges.append({"term": term, **locate_quote(
            record_text, term, "触发短语", require_unique=False)})

    kind = value.get("kind", "unknown")
    if kind not in KINDS:
        raise ValueError("passive.kind 只能是 meme／event／fact／unknown")
    scope = value.get("scope", "general")
    if not isinstance(scope, str) or scope not in set(allowed_scopes):
        raise ValueError("passive.scope 不在宿主允许范围内，不能由模型创建任意跨项目范围")

    source_ranges = []
    raw_ranges = value.get("source_ranges") or []
    if not isinstance(raw_ranges, list):
        raise ValueError("passive.source_ranges 必须是数组")
    for number, item in enumerate(raw_ranges, 1):
        if not isinstance(item, dict) or set(item) - {"type", "quote"}:
            raise ValueError(f"passive.source_ranges 第 {number} 条字段不合法")
        range_type = item.get("type")
        if range_type not in RANGE_TYPES:
            raise ValueError("passive.source_ranges.type 只能是 formation／support／revision")
        source_ranges.append({"type": range_type,
                              **locate_quote(record_text, item.get("quote"), "来源范围")})
    if short_terms and not any(item["type"] == "formation" for item in source_ranges):
        raise ValueError("短句许可必须同时提供可回指原文的 formation 来源范围")

    context_refs = []
    raw_refs = value.get("context_refs") or []
    if not isinstance(raw_refs, list):
        raise ValueError("passive.context_refs 必须是数组")
    for number, item in enumerate(raw_refs, 1):
        if not isinstance(item, dict) or set(item) - {"recordId", "source_ranges"}:
            raise ValueError(f"passive.context_refs 第 {number} 条字段不合法")
        ref_id = item.get("recordId")
        if not isinstance(ref_id, str) or not re.fullmatch(r"[0-9a-f]{16}", ref_id):
            raise ValueError(f"passive.context_refs 第 {number} 条 recordId 不合法")
        if ref_id == record_id:
            raise ValueError("passive.context_refs 不能引用自身")
        ref = record_lookup(ref_id)
        if ref is None or ref.get("state") != "active":
            raise ValueError(f"必要引用 recordId={ref_id} 不存在或已失效")
        if ref.get("scope", "general") != scope:
            raise ValueError(f"必要引用 recordId={ref_id} 不在同一 scope")
        ranges = []
        for raw_range in item.get("source_ranges") or []:
            if not isinstance(raw_range, dict) or set(raw_range) - {"type", "quote"}:
                raise ValueError(f"必要引用 recordId={ref_id} 的来源范围字段不合法")
            range_type = raw_range.get("type")
            if range_type not in RANGE_TYPES:
                raise ValueError("必要引用来源范围 type 不合法")
            ranges.append({"type": range_type,
                           **locate_quote(ref["text"], raw_range.get("quote"), "必要引用范围")})
        if not ranges:
            raise ValueError(f"必要引用 recordId={ref_id} 必须提供可核验 source_ranges")
        context_refs.append({"recordId": ref_id, "sourceSignature": ref["sourceSignature"],
                             "source_ranges": ranges})

    requested_episode = value.get("episode_id")
    if requested_episode is not None:
        if not isinstance(requested_episode, str) or not re.fullmatch(r"ep_[0-9a-f]{16}", requested_episode):
            raise ValueError("passive.episode_id 必须是服务端返回的 ep_ 加 16 位标识")
        known = [item for item in known_records if isinstance(item, dict)
                 and item.get("episode_id") == requested_episode]
        if not known or any(item.get("scope") != scope for item in known):
            raise ValueError("episode_id 只能续用同一范围已有的服务端标识")
        episode_id = requested_episode
    else:
        episode_id = (prior or {}).get("episode_id") or f"ep_{_digest([record_id, scope])[:16]}"

    entry = {"recordId": record_id, "sourceSignature": source_signature(record_text),
             "trigger_terms": trigger_terms, "short_trigger_terms": short_terms,
             "trigger_ranges": trigger_ranges, "kind": kind, "scope": scope,
             "episode_id": episode_id, "context_refs": context_refs,
             "source_ranges": source_ranges}
    entry["revision"] = _digest(entry)
    return entry


class PassiveMetadataStore:
    """按 recordId 保存完整版本；一次替换只会发布 ready 的整条记录。"""

    def __init__(self, path, allowed_scopes=("general",)):
        self.path = Path(path) if path is not None else None
        self.allowed_scopes = tuple(allowed_scopes)

    def read(self):
        if self.path is None or not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"被动元数据账本无法读取（{type(exc).__name__}）") from None
        if not isinstance(data, dict) or data.get("version") != 1 \
                or not isinstance(data.get("records"), dict):
            raise ValueError("被动元数据账本结构无效")
        return data["records"]

    def plan_put(self, entry):
        records = self.read()
        records[entry["recordId"]] = entry
        payload = {"version": 1, "records": records}
        return {"path": self.path,
                "content": json.dumps(payload, ensure_ascii=False, indent=2) + "\n"}

    def plan_remove(self, record_id):
        records = self.read()
        if record_id not in records:
            return None
        del records[record_id]
        return {"path": self.path,
                "content": json.dumps({"version": 1, "records": records},
                                      ensure_ascii=False, indent=2) + "\n"}


def _selftest():
    text = "用户：门口纸箱又堵住了。\n助手：叫它纸箱飞船吧。\n用户：呵呵，纸箱飞船起飞。"
    rid = "0123456789abcdef"
    ref_text = "用户：认真抱怨时不要接纸箱飞船这个梗。"
    ref_id = "fedcba9876543210"

    def lookup(value):
        if value == ref_id:
            return {"state": "active", "text": ref_text,
                    "sourceSignature": _digest(ref_text)}
        return None

    entry = validate_passive({
        "trigger_terms": ["纸箱飞船", "呵呵"], "short_trigger_terms": ["呵呵"],
        "kind": "meme", "source_ranges": [{"type": "formation", "quote": "叫它纸箱飞船吧"}],
        "context_refs": [{"recordId": ref_id, "source_ranges": [
            {"type": "revision", "quote": "认真抱怨时不要接纸箱飞船这个梗"}]}]},
        record_id=rid, record_text=text, record_lookup=lookup)
    repeated = validate_passive({
        "trigger_terms": ["纸箱飞船", "呵呵"], "short_trigger_terms": ["呵呵"],
        "kind": "meme", "source_ranges": [{"type": "formation", "quote": "叫它纸箱飞船吧"}],
        "context_refs": [{"recordId": ref_id, "source_ranges": [
            {"type": "revision", "quote": "认真抱怨时不要接纸箱飞船这个梗"}]}]},
        record_id=rid, record_text=text, record_lookup=lookup, prior=entry)
    assert entry["revision"] == repeated["revision"], "重复元数据不得增加 revision"
    assert entry["source_ranges"][0]["quote"] == "叫它纸箱飞船吧"
    continued_text = "用户：后来又说纸箱飞船已经收进储物间。"
    continued = validate_passive({
        "trigger_terms": ["纸箱飞船"], "kind": "event",
        "episode_id": entry["episode_id"],
        "source_ranges": [{"type": "support", "quote": "纸箱飞船已经收进储物间"}]},
        record_id="1111111111111111", record_text=continued_text,
        record_lookup=lookup, known_records=[entry])
    assert continued["episode_id"] == entry["episode_id"], \
        "同 scope 的后续记录应能显式续用服务端 episode_id"
    try:
        validate_passive({
            "kind": "event", "context_refs": [{"recordId": ref_id, "source_ranges": [
                {"type": "revision", "quote": "认真抱怨时不要接纸箱飞船这个梗"}]}]},
            record_id=rid, record_text=text,
            record_lookup=lambda value: ({"state": "retracted", "text": ref_text,
                                          "sourceSignature": _digest(ref_text)}
                                         if value == ref_id else None))
    except ValueError as exc:
        assert "失效" in str(exc)
    else:
        raise AssertionError("必要背景撤回后旧引用必须失效")
    try:
        validate_passive({"trigger_terms": ["呵呵"], "short_trigger_terms": ["呵呵"]},
                         record_id=rid, record_text=text, record_lookup=lookup)
    except ValueError as exc:
        assert "formation" in str(exc)
    else:
        raise AssertionError("首次普通出现不能自动形成短句许可")
    print("selftest 通过：W2 被动元数据范围、许可、引用与稳定版本")


if __name__ == "__main__":
    _selftest()
