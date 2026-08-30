#!/usr/bin/env python3
"""未解决事项 sidecar：人工／AI 显式维护，不做内容匹配或自动判断。"""

import argparse
import os
import re
import secrets
import tempfile
from dataclasses import dataclass
from pathlib import Path


FILENAME = "未解决.md"
HEADER = "# 未解决\n\n<!-- 一行一条；关闭时删除整行，不写已完成记录。 -->\n"
_ROW = re.compile(
    r"^- (U-[0-9a-f]{16})｜([^｜\r\n]+)｜初始：([^｜\r\n]+)｜更新：([^｜\r\n]+)$")


class UnresolvedRequestError(ValueError):
    """调用方给的 action／ID／摘要不合法。"""


class UnresolvedStoreError(ValueError):
    """sidecar 自身格式、编码、权限或写盘失败。"""


@dataclass(frozen=True)
class UnresolvedItem:
    id: str
    summary: str
    initial: str
    updated: str = "—"


def _clean_field(value, label):
    if not isinstance(value, str) or not value.strip():
        raise UnresolvedRequestError(f"{label} 必须是非空字符串")
    value = value.strip()
    if "｜" in value or "\n" in value or "\r" in value:
        raise UnresolvedRequestError(f"{label} 必须是单行，且不能含分隔符｜")
    return value


def parse_text(text):
    """返回 (合法项, 错误行号)；读路径的异常由 UnresolvedStore 归类。"""
    items, bad_lines, seen = [], [], set()
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line == "# 未解决" or line.startswith("<!--"):
            continue
        match = _ROW.fullmatch(line)
        if not match:
            bad_lines.append(number)
            continue
        item = UnresolvedItem(*match.groups())
        if item.id in seen:
            bad_lines.append(number)
            continue
        seen.add(item.id)
        items.append(item)
    return items, bad_lines


def render(items):
    lines = [HEADER.rstrip(), ""]
    lines.extend(
        f"- {item.id}｜{item.summary}｜初始：{item.initial}｜更新：{item.updated}"
        for item in items)
    return "\n".join(lines).rstrip() + "\n"


def validate_ops(ops):
    if not isinstance(ops, list) or not ops:
        raise UnresolvedRequestError(
            "unresolvedOps 必须是非空数组；无变化也要明确传 [{\"action\":\"none\"}]")
    if any(not isinstance(op, dict) for op in ops):
        raise UnresolvedRequestError("unresolvedOps 每一项都必须是对象")
    actions = [op.get("action") for op in ops]
    allowed = {"open", "update", "close", "none"}
    if any(action not in allowed for action in actions):
        raise UnresolvedRequestError("action 只能是 open／update／close／none")
    if "none" in actions and (len(actions) != 1 or set(ops[0]) != {"action"}):
        raise UnresolvedRequestError("none 必须单独出现，且不能带其它字段")
    if actions.count("open") > 1:
        raise UnresolvedRequestError("一次调用最多 open 一条；多件事请分别写回")
    normalized = []
    for op in ops:
        action = op["action"]
        if action == "none":
            normalized.append({"action": "none"})
        elif action == "open":
            normalized.append({"action": action,
                               "summary": _clean_field(op.get("summary"), "summary")})
        elif action == "update":
            normalized.append({"action": action,
                               "id": _clean_field(op.get("id"), "id"),
                               "summary": _clean_field(op.get("summary"), "summary")})
        else:
            normalized.append({"action": action,
                               "id": _clean_field(op.get("id"), "id")})
    return normalized


class UnresolvedStore:
    def __init__(self, path):
        self.path = Path(path) if path is not None else None

    def read(self, allow_partial=False):
        if self.path is None or not self.path.exists():
            return ([], []) if allow_partial else []
        try:
            text = self.path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise UnresolvedStoreError(
                f"{self.path} 无法读取（{type(exc).__name__}）") from None
        items, bad_lines = parse_text(text)
        if bad_lines and not allow_partial:
            raise UnresolvedStoreError(
                f"{self.path} 格式损坏，错误行：{'、'.join(map(str, bad_lines))}")
        return (items, bad_lines) if allow_partial else items

    def apply(self, ops, source="manual"):
        normalized = validate_ops(ops)
        if normalized == [{"action": "none"}]:
            # none 也真读一遍；清单坏了不能冒充已经复核成功。
            self.read()
            return "reviewed", []
        if self.path is None:
            raise UnresolvedStoreError("服务器没有配置未解决清单路径")
        source = _clean_field(source, "source")
        items = self.read()
        by_id = {item.id: item for item in items}
        changed = []
        for op in normalized:
            action = op["action"]
            if action == "open":
                while True:
                    item_id = "U-" + secrets.token_hex(8)
                    if item_id not in by_id:
                        break
                item = UnresolvedItem(item_id, op["summary"], source)
                items.append(item)
                by_id[item_id] = item
                changed.append(item_id)
                continue
            item_id = op["id"]
            if item_id not in by_id:
                raise UnresolvedRequestError(f"没有找到未解决事项 {item_id}")
            if action == "close":
                items = [item for item in items if item.id != item_id]
                del by_id[item_id]
            else:
                old = by_id[item_id]
                new = UnresolvedItem(old.id, op["summary"], old.initial, source)
                items = [new if item.id == item_id else item for item in items]
                by_id[item_id] = new
            changed.append(item_id)
        self._write(render(items))
        return "updated", changed

    def _write(self, content):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp",
                                        dir=self.path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(name, self.path)
            finally:
                Path(name).unlink(missing_ok=True)
        except OSError as exc:
            raise UnresolvedStoreError(
                f"{self.path} 写入失败（{type(exc).__name__}）") from None


def format_block(items, bad_lines=()):
    if not items and not bad_lines:
        return None
    lines = ["【当前未解决】以下事项仍被明确标记为未结束；这是当前集合，优先于后面的历史快照："]
    for item in items:
        source = item.updated if item.updated != "—" else item.initial
        lines.append(f"- [{item.id}｜{source}] {item.summary}")
    if bad_lines:
        lines.append("- ⚠ 未解决清单另有格式错误行：" + "、".join(map(str, bad_lines))
                     + "；这些行没有被静默当成有效事项。")
    return "\n".join(lines)


def source_record_ids(items):
    ids = set()
    for item in items:
        for source in (item.initial, item.updated):
            match = re.search(r"#record=([0-9a-f]{16})(?:$|[^0-9a-f])", source)
            if match:
                ids.add(match.group(1))
    return ids


def _selftest():
    import tempfile as _tempfile
    with _tempfile.TemporaryDirectory() as td:
        store = UnresolvedStore(Path(td) / FILENAME)
        assert store.read() == []
        status, changed = store.apply([{"action": "open", "summary": "住宿还没订"}],
                                      "timeline/window_01.md#record=0123456789abcdef")
        assert status == "updated" and len(changed) == 1
        item_id = changed[0]
        store.apply([{"action": "update", "id": item_id, "summary": "只差民宿确认"}],
                    "thread/window_02@2026-08-31T21:00:00+08:00")
        item = store.read()[0]
        assert item.summary == "只差民宿确认" and "window_01" in item.initial
        assert "window_02" in item.updated
        assert item_id in format_block(store.read())
        store.apply([{"action": "close", "id": item_id}], "manual")
        assert store.read() == [] and item_id not in store.path.read_text(encoding="utf-8")
        store.path.write_text("# 未解决\n坏行\n", encoding="utf-8")
        assert store.read(allow_partial=True)[1] == [2]
        try:
            store.apply([{"action": "none"}], "manual")
            assert False, "坏文件不能冒充 reviewed"
        except UnresolvedStoreError:
            pass
        for bad in ([], [{"action": "none", "id": "x"}],
                    [{"action": "open", "summary": "两｜段"}]):
            try:
                validate_ops(bad)
                assert False, f"坏操作该被拒：{bad}"
            except UnresolvedRequestError:
                pass
    print("selftest ok（缺失/增改删/来源/坏行/参数分类）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        ap.print_help()
