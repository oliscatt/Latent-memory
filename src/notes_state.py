#!/usr/bin/env python3
"""便条 sidecar：对方在页面上写给 TA 的话（“这条记错了”“这件事已经变了”）。

记忆只有 TA 自己能改：页面只能往这里添一张便条，改不改正文由 TA 在下次开场看到后自己决定——
说得对，TA 用 latent_correct／latent_supersede 改，再记一笔 changed；不认同，记 declined，
写一两句为什么回给对方。

存法：语料目录下 `便条.jsonl`，只追加不改写。一行一个事件：
  {"type": "note",  "id": "N-…", "at": ISO, "text": …, "quote"?: …, "recordId"?: …, "kind"?: "wrong"|"changed"}
  {"type": "reply", "id": "N-…", "at": ISO, "decision": "changed"|"declined", "reply"?: …}
状态由事件推出来：没回 → wait；changed → done；declined → said。
"""

import argparse
import json
import os
import re
import secrets
import threading
from pathlib import Path

FILENAME = "便条.jsonl"
MAX_TEXT = 500          # 一张便条写不下更多；超了 400，不截断（截断会改掉对方的原话）
MAX_QUOTE = 300
MAX_WAITING = 50        # 等 TA 看的上限：再多就是有人在刷，不是在说话
_ID = re.compile(r"N-[0-9a-f]{16}")
_RECORD = re.compile(r"[0-9a-f]{16}")


class NoteRequestError(ValueError):
    """调用方给的内容不合法（空、太长、ID 不对、已经回过）。"""


class NoteStoreError(ValueError):
    """便条文件自身读不了、写不进。"""


def _text(value, label, limit, required=True):
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        if required:
            raise NoteRequestError(f"{label} 必须是非空字符串")
        return None
    value = value.strip()
    if len(value) > limit:
        raise NoteRequestError(f"{label} 最多 {limit} 字，现在 {len(value)} 字")
    return value


class NoteStore:
    def __init__(self, path):
        self.path = Path(path) if path is not None else None
        self._lock = threading.Lock()   # 页面和工具调用可能同时写：读状态与追加要成一步

    def _events(self):
        if self.path is None or not self.path.exists():
            return [], []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            raise NoteStoreError(f"{self.path} 无法读取（{type(exc).__name__}）") from None
        events, bad = [], []
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
                if ev.get("type") not in ("note", "reply") or not _ID.fullmatch(str(ev.get("id"))):
                    raise ValueError
                events.append(ev)
            except ValueError:
                bad.append(number)      # 坏行跳过但要报出来，不冒充“没有便条”
        return events, bad

    def read(self, allow_partial=True):
        """→ (便条列表，按写的先后；坏行号)。每张：id/at/text/quote/recordId/kind/state/reply/repliedAt。"""
        events, bad = self._events()
        if bad and not allow_partial:
            raise NoteStoreError(f"{self.path} 有坏行：{'、'.join(map(str, bad))}")
        notes = {}
        for ev in events:
            if ev["type"] == "note" and ev["id"] not in notes:
                notes[ev["id"]] = {k: ev.get(k) for k in ("id", "at", "text", "quote", "recordId", "kind")} | {"state": "wait"}
            elif ev["type"] == "reply" and ev["id"] in notes and notes[ev["id"]]["state"] == "wait":
                n = notes[ev["id"]]
                n["state"] = "done" if ev.get("decision") == "changed" else "said"
                n["reply"], n["repliedAt"] = ev.get("reply"), ev.get("at")
        return list(notes.values()), bad

    def _append(self, event):
        if self.path is None:
            raise NoteStoreError("服务器没有配置语料目录，便条没处放")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except OSError as exc:
            raise NoteStoreError(f"{self.path} 写入失败（{type(exc).__name__}）") from None

    def add(self, text, at, quote=None, record_id=None, kind=None):
        """页面写一张便条。→ 新便条（state=wait）。"""
        text = _text(text, "便条", MAX_TEXT)
        quote = _text(quote, "引用的那段", MAX_QUOTE, required=False)
        if record_id is not None and not (isinstance(record_id, str) and _RECORD.fullmatch(record_id)):
            raise NoteRequestError("recordId 必须是 16 位小写十六进制")
        if kind is not None and kind not in ("wrong", "changed"):
            raise NoteRequestError("kind 只能是 wrong（记错了）或 changed（已经变了）")
        with self._lock:
            notes, _ = self.read()
            if sum(n["state"] == "wait" for n in notes) >= MAX_WAITING:
                raise NoteRequestError(f"已经有 {MAX_WAITING} 张在等 TA 看了，等 TA 看过再写")
            seen = {n["id"] for n in notes}
            while (nid := "N-" + secrets.token_hex(8)) in seen:
                pass
            event = {"type": "note", "id": nid, "at": at, "text": text}
            event |= {k: v for k, v in (("quote", quote), ("recordId", record_id), ("kind", kind)) if v}
            self._append(event)
        return {k: event.get(k) for k in ("id", "at", "text", "quote", "recordId", "kind")} | {"state": "wait"}

    def reply(self, note_id, decision, at, reply=None):
        """TA 回一张便条：changed＝认同、已经改好；declined＝不改，reply 写为什么。"""
        if not isinstance(note_id, str) or not _ID.fullmatch(note_id):
            raise NoteRequestError("id 是便条开头那个 N-…（16 位十六进制）")
        if decision not in ("changed", "declined"):
            raise NoteRequestError("decision 只能是 changed（认同、已改）或 declined（不改）")
        reply = _text(reply, "reply", MAX_TEXT, required=decision == "declined")
        with self._lock:
            notes, _ = self.read()
            note = next((n for n in notes if n["id"] == note_id), None)
            if note is None:
                raise NoteRequestError(f"没有这张便条：{note_id}")
            if note["state"] != "wait":
                raise NoteRequestError(f"{note_id} 已经回过了，不用再回")
            event = {"type": "reply", "id": note_id, "at": at, "decision": decision}
            if reply:
                event["reply"] = reply
            self._append(event)
        return note | {"state": "done" if decision == "changed" else "said", "reply": reply, "repliedAt": at}


KIND = {"wrong": "记错了", "changed": "已经变了"}


def format_block(notes, bad_lines=()):
    """开场最先给的那一段：只列还在等 TA 看的。没有就 None（不占开场的字）。"""
    waiting = [n for n in notes if n["state"] == "wait"]
    if not waiting and not bad_lines:
        return None
    lines = ["【对方写给你的便条】对方在页面上写给你的，先看这些。对照记忆自己判断，不要因为是对方写的就照改："
             "说得对，用 latent_correct（从来不对）或 latent_supersede（后来变了）改正文，再用 latent_note_reply 记 changed；"
             "不认同，用 latent_note_reply 记 declined，用一两句话告诉对方为什么。"]
    for n in waiting:
        about = [f"针对：{n['quote']}" if n.get("quote") else None,
                 f"record={n['recordId']}" if n.get("recordId") else None,
                 KIND.get(n.get("kind"))]
        about = "｜".join(x for x in about if x)
        lines.append(f"- [{n['id']}｜{str(n.get('at') or '')[:10]} 写的] {n['text']}" + (f"（{about}）" if about else ""))
    if bad_lines:
        lines.append("- ⚠ 便条文件另有格式错误行：" + "、".join(map(str, bad_lines)) + "；这些行没有被静默当成没有。")
    return "\n".join(lines)


def _selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        st = NoteStore(Path(td) / FILENAME)
        # 1. 没文件 = 没便条，开场不占字
        assert st.read() == ([], []) and format_block([]) is None
        # 2. 写一张带引用的、一张不带的；状态都是 wait，开场按先后列出来，带 record 与“记错了”
        a = st.add("我身高 164，不是 163。", "2026-10-06T10:00:00+08:00",
                   quote="她 163", record_id="0123456789abcdef", kind="wrong")
        b = st.add("合作项目那条已经结了。", "2026-10-06T10:05:00+08:00")
        notes, bad = st.read()
        assert [n["state"] for n in notes] == ["wait", "wait"] and not bad
        blk = format_block(notes)
        assert blk.startswith("【对方写给你的便条】") and a["id"] in blk and "record=0123456789abcdef" in blk \
            and "记错了" in blk and blk.index(a["id"]) < blk.index(b["id"]), blk
        # 3. 回：changed → done；declined 必须写理由 → said；回过的不许再回；回过的不再进开场
        st.reply(a["id"], "changed", "2026-10-06T11:00:00+08:00")
        for bad_call in (lambda: st.reply(b["id"], "declined", "t"),          # 不改却不说为什么
                         lambda: st.reply(a["id"], "declined", "t", "x"),     # 已经回过
                         lambda: st.reply("N-0000000000000000", "changed", "t"),
                         lambda: st.reply(b["id"], "maybe", "t")):
            try:
                bad_call()
                raise AssertionError("该拒的没拒")
            except NoteRequestError:
                pass
        st.reply(b["id"], "declined", "2026-10-06T11:01:00+08:00", "还没结：验收单还没签。")
        notes, _ = st.read()
        assert [(n["state"], n["reply"]) for n in notes] == [("done", None), ("said", "还没结：验收单还没签。")]
        assert format_block(notes) is None
        # 4. 参数：空、太长、坏 recordId、坏 kind 都 400 式拒绝，不截断
        for args in (("  ",), ("x" * (MAX_TEXT + 1),)):
            try:
                st.add(*args, at="t")
                raise AssertionError("该拒的没拒")
            except NoteRequestError:
                pass
        for kw in ({"record_id": "XYZ"}, {"kind": "maybe"}):
            try:
                st.add("x", "t", **kw)
                raise AssertionError("该拒的没拒")
            except NoteRequestError:
                pass
        # 5. 坏行跳过但报出来；等着的超过上限就不让再写
        with open(st.path, "a", encoding="utf-8") as f:
            f.write("{坏了\n")
        notes, bad = st.read()
        assert len(notes) == 2 and bad == [5], bad
        assert "格式错误行：5" in format_block(notes, bad)
        for i in range(MAX_WAITING):
            st.add(f"第 {i} 张", "t")
        try:
            st.add("再来一张", "t")
            raise AssertionError("上限没拦住")
        except NoteRequestError:
            pass
    # 6. 没配语料目录：读是空，写明确报错
    assert NoteStore(None).read() == ([], [])
    try:
        NoteStore(None).add("x", "t")
        raise AssertionError("没路径还写进去了")
    except NoteStoreError:
        pass
    print("selftest ok（没文件/写/开场列/回 changed·declined/重复回/参数/坏行/上限/无路径）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        _selftest()
