#!/usr/bin/env python3
"""把 DeepSeek Harness（DSH）的会话事件流导出成后台落档要的通用 JSONL。

DSH 把每个会话存成 ``~/.dsh/sessions/<项目>/<会话>/session.v4.jsonl.zstd``——
每行一个事件。用户与模型的话分别在 ``user/message``、``assistant/message`` 里，
其余是思考块、工具调用、工具结果、子代理回流、注入提醒、尝试事件等噪声。

这个导出器做四件事：

1. **只取人说的话**：两类消息事件、只要 ``content`` 里 ``type == "text"`` 的块。
2. **过滤注入消息**：``data.source.kind`` 不是缺省、也不是 ``"user"`` 的用户消息
   （定时提醒、插件注入的历史召回等）一律跳过——它们不是她说的。
3. **按消息 id 去重**：同一事件可能被重放；跨会话同名 id 只保留第一次。
4. **增量**：``--state`` 记住每个会话已导出的条数，重跑只追加新的；
   会话被压缩重写过（条数变少）则从头重导。

取不到时间的消息也跳过：落档端靠时间推游标，一行空时间会让整份来源读不进去。

时间字段按"事件顶层 ``time``（毫秒 epoch）→ ``data.time`` → ``ts``"依次取，
毫秒自动除以 1000。输出行形如::

    {"id": "...", "ts": "2026-11-01T13:07:00+08:00", "role": "user", "text": "..."}

用法::

    # 列出可导出的会话
    python3 dsh_export.py --list

    # 增量导出一个项目的全部会话
    python3 dsh_export.py --project <项目目录名> --out chat.jsonl --state export.state.json

    # 只导一个会话，且只要 2026-10-30 之后
    python3 dsh_export.py --session ~/.dsh/sessions/<项目>/<会话> --since 2026-10-30

导出失败时不会覆盖上一次成功的输出：新内容先在内存里攒好，最后一次性追加。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from typing import Iterable, Iterator

SESS_ROOT = os.path.join(os.path.expanduser("~"), ".dsh", "sessions")
SESSION_FILE = "session.v4.jsonl.zstd"
DEFAULT_TZ = timedelta(hours=8)


def make_tz(spec: str | None) -> timezone:
    """``+08:00`` / ``-05:30`` / ``UTC`` 三种写法；缺省 +08:00（本导出器原产地）。"""
    if not spec:
        return timezone(DEFAULT_TZ)
    if spec.upper() == "UTC":
        return timezone.utc
    sign = 1 if spec[0] != "-" else -1
    body = spec.lstrip("+-")
    hours, _, minutes = body.partition(":")
    return timezone(sign * timedelta(hours=int(hours), minutes=int(minutes or 0)))


def to_seconds(value) -> float | None:
    """把 ``time`` / ``ts`` 归一成秒。支持毫秒 epoch、秒 epoch、ISO 字符串。"""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return value / 1000.0 if value > 1e11 else float(value)
    text = str(value).strip()
    if text.isdigit():
        return to_seconds(int(text))
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def iso(seconds: float | None, tz: timezone) -> str:
    if seconds is None:
        return ""
    return datetime.fromtimestamp(seconds, tz).isoformat(timespec="seconds")


def text_of(content) -> str:
    """只要 text 块：思考、工具调用、工具结果都在别的 type 里，天然被跳过。"""
    parts = []
    for block in content or []:
        if isinstance(block, dict) and block.get("type") == "text":
            body = (block.get("text") or "").strip()
            if body:
                parts.append(body)
    return "\n".join(parts)


def records_of(events: Iterable[dict], tz: timezone,
               seen: set[str] | None = None) -> Iterator[dict]:
    """把事件流折成落档记录。``seen`` 传进来可跨会话去重。"""
    seen = set() if seen is None else seen
    for event in events:
        kind = event.get("type")
        if kind not in ("user/message", "assistant/message"):
            continue
        data = event.get("data") or {}

        if kind == "user/message":
            source = (data.get("source") or {}).get("kind")
            if source not in (None, "user"):
                continue                      # 定时提醒、插件注入的召回、子代理回流
            content = data.get("content")
            message_id = data.get("id")
            role = "user"
        else:
            # 不同版本的 DSH 把助手消息放在 data.message 或 data 本身，两种都认
            carrier = data.get("message") or data
            content = carrier.get("content")
            message_id = carrier.get("id") or data.get("id")
            role = "assistant"

        text = text_of(content)
        if not text or not message_id or message_id in seen:
            continue
        seconds = to_seconds(event.get("time") or data.get("time") or event.get("ts"))
        if seconds is None:
            continue                          # 落档端按时间推游标，没有时间的行会让整份来源读不进去
        seen.add(message_id)

        stamp = iso(seconds, tz)
        yield {
            "id": message_id,
            "ts": stamp,
            "role": role,
            "text": text,
            # 时间+角色+正文的短哈希，供自己核对导出结果；落档端只认 id 与 ts，不读这个字段
            "key": hashlib.sha1(f"{stamp}{role}{text}".encode()).hexdigest()[:16],
        }


def read_events(path: str) -> Iterator[dict]:
    """``zstd -dc`` 解压后逐行解析；不要求 Python 侧装 zstandard。"""
    try:
        done = subprocess.run(["zstd", "-dc", path], capture_output=True, check=False)
    except FileNotFoundError as exc:
        raise SystemExit("需要系统里有 zstd 命令（Android/Termux: pkg install zstd）") from exc
    if done.returncode != 0:
        raise SystemExit(f"解压失败 {path}：{done.stderr.decode('utf-8', 'replace')[:200]}")
    for line in done.stdout.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def sessions_under(project_dir: str) -> list[str]:
    found = []
    for name in sorted(os.listdir(project_dir)):
        path = os.path.join(project_dir, name, SESSION_FILE)
        if os.path.exists(path):
            found.append(path)
    return found


def load_state(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DSH 会话 → 落档通用 JSONL")
    parser.add_argument("--sessions-root", default=SESS_ROOT)
    parser.add_argument("--project", help="~/.dsh/sessions 下的项目目录名")
    parser.add_argument("--session", action="append", default=[],
                        help="直接给会话目录或压缩文件，可重复")
    parser.add_argument("--out", default="dsh-chat.jsonl")
    parser.add_argument("--state", default="dsh-export.state.json")
    parser.add_argument("--since", help="YYYY-MM-DD，只要这天之后（按消息时间）")
    parser.add_argument("--tz", help="输出时区，如 +08:00 / -05:30 / UTC，默认 +08:00")
    parser.add_argument("--list", action="store_true", help="只列出可导出的会话")
    args = parser.parse_args(argv)

    tz = make_tz(args.tz)
    paths: list[str] = []
    for item in args.session:
        candidate = item
        if os.path.isdir(candidate):
            candidate = os.path.join(candidate, SESSION_FILE)
        paths.append(candidate)
    if args.project:
        project_dir = os.path.join(args.sessions_root, args.project)
        if not os.path.isdir(project_dir):
            raise SystemExit(f"找不到项目目录：{project_dir}")
        paths.extend(sessions_under(project_dir))
    if not paths:
        root = args.sessions_root
        if os.path.isdir(root):
            for project in sorted(os.listdir(root)):
                full = os.path.join(root, project)
                if os.path.isdir(full):
                    paths.extend(sessions_under(full))

    if args.list:
        for path in paths:
            print(os.path.basename(os.path.dirname(path)), os.path.getsize(path))
        return 0
    if not paths:
        raise SystemExit("没有找到会话；用 --list 看看会话在哪")

    since = None
    if args.since:
        since = datetime.fromisoformat(args.since).replace(tzinfo=tz).timestamp()

    state = load_state(args.state)
    seen: set[str] = set()
    pending: list[dict] = []
    for path in paths:
        records = list(records_of(read_events(path), tz, seen))
        already = state.get(path, 0)
        if already > len(records):            # 会话被压缩重写：从头来
            already = 0
        for record in records[already:]:
            if since is not None and record["ts"]:
                if datetime.fromisoformat(record["ts"]).timestamp() < since:
                    continue
            pending.append(record)
        state[path] = len(records)

    if pending:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "a", encoding="utf-8") as handle:
            for record in pending:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(args.state, "w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"新增 {len(pending)} 条 → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
