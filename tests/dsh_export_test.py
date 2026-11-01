#!/usr/bin/env python3
"""``examples/host-dsh/dsh_export.py`` 的夹具测试：只用虚构事件，不碰真实会话。

跑法::

    python3 tests/dsh_export_test.py

判据（全部断言，失败即非零退出）：

1. 注入消息被丢弃 —— ``source.kind`` 为 ``schedule`` / ``plugin`` 的用户消息不进输出。
2. 噪声被丢弃 —— 思考块、工具调用、工具结果、空文本消息都不进输出。
3. 去重 —— 同一个消息 id 重放两次只保留一条。
4. 时间归一 —— 毫秒 epoch 与 ISO 字符串都能解析，且按 ``--tz`` 输出。
5. 两种布局 —— 助手消息在 ``data.message.content`` 与 ``data.content`` 都认。
6. 增量 —— 同一会话第二次导出新增 0 条；会话被重写变短时从头重导。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXAMPLES = os.path.join(ROOT, "examples", "host-dsh")
FIXTURE = os.path.join(EXAMPLES, "fixture_events.jsonl")

sys.path.insert(0, EXAMPLES)
import dsh_export  # noqa: E402


def load_fixture() -> list[dict]:
    with open(FIXTURE, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_records() -> None:
    tz = dsh_export.make_tz("+08:00")
    records = list(dsh_export.records_of(load_fixture(), tz))

    ids = [r["id"] for r in records]
    assert ids == ["m0001", "a0001", "a0002"], f"导出集合不对：{ids}"
    assert [r["role"] for r in records] == ["user", "assistant", "assistant"]

    first = records[0]
    assert first["text"] == "今天风很大，我把窗关上了。"
    assert first["ts"].endswith("+08:00"), f"时区没生效：{first['ts']}"
    # 1762000000 秒按 +08:00 是 2025-11-01 20:26:40
    assert first["ts"] == "2025-11-01T20:26:40+08:00", first["ts"]

    # 毫秒 epoch 被除以 1000：a0001 的 1762000120000 与 m0001 正好差 120 秒
    span = datetime.fromisoformat(records[1]["ts"]) - datetime.fromisoformat(records[0]["ts"])
    assert span.total_seconds() == 120, f"毫秒换算不对，差值 {span}"

    # 第二种布局 + ISO 时间
    assert records[2]["id"] == "a0002"
    assert records[2]["ts"] == "2026-11-01T13:20:00+08:00", records[2]["ts"]

    # 稳定键存在且长度固定
    assert all(len(r["key"]) == 16 for r in records)


def test_cli_incremental(workdir: str) -> None:
    zstd = shutil.which("zstd")
    if not zstd:
        print("跳过 CLI 增量检查：本机没有 zstd")
        return

    session_dir = os.path.join(workdir, "sessions", "示例项目", "示例会话")
    os.makedirs(session_dir, exist_ok=True)
    raw = os.path.join(session_dir, "events.jsonl")
    shutil.copyfile(FIXTURE, raw)
    packed = os.path.join(session_dir, dsh_export.SESSION_FILE)
    subprocess.run([zstd, "-q", "-f", raw, "-o", packed], check=True)

    out = os.path.join(workdir, "chat.jsonl")
    state = os.path.join(workdir, "state.json")
    argv = ["--session", session_dir, "--out", out, "--state", state, "--tz", "+08:00"]

    dsh_export.main(argv)
    with open(out, encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle if line.strip()]
    assert [r["id"] for r in lines] == ["m0001", "a0001", "a0002"], lines

    # 再跑一次：0 新增
    dsh_export.main(argv)
    with open(out, encoding="utf-8") as handle:
        again = [json.loads(line) for line in handle if line.strip()]
    assert len(again) == 3, f"增量失效，重跑又加了 {len(again) - 3} 条"

    # 会话被压缩重写成更短：游标落后 → 从头重导
    with open(raw, encoding="utf-8") as handle:
        keep = [line for line in handle if line.strip()][:2]
    with open(raw, "w", encoding="utf-8") as handle:
        handle.writelines(keep)
    subprocess.run([zstd, "-q", "-f", raw, "-o", packed], check=True)
    with open(state, "w", encoding="utf-8") as handle:
        json.dump({packed: 99}, handle)
    dsh_export.main(argv)
    with open(out, encoding="utf-8") as handle:
        after = [json.loads(line) for line in handle if line.strip()]
    assert after[-1]["id"] == "m0001", "重写变短后没有从头重导"


def main() -> int:
    test_records()
    workdir = tempfile.mkdtemp(prefix="dsh-export-test-")
    try:
        test_cli_incremental(workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    print("dsh_export_test ok（过滤注入 / 丢弃噪声 / 去重 / 时间归一 / 两种布局 / 增量）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
