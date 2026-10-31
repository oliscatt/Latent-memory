"""公仓 #62：无查询换窗的日期可信度回归。"""
from datetime import datetime
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from memory_retrieval import MemoryIndex, load_corpus
from mcp_server import MemoryServer


def call(server, name, args, now):
    request = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": name, "arguments": args}}
    result = server.handle(request, now=now)["result"]
    assert not result.get("isError"), result
    return result["content"][0]["text"]


class RecentRecallTest(unittest.TestCase):
    def test_opening_and_search(self):
        now = datetime(2026, 10, 31, 12).timestamp()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus"
            corpus.mkdir()
            old = root / "旧材料.md"
            old.write_text("青陶残片的清理方法。", encoding="utf-8")
            os.utime(old, (now - 365 * 86400,) * 2)
            copied = corpus / "旧材料.md"
            shutil.copyfile(old, copied)
            os.utime(copied, (now,) * 2)
            (corpus / "记录.md").write_text("## 2026-10-31 当天巡检\n巡检已完成。", encoding="utf-8")
            (corpus / "计划.md").write_text("## 2026-11-30 琥珀展览\n展览筹备计划。", encoding="utf-8")
            loaded = load_corpus(corpus)
            self.assertEqual(len(loaded.chunks), 3)
            rows = list(zip(loaded.chunks, loaded.meta))
            for sequence in (rows, reversed(rows)):
                index = MemoryIndex()
                for body, meta in sequence:
                    index.add(body, meta)
                index.build()
                server = MemoryServer(index=index, recall_topN=1)
                for query, body in (("青陶残片", "青陶残片的清理方法"), ("琥珀展览", "展览筹备计划")):
                    self.assertIn(body, call(server, "latent_search", {"query": query}, now))
                opening = call(server, "latent_session_start", {}, now)
                self.assertIn("当天巡检", opening)
                self.assertNotIn("青陶残片", opening)
                self.assertNotIn("琥珀展览", opening)

    def test_clock_boundary_and_legacy_metadata(self):
        now = 1800000000
        index = MemoryIndex()
        for delta in (86399, 86400, 86401):
            index.add(str(delta), {"timestamp": now + delta})
        scores = {row["text"]: row["score"] for row in index.recall_recent(now=now)}
        self.assertEqual(scores, {"86399": 1, "86400": 1, "86401": 0})
        only = MemoryIndex()
        only.add("只有文件时间", {"timestamp": now, "timestamp_source": "mtime"})
        only.add("较早复制但常被提起", {"timestamp": now - 86400, "timestamp_source": "mtime"})
        only.weights[1] = 2
        result = only.recall_recent(now=now)
        self.assertEqual([(row["id"], row["score"]) for row in result], [(1, 0), (0, 0)])


if __name__ == "__main__":
    unittest.main()
