"""事实模式专项：中性虚构事实、假向量，不联网、不写库。"""
from pathlib import Path
import datetime as dt
import json
import os
import re
import sys
import tempfile
import time
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import passive_recall as pr
from passive_facts import FactIndex, FactCooldown, COOLDOWN_SECONDS

FACTS = [
    {"id": "a", "block": "blk-a", "fact": "她说九月一滴酒都没喝，只是有时候会想。", "event_date": "2026-01-02", "written": "2026-01-02", "tag": "life", "kind": "event"},
    {"id": "b", "block": "blk-b", "fact": "她在海边小镇吃了一碗很烫的汤饭。", "event_date": None, "written": "2026-01-03", "tag": "life", "kind": "event"},
    {"id": "c", "block": "blk-c", "fact": "她把备份脚本改成每半小时拉一次，保留最新三份。", "event_date": "2026-01-04", "written": "2026-01-04", "tag": "work", "kind": "state"},
    {"id": "d", "block": "blk-d", "fact": "第九次开闸复盘：测试句错联一处。", "event_date": "2026-01-05", "written": "2026-01-05", "tag": "meta", "kind": "event"},
    {"id": "e", "block": "blk-e", "fact": "她今天刚说想吃楼下那家面，还没去。", "event_date": "2026-01-10", "written": "2026-01-10", "tag": "life", "kind": "event"},
    {"id": "f", "block": "blk-f", "fact": "她给猫买了一个会发光的球，猫不理。", "event_date": "2026-01-06", "written": "2026-01-06", "tag": "life", "kind": "event"},
]
TODAY = "2026-01-10"


class FakeProvider:
    """字符袋向量：共享的字越多越近；足够让排序可预期。"""
    id = "fake-char-bag"

    def embed(self, texts, is_query=False):
        out = []
        for text in texts:
            vec = [0.0] * 64
            for ch in text:
                vec[ord(ch) % 64] += 1.0
            out.append(vec)
        return out


class StubIndex:
    chunks, retracted, superseded = [], set(), set()
    time_context = None

    def __init__(self, today, live_blocks=None):
        self.fixed_now = time.mktime(dt.date.fromisoformat(today).timetuple()) + 12 * 3600
        self.meta = [{"record_id": b, "source": f"{b}.md"} for b in (live_blocks or [])]


def _write(tmp):
    path = Path(tmp) / "facts.jsonl"
    path.write_text("".join(json.dumps(f, ensure_ascii=False) + "\n" for f in FACTS), encoding="utf-8")
    return path


def _ask(service, text, n, session="s"):
    return service.candidate({"userInput": text, "turn": {
        "sessionId": session, "turnId": str(n), "deliveryId": f"{session}-{n}"}})


def pr_fact_index_or_none(root):
    from passive_facts import fact_index_from_env
    return fact_index_from_env(None, default_root=root)[0]


class FactModeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = _write(self.tmp.name)
        self.cool_path = Path(self.tmp.name) / "cool.json"
        # 服务端默认把冷却与向量缓存放 ~/.cache；测试一律落临时目录，不碰真实主目录。
        self._old_state = os.environ.get("LATENT_PASSIVE_FACT_STATE_DIR")
        os.environ["LATENT_PASSIVE_FACT_STATE_DIR"] = str(Path(self.tmp.name) / "state")

    def tearDown(self):
        if self._old_state is None:
            os.environ.pop("LATENT_PASSIVE_FACT_STATE_DIR", None)
        else:
            os.environ["LATENT_PASSIVE_FACT_STATE_DIR"] = self._old_state
        self.tmp.cleanup()

    def service(self, today=TODAY, provider=FakeProvider(), live_blocks=None):
        index = FactIndex(self.path, provider=provider)
        return pr.PassiveRecallService(StubIndex(today, live_blocks), assemble_candidates=True,
                                       fact_index=index, fact_cooldown=FactCooldown(self.cool_path))

    def test_load_filters_meta_and_ids_are_stable(self):
        index = FactIndex(self.path)
        self.assertEqual(len(index.rows), 5)
        self.assertNotIn("第九次", "".join(r["text"] for r in index.rows))
        again = FactIndex(self.path)
        self.assertEqual(sorted(index.by_id), sorted(again.by_id))

    def test_top_two_and_segments_fit(self):
        result = _ask(self.service(), "九月想喝酒", 1)
        self.assertEqual(result["status"], "ready", result)
        self.assertIn(len(result["records"]), (1, 2))
        self.assertIn("fact_top2", result["reasonCodes"])
        content = result["content"]
        segments = re.split(r"(?=〔来源：)", content)[1:]
        self.assertEqual(len(segments), len(result["records"]))
        for seg in segments:
            self.assertLessEqual(len(seg.encode("utf-8")), 900)
        self.assertNotIn("〔使用说明〕", content)
        self.assertIn("九月一滴酒", content)

    def test_same_day_written_excluded(self):
        result = _ask(self.service(), "楼下那家面", 1)
        self.assertNotIn("楼下那家面", result.get("content", ""))

    def test_cooldown_does_not_dig_deeper_and_expires(self):
        first = _ask(self.service(), "九月想喝酒", 1)
        ids1 = {r["recordId"] for r in first["records"]}
        # 同一句换窗口再问：前两名都在冷却，这一轮什么都不递，不由第 3、4 名顶上。
        second = _ask(self.service(), "九月想喝酒", 2, session="other")
        self.assertEqual(second["reasonCodes"], ["fact_cooled"], second)
        later = (dt.date.fromisoformat(TODAY) + dt.timedelta(days=1)).isoformat()  # 隔天同一时刻＝过了 12 小时
        third = _ask(self.service(today=later), "九月想喝酒", 3, session="third")
        self.assertEqual(ids1, {r["recordId"] for r in third.get("records", [])})

    def test_cooldown_12h_boundary_and_legacy_date_entry(self):
        self.assertEqual(COOLDOWN_SECONDS, 12 * 3600)
        cool = FactCooldown(None)
        cool.mark(["x"], 1000.0)
        self.assertTrue(cool.cooling("x", 1000.0 + COOLDOWN_SECONDS - 1))
        self.assertFalse(cool.cooling("x", 1000.0 + COOLDOWN_SECONDS))
        # 2026.09.30 前的冷却文件存的是日期，按那天本地零点折算，照样能读、照样按 12 小时解冻。
        self.cool_path.write_text(json.dumps({"version": 1, "entries": {"y": TODAY}}), encoding="utf-8")
        legacy = FactCooldown(self.cool_path)
        midnight = time.mktime(dt.date.fromisoformat(TODAY).timetuple())
        self.assertTrue(legacy.cooling("y", midnight + 11 * 3600))
        self.assertFalse(legacy.cooling("y", midnight + 13 * 3600))

    def test_floor_drops_unrelated(self):
        result = _ask(self.service(), "qqqq zzzz xxxx", 1)
        self.assertEqual(result["reasonCodes"], ["fact_below_floor"], result)

    def test_same_delivery_retry_does_not_remark(self):
        service = self.service()
        _ask(service, "九月想喝酒", 1)
        before = self.cool_path.read_text(encoding="utf-8")
        _ask(service, "九月想喝酒", 1)
        self.assertEqual(before, self.cool_path.read_text(encoding="utf-8"))

    def test_gate_still_drops_noise(self):
        service = self.service()
        self.assertEqual(_ask(service, "嗯", 1)["reasonCodes"], ["low_information"])
        self.assertEqual(_ask(service, "/help", 2)["reasonCodes"], ["control_command"])

    def test_no_embedding_is_empty(self):
        result = _ask(self.service(provider=None), "想喝酒", 1)
        self.assertEqual(result["reasonCodes"], ["fact_no_embedding"])

    def test_facts_from_retracted_block_not_delivered(self):
        live = [f"blk-{k}" for k in "bcdef"]          # blk-a 已撤回／取代，不在现行正文里
        result = _ask(self.service(live_blocks=live), "九月想喝酒", 1)
        self.assertNotIn("九月一滴酒", result.get("content", ""))
        full = _ask(self.service(live_blocks=live + ["blk-a"]), "九月想喝酒", 2, session="x")
        self.assertIn("九月一滴酒", full.get("content", ""))

    def test_fact_without_block_traced_by_source_file(self):
        extra = {"id": "g", "fact": "她说九月一滴酒都没喝这件事让她挺骄傲。", "event_date": None,
                 "written": "2026-01-07", "block": None, "source_file": "window_58_x.md",
                 "tag": "life", "kind": "event"}
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(extra, ensure_ascii=False) + "\n")
        live = [f"blk-{k}" for k in "bcdef"]
        gone = _ask(self.service(live_blocks=live), "九月一滴酒都没喝", 1)
        self.assertNotIn("挺骄傲", gone.get("content", ""), "来源文件不在语料里就不递")
        here = _ask(self.service(live_blocks=live + ["window_58_x"]), "九月一滴酒都没喝", 2, session="y")
        self.assertIn("挺骄傲", here.get("content", ""))

    def test_state_fact_labelled_as_then(self):
        result = _ask(self.service(), "备份脚本每半小时拉一次", 1)
        self.assertIn("当时的状态", result.get("content", ""), result)

    def test_background_build_warms_then_serves(self):
        import threading
        gate = threading.Event()

        class Slow(FakeProvider):
            def embed(self, texts, is_query=False):
                if not is_query:
                    gate.wait(5)
                return super().embed(texts, is_query)
        index = FactIndex(self.path, provider=Slow(), background=True)
        service = pr.PassiveRecallService(StubIndex(TODAY), assemble_candidates=True,
                                          fact_index=index, fact_cooldown=FactCooldown(self.cool_path))
        self.assertEqual(_ask(service, "九月想喝酒", 1)["reasonCodes"], ["fact_index_warming"])
        gate.set()
        for _ in range(50):
            if index.ready:
                break
            time.sleep(0.05)
        self.assertEqual(_ask(service, "九月想喝酒", 2)["status"], "ready")

    def test_directory_reload_picks_up_increment(self):
        from passive_facts import append_facts
        root = Path(self.tmp.name) / "事实库"
        (root / "全量").mkdir(parents=True)
        self.path.rename(root / "全量" / "facts.jsonl")
        index = FactIndex(root, provider=FakeProvider())
        self.assertEqual(index.root, root)
        before = len(index.rows)
        append_facts(root, "local", "2026-01-08", "blk-new", [
            {"fact": "她说周末想去看海边的日落，想带上拍立得。", "tag": "life"}])
        self.assertTrue(index.reload_if_changed(min_interval=0))
        for _ in range(50):
            if len(index.rows) == before + 1 and index.vectors is not None \
                    and len(index.vectors) == before + 1:
                break
            time.sleep(0.05)
        self.assertEqual(len(index.rows), before + 1)
        self.assertEqual(len(index.vectors), before + 1)
        self.assertFalse(index.reload_if_changed(min_interval=0), "没变动不该重读")

    def test_append_facts_rejects_bad_batch_whole(self):
        from passive_facts import append_facts
        root = Path(self.tmp.name) / "lib"
        with self.assertRaises(ValueError):
            append_facts(root, "local", "2026-01-08", "blk", [
                {"fact": "她说周末想去看海边的日落。"}, {"fact": "短"}])
        self.assertFalse((root / "增量").exists(), "整批拒收，不半写")

    def test_latent_append_writes_facts_linked_to_record(self):
        try:
            import mcp_server
        except SyntaxError:
            self.skipTest("mcp_server 需要 Python 3.12+")
        from memory_retrieval import MemoryIndex
        root = Path(self.tmp.name) / "事实库"
        (root / "全量").mkdir(parents=True)
        self.path.rename(root / "全量" / "facts.jsonl")
        corpus = Path(self.tmp.name) / "timeline"
        corpus.mkdir()
        old = os.environ.get("LATENT_PASSIVE_FACTS")
        os.environ["LATENT_PASSIVE_FACTS"] = str(root)
        try:
            srv = mcp_server.MemoryServer(index=MemoryIndex().build(), corpus_dir=str(corpus),
                                          enable_passive_recall=True)
            with self.assertRaises(mcp_server.ToolError):
                srv._tool_memory_append({"text": "她今天买了新耳机。", "current_state": "已买",
                                         "facts": [{"fact": "短"}]})
            self.assertEqual(list(corpus.glob("*.md")), [], "事实不合格时正文也不写")
            out = srv._tool_memory_append({
                "text": "她今天买了新耳机，花了三百块，说音质比旧的好很多。", "current_state": "已买到手",
                "facts": [{"fact": "她买了新耳机，花了三百块，说音质比旧的好很多。", "kind": "event"}]})
        finally:
            if old is None:
                os.environ.pop("LATENT_PASSIVE_FACTS", None)
            else:
                os.environ["LATENT_PASSIVE_FACTS"] = old
        self.assertIn("factsStatus=saved", out)
        rid = re.search(r"recordId=([0-9a-f]{16})", out).group(1)
        written = [json.loads(l) for p in (root / "增量" / "local").glob("*.jsonl")
                   for l in p.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([w["block"] for w in written], [rid])

    def test_default_fact_root_next_to_corpus_without_config(self):
        try:
            import mcp_server
        except SyntaxError:
            self.skipTest("mcp_server 需要 Python 3.12+")
        from memory_retrieval import MemoryIndex
        self.assertIn("facts", mcp_server.INSTRUCTIONS, "写回带事实的规矩要写在服务端说明里，不靠人格文件")
        corpus = Path(self.tmp.name) / "timeline"
        corpus.mkdir()
        old = os.environ.pop("LATENT_PASSIVE_FACTS", None)
        try:
            srv = mcp_server.MemoryServer(index=MemoryIndex().build(), corpus_dir=str(corpus),
                                          enable_passive_recall=True)
            self.assertIsNone(srv.passive.fact_index, "还没回填全量时不启用事实模式")
            out = srv._tool_memory_append({
                "text": "对方今天第一次去游泳，游了五百米，说腿抽筋了一次。", "current_state": "已发生",
                "facts": ["对方第一次去游泳，游了五百米，腿抽筋了一次。"]})
            self.assertIn("factsStatus=saved", out)
            root = Path(self.tmp.name) / "事实库"
            self.assertEqual(len(list((root / "增量" / "local").glob("*.jsonl"))), 1)
            (root / "全量-测试").mkdir()
            srv2 = mcp_server.MemoryServer(index=MemoryIndex().build(), corpus_dir=str(corpus),
                                           enable_passive_recall=True)
            self.assertIsNotNone(srv2.passive.fact_index, "回填全量后默认启用")
        finally:
            if old is not None:
                os.environ["LATENT_PASSIVE_FACTS"] = old

    def _backfill_index(self):
        from memory_retrieval import MemoryIndex
        index = MemoryIndex()
        rows = [("周六她在楼下面馆吃了一碗牛肉面，说汤太咸。", "timeline"),
                ("她把阳台的绿萝换了一个蓝色的新花盆。", "timeline"),
                ("索引摘要：吃面、换花盆。", "index"),
                ("一条被部署插件隐藏的记录。", "timeline"),
                ("她说要去学游泳，后来撤回说是记错了。", "timeline")]
        for n, (text, layer) in enumerate(rows):
            index.add(text, {"source": f"b{n}.md", "chunk_index": 0, "layer": layer,
                             "local_date": "2026-01-0" + str(n + 1)})
        index.build()
        index.retracted.add(4)
        index.hidden_indices = lambda now=None: frozenset({3})   # 模拟插件隐藏第 4 块
        return index

    def test_backfill_skips_index_layer_hidden_and_retracted(self):
        from passive_facts import FactBackfill
        blocks = FactBackfill(Path(self.tmp.name) / "事实库").blocks(self._backfill_index())
        self.assertEqual([t for _r, _d, t in blocks],
                         ["周六她在楼下面馆吃了一碗牛肉面，说汤太咸。", "她把阳台的绿萝换了一个蓝色的新花盆。"],
                         "只拆 timeline 层；插件隐藏的、撤回的都不发给 AI")

    def test_backfill_submit_validates_per_block_and_resumes(self):
        from passive_facts import FactBackfill, BACKFILL_DIR
        index, root = self._backfill_index(), Path(self.tmp.name) / "事实库"
        first = FactBackfill(root).next_batch(index, limit=1)
        self.assertEqual(len(first), 1)
        a = first[0]["block"]
        b = FactBackfill(root).next_batch(index, limit=5)[1]["block"]
        blocks, facts, errors = FactBackfill(root).submit(index, [
            {"block": a, "facts": [{"fact": "她周六在楼下面馆吃了一碗牛肉面，嫌汤太咸。", "event_date": "2026-01-01"}]},
            {"block": b, "facts": [{"fact": "太短"}]},
            {"block": "不存在", "facts": []}])
        self.assertEqual((blocks, facts), (1, 1))
        self.assertEqual([e[0] for e in errors], [b, "不存在"], "坏块整块退回，好块照收")
        again = FactBackfill(root)
        self.assertEqual([x["block"] for x in again.next_batch(index)], [b], "进度落盘，换个实例接着拆")
        _n, _f, dup = again.submit(index, [{"block": a, "facts": []}])
        self.assertTrue(dup and "已经交过" in dup[0][1])
        self.assertTrue((root / BACKFILL_DIR / "facts.jsonl").is_file())
        self.assertIsNone(pr_fact_index_or_none(root), "拆到一半不开启事实模式")
        self.assertEqual(FactIndex(root).rows, [], "暂存里拆到一半的事实不算现行事实")
        with self.assertRaises(ValueError):
            again.finish(index, "2026-01-10")
        self.assertEqual(again.submit(index, [{"block": b, "facts": []}])[:2], (1, 0), "没什么可拆的块交空数组")
        kept, merged, name = again.finish(index, "2026-01-10")
        self.assertEqual((kept, merged, name), (1, 0, "全量-2026-01-10"))
        self.assertFalse((root / BACKFILL_DIR).exists(), "落库后暂存清掉")
        rows = [json.loads(l) for l in (root / name / "facts.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual((rows[0]["block"], rows[0]["written"]), (a, "2026-01-01"), "written 取块日期")
        self.assertIsNotNone(pr_fact_index_or_none(root), "有全量才开启事实模式")

    def test_empty_block_not_resent_second_round(self):
        from passive_facts import FactBackfill
        index, root = self._backfill_index(), Path(self.tmp.name) / "事实库"
        first = FactBackfill(root)
        a, b = [x["block"] for x in first.next_batch(index)]
        first.submit(index, [{"block": a, "facts": ["她周六在楼下面馆吃了一碗牛肉面，嫌汤太咸。"]},
                             {"block": b, "facts": []}])
        kept, _merged, name = first.finish(index, "2026-01-10")
        self.assertEqual(kept, 1)
        self.assertEqual(set(json.loads((root / name / "blocks.json").read_text(encoding="utf-8"))), {a, b},
                         "blocks.json 要记下本轮处理过的全部块，含交空的")
        again = FactBackfill(root)
        self.assertEqual(again.next_batch(index), [], "交空的块第二轮不再发")
        self.assertEqual(again.finish(index, "2026-01-11"), (0, 0, None), "没有新块时 finish 不报错、不生成新全量")
        self.assertEqual(len(list(root.glob("全量*"))), 1)

    def test_all_duplicate_round_records_blocks_without_new_full(self):
        from passive_facts import FactBackfill
        from memory_retrieval import MemoryIndex
        index, root = self._backfill_index(), Path(self.tmp.name) / "事实库"
        first = FactBackfill(root)
        a, b = [x["block"] for x in first.next_batch(index)]
        first.submit(index, [{"block": a, "facts": ["她周六在楼下面馆吃了一碗牛肉面，嫌汤太咸。"]},
                             {"block": b, "facts": []}])
        first.finish(index, "2026-01-10")
        grown = MemoryIndex()
        for text, meta in zip(index.chunks, index.meta):
            grown.add(text, dict(meta))
        grown.add("又提到周六楼下那碗牛肉面，还是嫌咸。", {"source": "b9.md", "chunk_index": 0,
                                                     "layer": "timeline", "local_date": "2026-01-01"})
        grown.build()
        grown.retracted, grown.hidden_indices = set(index.retracted), index.hidden_indices
        again = FactBackfill(root)
        [c] = [x["block"] for x in again.next_batch(grown)]
        again.submit(grown, [{"block": c, "facts": [{"fact": "她周六在楼下面馆吃了一碗牛肉面，嫌汤太咸了"}]}])
        kept, merged, name = again.finish(grown, "2026-01-11")
        self.assertEqual((kept, merged, name), (0, 1, None), "全部与已有事实重复：不报错、不生成新全量")
        self.assertEqual(FactBackfill(root).next_batch(grown), [], "被判重复的块也记下，下次不再发")
        self.assertEqual(len(list(root.glob("全量*"))), 1)

    def test_first_round_all_empty_keeps_progress_without_enabling_fact_mode(self):
        from passive_facts import FactBackfill
        index, root = self._backfill_index(), Path(self.tmp.name) / "事实库"
        first = FactBackfill(root)
        first.submit(index, [{"block": x["block"], "facts": []} for x in first.next_batch(index)])
        self.assertEqual(first.finish(index, "2026-01-10"), (0, 0, None))
        self.assertEqual(list(root.glob("全量*")), [], "一条事实都没有就不建全量，免得误开事实模式")
        self.assertIsNone(pr_fact_index_or_none(root))
        self.assertEqual(FactBackfill(root).next_batch(index), [], "交空的块同样不再重发")

    def test_backfill_dedup_same_day_only(self):
        from passive_facts import dedup_facts
        rows = [{"id": "1", "fact": "她周六在楼下面馆吃了一碗牛肉面", "written": "2026-01-01", "event_date": None},
                {"id": "2", "fact": "她周六在楼下面馆吃了一碗牛肉面，嫌汤太咸", "written": "2026-01-01", "event_date": None},
                {"id": "3", "fact": "她周六在楼下面馆吃了一碗牛肉面", "written": "2026-02-01", "event_date": None}]
        kept, dups = dedup_facts(rows)
        self.assertEqual([r["id"] for r in kept], ["2", "3"], "同日近似留长的；跨天的不合并")
        self.assertEqual(dups, [("1", "2")])

    def test_backfill_tool_end_to_end_activates_fact_mode(self):
        try:
            import mcp_server
        except SyntaxError:
            self.skipTest("mcp_server 需要 Python 3.12+")
        corpus = Path(self.tmp.name) / "timeline"
        corpus.mkdir()
        old = os.environ.pop("LATENT_PASSIVE_FACTS", None)
        try:
            plain = mcp_server.MemoryServer(index=self._backfill_index(), corpus_dir=str(corpus))
            self.assertNotIn(mcp_server.FACT_BACKFILL_TOOL, {t["name"] for t in plain.tools},
                             "没开自动浮现就不出现回填工具，默认八个工具不变")
            srv = mcp_server.MemoryServer(index=self._backfill_index(), corpus_dir=str(corpus),
                                          enable_passive_recall=True)
            self.assertIn(mcp_server.FACT_BACKFILL_TOOL, {t["name"] for t in srv.tools})
            self.assertIsNone(srv.passive.fact_index)
            call = srv._tool_fact_backfill
            with self.assertRaises(mcp_server.ToolError) as no_embed:
                call({"action": "next"})
            self.assertIn("--embed", str(no_embed.exception), "没配向量就不许开工：建完自动浮现会整个哑掉")
            self.assertIn("⚠", call({"action": "status"}))
            srv.index.embed, srv.index.provider = True, FakeProvider()
            batch = json.loads(call({"action": "next"}).split("（items 每项 {block, facts}）：\n", 1)[1])
            self.assertEqual(len(batch), 2)
            bad = srv._call_tool(1, {"name": mcp_server.FACT_BACKFILL_TOOL,
                                     "arguments": {"action": "submit", "items": "x"}})
            self.assertTrue(bad["result"]["isError"])
            self.assertIn("写对", bad["result"]["content"][0]["text"])
            out = call({"action": "submit", "items": [
                {"block": batch[0]["block"], "facts": ["她周六在楼下面馆吃了一碗牛肉面，嫌汤太咸。"]},
                {"block": batch[1]["block"], "facts": [{"fact": "她给阳台的绿萝换了一个蓝色的新花盆。", "kind": "state"}]}]})
            self.assertIn("收下 2 块、2 条事实", out)
            self.assertIn("全量事实库已生成", call({"action": "finish"}))
            self.assertIsNotNone(srv.passive.fact_index, "finish 后当场开启事实模式，不用重启")
        finally:
            if old is not None:
                os.environ["LATENT_PASSIVE_FACTS"] = old

    def test_block_mode_gate_unchanged(self):
        admitted, reason, _ = pr._input_gate("想喝酒", short_terms=set(), index=None)
        self.assertEqual((admitted, reason), (False, "low_information"))


if __name__ == "__main__":
    unittest.main()
