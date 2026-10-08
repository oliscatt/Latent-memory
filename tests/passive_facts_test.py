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


def _lead_of(service, text, exclude):
    ranked = service.fact_index.ranked(text, exclude=exclude, top=None)
    rest = [score for _row, score in ranked[1:11]]
    return f"fact_lead:{ranked[0][1] - sum(rest) / len(rest):.3f}"


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

    def test_origin_derived_empty_and_unrecorded(self):
        """#43 第 2 步 P1（事实模式）：同一句不带 origin 会 ready，声明为派生就回空、不登记、不记冷却。"""
        service = self.service()
        ask = {"userInput": "九月想喝酒", "turn": {"sessionId": "s", "turnId": "1", "deliveryId": "d1"},
               "origin": "derived"}
        result = service.candidate(ask)
        self.assertEqual((result["status"], result["reasonCodes"]), ("empty", ["origin_derived"]))
        self.assertNotIn("records", result)
        self.assertEqual((service._deliveries, self.cool_path.exists()), ({}, False))
        self.assertNotIn("d1", json.dumps(service.inspect()))
        main = service.candidate(dict(ask, origin="main", turn=dict(ask["turn"], deliveryId="d2")))
        self.assertEqual(main["status"], "ready", "主会话照常浮")

    def test_origin_bad_value_rejected(self):
        """P3：origin 只认 main／derived，报错写明可选值。"""
        with self.assertRaisesRegex(pr.PassiveRecallRequestError, "main.*derived"):
            self.service().candidate({"userInput": "九月想喝酒", "origin": "subagent", "turn": {
                "sessionId": "s", "turnId": "1", "deliveryId": "d1"}})

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

    def test_margin_code_top1_minus_top2_before_floor_and_cooldown(self):
        """ready 末尾带 fact_margin:<排除后第 1 名减第 2 名>、fact_lead:<第 1 名减第 2～11 名均值>，
        不看下限与冷却；其余原因码与加它们之前一样。"""
        exclude = lambda row: (row["meta"].get("written") or "") >= TODAY

        def margin_of(service, text):
            (_r1, s1), (_r2, s2) = service.fact_index.ranked(text, exclude=exclude, top=2)
            return f"fact_margin:{s1 - s2:.3f}"

        # 排除后只剩 1 条（今天＝01-03，只有 a 写在之前）：没有第 2 名，两个码都不给。
        single = _ask(self.service(today="2026-01-03"), "九月想喝酒", 0)
        self.assertEqual(single["status"], "ready", single)
        self.assertFalse([c for c in single["reasonCodes"]
                          if c.startswith(("fact_margin:", "fact_lead:"))], single)
        self.cool_path.unlink()
        # 第 2 名低于下限（a 0.55／c 0.13）：只递 1 条，分差照算。
        service = self.service()
        codes = _ask(service, "九月想喝酒", 1)["reasonCodes"]
        self.assertEqual(codes, ["fact_mode_signal", "w4_assembled", "fact_top2",
                                 codes[3], margin_of(service, "九月想喝酒"),
                                 _lead_of(service, "九月想喝酒", exclude)], codes)
        self.assertTrue(codes[3].startswith("fact_score:") and all(len(c) <= 20 for c in codes[-2:]), codes)
        # 第 1 名在冷却（b 0.54／f 0.50）：只递 f，分差仍是 b 减 f，不往下顺延。
        service = self.service()
        top1 = service.fact_index.ranked("她的猫", exclude=exclude, top=1)[0][0]
        service.fact_cooldown.mark([pr._chunk_key(top1["text"])], service._now())
        result = _ask(service, "她的猫", 2)
        self.assertEqual(len(result["records"]), 1, result)
        self.assertEqual(result["reasonCodes"][-2:], [margin_of(service, "她的猫"),
                                                      _lead_of(service, "她的猫", exclude)])

    def test_lead_code_mean_of_ranks_two_to_eleven(self):
        """fact_lead＝第 1 名减第 2～11 名的算术平均；候选不足 11 名按实有的算，只剩 1 名不给，空结果不给。"""
        exclude = lambda row: (row["meta"].get("written") or "") >= TODAY
        # 夹具排除后 4 条候选（a、b、c、f）：按第 2～4 名的均值算。
        service = self.service()
        ranked = service.fact_index.ranked("九月想喝酒", exclude=exclude, top=None)
        self.assertEqual(len(ranked), 4)
        scores = [s for _r, s in ranked]
        want = f"fact_lead:{scores[0] - sum(scores[1:]) / 3:.3f}"
        self.assertEqual(_ask(service, "九月想喝酒", 1)["reasonCodes"][-1], want)
        # 再写 14 条（排除后 18 条候选）：只取第 2～11 名，第 12 名以后不进均值。
        extra = [{"id": f"x{i:02d}", "block": f"blk-x{i:02d}", "fact": f"她九月去第{i}家店喝了一杯酒。",
                  "event_date": None, "written": "2026-01-05", "tag": "life", "kind": "event"} for i in range(14)]
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.writelines(json.dumps(f, ensure_ascii=False) + "\n" for f in extra)
        self.cool_path.unlink()
        service = self.service()
        ranked = service.fact_index.ranked("九月想喝酒", exclude=exclude, top=None)
        self.assertEqual(len(ranked), 18)
        scores = [s for _r, s in ranked]
        want = f"fact_lead:{scores[0] - sum(scores[1:11]) / 10:.3f}"
        self.assertNotEqual(want, f"fact_lead:{scores[0] - sum(scores[1:]) / 17:.3f}",
                            "夹具得让第 12 名以后的分数改变均值，才验得出只取到第 11 名")
        codes = _ask(service, "九月想喝酒", 2)["reasonCodes"]
        self.assertEqual(codes[-1], want, codes)
        self.assertEqual(sum(c.startswith("fact_lead:") for c in codes), 1, codes)
        # 空结果（与哪条都不够像）没有它。
        self.assertEqual(_ask(service, "qqqq zzzz xxxx", 3)["reasonCodes"], ["fact_below_floor"])

    def test_lead_code_does_not_change_delivery(self):
        """fact_lead 不参与任何门槛和排序：递哪几条只由下限、冷却和前 2 名决定，其余原因码与加它之前同形。"""
        exclude = lambda row: (row["meta"].get("written") or "") >= TODAY
        service = self.service()
        inputs = ["九月想喝酒", "她的猫", "九月想喝酒", "海边的汤饭", "猫和球", "qqqq zzzz xxxx"]
        for n, text in enumerate(inputs):
            ranked = service.fact_index.ranked(text, exclude=exclude, top=2)
            want = [pr._chunk_key(row["text"]) for row, score in ranked
                    if score >= pr.FACT_FLOOR
                    and not service.fact_cooldown.cooling(pr._chunk_key(row["text"]), service._now())]
            result = _ask(service, text, n)
            self.assertEqual([r["recordId"] for r in result.get("records", [])], want, (text, result))
            codes = [c for c in result["reasonCodes"] if not c.startswith("fact_lead:")]
            if result["status"] == "ready":
                self.assertEqual(codes[:3], ["fact_mode_signal", "w4_assembled", "fact_top2"], codes)
                self.assertTrue(all(c.startswith("fact_score:") for c in codes[3:-1]), codes)
                self.assertTrue(codes[-1].startswith("fact_margin:"), codes)
            else:
                self.assertEqual(codes, result["reasonCodes"], "空结果不带 fact_lead")
        self.assertTrue(self.cool_path.exists())

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

    def test_topic_field_not_read(self):
        """#45 第 3 步 P4：同一份事实库每行加上 topic 和不加，逐句问两遍（第二遍看冷却），candidate() 逐位一致。"""
        asks = ["九月想喝酒", "备份脚本每半小时拉一次", "海边小镇的汤饭", "给猫买的发光球", "楼下那家面", "qqqq zzzz xxxx"]

        def run(root, facts):
            root.mkdir()
            path = root / "facts.jsonl"
            path.write_text("".join(json.dumps(f, ensure_ascii=False) + "\n" for f in facts), encoding="utf-8")
            index = FactIndex(path, provider=FakeProvider())
            self.assertFalse(any("topic" in row["meta"] for row in index.rows), "topic 不进 meta")
            service = pr.PassiveRecallService(StubIndex(TODAY), assemble_candidates=True, fact_index=index,
                                              fact_cooldown=FactCooldown(root / "cool.json"))
            return [json.dumps(_ask(service, q, n, session=f"s{r}"), ensure_ascii=False, sort_keys=True)
                    for r in range(2) for n, q in enumerate(asks)]

        base = Path(self.tmp.name)
        tagged = [dict(f, topic="备份" if f["kind"] == "state" else f"话题{f['id']}") for f in FACTS]
        plain = run(base / "plain", FACTS)
        self.assertEqual(plain, run(base / "topic", tagged))
        self.assertTrue(any('"status": "ready"' in r for r in plain) and any("fact_cooled" in r for r in plain),
                        "比的里面要有递出的轮次和冷却的轮次")

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
            {"fact": "她说周末想去看海边的日落，想带上相机。", "tag": "life"}])
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

    def test_facts_error_names_item_length_and_right_shape(self):
        # 判据：报错要指出第几条、哪一项、实际字符数与上下限，“写对”是 facts 自己的形状且能直接过校验。
        try:
            import mcp_server
        except SyntaxError:
            self.skipTest("mcp_server 需要 Python 3.12+")
        from memory_retrieval import MemoryIndex
        from passive_facts import append_facts
        long_fact = ("她把阳台的绿萝换了一个蓝色的新花盆，" * 10)[:137]
        facts = [{"fact": f"她说第{n}个周末想去看海边的日落。", "tag": "life", "kind": "event"} for n in range(8)]
        facts[2] = {"fact": long_fact, "tag": "life", "kind": "event"}
        corpus = Path(self.tmp.name) / "timeline"
        corpus.mkdir()
        srv = mcp_server.MemoryServer(index=MemoryIndex().build(), corpus_dir=str(corpus))
        with self.assertRaises(mcp_server.ToolError) as caught:
            srv._tool_memory_append({"text": "她今天换了花盆。", "current_state": "已换", "facts": facts})
        msg = str(caught.exception)
        for needle in ("facts[2]", "137", "8～120", "按字符数算"):
            self.assertIn(needle, msg)
        right = json.loads(next(l for l in msg.splitlines() if l.startswith("写对："))[3:])
        self.assertTrue(right["facts"])
        self.assertTrue(all(set(item) == {"fact", "tag", "kind"} for item in right["facts"]), right)
        self.assertEqual(append_facts(Path(self.tmp.name) / "lib", "local", "2026-01-08", "blk", right["facts"]),
                         len(right["facts"]), "“写对”要能直接通过校验")
        with self.assertRaisesRegex(ValueError, r"facts\[1\]\.tag"):
            append_facts(Path(self.tmp.name) / "lib", "local", "2026-01-08", "blk",
                         [facts[0], {"fact": facts[1]["fact"], "tag": "home"}])

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

    def test_backfill_tool_can_be_switched_off_by_deployment(self):
        try:
            import mcp_server
        except SyntaxError:
            self.skipTest("mcp_server 需要 Python 3.12+")
        corpus = Path(self.tmp.name) / "timeline"
        corpus.mkdir()
        old = os.environ.get("LATENT_FACT_BACKFILL")
        os.environ["LATENT_FACT_BACKFILL"] = "off"
        try:
            srv = mcp_server.MemoryServer(index=self._backfill_index(), corpus_dir=str(corpus),
                                          enable_passive_recall=True)
            names = {t["name"] for t in srv.tools}
            self.assertNotIn(mcp_server.FACT_BACKFILL_TOOL, names, "部署关掉回填工具就不许列出")
            self.assertIn(mcp_server.PASSIVE_RECALL_TOOL, names, "关回填工具不影响隐藏入口")
            blind = srv._call_tool(1, {"name": mcp_server.FACT_BACKFILL_TOOL,
                                       "arguments": {"action": "status"}}, hidden_ok=True)
            self.assertIn("未知工具", blind["error"]["message"], "模型硬调也当未知工具")
        finally:
            if old is None:
                os.environ.pop("LATENT_FACT_BACKFILL", None)
            else:
                os.environ["LATENT_FACT_BACKFILL"] = old

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
            # 事实向量在后台线程里算、算完写缓存到 state 目录；等它收尾再清临时目录，
            # 否则 tearDown 删目录时撞上正在写的缓存文件（约 1/25 偶发 Directory not empty）
            deadline = time.monotonic() + 10
            while not (srv.passive.fact_index.ready or srv.passive.fact_index.error) \
                    and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(srv.passive.fact_index.ready, srv.passive.fact_index.error)
        finally:
            if old is not None:
                os.environ["LATENT_PASSIVE_FACTS"] = old

    def test_block_mode_gate_unchanged(self):
        admitted, reason, _ = pr._input_gate("想喝酒", short_terms=set(), index=None)
        self.assertEqual((admitted, reason), (False, "low_information"))


# issue #45 第 1 步：同源成组＋状态软标注。夹具是 issue #45 第 7 节的虚构考研数据，S1～S7 是本组判据。
# 向量是手写的，只验逻辑、不验真实模型效果。
KAOYAN = [
    ("f1", "blk-0101", "2026-01-01", "阿离的考研冲刺校是 B 校。", (2, 1, 0, 0, 0)),
    ("f2", "blk-0101", "2026-01-01", "阿离的考研稳妥校是 C 校、D 校。", (2, 0, 1, 0, 0)),
    ("f3", "blk-0101", "2026-01-01", "阿离的考研保底校是 A 校。", (2, 0, 0, 1, 0)),
    ("f4", "blk-0601", "2026-06-01", "阿离改主意了，考研只报 E 校。", (2, 0, 0, 0, 1)),
]
ASK_ALL = "我考试那事你还记得吧"          # 四条都过下限，前 2 名是 f1、f4
ASK_SAFE = "我的保底校是哪所来着"         # f1～f3 过下限，f4 不过
ASK_ONLY = "那个保底校现在还作数吗"       # 只有 f3 过下限
OLDER = "这是较早的状态，可能已被更新"
# 跨块相似度默认关（STATE_SIM 为 None）；要测打开后的行为就临时设成这个值。
CROSS_SIM = 0.6


class TableProvider:
    """按原文查表给向量，查不到就报错——夹具里每个相似关系都是写死的。"""
    id = "table"

    def __init__(self, table):
        self.table = table

    def embed(self, texts, is_query=False):
        return [list(self.table[text]) for text in texts]


class FactGroupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cool_path = Path(self.tmp.name) / "cool.json"

    def tearDown(self):
        self.tmp.cleanup()

    def service(self, facts=KAOYAN, kind="state", max_piece_bytes=pr.DEFAULT_MAX_PIECE_BYTES):
        path = Path(self.tmp.name) / "facts.jsonl"
        # day 是一个日期（事件日＝写入日），或 (事件日, 写入日)。
        path.write_text("".join(json.dumps({"id": fid, "block": block, "fact": text,
                                            "event_date": day if isinstance(day, str) else day[0],
                                            "written": day if isinstance(day, str) else day[1],
                                            "tag": "life", "kind": kind}, ensure_ascii=False) + "\n"
                                for fid, block, day, text, _vec in facts), encoding="utf-8")
        table = {text: vec for *_rest, text, vec in facts}
        table.update({ASK_ALL: (3, 0.6, 0.2, 0.4, 0.5), ASK_SAFE: (1.2, 0.4, 0.4, 1.5, -1.8),
                      ASK_ONLY: (0.5, 0.3, 0.2, 2, -1), "我那时候住哪来着": (0.2, 0, 0, 1.5, -0.6),
                      "那 B 校和 E 校最后怎么定的": (1, 0, 0, 1, 0)})
        self.index = FactIndex(path, provider=TableProvider(table))
        return pr.PassiveRecallService(StubIndex("2026-07-01"), assemble_candidates=True,
                                       max_piece_bytes=max_piece_bytes, fact_index=self.index,
                                       fact_cooldown=FactCooldown(self.cool_path))

    @staticmethod
    def segments(result):
        """{事实原文: 来源行}，按递送顺序。"""
        out = {}
        for seg in re.split(r"(?=〔来源：)", result.get("content", ""))[1:]:
            head, body = seg.split("\n", 1)
            out[body.replace("〔历史证据结束〕", "").strip()] = head
        return out

    def text(self, fid):
        return next(text for f, _b, _d, text, _v in KAOYAN if f == fid)

    def test_fixture_preconditions(self):
        self.service()
        rows = {fid: self.index.by_id[pr._chunk_key(text)] for fid, _b, _d, text, _v in KAOYAN}
        for fid in ("f1", "f2", "f3"):
            self.assertGreaterEqual(self.index.similarity(rows["f4"], rows[fid]), CROSS_SIM)
        scored = dict((row["text"], s) for row, s in self.index.ranked(ASK_ALL, top=None))
        self.assertTrue(all(s >= pr.FACT_FLOOR for s in scored.values()), scored)
        self.assertEqual(sorted(scored, key=lambda t: -scored[t])[:2], [self.text("f1"), self.text("f4")])

    def test_s1_s2_group_and_older_label(self):
        # 默认：跨块不比；f1～f3 同块同一天也不互标——一轮四条都不带“较早”。
        self.assertIsNone(pr.STATE_SIM)
        plain = _ask(self.service(), ASK_ALL, 1)
        self.assertEqual(list(self.segments(plain)), [self.text(f) for f in ("f1", "f3", "f2", "f4")])
        self.assertIn("fact_group", plain["reasonCodes"])
        self.assertNotIn(OLDER, plain["content"])
        self.assertNotIn("fact_state_older", plain["reasonCodes"])
        # 设了 LATENT_PASSIVE_STATE_SIM 才比跨块：f4 与 f1～f3 余弦 0.8，较早的三条带标注。
        self.cool_path.unlink()
        with self.patched("STATE_SIM", CROSS_SIM):
            result = _ask(self.service(), ASK_ALL, 2)
        self.assertEqual(result["status"], "ready", result)
        segs = self.segments(result)
        self.assertEqual(list(segs), [self.text(f) for f in ("f1", "f3", "f2", "f4")])
        self.assertIn("fact_group", result["reasonCodes"])
        self.assertIn("fact_state_older", result["reasonCodes"])
        for fid in ("f1", "f2", "f3"):
            self.assertIn(OLDER, segs[self.text(fid)], fid)
        self.assertNotIn(OLDER, segs[self.text("f4")])
        self.assertTrue(all("当时的状态" in head for head in segs.values()))
        with self.patched("STATE_SIM", CROSS_SIM):
            self.assertEqual(result["content"], self.service()._assemble(result["records"]))

    def test_s2_dissimilar_states_not_labelled(self):
        # 变异检查补的：跨块比较打开时，不同块、余弦 < 线的两条状态，日期再有先后也不互标。
        home = ("f5", "blk-0301", "2026-03-01", "阿离现在住在学校旁边的出租屋。", (0, 0, 0, 1, -1))
        with self.patched("STATE_SIM", CROSS_SIM):
            result = _ask(self.service(KAOYAN + [home]), "我那时候住哪来着", 1)
        rows = {r["text"]: r for r in self.index.rows}
        self.assertLess(self.index.similarity(rows[self.text("f3")], rows[home[3]]), CROSS_SIM)
        self.assertEqual(list(self.segments(result)), [home[3], self.text("f3")], result)
        self.assertNotIn(OLDER, result["content"])

    def test_s2_same_block_counts_as_related(self):
        # 同块就算同一件事：余弦再低、跨块比较关着（默认），事件日较早的那条也标。
        recap = ("f6", "blk-0601", ("2026-01-15", "2026-06-01"), "阿离一月时还打算冲 B 校。", (0, 0, 0, 1, -1))
        result = _ask(self.service([KAOYAN[3], recap]), "那 B 校和 E 校最后怎么定的", 1)
        rows = {r["text"]: r for r in self.index.rows}
        self.assertLess(self.index.similarity(rows[self.text("f4")], rows[recap[3]]), CROSS_SIM)
        segs = self.segments(result)
        self.assertEqual(set(segs), {self.text("f4"), recap[3]}, result)
        self.assertIn(OLDER, segs[recap[3]])
        self.assertNotIn(OLDER, segs[self.text("f4")])

    def test_s3_no_label_without_newer_state(self):
        result = _ask(self.service(), ASK_SAFE, 1)
        segs = self.segments(result)
        self.assertEqual(set(segs), {self.text(f) for f in ("f1", "f2", "f3")}, result)
        self.assertNotIn(OLDER, result["content"])
        self.assertNotIn("fact_state_older", result["reasonCodes"])

    def test_s4_lone_old_state_not_labelled_and_siblings_need_floor(self):
        result = _ask(self.service(), ASK_ONLY, 1)
        self.assertEqual(list(self.segments(result)), [self.text("f3")], result)
        self.assertNotIn(OLDER, result["content"])
        self.assertNotIn("fact_group", result["reasonCodes"])

    def test_s5_at_most_three_siblings(self):
        six = [(f"g{i}", "blk-six", "2026-01-01", f"阿离那年考研准备的第{i}件小事。", (2, 0.1 * i, 0, 0, 0))
               for i in range(1, 7)]
        result = _ask(self.service(six, kind="event"), ASK_ALL, 1)
        self.assertEqual(len(result["records"]), 2 + pr.FACT_SIBLINGS, result)

    def test_s6_budget(self):
        result = _ask(self.service(), ASK_ALL, 1)
        self.assertLessEqual(len(result["content"].encode("utf-8")), pr.FACT_GROUP_BYTES)
        long = [(fid, block, day, text.rstrip("。") + "，" + "那天她把每所学校的分数线和往年报录比都抄在本子上" * 4 + "。", vec)
                for fid, block, day, text, vec in KAOYAN]
        cut = _ask(self.service(long), ASK_ALL, 1)
        self.assertLessEqual(len(cut["content"].encode("utf-8")), pr.FACT_GROUP_BYTES, cut)
        self.assertLess(len(cut["records"]), 4, "长事实时兄弟行要被预算截掉")
        bodies = list(self.segments(cut))
        self.assertTrue(bodies[0].startswith("阿离的考研冲刺校") and any(b.startswith("阿离改主意") for b in bodies))
        # 前 2 名本身超预算时照旧递，只是不再带兄弟行。
        self.cool_path.unlink()
        with self.patched("FACT_GROUP_BYTES", 100):
            tight = _ask(self.service(), ASK_ALL, 1)
        self.assertEqual(len(tight["records"]), 2, tight)

    def test_item4_group_about_1300_bytes_trimmed_to_default(self):
        # 上线前清单第 4 件 b：同一块 6 条都过下限，不设上限时整段约 1300 字节；默认上限下截到 1000 以内。
        six = [(f"g{i}", "blk-six", "2026-01-01", f"阿离那年考研准备的第{i}件小事：" + "把真题按年份装订好，每天早上先做一套英语再去图书馆占座" + "，晚上回宿舍对答案，周末把错题本重新抄一遍再讲给我听。",
                (2, 0.1 * i, 0, 0, 0)) for i in range(1, 7)]
        with self.patched("FACT_GROUP_BYTES", 10 ** 6):
            full = _ask(self.service(six, kind="event"), ASK_ALL, 1)
        size = len(full["content"].encode("utf-8"))
        self.assertTrue(1200 <= size <= 1400 and len(full["records"]) == 5, (size, len(full["records"])))
        self.cool_path.unlink()
        cut = _ask(self.service(six, kind="event"), ASK_ALL, 2)
        self.assertEqual(pr.FACT_GROUP_BYTES, 1000)
        self.assertLessEqual(len(cut["content"].encode("utf-8")), 1000, cut)
        self.assertLess(len(cut["records"]), 5)
        self.assertEqual([r["recordId"] for r in cut["records"]][:2], [r["recordId"] for r in full["records"]][:2])

    @staticmethod
    def patched(name, value):
        """临时改 passive_recall 里按名字导入的常量（FACT_GROUP_BYTES、STATE_SIM）。"""
        import contextlib
        @contextlib.contextmanager
        def swap():
            saved = getattr(pr, name)
            setattr(pr, name, value)
            try:
                yield
            finally:
                setattr(pr, name, saved)
        return swap()

    def test_s7_cooldown(self):
        service = self.service()
        FactCooldown(self.cool_path).mark([pr._chunk_key(self.text("f2"))], service._now())
        result = _ask(self.service(), ASK_ALL, 1)
        self.assertNotIn(self.text("f2"), result["content"], "冷却中的兄弟行不带")
        self.assertEqual(len(result["records"]), 3)
        self.cool_path.unlink()
        FactCooldown(self.cool_path).mark([pr._chunk_key(self.text(f)) for f in ("f1", "f4")], service._now())
        cooled = _ask(self.service(), ASK_ALL, 2)
        self.assertEqual(cooled["reasonCodes"], ["fact_cooled"], "前 2 名都冷却时不由兄弟行顶上")



class SnippetProvider:
    """按原文片段查表给单位向量：先整句对上，再找第一个出现在文本里的片段；都对不上就报错。记调用次数。"""
    id = "snippet"

    def __init__(self, table):
        self.table, self.calls = table, 0

    def hit_floor(self):
        return 0.3

    def embed(self, texts, is_query=False):
        self.calls += 1
        out = []
        for text in texts:
            vec = self.table.get(text) or next(v for k, v in self.table.items() if k in text)
            norm = sum(x * x for x in vec) ** 0.5
            out.append([x / norm for x in vec])
        return out


def _pad(vec):
    return tuple(vec) + (0, 0)


# #45 第 2 步夹具：f1～f4 写成 latent_append 记录。R1 与 R4 余弦约 0.68，R2、R3 与谁都是 0。
HINT_TABLE = {**{text: _pad(vec) for _f, _b, _d, text, vec in KAOYAN},
              ASK_ALL: _pad((3, 0.6, 0.2, 0.4, 0.5)),
              "冲刺校是 B 校": (2, 1, 1, 1, 0, 0, 0), "改主意": (2, 0, 0, 0, 1, 0, 0),
              "会发光的球": (0, 0, 0, 0, 0, 1, 0), "备份脚本": (0, 0, 0, 0, 0, 0, 1)}
HINT_RECORDS = [
    ("R1", (2026, 1, 1), "阿离定了考研志愿：冲刺校是 B 校，稳妥校是 C 校、D 校，保底校是 A 校。",
     "志愿已定：冲刺 B、稳妥 C 和 D、保底 A", ["f1", "f2", "f3"]),
    ("R2", (2026, 2, 1), "阿离给猫买了一个会发光的球，猫不理它。", "球放在客厅，猫还是不理", []),
    ("R3", (2026, 3, 1), "阿离把备份脚本改成每半小时拉一次，保留最新三份。", "已改好，在跑", []),
    ("R4", (2026, 6, 1), "阿离改主意了，考研只报 E 校，B、C、D、A 都不报了。", "考研只报 E 校", ["f4"]),
]
HINT_ENV = {"LATENT_SUPERSEDE_HINT_MIN": "0.6", "LATENT_SUPERSEDE_HINT_LEAD": "0.2"}


def _epoch(ymd, hour=10):
    return dt.datetime(*ymd, hour, tzinfo=dt.timezone(dt.timedelta(hours=8))).timestamp()


class SupersedeHintTests(unittest.TestCase):
    """#45 第 2 步判据 H1～H5、L1～L4、C1、T1（见任务卡第三节）；T2 在 mcp_server.py 自检里。"""

    def setUp(self):
        try:
            import mcp_server
        except SyntaxError:
            self.skipTest("mcp_server 需要 Python 3.12+")
        self.mcp = mcp_server
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.corpus, self.facts = base / "timeline", base / "事实库"
        self.log = base / "state" / "hints.jsonl"
        self.corpus.mkdir()
        self.facts.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, hint=True):
        """按日期依次写 R1～R4，返回 (server, {名字: recordId}, {名字: 回执})。"""
        from unittest import mock
        from memory_retrieval import MemoryIndex, load_corpus
        from passive_facts import FactIndex
        self.provider = SnippetProvider(HINT_TABLE)
        env = {"LATENT_PASSIVE_FACTS": str(self.facts),
               "LATENT_PASSIVE_FACT_STATE_DIR": str(Path(self.tmp.name) / "state"),
               "LATENT_SUPERSEDE_HINT_LOG": str(self.log), **(HINT_ENV if hint else {})}
        with mock.patch.dict(os.environ, env):
            if not hint:
                for key in HINT_ENV:
                    os.environ.pop(key, None)
            srv = self.mcp.MemoryServer(
                index=MemoryIndex(embed=True, provider=self.provider).build(),
                corpus_dir=str(self.corpus), enable_passive_recall=True,
                loader=lambda: load_corpus(str(self.corpus), embed=True, provider=self.provider,
                                           cache_path=""))
        ids, receipts = {}, {}
        facts = {fid: text for fid, _b, _d, text, _v in KAOYAN}
        for name, ymd, text, state, fids in HINT_RECORDS:
            args = {"text": text, "current_state": state}
            if fids:
                args["facts"] = [{"fact": facts[f], "kind": "state", "event_date": "%04d-%02d-%02d" % ymd}
                                 for f in fids]
            if name == "R3":    # 带一条索引摘要，L3 要拿它当“索引摘要不算 timeline 记录”的反例
                args["indexEvidence"] = [{"type": "event", "quote": "把备份脚本改成每半小时拉一次"}]
            receipts[name] = srv._tool_memory_append(args, now=_epoch(ymd))
            ids[name] = re.search(r"recordId=([0-9a-f]{16})", receipts[name]).group(1)
        # 事实库是启动时建的空库，增量靠后台按分钟重读；这里直接换成同步建好的那份，只为夹具确定
        srv.passive.fact_index = FactIndex(self.facts, provider=self.provider)
        return srv, ids, receipts

    def ask(self, srv, ymd, n):
        srv.index.fixed_now = _epoch(ymd, 12)
        return _ask(srv.passive, ASK_ALL, n, session=f"s{n}")

    def test_h1_h2_c1_hint_lists_only_the_old_state(self):
        srv, ids, receipts = self.build()
        self.assertIn("supersedeHint：库里有 1 条现行记录", receipts["R4"])
        self.assertIn(f"recordId={ids['R1']}（2026-01-01，当时的状态：志愿已定：冲刺 B、稳妥 C 和 D、保底 A）",
                      receipts["R4"])
        self.assertIn(f"supersedes＝那条的 recordId，by={ids['R4']}，不带 text；不是就什么都不用做", receipts["R4"])
        for other in ("R2", "R3"):
            self.assertNotIn(ids[other] + "（", receipts["R4"], "只列 1 条")
        for name in ("R1", "R2", "R3"):
            self.assertNotIn("supersedeHint", receipts[name], f"{name} 不相干，不提示")
        rows = [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([(r["record"], r["candidates"]) for r in rows], [(ids["R4"], [ids["R1"]])])
        self.assertEqual(self.mcp.supersede_hint_stats(self.log), {"hinted": 1, "linked": 0, "ignored": 1})
        self.assertIsNone(re.search(r"[一-鿿]", self.log.read_text(encoding="utf-8")), "账本不落正文")

    def test_h3_off_by_default(self):
        srv, ids, receipts = self.build(hint=False)
        self.assertNotIn("supersedeHint", receipts["R4"])
        self.assertFalse(self.log.exists(), "没开提示就不生成账本")
        default = self.mcp.MemoryServer(index=srv.index, corpus_dir=str(self.corpus)).hint_log_path
        self.assertFalse(default.resolve().is_relative_to(self.corpus.resolve().parent), "账本默认不在语料旁")

    def test_h4_no_extra_embedding_calls(self):
        srv, ids, _ = self.build()
        before = self.provider.calls
        self.assertIn(ids["R1"], srv._supersede_hint(ids["R4"], now=_epoch((2026, 6, 1))))
        self.assertEqual(self.provider.calls, before, "算提示只用现成向量")

    def test_h5_pick_rule(self):
        from memory_retrieval import pick_supersede_hint
        self.assertEqual(pick_supersede_hint([(0.9, 1), (0.6, 2), (0.5, 3)], 0.6, 0.2), [(0.9, 1)])
        self.assertEqual(pick_supersede_hint([(0.9, 1), (0.85, 2), (0.5, 3)], 0.6, 0.2), [(0.9, 1), (0.85, 2)])
        self.assertEqual(pick_supersede_hint([(0.9, 1), (0.85, 2), (0.8, 3)], 0.6, 0.2), [], "分不开就不提")
        self.assertEqual(pick_supersede_hint([(0.55, 1)], 0.6, 0.2), [], "不过线不提")
        self.assertEqual(pick_supersede_hint([(0.7, 1)], 0.6, 0.2), [(0.7, 1)], "只有一名时第二名按 0 算")
        self.assertEqual(pick_supersede_hint([], 0.6, 0.2), [])

    def test_l1_l2_c1_link_only(self):
        srv, ids, _ = self.build()
        before = self.ask(srv, (2026, 7, 1), 1)
        self.assertIn(KAOYAN[0][3], before["content"], "补链前 f1 会递")
        self.assertIn("冲刺校是 B 校", srv._tool_memory_search({"query": "冲刺校是 B 校"})["text"])
        md_before = {p: p.read_bytes() for p in self.corpus.rglob("*.md")}
        chunks_before = len(srv.index.chunks)
        signature = self.mcp._corpus_signature(srv.source_dirs)
        reloads = []
        srv._reload_from_disk = lambda: reloads.append(1)
        out = srv._call_tool(1, {"name": "latent_supersede",
                                 "arguments": {"supersedes": ids["R1"], "by": ids["R4"]}})["result"]
        self.assertFalse(out["isError"], out)
        self.assertIn("只写了取代账本，没有写新正文", out["content"][0]["text"])
        self.assertEqual({p: p.read_bytes() for p in self.corpus.rglob("*.md")}, md_before, "不产生新正文块")
        self.assertEqual((len(srv.index.chunks), reloads), (chunks_before, []), "块数不变、不整库重建")
        self.assertEqual(self.mcp._corpus_signature(srv.source_dirs), signature)
        r1 = next(m for m in srv.index.meta if m.get("record_id") == ids["R1"])
        self.assertEqual(r1["status"], "superseded")
        after = self.ask(srv, (2026, 7, 2), 2)        # 隔天再问，前一轮的冷却已过
        for fid, _b, _d, text, _v in KAOYAN:
            (self.assertIn if fid == "f4" else self.assertNotIn)(text, after["content"], fid)
        try:
            search = srv._tool_memory_search({"query": "冲刺校是 B 校"})["text"]
        except self.mcp.ToolError:
            search = ""
        self.assertNotIn("冲刺校是 B 校", search)
        self.assertEqual(self.mcp.supersede_hint_stats(self.log), {"hinted": 1, "linked": 1, "ignored": 0})

    def test_h6_superseded_record_no_longer_a_candidate(self):
        """判据写定后补的：第七节写了候选池去掉已被取代的记录，判据里漏了这条（变异“候选池不去掉已被取代的”
        只有这条会红）。补链后再写一条和 R1 一样的，提示只能指向现行的 R4，不能指回已退场的 R1。"""
        srv, ids, _ = self.build()
        srv._tool_memory_supersede({"supersedes": ids["R1"], "by": ids["R4"]})
        out = srv._tool_memory_append({"text": "阿离又提起冲刺校是 B 校的事。", "current_state": "只是聊起"},
                                      now=_epoch((2026, 6, 2)))
        self.assertIn(f"recordId={ids['R4']}（", out)
        self.assertNotIn(ids["R1"], out)

    def test_h7_same_day_records_are_not_candidates(self):
        """与新记录同一天（记录自己的日期，东八区自然日）的记录不进候选。实测标定按 5% 选出的线
        只提示 3 条，第一名都和新记录同一天，多半是同一件事接着写，不是旧状态被取代。"""
        srv, ids, _ = self.build()

        def pool(rid):
            idx = next(i for i, m in enumerate(srv.index.meta)
                       if m.get("layer", "timeline") == "timeline" and m.get("record_id") == rid)
            return {srv.index.meta[i].get("record_id") for i in srv.index.supersede_pool(idx)}

        def write(text, when):
            out = srv._tool_memory_append({"text": text, "current_state": "接着说"}, now=when)
            return re.search(r"recordId=([0-9a-f]{16})", out).group(1), out

        # 同一天：R4（6 月 1 日 10 点）当晚又写一条几乎一样的（余弦 1.0），R4 不进候选；更早的 R1 照常进、照常提示
        r5, out = write("阿离改主意了，晚上又说了一遍只报 E 校。", _epoch((2026, 6, 1), 22))
        self.assertNotIn(ids["R4"], pool(r5))
        self.assertIn(ids["R1"], pool(r5))
        self.assertIn(f"recordId={ids['R1']}（", out)
        self.assertNotIn(ids["R4"], out)
        # 前一天的照常进
        r6, _ = write("阿离改主意了，第二天又确认只报 E 校。", _epoch((2026, 6, 2), 10))
        self.assertLessEqual({ids["R4"], r5}, pool(r6))
        # 跨午夜：两对都选在 UTC 日期与东八区日期不一致的时刻，按 UTC 算日期两条都会判反
        late, _ = write("会发光的球滚到了沙发底下。", _epoch((2026, 6, 10), 23) + 50 * 60)
        early, _ = write("会发光的球又被猫推出来了。", _epoch((2026, 6, 11), 0) + 10 * 60)
        self.assertIn(late, pool(early), "前一天 23:50 的照常进（UTC 下两条同一天）")
        night, _ = write("会发光的球被收进了柜子。", _epoch((2026, 6, 11), 23) + 50 * 60)
        self.assertNotIn(early, pool(night), "同一天 00:10 的不进（UTC 下两条不同天）")

    def test_l3_bad_args_are_fixable_and_write_nothing(self):
        srv, ids, _ = self.build()
        srv._tool_memory_supersede({"supersedes": ids["R1"], "by": ids["R4"]})
        index_chunk = next(c for c, m in zip(srv.index.chunks, srv.index.meta) if m.get("layer") == "index")
        r1, r2, r4 = ids["R1"], ids["R2"], ids["R4"]
        cases = [
            ({"supersedes": r2, "by": r4, "text": "又写一遍"}, "去掉这些字段"),
            ({"supersedes": r2, "by": r4, "current_state": "又写一遍"}, "去掉这些字段"),
            ({"supersedes": r2}, "就带 by＝新记录的 recordId"),
            ({"supersedes": r2, "by": "R4"}, "16 位小写十六进制 recordId"),
            ({"supersedes": "0" * 16, "by": r4}, "请先 latent_search 核对"),
            ({"supersedes": r2, "by": "0" * 16}, "by 要填 latent_append"),
            ({"supersedes": r2, "by": pr._chunk_key(index_chunk)}, "索引摘要不算"),
            ({"supersedes": r2, "by": r2}, "是不是填重了"),
            ({"supersedes": r1, "by": r2}, "请沿 superseded_by 使用当前链尾"),
            ({"supersedes": r2, "by": r4}, "请把它接在链尾上"),
            ({"supersedes": r4, "by": r1}, "请核对两条哪条是新的"),
        ]
        ledger = (self.corpus / ".supersessions.json").read_bytes()
        hints = self.log.read_bytes()
        for args, fix in cases:
            out = srv._call_tool(1, {"name": "latent_supersede", "arguments": args})["result"]
            self.assertTrue(out["isError"], args)
            self.assertIn(fix, out["content"][0]["text"], args)
        self.assertEqual(((self.corpus / ".supersessions.json").read_bytes(), self.log.read_bytes()),
                         (ledger, hints), "报错时什么都不写")

    def test_l4_preflight_writes_nothing(self):
        srv, ids, _ = self.build()
        out = srv._tool_memory_supersede({"mode": "preflight", "supersedes": ids["R1"], "by": ids["R4"]})
        self.assertIn("预检通过", out)
        self.assertFalse((self.corpus / ".supersessions.json").exists())
        self.assertEqual(self.mcp.supersede_hint_stats(self.log)["linked"], 0)

    def test_t1_tool_counts_unchanged(self):
        from unittest import mock
        from memory_retrieval import MemoryIndex
        with mock.patch.dict(os.environ, {"LATENT_FACT_BACKFILL": "off",
                                          "LATENT_PASSIVE_FACTS": str(self.facts),
                                          "LATENT_PASSIVE_FACT_STATE_DIR": str(Path(self.tmp.name) / "state")}):
            srv = self.mcp.MemoryServer(index=MemoryIndex().build(), corpus_dir=str(self.corpus),
                                        enable_passive_recall=True)

        def listed(ok):
            return [t["name"] for t in srv.handle(
                {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, hidden_ok=ok)["result"]["tools"]]
        names = [t["name"] for t in self.mcp.TOOLS]
        self.assertEqual(listed(False), names)
        self.assertEqual(listed(True), names + [self.mcp.PASSIVE_RECALL_TOOL])
        self.assertEqual((len(listed(False)), len(listed(True))), (8, 9))


class FactFloorCalibrationTests(unittest.TestCase):
    def test_tool_selftest(self):
        """tests/fact_floor_calibration.py 随包；它的自检（判据 1a～1c）跟着发布检查跑，不靠人记得单独跑。"""
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import fact_floor_calibration
        fact_floor_calibration.selftest()


if __name__ == "__main__":
    unittest.main()
