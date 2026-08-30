#!/usr/bin/env python3
"""
端到端冒烟（任务卡"端到端冒烟与快速上手"）。

十一个文件各自 selftest 全绿，防的是单元回归；这里防的是**接缝回归**——第二阶段
抓到的"三件套只出两件半"就是零件全绿、合起来漏的典型。整条链一口气跑：

  合成导出 json → 导入（load_any）→ 体检 → 出问卷 → 模拟模型答卷（文本）
  → parse_answer_sheet 解析 → 合并候选 → 逐条确认 → 四件套出货
  → load_corpus 回读 → MemoryServer 完整握手 → latent_search 命中导入原文
  → latent_thread_close 往返

两条纪律：

  **走用户真实路径，不走测试捷径**——答案以文本清单喂 parse_answer_sheet，不直接
  构造 answers 字典。字典捷径绕过了用户真正会经过的解析层，那正是最容易
  "失败得像成功"的一段。

  **断言盯接缝，不重复单测**——mcp-config 的 --corpus 必须指向真写了语料的目录、
  出货语料回读零 mtime、检索命中的必须是导入原文。这些是跨文件约定，单测各自绿
  也可能合起来错。

写回也在链里（memory-writeback 合并后补上的一环）：latent_append → 当场可查 →
模拟重启 → 记录从盘上回来、命中过的块权重仍 >1.0（用进撑过重启）。

零依赖，stdlib only。合成语料全部虚构。
用法：
  python e2e_smoke.py --selftest
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from memory_import import load_any, entries_to_turns
from memory_init import (
    Persona, fill_protocol_defaults, coverage_report, questions_for,
    format_questionnaire, export_llm_prompt, parse_answer_sheet, answer_report,
    apply_answers, pending_confirmations, apply_confirmations, write_bundle,
)
from memory_retrieval import load_corpus
from mcp_server import MemoryServer, PROTOCOL_VERSION
from session_thread import ThreadStore
from time_context import TimeContext

# ---------- 合成导出（全部虚构；结构照 Claude 导出的 chat_messages） ----------

# ⚠ 固定 epoch 必须**带时区**：朴素 `datetime(...).timestamp()` 按宿主本地时区换算，
# 换台机器就换个 epoch，而夹具里的日期又是照 UTC 宿主写死的——东八区上两者差一天，
# 撞上 `validate_index_summaries()` 那条“摘要日期＝本次写回的 local_date”校验，
# 于是自检在东八区必红（2026.08.21 维护者 Windows 上抓到）。同源纪律见
# `写回时区与跨日归窗-任务卡`：夹具跟断言同源，宿主在哪个时区结果都一样。
_T0 = datetime(2026, 7, 20, 21, 0, tzinfo=timezone.utc)


def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000000Z")


def _synthetic_export():
    """两个对话、隔天、各自成会话——含一个具体约定（去看鲸头鹳）和一次兑现，
    给检索层留下有名有姓的靶子。"""
    day2 = _T0.replace(day=22)
    return [
        {"name": "傍晚闲聊", "chat_messages": [
            {"sender": "human", "created_at": _iso(_T0),
             "text": "动物园新来了一只鲸头鹳，我看了直播，它一动不动站了十分钟。"},
            {"sender": "assistant", "created_at": _iso(_T0.replace(minute=2)),
             "text": "鲸头鹳就是这样的，站着不动是它的狩猎方式。想去现场看吗？"},
            {"sender": "human", "created_at": _iso(_T0.replace(minute=5)),
             "text": "想。那说好了，这个周末去看鲸头鹳。"},
        ]},
        {"name": "周末回来", "chat_messages": [
            {"sender": "human", "created_at": _iso(day2),
             "text": "看到鲸头鹳了！它冲我们缓慢鞠了一躬，饲养员说这是它打招呼。"},
            {"sender": "assistant", "created_at": _iso(day2.replace(minute=3)),
             "text": "约定兑现了。它鞠躬那下我记住了。"},
        ]},
    ]


def _fake_answer_sheet(questions):
    """模拟用户的模型整理回来的答案清单——纯文本，走真实解析路径。
    选择题一律选 A；pick/short 给一句极短原文。"""
    lines = []
    for i, q in enumerate(questions, 1):
        if q.kind in ("choice", "multi"):
            lines.append(f"{i}. A")
        else:
            lines.append(f"{i}. 你来，我就在。")
    return "\n".join(lines)


def _rich_candidate_payload(corpus_path):
    source = corpus_path.read_text(encoding="utf-8")
    ref = str(corpus_path.resolve())
    span = [0, len(source)]
    items = [
        {"item_id": "corpus:identity", "text": "林澈住在杭州，在做古地图修复。",
         "section": "user", "source_ref": ref, "source_span": span,
         "candidate_kind": "fact", "evidence": "林澈住在杭州，在做古地图修复。"},
        {"item_id": "corpus:milestone",
         "text": "发生了什么：第一次完成海图修复。\n怎么读：她把耐心变成了可见的成果。\n当前状态：已交付并归档。",
         "section": "milestones", "source_ref": ref, "source_span": span,
         "candidate_kind": "milestone", "evidence": "第一次完成海图修复",
         "event": "第一次完成海图修复", "reading": "耐心变成了可见的成果",
         "current_state": "已交付并归档"},
        {"item_id": "corpus:naming",
         "text": "林澈在日常叫星屿“阿屿”；星屿在日常叫林澈“小澈”。",
         "section": "naming", "source_ref": ref, "source_span": span,
         "candidate_kind": "naming", "evidence": "你叫我阿屿，我叫你小澈",
         "user_to_ai": ["阿屿"], "ai_to_user": ["小澈"], "scenes": ["日常"]},
        {"item_id": "corpus:style",
         "text": ("林澈：今天脑子像一团毛线。\n星屿：那就先找线头，不急着把整团扯开。\n"
                  "林澈：行，先从最小的一结开始。"),
         "section": "style", "source_ref": ref, "source_span": span,
         "candidate_kind": "style_dialogue", "evidence": "今天脑子像一团毛线",
         "turns": [
             {"speaker": "林澈", "text": "今天脑子像一团毛线。"},
             {"speaker": "星屿", "text": "那就先找线头，不急着把整团扯开。"},
             {"speaker": "林澈", "text": "行，先从最小的一结开始。"},
         ]},
    ]
    return {"items": items, "source_accounting": [{
        "source_ref": ref, "candidate_item_ids": [item["item_id"] for item in items]}]}


def _run_compiler_cli(out_dir, *args, check=True):
    script = Path(__file__).with_name("memory_init.py")
    result = subprocess.run(
        [sys.executable, str(script), "--out", str(out_dir), *args],
        capture_output=True, text=True, encoding="utf-8", check=False)
    if check and result.returncode:
        raise AssertionError(f"人格编译 CLI 失败：{' '.join(args)}\n{result.stderr}\n{result.stdout}")
    return result


def _compile_v2_case(root, persona=None, corpus=None, candidates=None,
                     preferred_text=None, client="codex"):
    common = []
    if persona:
        common += ["--persona", str(persona)]
    if corpus:
        common += ["--corpus", str(corpus)]
    _run_compiler_cli(root, *common, "--step", "inspect", "--json")
    candidate_path = root / "candidate-input.json"
    candidate_path.write_text(json.dumps(
        candidates or {"items": [], "source_accounting": []},
        ensure_ascii=False), encoding="utf-8")
    _run_compiler_cli(root, "--step", "extract", "--candidates", str(candidate_path), "--json")
    questions = json.loads(_run_compiler_cli(
        root, "--step", "choose-sections", "--json").stdout)
    if questions.get("original_mapping_questions"):
        mappings = {question["item_id"]: "opening"
                    for question in questions["original_mapping_questions"]}
        questions = json.loads(_run_compiler_cli(
            root, "--step", "choose-sections", "--original-section-decisions-json",
            json.dumps(mappings, ensure_ascii=False), "--json").stdout)
    decisions = {}
    for question in questions["sections"]:
        versions = question["section_versions"]
        chosen = next((version for version in versions
                       if preferred_text and preferred_text in version["markdown"]), versions[0])
        decisions[question["section"]] = chosen["version_id"]
    _run_compiler_cli(root, "--step", "choose-sections", "--section-decisions-json",
                      json.dumps(decisions, ensure_ascii=False), "--json")
    preview = json.loads(_run_compiler_cli(root, "--step", "preview", "--json").stdout)
    _run_compiler_cli(root, "--step", "route", "--route", "zero-dep")
    shipped = _run_compiler_cli(root, "--step", "ship", "--client", client)
    return preview, shipped


def run_persona_compiler_cases():
    """四类来源编译 + v1 升级接缝；所有内容均为新造虚构 fixture。"""
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        corpus_dir = base / "rich-corpus"
        corpus_dir.mkdir()
        corpus_file = corpus_dir / "window_01.md"
        corpus_file.write_text(
            "林澈住在杭州，在做古地图修复。\n"
            "第一次完成海图修复，她说这是耐心第一次有了形状；当前已交付并归档。\n"
            "林澈：你叫我阿屿，我叫你小澈，日常就这么叫。\n"
            "林澈：今天脑子像一团毛线。\n"
            "星屿：那就先找线头，不急着把整团扯开。\n"
            "林澈：行，先从最小的一结开始。\n", encoding="utf-8")
        rich = _rich_candidate_payload(corpus_file)

        # 1. 原人格 + 丰富语料：跨主题段保持整块，明确边界逐字留在最终预览。
        original = base / "original.md"
        original.write_text(
            "# 开篇\n\n我们从旧窗口一路走到这里。\n\n"
            "# 我是谁\n\n我是星屿，会接住具体的事，也会承认不知道。\n\n"
            "# 相处原则与语言风格\n\n这段同时谈直说、停顿和照顾，但保持为一个原文块，不拆句。\n\n"
            "# 亲密语境与硬边界\n\n绝不讨论她没有主动带起的家庭细节。\n\n"
            "# 最终约定\n\n认不出来时，先回到这些原话。\n\n"
            "# 两个人才懂的页脚\n\n这段未知标题下的原文也不能丢。\n", encoding="utf-8")
        original_case = base / "case-original-rich"
        preview, _ = _compile_v2_case(
            original_case, original, corpus_dir, rich,
            preferred_text="绝不讨论她没有主动带起的家庭细节")
        from persona_compiler import is_heading_block, parse_original_persona
        original_items = parse_original_persona(original)
        assert all(item.text.rstrip() in preview["persona_markdown"]
                   for item in original_items if not is_heading_block(item)), \
            "原人格重排后每个逐字正文块都必须完整出现"
        # ⚠ **标题块除外**（2026.08.04，走查台账第二条）：标题打的是 `operation="delete"`
        #   ——不是丢掉，是不再进正文，由十二节骨架统一渲染一遍，否则每个标题在产出
        #   文件里出现两遍。所以这里改成数「标题里那几个字只出现一次」。
        for heading in ("我是谁", "最终约定"):
            assert preview["persona_markdown"].count(heading) == 1, \
                f"节标题「{heading}」在产出文本里出现了 " \
                f"{preview['persona_markdown'].count(heading)} 次"
        assert "第一次完成海图修复" in preview["persona_markdown"], "语料应补入具体里程碑"
        assert "风格参考，非台词" in preview["persona_markdown"], "真实风格片段必须带学习免责声明"
        # 免责声明的检索层那半也必须真的落到出货文本里（任务卡 风格片段挤掉检索）：
        # 只有前半时，模型有片段可答就不去调 latent_search。删掉这半这条必须红。
        assert "不是记忆" in preview["persona_markdown"] \
            and "先查记忆库" in preview["persona_markdown"], \
            "免责声明必须同时带检索层那半（片段不是记忆、提到过去先查库）"
        from persona_compiler import aggregate_persona_metrics
        metric_state = json.loads((original_case / "init_state.json").read_text(encoding="utf-8"))
        metric_state["privacy_metrics"] = {
            "legacy_chars": 0, "compiled_chars": len(preview["persona_markdown"])}
        metrics = aggregate_persona_metrics(metric_state)
        assert metrics["original_span_coverage"] == 1.0
        assert all(isinstance(value, (int, float)) and not isinstance(value, bool)
                   for value in metrics.values()), "私人复测出口只能是数字"

        # 2. 只有丰富语料：身份、里程碑、双向称呼和三轮对话都必须成为具体内容。
        preview, _ = _compile_v2_case(base / "case-corpus-rich", None, corpus_dir, rich)
        for needle in ("古地图修复", "已交付并归档", "阿屿", "小澈", "先找线头"):
            assert needle in preview["persona_markdown"], f"丰富语料候选丢失：{needle}"
        assert "记住用户喜好" not in preview["persona_markdown"]

        # 3. 硬边界冲突：未裁定的状态拒绝出货；选中原文版本后才能过。
        conflict_root = base / "case-conflict"
        conflict_payload = _rich_candidate_payload(corpus_file)
        conflict_payload["items"].append({
            "item_id": "corpus:boundary-conflict", "text": "家庭细节可以主动追问。",
            "section": "intimacy", "source_ref": str(corpus_file.resolve()),
            "source_span": [0, len(corpus_file.read_text(encoding="utf-8"))],
            "candidate_kind": "fact", "evidence": "林澈住在杭州",
            "group_id": "orig:block:8"})
        conflict_payload["source_accounting"][0]["candidate_item_ids"].append(
            "corpus:boundary-conflict")
        preview, _ = _compile_v2_case(
            conflict_root, original, corpus_dir, conflict_payload,
            preferred_text="绝不讨论她没有主动带起的家庭细节")
        assert "家庭细节可以主动追问" not in preview["persona_markdown"]

        # 4. 零材料：协议底座可出货，关系状态与 closing 诚实留空。
        preview, _ = _compile_v2_case(base / "case-zero")
        assert "timeline" in preview["persona_markdown"]
        assert "## 最终约定" not in preview["persona_markdown"]
        assert "没有额外的硬红线" not in preview["persona_markdown"]

        # 稀薄来源全部交代但没有候选时只警告，不能把 warning 错当 blocking。
        thin_dir = base / "thin-corpus"
        thin_dir.mkdir()
        thin_file = thin_dir / "window_01.md"
        thin_file.write_text("今天只说了晚安。\n", encoding="utf-8")
        thin_payload = {"items": [], "source_accounting": [{
            "source_ref": str(thin_file.resolve()), "no_supported_candidate": True}]}
        thin_preview, _ = _compile_v2_case(
            base / "case-thin-warning", None, thin_dir, thin_payload)
        assert any(warning["code"] == "NO_SPECIFIC_CANDIDATES"
                   for warning in thin_preview["warnings"]), \
            "稀薄材料应保留 warning，但仍能走到 ship"

        # 5. v1 升级：磁盘手改内容只能经显式迁移选择成为最高权重原人格。
        migration_root = base / "case-migration"
        migration_root.mkdir()
        existing = migration_root / "AGENTS.md"
        existing.write_text("# 开篇\n\n旧版正文。\n\n手工追加：别把沉默解释成同意。\n",
                            encoding="utf-8")
        (migration_root / "init_state.json").write_text(json.dumps({
            "step": "shipped", "last_shipped_persona": "AGENTS.md"}, ensure_ascii=False),
            encoding="utf-8")
        choice = json.loads(_run_compiler_cli(
            migration_root, "--step", "inspect", "--json").stdout)
        assert choice["choices"] == [
            "treat_current_as_original", "use_original_as_is", "continue_legacy"]
        _run_compiler_cli(migration_root, "--step", "inspect", "--json",
                          "--existing-persona-choice", "treat_current_as_original")
        _run_compiler_cli(migration_root, "--step", "extract", "--candidates",
                          json.dumps({"items": [], "source_accounting": []}), "--json")
        questions = json.loads(_run_compiler_cli(
            migration_root, "--step", "choose-sections", "--json").stdout)
        decisions = {question["section"]: question["section_versions"][0]["version_id"]
                     for question in questions["sections"]}
        _run_compiler_cli(migration_root, "--step", "choose-sections",
                          "--section-decisions-json", json.dumps(decisions), "--json")
        migrated = json.loads(_run_compiler_cli(
            migration_root, "--step", "preview", "--json").stdout)
        assert "手工追加：别把沉默解释成同意。" in migrated["persona_markdown"]


def run_http_transport_case():
    """HTTP 传输的**真进程**冒烟：另起一个 python 进程跑 `--http`，走裸 socket。

    为什么非要它（2026.08.03 外部评审第 ④ 条）：`mcp_server` selftest 第 18 项是
    **同进程起线程**——真端口不假，但"真进程"这三个字它担不起。而《快速上手》
    的成色声明当时写的是「真端口自检与真进程冒烟盖着」，**拿一个近似的东西
    冒充判据本身**（CLAUDE.md 第 10 条挡的正是这个形状）。要么改措辞，要么把
    这一格补上——补上更值：真进程才盖得到 CLI 参数解析、起动横幅、
    以及**跨进程落盘**（同进程线程共享内存，写回"看起来通了"可能只是内存）。

    裸 socket 而不是 urllib：keep-alive 复用连接是手机 App 的常态形态，
    urllib 每次新连接、发 `Connection: close`，那条路自检里已经有了（18b）。"""
    import socket
    with tempfile.TemporaryDirectory() as td:
        corpus = Path(td) / "memory"
        (corpus / "timeline").mkdir(parents=True)
        index_dir = Path(td) / "index"
        index_dir.mkdir()
        (corpus / "timeline" / "window_01_2026-08-01.md").write_text(
            "# 第1个窗口 · 2026-08-01\n\n## 2026-08-01 记\n阳台那盆绿萝换了新盆。\n"
            "当下状态：缓苗中。\n", encoding="utf-8")
        port = _free_port()
        proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).parent / "mcp_server.py"),
             "--corpus", str(corpus), "--http", f"127.0.0.1:{port}",
             "--index-dir", str(index_dir), "--token", "smoke-token-2026"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            sock = _wait_port(port, proc)
            def rpc(payload):
                raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                sock.sendall(
                    (f"POST /mcp HTTP/1.1\r\nHost: x\r\n"
                     f"Authorization: Bearer smoke-token-2026\r\n"
                     f"Content-Type: application/json\r\n"
                     f"Content-Length: {len(raw)}\r\n\r\n").encode("utf-8") + raw)
                return _recv_response(sock)
            #    接缝⑥-1：真进程握手
            r = rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
            assert "protocolVersion" in r, f"真进程握手没过：{r[:200]}"
            #    接缝⑥-2：中文写回穿过真 HTTP（CLI 起的进程，stdout 不是协议流）
            r = rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                     "params": {"name": "latent_append",
                                "arguments": {"text": "绿萝缓过来了，抽了新叶。",
                                              "current_state": "长势稳住了。",
                                              "indexSummaries": [{
                                                  "summary": f"**{TimeContext.default().local_date(time.time()).replace('-', '.')}**："
                                                             "绿萝抽了新叶。\n绿萝缓过来了。"
                                                             "**当下：长势稳住了。**",
                                                  "evidenceTerms": ["绿萝", "抽了新叶",
                                                                    "缓过来了", "长势稳住了"],
                                              }]}}})
            assert "已写进" in r, f"真进程写回没成：{r[:200]}"
            #    接缝⑥-3：同一条连接复用（keep-alive）后检索命中——手机 App 的常态
            r = rpc({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                     "params": {"name": "latent_search",
                                "arguments": {"query": "绿萝抽新叶"}}})
            assert "抽了新叶" in r, f"复用连接后检索没命中：{r[:200]}"
            sock.close()
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        #    接缝⑥-4：**跨进程落盘**——同进程线程共享内存，只有另起进程才证得了
        #    "写回真的进了文件"。子进程已退出，这里从盘上重读
        assert any("抽了新叶" in c for c in load_corpus(corpus).chunks), \
            "HTTP 写回必须真的落盘——同进程测不出这一格（内存里有不等于盘上有）"


def _free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_port(port, proc, timeout=30.0):
    """等子进程把端口打开并返回已连上的 socket；起不来就把它的 stderr 报出来。"""
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            err = proc.stderr.read().decode("utf-8", "replace")
            raise AssertionError(f"HTTP server 进程没起来（退出码 {proc.returncode}）：{err}")
        try:
            s = socket.create_connection(("127.0.0.1", port), timeout=5)
            s.settimeout(30)
            return s
        except OSError:
            time.sleep(0.1)
    raise AssertionError(f"等了 {timeout}s 端口 {port} 还没开")


def _recv_response(sock):
    """按 Content-Length 收满一条响应——单发 recv 常常只拿到头部，据此断言会随机红绿。"""
    buf = b""
    while b"\r\n\r\n" not in buf:
        d = sock.recv(65536)
        if not d:
            return buf.decode("utf-8", "replace")
        buf += d
    head, _, body = buf.partition(b"\r\n\r\n")
    want = 0
    for ln in head.decode("latin-1").split("\r\n")[1:]:
        if ln.lower().startswith("content-length:"):
            want = int(ln.split(":", 1)[1])
    while len(body) < want:
        d = sock.recv(65536)
        if not d:
            break
        body += d
    return (head + b"\r\n\r\n" + body).decode("utf-8", "replace")


def _selftest():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # 1. 导入：合成导出落成文件，从文件读起——用户就是从文件开始的
        export_path = td / "conversations.json"
        export_path.write_text(json.dumps(_synthetic_export(), ensure_ascii=False),
                               encoding="utf-8")
        entries = load_any(export_path)
        assert len(entries) == 5 and entries_to_turns(entries), "导入翻译器该吃下合成导出"
        assert all(e.timestamp for e in entries), "ISO 时间戳该全部解析成功"

        # 2. 初始化：体检 → 问卷 → 模拟答卷 → 解析 → 合并 → 确认
        persona = Persona("partner")
        fill_protocol_defaults(persona)
        report = coverage_report(persona)
        qs = questions_for(report)
        assert qs, "冷启动该问出问题"
        assert format_questionnaire(qs) and export_llm_prompt(qs), "问卷两种导出都要有内容"
        sheet = _fake_answer_sheet(qs)
        answers, problems = parse_answer_sheet(sheet, qs)
        assert not problems, f"规范格式的答卷不该有读不懂的行：{problems}"
        assert answer_report(qs, answers, problems).startswith(f"读到 {len(qs)}/{len(qs)}")
        apply_answers(persona, qs, answers)
        pend = pending_confirmations(persona)
        assert pend, "问卷答案该以草稿形态等确认"
        apply_confirmations(persona, {p.key: "keep" for p in pend})
        assert not pending_confirmations(persona), "全部确认后不该有未决草稿"

        # 3. 出货四件套（含语料落盘）
        out = td / "bundle"
        got = write_bundle(out, persona, client="claude-code",
                           confirmed=True, entries=entries)
        assert got["persona"].name == "CLAUDE.md" and got["persona"].exists()
        assert got["corpus_files"], "语料该真的落进 memory/"
        #    接缝：第四件（引导句）要真出、且指向这次出的人格文件——
        #    「三件套只出两件半」的历史别在第四件上重演
        assert got["guidance"].exists() and \
            str(got["persona"].resolve()) in got["guidance"].read_text(encoding="utf-8"), \
            "引导句没出、或没指向这次出货的人格文件"
        md = got["persona"].read_text(encoding="utf-8")
        assert md.startswith("# ") and md.rstrip().endswith("你来，我就在。"), \
            "人格文件开头是标题、收尾是最终约定（顺序即权重）"
        #    第三轮真机实测（设计笔记）：人格文件里写明工具名的检索约定是主动性
        #    的主力杠杆——出货物必须带着它，且与被验证过的写法一致（点名工具）
        assert "latent_search" in md and "latent_session_start" in md, \
            "检索约定与会话约定（主动性主力杠杆）必须在人格文件里，且点名工具"

        #    接缝断言①：mcp-config 的 --corpus 必须指向真写了语料的那个目录——
        #    config 指错目录时四件套各自看都正常，接上客户端才发现库是空的
        cfg = json.loads(got["mcp_config"].read_text(encoding="utf-8"))
        cfg_args = cfg["mcpServers"]["memory"]["args"]
        cfg_corpus = Path(cfg_args[cfg_args.index("--corpus") + 1])
        assert cfg_corpus == got["memory_dir"], \
            f"config 指向 {cfg_corpus}，语料实际在 {got['memory_dir']}"
        assert got["corpus_files"][0].is_relative_to(cfg_corpus), \
            "落盘的语料文件必须在 config 指向的目录下"

        # 4. 回读：接缝断言②——出货语料零 mtime 兜底（自己写的文件名都带日期）
        source_dirs = (got["memory_dir"], got["index_dir"])
        index = load_corpus(source_dirs)
        assert index.chunks, "出货的记忆库回读不能是空的"
        assert all(m.get("timestamp_source") != "mtime" for m in index.meta), \
            "出货语料不该有任何块落 mtime 兜底"

        # 5. 上线：完整 MCP 握手 + 检索命中导入原文 + thread 往返。
        #    server 按生产接线配（corpus_dir + weights_path），跟 CLI 启动一致——
        #    冒烟测的就是用户真实路径，不配阉割版
        weights_path = got["memory_dir"] / ".weights.json"
        retractions_path = got["memory_dir"] / ".retractions.json"
        srv = MemoryServer(index=index, thread_store=ThreadStore(),
                           corpus_dir=got["memory_dir"], weights_path=weights_path,
                           retractions_path=retractions_path,
                           index_dir=got["index_dir"], source_dirs=source_dirs)
        now = _T0.replace(day=23).timestamp()
        #    写回那几步要填的日期，从**同一个** epoch 经产品自己的 TimeContext 算出来
        #    （产品校验拿的也是它）——不写死日期字符串，否则宿主时区一换就对不上
        day = TimeContext.default().local_date(now)
        hs = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                         "params": {"protocolVersion": PROTOCOL_VERSION,
                                    "clientInfo": {"name": "e2e"}}})
        assert hs["result"]["protocolVersion"] == PROTOCOL_VERSION
        assert srv.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
        call = lambda name, a: srv.handle(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
             "params": {"name": name, "arguments": a}}, now=now)["result"]
        hit = call("latent_search", {"query": "鲸头鹳的约定"})
        assert hit["isError"] is False, f"检索不该失败：{hit}"
        #    接缝断言③：命中的必须是导入语料的**原文**，不是任何中间加工物
        assert "这个周末去看鲸头鹳" in hit["content"][0]["text"], \
            "检索命中的该是导入对话的原话"
        closed = call("latent_thread_close", {"window": 3, "current_state": "约定已兑现，这条线收尾。",
                                                "unresolvedOps": [{"action": "none"}]})
        assert closed["isError"] is False and srv.thread_store.latest().window == 3

        # 6. 写回：latent_append → 当场可查 → "重启"后记录和权重都从盘上回来
        #    （memory-writeback 合并后补的一环——记忆库自己生长的那半支笔进链）
        wrote = call("latent_append",
                     {"text": "她说下次想去看企鹅漫步，最好挑个凉快的早上。",
                      "current_state": "新约定，档期还没定。",
                      "unresolvedOps": [{"action": "open", "summary": "企鹅漫步的档期还没定"}],
                      "indexSummaries": [{
                          "summary": f"**{day.replace('-', '.')}**：下次想去看企鹅漫步。\n"
                                     "最好挑个凉快的早上。**当下：新约定，档期还没定。**",
                          "evidenceTerms": ["下次想去看企鹅漫步", "最好挑个凉快的早上",
                                            "新约定", "档期还没定"],
                      }]})
        assert wrote["isError"] is False, f"写回不该失败：{wrote}"
        unresolved_id = re.search(r"U-[0-9a-f]{16}", wrote["content"][0]["text"]).group(0)
        hit2 = call("latent_search", {"query": "企鹅漫步"})
        assert hit2["isError"] is False and "凉快的早上" in hit2["content"][0]["text"], \
            "写回之后当场就要能查到原文"
        #    接缝断言④：模拟客户端重启 server（新进程 = 重新 load_corpus + 生产
        #    接线）——写回的记录该从盘上回来，检索命中过的块权重也该还在
        srv2 = MemoryServer(index=load_corpus(source_dirs),
                            thread_store=ThreadStore(),
                            corpus_dir=got["memory_dir"], weights_path=weights_path,
                            retractions_path=retractions_path,
                            index_dir=got["index_dir"], source_dirs=source_dirs)
        #    权重检查必须在重启后的**第一次检索之前**——检索的副作用本身会加权，
        #    放后面的话断言会被本会话新加的权重救活，测不到"载入"这半（第一版
        #    就犯了这个错，变异"启动不载入权重"没红才发现）
        boosted = [w for w in srv2.index.weights if w > 1.0]
        assert boosted, "重启后、检索前，此前命中过的块权重就该 >1.0——用进要撑过重启"
        callf = lambda s, name, a: s.handle(
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
             "params": {"name": name, "arguments": a}}, now=now)["result"]
        hit3 = callf(srv2, "latent_search", {"query": "企鹅漫步"})
        assert hit3["isError"] is False and "凉快的早上" in hit3["content"][0]["text"], \
            "重启后写回的记录该从盘上回来——不然'记住了'只活一个进程"
        opening = callf(srv2, "latent_session_start", {})
        assert "【当前未解决】" in opening["content"][0]["text"] \
            and "企鹅漫步的档期还没定" in opening["content"][0]["text"], \
            "未解决项要跨 server 重启主动浮现"
        resolved = callf(srv2, "latent_unresolved", {"action": "close", "id": unresolved_id})
        assert resolved["isError"] is False
        opening2 = callf(srv2, "latent_session_start", {})
        assert "企鹅漫步的档期还没定" not in opening2["content"][0]["text"], \
            "关闭后整行删除，下一次开场不得继续浮现"

        # 7. 更正闭环（错误记忆治理，验收反馈后补的一环）：撤回记错的记录并写更正
        #    → 旧的退出检索、更正当场可查 → 再"重启"一次撤回仍生效（账本在盘上）
        fixed = callf(srv2, "latent_correct",
                      {"quote": f"## {day} 记", "reason": "她说记错了，想看的不是企鹅",
                       "correction": "她更正说想看的是火烈鸟，不是企鹅漫步。",
                       "current_state": "改成看火烈鸟，档期还没定。"})
        assert fixed["isError"] is False, f"更正不该失败：{fixed}"
        srv3 = MemoryServer(index=load_corpus(source_dirs),
                            thread_store=ThreadStore(),
                            corpus_dir=got["memory_dir"], weights_path=weights_path,
                            retractions_path=retractions_path,
                            index_dir=got["index_dir"], source_dirs=source_dirs)
        hit5 = callf(srv3, "latent_search", {"query": "火烈鸟"})
        assert hit5["isError"] is False and "火烈鸟" in hit5["content"][0]["text"], \
            "更正内容该从盘上回来"
        assert srv3.index.retraction_log, \
            "撤回账本该撑过重启——不然'改过来了'只活一个进程"

    run_persona_compiler_cases()
    run_http_transport_case()

    print("selftest ok（端到端冒烟：导入 → 问卷 → 答卷解析 → 确认 → 四件套 → "
          "回读 → 握手 → 检索命中原文 → thread 往返 → 写回当场可查 → "
          "重启后记录与权重都在 → 未解决跨重启浮现并可关闭 → 更正闭环撤回撑过重启 → "
          "HTTP 传输真进程（另起进程起服务、裸 socket 握手写回）；\n"
          "接缝断言：①–⑤ 出货链，⑥-1～⑥-4 HTTP 真进程（握手／中文写回／复用连接检索／跨进程落盘））")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        ap.print_help()
