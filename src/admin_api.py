#!/usr/bin/env python3
"""
给人看的只读口子（记忆可视化"山屋"用）：`/admin/api/...`。

为什么单独一个口子：`/mcp` 是给模型的工具口，页面拿不到"列出全部记忆""看统计"。
这里只读、单独一把钥匙（`--admin-token`）、默认关；不进工具表，模型不知道它在。
HTTP 那层（鉴权、Origin）在 mcp_server.make_http_server 里，本文件只管"问什么、回什么"。
唯一能写的是贴便条（POST /notes）：对方写给 TA 的话，改不改正文由 TA 下次开场看到后自己定。

每个参数都有上限，超了回 400：一个请求不许把建库拖死。

零依赖，stdlib only。用法：python admin_api.py --selftest
"""

import json
import random
import re
import sys
import threading
import time
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

from memory_retrieval import _chunk_key
from notes_state import NoteRequestError, NoteStoreError
from session_recall import read_recall_log, RECALL_LOG_FILENAME

MAX_N = 50            # top / random
MAX_RECALL_DAYS = 30
MAX_SPAN_DAYS = 90    # days 的 from..to
MAX_Q = 200           # search 的 q
LIST_TEXT = 600       # 列表里正文截断；要全文走 /records/<id>
HEALTH_TTL = 300      # 体检要建库，结果缓存五分钟


class BadRequest(ValueError):
    pass


def _int(q, key, default, lo, hi):
    raw = q.get(key, [None])[0]
    if raw is None:
        return default
    try:
        v = int(raw)
    except ValueError:
        raise BadRequest(f"{key} 要是整数")
    if not lo <= v <= hi:
        raise BadRequest(f"{key} 要在 {lo}～{hi} 之间")
    return v


def _day(q, key, default=None):
    raw = q.get(key, [None])[0]
    if raw is None:
        return default
    try:
        return date.fromisoformat(raw)
    except ValueError:
        raise BadRequest(f"{key} 要是 YYYY-MM-DD")


def milestones_section(text):
    """人格文件里"里程碑"那一节：从标题含"里程碑"的那行，到下一个同级或更高级标题为止。
    节外的一个字都不回（人格文件别的节可能很私人）。"""
    lines = text.splitlines()
    start = level = None
    for i, line in enumerate(lines):
        m = re.match(r"^(#{1,6})\s", line)
        if start is None:
            if m and "里程碑" in line:
                start, level = i, len(m.group(1))
        elif m and len(m.group(1)) <= level:
            return "\n".join(lines[start:i]).strip()
    return "\n".join(lines[start:]).strip() if start is not None else ""


def milestone_items(section):
    """节里一条一条：`**名字 · 时间**：正文` 或 `- **名字**：正文` 这两种写法都认。"""
    items = []
    for m in re.finditer(r"^(?:-\s*)?\*\*(.+?)\*\*[：:]?\s*(.*?)(?=^\s*(?:-\s*)?\*\*|\Z)", section, re.M | re.S):
        items.append({"title": m.group(1).strip(), "body": m.group(2).strip()})
    return items


def display_title(heading, text):
    """给页面看的一行标题：有像样的小标题就用；"当下状态"这类通用标题、索引行没标题，
    就取正文第一句（索引行取"事件："后面那句），去掉 markdown 记号。"""
    h = (heading or "").strip()
    if h and h not in ("当下状态", "当下", "#PREAMBLE") and not re.match(r"window_\d+", h):
        return h
    body = re.sub(r"^#+ .*$", "", text, flags=re.M)
    m = re.search(r"事件：\*\*\s*(.+?)(?:；|\n|$)", body)
    s = m.group(1) if m else body.strip()
    s = re.sub(r"[*`#>]|^\s*-\s*", "", s.split("\n")[0]).strip()
    return re.split(r"[。；！？]", s)[0][:28]


class AdminAPI:
    """server：mcp_server.MemoryServer；lock：跟工具调用共用的那把（索引不是线程安全的）。"""

    def __init__(self, server, lock=None, persona_path=None, diagnose=None, fact_path=None):
        self.server = server
        self.lock = lock or threading.Lock()
        self.persona_path = Path(persona_path) if persona_path else None
        self.diagnose = diagnose          # 无参可调，返回 doctor_json；没给就没有 /health
        self.fact_path = Path(fact_path) if fact_path else None
        self._health = (0.0, None)

    # ---------- 路由 ----------
    def handle(self, path, query):
        """→ (code, dict)。path 已去掉 /admin/api 前缀。"""
        routes = {"/health": self.health, "/days": self.days, "/top": self.top,
                  "/facts/stats": self.fact_stats, "/facts": self.facts, "/facts/search": self.fact_search,
                  "/facts/random": self.fact_random, "/unresolved": self.unresolved,
                  "/recalls": self.recalls, "/thread": self.thread, "/milestones": self.milestones,
                  "/notes": self.notes}
        try:
            if path.startswith("/records/"):
                return 200, self.record(path[len("/records/"):])
            fn = routes.get(path.rstrip("/"))
            if fn is None:
                return 404, {"error": f"没有这个接口：{path}"}
            if fn == self.health:       # 体检自己另建一份索引、要好几秒，不占着工具调用的锁
                return 200, fn(query)
            with self.lock:
                return 200, fn(query)
        except BadRequest as e:
            return 400, {"error": str(e)}
        except LookupError as e:
            return 404, {"error": str(e)}

    def handle_post(self, path, data):
        """POST 只有一条路：贴便条。→ (code, dict)。"""
        if path.rstrip("/") != "/notes":
            return 404, {"error": f"没有这个接口：{path}"}
        if not isinstance(data, dict):
            return 400, {"error": "请求体要是对象：{text, quote?, recordId?, kind?}"}
        at = self.server.time_context.isoformat(time.time())
        try:
            return 200, self.server.notes_store.add(
                data.get("text"), at, quote=data.get("quote"),
                record_id=data.get("recordId"), kind=data.get("kind"))
        except NoteRequestError as e:
            return 400, {"error": str(e)}
        except NoteStoreError as e:
            return 500, {"error": str(e)}

    def notes(self, q):
        """写过的便条，新的在前：等 TA 看／改好了／TA 回了话。"""
        notes, bad = self.server.notes_store.read()
        return {"notes": notes[::-1], "badLines": bad}

    # ---------- 正文 ----------
    def _chunks(self):
        idx = self.server.index
        gone = getattr(idx, "retracted", set()) | getattr(idx, "superseded", set())
        for i, text in enumerate(idx.chunks):
            meta = idx.meta[i]
            yield i, text, meta, i in gone

    def _row(self, i, text, meta, gone, full=False):
        w = self.server.index.weights[i]
        return {"recordId": meta.get("record_id") or _chunk_key(text),
                "date": meta.get("local_date"), "heading": meta.get("heading"),
                "title": display_title(meta.get("heading"), text),
                "source": meta.get("source"), "layer": meta.get("layer"),
                "status": "gone" if gone else meta.get("status", "current"),
                "recalls": max(0, round((w - 1) / 0.05)),
                "text": text if full or len(text) <= LIST_TEXT else text[:LIST_TEXT] + "…"}

    def _latest_record_day(self):
        days = [m.get("local_date") for _, _, m, _ in self._chunks() if m.get("local_date")]
        return date.fromisoformat(max(days)) if days else date.today()

    def days(self, q):
        # 不给 to 就到库里最近一条为止（线上就是今天；演示库停在录的那天，过几天打开也不空）
        d1 = _day(q, "to", None) or self._latest_record_day()
        d0 = _day(q, "from", d1 - timedelta(days=34))
        if d0 > d1 or (d1 - d0).days > MAX_SPAN_DAYS:
            raise BadRequest(f"from..to 最多跨 {MAX_SPAN_DAYS} 天")
        out = {}
        for i, text, meta, gone in self._chunks():
            day = meta.get("local_date")
            if day and d0.isoformat() <= day <= d1.isoformat():
                out.setdefault(day, []).append(self._row(i, text, meta, gone))
        return {"from": d0.isoformat(), "to": d1.isoformat(), "days": out}

    def record(self, rid):
        if not re.fullmatch(r"[0-9a-f]{16}", rid):
            raise BadRequest("recordId 是 16 位十六进制")
        with self.lock:
            for i, text, meta, gone in self._chunks():
                if (meta.get("record_id") or _chunk_key(text)) == rid:
                    return self._row(i, text, meta, gone, full=True)
        raise LookupError(f"没有这条记录：{rid}")

    def top(self, q):
        n = _int(q, "n", 5, 1, MAX_N)
        rows = [(self.server.index.weights[i], i, t, m, g) for i, t, m, g in self._chunks() if not g]
        rows.sort(key=lambda r: -r[0])
        return {"items": [self._row(i, t, m, g) for _, i, t, m, g in rows[:n]]}

    # ---------- 事实 ----------
    def _facts(self):
        """直接读事实库的 jsonl（FactIndex 不收 meta，事实簿要数"记忆本身"那一类）。
        跳过去重记录和回填中途的，跟 FactIndex 认的是同一批文件。"""
        if self.fact_path is None or not self.fact_path.exists():
            return []
        files = [self.fact_path] if self.fact_path.is_file() else sorted(
            p for p in self.fact_path.rglob("*.jsonl")
            if not p.name.endswith(".dups.jsonl") and "回填中" not in p.parts)
        out = []
        for p in files:
            for line in p.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if (r.get("fact") or "").strip():
                    out.append({"fact": r["fact"].strip(), "date": r.get("event_date") or r.get("written"),
                                "tag": r.get("tag"), "kind": r.get("kind")})
        return out

    def fact_stats(self, q):
        facts = self._facts()
        latest = max((f["date"] for f in facts if f["date"]), default=None)
        d1 = _day(q, "to", None) or (date.fromisoformat(latest) if latest else date.today())
        d0 = _day(q, "from", d1 - timedelta(days=34))
        if d0 > d1 or (d1 - d0).days > MAX_SPAN_DAYS:
            raise BadRequest(f"from..to 最多跨 {MAX_SPAN_DAYS} 天")
        per, total = {}, {}
        for f in facts:
            total[f["tag"]] = total.get(f["tag"], 0) + 1
            if f["date"] and d0.isoformat() <= f["date"] <= d1.isoformat():
                day = per.setdefault(f["date"], {})
                day[f["tag"]] = day.get(f["tag"], 0) + 1
        return {"from": d0.isoformat(), "to": d1.isoformat(), "latest": latest, "perDay": per, "total": total}

    def facts(self, q):
        """某一天（day），或一段日子（from..to，最多 90 天）的事实；事实簿一次取整段，在页面里翻。"""
        tag = q.get("tag", [None])[0]
        day = _day(q, "day")
        if day is not None:
            d0 = d1 = day
        else:
            d0, d1 = _day(q, "from"), _day(q, "to")
            if d0 is None or d1 is None:
                raise BadRequest("要给 day，或者 from 和 to")
            if d0 > d1 or (d1 - d0).days > MAX_SPAN_DAYS:
                raise BadRequest(f"from..to 最多跨 {MAX_SPAN_DAYS} 天")
        rows = [f for f in self._facts() if f["date"] and d0.isoformat() <= f["date"] <= d1.isoformat()
                and (tag is None or f["tag"] == tag)]
        return {"from": d0.isoformat(), "to": d1.isoformat(), "items": rows[:20000]}

    def fact_search(self, q):
        """一句话找相似：按字的二元组重合打分（零依赖；配了向量的部署以后换成向量）。"""
        s = (q.get("q", [""])[0] or "").strip()
        if not s or len(s) > MAX_Q:
            raise BadRequest(f"q 要有内容、最多 {MAX_Q} 字")
        grams = lambda t: {t[i:i + 2] for i in range(len(t) - 1)} or {t}
        g = grams(s)
        scored = []
        for f in self._facts():
            if f["tag"] == "meta":
                continue
            score = len(g & grams(f["fact"])) / len(g)
            if score > 0:
                scored.append((score, f))
        scored.sort(key=lambda x: -x[0])
        return {"q": s, "items": [dict(f, score=round(sc, 3)) for sc, f in scored[:12]]}

    def fact_random(self, q):
        n = _int(q, "n", 3, 1, MAX_N)
        pool = [f for f in self._facts() if f["tag"] != "meta"]
        return {"items": random.sample(pool, min(n, len(pool)))}

    # ---------- 其余几页 ----------
    def _window_days(self):
        """窗口号 → 那个窗口的日子（按 thread 收尾时刻，记忆所有者时区）。"""
        out = {}
        tc = getattr(self.server, "time_context", None)
        for t in self.server.thread_store.all():
            out[t.window] = tc.local_date(t.ended_at) if tc else date.fromtimestamp(t.ended_at).isoformat()
        return out

    def _where(self, src, days):
        """未解决的来源（index/window_12.md、timeline/2026-10-04.md、…#record=…）→ 窗口号和日子。"""
        w = re.search(r"window_(\d+)", src or "")
        d = re.search(r"(\d{4}-\d\d-\d\d)", src or "")
        win = int(w.group(1)) if w else None
        return win, (d.group(1) if d else days.get(win))

    def unresolved(self, q):
        items, bad = self.server.unresolved_store.read(allow_partial=True)
        days = self._window_days()
        out = []
        for x in items:
            w0, d0 = self._where(x.initial, days)
            w1, d1 = self._where(x.updated, days)
            out.append({"id": x.id, "summary": x.summary, "initial": x.initial, "updated": x.updated,
                        "window": w0, "initialDate": d0, "updatedDate": d1 or d0})
        return {"items": out, "badLines": bad}

    def recalls(self, q):
        days = _int(q, "days", 7, 1, MAX_RECALL_DAYS)
        corpus = getattr(self.server, "corpus_dir", None)
        if not corpus:
            return {"items": []}
        # "近 N 天"按最近一条记录往回数：线上最近一条就是刚才，跟按此刻数一样；
        # 演示库停在录制那天，过几天再打开也不会空
        rows = read_recall_log(Path(corpus) / RECALL_LOG_FILENAME, days=36500)
        if rows:
            last = max(r["at"] for r in rows)
            rows = [r for r in rows if r["at"] >= last - days * 86400]
        by_key = {}
        for i, text, meta, gone in self._chunks():
            by_key[meta.get("record_id") or _chunk_key(text)] = (meta, text)
        for r in rows:
            for ref in r.get("recall") or []:
                meta, text = by_key.get(ref["recordId"], ({}, ""))
                ref.update(heading=meta.get("heading"), snippet=text[:160])
        return {"items": rows}

    def thread(self, q):
        n = _int(q, "n", 1, 1, MAX_RECALL_DAYS)
        allt = self.server.thread_store.all()
        allt = sorted(allt, key=lambda t: (t.window, t.ended_at))[-n:]
        t = self.server.thread_store.latest()
        return {"thread": asdict(t) if t else None, "items": [asdict(x) for x in allt]}

    def milestones(self, q):
        if self.persona_path is None or not self.persona_path.exists():
            return {"items": []}
        sec = milestones_section(self.persona_path.read_text(encoding="utf-8"))
        return {"items": milestone_items(sec)}

    def health(self, q):
        if self.diagnose is None:
            raise LookupError("这个部署没开体检")
        at, cached = self._health
        if cached is None or time.time() - at > HEALTH_TTL:
            cached = self.diagnose()
            self._health = (time.time(), cached)
        return cached


# ---------- selftest（合成数据，全部虚构） ----------

def _selftest():
    import tempfile
    from memory_retrieval import MemoryIndex
    from session_thread import ThreadStore, close_thread
    from unresolved_state import UnresolvedStore

    now = time.time()
    idx = MemoryIndex()
    idx.add("## 修咖啡机\n加热管不工作，拆开发现保险丝熔断。", {"heading": "修咖啡机", "local_date": date.today().isoformat(), "timestamp": now})
    idx.add("## 阳台\n整理了阳台的花盆。", {"heading": "阳台", "local_date": date.today().isoformat(), "timestamp": now})
    idx.weights[0] = 1.35
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "事实库").mkdir()
        (td / "事实库" / "f.jsonl").write_text(
            '{"fact": "咖啡机保险丝换好了。", "tag": "life", "event_date": "%s"}\n'
            '{"fact": "这一条是 meta。", "tag": "meta", "event_date": "%s"}\n' % ((date.today().isoformat(),) * 2),
            encoding="utf-8")
        (td / "persona.md").write_text("# 人格\n\n## 我们之间\n私人的话，不该出来。\n\n## 里程碑索引\n\n"
                                       "**第一次修好咖啡机 · 第 3 窗**：她拆，我念说明书。\n\n"
                                       "**阳台的花 · 第 5 窗**：一起选了盆。\n\n## 别的节\n也不该出来。\n",
                                       encoding="utf-8")
        us = UnresolvedStore(td / "未解决.md")
        us.apply([{"action": "open", "summary": "周末去哪还没定"}], "timeline/window_03.md")
        ts = ThreadStore()
        ts.append(close_thread(3, now - 100, now - 50, ("咖啡机",), "修好了"))

        class S:  # MemoryServer 的最小替身：只要这几样
            pass
        s = S()
        s.index, s.unresolved_store, s.thread_store, s.corpus_dir = idx, us, ts, str(td)
        api = AdminAPI(s, persona_path=td / "persona.md", fact_path=td / "事实库",
                       diagnose=lambda: {"status": "ok", "stats": {"files": 1}})

        code, top = api.handle("/top", {"n": ["1"]})
        assert code == 200 and top["items"][0]["heading"] == "修咖啡机" and top["items"][0]["recalls"] == 7
        rid = top["items"][0]["recordId"]
        assert api.handle(f"/records/{rid}", {})[1]["text"].startswith("## 修咖啡机")
        assert api.handle("/records/xyz", {})[0] == 400
        assert api.handle("/records/" + "0" * 16, {})[0] == 404
        assert len(api.handle("/days", {})[1]["days"][date.today().isoformat()]) == 2

        # 参数上限：超了 400，不许拖死
        assert api.handle("/top", {"n": ["51"]})[0] == 400
        assert api.handle("/facts/random", {"n": ["0"]})[0] == 400
        assert api.handle("/recalls", {"days": ["31"]})[0] == 400
        assert api.handle("/days", {"from": ["2026-01-01"], "to": ["2026-06-01"]})[0] == 400
        assert api.handle("/facts/search", {"q": ["长" * 201]})[0] == 400
        assert api.handle("/nope", {})[0] == 404

        st = api.handle("/facts/stats", {})[1]
        assert st["total"] == {"life": 1, "meta": 1}
        assert [f["fact"] for f in api.handle("/facts/random", {"n": ["5"]})[1]["items"]] == ["咖啡机保险丝换好了。"], \
            "meta 不浮"
        assert api.handle("/facts/search", {"q": ["保险丝"]})[1]["items"][0]["fact"] == "咖啡机保险丝换好了。"
        assert api.handle("/unresolved", {})[1]["items"][0]["summary"] == "周末去哪还没定"
        assert api.handle("/thread", {})[1]["thread"]["window"] == 3

        # 里程碑：只回那一节，节外一个字都不出来
        ms = api.handle("/milestones", {})[1]
        assert [m["title"] for m in ms["items"]] == ["第一次修好咖啡机 · 第 3 窗", "阳台的花 · 第 5 窗"]
        blob = json.dumps(ms, ensure_ascii=False)
        assert "不该出来" not in blob, "人格文件节外的内容漏出来了"

        assert api.handle("/health", {})[1]["status"] == "ok"
        assert api.handle("/recalls", {})[1] == {"items": []}
    print("selftest ok（admin_api：十个接口、参数上限、里程碑只回那一节、meta 不浮）")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
