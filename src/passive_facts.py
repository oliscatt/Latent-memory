"""事实模式：捞交给事实库，判交给回话的模型。

事实库是一句一条的 JSONL（id／fact／event_date／written／block／tag／kind），每条当一「块」，
这样 recordId／ranges／签名／宿主切段与账本都沿用块路径的机制，宿主一行不改。

- 每轮：原句一个查询向量，与全部非 meta 事实点积，排除写入日不早于今天的与来源块已撤回／被取代的，取前 2；
  低于噪音下限 FACT_FLOOR 的、冷却中的再去掉，不由后面的名次顶上；
- 同源成组：前 2 名所在块（同一条记忆）里另外几条也过下限、不在冷却的，一并递出，最多 FACT_SIBLINGS 条，
  整段不超过 FACT_GROUP_BYTES 字节；
- 状态软标注：同一轮里两条 kind=state 同块、事件日不同，较早的那条标“较早的状态”，只标不删；
  设了 STATE_SIM 才把不同块、彼此余弦 ≥ 它的也算进来（默认关）；
- 冷却：递出的事实 12 小时内不再递，跨窗口生效，状态落盘在服务端；
- 没有 embedding（零依赖档）时暂不浮，返回 fact_no_embedding（暗号模式另做）。
"""
from __future__ import annotations

from array import array
import datetime as _dt
import json
import math
import os
from pathlib import Path
import shutil
import threading
import time

from memory_retrieval import _chunk_key

FACT_TOP = 2
# 噪音下限：只挡「跟哪件事都不沾边」的句子，不是准入门槛（相关与否仍交给回话的模型）。
# 默认 0.42 是在 voyage-3.5 上定的（开发集 89 句：砍掉约两成无用递送、该接的 28 道一道不丢）；
# 换了向量模型余弦分布会变，用 LATENT_PASSIVE_FACT_FLOOR 调。
FACT_FLOOR = float(os.environ.get("LATENT_PASSIVE_FACT_FLOOR") or 0.42)
# 同源成组：一条记忆拆出的几条是同一件事的几个面，前 2 名之外每轮最多再带这么多条同块的。
FACT_SIBLINGS = 3
# 带兄弟行时整段递送文本的字节上限（前 2 名不受它限制）。默认 1000：有的宿主单轮上限 1200 字节
# 是连外壳一起算的，这里留出余量，正常情况下一轮到宿主不用再裁。
FACT_GROUP_BYTES = int(os.environ.get("LATENT_PASSIVE_FACT_GROUP_BYTES") or 1000)
# 不同块的两条状态事实算“说的是同一件事”的余弦线。默认不设＝跨块比较关闭，只在同一条记忆里比。
# 在维护者自用语料上量过（voyage-3.5）：不相干的状态对余弦中位数 0.585、P95 0.690，0.6 会误标约四成；
# 没有取代记录可作正例时定不了线。要开先在自己的语料上标定。
STATE_SIM = float(os.environ["LATENT_PASSIVE_STATE_SIM"]) if os.environ.get("LATENT_PASSIVE_STATE_SIM") else None
# 同一条事实递出后多久不再递（12 小时）。
COOLDOWN_SECONDS = 12 * 3600
FACT_LAYER = "fact"
_dot = getattr(math, "sumprod", None) or (lambda a, b: sum(x * y for x, y in zip(a, b)))


def _unit(vec):
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return array("f", (x / norm for x in vec))


class FactIndex:
    """只读事实索引。向量存成 float32 数组：一万条×1024 维约 40MB，而 Python float 列表要 300MB+。

    path 可以是单个 facts.jsonl，也可以是事实库目录（读目录下全部 *.jsonl：全量＋各端增量，
    跳过 *.dups.jsonl）。文件有变动时 reload_if_changed() 在后台重读，只给新事实算向量，
    重读期间旧快照照常服务。"""

    def __init__(self, path, provider=None, cache=None, background=False):
        self.path = Path(path)
        self.provider = provider
        self.cache = cache
        self.ready = provider is None
        self.error = None
        self._lock = threading.Lock()
        self._reloading = False
        self._checked_at = 0.0
        self._signature = self._scan()
        rows, by_id = self._read()
        self._snapshot = (rows, by_id, None)
        if provider is None:
            return
        if background:
            # 首次上线要给全部事实算向量（约一万条、云端向量服务实测约 9 分钟）：放后台，不拖住服务启动；
            # 算完之前事实模式回 fact_index_warming，不浮。
            threading.Thread(target=self._build_safely, args=(rows, by_id), daemon=True).start()
        else:
            self._install(rows, by_id, self._vectors(rows))

    @property
    def root(self):
        """事实库目录（增量写入落在它下面）；单文件模式没有目录，返回 None。"""
        return self.path if self.path.is_dir() else None

    @property
    def rows(self):
        return self._snapshot[0]

    @property
    def by_id(self):
        return self._snapshot[1]

    @property
    def vectors(self):
        return self._snapshot[2]

    def _files(self):
        if self.path.is_dir():
            # 回填暂存与去重记录都不是现行事实：前者拆到一半，后者是被并掉的重复。
            return sorted(p for p in self.path.rglob("*.jsonl")
                          if not p.name.endswith(".dups.jsonl")
                          and BACKFILL_DIR not in p.relative_to(self.path).parts
                          and not any(part.endswith(".tmp") for part in p.relative_to(self.path).parts))
        return [self.path]

    def _scan(self):
        out = []
        for p in self._files():
            try:
                st = p.stat()
            except OSError:
                continue
            out.append((str(p), st.st_mtime_ns, st.st_size))
        return tuple(out)

    def _read(self):
        rows, by_id = [], {}
        for path in self._files():
            rel = path.relative_to(self.path).as_posix() if self.path.is_dir() else path.name
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("tag") == "meta":
                    continue
                text = str(item.get("fact") or "").strip()
                if not text:
                    continue
                record_id = _chunk_key(text)
                if record_id in by_id:
                    continue
                row = {"id": len(rows), "text": text, "meta": {
                    "source": rel, "heading": item.get("id"), "chunk_index": n,
                    "layer": FACT_LAYER, "local_date": item.get("event_date") or item.get("written"),
                    "written": item.get("written"), "timestamp_source": FACT_LAYER,
                    "tag": item.get("tag"), "kind": item.get("kind", "event"),
                    "block": item.get("block"),
                    # 没带块号的事实（不经 latent_append、写入方也没按切块规则自己算块号）：记来源文件名，按「那个文件还有现行正文」核对。
                    "source_file": item.get("source_file"),
                }}
                by_id[record_id] = row
                rows.append(row)
        return rows, by_id

    def _build_safely(self, rows, by_id):
        try:
            self._install(rows, by_id, self._vectors(rows))
        except Exception as exc:  # 后台失败要留痕，别静默当成「还在热身」
            self.error = repr(exc)
        finally:
            self._reloading = False

    def _install(self, rows, by_id, vectors):
        self._snapshot = (rows, by_id, vectors)
        self.ready = True

    def _vectors(self, rows):
        # 逐条单位化成 float32，不在内存里攒一整份 Python float 列表（峰值会冲到 400MB）。
        provider, cache = self.provider, self.cache
        texts = [row["text"] for row in rows]
        vectors = [cache.get(t) if cache is not None else None for t in texts]
        todo = [i for i, v in enumerate(vectors) if v is None]
        for start in range(0, len(todo), 64):
            batch = todo[start:start + 64]
            fresh = provider.embed([texts[i] for i in batch])
            if len(fresh) != len(batch):
                raise RuntimeError("事实向量条数对不上")
            for i, vec in zip(batch, fresh):
                if not any(vec):
                    raise RuntimeError("事实向量准备失败，不用全零向量冒充成功")
                vectors[i] = _unit(vec)
                if cache is not None:
                    cache.put(texts[i], vectors[i])
        if cache is not None:
            cache.save()
        return vectors

    def reload_if_changed(self, min_interval=60.0, now=None):
        """文件有变动就在后台重读（最多每 min_interval 秒查一次）；返回是否启动了重读。"""
        import time as _time
        now = _time.monotonic() if now is None else now
        with self._lock:
            if self._reloading or now - self._checked_at < min_interval:
                return False
            self._checked_at = now
            signature = self._scan()
            if signature == self._signature:
                return False
            self._signature = signature
            self._reloading = True
        rows, by_id = self._read()
        if self.provider is None:
            self._install(rows, by_id, None)
            self._reloading = False
            return True
        threading.Thread(target=self._build_safely, args=(rows, by_id), daemon=True).start()
        return True

    def row(self, record_id):
        return self.by_id.get(record_id)

    def similarity(self, a, b):
        """两条事实向量的余弦；按正文在同一份快照里找，后台重读换了快照也不会错位。找不到给 0。"""
        _rows, by_id, vectors = self._snapshot
        ia, ib = by_id.get(_chunk_key(a["text"])), by_id.get(_chunk_key(b["text"]))
        if vectors is None or ia is None or ib is None:
            return 0.0
        return _dot(vectors[ia["id"]], vectors[ib["id"]])

    def ranked(self, query, *, exclude=lambda row: False, top=FACT_TOP):
        """按与原句的相似度取前 top 条（top=None 给全部）；没有向量返回 None（调用方回 fact_no_embedding）。"""
        rows, _by_id, vectors = self._snapshot
        if vectors is None or self.provider is None or not self.ready:
            return None
        qv = _unit(self.provider.embed([query], is_query=True)[0])
        scored = [(_dot(vec, qv), row["id"]) for vec, row in zip(vectors, rows)
                  if not exclude(row)]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [(rows[i], score) for score, i in scored[:top]]


def _fact_rows(block, written, facts, *, allow_empty=False):
    """校验一批事实并编好号，返回待写的行。facts 每项 {fact, tag?, kind?, event_date?} 或一句字符串：
    fact 一句 8–120 字；tag 只收 life／work／meta，kind 只收 event／state。格式不对整批拒收（抛
    ValueError），不半写。编号 f＋sha256(块号|事实)[:12]，同一块同一句重交不变号。"""
    import hashlib
    if not isinstance(facts, list) or (not facts and not allow_empty):
        raise ValueError("facts 必须是非空数组")
    rows = []
    for item in facts:
        if isinstance(item, str):
            item = {"fact": item}
        if not isinstance(item, dict):
            raise ValueError("facts 每项必须是对象或字符串")
        text = str(item.get("fact") or "").strip()
        if not 8 <= len(text) <= 120:
            raise ValueError(f"事实要一句话、8–120 字：{text[:20]}…")
        tag = item.get("tag", "life")
        kind = item.get("kind", "event")
        event_date = item.get("event_date")
        if tag not in {"life", "work", "meta"} or kind not in {"event", "state"}:
            raise ValueError("tag 只收 life／work／meta，kind 只收 event／state")
        if event_date is not None:
            try:
                _dt.date.fromisoformat(str(event_date))
            except ValueError:
                raise ValueError(f"event_date 要写成 YYYY-MM-DD，拿不准就省略：{event_date}") from None
        fid = "f" + hashlib.sha256(f"{block}|{text}".encode()).hexdigest()[:12]
        rows.append({"id": fid, "fact": text, "event_date": event_date, "written": written,
                     "block": block, "tag": tag, "kind": kind})
    return rows


def append_facts(root, side, local_date, block, facts):
    """把一次写回里提炼的事实追加到 <事实库>/增量/<side>/<日期>.jsonl；返回写入条数。"""
    rows = _fact_rows(block, local_date, facts)
    target = Path(root) / "增量" / side / f"{local_date}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    return len(rows)


# ==== 全量回填 ==============================================================
# 事实模式要先有一份「全量」：把已有记忆的每个 timeline 块拆成一句一条的事实。拆的是用户自己那台
# 带人格文件的 AI（它认得用户是谁），服务端只发块、发规则、逐条校验、记进度、最后去重落库。
# 拆到一半的东西放在「回填中/」，不叫「全量*」，所以不会半截开启事实模式；finish 才改名成全量。
BACKFILL_DIR = "回填中"
BACKFILL_RULES = """事实提炼规则（每个块拆成若干条，一行一条）：
1. 一条只讲一件事，一句中文，20–70 字（硬限 8–120 字）。主语写清：用户用你平时对 TA 的称呼或名字，
   你自己写“我”，别人写名字或身份。不写“今天”“昨晚”这类相对时间，换成具体日期或删掉。
2. 保小事：吃了什么、一句玩笑、一个称呼、一件衣服、一个具体数字、谁说了哪句原话——正是要留的，
   不许概括成“聊了很多”“气氛很好”。能带原话的带短原话（15 字以内，用「」）。
3. 买的、搭的、定的东西单独成条：用户买了什么（多少钱、哪天到）、你们搭了什么、定了什么约定或决定
   （谁拍板、哪天）。这类一句话的事实最容易在叙事里被略过。
4. 不写空泛总结和心理分析结论，只写发生了什么、用户是什么情况、约定了什么、喜欢／不喜欢什么。
5. event_date：事实里写明了日期或明显就是块当天，就填 YYYY-MM-DD；拿不准就省略。
6. tag：life＝生活、身体、感情、出行、吃喝、学业工作的日常；work＝用户在做的具体项目工作内容；
   meta＝关于记忆系统本身、测试、复盘的事（照提，但不会被自动浮现）。
7. kind：event＝那天发生了什么；state＝现在是什么状态、以后可能变的事（住哪、在用什么、某个待办还没做）。
   拿不准填 event。
8. 同一块里重复说的同一件事只写一条。finish 只合并同一天写下的近似事实，跨天的不合并，
   所以别把别的块里已经拆过的旧事再拆一遍（比如这块在回顾前几天的事）。
9. 块里没有值得拆的（纯寒暄、纯技术流水账）就交一个空数组，块同样算处理过。
10. 用户明说过“别记”的内容不拆。只写块里有的，不补、不猜。"""
_BACKFILL_DEDUP = 0.6


def _dedup_grams(text):
    import re
    text = re.sub(r"[\s，。、；：「」“”\"'（）()！!？?·…—-]", "", text)
    return {text[i:i + 2] for i in range(len(text) - 1)}


def _near_dup(a, b):
    ga, gb = _dedup_grams(a["fact"]), _dedup_grams(b["fact"])
    return bool(ga and gb) and len(ga & gb) / min(len(ga), len(gb)) >= _BACKFILL_DEDUP


def dedup_facts(rows):
    """只合并「同一天写下的同一件事」：写入日或事件日相同，且字符二元组重合 ≥ 0.6；留较长那条。
    跨天的相似事实一律保留（那是不同的事）。返回 (保留的行, [(被并掉的 id, 留下的 id)])。"""
    from collections import defaultdict
    grams = {row["id"]: _dedup_grams(row["fact"]) for row in rows}
    buckets = defaultdict(list)
    for row in rows:
        for key in {row.get("written"), row.get("event_date")} - {None}:
            buckets[key].append(row)
    dup_of = {}
    for items in buckets.values():
        items.sort(key=lambda row: -len(row["fact"]))
        for i, a in enumerate(items):
            if a["id"] in dup_of:
                continue
            for b in items[i + 1:]:
                if b["id"] in dup_of or b["id"] == a["id"]:
                    continue
                ga, gb = grams[a["id"]], grams[b["id"]]
                if ga and gb and len(ga & gb) / min(len(ga), len(gb)) >= _BACKFILL_DEDUP:
                    dup_of[b["id"]] = a["id"]
    seen, kept = set(), []
    for row in rows:
        if row["id"] in seen or row["id"] in dup_of:
            continue
        seen.add(row["id"])
        kept.append(row)
    return kept, sorted(dup_of.items())


class FactBackfill:
    """回填进度与暂存：<事实库>/回填中/{progress.json, facts.jsonl}。进度按块号记，跨会话续拆。"""

    def __init__(self, root):
        self.root = Path(root)
        self.dir = self.root / BACKFILL_DIR
        self.progress_path = self.dir / "progress.json"
        self.staged_path = self.dir / "facts.jsonl"
        self._lock = threading.RLock()

    def existing_rows(self):
        """事实库里已有的现行事实（全量＋增量，不含回填暂存与去重记录）。"""
        return FactIndex(self.root).rows if self.root.is_dir() else []

    def _full_dirs(self):
        return sorted(p for p in self.root.glob("全量*") if p.is_dir()) if self.root.is_dir() else []

    def recorded_blocks(self):
        """各份全量的 blocks.json 里记下的已处理块号（含交空的、被判重复的）。删掉某份全量，
        它记的块也跟着回到“没拆过”，与事实本身同进退。"""
        out = set()
        for full in self._full_dirs():
            path = full / "blocks.json"
            if path.is_file():
                out.update(json.loads(path.read_text(encoding="utf-8")))
        return out

    @staticmethod
    def _write_blocks(path, blocks):
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(sorted(blocks), ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)

    def blocks(self, index, now=None):
        """待拆的块：timeline 层、没撤回、没被部署插件隐藏、事实库里还没有它拆出的事实；
        同一正文只算一次。返回 [(recordId, 日期, 正文)]。已有事实的块（上次回填过、或写回时
        带过 facts）不再发——重跑回填只补新块，不会把同一批事实再拆一遍。"""
        covered = ({row["meta"].get("block") for row in self.existing_rows()} | self.recorded_blocks()) - {None}
        return [item for item in self._timeline_blocks(index, now) if item[0] not in covered]

    @staticmethod
    def _timeline_blocks(index, now=None):
        hidden = index.hidden_indices(now) if hasattr(index, "hidden_indices") else frozenset()
        retracted = getattr(index, "retracted", set())
        seen, out = set(), []
        for i, (text, meta) in enumerate(zip(index.chunks, index.meta)):
            if meta.get("layer", "timeline") != "timeline" or i in retracted or i in hidden:
                continue
            rid = _chunk_key(text)
            if rid in seen:
                continue
            seen.add(rid)
            out.append((rid, meta.get("local_date"), text))
        return out

    def done(self):
        if not self.progress_path.is_file():
            return {}
        return json.loads(self.progress_path.read_text(encoding="utf-8")).get("done", {})

    def _save_done(self, done):
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.progress_path.with_name("progress.json.tmp")
        tmp.write_text(json.dumps({"version": 1, "done": done}, ensure_ascii=False, sort_keys=True),
                       encoding="utf-8")
        os.replace(tmp, self.progress_path)

    def full_exists(self):
        return self.root.is_dir() and any(p.is_dir() for p in self.root.glob("全量*"))

    def status_line(self, index, now=None):
        blocks, done = self.blocks(index, now), self.done()
        left = sum(1 for rid, _d, _t in blocks if rid not in done)
        staged = sum(done.values())
        return (f"回填进度：{len(blocks) - left}/{len(blocks)} 块已拆，暂存 {staged} 条事实，还剩 {left} 块。"
                + ("已有全量事实库；这里只算还没有事实的新块，拆完会另存一份只含新块的全量。"
                   if self.full_exists() else ""))

    def next_batch(self, index, now=None, limit=8, max_chars=6000):
        done = self.done()
        batch, used = [], 0
        for rid, day, text in self.blocks(index, now):
            if rid in done:
                continue
            if batch and (len(batch) >= limit or used + len(text) > max_chars):
                break
            batch.append({"block": rid, "date": day, "text": text})
            used += len(text)
        return batch

    def submit(self, index, items, now=None):
        """items＝[{block, facts}]。逐块校验，合格的块整块写进暂存并记为已拆，不合格的整块退回。
        返回 (收下的块数, 收下的事实条数, [(块号, 原因)])。"""
        if not isinstance(items, list) or not items:
            raise ValueError("items 必须是非空数组，每项 {block, facts}")
        with self._lock:
            known = {rid: day for rid, day, _t in self.blocks(index, now)}
            done = self.done()
            accepted, lines, errors = {}, [], []
            for item in items:
                block = item.get("block") if isinstance(item, dict) else None
                if block not in known:
                    errors.append((str(block), "不是待拆的块号（照 next 返回的 block 原样抄）"))
                    continue
                if block in done or block in accepted:
                    errors.append((block, "这个块已经交过，不用再交"))
                    continue
                try:
                    rows = _fact_rows(block, known[block], item.get("facts"), allow_empty=True)
                except ValueError as e:
                    errors.append((block, str(e)))
                    continue
                accepted[block] = len(rows)
                lines.extend(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
            if accepted:
                self.dir.mkdir(parents=True, exist_ok=True)
                with open(self.staged_path, "a", encoding="utf-8") as fh:
                    fh.write("".join(lines))
                self._save_done({**done, **accepted})
            return len(accepted), sum(accepted.values()), errors

    def finish(self, index, local_date, now=None):
        """全部块拆完才落库：去重后写 <事实库>/全量-<日期>/{facts.jsonl, facts.dups.jsonl, blocks.json}，
        暂存清掉。blocks.json 记本轮处理过的全部块（含交空的、被判重复的），下次不再发。
        返回 (保留条数, 并掉条数, 全量目录名或 None)。

        没有新事实时不报错、不生成新全量：库里已有全量就把块号合进最新那份的 blocks.json、清暂存；
        一份全量都还没有（整批都是交空的块）就把进度留在暂存里——同样不再重发，也不会因为一份
        空全量而误开事实模式。"""
        with self._lock:
            blocks, done = self.blocks(index, now), self.done()
            left = [rid for rid, _d, _t in blocks if rid not in done]
            if left:
                raise ValueError(f"还有 {len(left)} 块没拆，先用 next／submit 拆完再 finish")
            rows = []
            if self.staged_path.is_file():
                rows = [json.loads(line) for line in self.staged_path.read_text(encoding="utf-8").splitlines()
                        if line.strip()]
            processed = set(done)
            kept, dups = dedup_facts(rows) if rows else ([], [])
            # 与库里已有的事实同一天近似的，留旧的、丢新的，免得两份全量里同一件事各一条。
            existing = self.existing_rows()
            by_day = {}
            for row in existing:
                meta = row["meta"]
                for key in {meta.get("written"), meta.get("local_date")} - {None}:
                    by_day.setdefault(key, []).append({"fact": row["text"]})
            fresh = []
            for row in kept:
                days = {row.get("written"), row.get("event_date")} - {None}
                old = next((e for d in days for e in by_day.get(d, ()) if _near_dup(row, e)), None)
                if old is None:
                    fresh.append(row)
                else:
                    dups.append((row["id"], "已有事实"))
            kept = fresh
            if not kept:
                fulls = self._full_dirs()
                if not fulls:
                    return 0, len(dups), None
                newest = fulls[-1] / "blocks.json"
                old = set(json.loads(newest.read_text(encoding="utf-8"))) if newest.is_file() else set()
                self._write_blocks(newest, old | processed)
                shutil.rmtree(self.dir, ignore_errors=True)
                return 0, len(dups), None
            target = self.root / f"全量-{local_date}"
            n = 1
            while target.exists():
                n += 1
                target = self.root / f"全量-{local_date}-{n}"
            staging = self.root / (target.name + ".tmp")
            staging.mkdir(parents=True)
            (staging / "facts.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept), encoding="utf-8")
            (staging / "facts.dups.jsonl").write_text(
                "".join(json.dumps({"dropped": a, "kept": b}) + "\n" for a, b in dups), encoding="utf-8")
            self._write_blocks(staging / "blocks.json", processed)
            os.replace(staging, target)
            shutil.rmtree(self.dir, ignore_errors=True)
            return len(kept), len(dups), target.name


class FactCooldown:
    """跨窗口冷却：{"version":1,"entries":{recordId: 递出时刻 epoch 秒}}。写临时文件再替换，读写持锁。
    条目也可能是 "YYYY-MM-DD"（按那天本地零点折算），读得懂，不必迁移文件。"""

    def __init__(self, path, seconds=COOLDOWN_SECONDS):
        self.path = Path(path) if path else None
        self.seconds = float(seconds)
        self._lock = threading.RLock()
        self._entries = {}
        if self.path is not None and self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("version") != 1 or not isinstance(data.get("entries"), dict):
                raise ValueError("冷却文件格式不认识")
            self._entries = dict(data["entries"])

    def cooling(self, record_id, now):
        last = self._entries.get(record_id)
        if not last:
            return False
        if isinstance(last, str):
            last = time.mktime(_dt.date.fromisoformat(last).timetuple())
        return now - last < self.seconds

    def mark(self, record_ids, now):
        with self._lock:
            for record_id in record_ids:
                self._entries[record_id] = int(now)
            if self.path is None:
                return
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps({"version": 1, "entries": self._entries},
                                      ensure_ascii=False, sort_keys=True), encoding="utf-8")
            os.replace(tmp, self.path)


class FactVectorCache:
    """事实向量的二进制缓存：同目录 <名>.vec（float32 连续存放）＋ <名>.vec.json（provider 与文本哈希顺序）。

    不用 VectorCache 的 JSON：一万条×1024 维写成文本约 90MB，读盘时整份文本要先常驻再逐条解析。"""

    def __init__(self, base, provider_id):
        import hashlib
        self._hash = lambda text: hashlib.sha1(text.encode("utf-8")).hexdigest()
        self.bin_path = Path(str(base) + ".vec")
        self.idx_path = Path(str(base) + ".vec.json")
        self.provider_id = provider_id
        self.slots, self.dim, self.dirty = {}, None, False
        self._blob = array("f")
        if self.idx_path.exists() and self.bin_path.exists():
            meta = json.loads(self.idx_path.read_text(encoding="utf-8"))
            if meta.get("provider") == provider_id and meta.get("dim"):
                self.dim = int(meta["dim"])
                with open(self.bin_path, "rb") as fh:
                    self._blob.fromfile(fh, self.dim * len(meta["keys"]))
                self.slots = {key: n for n, key in enumerate(meta["keys"])}

    def get(self, text):
        n = self.slots.get(self._hash(text))
        if n is None:
            return None
        return self._blob[n * self.dim:(n + 1) * self.dim]

    def put(self, text, vec):
        if self.dim is None:
            self.dim = len(vec)
        self.slots[self._hash(text)] = len(self.slots)
        self._blob.extend(vec)
        self.dirty = True

    def save(self):
        if not self.dirty:
            return
        keys = [key for key, _ in sorted(self.slots.items(), key=lambda item: item[1])]
        tmp = Path(str(self.bin_path) + ".tmp")
        with open(tmp, "wb") as fh:
            self._blob.tofile(fh)
        os.replace(tmp, self.bin_path)
        self.idx_path.write_text(json.dumps({"provider": self.provider_id, "dim": self.dim,
                                             "keys": keys}), encoding="utf-8")
        self.dirty = False


def fact_index_from_env(provider=None, default_root=None):
    """事实模式何时启用：显式配了 LATENT_PASSIVE_FACTS（文件或目录），或默认事实库目录里已有
    全量回填（「全量*」子目录）。只有零星增量、还没回填历史时不启用——只靠几条新事实浮现，
    比块路径还差。增量写入不受影响，照样落在默认目录里攒着。"""
    path = os.environ.get("LATENT_PASSIVE_FACTS")
    if not path and default_root is not None:
        root = Path(default_root)
        if root.is_dir() and any(p.is_dir() for p in root.glob("全量*")):
            path = str(root)
    if not path:
        return None, None
    base = Path(path)
    # 冷却与向量缓存绝不放进语料仓：那边有自动提交，34MB 的二进制和每轮变的冷却文件会被一并 commit。
    state = Path(os.environ.get("LATENT_PASSIVE_FACT_STATE_DIR")
                 or Path.home() / ".cache" / "latent-passive-facts")
    state.mkdir(parents=True, exist_ok=True)
    cache = FactVectorCache(str(state / "facts"), provider.id) if provider is not None else None
    index = FactIndex(base, provider=provider, cache=cache, background=True)
    cooldown = FactCooldown(str(state / "cooldown.json"))
    return index, cooldown

if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        # 发布检查逐个跑 src/*.py --selftest：事实模式的判据在 tests/passive_facts_test.py，这里直接跑它。
        import unittest
        here = Path(__file__).resolve()
        sys.path.insert(0, str(here.parents[1] / "tests"))
        suite = unittest.defaultTestLoader.loadTestsFromName("passive_facts_test")
        result = unittest.TextTestRunner(verbosity=1).run(suite)
        print("selftest 事实模式：" + ("通过" if result.wasSuccessful() else "失败"))
        sys.exit(0 if result.wasSuccessful() else 1)
