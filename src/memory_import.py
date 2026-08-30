#!/usr/bin/env python3
"""
通用导入翻译器层参考实现（README 核心想法第 1 条 + 设计笔记"通用导入"一节，
任务卡"通用导入翻译器层"）。

中间标准格式一个：MemoryEntry {时间戳, 说话人, 文本, 标签?}（设计笔记 原设想）。
翻译器六个（⚠ 原文写「四个」，Claude memories.json 与 projects/*.json 两支加进来时
这行没跟着改；selftest 打印那行写的一直是六个），各来源 → 统一中间格式：
  parse_chatgpt_export   ChatGPT conversations.json（mapping 树，沿 current_node
                         回溯取当前分支——编辑重发产生的废弃分支不混进来）
  parse_claude_export    Claude 导出 json（chat_messages 列表，ISO 时间戳）
  parse_claude_memories  Claude 导出包里的 memories.json——Claude 自己蒸馏的用户
                         记忆摘要，叙事体长文本，整条导入不切散
  parse_claude_project   Claude 导出包里的 projects/*.json——项目知识库文档
                         （docs 列表），每份文档整条导入
  parse_line_chat        通用逐行聊天文本：时间戳行式（"2026-07-15 21:03 说话人"+
                         内容行，微信第三方导出工具的常见形态）和"说话人：内容"式
  parse_markdown_timeline 我们自己的 timeline md——复用 memory_retrieval 的
                         timeline_chunks（切块+时间戳解析逻辑只有一份，不复制）

流水线两头接上现有消费方：
  entries_to_index → 建好的 MemoryIndex（切块走会话自然边界：时间间隔断开，呼应
                     chunking 实验"自然边界优于机械字数"；chunk meta 带时间戳，
                     recall_recent 对导入语料同样生效）
  entries_to_turns → [(speaker, text)]，draft_extraction 那套提炼函数直接可用
                     （它 docstring 里"留给翻译器层再补"的口从这里接上）
入口 load_any(path)：按扩展名+结构嗅探分发——"选个文件，点导入"的体验目标；
认不出的结构明确报错，不静默猜。
目录入口 load_dir(dir) / corpus_dir_files(dir)：递归吃一整个语料目录（用户的语料
常常就是解压出来的一堆文件），逐个走 load_any——**认不出的逐个报出是哪个文件、
不计进统计**，那条纪律一个字没稀释，只是多认了一种输入（2026.08.04，走查台账第四条）。

**诚实边界（核对状态分翻译器记，别混）**：
  1. Claude 全家翻译器**已与真实导出两轮核对**（2026.07.30，维护者提供官方导出，
     语料不入库、只核结构+聚合冒烟）：conversations.json 字段名/sender 枚举
     （human、assistant）/时间戳形状（ISO 微秒+Z）/空消息跳过全部对上，且全量
     样本（12 对话、2153 条非空消息）把正文提取和 entries_to_index→recall 整条
     管线真跑通（396 块全带时间戳、recall 按新鲜度降序）；content 块实测有
     text/thinking/tool_use/tool_result 四种类型，只取 text 是对的——thinking
     的正文键叫 thinking 不叫 text，工具块是过程记录，都不该进记忆；
     memories.json、projects/*.json 结构同批核对。
     另一条实测经验：官方导出选"最近 30 天"会把窗口外的对话全部框空（第一轮
     就踩了——最后一条 app 对话在窗口外 5 天，导出近乎全空），导记忆一律选全量。
  2. ChatGPT 翻译器字段名按公开已知结构写。**2026.08.03 维护者实测：ChatGPT、
     Kelivo、Operit 三家导出的 json 都能被 load_any 认出并正常建库起服务**——
     所以"认不认得出"这一格不再空着。
     ⚠ **但"起得来"不等于"字段语义对"，这两件事这里刻意分开记**：load_any 的分发
     是**按顶层键嗅探**（mapping / chat_messages / conversations_memory / docs），
     键匹配上就会进对应翻译器；说话人枚举归一得对不对、时间戳形状认得对不对、
     空消息该不该跳，嗅探一概不管。上面第 1 条 Claude 那家敢写"已核对"，
     是因为真做过两轮字段级比对并留了具体数字（12 对话、2153 条消息、396 块）。
     **这一档还没有那种证据，所以接真样本前仍然先跑 --stats 核对
     条数/说话人/时间范围再信。**
     ⚠ 还有一格没答：Kelivo／Operit 的 json 具体命中的是哪个翻译器（多半是
     ChatGPT 那支，因为不少 App 照抄了它的导出形状）。这决定要不要给它们单独写
     翻译器——**在答案出来之前，别把"它们能跑"写成"我们支持这两家的格式"**。
  3. 微信官方没有导出功能、第三方工具格式五花八门——不猜着写某个具体工具的
     解析器，parse_line_chat 按最常见形态覆盖，对不上的等拿到真实样本再补。

零依赖，stdlib only。
用法：
  python memory_import.py --selftest
  python memory_import.py --stats <导出文件或语料目录>   # 接真实样本前的核对步骤
                                                # 目录递归；认不出的文件逐个报出、退出码 1
"""

import argparse
import json
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# 同目录模块，import 不触发其 CLI
from memory_retrieval import MemoryIndex, timeline_chunks

# 会话切块参数：半小时没说话算一次会话边界、单块字符上限——都是经验值，
# 跟检索层 +0.05/k=60 同一待遇，等真实评估再调
DEFAULT_GAP_SECONDS = 1800
DEFAULT_MAX_CHARS = 800


@dataclass
class MemoryEntry:
    """中间标准格式（设计笔记"通用导入"：{时间戳, 说话人, 文本, 标签?}）。
    timestamp 是 epoch 秒或 None；speaker 叙事体语料（timeline）为空串；
    tags 第一个元素约定为来源（chatgpt/claude/chatlog/timeline）。

    ⚠ **`timestamp_source` 不是装饰，是「这个时间戳能不能拿来下结论」**
    （2026.08.04 审查侧黑盒抓到）：timeline 那条链的时间戳有优先级
    （`chunk_head_slash` / `file_head` / `heading` / `chunk_head` / **`mtime`**），
    而 `mtime` 是**没有日期依据时的兜底**——拷贝一遍目录、重新 clone，它就变成"刚刚"。
    `parse_markdown_timeline` 原先只取 `timestamp` 把来源丢了，于是语料目录里
    **随便一份今天改过的 README.md 就能把「这份语料最新停在哪天」顶成今天**，
    第 0 步那道旧快照体检当场失效、且不报错。**取时间戳的地方必须能看见它的来源。**
    ⚠ 只有 timeline 那条链会填这个字段；其余翻译器的时间戳来自导出文件里的真实时间，
    留空即可——**判据是「是不是 mtime」，不是「有没有标注」**，所以留空按可信算。"""
    text: str
    speaker: str = ""
    timestamp: float = None
    tags: tuple = ()
    timestamp_source: str = ""


# ---------- 翻译器一：ChatGPT conversations.json ----------

def _chatgpt_conv_entries(conv):
    mapping = conv.get("mapping") or {}
    cur = conv.get("current_node")
    if cur in mapping:
        # 沿 current_node 回溯到根：这是当前对话分支——编辑重发会在 mapping 树里
        # 留下废弃分支，全收会把"被编辑掉的旧问法/旧回答"混进记忆
        node_ids = []
        while cur:
            node_ids.append(cur)
            cur = (mapping.get(cur) or {}).get("parent")
        node_ids.reverse()
    else:
        # 没有 current_node 的兜底：按 create_time 排。局限如实标：废弃分支会混进来
        node_ids = sorted(mapping, key=lambda k: ((mapping[k].get("message") or {}).get("create_time") or 0))
    title = conv.get("title") or ""
    out = []
    for nid in node_ids:
        msg = (mapping.get(nid) or {}).get("message")
        if not msg:
            continue
        if ((msg.get("author") or {}).get("role")) not in ("user", "assistant"):
            continue  # system/tool 节点不是对话内容
        parts = ((msg.get("content") or {}).get("parts")) or []
        text = "\n".join(p for p in parts if isinstance(p, str)).strip()
        if not text:
            continue
        out.append(MemoryEntry(text=text, speaker=msg["author"]["role"], timestamp=msg.get("create_time"),
                               tags=("chatgpt", title) if title else ("chatgpt",)))
    return out


def parse_chatgpt_export(data):
    """conversations.json（单个对话 dict 或对话列表）→ [MemoryEntry]。"""
    entries = []
    for conv in (data if isinstance(data, list) else [data]):
        entries.extend(_chatgpt_conv_entries(conv))
    return entries


# ---------- 翻译器二：Claude 导出 json ----------

def _iso_ts(s):
    """ISO8601 → epoch 秒；解析不了返回 None 不抛（时间戳缺失下游本来就有兜底）。"""
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError, TypeError):
        return None


def parse_claude_export(data):
    """Claude 导出（单个对话 dict 或列表，chat_messages 结构）→ [MemoryEntry]。
    sender 的 human 归一成 user，跟 ChatGPT 翻译器出口一致。"""
    entries = []
    for conv in (data if isinstance(data, list) else [data]):
        title = conv.get("name") or ""
        for m in conv.get("chat_messages") or []:
            text = (m.get("text") or "\n".join(
                b.get("text", "") for b in (m.get("content") or []) if b.get("type") == "text")).strip()
            if not text:
                continue
            speaker = {"human": "user"}.get(m.get("sender"), m.get("sender") or "")
            entries.append(MemoryEntry(text=text, speaker=speaker, timestamp=_iso_ts(m.get("created_at")),
                                       tags=("claude", title) if title else ("claude",)))
    return entries


def parse_claude_project(data):
    """Claude 导出包里的 projects/*.json：项目知识库（docs: [{filename, content,
    created_at}]），是用户主动整理上传的资料不是对话。每份文档整条导入（它天然
    是完整单元），时间戳取文档 created_at。结构与真实导出核对过（2026.07.30）。"""
    entries = []
    for proj in (data if isinstance(data, list) else [data]):
        pname = proj.get("name") or ""
        for doc in proj.get("docs") or []:
            text = (doc.get("content") or "").strip()
            if not text:
                continue
            entries.append(MemoryEntry(text=text, speaker="",
                                       timestamp=_iso_ts(doc.get("created_at")),
                                       tags=("claude_project", pname, doc.get("filename") or "")))
    return entries


def parse_claude_memories(data):
    """Claude 导出包里的 memories.json：[{conversations_memory: 长文本, account_uuid}]。
    是 Claude 自己蒸馏的用户记忆摘要——叙事体、无时间戳，整条当一个条目导入，
    不切散（它本身已是浓缩产物，再切只会打碎语义）。结构与真实导出核对过（2026.07.30）。"""
    entries = []
    for item in (data if isinstance(data, list) else [data]):
        text = (item.get("conversations_memory") or "").strip()
        if text:
            entries.append(MemoryEntry(text=text, speaker="", tags=("claude_memory",)))
    return entries


# ---------- 翻译器三：通用逐行聊天文本 ----------

_LINE_TS_RE = re.compile(
    r"^(20\d{2})[-./](\d{1,2})[-./](\d{1,2})\s+(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(.*)$")


def parse_line_chat(text):
    """通用逐行聊天文本 → [MemoryEntry]。两种形态：
      A) 时间戳行"2026-07-15 21:03[:22] 说话人"，后续行是内容，直到下一个时间戳行
      B) "说话人：内容"单行（draft_extraction.parse_transcript 同款），无时间戳
    微信第三方导出工具格式不一，这里按最常见形态写——接真实样本前先 --stats 核对。"""
    entries, cur = [], None  # cur = [speaker, ts, lines]

    def flush():
        nonlocal cur
        if cur and cur[2]:
            entries.append(MemoryEntry(text="\n".join(cur[2]).strip(), speaker=cur[0],
                                       timestamp=cur[1], tags=("chatlog",)))
        cur = None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _LINE_TS_RE.match(line)
        if m:
            flush()
            y, mo, d, hh, mm, ss, speaker = m.groups()
            try:
                ts = datetime(int(y), int(mo), int(d), int(hh), int(mm), int(ss or 0)).timestamp()
            except ValueError:
                ts = None  # "2026-13-40 25:99"这类假日期行：当成时间未知，内容照收
            cur = [speaker.strip(), ts, []]
            continue
        if cur is not None:
            cur[2].append(line)
            continue
        sep = "：" if "：" in line else (":" if ":" in line else None)
        if sep:
            speaker, _, content = line.partition(sep)
            if speaker.strip() and content.strip():
                entries.append(MemoryEntry(text=content.strip(), speaker=speaker.strip(), tags=("chatlog",)))
    flush()
    return entries


# ---------- 翻译器四：markdown timeline ----------

def parse_markdown_timeline(text, filename="timeline.md", mtime=0.0):
    """timeline md → [MemoryEntry]。切块+时间戳解析全部复用 memory_retrieval 的
    timeline_chunks（chunk_head/file_head/mtime 那套优先级），这里只包一层中间格式。
    叙事体没有说话人，speaker 为空串——entries_to_turns 会把它筛掉，不进对话轮。"""
    entries = []
    for chunk, meta in timeline_chunks(text, filename, mtime):
        entries.append(MemoryEntry(text=chunk, speaker="", timestamp=meta["timestamp"],
                                   tags=("timeline", meta["heading"]) if meta["heading"] else ("timeline",),
                                   timestamp_source=meta.get("timestamp_source", "")))
    return entries


# ---------- 流水线出口：两个消费方 ----------

def entries_to_index(entries, embed=False, gap_seconds=DEFAULT_GAP_SECONDS,
                     max_chars=DEFAULT_MAX_CHARS, source="import"):
    """中间格式 → 建好的 MemoryIndex（统一流水线的"切块→存库"段）。

    切块走会话自然边界（chunking 实验结论：自然边界优于机械字数）：相邻条目时间
    间隔超 gap_seconds 断开；块超 max_chars 也断开，但单条超长条目不硬切（timeline
    叙事块整条进，不重切）。chunk meta 带首条时间戳（timestamp_source="import"），
    recall_recent 对导入语料同样生效——换窗召回不区分语料来自哪个 app。"""
    index = MemoryIndex(embed=embed)
    buf, buf_ts, last_ts, size = [], None, None, 0

    def flush():
        nonlocal buf, buf_ts, size
        if buf:
            meta = {"source": source, "chunk_index": len(index.chunks)}
            if buf_ts is not None:
                meta["timestamp"], meta["timestamp_source"] = buf_ts, "import"
            index.add("\n".join(buf), meta)
        buf, buf_ts, size = [], None, 0

    for e in entries:
        line = f"{e.speaker}：{e.text}" if e.speaker else e.text
        gap = (e.timestamp is not None and last_ts is not None
               and e.timestamp - last_ts > gap_seconds)
        if buf and (gap or size + len(line) > max_chars):
            flush()
        buf.append(line)
        size += len(line)
        if buf_ts is None:
            buf_ts = e.timestamp
        if e.timestamp is not None:
            last_ts = e.timestamp
    flush()
    return index.build()


def entries_to_turns(entries):
    """中间格式 → [(speaker, text)] 对话轮，draft_extraction 的提炼函数直接可用。
    无说话人的叙事体条目（timeline）不是对话轮，筛掉。"""
    return [(e.speaker, e.text) for e in entries if e.speaker]


# ---------- 入口：选个文件，点导入 ----------

def load_any(path):
    """按扩展名+结构嗅探分发到对应翻译器 → [MemoryEntry]。
    认不出的结构明确报错，不静默猜成某种格式。"""
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        data = json.loads(raw)
        sample = data[0] if isinstance(data, list) and data else data
        if isinstance(sample, dict) and "mapping" in sample:
            return parse_chatgpt_export(data)
        if isinstance(sample, dict) and "chat_messages" in sample:
            return parse_claude_export(data)
        if isinstance(sample, dict) and "conversations_memory" in sample:
            return parse_claude_memories(data)
        if isinstance(sample, dict) and isinstance(sample.get("docs"), list):
            return parse_claude_project(data)
        raise ValueError(f"认不出的 json 导出结构：{path}"
                         "（mapping / chat_messages / conversations_memory / docs 都没有）")
    if p.suffix.lower() in (".md", ".markdown"):
        return parse_markdown_timeline(raw, p.name, p.stat().st_mtime)
    return parse_line_chat(raw)


def corpus_dir_files(directory):
    """语料目录 → 参与统计的文件（递归、排序、跳过隐藏项）。

    **递归不是「更全」，是范围要跟建库对齐**：`persona_compiler.py`、
    `memory_init.py`、`mcp_server` 体检那条链读语料目录一律 `rglob("*")`。
    这里只看一层的话，**体检的范围会小于实际建库的范围**——用户把旧的那批
    放在子目录里，第 0 步照样报不出旧快照，比崩掉更坏（崩掉至少看得见）。

    ⚠ 路径任何一段以 `.` 开头的一律跳过：语料目录里会有本项目自己写的 sidecar
    （`.weights.json` 每次检索都落盘，见 `mcp_server._corpus_signature`），
    那不是用户语料。不跳的话跑过服务的用户会被报一堆「认不出格式」，
    **而那些噪声会把真正认不出的那份名单淹掉**。"""
    root = Path(directory)
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts))


def load_dir(directory):
    """目录 → ([MemoryEntry], [(路径, 原因)])：逐个文件走 load_any。

    ⚠ **load_any 那条纪律一个字不动**——单文件认不出的结构照旧抛。这里只是让
    「一个目录」也成为合法输入，**不是把纪律稀释成静默跳过**：认不出的文件
    逐个记下**是哪个文件、为什么**，由调用方原样报给人看。

    一个坏文件不顶掉整份汇总（用户的语料目录里混进一张图、一个 pdf 是常态，
    为它整个不出结果，第 0 步就又被跳过去了），但**它没被统计进去这件事必须说出来**
    ——否则用户会以为上面那个条数覆盖了目录里所有东西。

    ⚠ **第三个返回值 `empties` 是 2026.08.04 审查侧黑盒补的**：原先只挡住**抛异常**的
    文件，挡不住**读得进去但一条都没解析出来**的（几行散文的 `笔记.txt` 就是）。
    那种文件被静默计进「N 个文件」，而它一条都没贡献——**单文件跑时用户看得见那个
    `0 条`，目录汇总里它被淹掉了**。「认不出就报出来，不猜也不静默跳过」这条纪律
    在目录这个新入口上原本有这个洞。
    ⚠ **它跟 `problems` 分两个桶、退出码也不同**：认不出（抛异常）是**错**，退出码 1；
    解析出 0 条多半是语料目录里那份 README，是**常态**，只报不红——
    让它也红会逼用户去清理一个本来就该允许存在的文件。"""
    entries, problems, empties = [], [], []
    for p in corpus_dir_files(directory):
        try:
            got = load_any(p)
        except (ValueError, UnicodeDecodeError, OSError) as exc:
            problems.append((p, f"{type(exc).__name__}: {exc}"))
            continue
        if not got:
            empties.append(p)          # 读得进去、一条也没解析出来：见 docstring
            continue
        entries.extend(got)
    return entries, problems, empties


def stats(path):
    """接真实样本前的核对步骤：只打聚合统计，不打内容（语料不该进终端记录）。

    文件和目录都吃。返回退出码：认不出的文件存在时返回 1（**不是崩**——没有
    traceback、汇总照打，只是不让第 0 步这条命令在脚本里被当成完全通过）。

    ⚠ **这个函数的目的不是「打统计」，是让第 0 步能发现旧快照**（走查台账第四条）：
    所以时间范围之外还要**单独把最新那个日期摆出来**并跟一句让人对一对的话，
    口径与 `mcp_server` 体检的「时间范围」那条同源。只报一个 `A ~ B` 区间的话，
    人眼会滑过去——2026.08.04 那次真实的旧快照就是被 `--doctor` 抓到的，不是这里。"""
    p = Path(path)
    if not p.exists():
        print(f"找不到这个路径：{path}")
        return 1
    if p.is_dir():
        entries, problems, empties = load_dir(p)
        counted = len(corpus_dir_files(p)) - len(problems) - len(empties)
        head = f"{counted} 个文件参与统计（目录，递归）；"
    else:
        entries, problems, empties = load_any(p), [], []
        head = ""
    speakers = sorted({e.speaker for e in entries if e.speaker})
    # ⚠ **落 mtime 的时间戳不参与任何结论**（2026.08.04 审查侧黑盒抓到）：它是
    #    「这块没有任何日期依据」的兜底，拷一遍目录就变成"刚刚"。混进来的话，
    #    语料目录里随便一份今天改过的 README.md 就能把「最新停在哪天」顶成今天，
    #    **第 0 步这道旧快照体检当场失效，而且不报错。**
    ts = [e.timestamp for e in entries
          if e.timestamp is not None and e.timestamp_source != "mtime"]
    n_mtime = sum(1 for e in entries
                  if e.timestamp is not None and e.timestamp_source == "mtime")
    fmt = lambda t: datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")  # noqa: E731
    line = (f"{head}{len(entries)} 条；说话人：{'/'.join(speakers) or '（叙事体，无说话人）'}"
            f"；带时间戳 {len(ts)} 条")
    if ts:
        line += f"；时间范围 {fmt(min(ts))} ~ {fmt(max(ts))}"
    print(line)
    if ts:
        print(f"⚠ 这份语料最新的一条停在 {datetime.fromtimestamp(max(ts)):%Y-%m-%d}"
              "——盯住这个日期，问 TA 一句：跟 TA 最近一次聊天真的是这天吗？"
              "对不上说明这份导出是旧快照，让 TA 重新导一份（时间范围选「全部」）再建库"
              "（条数再漂亮也救不了停在过去的语料）。")
    elif n_mtime:
        # ⚠ 说法要跟抬头那个「带时间戳 0 条」对得上：那个数只数可信的，
        #    这里再说一次「有 N 条带时间戳」，同一屏里两个数打架，人会以为程序错了。
        print(f"⚠ 另有 {n_mtime} 条**只有文件修改时间**（没有任何日期依据，因此没算进上面"
              "那个「带时间戳」数）"
              "——「这份导出是不是旧快照」这道体检**在这份语料上无效**，别当成查过了。"
              "修法是把日期写进文件名（window_04_2026-06-17.md 这种）或标题行。")
    else:
        print("⚠ 一条都没有时间戳，算不出时间范围——这份语料上「导出是不是旧快照」"
              "这道体检无效，别当成查过了；换窗召回按时间新鲜度排序，也没有排序依据。")
    if ts and n_mtime:
        print(f"⚠ 另有 {n_mtime} 条只有文件修改时间，**没有算进上面那个日期**"
              "——它们没有任何日期依据，拷一遍目录就会变成「刚刚」，"
              "算进去会把旧快照伪装成新语料。")
    if empties:
        print(f"⚠ 另有 {len(empties)} 个文件读得进去、但一条都没解析出来"
              "（多半是 README／随手记这类不是语料的东西，已从上面的文件数里剔除）：")
        for e in empties:
            print(f"    - {e}")
    if problems:
        print(f"⚠ 另有 {len(problems)} 个文件认不出格式，没有计入上面的数"
              "（认不出就报出来，不猜也不静默跳过）：")
        for bad, why in problems:
            print(f"    - {bad}：{why}")
    return 1 if problems else 0


# ---------- selftest（合成 fixture，全部虚构，不含任何真实语料） ----------

_CHATGPT_FIXTURE = {
    "title": "修咖啡机",
    "current_node": "n3",
    "mapping": {
        "root": {"message": None, "parent": None, "children": ["sys"]},
        "sys": {"message": {"author": {"role": "system"}, "create_time": None,
                            "content": {"content_type": "text", "parts": [""]}},
                "parent": "root", "children": ["n1"]},
        "n1": {"message": {"author": {"role": "user"}, "create_time": 1752500000.0,
                           "content": {"content_type": "text", "parts": ["咖啡机又坏了，怎么办"]}},
               "parent": "sys", "children": ["n2", "n2_edited"]},
        "n2_edited": {"message": {"author": {"role": "assistant"}, "create_time": 1752500100.0,
                                  "content": {"content_type": "text", "parts": ["被编辑掉的旧回答"]}},
                      "parent": "n1", "children": []},
        "n2": {"message": {"author": {"role": "assistant"}, "create_time": 1752500060.0,
                           "content": {"content_type": "text", "parts": ["检查保险丝有没有熔断"]}},
               "parent": "n1", "children": ["n3"]},
        "n3": {"message": {"author": {"role": "user"}, "create_time": 1752500120.0,
                           "content": {"content_type": "text", "parts": ["果然是保险丝"]}},
               "parent": "n2", "children": []},
    },
}

# fixture 形状照 2026.07.30 真实导出核对结果构造（内容虚构）：时间戳 ISO 微秒+Z、
# 消息带 uuid/attachments/files 等我们不用的键（解析器要容忍）、含一条真实样本里
# 实际存在的空消息（text 空且 content 空）
_CLAUDE_FIXTURE = [{
    "uuid": "0" * 36,
    "name": "种薄荷",
    "summary": "",
    "created_at": "2026-07-15T21:03:00.000000Z",
    "chat_messages": [
        {"uuid": "1" * 36, "sender": "human", "text": "薄荷又蔫了", "content": [],
         "created_at": "2026-07-15T21:03:22.000000Z", "attachments": [], "files": []},
        {"uuid": "2" * 36, "sender": "assistant", "text": "",
         "content": [
             # 真实导出的 content 块有四种类型；thinking 的正文键叫 thinking、
             # 工具块是过程记录——只有 text 类型该进记忆，混在这里当回归守护
             {"type": "thinking", "thinking": "思考过程不该进记忆", "summaries": [], "cut_off": False},
             {"type": "tool_use", "name": "web_search", "input": {}, "message": ""},
             {"type": "tool_result", "name": "web_search", "content": [], "is_error": False},
             {"type": "text", "text": "盆底是不是积水了", "citations": [], "flags": []},
         ],
         "created_at": "2026-07-15T21:04:00.000000Z", "attachments": [], "files": []},
        {"uuid": "3" * 36, "sender": "human", "text": "", "content": [],
         "created_at": "2026-07-15T21:05:00.000000Z", "attachments": [], "files": []},
    ],
}]

_CLAUDE_MEMORIES_FIXTURE = [{"conversations_memory": "喜欢薄荷茶；阳台养了两盆植物；读小说偏爱结构复杂的。",
                             "account_uuid": "a" * 36}]

_CLAUDE_PROJECT_FIXTURE = {
    "uuid": "c" * 36, "name": "阳台种植笔记", "description": "", "is_private": True,
    "prompt_template": "", "created_at": "2026-07-01T10:00:00.000000+00:00",
    "docs": [
        {"uuid": "d" * 36, "filename": "浇水表.md", "content": "薄荷两天一浇，正午不浇。",
         "created_at": "2026-07-01T10:05:00.000000+00:00"},
        {"uuid": "e" * 36, "filename": "空文档.md", "content": "",
         "created_at": "2026-07-02T10:05:00.000000+00:00"},
    ],
}

_LINECHAT_FIXTURE = """2026-07-15 21:03 林岸
今天真的好累
2026-07-15 21:04:10 星回
先歇会儿，活儿明天还在
跑不掉的
"""

_TIMELINE_FIXTURE = "# window_30_晒被子（2026.07.15）\n\n## 上午\n把被子都晒了。\n\n## 傍晚\n收被子，带着太阳味。\n"


def _selftest():
    # 1.【变异靶心：current_node 回溯】ChatGPT 翻译器取当前分支：3 条、按序、
    #    编辑废弃分支不混入、system 节点被滤掉、epoch 时间戳原样带出
    e = parse_chatgpt_export(_CHATGPT_FIXTURE)
    assert [(x.speaker, x.text) for x in e] == [
        ("user", "咖啡机又坏了，怎么办"), ("assistant", "检查保险丝有没有熔断"), ("user", "果然是保险丝")], \
        f"当前分支该是 3 条对话，实际 {[(x.speaker, x.text) for x in e]}"
    assert all("被编辑掉" not in x.text for x in e), "废弃分支不该混进记忆"
    assert e[0].timestamp == 1752500000.0 and e[0].tags == ("chatgpt", "修咖啡机")

    # 2. 没有 current_node 的兜底：按 create_time 排——废弃分支会混进来，局限如实测出
    no_cur = {k: v for k, v in _CHATGPT_FIXTURE.items() if k != "current_node"}
    e2 = parse_chatgpt_export(no_cur)
    assert len(e2) == 4 and e2[2].text == "被编辑掉的旧回答", "兜底路径按时间排、含废弃分支（已知局限）"

    # 3. Claude 翻译器（fixture 形状照真实导出核对结果）：text 字段和 content 块两种
    #    形态都吃，human 归一成 user，微秒+Z 时间戳解析，空消息（真实样本里存在）跳过
    e3 = parse_claude_export(_CLAUDE_FIXTURE)
    assert [(x.speaker, x.text) for x in e3] == [("user", "薄荷又蔫了"), ("assistant", "盆底是不是积水了")], \
        "两条非空消息进、一条空消息跳过；thinking/tool_use/tool_result 块不进记忆"
    assert e3[0].timestamp is not None and e3[1].timestamp - e3[0].timestamp == 38.0, "ISO 时间戳差 38 秒"
    assert e3[0].tags == ("claude", "种薄荷")

    # 4. 逐行文本翻译器：时间戳行式（含多行内容归并、秒可省）和"说话人：内容"式
    e4 = parse_line_chat(_LINECHAT_FIXTURE)
    assert len(e4) == 2 and e4[0].speaker == "林岸" and e4[1].text == "先歇会儿，活儿明天还在\n跑不掉的"
    assert e4[1].timestamp - e4[0].timestamp == 70.0, "21:03 → 21:04:10 差 70 秒"
    e4b = parse_line_chat("林岸：不想歇\n星回：那就再撑十分钟")
    assert [(x.speaker, x.text) for x in e4b] == [("林岸", "不想歇"), ("星回", "那就再撑十分钟")]

    # 5. timeline 翻译器：时间戳走 timeline_chunks 那套（标题行完整日期），无说话人
    e5 = parse_markdown_timeline(_TIMELINE_FIXTURE, "window_30_晒被子.md", mtime=1752500000.0)
    assert len(e5) >= 2 and all(x.speaker == "" for x in e5)
    assert all(datetime.fromtimestamp(x.timestamp).strftime("%Y%m%d") == "20260715" for x in e5), \
        "timeline 条目时间戳该来自标题行日期，不是 mtime"

    # 6. 消费方一（draft_extraction）真接上：翻译器出口 → 对话轮 → 风格候选
    from draft_extraction import extract_style_candidates
    turns = entries_to_turns(e4 + e4b)
    assert ("星回", "那就再撑十分钟") in turns and all(who for who, _ in turns)
    assert entries_to_turns(e5) == [], "叙事体条目不进对话轮"
    cands = extract_style_candidates(turns, "星回", topN=2)
    assert cands and all(not c.confirmed for c in cands), "提炼函数直接吃翻译器出口"

    # 7.【变异靶心：会话间隙切块 + 时间戳传导】消费方二（MemoryIndex）真接上
    t0 = 1752500000.0
    convo = [MemoryEntry("今天真的好累", "林岸", t0),
             MemoryEntry("先歇会儿", "星回", t0 + 60),
             MemoryEntry("昨晚那杯薄荷茶不错", "林岸", t0 + 7200)]
    idx = entries_to_index(convo, gap_seconds=1800)
    assert len(idx.chunks) == 2, "隔两小时是新会话，该切成两块"
    assert idx.chunks[0] == "林岸：今天真的好累\n星回：先歇会儿", "同会话归并且带说话人前缀"
    assert idx.meta[0]["timestamp"] == t0 and idx.meta[1]["timestamp"] == t0 + 7200, "块时间戳=首条时间戳"
    assert idx.retrieve("薄荷茶", topN=1)[0]["id"] == 1, "导入后可检索"
    assert idx.recall_recent(topN=2, now=t0 + 7300)[0]["id"] == 1, "recall_recent 对导入语料生效"
    #    单条超长不硬切
    assert len(entries_to_index([MemoryEntry("长" * 2000, "林岸", t0)], max_chars=800).chunks) == 1

    # 8. load_any 分发：四种文件各进各的翻译器；认不出的 json 明确报错
    with tempfile.TemporaryDirectory() as td:
        cases = [("c.json", json.dumps([_CHATGPT_FIXTURE]), "chatgpt"),
                 ("k.json", json.dumps(_CLAUDE_FIXTURE), "claude"),
                 ("t.md", _TIMELINE_FIXTURE, "timeline"),
                 ("w.txt", _LINECHAT_FIXTURE, "chatlog")]
        for name, content, tag in cases:
            Path(td, name).write_text(content, encoding="utf-8")
            got = load_any(Path(td, name))
            assert got and got[0].tags[0] == tag, f"{name} 该分发到 {tag} 翻译器"
        Path(td, "bad.json").write_text('{"foo": 1}', encoding="utf-8")
        try:
            load_any(Path(td, "bad.json"))
            assert False, "认不出的结构该报错，不能静默猜"
        except ValueError:
            pass

    # 9. 空输入兜底：不崩，空进空出
    assert parse_line_chat("") == [] and entries_to_turns([]) == []
    empty = entries_to_index([])
    assert empty.retrieve("任意", topN=3) == [] and empty.recall_recent(topN=3) == []

    # 10. Claude memories.json 翻译器：整条导入不切散、无说话人不进对话轮、分发认得出
    e10 = parse_claude_memories(_CLAUDE_MEMORIES_FIXTURE)
    assert len(e10) == 1 and e10[0].tags == ("claude_memory",) and e10[0].speaker == ""
    assert "薄荷茶" in e10[0].text and entries_to_turns(e10) == []
    assert parse_claude_memories([{"conversations_memory": "", "account_uuid": "b" * 36}]) == [], "空摘要不产条目"
    with tempfile.TemporaryDirectory() as td:
        Path(td, "m.json").write_text(json.dumps(_CLAUDE_MEMORIES_FIXTURE), encoding="utf-8")
        assert load_any(Path(td, "m.json"))[0].tags == ("claude_memory",), "load_any 认得 memories.json"

    # 11. Claude projects/*.json 翻译器：知识库文档整条导入、空文档跳过、时间戳解析、分发认得出
    e11 = parse_claude_project(_CLAUDE_PROJECT_FIXTURE)
    assert len(e11) == 1 and e11[0].tags == ("claude_project", "阳台种植笔记", "浇水表.md"), \
        "一份有内容的文档进、空文档跳过"
    assert e11[0].timestamp is not None and e11[0].speaker == ""
    with tempfile.TemporaryDirectory() as td:
        Path(td, "p.json").write_text(json.dumps(_CLAUDE_PROJECT_FIXTURE), encoding="utf-8")
        assert load_any(Path(td, "p.json"))[0].tags[0] == "claude_project", "load_any 认得 projects/*.json"

    # 12. CLI 入口必须把 stdout 锁成 UTF-8（`--stats` 打的是中文，语料带 emoji 时
    #     在 cp936 下会 UnicodeEncodeError）。⚠ **在 Linux／默认 UTF-8 的机器上这条
    #     恒真，在那儿跑不算验过**：变异要在 `PYTHONIOENCODING=gbk` 下跑——删掉
    #     `__main__` 里的 `sys.stdout.reconfigure(...)` 必须转红，加回去复绿。
    assert (getattr(sys.stdout, "encoding", "") or "").lower().replace("-", "") == "utf8", \
        f"CLI 入口没把 stdout 锁成 UTF-8（当前 {sys.stdout.encoding}）：" \
        "中文 Windows（cp936）下 --stats 遇到 emoji 会 UnicodeEncodeError"

    # 13.【变异靶心：--stats 吃目录 + 第 0 步真能发现旧快照】走查台账第四条。
    #     ⚠ **夹具是判据的一部分，不许简化**：一个文件的目录证明不了「吃目录」、
    #     全同一种格式证明不了分发、没有时间戳会让靶心二恒真（那三种都是空跑）。
    #     所以这里的夹具**必须**是：≥3 个文件、≥2 种格式、带真时间戳、含子目录、
    #     另有一份认不出格式的文件、外加一个隐藏 sidecar。
    import io
    import contextlib
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "子目录").mkdir()
        Path(root, "c.json").write_text(json.dumps([_CHATGPT_FIXTURE]), encoding="utf-8")
        Path(root, "w.txt").write_text(_LINECHAT_FIXTURE, encoding="utf-8")
        Path(root, "子目录", "t.md").write_text(_TIMELINE_FIXTURE, encoding="utf-8")
        Path(root, ".weights.json").write_text("{}", encoding="utf-8")  # sidecar，不是语料
        # 13a. 目录被递归收齐、隐藏 sidecar 被跳过
        got = corpus_dir_files(root)
        assert [p.name for p in got] == ["c.json", "w.txt", "t.md"], \
            f"目录要递归收齐并跳过隐藏 sidecar，实际 {[p.name for p in got]}"
        # 13b.【靶心一】目录跑得通，且混合格式各进各的翻译器（不是只认了一种）
        d_entries, d_problems, d_empties = load_dir(root)
        assert d_problems == [] and d_empties == [], f"这三个都该认得出且都有内容：{d_problems} {d_empties}"
        assert {e.tags[0] for e in d_entries} == {"chatgpt", "chatlog", "timeline"}, \
            "三种格式要各进各的翻译器——只认出一种说明目录那层把分发吃掉了"
        # 13c.【靶心二】汇总里要有时间范围，**且最新那个日期旁边跟着那句让人对一对的话**
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = stats(root)
        out = buf.getvalue()
        #      ⚠ 最早 2025-07-14、最新 2026-07-15 **必须是两个不同的日期**，
        #      否则「单独摆出来的是不是最新那个」这条断言恒真（拿最早的顶上也过）
        assert rc == 0 and "时间范围 2025-07-14" in out and "~ 2026-07-15" in out, \
            f"目录汇总要打出时间范围（最早~最晚）：{out}"
        assert "最新的一条停在 2026-07-15" in out and "真的是这天吗" in out, \
            f"最新日期要单独摆出来并跟一句让人对一对——只报区间人眼会滑过去：{out}"
        # 13d.【靶心三】混进认不出格式的文件：报出是哪个文件，不静默跳过、也不整个崩掉
        Path(root, "坏.json").write_text('{"foo": 1}', encoding="utf-8")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = stats(root)
        out = buf.getvalue()
        assert "时间范围" in out, f"一个坏文件不该顶掉整份汇总：{out}"
        assert "坏.json" in out and "没有计入上面的数" in out, \
            f"认不出的要报出是哪个文件、并说明它没进统计（不静默跳过）：{out}"
        assert rc == 1, "有认不出的文件时退出码要非 0（不是崩，是别被当成完全通过）"
        # 13e. 没有时间戳的语料：⚠ 说清这道体检无效，不许沉默
        with tempfile.TemporaryDirectory() as td2:
            Path(td2, "n.txt").write_text("林岸：随便说一句\n星回：嗯", encoding="utf-8")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                stats(Path(td2))
            assert "这道体检无效" in buf.getvalue(), \
                f"一条时间戳都没有时要说清第 0 步这道体检无效：{buf.getvalue()}"
        # 13f.【靶心四·mtime 不许顶掉「最新停在哪天」】2026.08.04 审查侧黑盒抓到：
        #     语料目录里随便一份**今天改过、没有日期**的 md（README／随手记那种）走
        #     mtime 兜底，会把整份语料的「最新一条停在」顶成今天，**第 0 步这道旧快照
        #     体检当场失效且不报错**——而这正是这张卡存在的全部理由。
        #     ⚠ **夹具的两个条件缺一不可**：那份 md 必须**没有任何日期依据**（否则走不到
        #     mtime 那一支）、且它的 mtime 必须**晚于**真语料里的日期（否则这条恒真）。
        #     这里靠 `_TIMELINE_FIXTURE` 里是 2025/2026 年的日期、而临时文件是"刚写的"
        #     来保证后者。变异：把 stats 里 ts 的 `!= "mtime"` 过滤去掉 → 这条红。
        with tempfile.TemporaryDirectory() as td3:
            Path(td3, "t.md").write_text(_TIMELINE_FIXTURE, encoding="utf-8")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                stats(Path(td3))
            base = buf.getvalue()
            assert "最新的一条停在 2026-07-15" in base, f"先确认干净时是对的：{base}"
            Path(td3, "README.md").write_text("# 项目说明\n\n这个文件夹是我导出来的语料\n",
                                              encoding="utf-8")
            #     先确认变异夹具真的打上了：那份 README 确实走了 mtime 兜底
            only_readme = parse_markdown_timeline(
                Path(td3, "README.md").read_text(encoding="utf-8"), "README.md",
                Path(td3, "README.md").stat().st_mtime)
            assert only_readme and all(e.timestamp_source == "mtime" for e in only_readme), \
                "夹具没打上：那份 README 没走 mtime 兜底，下面这条就测不出东西了"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                stats(Path(td3))
            out3 = buf.getvalue()
            assert "最新的一条停在 2026-07-15" in out3, \
                f"一份今天改过的 README 不许把「最新停在哪天」顶掉：{out3}"
            assert "只有文件修改时间" in out3 and "没有算进上面那个日期" in out3, \
                f"落 mtime 的条目要如实披露，不能只是悄悄排除掉：{out3}"
        # 13g.【靶心五·读得进去但解析出 0 条的文件要报出来】同日同一轮黑盒抓到：
        #     原先只挡住**抛异常**的文件，几行散文的「笔记.txt」读得进去、解析出 0 条，
        #     被静默计进「N 个文件」——单文件跑看得见那个 `0 条`，目录汇总里被淹掉。
        #     ⚠ 它跟认不出的分两个桶：这类是常态（语料目录里就该允许有 README），
        #     **只报不红**；退出码仍只由 problems 决定。
        with tempfile.TemporaryDirectory() as td4:
            Path(td4, "t.md").write_text(_TIMELINE_FIXTURE, encoding="utf-8")
            Path(td4, "笔记.txt").write_text("这是我随手写的一份笔记\n没有对话结构\n",
                                             encoding="utf-8")
            assert load_any(Path(td4, "笔记.txt")) == [], \
                "夹具没打上：这份笔记居然解析出东西了，下面测不出静默计入"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc4 = stats(Path(td4))
            out4 = buf.getvalue()
            assert "笔记.txt" in out4 and "一条都没解析出来" in out4, \
                f"解析出 0 条的文件要点名报出来，不许静默计进文件数：{out4}"
            assert "1 个文件参与统计" in out4, \
                f"文件数要只数真正贡献了条目的那些：{out4}"
            assert rc4 == 0, "解析出 0 条是常态（README 那类），只报不红——退出码只由认不出的决定"
        # 13h.【全部时间戳都只有 mtime 这一支】2026.08.04 审查侧终验时补：
        #     这一支原先没有任何断言看着（13f 走的是「有真日期 ＋ 混进 mtime」那支）。
        #     ⚠ 它还得跟抬头那个数**说法一致**：抬头 `带时间戳 0 条` 只数可信的，
        #     这里若再说一句「有 1 条带时间戳」，同一屏两个数打架，人会当成程序错了。
        with tempfile.TemporaryDirectory() as td5:
            Path(td5, "README.md").write_text("# 项目说明\n\n这个文件夹是我导出来的语料\n",
                                              encoding="utf-8")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc5 = stats(Path(td5))
            out5 = buf.getvalue()
            assert "带时间戳 0 条" in out5 and "在这份语料上无效" in out5, \
                f"全是 mtime 时不许编出一个「最新停在 X」，要说这道体检无效：{out5}"
            assert "条带时间戳，但" not in out5, \
                f"抬头说 0 条、下面说「有 1 条带时间戳」，同一屏两个数打架：{out5}"
            assert rc5 == 0, "只有 mtime 不是错（是语料本身没日期），只报不红"

    print("selftest ok（13项断言：六个翻译器 / 两个消费方真接上 / 分发与兜底 / "
          "CLI 入口把 stdout 锁成 UTF-8（⚠ 变异要在 PYTHONIOENCODING=gbk 下跑，"
          "默认 UTF-8 的机器上这条恒真）/ --stats 吃目录且报得出旧快照（⚠ 含：落 mtime 的时间戳不参与「最新停在哪天」、解析出 0 条的文件要点名报出））")


if __name__ == "__main__":
    # 中文 Windows（cp936/GBK）下 stdout 按系统区域编码写，`--stats` 打的说话人、
    # 时间范围是中文，语料里带 emoji 时直接 UnicodeEncodeError
    # （缘由与判据见 memory_init.py 同一处注释）。改的是编码，不是 ensure_ascii。
    # ⚠ 只放在 `__main__` 里：被 import 时不许改掉调用方进程的 stdout。
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--stats", metavar="PATH",
                    help="导入前核对：条数/说话人/时间范围（不打内容）。文件或目录都吃，目录递归")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    elif args.stats:
        sys.exit(stats(args.stats))
    else:
        ap.print_help()
