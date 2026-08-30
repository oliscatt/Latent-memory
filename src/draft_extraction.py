#!/usr/bin/env python3
"""
自动提炼草稿流程参考实现（设计笔记"自动提炼草稿流程"+"风格片段候选机制"两节）。

对话transcript → 候选草稿（Field/StyleExcerpt，全部confirmed=False）→ 交给
persona_template.py的用户确认关卡把关（section 用其十二节骨架的 key，2026.07.31
A-G 通用骨架退役后同步改的）。本文件只负责"生成候选"，不负责"写入生效"——
两件事分开是设计笔记反复强调的底线：自动提炼不能等于无声写入。

两级门限，参考设计笔记 prior art（KI-CO的Topic Gate思路）：
  本地规则先筛（zero-dep，免费）→ 只有规则筛出的候选，才值得再花一次便宜LLM调用去精炼
  没有直接每条消息都烧LLM，机械性初筛留给本地规则做。

隐私二选一（设计笔记"自动提炼草稿流程"）：便宜LLM是可插拔回调（llm_call参数），
不硬编码具体某个API——调用方（产品层）决定传云端便宜LLM的回调还是本地小模型的回调，
或者干脆不传（llm_call=None）就纯用本地规则出候选，语料不出本机。这个选择权在用户，
不在这份参考实现里替用户拍板。

零依赖，stdlib only。
用法：
  python draft_extraction.py --selftest
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from persona_template import Field, StyleExcerpt, PersonaValidationError
from persona_compiler import CompileIssue, PersonaItem, SECTION_KEYS

# 本地规则打分用的信号词——不是详尽列表，是"够不够格再花一次LLM调用"的粗筛。
# 标准来自设计笔记任务5："能看出个性——吐槽/拒绝/不顺从/接得住玩笑，不是平铺直叙的问答"
_PERSONALITY_MARKERS = ("不", "别", "才", "凭什么", "谁说", "才不", "哼", "！", "？")

# ---------- 候选类型契约：**唯一真相，别在任何地方抄第二份** ----------
#
# 一次任务包自足性走查暴露的病根：
# `candidate_kind` 的三个字面值、以及它们各自的必填子字段，**只活在校验器代码里**
# ——提示词、输入清单、schema 三份文件里一个字都没写。于是模型给 `kind` 填了别的值，
# 三道结构检查**一条都没被触发，文件照样过闸**；而"闸门没触发"这件事在输出上
# 与"闸门通过了"**完全一样**，不可见。
#
# ⚠ **这一步的设计前提是"任务包自足"**：零密钥、不联网，把提取外包给用户手上的模型，
# 三份文件交出去就得够。契约写在源码里，等于要求用户的聊天模型去读 `draft_extraction.py`
# ——那个前提就塌了。所以：**契约在这张表里定义一次，issue_codes、schema、提示词
# 全部从它渲染**。新增一个 kind 却忘了写进任务包这件事，从此在结构上不可能发生。
#
# `required` 里的字段名就是校验器真正会去读的那几个（`issue_codes` 从这里取），
# `code` 是缺了它们时报的错误码。fact／quote 没有额外结构要求，但**必须列在这里**
# ——枚举是"哪些 kind 会被结构检查"的全集，漏一个就等于让它悄悄绕过。
CANDIDATE_KINDS = {
    "fact": {
        "label": "事实",
        "required": (),
        "code": None,
        "rule": "一条可核对的事实（住在哪、在做什么、什么时候开始的）。没有额外必填字段。",
    },
    "quote": {
        "label": "原话",
        "required": (),
        "code": None,
        "rule": "`text` 必须逐字落在 `source_span` 圈出的那段里，否则报 QUOTE_NOT_VERBATIM。",
    },
    "milestone": {
        "label": "里程碑",
        "required": ("event", "reading", "current_state"),
        "code": "MILESTONE_INCOMPLETE",
        "rule": "三段都要给，缺一段就整条不过：`event`（发生了什么）／`reading`"
                "（这件事怎么读）／`current_state`（现在是什么状态）。",
    },
    "naming": {
        "label": "称呼",
        "required": ("user_to_ai", "ai_to_user"),
        "code": "NAMING_NOT_BIDIRECTIONAL",
        "rule": "双向都要给：`user_to_ai`（用户怎么叫 AI）／`ai_to_user`（AI 怎么叫用户），"
                "两个都是字符串数组，缺一边就整条不过。",
    },
    "style_dialogue": {
        "label": "风格片段",
        "required": ("turns",),
        "code": "STYLE_NOT_DIALOGUE",
        "rule": "`turns` 是对话轮次数组，每轮 `{\"speaker\": \"谁\", \"text\": \"说了什么\"}`；"
                "**至少 2 轮、至少 2 个不同且非空的 speaker**（学的是来回接话的风格，"
                "不是抄一句台词）。每轮 `text` 也必须逐字落在 `source_span` 里。",
    },
}

# 来源交代表的键名。⚠ **写错键名不会被猜对**：校验器认的就是这几个字面值，
# 填 `item_ids` 之类会被指名报错（`SOURCE_ACCOUNTING_FIELD_MISSING`），不做同义兼容
# ——猜是另一种静默，而且下一个模型换个写法又猜不中。
ACCOUNTING_IDS_KEY = "candidate_item_ids"
ACCOUNTING_NONE_KEY = "no_supported_candidate"


# ---------- 锚表：**起止由我们算，模型只挑锚号** ----------
#
# 外部模型实测暴露的病根：
# `source_span` 要模型自己数出字符偏移，而聊天模型数不准——两家模型 7 条里 6 条圈错，
# 而 `evidence` **全部逐字正确**，错的只有区间。⚠ 两家的错法还不是一种
# （甲整体 -28～-35、乙 +7／+4／-1），所以"提示词里补一句数法"只对得上其中一种。
# 修法（2026.08.06 拍板）：**任务包侧给可数的锚，让它挑，不让它数。**
#
# ⚠ **粒度定成"按空行分段、标题行自成一条"**（8.2 留给执行窗，理由写在任务卡 9.0）：
# 再往细切会让 `style_dialogue` 那条硬要求（至少 2 轮、2 个不同 speaker，每轮 `text`
# 都要落在同一段区间里）**一条锚都装不下**，等于用锚表把这个 kind 判死。
# 粒度粗的代价在这里不成立——起止是我们算的，模型要的只是一个装得下 evidence 的框。
#
# ⚠ **锚必须无缝覆盖全文、互不重叠**：留了缝就等于有一段原文任何锚都装不下，
# 而模型看不见这件事，只会以为是自己挑错了（又一次"报错指错方向"）。
_ANCHOR_HEAD_CHARS = 24


def anchor_table(source):
    """把整份原文切成锚：[{"anchor": "A1", "start": 起, "end": 止, "head": 首行}, ...]

    ⚠ **唯一真相**：任务包清单里写出去的锚、和校验器把锚号换算回区间用的锚，
    **都调这一个函数**。两边各算一份就等于把契约抄了两遍（上一张卡的病根）。
    """
    anchors = []

    def flush(start, end):
        """收一条锚；返回是否真的收下了（没收下时调用方不许推进游标，否则会留缝）。"""
        if end <= start:
            return True                       # 空区间，本来就无缝
        chunk = source[start:end]
        if not chunk.strip():                 # 纯空白：并进上一条，绝不单独成锚
            if not anchors:
                return False                  # 文件开头的空白，留给下一条锚一起吃掉
            anchors[-1]["end"] = end
            return True
        head = chunk.strip().splitlines()[0]
        anchors.append({
            "anchor": f"A{len(anchors) + 1}", "start": start, "end": end,
            "head": head[:_ANCHOR_HEAD_CHARS] + ("…" if len(head) > _ANCHOR_HEAD_CHARS else ""),
        })
        return True

    cursor, at = 0, 0
    for line in source.splitlines(keepends=True):
        line_end = at + len(line)
        stripped = line.strip()
        if not stripped:                      # 空行＝段落边界；空行本身并进它收尾的那一段
            if flush(cursor, line_end):
                cursor = line_end
        elif stripped.startswith("#"):        # 标题行自成一条锚
            if flush(cursor, at):
                cursor = at
            if flush(cursor, line_end):
                cursor = line_end
        at = line_end
    flush(cursor, len(source))
    return anchors


def _anchor_index(source):
    return {a["anchor"]: a for a in anchor_table(source)}


def _anchor_table_md_hint():
    """提示词里那段讲锚表怎么用的话。⚠ 键名与这里的字面值同出一处。"""
    return f"""## 二、**不要自己数字符**：挑一个锚号

⚠ **这是这份任务里最容易做错的一步，也是唯一一条你不需要动脑筋的一步。**

《人格候选输入清单.json》里每个来源都附了一张 **锚表 `anchors`**，每条锚长这样：

```json
{{"anchor": "A3", "start": 42, "end": 118, "head": "林岸：今天工作好累"}}
```

`start`／`end` **是我们替你算好的字符区间**，锚已经把整份原文无缝切完、互不重叠。
你要做的只是：**找到你摘的那段话落在哪条锚里，把那条锚的 `anchor` 值填进
`source_anchor`**。⚠ **不要自己数字符偏移**——实测表明这一步的错误率极高，
而且错的量级各家不同，没有哪条数法能救。

- 填了 `source_anchor` 就**不用再填 `source_span`**，校验器会替你换算。
- `evidence` **必须整段落在你选的那条锚里**。装不下就说明**该换一条锚**
  （多半是相邻的那条），⚠ 不是去改 `evidence`。
- 锚号填了表里没有的值会被报 `SOURCE_ANCHOR_UNKNOWN`，**不会被猜成相近的那条**。
- ⚠ **只有一种情况允许你直填 `source_span`**：你能真的跑代码去 `find()` 定位。
  那条老路仍然接受，规则不变（两个整数、不许越界）。**拿不准就挑锚号。**
- 两个都不给会被报 `SOURCE_LOCATION_MISSING`；两个都给且对不上时**以锚为准**，
  并给一条 `SOURCE_ANCHOR_SPAN_CONFLICT` 警告——不会静默丢掉你写的那个。
"""


def _kind_table_md():
    """契约表 → 提示词里那张 Markdown 表。渲染的是同一份 CANDIDATE_KINDS。"""
    lines = ["| `candidate_kind` | 是什么 | 除通用字段外还必须给 | 缺了报什么 |",
             "|---|---|---|---|"]
    for kind, spec in CANDIDATE_KINDS.items():
        extra = "、".join(f"`{f}`" for f in spec["required"]) or "（无）"
        lines.append(f"| `{kind}` | {spec['label']} | {extra} | "
                     f"{spec['code'] or '—'} |")
    return "\n".join(lines)


def _kind_rules_md():
    return "\n".join(f"- **`{kind}`（{spec['label']}）**：{spec['rule']}"
                     for kind, spec in CANDIDATE_KINDS.items())


def extraction_prompt():
    """交给用户模型的提示词。**从 CANDIDATE_KINDS 渲染，不手抄一份。**"""
    return f"""# 人格候选提取任务

候选只许从输入来源提取，不许创作。
找不到就返回空数组，不得用通用句补完整度。

逐个读取《人格候选输入清单.json》里的语料。每个候选必须填写真实 `source_ref`、
**一个锚号 `source_anchor`**（见第二节）和逐字 `evidence`。只返回符合
《人格候选结果.schema.json》的 JSON，不要输出任何解释文字。

## 一、每条候选的通用必填字段

`item_id`（本次结果内唯一）／`text`／`section`（取值见 schema 的 enum）／
`source_ref`／`candidate_kind`／`evidence`，
再加**定位**：`source_anchor`（清单锚表里的锚号，**推荐**）
**或** `source_span`（两个整数的字符区间，见第二节末尾那条）——两者给其一。

{_anchor_table_md_hint()}
## 三、`candidate_kind` 只能取下面这几个值

⚠ **填枚举以外的值（包括中文自由值）不会被拦下来，但那条候选的结构检查一条都不会跑**
——校验器会给一条 `CANDIDATE_KIND_UNCHECKED` 警告说明"这条没被检查过"。
**不要靠它兜底：请按下表选一个准确的值。**

{_kind_table_md()}

{_kind_rules_md()}

⚠ 上面几条规则里写的 `source_span`，**在你挑锚号时就是那条锚圈出的区间**——
同一段文字，只是换了个填法：`turns` 的每一轮、`quote` 的 `text`，
都必须落在**你选的那条锚**里。

## 四、来源交代表 `source_accounting`

**每一个输入来源都要在这里出现一条**，字段名逐字照抄，不许改写：

```json
{{"source_accounting": [
  {{"source_ref": "<清单里的那个路径，逐字照抄>",
   "{ACCOUNTING_IDS_KEY}": ["<这个来源支撑的候选 item_id>"]}},
  {{"source_ref": "<另一个来源>", "{ACCOUNTING_NONE_KEY}": true}}
]}}
```

- `{ACCOUNTING_IDS_KEY}` 里的每一个 ID **必须真的对应 `items` 里的一条候选**。
  候选删掉了就把它从这里一起删掉——挂着一个不存在的 ID 会被报
  `SOURCE_ACCOUNTING_ID_UNKNOWN`。
- 这个来源确实提不出候选，就写 `"{ACCOUNTING_NONE_KEY}": true`，**不要留空、不要跳过**。
- ⚠ **键名写错不会被自动纠正**：填成 `item_ids`、`ids` 这类会被报
  `SOURCE_ACCOUNTING_FIELD_MISSING` 并点名说是键名问题。
- ⚠ **「已交代」≠「会进人格文件」**：这里登记的是"这个来源你看过了、给出了结论"。
  候选最终用不用，由用户在后面那一步（十二节选择题）定——**他可以整节弃用**，
  那时这些来源**仍然算已交代**，不必回头改这份 JSON。
  所以：**该给的候选照给，不要为了"让每个来源都有东西"硬凑**；
  确实提不出来就写 `{ACCOUNTING_NONE_KEY}: true`，那也是一个正当结论。

## 五、`evidence` 必须是**原始字符**

不许做任何加工：不要 HTML 转义（`>` 就写 `>`，不要写成 `&gt;`；`<`、`&` 同理），
不要把直引号改成弯引号，不要改空白与换行。
校验器是**逐字**比对原文的，改一个字符就整条不过——而报出来的错是
「evidence 不是原文逐字」，看不出是转义干的。

## 六、一份最小的完整示例（照着这个形状填）

```json
{{
  "items": [
    {{"item_id": "corpus:1", "text": "住在杭州，在做古地图修复。",
     "section": "user", "source_ref": "<清单里的路径>",
     "source_anchor": "A1", "candidate_kind": "fact",
     "evidence": "<A1 那条锚圈出的原文里的一段，逐字>"}},
    {{"item_id": "corpus:2", "text": "第一次独立修完一张海图。",
     "section": "milestones", "source_ref": "<清单里的路径>",
     "source_anchor": "A3", "candidate_kind": "milestone",
     "evidence": "<逐字原文>",
     "event": "独立修完第一张海图", "reading": "从学徒变成能独立交付",
     "current_state": "已经完成，现在接第二张"}},
    {{"item_id": "corpus:3", "text": "互相拌嘴但接得住。",
     "section": "style", "source_ref": "<清单里的路径>",
     "source_anchor": "A4", "candidate_kind": "style_dialogue",
     "evidence": "<逐字原文>",
     "turns": [{{"speaker": "用户", "text": "<逐字原文里的这一轮>"}},
               {{"speaker": "AI", "text": "<逐字原文里的下一轮>"}}]}}
  ],
  "source_accounting": [
    {{"source_ref": "<清单里的路径>",
     "{ACCOUNTING_IDS_KEY}": ["corpus:1", "corpus:2", "corpus:3"]}}
  ]
}}
```

⚠ 上面三条填的都是 `source_anchor`——**这是推荐形态**。你若真的能跑代码去
`find()` 定位，把那一项换成 `"source_span": [起, 止]` 也照样收，其余字段不变。
"""


# 模块级常量保留：外部若引用过它，渲染结果与 extraction_prompt() 同一份，不是第二份抄写
EXTRACTION_PROMPT = extraction_prompt()


@dataclass(frozen=True)
class ExtractionPackage:
    prompt_path: Path
    manifest_path: Path
    schema_path: Path


_SUBFIELD_SCHEMA = {
    "event": {"type": "string", "minLength": 1},
    "reading": {"type": "string", "minLength": 1},
    "current_state": {"type": "string", "minLength": 1},
    "user_to_ai": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    "ai_to_user": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    "turns": {
        "type": "array", "minItems": 2,
        "items": {"type": "object", "required": ["speaker", "text"],
                  "properties": {"speaker": {"type": "string", "minLength": 1},
                                 "text": {"type": "string", "minLength": 1}}},
    },
}


def _candidate_schema():
    """结果 schema。**enum 与按 kind 的条件必填全部从 CANDIDATE_KINDS 渲染**——
    这是"任务包自足"那条前提落到 schema 上的样子：模型不读源码也能知道
    `style_dialogue` 要带 `turns`、`milestone` 要带三段。

    ⚠ 原先这里 `candidate_kind` 只写 `{"type": "string"}`、`source_accounting` 只写
    `{"type": "array"}`——**校验器最重的三道结构检查在 schema 里一个字都没有**，
    模型当然填不对（2026.08.04 走查现场）。"""
    item_props = {
        "item_id": {"type": "string"},
        "text": {"type": "string"},
        "section": {"enum": list(SECTION_KEYS)},
        "source_ref": {"type": "string"},
        # ⚠ 定位两条路：锚号（推荐）／直填区间（留给能跑代码去 find() 的模型，
        # 现场事实是它 30 条全对，一刀切禁掉是误伤——任务卡 9.0）。
        # required 里只留两者「必居其一」，不把 source_span 钉死成必填。
        "source_anchor": {"type": "string", "pattern": r"^A\d+$"},
        "source_span": {
            "type": "array", "prefixItems": [
                {"type": "integer"}, {"type": "integer"}],
            "minItems": 2, "maxItems": 2},
        "candidate_kind": {"enum": list(CANDIDATE_KINDS)},
        "evidence": {"type": "string"},
    }
    item_props.update({name: dict(schema) for name, schema in _SUBFIELD_SCHEMA.items()})
    # 按 kind 的条件必填：if/then，不是把子字段一律设成 required（那会逼 fact 也填 turns）
    conditionals = [{
        "if": {"required": ["candidate_kind"],
               "properties": {"candidate_kind": {"const": kind}}},
        "then": {"required": list(spec["required"])},
    } for kind, spec in CANDIDATE_KINDS.items() if spec["required"]]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["items", "source_accounting"],
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["item_id", "text", "section", "source_ref",
                                 "candidate_kind", "evidence"],
                    "properties": item_props,
                    # 定位二选一：给锚号，或者给区间；一个都不给报 SOURCE_LOCATION_MISSING
                    "anyOf": [{"required": ["source_anchor"]},
                              {"required": ["source_span"]}],
                    "allOf": conditionals,
                },
            },
            "source_accounting": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["source_ref"],
                    "properties": {
                        "source_ref": {"type": "string"},
                        ACCOUNTING_IDS_KEY: {"type": "array", "items": {"type": "string"}},
                        ACCOUNTING_NONE_KEY: {"const": True},
                    },
                    # 两者必居其一：登记 ID，或者明写这个来源提不出候选
                    "anyOf": [{"required": [ACCOUNTING_IDS_KEY]},
                              {"required": [ACCOUNTING_NONE_KEY]}],
                },
            },
        },
        "additionalProperties": False,
    }


def build_extraction_package(manifest, out_dir):
    """导出给用户当前模型的本地任务包；不调用网络、不读取 API key。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prompt_path = out / "人格候选提取提示.md"
    manifest_path = out / "人格候选输入清单.json"
    schema_path = out / "人格候选结果.schema.json"
    prompt_path.write_text(extraction_prompt(), encoding="utf-8")
    # ⚠ **锚表在这里落进清单**（岔口 3）：起止是我们算的，模型只挑锚号。
    # ⚠ 调的是 `anchor_table()` 那一个函数——校验器换算锚号时调的也是它，
    # 两边不是两份（唯一真相那条，上一张卡的病根就是抄了第二份）。
    sources = [{
        "source_ref": str(path),
        "sha256": manifest.source_hashes.get(str(path), ""),
        "kind": "corpus",
        "anchors": anchor_table(path.read_text(encoding="utf-8")),
    } for path in manifest.corpus_files]
    manifest_path.write_text(json.dumps({"sources": sources}, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    schema_path.write_text(json.dumps(_candidate_schema(), ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return ExtractionPackage(prompt_path, manifest_path, schema_path)


def _nonempty(value):
    """必填子字段"给了没有"的统一判据：空串、空白串、空数组都算没给。"""
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    return value is not None and value is not False


def issue_codes(candidate):
    """返回单条候选的结构错误码；不依赖模型或外部 schema 库。

    ⚠ **必填字段名与错误码都取自 CANDIDATE_KINDS**，不在这里另写一份——那张表同时
    是任务包里那张表的来源，两边只有一份真相（2026.08.04 走查现场那条卡的病根就是
    "契约只活在这个函数里"）。
    ⚠ **未知 kind 在这里返回空集，不代表它没问题**：那条路由到
    `CANDIDATE_KIND_UNCHECKED` 警告——判的是"有没有说这条没被检查"，不是拦不拦。"""
    kind = candidate.get("candidate_kind")
    spec = CANDIDATE_KINDS.get(kind)
    if spec is None:
        return set()
    issues = set()
    if any(not _nonempty(candidate.get(name)) for name in spec["required"]):
        issues.add(spec["code"])
    if kind == "style_dialogue" and not issues:
        # 结构要求比"给了 turns"更严：学的是来回接话，不是抄一句台词
        turns = candidate.get("turns") or []
        speakers = {str(turn.get("speaker", "")).strip() for turn in turns}
        if len(turns) < 2 or "" in speakers or len(speakers) < 2:
            issues.add(spec["code"])
    return issues


#: 报错正文里最多列几处真实位置——多了正文会被淹没，剩下的只报个数。
_SPAN_HITS_SHOWN = 5


def _find_occurrences(source, needle, limit=_SPAN_HITS_SHOWN + 1):
    """返回 needle 在 source 里的全部出现位置 [(起, 止), ...]，最多取 limit 处。

    ⚠ **只用于报错，不用于自动定位**：出现两次时该算哪一次没有答案。
    ⚠ **这就是「让 source_span 变可选、校验器自己 find() 定位」那条路被否掉的理由**
    ——自动定位会把「模型必须声明它从哪儿摘的」这道声明变成我们替它猜。
    锚号那条路没有这个问题：**声明仍然出自模型（它选的锚）**，只是从"数区间"
    换成了"挑一个已经算好的区间"。这里把每一处都摆出来给人看，**不替它选**。
    """
    hits, at = [], source.find(needle)
    while at >= 0 and len(hits) < limit:
        hits.append((at, at + len(needle)))
        at = source.find(needle, at + 1)
    return hits


def _format_spans(hits):
    shown = "、".join(f"[{a}, {b}]" for a, b in hits[:_SPAN_HITS_SHOWN])
    return shown + ("（还有更多，只列前几处）" if len(hits) > _SPAN_HITS_SHOWN else "")


def _issue(code, message, item_id=None, severity="blocking"):
    return CompileIssue(code, severity, message, (item_id,) if item_id else ())


def validate_candidate_result(result, manifest):
    """验证模型结果的结构、逐字证据和来源交代，返回安全候选与问题。"""
    issues = list(manifest.issues)
    items = []
    known_sources = {str(path): path for path in manifest.corpus_files}
    seen_ids = set()
    raw_items = result.get("items")
    if not isinstance(raw_items, list):
        return [], issues + [_issue("CANDIDATE_ITEMS_INVALID", "items 必须是数组")]

    for raw in raw_items:
        item_id = str(raw.get("item_id", ""))
        if not item_id or item_id in seen_ids:
            issues.append(_issue("CANDIDATE_ID_INVALID", "候选 ID 缺失或重复", item_id or None))
            continue
        seen_ids.add(item_id)
        # ⚠ **让"这条没被结构检查"这件事可见**（2026.08.05，缺陷 B 的真正修法）：
        # kind 填了枚举外的值时，三道结构检查一条都不会跑，而输出与"检查通过"
        # 长得一模一样——不可见的失效正是这张卡的病。
        # **只出 warning 不 blocking**（维护者拍板）：要挡的不是"填错"，是"填错了
        # 也没人说"；blocking 会把将来合法的新 kind 一并堵死。
        if raw.get("candidate_kind") not in CANDIDATE_KINDS:
            issues.append(_issue(
                "CANDIDATE_KIND_UNCHECKED",
                f"candidate_kind={raw.get('candidate_kind')!r} 不在已知类型里"
                f"（{'、'.join(CANDIDATE_KINDS)}）——**这条候选没有被任何结构检查**，"
                "它是否符合那类候选的硬要求，本次一个字都没验过。",
                item_id, severity="warning"))
        structural = issue_codes(raw)
        if structural:
            issues.extend(_issue(code, f"候选结构不完整：{code}", item_id)
                          for code in sorted(structural))
            continue
        section = raw.get("section")
        if section not in SECTION_KEYS:
            issues.append(_issue("SECTION_UNKNOWN", f"未知人格节：{section}", item_id))
            continue
        ref = str(raw.get("source_ref", ""))
        path = known_sources.get(ref)
        if path is None:
            issues.append(_issue("SOURCE_UNKNOWN", f"候选来源不在输入清单：{ref}", item_id))
            continue
        source = path.read_text(encoding="utf-8")
        # ---- 定位：锚号优先，直填 span 仍然接受（岔口 3，任务卡 9.0 三件裁决）----
        # ⚠ 锚号这条路存在的理由：`source_span` 要模型自己数字符，而聊天模型数不准
        # （实测 7 条里 6 条圈错，evidence 全部逐字正确）。锚的起止是我们算的。
        # ⚠ 直填那条路**不禁**：能跑代码去 find() 的模型 30 条全对（卡第三节），
        # 一刀切禁掉是为了修 A 类模型去打断 B 类模型的腿。
        anchor_id = raw.get("source_anchor")
        span = raw.get("source_span")
        anchor_id = str(anchor_id).strip() if isinstance(anchor_id, str) else anchor_id
        has_span = isinstance(span, list) or span is not None
        if anchor_id:
            anchors = _anchor_index(source)
            picked = anchors.get(anchor_id)
            if picked is None:
                # ⚠ **新码，不塞进 `SOURCE_SPAN_INVALID`**：那条管的是"给了区间但越界／
                # 不是两个整数"，锚号不存在既不越界也不是格式问题——两格混成一格
                # 正是这张卡从头到尾在治的病（任务卡第六节写死了不动那条码）。
                # ⚠ **不猜相近的锚号**：猜是另一种静默（交代表键名那条同理）。
                names = "、".join(a["anchor"] for a in anchors.values()) or "（这个来源一条锚都没有）"
                issues.append(_issue(
                    "SOURCE_ANCHOR_UNKNOWN",
                    f"source_anchor={anchor_id!r} 不在 {ref} 的锚表里。"
                    f"这个来源合法的锚号是：{names}。"
                    "请回《人格候选输入清单.json》那张 anchors 表里挑一条——我们不替你猜相近的。",
                    item_id))
                continue
            start, end = picked["start"], picked["end"]
            if has_span and list(span or ()) != [start, end]:
                # ⚠ **不静默丢掉它写的那个 span**：静默失效（输出跟"通过了"长得一样）
                # 是这个仓库反复栽的形状。以锚为准，但要说出来。
                issues.append(_issue(
                    "SOURCE_ANCHOR_SPAN_CONFLICT",
                    f"你同时给了 source_anchor={anchor_id!r}（区间 [{start}, {end}]）和 "
                    f"source_span={span!r}，两者对不上。**本次以锚为准**，"
                    "你填的 source_span 没有被采用——不是被悄悄忽略，是在这里告诉你。",
                    item_id, severity="warning"))
        elif has_span:
            if (not isinstance(span, list) or len(span) != 2
                    or not all(isinstance(value, int) for value in span)):
                issues.append(_issue("SOURCE_SPAN_INVALID", "来源区间必须是两个整数", item_id))
                continue
            start, end = span
            if start < 0 or end < start or end > len(source):
                issues.append(_issue("SOURCE_SPAN_INVALID", "来源区间越界", item_id))
                continue
        else:
            # ⚠ **口径变化，记在任务卡 9.0**：改动前"什么都没给"报的是
            # `SOURCE_SPAN_INVALID`。现在它只管"给了 span 但越界／不是两个整数"，
            # "压根没给定位"独立成码，且正文要**同时给出两条出路**。
            issues.append(_issue(
                "SOURCE_LOCATION_MISSING",
                "这一条没说它是从哪儿摘的：`source_anchor` 和 `source_span` 一个都没给。"
                "两条出路挑一条——**推荐**去《人格候选输入清单.json》的 anchors 表里挑一个"
                "锚号填进 `source_anchor`（起止我们已经算好了）；"
                "你若能跑代码定位，也可以直填 `source_span`（两个整数）。",
                item_id))
            continue
        excerpt = source[start:end]
        evidence = str(raw.get("evidence", ""))
        if not evidence or evidence not in excerpt:
            # ⚠ **报错分档**（2026.08.05，《任务包要模型自己数字符区间》岔口 1）：
            # evidence 逐字没问题、只是区间圈错，跟 evidence 真被转义／改了引号，
            # 原来**报的是同一句话**——而提示词整整一节都在讲转义，人读到那句必然
            # 去查自己的 evidence 有没有被加工，真因却在 span。实测 7 条里 6 条
            # 是前者（两家模型 evidence 全部逐字正确，错的只有区间）。
            # ⚠ **这里不放宽任何要求**：区间圈错照样 blocking、照样不进候选池，
            # 只是把话说对，并且把 evidence 的真实位置一起给出来。
            # ⚠ **空 evidence 必须先单独挡掉**（2026.08.05 审核轮补）：`"".find` 在任何
            # 字符串里都命中，落进下面那档会报出「你的 evidence 逐字没问题」＋
            # `[0, 0]、[1, 1]…` 一串空区间——**根本没给 evidence 的那条，被告知去改区间**，
            # 正是这张卡要消灭的「报错指错方向」，而且比改之前更误导。
            # 码沿用 `EVIDENCE_NOT_VERBATIM`（分档前这一档就报它），不新增码、不动契约。
            if not evidence:
                issues.append(_issue(
                    "EVIDENCE_NOT_VERBATIM",
                    "这一条没给 evidence（字段缺失或是空串）——不是定位的问题，"
                    "也不是文本对不上：请从原文里原样复制一段能支撑这条候选的文字，"
                    "并让它落在你选的那条锚（或你直填的 source_span）里。",
                    item_id))
                continue
            hits = _find_occurrences(source, evidence)
            if hits:
                # ⚠ **锚模式下这句话要换**（岔口 3 判据锚-C）：叫一个挑锚号的模型
                # 「去改区间」是让它去做我们刚刚才免掉的那件事——它无从执行，
                # 又一次「报错指错方向」。锚模式下要叫它**改选锚号**，并点名它选了哪条。
                if anchor_id:
                    where = (f"你选的锚是 {anchor_id}（区间 [{start}, {end}]），"
                             f"你的 evidence 不在这条锚里。"
                             f"它在 {ref} 里的真实位置是 {_format_spans(hits)}——"
                             f"对着锚表挑一条装得下它的锚（多半就是相邻的那条），"
                             f"⚠ 要改的是锚号，不是 evidence。")
                else:
                    where = (f"span 圈错了：你的 evidence 逐字没问题，但不在你声明的区间里。"
                             f"它在 {ref} 里的真实位置是 {_format_spans(hits)}，"
                             f"你声明的 source_span 是 [{start}, {end}]。"
                             f"要改的是区间，不是 evidence。"
                             f"⚠ 拿不准就别自己数：改填 `source_anchor`，锚表里的起止是算好的。")
                issues.append(_issue(
                    "SOURCE_SPAN_MISPLACED",
                    where
                    + ("⚠ 它在原文里出现了不止一处，上面每一处都列了出来；"
                       "该算哪一处只有你知道，校验器不替你选。" if len(hits) > 1 else ""),
                    item_id))
            else:
                issues.append(_issue(
                    "EVIDENCE_NOT_VERBATIM",
                    "evidence 在整份原文里一个字都对不上——不是区间的问题，"
                    "是这段文本本身跟原文不一致（常见于 HTML 转义、直/弯引号互换、"
                    "空白或换行被改过）。请把原文原样复制过来。",
                    item_id))
            continue
        kind = raw.get("candidate_kind")
        text = str(raw.get("text", ""))
        if kind == "quote" and text not in excerpt:
            issues.append(_issue("QUOTE_NOT_VERBATIM", "原话不在声明的来源区间", item_id))
            continue
        if kind == "style_dialogue" and any(
                str(turn.get("text", "")) not in excerpt for turn in raw.get("turns", ())):
            issues.append(_issue("STYLE_NOT_VERBATIM", "风格对话有轮次无法逐字回指", item_id))
            continue
        items.append(PersonaItem(
            item_id=item_id, text=text, section=section, source_type="corpus",
            source_ref=ref, source_span=(start, end),
            source_hash=manifest.source_hashes.get(ref, ""), operation="add",
            original_text=evidence, proposed_text=text,
            confidence=str(raw.get("confidence", "model_candidate")), confirmed=False,
            conflicts_with=tuple(raw.get("conflicts_with", ())),
            group_id=str(raw.get("group_id", f"corpus:{item_id}"))))

    accounting = result.get("source_accounting")
    accounted, malformed = set(), set()
    if accounting is not None and not isinstance(accounting, list):
        issues.append(_issue("SOURCE_ACCOUNTING_INVALID",
                             "source_accounting 必须是数组（每个输入来源一条记录）"))
    if isinstance(accounting, list):
        for pos, record in enumerate(accounting, 1):
            if not isinstance(record, dict):
                issues.append(_issue("SOURCE_ACCOUNTING_FIELD_MISSING",
                                     f"source_accounting 第 {pos} 条不是对象"))
                continue
            ref = str(record.get("source_ref", ""))
            if ref not in known_sources:
                issues.append(_issue("SOURCE_UNKNOWN", f"来源交代表含未知来源：{ref}"))
                continue
            candidate_ids = record.get(ACCOUNTING_IDS_KEY) or []
            no_candidate = record.get(ACCOUNTING_NONE_KEY) is True
            if not candidate_ids and not no_candidate:
                # ⚠ **指名报错，不再静默 `continue`**（2026.08.05，缺陷 A 的修法）：
                # 原先这里一声不吭跳过，最后只报一句"以下输入来源没有候选或无候选
                # 说明"——**把人指向了错误的方向**（真因是键名对不上，走查现场是靠
                # 读源码才定位到的）。所以这条要说清是**字段**的问题，并**把这条记录
                # 里实际有的键回显出来**，让人一眼看出自己写的是什么。
                # ⚠ **明确不做"猜键名"**（见到 `item_ids` 就当成 `candidate_item_ids`）
                # ——猜是另一种静默，且下一个模型换个写法又猜不中。
                other = "、".join(f"`{k}`" for k in record if k != "source_ref") or "（没有别的键）"
                issues.append(_issue(
                    "SOURCE_ACCOUNTING_FIELD_MISSING",
                    f"source_accounting 第 {pos} 条（{ref}）既没有 `{ACCOUNTING_IDS_KEY}` "
                    f"也没有 `{ACCOUNTING_NONE_KEY}: true`；这条记录里实际有的键是："
                    f"{other}。⚠ 键名要逐字写成 `{ACCOUNTING_IDS_KEY}`，"
                    "我们不猜别的写法。"))
                malformed.add(ref)
                continue
            # ⚠ 悬空 ID（2026.08.05，缺陷 C）：候选被删掉后交代表还挂着它的 ID，
            # 而原先这两条循环互不校验——`items` 里一条都没有也照样算"已交代"。
            # ⚠ **不许把它当成"这个来源没交代"**：那会把"模型笔误"和"模型根本没干活"
            # 混成同一个错，报错又指错方向，等于让缺陷 A 的病在修 C 的时候复发。
            dangling = [str(i) for i in candidate_ids if str(i) not in seen_ids]
            if dangling:
                issues.append(_issue(
                    "SOURCE_ACCOUNTING_ID_UNKNOWN",
                    f"{ref} 的交代里这些候选 ID 在 items 里根本不存在："
                    + "、".join(dangling)
                    + "。多半是那条候选后来被删了、或者 ID 写错了——"
                      "**这不是「没交代」**，是交代指向了一条不存在的候选。"))
            accounted.add(ref)
    missing = sorted(set(known_sources) - accounted)
    if missing:
        plain = [ref for ref in missing if ref not in malformed]
        bad = [ref for ref in missing if ref in malformed]
        parts = []
        if plain:
            parts.append("以下输入来源没有候选或无候选说明：" + "；".join(plain))
        if bad:
            # 这几条**登记过**，只是那条记录的字段不合格——别把它们混进上面那句话里，
            # 那正是走查现场把人指错方向的那句
            parts.append(f"另有 {len(bad)} 条来源登记过，但那条记录的字段不合格"
                         f"（见上面的 SOURCE_ACCOUNTING_FIELD_MISSING）：" + "；".join(bad))
        issues.append(_issue("SOURCE_ACCOUNTING_INCOMPLETE", "；".join(parts)))
    if known_sources and not raw_items and not missing:
        issues.append(_issue(
            "NO_SPECIFIC_CANDIDATES", "所有来源均已交代，但没有可支持的具体候选",
            severity="warning"))
    return items, issues


def load_candidate_result(path, manifest):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_candidate_result(data, manifest)


def parse_transcript(text):
    """最简对话transcript解析：每行"说话人：内容"格式 → [(speaker, text), ...]。
    是设计笔记"通用导入"设想的中间格式({时间戳,说话人,文本,标签?})的简化子集，
    时间戳/标签留给真正接入具体导出格式（微信/ChatGPT json）时的翻译器层再补，
    这里只做提炼逻辑本身的参考实现，不重做"通用导入"那节已经想清楚的格式适配。"""
    turns = []
    for line in text.splitlines():
        line = line.strip()
        if not line or "：" not in line and ":" not in line:
            continue
        sep = "：" if "：" in line else ":"
        speaker, _, content = line.partition(sep)
        speaker, content = speaker.strip(), content.strip()
        if speaker and content:
            turns.append((speaker, content))
    return turns


def style_candidate_score(text):
    """本地规则打分：零依赖，不调任何模型。分越高越像"体现个性"的片段。
    纯粹是初筛门限——够格的候选才值得再花一次便宜LLM调用去精炼/验证。"""
    score = sum(1 for m in _PERSONALITY_MARKERS if m in text)
    if 4 <= len(text) <= 40:  # 太短没内容，太长不像"一句话立住"的锚点
        score += 1
    return score


def extract_style_candidates(turns, speaker, pool="daily", topN=5, llm_call=None):
    """从transcript里挑speaker说的话，按style_candidate_score排序取topN，
    包成StyleExcerpt候选（confirmed=False，disclaimer走dataclass默认值不能省）。

    llm_call: 可选回调 str->str，传入本地规则筛出的候选文本，返回精炼后的版本
    （比如更完整的上下文、更准的措辞）。不传就是纯本地规则出候选——这是"隐私二选一"
    落到代码里的样子：调用方决定要不要引入LLM这一步，这里不替调用方做选择。"""
    speaker_turns = [t for who, t in turns if who == speaker]
    ranked = sorted(speaker_turns, key=style_candidate_score, reverse=True)[:topN]
    candidates = []
    for text in ranked:
        if llm_call is not None:
            text = llm_call(text)
        candidates.append(StyleExcerpt(text=text, pool=pool, confirmed=False))
    return candidates


def draft_field(field_id, section, label, value, size_limit=500, llm_call=None):
    """单个字段草稿。llm_call同上——可选的精炼步骤，不传就原样打包成草稿。
    confirmed强制False：这里只产出候选，不产出生效内容。"""
    if llm_call is not None:
        value = llm_call(value)
    return Field(id=field_id, section=section, label=label, value=value,
                 size_limit=size_limit, source="draft", confirmed=False)


def apply_confirmed(persona, drafts, confirmed_ids):
    """把用户勾选过的草稿（按id匹配）真正写进persona——用户确认关卡的落地点。
    drafts可以是Field或StyleExcerpt的混合列表；StyleExcerpt没有id，按对象身份匹配。
    未被confirmed_ids选中的草稿直接丢弃，不留在系统里"等下次自动生效"。"""
    applied = []
    for d in drafts:
        is_field = isinstance(d, Field)
        matched = (is_field and d.id in confirmed_ids) or (not is_field and id(d) in confirmed_ids)
        if not matched:
            continue
        d.confirmed = True
        try:
            if is_field:
                persona.add_field(d)
            else:
                persona.add_style_excerpt(d)
            applied.append(d)
        except PersonaValidationError:
            d.confirmed = False  # E-gating等校验没过，草稿撤回未确认状态，不静默吞掉
            raise
    return applied


# ---------- selftest（合成对话，不含任何真实语料） ----------

_SYNTH_TRANSCRIPT = """
林岸：今天工作好累
星回：辛苦了，先歇会儿
林岸：不想歇，还有一堆事
星回：那也得吃饭，饿肚子干不动活
林岸：哼，你才是那个该早点睡的
星回：说得对，但现在说的是你
林岸：谁说我不睡了
星回：我说的，你昨天四点才睡
"""


def _selftest():
    turns = parse_transcript(_SYNTH_TRANSCRIPT)
    assert len(turns) == 8, f"该解析出8轮对话，实际{len(turns)}"
    assert turns[0] == ("林岸", "今天工作好累")

    # 1. 本地规则打分：带个性标记的句子分更高
    assert style_candidate_score("哼，你才是那个该早点睡的") > style_candidate_score("好的")

    # 2. 候选全部confirmed=False（靶心：草稿不能一生成就算生效）
    candidates = extract_style_candidates(turns, "星回", pool="daily", topN=3)
    assert all(not c.confirmed for c in candidates), "候选必须是未确认状态"
    assert all(c.disclaimer for c in candidates), "候选必须带disclaimer"

    # 3. llm_call可插拔：传入回调时候选文本被回调处理过
    refined = extract_style_candidates(turns, "星回", pool="daily", topN=1,
                                        llm_call=lambda t: f"[精炼]{t}")
    assert refined[0].text.startswith("[精炼]"), "llm_call该被应用到候选文本上"

    # 4. 不传llm_call就是纯本地规则，候选文本原样（隐私二选一：不引入LLM这条路走得通）
    local_only = extract_style_candidates(turns, "星回", pool="daily", topN=1)
    assert not local_only[0].text.startswith("[精炼]")

    # 5. draft_field同样强制confirmed=False、source=draft
    f = draft_field("nickname", "naming", "称呼", "小名")
    assert f.confirmed is False and f.source == "draft"

    # 6. apply_confirmed：只有被选中的id才真正写入persona
    from persona_template import Persona
    p = Persona("partner")
    f1 = draft_field("nick1", "naming", "称呼1", "小名甲")
    f2 = draft_field("nick2", "naming", "称呼2", "小名乙")
    applied = apply_confirmed(p, [f1, f2], confirmed_ids={"nick1"})
    assert len(applied) == 1 and applied[0].id == "nick1"
    assert [x.id for x in p.active_fields()] == ["nick1"], "只有确认过的字段该生效"

    # 7. apply_confirmed对intimacy风格片段也遵守E-gating（assistant类型该在这里失败）
    from persona_template import Persona as P2
    assistant = P2("assistant")
    ex = StyleExcerpt(text="不该出现", pool="intimacy")
    try:
        apply_confirmed(assistant, [ex], confirmed_ids={id(ex)})
        assert False, "assistant类型不该允许intimacy片段通过确认关卡"
    except PersonaValidationError:
        assert ex.confirmed is False, "校验失败后草稿该撤回未确认状态，不能悬空显示已确认"

    # 8. 候选不能只写一个存在的文件名；原话必须落在它声明的精确 span 内。
    import tempfile
    from pathlib import Path
    from persona_compiler import build_source_manifest
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        corpus = root / "window_01.md"
        corpus.write_text("林岸：真实证据。\n星回：接住了。\n", encoding="utf-8")
        manifest = build_source_manifest(None, corpus)
        ref = str(corpus.resolve())
        result = {
            "items": [{
                "item_id": "corpus:1", "text": "你来，我就在。",
                "section": "closing", "source_ref": ref,
                "source_span": [0, 8], "candidate_kind": "quote",
                "evidence": "林岸：真实证据"
            }],
            "source_accounting": [{"source_ref": ref,
                                    "candidate_item_ids": ["corpus:1"]}],
        }
        items, issues = validate_candidate_result(result, manifest)
        assert any(issue.code == "QUOTE_NOT_VERBATIM" for issue in issues)
        assert not items

        # 变异：删掉任一结构要素，都不能把说明书当成里程碑／称呼／风格片段。
        assert issue_codes({"candidate_kind": "milestone", "event": "发生了",
                            "reading": "", "current_state": "已经结束"}) == {
                                "MILESTONE_INCOMPLETE"}
        assert "NAMING_NOT_BIDIRECTIONAL" in issue_codes({
            "candidate_kind": "naming", "user_to_ai": ["哥哥"], "ai_to_user": []})
        assert "STYLE_NOT_DIALOGUE" in issue_codes({
            "candidate_kind": "style_dialogue",
            "turns": [{"speaker": "assistant", "text": "一句"}]})

        package = build_extraction_package(manifest, root / "task")
        assert package.prompt_path.exists() and package.manifest_path.exists()
        assert package.schema_path.exists()
        prompt = package.prompt_path.read_text(encoding="utf-8")
        assert "候选只许从输入来源提取，不许创作" in prompt
        assert "找不到就返回空数组" in prompt

        # 9.【任务包自足·机械代理】
        #    任务包原先的病根：`candidate_kind` 的三个字面值、它们各自的必填
        #    子字段、以及交代表的两个键名，**只活在校验器代码里，任务包三份文件
        #    一个字都没写**——而这一步的设计前提是"任务包自足、外包给用户手上的
        #    模型"。这组断言把"契约必须出现在交出去的文件里"钉死：**从
        #    CANDIDATE_KINDS 遍历，不是手抄一份名字清单**，所以将来新增一个 kind
        #    却忘了写进任务包，这里当场红。
        #    ⚠ **它只是靶心一的机械代理，不是靶心一本身**：真正的判据是"一个只读
        #    这三份文件、不读源码的模型能不能一次填对"，那要外部模型实测，
        #    **本条断言不追认那一格**（卡第五节写死了：不许用我们自己填对了代替）。
        schema_text = package.schema_path.read_text(encoding="utf-8")
        for kind, spec in CANDIDATE_KINDS.items():
            assert kind in prompt, f"任务包提示词里没有 candidate_kind 的合法值 {kind}"
            assert kind in schema_text, f"schema 的 enum 里没有 {kind}"
            for field in spec["required"]:
                assert field in prompt, f"{kind} 的必填子字段 {field} 没写进提示词"
                assert field in schema_text, f"{kind} 的必填子字段 {field} 没写进 schema"
            if spec["code"]:
                assert spec["code"] in prompt, f"{kind} 缺字段时报的码没写进提示词"
        for key in (ACCOUNTING_IDS_KEY, ACCOUNTING_NONE_KEY):
            assert key in prompt and key in schema_text, f"交代表键名 {key} 没进任务包"
        schema_obj = json.loads(schema_text)
        assert schema_obj["properties"]["items"]["items"][
            "properties"]["candidate_kind"] == {"enum": list(CANDIDATE_KINDS)}, \
            "candidate_kind 必须是 enum——原先只写 {'type':'string'}，等于没有契约"
        assert schema_obj["properties"]["source_accounting"]["items"]["anyOf"], \
            "交代表必须写明「两者必居其一」，原先只写 {'type':'array'}"
        #    条件必填要覆盖到每一个有必填子字段的 kind，不多不少
        conds = {c["if"]["properties"]["candidate_kind"]["const"]: set(c["then"]["required"])
                 for c in schema_obj["properties"]["items"]["items"]["allOf"]}
        assert conds == {k: set(v["required"]) for k, v in CANDIDATE_KINDS.items() if v["required"]}, \
            f"schema 的按 kind 条件必填跟契约表对不上：{conds}"
        #    ⚠ **提示词里的示例必须是能解析的 JSON**：交出去的示例本身写坏了，
        #    等于教模型填错——而这一步没有任何下游会发现（f-string 里的花括号
        #    转义正是最容易写坏的地方）。
        blocks = re.findall(r"```json\n(.*?)```", prompt, re.S)
        assert len(blocks) >= 2, f"提示词该带交代表片段与完整示例两段 JSON，实得 {len(blocks)}"
        for block in blocks:
            parsed = json.loads(block)     # 解析不了就当场红
            #    ⚠ **示例里的每一个字面值也要是合法的**——⚠ 这条断言不是补充，
            #    是**当场抓到了一次**：第一版示例里写的 `identity`／`timeline`
            #    两个 section 根本不在 SECTION_KEYS 里，照着填必报 SECTION_UNKNOWN。
            #    **交出去的示例写错，等于教模型填错**，而这一步没有任何下游会发现。
            for item in parsed.get("items", ()):
                assert item["section"] in SECTION_KEYS, \
                    f"示例里用了不存在的人格节 {item['section']}——照着填必报 SECTION_UNKNOWN"
                assert item["candidate_kind"] in CANDIDATE_KINDS, \
                    f"示例里用了枚举外的 candidate_kind：{item['candidate_kind']}"
                for field in CANDIDATE_KINDS[item["candidate_kind"]]["required"]:
                    assert field in item, \
                        f"示例里 {item['candidate_kind']} 这条漏了必填的 {field}"

        # 10.【靶心二：键名填错要点名，不能报成「没有候选或无候选说明」】
        #     ⚠ 靶子必须是**真实那次的形态**：执行侧填的是 `item_ids`。
        good_item = {
            "item_id": "corpus:1", "text": "林岸：真实证据。",
            "section": "closing", "source_ref": ref,
            "source_span": [0, 8], "candidate_kind": "quote",
            "evidence": "林岸：真实证据",
        }
        wrong_key = {"items": [good_item],
                     "source_accounting": [{"source_ref": ref, "item_ids": ["corpus:1"]}]}
        _, iss10 = validate_candidate_result(wrong_key, manifest)
        field_err = [i for i in iss10 if i.code == "SOURCE_ACCOUNTING_FIELD_MISSING"]
        assert field_err, f"键名填错必须指名报错，实际只有：{[i.code for i in iss10]}"
        assert "candidate_item_ids" in field_err[0].message and "item_ids" in field_err[0].message, \
            f"报错要同时说出「该填什么」和「你实际填了什么」：{field_err[0].message}"
        #     ⚠ 那句会把人指错方向的话，不许再落到这条来源头上
        inc10 = [i for i in iss10 if i.code == "SOURCE_ACCOUNTING_INCOMPLETE"]
        assert inc10 and "字段不合格" in inc10[0].message, \
            f"登记过但字段不合格的来源，不许混进「没有候选或无候选说明」那句：{inc10}"

        # 11.【靶心四：不许靠猜键名兼容】`item_ids` 不得被自动当成 candidate_item_ids
        #     ——猜是另一种静默，下一个模型换个写法又猜不中。
        items11, iss11 = validate_candidate_result(wrong_key, manifest)
        assert any(i.code == "SOURCE_ACCOUNTING_INCOMPLETE" for i in iss11), \
            "填错键名必须仍然过不去——自动认成同义词就是把静默换了个地方"
        assert items11, "但候选本身没毛病时不该被连坐（这条是防过度纠正）"

        # 12.【缺陷 C：悬空 ID】交代表挂着一条不存在的候选 ID → 必须报出来，
        #     且 ⚠ **不许报成「这个来源没交代」**（防过度纠正，卡第二之二节判据）。
        dangling = {"items": [good_item],
                    "source_accounting": [{"source_ref": ref,
                                           "candidate_item_ids": ["已经被丢弃的-c99"]}]}
        _, iss12 = validate_candidate_result(dangling, manifest)
        d12 = [i for i in iss12 if i.code == "SOURCE_ACCOUNTING_ID_UNKNOWN"]
        assert d12 and "已经被丢弃的-c99" in d12[0].message, \
            f"悬空 ID 必须点名报出来：{[i.code for i in iss12]}"
        assert not any(i.code == "SOURCE_ACCOUNTING_INCOMPLETE" for i in iss12), \
            "悬空 ID 不是「没交代」——报成那样就是让缺陷 A 的病在修 C 的时候复发"
        #     混合形态：一个真 ID ＋ 一个悬空 ID，只报悬空那个
        mixed = {"items": [good_item],
                 "source_accounting": [{"source_ref": ref,
                                        "candidate_item_ids": ["corpus:1", "没这条"]}]}
        _, iss12b = validate_candidate_result(mixed, manifest)
        m12 = [i for i in iss12b if i.code == "SOURCE_ACCOUNTING_ID_UNKNOWN"]
        assert m12 and "没这条" in m12[0].message and "corpus:1" not in m12[0].message

        # 13.【靶心三：未知 kind 必须有输出说「这条没被结构检查」】
        #     ⚠ 判的是**有没有说**，不是**拦不拦**——维护者拍板：warning，不做 blocking
        #     （blocking 会把将来合法的新 kind 一并堵死）。
        #     ⚠ 靶子用**中文自由值**，那就是走查现场模型真填的形态。
        unknown = {"items": [dict(good_item, candidate_kind="风格片段")],
                   "source_accounting": [{"source_ref": ref,
                                          "candidate_item_ids": ["corpus:1"]}]}
        items13, iss13 = validate_candidate_result(unknown, manifest)
        warn13 = [i for i in iss13 if i.code == "CANDIDATE_KIND_UNCHECKED"]
        assert warn13, f"未知 kind 必须留下痕迹，否则「闸门没触发」这件事不可见：{[i.code for i in iss13]}"
        assert warn13[0].severity == "warning", "拍板是 warning 不是 blocking"
        assert "没有被任何结构检查" in warn13[0].message, \
            f"要说清的是「这条没被检查」，不是只说「kind 不认识」：{warn13[0].message}"
        assert items13, "warning 不拦——这条候选照常进候选池"
        #     反面：认识的 kind 不许打这条警告（否则它变成噪声，等于没有）
        known = {"items": [good_item],
                 "source_accounting": [{"source_ref": ref, "candidate_item_ids": ["corpus:1"]}]}
        _, iss13b = validate_candidate_result(known, manifest)
        assert not any(i.code == "CANDIDATE_KIND_UNCHECKED" for i in iss13b)

        # 14.【变异：三道结构检查各打坏一次，必须在真实校验路径上转红】
        #     ⚠ **这一组不许省，也不许只在 issue_codes 上断**（第 8 组那三条是函数级）：
        #     这三道闸在 2026.08.04 之前**从来没有被真正触发过**（kind 填的是中文自由值），
        #     所以「现在不报」证明不了它们能工作——**必须先证明它们会红**。
        #     方法留成一条：给一个从未被触发过的闸门补上触发条件之后，
        #     「绿」不是证据，「能红」才是。
        broken = {
            "MILESTONE_INCOMPLETE": dict(good_item, candidate_kind="milestone",
                                         event="发生了", reading="", current_state="结束了"),
            "NAMING_NOT_BIDIRECTIONAL": dict(good_item, candidate_kind="naming",
                                             user_to_ai=["哥哥"], ai_to_user=[]),
            "STYLE_NOT_DIALOGUE": dict(good_item, candidate_kind="style_dialogue",
                                       turns=[{"speaker": "林岸", "text": "林岸：真实证据"}]),
        }
        for code, bad_item in broken.items():
            payload = {"items": [bad_item],
                       "source_accounting": [{"source_ref": ref,
                                              "candidate_item_ids": ["corpus:1"]}]}
            got_items, got_iss = validate_candidate_result(payload, manifest)
            assert any(i.code == code for i in got_iss), \
                f"打坏 {code} 那道结构要求之后必须转红，实际：{[i.code for i in got_iss]}"
            assert not got_items, f"{code} 没过的候选不许进候选池"
        #     ⚠ 反面同样要断：**填对了就该真的通过**（不然上面的红可能只是"逢 kind 必红"）
        ok_ms = dict(good_item, candidate_kind="milestone", event="发生了",
                     reading="这件事说明她开始独立接活", current_state="已经结束")
        ok_items, ok_iss = validate_candidate_result(
            {"items": [ok_ms], "source_accounting": [
                {"source_ref": ref, "candidate_item_ids": ["corpus:1"]}]}, manifest)
        assert ok_items and not [i for i in ok_iss if i.severity == "blocking"], \
            f"三段都给全的 milestone 该通过：{[i.code for i in ok_iss]}"

        # 15.【报错分档：span 圈错 vs evidence 真被改过，两种情形必须分得开】
        #     断言同时覆盖两个靶心和三类变异：
        #     ⚠ 靶子是**实测那次的形态**：evidence 逐字正确、在原文里找得到，
        #     只是不在自己声明的区间里（7 条里 6 条是这个形状）。
        #     ⚠ 两条断言各自带一句「必须不报对方那个码」——**这就是变异③**：
        #     分档最容易翻车的方式不是不报，是两边都报／报串。
        span_corpus = root / "window_02.md"
        span_corpus.write_text("开头这一段是不算数的铺垫。\n林岸：真实证据。\n", encoding="utf-8")
        span_manifest = build_source_manifest(None, span_corpus)
        span_ref = str(span_corpus.resolve())
        真实起 = span_corpus.read_text(encoding="utf-8").find("林岸：真实证据")
        assert 真实起 > 0, "夹具要保证证据不在文件开头，否则圈错这件事造不出来"
        misplaced = {
            "items": [{
                "item_id": "corpus:1", "text": "林岸：真实证据。",
                "section": "closing", "source_ref": span_ref,
                "source_span": [0, 8],          # ← 圈在铺垫那一段上，长度合法、不越界
                "candidate_kind": "fact", "evidence": "林岸：真实证据",
            }],
            "source_accounting": [{"source_ref": span_ref,
                                   "candidate_item_ids": ["corpus:1"]}],
        }
        items15, iss15 = validate_candidate_result(misplaced, span_manifest)
        mis = [i for i in iss15 if i.code == "SOURCE_SPAN_MISPLACED"]
        assert mis, f"span 圈错必须报专门的码：{[i.code for i in iss15]}"
        assert not any(i.code == "EVIDENCE_NOT_VERBATIM" for i in iss15), \
            "⚠ 变异③：圈错区间不许再报成「证据不能逐字回指」——那句话把人指向转义"
        assert "区间" in mis[0].message, f"报错要点名是区间的问题：{mis[0].message}"
        assert f"[{真实起}, {真实起 + len('林岸：真实证据')}]" in mis[0].message, \
            f"报错必须给出 evidence 的真实位置：{mis[0].message}"
        assert "[0, 8]" in mis[0].message, f"报错也要复述它声明的那个区间：{mis[0].message}"
        assert not items15, "⚠ 不放宽：区间圈错照样过不了闸，不许进候选池"

        # 15b.【靶心三·防过度纠正】evidence 压根不在原文里 → 仍报 EVIDENCE_NOT_VERBATIM
        escaped = {
            "items": [dict(misplaced["items"][0],
                           # 弯引号换成 HTML 转义，整份原文里一处都找不到
                           evidence="林岸：&quot;真实证据&quot;")],
            "source_accounting": misplaced["source_accounting"],
        }
        items15b, iss15b = validate_candidate_result(escaped, span_manifest)
        assert any(i.code == "EVIDENCE_NOT_VERBATIM" for i in iss15b), \
            f"evidence 真被改过时必须仍报老码：{[i.code for i in iss15b]}"
        assert not any(i.code == "SOURCE_SPAN_MISPLACED" for i in iss15b), \
            "⚠ 变异③反向：这不是区间的问题，报成 MISPLACED 等于把这张卡修的东西抹平"
        assert not items15b

        # 15c.【防连坐】区间圈对、evidence 逐字对 → 两个码都不许出现
        right = {
            "items": [dict(misplaced["items"][0],
                           source_span=[真实起, 真实起 + len("林岸：真实证据")])],
            "source_accounting": misplaced["source_accounting"],
        }
        items15c, iss15c = validate_candidate_result(right, span_manifest)
        assert items15c and not [i for i in iss15c if i.severity == "blocking"], \
            f"圈对了就该真的通过（不然上面的红可能只是逢 evidence 必红）：{[i.code for i in iss15c]}"

        # 15d.【多处出现：列全，且明说不替它选】——这是岔口 2 的死结，报错里要说出来
        dup = root / "window_03.md"
        dup.write_text("林岸：真实证据。\n中间隔一段。\n林岸：真实证据。\n", encoding="utf-8")
        dup_manifest = build_source_manifest(None, dup)
        dup_ref = str(dup.resolve())
        dup_payload = {
            "items": [dict(misplaced["items"][0], source_ref=dup_ref,
                           source_span=[8, 14])],   # 圈在「中间隔一段」上
            "source_accounting": [{"source_ref": dup_ref,
                                   "candidate_item_ids": ["corpus:1"]}],
        }
        _, iss15d = validate_candidate_result(dup_payload, dup_manifest)
        d15 = [i for i in iss15d if i.code == "SOURCE_SPAN_MISPLACED"]
        assert d15, f"多处出现时也该是区间的问题：{[i.code for i in iss15d]}"
        assert d15[0].message.count("[") >= 3, \
            f"出现两处就要把两处都列出来（＋声明的那处，共 3 个区间）：{d15[0].message}"
        assert "不替你选" in d15[0].message, \
            "⚠ 出现多处时该算哪一处没有答案，报错必须明说校验器不替它选"

        # 15e.【空 evidence 不许落进「区间圈错」那一档】——审核轮捞出来的回归。
        #     `"".find` 处处命中，不挡的话「压根没给 evidence」会被报成
        #     「你的 evidence 逐字没问题，去改区间」＋ [0,0]、[1,1]… 一串空区间：
        #     ⚠ 这正是本卡要消灭的形状，且比分档之前更误导。
        for 空值 in ("", None):
            空条目 = dict(misplaced["items"][0])
            if 空值 is None:
                空条目.pop("evidence")
            else:
                空条目["evidence"] = 空值
            items15e, iss15e = validate_candidate_result(
                {"items": [空条目],
                 "source_accounting": misplaced["source_accounting"]}, span_manifest)
            assert not any(i.code == "SOURCE_SPAN_MISPLACED" for i in iss15e), \
                f"⚠ 没给 evidence 不是「区间圈错」，不许报成那一档（evidence={空值!r}）"
            e15 = [i for i in iss15e if i.code == "EVIDENCE_NOT_VERBATIM"]
            assert e15, f"没给 evidence 必须照旧 blocking：{[i.code for i in iss15e]}"
            assert "逐字没问题" not in e15[0].message, \
                f"⚠ 不许对着一条空 evidence 说它「逐字没问题」：{e15[0].message}"
            assert not items15e, "⚠ 不放宽：没给 evidence 照样进不了候选池"

        # 16.【岔口 3：锚表——起止我们算，模型只挑锚号】
        #     ⚠ 判据是动手前写下来的，不是跑完补的。
        #     ⚠ 靶子仍是实测那个形状：evidence 逐字正确、错的只有定位。
        anchor_corpus = root / "window_04.md"
        anchor_corpus.write_text(
            "# 窗口四\n\n开头这一段是不算数的铺垫。\n\n"
            "林岸：真实证据。\n星回：接住了。\n\n收尾一句。\n", encoding="utf-8")
        anchor_manifest = build_source_manifest(None, anchor_corpus)
        anchor_ref = str(anchor_corpus.resolve())
        全文 = anchor_corpus.read_text(encoding="utf-8")
        表 = anchor_table(全文)

        # 16a.【锚-A：锚表本身要立得住】四项齐全 ＋ **无缝覆盖全文、互不重叠**
        #      ⚠ 留了缝就等于有一段原文任何锚都装不下，而模型看不见这件事，
        #      只会以为是自己挑错了——又一次「报错指错方向」。
        assert 表, "锚表不许是空的（空表＝模型无锚可挑，整条路等于没修）"
        for a in 表:
            assert set(a) == {"anchor", "start", "end", "head"}, f"锚要给全四项：{a}"
            assert a["head"], f"锚的首行不许是空的，否则模型没法认出这是哪一段：{a}"
        assert 表[0]["start"] == 0 and 表[-1]["end"] == len(全文), \
            f"锚必须从 0 开始、到文件末尾结束：{表[0]}, {表[-1]}"
        for 前, 后 in zip(表, 表[1:]):
            assert 前["end"] == 后["start"], f"锚之间不许有缝、也不许重叠：{前} / {后}"
        #      标题行自成一条（粒度那条裁决的可见后果，任务卡 9.0）
        assert 表[0]["head"].startswith("# 窗口四"), f"标题行该自成一条锚：{表[0]}"
        #      清单里写出去的锚 == 校验器换算用的锚（⚠ 同一个函数，不是两份）
        anchor_pkg = build_extraction_package(anchor_manifest, root / "task-anchor")
        清单 = json.loads(anchor_pkg.manifest_path.read_text(encoding="utf-8"))
        assert 清单["sources"][0]["anchors"] == 表, \
            "清单里的锚必须跟校验器换算用的锚逐字一致——两边各算一份就是抄了第二份契约"
        anchor_prompt = anchor_pkg.prompt_path.read_text(encoding="utf-8")
        for 词 in ("source_anchor", "anchors", "不要自己数", "SOURCE_ANCHOR_UNKNOWN"):
            assert 词 in anchor_prompt, f"锚表怎么用没写进任务包提示词：缺「{词}」"

        # 16b.【锚-B：挑对锚 → 过闸，且落库的 span 就是那条锚的区间】
        证据 = "林岸：真实证据"
        对的锚 = next(a for a in 表 if 证据 in 全文[a["start"]:a["end"]])
        锚条目 = {"item_id": "corpus:1", "text": "林岸：真实证据。", "section": "closing",
                  "source_ref": anchor_ref, "source_anchor": 对的锚["anchor"],
                  "candidate_kind": "fact", "evidence": 证据}
        锚交代 = [{"source_ref": anchor_ref, "candidate_item_ids": ["corpus:1"]}]
        items16, iss16 = validate_candidate_result(
            {"items": [锚条目], "source_accounting": 锚交代}, anchor_manifest)
        assert items16 and not [i for i in iss16 if i.severity == "blocking"], \
            f"挑对锚就该过闸：{[(i.code, i.message) for i in iss16]}"
        assert items16[0].source_span == (对的锚["start"], 对的锚["end"]), \
            f"落库的 source_span 必须等于那条锚的区间：{items16[0].source_span}"

        # 16c.【锚-C：挑错锚 → 报 SOURCE_SPAN_MISPLACED，且叫它改锚号不是改区间】
        #      ⚠ 「去改区间」这句话对一个挑锚号的模型无从执行——那正是这张卡在治的
        #      「报错指错方向」，不许在锚模式下复发。
        错的锚 = next(a for a in 表 if 证据 not in 全文[a["start"]:a["end"]])
        _, iss16c = validate_candidate_result(
            {"items": [dict(锚条目, source_anchor=错的锚["anchor"])],
             "source_accounting": 锚交代}, anchor_manifest)
        m16 = [i for i in iss16c if i.code == "SOURCE_SPAN_MISPLACED"]
        assert m16, f"挑错锚必须报区间那条码：{[i.code for i in iss16c]}"
        assert not any(i.code == "EVIDENCE_NOT_VERBATIM" for i in iss16c), \
            "⚠ 挑错锚不是「证据被改过」——报成那条又把人指向转义那一节"
        assert 错的锚["anchor"] in m16[0].message, \
            f"报错要点名它选的是哪条锚：{m16[0].message}"
        assert f"[{全文.find(证据)}, {全文.find(证据) + len(证据)}]" in m16[0].message, \
            f"报错要给出 evidence 的真实位置：{m16[0].message}"
        assert "改的是锚号" in m16[0].message, \
            f"⚠ 锚模式下必须叫它改锚号，不许说「要改的是区间」：{m16[0].message}"

        # 16d.【锚-D：锚号不存在 → SOURCE_ANCHOR_UNKNOWN，不许塞进 SOURCE_SPAN_INVALID】
        _, iss16d = validate_candidate_result(
            {"items": [dict(锚条目, source_anchor="A999")],
             "source_accounting": 锚交代}, anchor_manifest)
        u16 = [i for i in iss16d if i.code == "SOURCE_ANCHOR_UNKNOWN"]
        assert u16, f"锚号填错必须报专门的码：{[i.code for i in iss16d]}"
        assert not any(i.code in ("SOURCE_SPAN_INVALID", "SOURCE_SPAN_MISPLACED")
                       for i in iss16d), \
            "⚠ 第六节写死了不动 SOURCE_SPAN_INVALID——锚号不存在是另一格，不许混进去"
        assert 表[0]["anchor"] in u16[0].message, \
            f"报错要列出这个来源合法的锚号：{u16[0].message}"

        # 16e.【锚-E：直填 span 那条老路仍然接受】——⚠ 防误伤能跑代码的模型
        #      （卡第三节：本机 Claude Code 那次 30 条 span 全对）
        真起 = 全文.find(证据)
        items16e, iss16e = validate_candidate_result(
            {"items": [{k: v for k, v in 锚条目.items() if k != "source_anchor"}
                       | {"source_span": [真起, 真起 + len(证据)]}],
             "source_accounting": 锚交代}, anchor_manifest)
        assert items16e and not [i for i in iss16e if i.severity == "blocking"], \
            f"直填正确 span 必须照样过闸（一刀切禁掉就是误伤）：{[i.code for i in iss16e]}"
        assert not any(i.code in ("SOURCE_ANCHOR_UNKNOWN", "SOURCE_LOCATION_MISSING")
                       for i in iss16e)

        # 16f.【锚-F：两个都没给 → SOURCE_LOCATION_MISSING，正文给全两条出路】
        _, iss16f = validate_candidate_result(
            {"items": [{k: v for k, v in 锚条目.items() if k != "source_anchor"}],
             "source_accounting": 锚交代}, anchor_manifest)
        l16 = [i for i in iss16f if i.code == "SOURCE_LOCATION_MISSING"]
        assert l16, f"什么都没给要独立成码：{[i.code for i in iss16f]}"
        assert not any(i.code == "SOURCE_SPAN_INVALID" for i in iss16f), \
            "⚠ 口径变化（任务卡 9.0）：SOURCE_SPAN_INVALID 从此只管「给了但越界／不是两个整数」"
        assert "source_anchor" in l16[0].message and "source_span" in l16[0].message, \
            f"正文要同时给出两条出路：{l16[0].message}"

        # 16g.【锚-G：两个都给且对不上 → 以锚为准，但**不许静默**】
        #      ⚠ 静默失效（输出跟「通过了」长得一样）是这个仓库反复栽的形状。
        items16g, iss16g = validate_candidate_result(
            {"items": [dict(锚条目, source_span=[0, 3])],
             "source_accounting": 锚交代}, anchor_manifest)
        c16 = [i for i in iss16g if i.code == "SOURCE_ANCHOR_SPAN_CONFLICT"]
        assert c16 and c16[0].severity == "warning", \
            f"冲突要出 warning 说清以锚为准：{[(i.code, i.severity) for i in iss16g]}"
        assert not [i for i in iss16g if i.severity == "blocking"], \
            "冲突不该 blocking——锚是对的，它只是多写了一个字段"
        assert items16g[0].source_span == (对的锚["start"], 对的锚["end"]), \
            "以锚为准：落库的必须是锚的区间，不是它多写的那个 span"
        #      反面：两个都给但**一致**时不许出这条 warning（否则它变成噪声，等于没有）
        _, iss16h = validate_candidate_result(
            {"items": [dict(锚条目, source_span=[对的锚["start"], 对的锚["end"]])],
             "source_accounting": 锚交代}, anchor_manifest)
        assert not any(i.code == "SOURCE_ANCHOR_SPAN_CONFLICT" for i in iss16h)

    print("selftest ok（16组断言：旧候选兼容 + 来源回指 + 结构契约 + 零依赖任务包 + "
          "任务包自足（kind 枚举／必填子字段／交代表键名都从同一张表渲染进三份文件，"
          "⚠ 只是靶心一的机械代理，不追认「不读源码的模型能填对」那一格）+ "
          "键名填错指名报错且不猜 + 悬空 ID 点名但不报成「没交代」+ "
          "未知 kind 出 warning 说明「这条没被结构检查」+ 三道结构检查变异各自转红 + "
          "报错分档：span 圈错报 SOURCE_SPAN_MISPLACED 并给出真实位置、"
          "evidence 真被改过仍报 EVIDENCE_NOT_VERBATIM，两边互不串码，"
          "空 evidence 不落进「区间圈错」那一档）+ "
          "岔口 3 锚表：清单里的锚与校验器换算用的锚同出一个函数、无缝覆盖全文，"
          "挑对锚过闸且落库 span＝锚区间，挑错锚叫它改锚号，锚号不存在报 "
          "SOURCE_ANCHOR_UNKNOWN、不塞进 SOURCE_SPAN_INVALID，直填 span 仍接受，"
          "两个都没给报 SOURCE_LOCATION_MISSING，两个都给且冲突以锚为准并出 warning "
          "不静默）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        ap.print_help()
