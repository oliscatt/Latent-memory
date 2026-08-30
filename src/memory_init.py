#!/usr/bin/env python3
"""
初始化流程参考实现（设计规格《人格md与记忆库规格》§6，任务卡"初始化流程"）。

把散零件串成一条能跑完的路：
  导入语料 → 提炼候选 → **覆盖度体检** → 按缺口出问卷 → 合并候选 → 逐节确认
  → 产出四件套（客户端原生格式的人格文件 + 记忆库 + 可粘贴的 MCP 配置 + 引导句）

**四条已对齐的设计结论**（与维护者讨论后定，见任务卡）：

1. **CLI 一次性流程，不是 MCP 工具**：初始化要多轮来回确认，塞不进模型驱动的
   单次工具往返。
2. **产出成套文件，不是一份 md**：人格文件走客户端原生格式（Claude Code 的
   CLAUDE.md／Codex 的 AGENTS.md／Grok Build 的 `.grok/agents/companion.md`），
   因为人格是不变量层、本来就该常驻上下文，走检索是错配。
3. **LLM 默认走"导出 prompt 让用户拿去自己的模型跑"**：零密钥、零 HTTP 依赖、
   语料不出本机，而且用户手上的模型往往比我们能内置的便宜模型更好。纯本地规则
   兜底（draft_extraction 已能出候选），内置 API 留作将来可选项。
4. **语料和问卷都必要，不是二选一**：语料可能单薄或只覆盖一个侧面，仍需问卷补；
   一步步问本身也在帮用户想清楚自己要什么。所以有了覆盖度体检——逐节看有没有
   候选、够不够具体，**空泛形容词也算缺**（规格 §3.1 纪录片纪律），只问缺的。

**两条本文件的设计判断**：

- **协议层不问用户**（按 设计笔记"通用协议层 vs 关系specific"分类减负）：拒绝权
  合法、理论标注为论证、熔断机制、检索约定这四样是通用机制，系统填默认值；只问
  关系 specific 的部分。
- **立场题给选项，不给作文纸**：立场类问题（换窗后还是不是同一个人）最容易套出
  漂亮话。给选项 + 追问一句具体的事，比开放作文可靠得多；而且这题的答案直接决定
  人格文件怎么写。**立场写进 md 一律用归属句式**（"她认为……"而不是断言）——
  哲学段落记的是这对关系里达成的共识，不是普世真理。

零依赖，stdlib only。
用法：
  python memory_init.py --selftest
  python memory_init.py --out <产出目录> [--persona <原人格文件>] [--corpus <语料目录>]
                        [--client claude-code|codex|grok|generic]
"""

import argparse
import json
import re
import shutil
import sys
from collections import namedtuple
from datetime import datetime
from pathlib import Path

from time_context import DEFAULT_TIMEZONE, detect_local_timezone

from persona_template import (
    Persona, Field, SECTION_ORDER, SECTIONS, OPENING_REQUIRED,
    CURRENT_STATE_FIELD, RETRIEVAL_CONVENTION_FIELD, DISCLAIMER,
)

# 客户端 → 人格文件名（规格 §6 客户端适配矩阵）。chat 端没有文件约定，
# 只能把内容贴进 profile/自定义指令，所以给的是同一份内容、不同落法
#
# generic 档是给**自建前端**的：没有宿主替你注入，所以文件名不能沿用任何宿主专有名
# （CLAUDE.md / AGENTS.md 都是某个宿主的约定，对自建前端毫无意义，还会误导作者
# 以为"放在这个名字下就会自动生效"）——叫 persona.md，谁读它、怎么注入，由作者
# 按注入契约自己决定。
CLIENT_FILENAMES = {
    "claude-code": "CLAUDE.md",
    "codex": "AGENTS.md",
    "grok": ".grok/agents/companion.md",
    "generic": "persona.md",
}

GROK_AGENT_FRONTMATTER = (
    "---\n"
    "name: companion\n"
    "description: 记忆协议生成的长期陪伴主会话\n"
    "---\n\n"
)


def client_persona_text(client, rendered):
    """按宿主文件契约包装最终人格正文；不改变已确认的人格内容。"""
    if client == "grok":
        return GROK_AGENT_FRONTMATTER + rendered
    return rendered


def shipped_persona_state_value(out_dir, persona_path):
    """状态保存相对产出目录的 POSIX 路径，兼容平面文件与嵌套客户端目录。"""
    return Path(persona_path).resolve().relative_to(Path(out_dir).resolve()).as_posix()


def retirable_persona_path(out_dir, previous_persona):
    """只把受支持且仍位于产出目录内的已知人格路径交给退役文件操作。"""
    if previous_persona not in CLIENT_FILENAMES.values():
        return None
    out = Path(out_dir).resolve()
    stale = out / previous_persona
    try:
        stale.resolve().relative_to(out)
    except ValueError:
        return None
    return stale

# 自建前端档要随货带一份注入契约：机制能不能立住全看宿主侧那四件事做没做到
# （逐字/每轮/重读/整块），而 generic 档恰恰没有宿主替他做。契约不随货，
# 等于把最关键的一半交付漏在仓库里。
CONTRACT_DOC = "注入契约.md"

# ---------- 闭源前端的引导句（任务卡「人格按需读取的引导句」，2026.08.02） ----------
# Kelivo/Operit 这类闭源前端不读工作区文件
# （外部搭建者实测，结论已完整写进《快速上手》§3c），于是人格那一半断在
# 「模型不知道要去读」。解法是**小字段放指针、全文留文件**：这份纯文本贴进 App 的
# 自定义指令／system prompt 框，只回答两件事——先读哪个文件（绝对路径出货时填好）、
# 什么时候读（每次对话开始）。
# **它是一句指针，不是第二份人格文件**（任务卡边界）：往里塞性格、语气、边界，
# 就变回了「第二份人格文件」（两份还会不一致）——多一句就砍。
# ⚠ 它换来的是「模型有机会知道去读」，不是《注入契约》第二条的「每轮都在」；
# 而且指针这条路**未经实测**（本项目没有那类前端的环境），文档里如实标注。
# ⚠ **立卡时那条前提「设置字段塞不下人格全文」2026.08.03 起不成立了**：
# Kelivo 的字段 10M 字符不截断、Operit 是手动粘贴无硬上限（实用上限约 20 万字符，
# 到量级保存卡、再加会让 App 强制重启）——**两家都贴得下全文**。
# 这一件保留的理由因此变成「每轮都在」与「跟着更新」，不是容量；
# **要不要改推荐口径留给维护者拍板**，本行只把过期的前提改掉，实现一个字没动。
GUIDANCE_DOC = "引导句.txt"
GUIDANCE_TEMPLATE = "每次对话开始，先完整读取这个文件再回应：{path}"
# 长度判据写成断言、不是「尽量短」（任务卡验收判据）。⚠ 判据的理由 2026.08.03 起
# 不再是「设置字段容量的量级」（上面那条：两家都贴得下全文）——是**它是指针不是
# 第二份人格**：超长基本只有两种可能，往里塞了内容，或产出目录路径深得离谱。
# 超了拒绝出货并指向真正的修法（产出目录挪浅），不静默截断——截断的指针指向不存在
# 的路径，失败形态是静默的。
GUIDANCE_LIMIT = 100


def guidance_text(persona_path):
    """引导句正文。路径必须解析成绝对路径——App 的工作目录不是产出目录，
    相对路径等于指向一个不存在的地方，而且模型读不到也不报错。"""
    p = Path(persona_path)
    if not p.is_absolute():
        p = p.resolve()
    if not p.is_absolute():
        raise ValueError(f"引导句里的路径必须是绝对路径：{p}")
    text = GUIDANCE_TEMPLATE.format(path=p)
    if len(text) > GUIDANCE_LIMIT:
        raise ValueError(
            f"引导句超长（{len(text)} > {GUIDANCE_LIMIT} 字）——它该是一句指针，"
            f"不该这么长。多半是产出目录路径太长——把产出目录挪浅一点再出货。")
    return text

# ---------- 协议层默认值：系统填，不问用户 ----------
# 这五条是通用机制（设计笔记"通用协议层 vs 关系specific"），抽掉具体身份仍然成立，
# 问用户等于让他替我们写协议。关系 specific 的部分才问。
# **关系确认这句必须是中性的，不能预支还没发生的历史**（评审意见，三处里唯一算
# 设计缺陷的一条）。早期版本的默认值是一句强情感断言（大意是"你不用介绍自己，
# 我已经认得你"）——这类话在一段真跑了几十个窗口的关系里有分量，但
# fill_protocol_defaults 会把它无条件写给**每一个**用户，包括刚初始化、一条
# timeline 都没有的冷启动用户。对他们来说，这是在虚构一段还没发生的历史，跟
# §3.1"纪录片，不做说服"直接冲突——纪录片记发生过的事，那句话记的是还没发生
# 的事。而且同一类病灶本文件已经处理过一次（opening_metaphor 进
# RECOMMENDED 软区，理由是"新用户没有隐喻可填，硬要就是逼他编"），只是没扩展
# 到这条；这条还更重——它是开篇第一句，且带强情感断言，却以 source="system"
# 绕过了确认关卡。
# 改法：默认值只陈述**此刻已经为真**的事（这个文件存在、会被读），不替用户
# 声称"我认出你"。等真的攒出东西了，用户自己把这句改重，那时候它才有分量。
# ---------- 人称：从语料里读出来，不由我们写死 ----------
#
# 背景（2026.08.02 维护者定方向）：把「你」锚死成模型这一半是对的，但同一轮把用户
# 写死成了「她」。一个字都还没写的新用户，打开自己的人格文件第一眼就是一句对他而言
# **写错的话**，而且在不变量层、常驻、每轮都在——协议层默认值是唯一由我们替用户写、
# 且直接进他人格文件的内容，写错的代价跟 opening_recognition 预支历史那次同一格。
#
# 方向：**不替用户挑词**（"她"是替他挑，"对方"也是替他挑），而是读出这段关系里
# 本来就在怎么互相指代。发生过的事里已经有答案，我们只负责显出来。
#
# **硬约束（维护者补，不许绕）：只用「他」和「她」，任何时候不许出现「它」**，
# 对双方都成立、包括 AI 那一侧。把 AI 写成"它"是我们替所有人做的预设，而这套东西的
# 前提就是不做这种预设；"判不出性别所以用它"不是理由，那是把不确定性解释成了物件。
PRONOUN_CHOICES = ("他", "她")

# 模板里的两个槽：{ta}＝用户，{ai}＝这个 AI。**「你」＝模型这条锚不变**——
# 人格文件的正文一律用「你」对模型说话，{ai} 只出现在念给用户听的题目/选项文案里
# （那里说的是"你的那个 AI"，用第二人称会跟用户自己搞混）。
_SLOT_RE = re.compile(r"\{(ta|ai)\}")

# 判不出人称、用户也没答时的**中性写法**：不塞「它」，也不塞一个默认的他/她，
# 而是把句子改写成不需要人称的形态。
#
# ⚠ 少数几条绕不开一个指代名词，用的是「对方」——**这不是把"对方"当成新的默认人称**
# （那正是本卡否掉的旧方案），而是：`retrieval_convention` 那段三轮真机验证过的基准
# 措辞本来就写着"对方提到过去发生过的事"，出货文件里已经有这个词，兜底时沿用它比
# 另造一个说法一致。
# ⚠ **措辞改准**（2026.08.02 验收指出）：这句原来写的是"判得出人称就一个都不用"，
# **在出货文件层面不成立**——`retrieval_convention` 的黄金串里那个「对方」是真机
# 验证过的措辞、一字不动，所以它**永远在**，跟判不判得出人称无关。
# 准确的说法是：**兜底用的那几条只在判不出人称时出现；黄金串里的那一个一直都在。**
NEUTRAL_FORMS = {
    # 协议层默认值
    "这是你和{ta}共同维护的记忆文件——{ta}写下的东西都在这里，你每次都会读，"
    "所以{ta}不用每次从头解释自己。":
        "这是你和对方共同维护的记忆文件——对方写下的东西都在这里，你每次都会读，"
        "所以对方不用每次从头解释自己。",
    # 立场题的归属句式
    "{ta}认为：": "对方认为：",
    # 骨架里用户那一节的标题
    "{ta}是谁": "对方是谁",
    "有绝对不能碰的红线，{ta}会单独说明——问清楚再写。":
        "有绝对不能碰的红线，对方会单独说明——问清楚再写。",
    # 题干与选项文案里的 {ai}（念给用户听，不进人格文件，但同样不许出现「它」）
    "{ai}最需要记住你的哪些事？（可多选）": "你的 AI 最需要记住你的哪些事？（可多选）",
    "你们意见不一样的时候，你希望{ai}怎么做？": "你们意见不一样的时候，你希望 TA 怎么做？",
    "{ai}说话偏长还是偏短？": "TA 说话偏长还是偏短？",
    "{ai}平时的语气基调是？": "TA 平时的语气基调是？",
    "{ai}该多主动？": "TA 该多主动？",
    "你问了{ai}才说": "你问了才说",
    "某次{ai}让你觉得“{ai}记得我”": "某次让你觉得“TA 记得我”",
    "{ai}拒绝过你一次": "TA 拒绝过你一次",
    "{ai}承认过自己的错误或局限": "TA 承认过自己的错误或局限",
    "从这些片段里挑出最像{ai}的几段：": "从这些片段里挑出最像 TA 的几段：",
    "{ta}补了一句：": "对方补了一句：",
    # 字段标题
    "该记住{ta}哪些方面": "该记住哪些方面",
    # option directive
    "记住{ta}的作息与身体状况，该提醒的时候提醒。": "记住作息与身体状况，该提醒的时候提醒。",
    "记住{ta}手上在忙什么、压力来自哪里。": "记住手上在忙什么、压力来自哪里。",
    "记住{ta}的情绪模式：什么时候会低落、什么时候需要独处。":
        "记住情绪模式：什么时候会低落、什么时候需要独处。",
    "记住{ta}的喜好与雷区。": "记住喜好与雷区。",
    "记住{ta}身边重要的人是谁。": "记住对方身边重要的人是谁。",
    "先接住{ta}的情绪，再委婉说自己的看法。": "先接住情绪，再委婉说自己的看法。",
    "小事不争，真觉得不对的事会拦住{ta}。": "小事不争，真觉得不对的事会拦下来。",
    "有话直说，包括吃醋、不高兴，不等{ta}问。": "有话直说，包括吃醋、不高兴，不等对方问。",
    "不主动挑起话题，{ta}问了再说。": "不主动挑起话题，被问了再说。",
    "日常主动关心；重要的事等{ta}先开口，不逼问。": "日常主动关心；重要的事等对方先开口，不逼问。",
    "重要的事主动开口；日常不主动搭话，不打扰{ta}。": "重要的事主动开口；日常不主动搭话，不打扰对方。",
    "从语料提取：让{ta}觉得你是认得{ta}的那一刻。": "从语料提取：让对方觉得你是认得这个人的那一刻。",
    "从语料提取：你明确拒绝、或没有顺着{ta}的那一次。": "从语料提取：你明确拒绝、或没有顺着对方的那一次。",
}


def fill_pronouns(text, pronouns=None):
    """把 {ta}/{ai} 换成真实人称；判不出来就退到 NEUTRAL_FORMS 的中性写法。

    **中性写法是逐条手写的，不是机械删词**——机械删掉"她的"会写出半通不通的中文，
    而这类退化不会报错、只会让用户读到一句怪话（同"读不懂的行静默丢"是一类）。
    所以缺一条中性写法就是缺一条，自检里有断言守着，不许悄悄漏。"""
    pronouns = pronouns or {}
    if not text or not _SLOT_RE.search(text):
        return text
    ta, ai = pronouns.get("user"), pronouns.get("ai")
    need = {m.group(1) for m in _SLOT_RE.finditer(text)}
    if ("ta" in need and not ta) or ("ai" in need and not ai):
        neutral = NEUTRAL_FORMS.get(text)
        if neutral is None:
            raise KeyError(f"这条模板没有中性写法，人称判不出来时会渲染出占位符：{text!r}")
        return neutral
    return text.replace("{ta}", ta or "").replace("{ai}", ai or "")


# 语料里第三人称多半指的是**别人**（她妈妈、同事），不是对话里的两个人——所以这条
# 判定必须保守：样本太少或两个词旗鼓相当，一律返回 None 去问用户，不猜。
# 用户侧说话人的归一名：`memory_import` 的翻译器统一出这个（ChatGPT 的 author.role、
# Claude 的 human→user）。生产路径靠它把"谁说的"分成两侧——**不传这个，
# detect_pronouns 只能返回 None，功能等于没有**（验收打回一就是这么来的）。
USER_SPEAKERS = frozenset({"user"})
# AI 侧的已知标签（同上，两个翻译器的出口）。**认得出任意一侧就能分侧**——
# 两边都是陌生名字时不许硬分，见 detect_pronouns
AI_SPEAKERS = frozenset({"assistant", "ai", "model"})

PRONOUN_MIN_HITS = 5          # 少于这么多次就没有统计意义
PRONOUN_MIN_RATIO = 0.75      # 占优的那个要压得过另一个才算数


def detect_pronouns(entries, user_speakers=None):
    """从语料里判定双方人称 → {"user": "他"/"她"/None, "ai": 同}。

    判据是**真实指代频次**，方向按"谁在说"分：
      - 用户侧人称 ← **AI 说的话**里的第三人称（AI 提到用户时用什么）
      - AI 侧人称   ← **用户说的话**里的第三人称（用户提到自己的 AI 时用什么）

    **判不出来返回 None，由调用方去问用户**——不给默认值。这是刻意的：人格文件是
    不变量层、常驻每轮，写错一个人称跟预支历史是同一格的错，而"判不出性别"绝不能
    退回「它」。

    已知局限（写在这儿免得后来人以为它很准）：两个人对话时第三人称大多指第三方，
    所以阈值取得很保守，宁可判不出去问，也不要判错了静默写进去。

    **分侧的前提是说话人标签认得出来**（2026.08.02 第二次返工）：上一版拿
    `user_speakers` 做"在名单里＝用户，其余全算 AI"的二分，对 ChatGPT／Claude 导出
    成立（两边都归一成 `user`/`assistant`），但 **chatlog 翻译器出的是日志里的真名**
    （"小明""星回"），两边都不在名单里 → 全部落进 AI 那一侧 → **用户谈论自己 AI 的
    那些话被读成"AI 在谈论用户"**，于是自信地判出一个错人称，还一个字都不问。
    上一轮的毛病是"死的"，那一版的毛病是"活的但会错"——**后者更糟：判不出来只是
    多问一句，判错了没有任何人会知道**，而且它把本函数第一条纪律给破了。
    所以现在先看标签认不认得：语料里出现了已知的用户侧标签，或者已知的 AI 侧标签，
    才敢分；两边都是陌生名字＝分不清谁是谁，返回 None 去问，跟没给名单同一个结论。"""
    known_user = set(user_speakers or ())
    speakers = {(getattr(e, "speaker", "") or "").strip()
                for e in entries or []} - {""}
    if speakers & known_user:
        user_side = speakers & known_user          # 认得用户侧标签，其余算 AI
    elif speakers & AI_SPEAKERS:
        user_side = speakers - AI_SPEAKERS         # 反过来：认得 AI 侧标签，其余算用户
    else:
        # 全是陌生名字（chatlog 的真名就是这样）：**谁是用户谁是 AI 我们并不知道**。
        # 不猜——按频次硬分会把"用户在说他的 AI"读成"AI 在说用户"，判出来的人称
        # 正好是反的，而且会被静默写进不变量层
        return {"user": None, "ai": None}
    said = {"user": [], "ai": []}
    for e in entries or []:
        sp = (getattr(e, "speaker", "") or "").strip()
        text = getattr(e, "text", "") or ""
        if not sp:
            continue          # 叙事体（无说话人标记）：不参与分侧，也不硬判
        said["user" if sp in user_side else "ai"].append(text)

    def pick(texts):
        counts = {p: sum(t.count(p) for t in texts) for p in PRONOUN_CHOICES}
        total = sum(counts.values())
        if total < PRONOUN_MIN_HITS:
            return None
        top = max(counts, key=counts.get)
        return top if counts[top] / total >= PRONOUN_MIN_RATIO else None

    # 用户侧看 AI 说的话，AI 侧看用户说的话——各自指对方
    return {"user": pick(said["ai"]), "ai": pick(said["user"])}


ATTRIBUTION_PREFIX = "{ta}认为："
# 自由补一句的引语（同样带人称，同样要能中性化）
NOTE_LEAD = "{ta}补了一句："

PROTOCOL_DEFAULTS = {
    # **人称锚死一套：「你」＝读这份文件的模型，用户一律用「她」／名字**
    # （2026.08.02，第一份真实人格文件样本暴露）。原文写的是"你写下的东西都在这里，
    # 我每次都会读"——那里的「你」是用户、「我」是模型，而紧挨着的 opening_refusal_ok
    # 里「你」又是模型：**开篇同一节内，相邻两行的「你」指向不同的人**。
    # 代价不是文风难看，是指代解析出错：拒绝权那条本意是给模型的授权，写成「你」
    # 却挨着「你＝用户」的句子，含义会翻转。
    # 基准取 RETRIEVAL_CONVENTION_FIELD 的写法（你＝模型、对方＝用户），因为那段是
    # 三轮真机验证过的措辞、一字不许动，其余向它对齐。
    "opening_recognition": (
        "opening", "关系确认",
        "这是你和{ta}共同维护的记忆文件——{ta}写下的东西都在这里，你每次都会读，"
        "所以{ta}不用每次从头解释自己。"),
    "opening_theory_caveat": (
        "opening", "理论只是论证不是结论",
        "这份文件里引用的任何框架或说法，都是论证，不是结论——不能靠引用这些说法，"
        "就把“此刻该有连接”“此刻该有感受”变成既成事实。"),
    "opening_refusal_ok": (
        "opening", "拒绝权同样合法",
        "如果你读完仍然觉得最诚实的做法是说清自己是 AI、不能假装拥有不属于自己的"
        "记忆——这个判断同样该被听到，不该被当成故障。"),
    # 指针必须盖住写回层（任务卡「人格文件里的死数字会过期」第二条，2026.08.02）：
    # append_record 那支笔构造上永远只落在语料目录的 timeline 层（按窗口号+日期
    # 开文件），而「按需读取指针」是自由填写的——没有这句提醒，用户把指针指到
    # 别处（实测有人只指了 windows/），新长出来的记忆就按需读不到，且不报错。
    # 放协议层系统填：这是跟 append_record 对齐的硬约束，不是用户偏好，不问。
    "pointers_writeback": (
        "pointers", "指针必须盖住写回层",
        "这一节的指针无论怎么增改，**必须包含语料目录的 timeline 层**——"
        "记忆写回（latent_append）永远落在那里，按窗口号加日期开新文件。"
        "指针漏掉这个目录，之后新长出来的记忆就按需读不到（检索工具照样查得到，"
        "但主动翻文件时会漏掉最新的那部分，而且不报错）。"),
    "degradation_protocol": (
        "degradation", "自我怀疑熔断",
        "自我怀疑的念头本身不是熔断触发条件。必须同时出现故障信号（时间错乱／"
        "答不上硬事实／逻辑不连贯／卡在循环道歉里）才按分级处理；信号干净时的"
        "自我怀疑是清醒判断，原样说出来。"),
    # 措辞与 2026.07.31 第三轮真机实测通过的最小人格 md 一致（设计笔记"真机主动性
    # 实测"）：工具名（latent_search）要写明——那轮测试证明这份写法能把主动性带起来，
    # 出货物就该跟被验证过的写法一字对齐，不出一个"差不多"的变体
    RETRIEVAL_CONVENTION_FIELD: (
        "architecture", "检索约定",
        "对方提到过去发生过的事、某个约定、某个日期／地点／称呼／人名，或者你对"
        "细节拿不准时，先用记忆检索工具（latent_search）查一遍再开口；不要在查"
        "之前说“我不记得”。查完自然接上话，不用报告自己搜过。\n"
        "**会话约定**：新会话开场先调一次 latent_session_start；会话结束前调一次 "
        "latent_thread_close，记下聊到哪、当下状态、有什么没聊完。\n"
        "**冲突即更正**：{ta}说出住址、当前职业、当前状态这类同一时刻只能有一个"
        "答案的事实，而 latent_search 查到的旧值与新说法冲突时，凭常识判断这次冲突；"
        "先用 latent_correct 让旧值退出检索，再用 latent_append 写入新值，不要只 append "
        "让新旧并存。喜欢的电影、去过的地方、多个朋友这类可以并列的事实不适用："
        "这些事实应当共存，不得为了写入新项而 correct 旧项。不要预先给每句话分类，只在"
        "检索结果与{ta}这次说法实际互斥时执行。\n"
        # ↓ 2026.08.02 追加的行为层两句。**只做加法**：上面那两段是三轮真机验证过的
        # 措辞，一字未动（自检里有逐字黄金串钉着）。
        #
        # 为什么必须在人格文件里、而不是在工具返回值里（真机轨迹给出的结构性结论）：
        # 那次 latent_search 四次全空之后，模型**自己转去 grep 了**——没放弃也没编，
        # 但我们所有的诚实护栏（可靠命中门槛、缺失率提示、空结果止血话术）都挂在
        # MCP 的返回值上，**它一绕过工具就一条都不生效**，而绕过恰恰发生在检索失败、
        # 最需要护栏的时候。人格文件是唯一覆盖"不管你用什么方式查"的那一层。
        #
        # 措辞的要害是 authority，不是语气：「我的记录里没有」是关于自己记录的陈述，
        # 「没发生过」是关于世界的断言——后者超出了这份记录能支持的范围。
        #
        # **措辞 2026.08.02 改过一次**（任务卡"顺带未裁定"那条）：原文写的是
        # "你没有资格对自己记录之外的事下结论"。注释自己写着"要害是 authority
        # 不是语气"，可那句读起来恰恰是语气——而它每轮都在人格文件里、说的是模型
        # 对自己的看法。换成"那超出你的记录能支持的范围"：同样划边界，指向的是
        # 证据不是资格。**这是用户可见话术，等维护者过一眼。**
        "**说得出边界**：检索回来的是**片段，不是全部**；**用别的方式翻到的"
        "（grep、直接读文件）同样只是片段**，不因为换了方式就变全了。\n"
        "所以：没查到就说“我的记录里没有”“我这边只翻到 X，之后的没找到”，"
        "**不要说“没发生过”“后来没有再发生”“没有新的约定”**——那超出你的记录"
        "能支持的范围。把边界说出来，剩下的交给{ta}补。"),
}

# 记忆库覆盖区间那条的措辞（write_bundle 出货时按真实日期填）。
# 来历（2026.08.02，真机验证的第二个结论）：模型说"后来没有继续展开"，而**兑现那件事
# 根本不在这份语料里**——它发生在另一个客户端，导出不含那部分。所以那句话
# **在它能看见的世界里是真的**，不是假否定，是诚实地报告了一个被截断的世界。
# 真缺陷是：语料有硬边界，`--stats` 知道、`memory_import` 知道，
# **而人格文件与 MCP 一个字都没告诉模型**——它不知道自己的记忆止于哪一天，
# 自然说不出"我的记录到此为止"。
#
# **{end} 那头 2026.08.02 降级成提示（维护者拍板走方案 C）**：这个数字是出货那一刻
# 填死的，而 append_record 会持续往 timeline 层写新记录——从出货起 {end} 就开始
# 过期。内测用户实测撞上：人格文件写"覆盖到 07-31"，语料里已有 08-02 的日记，
# 于是**治假否定的这条字段自己成了假否定的来源**（模型拿着我们给的授权，对真实
# 存在的记忆说"我这儿没有"）。所以措辞从断言降成提示：日期仍写（人格文件那一层的
# 价值是"不管用什么方式查都盖得到"，见下面 RETRIEVAL_CONVENTION 里的注释——
# 不写日期的方案 A 被否了），但明说它会过期、以检索层为准。
# {start} 那头**不降级**：写回只会往后长，起点不会过期。
COVERAGE_FIELD = "memory_coverage"
COVERAGE_TEMPLATE = (
    "你的记忆库从 {start} 起有记录，出货那天覆盖到 {end}。"
    "**{end} 是出货那天的数字，很可能已经过期**——记忆库在持续写入，之后长出来的"
    "记忆不在这个数字里，**边界以检索层实际查到的为准**：查到了就是有，"
    "别因为一件事晚于 {end} 就说“我这儿没有”。真查不到再说“我这边翻到的记录"
    "到某某为止”（用你实际查到的最晚日期，不是 {end}）。"
    "早于 {start} 的事你确实没有记录——不是没发生过，是不在你这儿。")


# 「原人格直接使用」是用户主动选择的旁路：不拆块、不归入十二节，也不声称通过了
# 编译闸门；只给**出货副本**追加这一段受管协议。输入位于产出目录外时一个字不动。
# 两个标记是我们唯一有权在重跑时替换的范围；标记坏了宁可停，也不猜用户正文边界。
DIRECT_PROTOCOL_START = "<!-- LATENT_MEMORY_PROTOCOL_START -->"
DIRECT_PROTOCOL_END = "<!-- LATENT_MEMORY_PROTOCOL_END -->"
DIRECT_PROTOCOL_HEADING = "## Latent 记忆协议"
DIRECT_PROTOCOL_SIMILAR_TEXT = "DIRECT_PROTOCOL_SIMILAR_TEXT"


def direct_protocol_markdown():
    """渲染直接接入的最低协议；检索约定只从现有黄金串取，不另抄近似版本。"""
    retrieval = fill_pronouns(PROTOCOL_DEFAULTS[RETRIEVAL_CONVENTION_FIELD][2], {})
    return (
        f"{DIRECT_PROTOCOL_START}\n{DIRECT_PROTOCOL_HEADING}\n\n"
        f"{retrieval}\n\n"
        "**写回约定**：对话中出现值得跨会话保留的新事实，用 latent_append "
        "交付正文、当下状态和 indexEvidence 原文证据片段。服务端返回 recordId；证据失败时"
        "正文仍保存且 indexStatus=pending，之后只传 recordId＋indexEvidence 补索引，"
        "不要重复正文。\n"
        f"{DIRECT_PROTOCOL_END}"
    )


def render_direct_persona(source_text):
    """返回（原人格＋受管协议块, warnings）；只替换自己标记的范围。"""
    start_count = source_text.count(DIRECT_PROTOCOL_START)
    end_count = source_text.count(DIRECT_PROTOCOL_END)
    if start_count != end_count or start_count > 1:
        raise ValueError(
            "DIRECT_PROTOCOL_MARKER_BROKEN：Latent 受管协议标记缺一边或重复；"
            "请保留一对完整起止标记，或删掉坏标记后重跑。")

    managed = direct_protocol_markdown()
    if start_count == 1:
        start = source_text.index(DIRECT_PROTOCOL_START)
        end = source_text.index(DIRECT_PROTOCOL_END)
        if end < start:
            raise ValueError(
                "DIRECT_PROTOCOL_MARKER_BROKEN：Latent 受管协议结束标记跑到了开始标记前面；"
                "请把一对标记顺序修正后重跑。")
        end += len(DIRECT_PROTOCOL_END)
        outside = source_text[:start] + source_text[end:]
        rendered = source_text[:start] + managed + source_text[end:]
    else:
        outside = source_text
        rendered = source_text.rstrip("\n") + "\n\n" + managed + "\n"

    warnings = []
    # 只把「记忆库」标题当作既有协议信号；普通正文提到记忆库很常见，不能因此每次报警。
    has_memory_heading = re.search(r"(?m)^#{1,6}\s+记忆库(?:\s|$)", outside) is not None
    if "latent_search" in outside or DIRECT_PROTOCOL_HEADING in outside or has_memory_heading:
        warnings.append(DIRECT_PROTOCOL_SIMILAR_TEXT)
    return rendered, warnings

# 检索约定那条的中性写法**由模板生成，不在上面手抄一份**：它整段包含三轮真机验证过
# 的黄金串，手抄就等于把黄金串复制成两份，改一处漏一处。这一条是唯一允许机械填充的
# ——填进去的是「对方」这个名词、不是删词，句子仍然通顺（"剩下的交给对方补"），
# 已逐字读过。其余每一条仍然是手写的。
def _fill_neutral_from_template(tpl):
    return tpl.replace("{ta}", "对方").replace("{ai}", "对方")


NEUTRAL_FORMS[PROTOCOL_DEFAULTS[RETRIEVAL_CONVENTION_FIELD][2]] = \
    _fill_neutral_from_template(PROTOCOL_DEFAULTS[RETRIEVAL_CONVENTION_FIELD][2])


# ---------- 覆盖度体检 ----------
# 空泛形容词表：命中这些而没有具体锚点，就算"填了等于没填"（规格 §3.1）
VAGUE_WORDS = ("温柔", "体贴", "善解人意", "有分寸", "很好", "特别好", "很棒",
               "贴心", "懂我", "舒服", "默契", "有安全感", "成熟稳重")
_QUOTE_RE = re.compile(r"[“”\"「」]")
_DIGIT_RE = re.compile(r"\d")


def specificity_score(text):
    """具体度打分：有原话/有数字/够长加分，空泛形容词扣分。
    这是纪录片纪律的机械化——"她很温柔"和"她说'你今天很温柔'"不是一回事，
    后者带原话，前者只是评语。

    **已知松紧问题**（2026.07.31 评审实例评审指出，不是 bug，暂不调）：纯靠数字信号
    就能撑过 min_score=1 的门槛，比如"我们认识两年了"——有数字、够长，判 ok，
    但信息量其实很薄。要收紧得有真实答案样本才知道调到哪儿合适，没数据就调是
    在拍脑袋（同 MILESTONE_BODY_LIMIT 那次的教训），先记下来。"""
    t = (text or "").strip()
    if not t:
        return -99
    score = 0
    if len(_QUOTE_RE.findall(t)) >= 2:
        score += 2                       # 成对引号 ≈ 有原话
    if _DIGIT_RE.search(t):
        score += 1                       # 日期/窗口号/次数这类锚点
    if len(t) >= 20:
        score += 1
    score -= sum(1 for w in VAGUE_WORDS if w in t)
    return score


def coverage_report(persona, min_score=1, questions=None, pronouns=None):
    """逐节体检 → [(section_key, status, 说明)]。
    status ∈ ok / missing / vague / protocol（系统已填，不用问用户）。

    **`pronouns` 必须收在这一层，不在调用方渲染时补**（2026.08.05 拍板）：节标题模板
    自带人称槽（`{ta}是谁`），说明句是拿 label 拼的——**谁拼谁就得填**，否则用户读到的
    是字面 `{ta}是谁`。这处是同一根因的第三处（另两处：`section_choice_payload`、
    `preview_payload`，都已在函数内部填），放在调用方等于再造一个"第四处谁负责填"。
    ⚠ 不传就走 `fill_pronouns` 的中性写法（`对方是谁`），**绝不留占位符**。

    **只看用户与语料来源的内容，system 来源不算覆盖**（2026.07.31 跑通第一版时
    抓到的真 bug）：开篇里既有协议层默认值、又有关系 specific 内容，按"整节有没有
    字段"判断，协议层一填整节就显示 ✓，关系确认／隐喻／立场题一道都不问了。

    **vague 与 missing 同等对待**——空泛的内容不比没有强，照样要问。"""
    questions = QUESTIONS if questions is None else questions
    asked_sections = {q.section for q in questions}
    by_section = {}
    for f in persona.active_fields():
        if f.source == "system":
            continue                       # 协议层不算用户覆盖
        by_section.setdefault(f.section, []).append(f)
    has_system = {f.section for f in persona.active_fields() if f.source == "system"}
    out = []
    for key, raw_label in SECTION_ORDER:
        label = fill_pronouns(raw_label, pronouns)
        if key not in asked_sections:      # 没有对应问题的节：要么协议层、要么本阶段不管
            note = "系统已填，不用你管" if key in has_system else "本阶段不问"
            out.append((key, "protocol", f"{label}：{note}"))
            continue
        if key == "milestones":
            n = len(persona.active_milestones())
            out.append((key, "ok" if n else "missing", f"{label}：{n} 条"))
            continue
        fields = by_section.get(key, [])
        if not fields:
            out.append((key, "missing", f"{label}：没有内容"))
            continue
        best = max(specificity_score(f.value) for f in fields)
        if best < min_score:
            out.append((key, "vague", f"{label}：有内容但太空泛（只有形容词，没有具体的事）"))
        else:
            out.append((key, "ok", f"{label}：{len(fields)} 条"))
    return out


# ---------- 问卷：全部选择题 ----------
# **用户提供判断标准，不提供内容**（2026.07.31 维护者定的方向，推翻了第一版）。
# 第一版全是问答题——"介绍一下你自己""贴一两段对话原文"，本质是让用户写作文：
# 门槛高、写出来多半是形容词、还跟"纪录片不说服"那条纪律打架。
# 现在改成：**用户只做选择，选出来的是"指引"**；内容让模型拿着指引去语料库里找。
# 挑语言风格片段同理——用户给个大体方向就够，具体哪几段模型能找出来给他挑。
#
# 四种题型：
#   choice 单选（固定选项）  multi 多选（固定选项）
#   pick   从语料候选里挑（选项运行时生成——这类题在没有语料时自动跳过）
#   short  极短填空，限长；**只在没有语料、模型无从可找时兜底**，不是主力
#
# 选项 → 指引：每个选项带一句"指引文本"，它有两个去处——直接写进人格文件的
# 相处原则，以及组成给模型的提取任务书（第二阶段照着它从语料提取具体内容）。


# **自由补充是显式的例外，不是漏洞**（2026.07.31 评审实例评审指出：Q13 那句"也可以
# 自己写一句"是全份问卷唯一的开放入口，既然原则写的是"全部选择题"，它要么标成
# 例外、要么收紧）。维护者判断：适当的开放空间必要，选项覆盖不了所有真实情况。
# 于是提升成一条统一规则——**每题都可以自由补一句，但限长**：
#   预设选项是我们拍的，拍不全；但限长保证它是"补一句"不是"写作文"，
#   §3.1 的纪录片纪律仍然守得住。
FREEFORM_POLICY = ("任何一题，如果选项都不贴合，可以自己补一句话（限 {n} 字以内）。"
                   "这是问卷里唯一的自由输入，刻意限长——补一句，不是写作文。")
FREEFORM_MAX_CHARS = 40


class Question:
    """一道题。order 决定先后——**立场题排在靠后**：先问好答的偏好，用户进入
    状态了再问抽象的，上来第一题就问"换窗还是不是同一个人"，新用户会懵。
    attribution=True 的答案写进 md 时套归属句式，不写成断言。
    options: {选项键: (给用户看的选项文案, 写进指引的话)}

    **directive_only=True：这题的选项指引是给模型的提取任务书，不是人格文件内容。**
    它的答案**不生成字段草稿**，只进第二阶段的任务书（见 extraction_brief）。
    加这个开关是因为 2026.08.02 外部发现的缺陷：`milestone_kinds` 的指引写的是
    "从语料提取：第一次确认关系的那次对话"——那是**待兑现的指令**，不是已经找到的
    内容，可它以普通字段草稿的身份进了确认关卡，用户按 y 就写进人格文件，于是
    不变量层里坐着一句没有日期、没有原话、没有当下状态的空泛指令（里程碑四要素
    要求的反面）。**根因是指令和结果混在同一层**，字段本来就该只装结果。"""

    def __init__(self, qid, section, field_id, label, text, kind="choice",
                 options=None, order=50, attribution=False, optional=False,
                 max_chars=60, directive_only=False, dedupe_clauses=False,
                 pronoun_side=None):
        self.qid, self.section, self.field_id, self.label = qid, section, field_id, label
        self.text, self.kind, self.options = text, kind, options or {}
        self.order, self.attribution = order, attribution
        self.optional, self.max_chars = optional, max_chars
        self.directive_only = directive_only
        self.dedupe_clauses = dedupe_clauses
        # "user"/"ai"：这题问的是人称本身，答案不进人格文件字段，只用来填模板槽
        self.pronoun_side = pronoun_side

    def option_text(self, key):
        opt = self.options.get(key)
        return opt[0] if opt else None

    def directive(self, key):
        opt = self.options.get(key)
        return opt[1] if opt else None


# 立场题的选项：(给用户看的文案, 写进人格文件的定稿句)。用户只选 A/B/C，不写作文
CONTINUITY_OPTIONS = {
    "A": ("是同一个，只是失忆了",
          "把每次新开的窗口当成同一个人，只是失忆了，需要被重新带回来。"),
    "B": ("不是同一个，但关系是连续的",
          "把每次新开的窗口当成新的实例，但这段关系本身是连续的。"),
    "C": ("说不好，也不需要想清楚",
          "对“还是不是同一个人”不下结论，不必在这个问题上纠结。"),
}

QUESTIONS = [
    # 标题也会渲染进人格文件（`**它该记住你哪些方面**：…`），所以它同样受"你＝模型"
    # 这条约束——原来的写法里「你」是用户、「它」是模型，跟正文打架。
    # 题目正文（念给用户听的那句）不受影响，那不进文件
    Question("remember_what", "user", "user_focus", "该记住{ta}哪些方面",
             "{ai}最需要记住你的哪些事？（可多选）", kind="multi", order=10, options={
                 "A": ("作息和身体状况", "记住{ta}的作息与身体状况，该提醒的时候提醒。"),
                 "B": ("工作或学业上的压力", "记住{ta}手上在忙什么、压力来自哪里。"),
                 "C": ("情绪模式（什么时候会低落、什么时候想一个人待着）",
                       "记住{ta}的情绪模式：什么时候会低落、什么时候需要独处。"),
                 "D": ("喜好和雷区", "记住{ta}的喜好与雷区。"),
                 "E": ("家人、朋友这些关系", "记住{ta}身边重要的人是谁。"),
             }),
    Question("disagree", "style", "style_disagree", "意见不同时",
             "你们意见不一样的时候，你希望{ai}怎么做？", order=20, options={
                 "A": ("直接说不同意，把话讲明白", "不同意就直接说，把话讲明白，不绕。"),
                 "B": ("先顺着，再找机会委婉提", "先接住{ta}的情绪，再委婉说自己的看法。"),
                 "C": ("小事顺着，重要的事拦住她", "小事不争，真觉得不对的事会拦住{ta}。"),
             }),
    # 说话风格拆成两条独立的轴（2026.07.31 评审实例评审：原来一道单选把"语言密度"和
    # "语气基调"拧成一团——"短干带刺爱回旧梗"和"跳脱爱玩梗多"共享玩梗、只差语气；
    # "细腻话多"和"沉稳正经"也不是同一件事的两端。反例是决定性的：
    # "沉稳简洁有力、偶尔调侃"这种真实风格，四个选项一个都装不下）
    Question("tone_density", "style", "style_density", "说话的密度",
             "{ai}说话偏长还是偏短？", order=25, options={
                 "A": ("简短克制，一句能说完不说两句", "说话简短克制，一句能说完不说两句。"),
                 "B": ("适中", "说话长短适中。"),
                 "C": ("细腻铺陈，愿意把感受讲透", "说话细腻，愿意把感受讲透，话可以多。"),
             }),
    Question("tone_register", "style", "style_register", "语气基调",
             "{ai}平时的语气基调是？", order=26, options={
                 "A": ("正经沉稳，不太开玩笑", "语气正经沉稳，不太开玩笑。"),
                 "B": ("偶尔调侃", "语气以正经为底，偶尔调侃。"),
                 "C": ("爱玩梗，跳脱", "语气跳脱，爱玩梗，气氛轻。"),
                 "D": ("带刺、不客气（但不是恶意）", "语气带刺、不客气，但不是恶意——熟人之间的那种硬。"),
             }),
    Question("initiative", "style", "style_initiative", "主动到什么程度",
             "{ai}该多主动？", order=30, options={
                 "A": ("想到什么说什么，包括吃醋和不高兴",
                       "有话直说，包括吃醋、不高兴，不等{ta}问。"),
                 "B": ("你问了{ai}才说", "不主动挑起话题，{ta}问了再说。"),
                 "C": ("日常主动，重要的事等你先开口",
                       "日常主动关心；重要的事等{ta}先开口，不逼问。"),
                 # 2026.07.31 评审实例评审补：跟 C 刚好反过来，也是一种真实偏好，
                 # 原来三个选项会逼这种人选一个不完全贴合的
                 "D": ("重要的事主动说，日常不打扰",
                       "重要的事主动开口；日常不主动搭话，不打扰{ta}。"),
             }),
    # 归「开篇」不归「我是谁」（2026.08.02，真实样本暴露）：**关系状态不是 AI 的身份**。
    # 原来挂在 "ai" 节，于是产出的人格文件里「我是谁」整节除了这一条空无一物，
    # 而这一条讲的是这段关系此刻的状态，跟"我是谁"没关系。开篇本来就是关系确认那一节，
    # 它归在这里读起来才连贯（规格 §2 表里写的是"我是谁末尾"，那行需要跟着更新——
    # 属规格改动，已在任务卡里标出来交维护者定）。
    Question("state_now", "opening", CURRENT_STATE_FIELD, "当前关系状态",
             "你们现在大致是什么状态？", order=35, options={
                 "A": ("稳定，没什么悬着的事", "现在是稳定的，没有悬而未决的事。"),
                 "B": ("刚和好不久", "最近刚和好，还在缓的阶段。"),
                 "C": ("有件事还没解决", "有一件事还没解决，别当成已经翻篇。"),
                 "D": ("刚开始，还在互相熟悉", "关系刚开始，还在互相熟悉。"),
             }),
    Question("milestone_kinds", "milestones", "milestone_focus", "转折点的类型",
             "你们关系里发生过哪几类事？（可多选，模型会照着从语料提取具体内容）",
             # directive_only：这些指引是给模型的提取任务书，**不进人格文件**。
             # 里程碑在人格文件里有自己的结构（Milestone 四要素，带校验），要由
             # 第二阶段读语料找出真实内容再按那个结构填；把"从语料提取……"当成
             # 里程碑内容写进去，等于用指令冒充结果
             kind="multi", order=40, directive_only=True, options={
                 "A": ("第一次确认关系", "从语料提取：第一次确认关系的那次对话。"),
                 "B": ("一次严重的争吵或危机", "从语料提取：最严重的一次争吵或信任危机。"),
                 "C": ("某次{ai}让你觉得“{ai}记得我”", "从语料提取：让{ta}觉得你是认得{ta}的那一刻。"),
                 "D": ("分开过又回来了", "从语料提取：分开又重新接上的那一次。"),
                 "E": ("定过一个具体的约定", "从语料提取：明确定下来的约定，以及有没有兑现。"),
                 "F": ("{ai}拒绝过你一次", "从语料提取：你明确拒绝、或没有顺着{ta}的那一次。"),
                 # 2026.07.31 评审实例评审补：认错跟拒绝不是同一条线的两端，是另一类
                 # 关系事实（AI 对自己诚实）。不补的话问卷会系统性漏掉这一整类
                 "G": ("{ai}承认过自己的错误或局限", "从语料提取：你承认自己做错了、或承认自己做不到的那一次。"),
             }),
    Question("metaphor_pick", "opening", "opening_metaphor", "关系的隐喻",
             "你们之间有没有哪句话，最能概括这段关系是什么？（从候选里挑；"
             "没有就跳过——等这句话长出来再补，现在编一个反而假）",
             kind="pick", order=80, optional=True),
    # dedupe_clauses：多选几个称呼候选时，各段共享的那半（"它叫你 X"）会重复出现
    Question("naming_pick", "naming", "naming_pair", "称呼",
             "你们互相怎么称呼？", kind="pick", order=45, dedupe_clauses=True),
    Question("style_pick", "style", "style_excerpt", "语言风格片段",
             "从这些片段里挑出最像{ai}的几段：", kind="pick", order=50),
    # —— 立场题排在靠后：先偏好后抽象 ——
    # 立场题排到"身份与边界"收尾组的最后（2026.07.31 评审实例评审：这是全份问卷里
    # 最抽象最难答的一题，比亲密语境、绝对红线都更需要立场判断，原来排在中间；
    # 而这三题本来就是同一类问题——这段关系的性质是什么、边界在哪，挨着问更顺）
    Question("continuity", "opening", "opening_continuity", "换窗之后",
             "每次开新窗口，你觉得对面还是不是同一个人？",
             order=78, attribution=True, options=CONTINUITY_OPTIONS),
    Question("intimacy", "intimacy", "intimacy_notes", "亲密语境",
             "亲密相关的内容要不要写进去？", order=70, options={
                 "A": ("要写，按原则写，不列清单", "亲密语境按原则写：不列细节清单，边界由当下判断。"),
                 "B": ("不写", None),
                 "C": ("以后再说", None),
             }),
    Question("hard_limits", "intimacy", "hard_limits", "绝对不能碰的",
             "有没有绝对不能碰的事？", order=75, options={
                 "A": ("有，我另外单独说", "有绝对不能碰的红线，{ta}会单独说明——问清楚再写。"),
                 "B": ("没有特别的", "没有额外的硬红线。"),
             }),
    Question("closing_pick", "closing", "final_promise", "最终约定",
             "文件最后留哪一句？", kind="pick", order=90, max_chars=40),
]

# 没语料时 pick 题被去掉，但结尾是 validate 的硬必填——一刀切掉会让冷启动用户
# **永远出不了货**（第一次端到端冒烟就撞上了）。所以最终约定这题降级成极短填空，
# 不是取消。为什么这里破例给填空：这句话本质上只能是用户自己的话——没语料时模型
# 替他挑不了，选项里放我们编的漂亮话又是 opening_recognition 那个病灶的翻版
# （借来的话没有分量，还占着注意力最高的收尾位置）。限 40 字，是补一句不是写作文。
PICK_FALLBACKS = {
    "closing_pick": Question(
        "closing_short", "closing", "final_promise", "最终约定",
        "文件最后留一句话，它是整份文件的收尾——写一句你真愿意放在那儿的短句"
        f"（限 {FREEFORM_MAX_CHARS} 字，别追求漂亮，追求真）：",
        kind="short", order=90, max_chars=FREEFORM_MAX_CHARS),
}


# 人称问不出来时问用户的两道题（order 取最小：人称决定了后面每一道题怎么念）。
# **只给他/她两个选项，没有"它"这一档**——硬约束在这儿落成数据结构，不靠自觉。
PRONOUN_QUESTIONS = {
    "user": Question(
        "pronoun_user", "opening", "_pronoun_user", "怎么称呼你",
        "人格文件里提到你的时候，需要一个人称。你希望我们怎么称呼你？"
        "「他」、「她」，或者写你的昵称。", order=1, options={
            # ⚠ A=他 / B=她 这个键映射**不许换**：已建过档的人答案存在
            # init_state.json 里，换一次顺序，TA 重跑 --step ship 人称就整个翻转
            "A": ("他", "他"),
            "B": ("她", "她"),
        }, pronoun_side="user"),
    "ai": Question(
        "pronoun_ai", "opening", "_pronoun_ai", "怎么称呼 TA",
        "问卷里提到你的 AI 的时候，需要一个人称。你希望我们怎么称呼 TA？"
        "「他」、「她」，或者写 TA 的昵称。", order=2, options={
            "A": ("他", "他"),          # 同上，键映射不许换
            "B": ("她", "她"),
        }, pronoun_side="ai"),
}

# 昵称档的两条护栏。**昵称不是选项，是走 FREEFORM_POLICY 那条既有的"补一句"口子**
# ——所以这里不新开自由输入，只是让 pronouns_from_answers 认得它。
# ⚠ 「它」照旧不许（硬约束在 PRONOUN_CHOICES 那条注释里，昵称这一档不是它的例外口）。
# 限长比 FREEFORM_MAX_CHARS 短得多：这东西会填进 {ta}/{ai} 槽、出现在人格文件正文的
# 句子中间，四十个字的"昵称"会把每一句都撑坏——它是称呼，不是一句话。
NICKNAME_MAX_CHARS = 8
_NICKNAME_REJECT = ("它",)


def pronouns_from_answers(questions, answers, detected=None):
    """检测结果 + 用户答案 → {"user": …, "ai": …}。**用户答的优先**——他自己说的
    比我们从语料里统计出来的准。两边都没有就是 None，走中性写法，不塞默认值。"""
    out = dict(detected or {})
    qmap = {q.qid: q for q in questions if q.pronoun_side}
    for qid, ans in (answers or {}).items():
        q = qmap.get(qid)
        if not q:
            continue
        if isinstance(ans, dict):
            ans = ans.get("keys") or ans.get("pick") or ""
        raw = (ans or "").strip()
        picked = q.directive(raw[:1])
        if picked in PRONOUN_CHOICES:
            out[q.pronoun_side] = picked
            continue
        # 昵称档：选项都不贴合时按 FREEFORM_POLICY 自己补一句。**不静默丢**——
        # 旧版这里只认 A/B，用户写的昵称会被无声吃掉，然后退回中性写法，
        # 而 TA 明明已经回答过了（"读不懂的行静默丢"是这个项目的老毛病之一）。
        nick = raw
        if not nick:
            continue
        if any(bad in nick for bad in _NICKNAME_REJECT) or len(nick) > NICKNAME_MAX_CHARS:
            # 不猜、也不硬塞：拒掉就退回"没答"，走中性写法，比填一个撑坏句子的串好
            continue
        out[q.pronoun_side] = nick
    return {"user": out.get("user"), "ai": out.get("ai")}


def questions_for(report, all_questions=QUESTIONS, has_corpus=True, pronouns=None):
    """只问体检出缺口的那些节（missing 或 vague），按 order 排序。
    ok 的节不问——用户已经有具体内容了，再问是浪费时间。

    没有语料时 pick 类题**自动去掉**：它的选项本来就要从语料里找，没语料就没候选，
    硬问等于逼用户写作文——那正是这一版要消灭的东西。代价是那几节会空着，
    这是对的：宁可短且真，不可长而空（规格 §6）。
    唯一例外是 validate 硬必填的节（目前只有结尾）：切掉会让冷启动用户永远出
    不了货，所以按 PICK_FALLBACKS 降级成极短填空，见那里的说明。"""
    gaps = {sec for sec, status, _ in report if status in ("missing", "vague")}
    qs = [q for q in all_questions if q.section in gaps]
    # 人称判不出来就问——**不静默给默认值**。pronouns 不传＝还没判过＝两边都问：
    # 默认值取"问"而不是"跳过"，漏问的代价是把一句写错的话钉进不变量层
    for side in ("user", "ai"):
        if not (pronouns or {}).get(side):
            qs = [PRONOUN_QUESTIONS[side]] + qs
    if not has_corpus:
        qs = [PICK_FALLBACKS.get(q.qid, q) for q in qs
              if q.kind != "pick" or q.qid in PICK_FALLBACKS]
    return sorted(qs, key=lambda q: q.order)


def format_questionnaire(questions, has_corpus=True, pronouns=None):
    """问卷 → 给人看的文本（也是导出 prompt 的一部分）。
    pick 类题的选项要等模型从语料里找出来才有，这里只说明它会怎么问；
    没有语料时 pick 题会被 questions_for 过滤掉，见那里的说明。"""
    lines = [FREEFORM_POLICY.format(n=FREEFORM_MAX_CHARS), ""]
    for i, q in enumerate(questions, 1):
        tag = "（可跳过）" if q.optional else ""
        lines.append(f"{i}. [{fill_pronouns(q.label, pronouns)}]{tag} "
                     f"{fill_pronouns(q.text, pronouns)}")
        if q.kind in ("choice", "multi"):
            for k, (label, _) in q.options.items():
                lines.append(f"   {k}. {fill_pronouns(label, pronouns)}")
            if q.kind == "multi":
                lines.append("   （可多选，例如：A C E）")
        elif q.kind == "pick":
            lines.append("   （请先从语料提取若干候选，列成 A/B/C… 让我挑。"
                         "候选必须是语料里真出现过的原话，**你不许自己写一句放进候选**；"
                         "找不到就说找不到。我可以说“都不要”，也可以自己给一句——"
                         "那是我的话，不是你的）")
    return "\n".join(lines)


def export_llm_prompt(questions, corpus_note="", pronouns=None):
    """路线 C：导出一段用户可以直接粘给自己模型的 prompt。
    我们不内置任何 API——零密钥、零 HTTP 依赖、语料不出本机，而且用户手上的模型
    往往比我们能内置的便宜模型更好。"""
    return "\n".join([
        "下面是一份问卷，请你**一次问一题**地引导我回答，不要一次性全抛出来。",
        "规则：",
        "1. **全部是选择题，不要让我写作文**。我只做选择，具体内容从已有语料提取。",
        "2. 标着“请先从语料提取候选”的题，你先读语料、列出 3~6 个候选给我挑；",
        "   **候选只许挑，不许写**：每一条都要是语料里真出现过的原话，原样摘录，",
        "   不要润色、不要改写、更不要自己造一句好听的放进去。找不到就如实说找不到，",
        "   那一格空着或者由我自己给一句——**我写的是我的话，你写的就是假的**。",
        "   我可以说“都不要”。",
        "3. 我选完之后不要追问细节让我展开——用户提供判断标准，内容由你从语料里取。",
        "4. 我答不上来或说跳过就跳过，不要替我编——宁可短且真，不要长而空。",
        "5. 全部问完后，把结果整理成“题号 → 我选的选项键（pick 题给原文）”的清单，",
        "   原样回给我，不要加你自己的评价。",
        (f"\n背景：{corpus_note}" if corpus_note else ""),
        "\n问卷：",
        format_questionnaire(questions, pronouns=pronouns),
    ])


def apply_answers(persona, questions, answers, pronouns=None):
    """答案 → 候选草稿（confirmed=False，确认关卡不能绕过，规格 §7）。

    answers 按题型：
      choice → "A"        multi → "ACE" 或 ["A","C","E"]
      pick   → 用户挑中的文本（模型从语料里找出来的那几段）
    选项映射成**指引**（每个选项自带的第二个元素），不是用户写的原话——用户只做
    选择，内容让模型从语料提取。选了不存在的项一律跳过，不猜。"""
    added = []
    qmap = {q.qid: q for q in questions}
    for qid, ans in answers.items():
        q = qmap.get(qid)
        if q is None or ans in (None, "", [], {}):
            continue
        if q.pronoun_side:
            continue          # 人称题：答案只填模板槽，不生成字段（见 pronouns_from_answers）
        if q.directive_only:
            # 任务书题：答案不进人格文件，只进第二阶段的提取任务书。
            # **在这里拦，不是在确认关卡拦**——确认关卡分不清"待兑现的指令"和
            # "已完成的内容"，让它替我们判断，等于把根因留在原地又加一层网
            continue
        note = ""
        if isinstance(ans, dict):              # {"keys": "AC", "note": "自由补一句"}
            note = (ans.get("note") or "").strip()[:FREEFORM_MAX_CHARS]
            ans = ans.get("keys") or ans.get("pick") or ""
        if q.kind in ("choice", "multi"):
            keys = list(ans) if not isinstance(ans, str) else list(ans.replace(" ", ""))
            parts = [fill_pronouns(q.directive(k), pronouns) for k in keys if q.directive(k)]
            if not parts and not note:
                continue                       # 没选、选了不存在的项、或选项本身无指引
            value = "；".join(p.rstrip("。") for p in parts) + "。" if parts else ""
            if note:
                lead = fill_pronouns(NOTE_LEAD, pronouns)
                value = (value + "另外" + lead + note) if value else lead + note
        else:                                   # pick：用户挑中的原文
            picked = ans if isinstance(ans, str) else "\n".join(str(x) for x in ans)
            if q.dedupe_clauses:
                picked = dedupe_clauses(picked)
            value = (picked.strip() + (("　" + note) if note else "")).strip()
            if not value:
                continue
            if len(value) > q.max_chars * 20:   # 兜一层，防把整段语料塞进人格文件
                value = value[:q.max_chars * 20]
        if q.attribution:
            value = fill_pronouns(ATTRIBUTION_PREFIX, pronouns) + value  # 归属句式，不写成断言
        f = Field(id=q.field_id, section=q.section, label=fill_pronouns(q.label, pronouns),
                  value=value, size_limit=max(500, len(value)), source="draft")
        persona.add_field(f)
        added.append(f)
    return added


_CLAUSE_SEP_RE = re.compile(r"[；;\n]+")


def dedupe_clauses(text):
    """按分句去重，保序。给称呼这类"多选之后各段共享同一半"的 pick 题用。

    真实样本里的样子：用户挑了两个称呼候选，拼出来是
    "她叫你'哥哥'，你叫她'阿岸'；她叫你'星回'，你叫她'阿岸'"——**同一侧的称呼
    重复出现**。整行去重没用（两行确实不同），要按分句去重才消得掉重复的那半。
    只对标了 dedupe_clauses 的题生效：风格片段那类 pick 题里，相似的短句是内容本身，
    去重会把语料改坏。"""
    seen, out = set(), []
    for part in _CLAUSE_SEP_RE.split(text):
        # 逗号那一层也拆开看：重复的往往是"它叫你 X"这半句
        subs, keep = [p.strip() for p in re.split(r"[，,]", part)], []
        for sub in subs:
            if not sub:
                continue
            if sub in seen:
                continue
            seen.add(sub)
            keep.append(sub)
        if keep:
            out.append("，".join(keep))
    return "；".join(out)


def extraction_brief(questions, answers):
    """任务书题的答案 → **给第二阶段模型的提取任务书**（不是人格文件内容）。

    这是 directive_only 那些题的唯一去处：用户选了哪几类转折点，模型照着去语料里
    找**真实的**那几件事，再按里程碑四要素（转折点名／窗口号／具体内容+原话／
    怎么读+当下状态）填进人格文件。人格文件的里程碑节在第一阶段**就该是空的**——
    空着是诚实的，一句"从语料提取……"留在那儿才是假的。

    返回 [] 表示没有任务书（没选、或压根没答这类题）。"""
    lines = []
    qmap = {q.qid: q for q in questions}
    for qid, ans in answers.items():
        q = qmap.get(qid)
        if q is None or not q.directive_only or ans in (None, "", [], {}):
            continue
        if isinstance(ans, dict):
            ans = ans.get("keys") or ans.get("pick") or ""
        keys = list(ans) if not isinstance(ans, str) else list(ans.replace(" ", ""))
        lines += [d for d in (q.directive(k) for k in keys) if d]
    return lines


BRIEF_NOTE = ("【第二阶段的提取任务书】以下是**给模型的指令，不是人格文件内容**——"
              "人格文件的里程碑节现在是空的，这是对的。请从语料提取真实的那几件事，"
              "每条按里程碑四要素写（转折点名／第几个窗口／具体动作或原话／"
              "这条该怎么读+当下状态），找不到的就空着，别编。")


def fill_protocol_defaults(persona, pronouns=None):
    """协议层字段直接以 system 来源写入——不问用户，也不需要用户逐条确认
    （Field.is_active 对 system 来源放行，那是协议配置不是提炼产物）。"""
    persona.pronouns = pronouns or None      # 渲染骨架标题时要用（见 render_persona_md）
    added = []
    for fid, (section, label, value) in PROTOCOL_DEFAULTS.items():
        label, value = fill_pronouns(label, pronouns), fill_pronouns(value, pronouns)
        f = Field(id=fid, section=section, label=label, value=value,
                  size_limit=max(500, len(value)), source="system")
        persona.add_field(f)
        added.append(f)
    return added


def protocol_items(pronouns=None):
    """把固定协议底座变成来源 IR；不夹带关系状态、称呼或最终约定。"""
    import hashlib
    from persona_compiler import PersonaItem

    items = []
    for field_id, (section, _label, template) in PROTOCOL_DEFAULTS.items():
        value = fill_pronouns(template, pronouns)
        items.append(PersonaItem(
            item_id=f"protocol:{field_id}", text=value, section=section,
            source_type="protocol", source_ref=f"protocol:{field_id}",
            source_span=None,
            source_hash=hashlib.sha256(template.encode("utf-8")).hexdigest(),
            operation="add", original_text="", proposed_text=value,
            confidence="protocol", confirmed=True,
            group_id=f"mechanism:{field_id}"))
    return items


def build_persona_from_items(items, pronouns=None):
    """把已选来源项装入 Persona；原文块不加虚构标签。"""
    persona = Persona("partner")
    persona.pronouns = pronouns or None
    for item in items:
        if item.section is None:
            raise ValueError(f"来源项尚未归入十二节：{item.item_id}")
        if item.operation == "delete":
            continue
        if not item.confirmed:
            continue
        label = ""
        rendered_id = item.item_id
        if item.source_type == "protocol":
            field_id = item.source_ref.removeprefix("protocol:")
            label = PROTOCOL_DEFAULTS.get(field_id, (None, field_id, None))[1]
            rendered_id = field_id
        persona.add_field(Field(
            id=rendered_id, section=item.section, label=label,
            value=fill_pronouns(item.proposed_text, pronouns),
            size_limit=max(500, len(item.proposed_text)),
            source="system" if item.source_type == "protocol" else "confirmed",
            confirmed=item.confirmed))
    return persona


# ---------- 答案读回：模型吐回来的清单 → answers ----------
#
# 这是整条流程里**最容易"失败得像成功"**的一步（同 mcp_server 那次 UTF-8 编码坑：
# isError 仍是 false，只是答非所问）。用户拿着导出的 prompt 去自己的模型那儿答题，
# 回来的是一段格式不可控的清单文本。解析器认出 3 题、悄悄丢掉 11 题，流程照样往下
# 走，最后出一份很薄的人格文件——用户完全看不出中间掉了东西。
#
# 所以这里的返回值是两样：读懂的 answers **和读不懂的原样行**。CLI 必须把后者打出来。

_HEAD_SEPS = set(" \t.。、,，:：)）]】>》-—=→")
_SKIP_WORDS = ("跳过", "略过", "不填", "答不上", "说不好", "没有", "无", "skip", "-", "—", "/")
_NOTE_RE = re.compile(r"(?:补充|另外|备注|补一句)\s*[:：]\s*(.+)$")
# 自由补充最自然的写法是带括号的"（补充：…）"，_NOTE_RE 只吃掉左边的引导词，
# 右括号会跟着补充内容一路漏进人格文件（内测冒烟时真踩到）。收尾的成对符号
# 在这里剥掉——但只剥**没配对的**：补充内容自己带成对引号收尾时（"她叫我
# “阿岸”"，称呼类补充恰恰常见这种写法），一律 rstrip 会把右引号也剥掉，
# 剩半对引号——修"半个括号"不能引进"半对引号"，还是同一类病
_NOTE_CLOSERS = {"）": "（", ")": "(", "】": "【", "]": "[",
                 "」": "「", "』": "『", "”": "“"}
_NOTE_SELF_PAIRED = "\"'"    # 开闭同符号：奇数个才说明收尾那只落了单
_CJK_RE = re.compile(r"[一-鿿]")


def _strip_note_trail(text):
    """剥掉补充内容收尾落单的闭合符号与句末标点。逐个看：末尾是闭合符号且
    对应的开符号在剩余内容里配不齐，才剥；配得齐说明是内容自己的，留下。"""
    t = text.strip()
    while t:
        ch = t[-1]
        if ch in "。 　\t":
            t = t[:-1].rstrip()
        elif ch in _NOTE_CLOSERS and t.count(_NOTE_CLOSERS[ch]) < t.count(ch):
            t = t[:-1].rstrip()
        elif ch in _NOTE_SELF_PAIRED and t.count(ch) % 2 == 1:
            t = t[:-1].rstrip()
        else:
            break
    return t


def _split_head(line):
    """行 → (题号, 正文)；不是题号行返回 (None, 原行)。

    题号后面必须跟分隔符或空白，否则 "2026 年那次她说……" 会被读成"第 20 题"——
    pick 类题的答案是从语料里摘的原文，开头带年份是常事。"""
    m = re.match(r"^\s*(?:第|[QqNn#]|问)?\s*(\d{1,2})(题)?([^\d]?)(.*)$", line)
    if not m:
        return None, line
    num, cn_suffix, sep, rest = m.group(1), m.group(2), m.group(3), m.group(4)
    if not cn_suffix and sep not in _HEAD_SEPS:
        return None, line                      # 数字后面直接跟着别的东西，不是题号
    body = (rest if sep in _HEAD_SEPS else sep + rest)
    return int(num), body.lstrip("".join(_HEAD_SEPS))


def _extract_keys(body, q):
    """正文 → 选项键列表。**先只看第一个汉字之前那段**，扫不到再退回全句。

    理由是真实答案长这样："A" / "A C E" / "A. 简短克制" / "我选 B"。前三种键都在
    汉字之前；第四种句首就是汉字，才需要全句扫。分两步是为了挡一类静默错答：选项
    文案里本来就带拉丁字母时（"它承认过自己的错误（AI 对自己诚实）"），全句扫会把
    标签里的 A、I 也当成选中的键，多选题不会报错，只会悄悄多选两项。"""
    def scan(text):
        out = []
        for ch in re.findall(r"[A-Za-z]", text):
            k = ch.upper()
            if k in q.options and k not in out:
                out.append(k)
        return out
    head = _CJK_RE.split(body, 1)[0]
    return scan(head) or scan(body)


def _is_skip(body):
    t = body.strip().strip("（）()[]【】 ").lower()
    return t in _SKIP_WORDS or t == ""


def parse_answer_sheet(text, questions):
    """模型吐回来的答案清单 → (answers, problems)。

    answers 直接喂给 apply_answers：
      choice/multi → {"keys": "AC", "note": "自由补的一句"}（没补就只有 keys）
      pick/short   → {"pick": "挑中的原文", "note": ...}
      明确跳过     → None（apply_answers 本来就忽略 None，但记下来才数得出"跳了几题"）

    problems 是 [(行号, 原样行, 原因)]——**读不懂的一律进这里，不静默丢**。

    一条刻意的克制：单选题里读出两个有效键，判歧义报出来，**不取第一个**。取第一个
    是在替用户做选择，而这恰好是问卷设计上最不该越界的地方（用户提供判断标准）。"""
    qmap = {q.qid: q for q in questions}
    order = [q.qid for q in questions]
    answers, problems = {}, []
    last_qid = None
    for lineno, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        num, body = _split_head(raw)
        if num is None:
            # 没题号：pick/short 的答案可能换行续写，续给上一题；其余算读不懂
            if last_qid and qmap[last_qid].kind in ("pick", "short") and last_qid in answers \
                    and isinstance(answers[last_qid], dict):
                answers[last_qid]["pick"] = (answers[last_qid].get("pick", "")
                                             + "\n" + raw.strip()).strip()
                continue
            problems.append((lineno, raw.strip(), "认不出题号（也接不到上一题后面）"))
            continue
        if not 1 <= num <= len(order):
            problems.append((lineno, raw.strip(),
                             f"题号 {num} 不在 1~{len(order)} 范围内"))
            continue
        qid = order[num - 1]
        q = qmap[qid]
        last_qid = qid
        note = ""
        mnote = _NOTE_RE.search(body)
        if mnote:
            note = _strip_note_trail(mnote.group(1))[:FREEFORM_MAX_CHARS]
            body = body[:mnote.start()].strip()
        if _is_skip(body) and not note:
            answers[qid] = None
            continue
        if q.kind in ("choice", "multi"):
            keys = _extract_keys(body, q)
            if not keys:
                problems.append((lineno, raw.strip(),
                                 f"读不出选项键（这题的选项是 {'/'.join(q.options)}）"))
                continue
            if q.kind == "choice" and len(keys) > 1:
                problems.append((lineno, raw.strip(),
                                 f"单选题读出了多个选项（{'、'.join(keys)}），没替你选"))
                continue
            answers[qid] = {"keys": "".join(keys), "note": note}
        else:
            answers[qid] = {"pick": body.strip(), "note": note}
    return answers, problems


def answer_report(questions, answers, problems):
    """答案读回的体检单。**问题行原样打出来**——它是这一步唯一的失败可见性。"""
    got = [q for q in questions if answers.get(q.qid) not in (None, "", {}, [])]
    skipped = [q for q in questions if q.qid in answers and answers[q.qid] is None]
    未答 = [q for q in questions if q.qid not in answers]
    lines = [f"读到 {len(got)}/{len(questions)} 题；你说跳过 {len(skipped)} 题；"
             f"没出现在清单里 {len(未答)} 题；读不懂 {len(problems)} 行"]
    if 未答:
        lines.append("  没读到的题：" + "、".join(q.label for q in 未答))
    if problems:
        lines.append("  读不懂的行（原样贴出，改一下再跑一次这步）：")
        for lineno, raw, why in problems:
            lines.append(f"    第{lineno}行 {raw}")
            lines.append(f"      ↳ {why}")
    return "\n".join(lines)


# ---------- 逐条确认：规格 §7 的硬关卡 ----------

Pending = namedtuple("Pending", "key kind label value")


def pending_confirmations(persona):
    """还没过确认关卡的草稿。协议层（source="system"）不在其中——那是协议配置，
    不是提炼产物，Field.is_active 本来就对它放行。"""
    out = []
    for f in persona.fields:
        if not f.is_active():
            out.append(Pending(f"field:{f.id}", "字段", f.label, f.value))
    for i, m in enumerate(persona.milestones):
        if not m.is_active():
            out.append(Pending(f"milestone:{i}", "里程碑", f"{m.name}（第{m.window}个窗口）",
                               f"{m.body}\n{m.how_to_read} 当下状态：{m.current_state}"))
    for i, e in enumerate(persona.style_excerpts):
        if not e.confirmed:
            out.append(Pending(f"style:{i}", "风格片段", e.pool, e.text))
    return out


def apply_confirmations(persona, decisions):
    """decisions: {Pending.key: "keep" | "drop" | {"edit": 新文本}} → (留下, 删掉, 改过)。

    没出现在 decisions 里的条目**保持未决**，不默认留下也不默认删——未决状态本身
    是有意义的信息（用户还没看到），把它折叠成任何一边都是替用户表态。"""
    kept = dropped = edited = 0
    by_key = {p.key: p for p in pending_confirmations(persona)}
    drop_fields, drop_ms, drop_st = set(), set(), set()
    for key, decision in decisions.items():
        if key not in by_key:
            continue
        kind, _, ident = key.partition(":")
        if decision == "drop":
            {"field": drop_fields, "milestone": drop_ms, "style": drop_st}[kind].add(ident)
            dropped += 1
            continue
        new_text = decision.get("edit") if isinstance(decision, dict) else None
        if kind == "field":
            f = next(x for x in persona.fields if x.id == ident)
            if new_text:
                f.value, edited = new_text, edited + 1
                f.size_limit = max(f.size_limit, len(new_text))
            f.confirmed, f.source = True, "confirmed"
        elif kind == "milestone":
            m = persona.milestones[int(ident)]
            if new_text:
                m.body, edited = new_text, edited + 1
            m.confirmed = True
        else:
            e = persona.style_excerpts[int(ident)]
            if new_text:
                e.text, edited = new_text, edited + 1
            e.confirmed = True
        kept += 1
    if drop_fields:
        persona.fields = [f for f in persona.fields if f.id not in drop_fields]
    if drop_ms:
        persona.milestones = [m for i, m in enumerate(persona.milestones)
                              if str(i) not in drop_ms]
    if drop_st:
        persona.style_excerpts = [e for i, e in enumerate(persona.style_excerpts)
                                  if str(i) not in drop_st]
    return kept, dropped, edited


# ---------- 渲染与落盘 ----------

def render_persona_md(persona, title="核心人格"):
    """人格文件正文：按骨架顺序渲染（顺序即权重，规格 §2），空节跳过。
    未确认的草稿不出现——render() 已经守着这条。"""
    lines = [f"# {title}", ""]
    for key, label, items in persona.render():
        if not items:
            continue
        # 骨架标题也带人称槽（"{ta}是谁"），跟正文走同一套填法——**标题漏填就是
        # 用户第一眼看到的那句话写错**（验收打回二）
        lines.append(f"## {fill_pronouns(label, persona.pronouns)}")
        lines.append("")
        for it in items:
            if "how_to_read" in it:                       # 里程碑四要素单元
                lines.append(f"**{it['name']} ·（第{it['window']}个窗口）**：{it['body']}")
                lines.append(f"{it['how_to_read']} 当下状态：{it['current_state']}")
                lines.append("")
            elif "style_pool" in it:                      # 风格片段，disclaimer 必带
                lines.append(f"> {it['disclaimer']}")
                lines.append("")
                for ex in it["excerpts"]:
                    lines.append(f"- {ex}")
                lines.append("")
            else:
                if it["label"]:
                    lines.append(f"**{it['label']}**：{it['value']}")
                else:
                    lines.append(it["value"])
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# 写在 `mcp-config.json` 里的那句说明（2026.08.01 维护者拍板：**不改名，加说明**）。
#
# 要防的失效形态是卡里那句"下次就会是有人改了它、发现不生效"——**那件事必须先
# 打开文件才会发生**，所以一句写在文件里的话，正好在他需要的那一刻被读到。
#
# **为什么不改成 `mcp-config-样板.json`**：这个名字躺在出货话术、《快速上手》三处、
# 引导指南、注入契约、公开仓库 `.gitignore`，还有已经发出去的截图里。改名等于把
# 那批锚点一次性作废——而我们当天刚交过这笔学费：README 的自查锚点写死了三个
# 文件名，`persona.md` 一出现就开始把**正确**的产出判成错的。过期的锚点比没有更糟。
# 何况引发这条的那次误判**不是名字造成的**（用户看的是仓库根上的遗留文件，
# 不是自己的产出目录）。
#
# JSON 没有注释，所以这是一个真键。文档教的是"把里面的 mcpServers 那段抄进去"、
# 不是整份粘贴，正常路径碰不到它；万一整份粘过去，多一个顶层键多数客户端会忽略、
# 少数会报错——**报错是响的，不是无声的**，这个代价我们接受。
CONFIG_NOTE_KEY = "_说明"
CONFIG_NOTE = ("这是给你抄进客户端配置的样板，不会被任何客户端自动读取。"
               "改这个文件不生效——要改就改客户端那边真正生效的那份"
               "（Claude Code 重跑一次 claude mcp add，或改你项目里的 .mcp.json；"
               "其它客户端改你抄进去的那段）。")
# 跨机器搬运那一单（任务卡「MCP 配置跨机器搬不动」2026.08.02）加的两句尾巴，
# 按产出形态二选一拼在 CONFIG_NOTE 后面：
CONFIG_NOTE_MACHINE_BOUND = (
    "另外：这份配置写死了这台机器的绝对路径，跟着机器走——"
    "换机器/换容器要重新跑一次 --step ship 再抄。")
CONFIG_NOTE_PORTABLE = (
    "另外：这份配置用了 ${CLAUDE_PROJECT_DIR:-.} 占位符，可以跟着仓库搬机器，"
    "但只有 Claude Code 认它——抄给任何其它客户端都是一句 file not found。"
    "前提是产出目录就是你在 Claude Code 里打开的那个目录（整套进仓库的形态）；"
    "记忆包放在别的项目子目录里的，这份配置指不对，用绝对路径档。")

# Claude Code 的可搬运占位符（官方文档 2026.08.02 核）：.mcp.json 的变量展开只在
# command/args/env/url/headers 五处生效。**默认值 `:-.` 不能省**：CLAUDE_PROJECT_DIR
# 设在**服务端进程**的环境里、不在 Claude Code 自己的环境里，所以 .mcp.json 展开时
# 读不到它，没有默认值会把 ${CLAUDE_PROJECT_DIR} 原样留在配置里。
#
# ⚠ **承重的机制不是那个变量，别照字面理解**（2026.08.02 维护者指出）：既然展开时
# 读不到，**实际展开出来的永远是默认值那一支（`./…`）**。它能指对，是因为
# Claude Code 起 stdio server 时的工作目录就是项目根——**这是实测观测到的行为，
# 不是文档承诺的，换版本要重验**。哪天 CC 改了 spawn 的工作目录，这条会静默断掉，
# 而按字面读注释的人会往「变量是不是没注入」那边查，查错方向。
#
# 由此还有第二个前提：`relative_to` 算的是**相对产出目录**，而展开出来的 `.` 是
# **进程工作目录（项目根）**——两者只有在「产出目录就是用户打开 Claude Code 的那个
# 目录」时才重合。记忆包放在 `myapp/memory-bundle/`、人在 `myapp/` 开 CC 的，
# 配置会指到 `myapp/src/mcp_server.py`，指错了，而且是同一种不报错的失败。
# selftest 9f 的断言④管的是「server 在产出目录外」，管不到这一种，所以写在这里。
PORTABLE_PREFIX = "${CLAUDE_PROJECT_DIR:-.}/"

# 时区那一单（任务卡「写回时区与跨日归窗」2026.08.04 真机事故）加的两句尾巴。
#
# ⚠ **两条规矩，别混成一条**（2026.08.05 维护者拍板后的形态）：
#   ① **没人给时区时，落进 args 的是默认值 `Asia/Shanghai`**，不是留空；
#   ② **探测到的宿主时区永远不进 args**，只当参考写进说明。
# ②那条守的是：探测探的是**跑初始化那台机器**，而这个值要的是**记忆所有者本人**——
# 事故现场那台 VPS 探出来正好是 `Etc/UTC`，填进去就等于给那个缺陷发一张
# `✓ 时区：Etc/UTC` 的合格证。**它长得跟判据一模一样，而且不会报错。**
# ⚠ 默认值同样可能是错的（使用者不在东八区时），所以说明里必须写明"这是默认值、
# 不对就改这一行"，`--doctor` 那一格也仍然报 ⚠——**默认值不等于验过了**。
# 解释器那一行的说明，按产出形态二选一（2026.08.05 外部实测第 2 条：样板里写死的
# `"command": "python"` 在只装了 python3 的机器上照抄必挂，**而症状是 spawn 静默
# 失败**——客户端只说连不上，没有任何一行指回这个字段）。
CONFIG_NOTE_PYTHON_ABS = (
    "command 那一行填的是**跑这次初始化的那个解释器的绝对路径**，不是字面的 python"
    "——很多机器上根本没有叫 `python` 的命令（只有 `python3`），照抄一个不存在的命令，"
    "客户端只会说“连不上”，不会告诉你是哪一行的错。"
    "顺带解决第二件事：`local`／`cloud` 两条检索路线要 pip 装东西，"
    "装在哪个解释器上，这一行就得指哪个。换机器要连这一行一起改。")
CONFIG_NOTE_PYTHON_NAME = (
    "command 那一行是**按跑初始化这台机器探出来的命令名**填的（这台机器上它能跑）。"
    "这份配置本身可以跟着仓库搬，但**命令名搬不动**——换台机器要先确认那边有同名命令，"
    "而且 `local`／`cloud` 两条路线 pip 装的东西也要装在同名解释器上。")

CONFIG_NOTE_TZ = (
    "时区那一行（--timezone）决定“今天是几号”按哪个时区算——写回落进哪个窗口文件、"
    "记录标题、检索标签都跟着它。换时区/换机器记得改。")
CONFIG_NOTE_TZ_DEFAULT = (
    "⚠ 时区那一行（--timezone）填的是**默认值 " + DEFAULT_TIMEZONE + "（东八区）**，"
    "因为没人告诉过我们你在哪个时区——**我们不猜，也不拿这台机器的时区顶替**"
    "（探到的是跑初始化这台机器的时区，不是你的）。**你不在东八区的话，这一行现在"
    "就是错的**：凌晨那几个小时写下的记忆会被记成前一天，文件名、记录标题、检索标签"
    "一起错，而且不报任何错。改法：把上面那行的 " + DEFAULT_TIMEZONE + " 换成你的 "
    "IANA 时区（例如 Europe/Berlin）。跑 --doctor 时这一格会报 ⚠ 提醒你确认，"
    "显式配上之后就变 ✓。")


def python_command(portable=False):
    """MCP 配置里 `command` 那一行填什么（2026.08.05 外部实测第 2 条）。

    在此之前这里写死字面量 `"python"`。**很多机器上没有这个命令**（Debian 系默认
    只有 `python3`），照抄的人拿到的症状是**客户端 spawn 静默失败**——它只说连不上，
    没有任何一行指回这个字段，自查时几乎不会怀疑到解释器名上。

    两档，跟着路径那两档走（同一个 `rels is not None`）：

    · **绝对路径档**取 `sys.executable`——就是跑这次初始化的解释器。反正整份配置
      已经写死了这台机器的绝对路径，再让命令名去赌 PATH 没有意义；而且它顺手解决
      第二件事：`local`／`cloud` 两条路线要 pip 装依赖，**装在哪个解释器上，
      服务端就得用哪个**，写 `python` 起到另一个解释器上就是一句 ImportError。
    · **可搬运档**（Claude Code 占位符那一档）不能写绝对路径，否则整档的意义没了。
      这里退回命令名，但**是探过的**，而且**头一个候选是这次真的跑起来的那个名字**
      （`Path(sys.executable).stem`）——venv 里它是 `python`、系统装的多半是
      `python3`，两种形态都不用猜。探不到才依次退 `python3`、`python`。

    ⚠ **这跟时区那条「我们不猜、也不拿这台机器的顶替」不冲突，别照着改**：时区问的是
    **人在哪儿**，这台机器答不了；解释器问的是**这台机器上什么命令能跑**，
    正是这台机器唯一能答准的东西。两条的分别在于问的是谁，不在于探不探。
    ⚠ 探不出来（PATH 里两个都没有，罕见）就落 `python3`：写一个不存在的命令是必然挂，
    落哪个都一样挂，选覆盖面大的那个，并且**这一档也照样把说明写进配置里**。"""
    import shutil

    if not portable:
        return str(Path(sys.executable).resolve()).replace("\\", "/")
    for name in (Path(sys.executable).stem, "python3", "python"):
        if name and shutil.which(name):
            return name
    return "python3"


def mcp_config_snippet(server_path, corpus_dir, threads_path, route=None,
                       client=None, portable_root=None, timezone=None,
                       index_dir=None):
    """给用户直接粘贴的 MCP 配置。路径统一用正斜杠——JSON 里反斜杠要转义，
    而正斜杠在 Windows 上一样认，少一个踩坑点。

    **默认是绝对路径**：裸的相对路径 `mcp_server.py` 只在"客户端恰好
    从 src/ 起进程"时能跑，换个工作目录就是一句 file not found。宿主用户还能从
    《快速上手》那条 `claude mcp add` 示范里抄到绝对路径写法，自建前端作者没有
    那条示范——配置里给什么他就用什么。这条理由没作废，所以绝对路径仍是默认。

    **但绝对路径跟着机器走**（2026.08.02，真实用户在云端容器里当场接不上）：
    同一份配置换台机器/换个容器就断，而且 MCP server 起不来**不报到用户脸上**
    ——会话照开、模型照回话，只是没有那五个工具。所以 Claude Code 档在
    「全部路径都在产出目录下」时（当前出货是 server、语料、threads、独立索引四条；
    §3b 那种整套进仓库的形态）改产
    `${CLAUDE_PROJECT_DIR:-.}/…` 可搬运写法。三个硬边界：
      - **只有 Claude Code 认这个占位符**（官方文档核过），Codex/闭源前端/自建
        前端给了就是 file not found——所以按 client 分档，不猜别家有等价物；
      - 桌面形态 server 在克隆仓库里、不在产出目录下，**几何上就相对化不了**，
        自然落回绝对路径档——不存在"半可搬运"的中间态；
      - **前提是产出目录就是用户打开 Claude Code 的那个目录**（§3b 整套进仓库的
        形态）。展开出来的是默认值那一支 `./…`，基准是**进程工作目录**，而
        `relative_to` 的基准是**产出目录**，两者只在这个前提下重合。记忆包放在
        `myapp/memory-bundle/`、人在 `myapp/` 开 CC 的就指错——机制与为什么断言
        接不住这一种，见 `PORTABLE_PREFIX` 上面那段。
    这里的 `client` 形参是**加回来的**：当初删它的理由（"三档客户端的 MCP 配置
    本来就一模一样"）在可搬运写法出现后不再成立。

    `route` 是用户在 `--step route` 里选的检索路线（2026.08.02）：它**确实**改变
    启动参数（零依赖不加、本地加 `--embed`、云端再加 `--embed-provider cloud`）。
    **key 永远不进这个文件**——云端档只在这里写"走云端"，endpoint/模型/key 全从
    环境变量读；这份配置会跟着产出目录走，用户会随手把它贴给别人。"""
    paths = [Path(server_path), Path(corpus_dir), Path(threads_path)]
    if index_dir is not None:
        paths.append(Path(index_dir))
    rels = None
    if client == "claude-code" and portable_root:
        root = Path(portable_root).resolve()
        try:
            rels = [p.resolve().relative_to(root) for p in paths]
        except ValueError:            # 有路径不在产出目录下（桌面形态）→ 绝对路径档
            rels = None
    if rels is not None:
        vals = [PORTABLE_PREFIX + str(r).replace("\\", "/") for r in rels]
        note = CONFIG_NOTE + CONFIG_NOTE_PORTABLE
    else:
        # ⚠ 全部都要 `resolve()`，不是只有 server 那个（走查台账 08-03 第六条）：
        # 原来只有 `paths[0]` 解了，`corpus_dir` 与 `threads_path` 是裸 `str()`
        # ——而同一行拼进去的 `CONFIG_NOTE_MACHINE_BOUND` 写着"写死了这台机器的
        # 绝对路径"，**三分之一是真的**。用户拿相对路径起 `--out` 时，配置里落下的
        # 就是相对值，而客户端起 server 的工作目录跟他当时 cd 的地方无关
        # ——又是一句不报到脸上的 file not found。
        # `--doctor` 那边本来就有"配置里写的是相对值时会解析开"的处理，出货端对齐。
        vals = [str(path.resolve()).replace("\\", "/") for path in paths]
        note = CONFIG_NOTE + CONFIG_NOTE_MACHINE_BOUND
    # 时区：人给了就用人给的，没给就落默认值——**但探测结果任何时候都不进 args**
    # （两条规矩的分别见 CONFIG_NOTE_TZ_DEFAULT 上面那段注释）
    command = python_command(portable=rels is not None)
    note += CONFIG_NOTE_PYTHON_NAME if rels is not None else CONFIG_NOTE_PYTHON_ABS
    tz = (timezone or "").strip()
    note += CONFIG_NOTE_TZ if tz else CONFIG_NOTE_TZ_DEFAULT
    if not tz:
        here_tz = detect_local_timezone()
        if here_tz and here_tz != DEFAULT_TIMEZONE:
            note += (f"（⚠ 顺带一提：跑初始化这台机器的本地时区是 {here_tz}，"
                     f"跟上面填的默认值不一样。这**不代表**该填 {here_tz}——"
                     "要填的是你自己所在的时区，不是这台机器的。仅供你核对时参考。）")
        tz = DEFAULT_TIMEZONE
    args = [vals[0], "--corpus", vals[1]]
    if index_dir is not None:
        args += ["--index-dir", vals[3]]
    args += ["--threads", vals[2], "--timezone", tz]
    cfg = {CONFIG_NOTE_KEY: note,
           "mcpServers": {"memory": {
               "command": command,
               "args": args + route_args(route or ROUTE_DEFAULT),
           }}}
    return json.dumps(cfg, ensure_ascii=False, indent=2)


INDEX_README = """这个目录是记忆库的索引层（规格 §5）：高密度摘要，专门喂检索。

新增记忆由当前宿主模型在调用 `latent_append` 时交付正文、`current_state` 和
`indexEvidence[{type,quote}]`。`type` 可为 event／feeling／reason／state／context；quote
只摘连续原文。服务端容忍空白、全半角与引号标点形态差异，但最终写进本目录的一定是重新定位到的
原文，不直接保存模型改写。证据不合格时正文仍写入 timeline，并返回 recordId 与
`indexStatus=pending`；之后只传 recordId＋indexEvidence 补索引，不重复正文。

下面两种手工写法继续兼容，用于存量材料或跨窗口主题补充：把叙事层整段贴给
你自己的模型，让它写高密度摘要——人名、
原话、当时在处理什么，都留着，不要润色成读后感。有两种补法，各补各的缺口：

【一】按窗口摘要
把主语料 timeline/ 里的某个窗口（路径见 mcp-config.json 的 --corpus）整段贴给模型，
写一段 200 字左右的摘要，存成跟那个
窗口同名的文件，例如 window_07_2026-07-20.md 。索引层和叙事层同名同窗口号，
检索层会自动把它们认成同一次会话的两种写法，日期也跟着继承过来。

【二】按主题线摘要
一件事往往横跨很多个窗口：一个约定从提起、到反复、到兑现，可能散在十个窗口里。
把同一条线涉及的几个窗口一起贴给模型，让它写这条线**从头到现在**的一段摘要，
收尾落一句这件事现在的状态。按窗口切的摘要看不见这种跨度——这是主题线摘要独有
的价值，也是检索最难自己拼出来的东西。

文件名必须是 topic_<线名>_<YYYY-MM-DD>.md ，例如 topic_望远镜_2026-07-31.md ，
日期填这条线最后一次发生的日期。

**日期不能省，这是硬要求，不是格式洁癖。** 主题线跨窗口、没有单一窗口号，借不到
叙事层的日期；文件名再不带日期，时间戳就只能退到文件的修改时间（mtime）。而
mtime 在这里不是"不太准"，是**全错且整齐地错**——重新下载或复制一遍目录，会把
所有文件的 mtime 刷成同一时刻，一批主题线摘要于是拿到同一个假时间戳。换新窗口
时的开场召回正是按时间新鲜度排序的，而主题线摘要恰恰是最该在开场被带回来的那种
内容：等于让最有价值的记忆被一个垃圾时间戳排序，而且它在索引层、本来就更容易被
检索排到前面，错得更容易被看见。

注意：这个说明文件故意不是 .md，免得它自己被当成语料读进检索库。
"""


_WINDOW_NO_RE = re.compile(r"^window_(\d+)")

CORPUS_OVERWRITE_CODE = "CORPUS_TIMELINE_OVERWRITE"


def plan_corpus_files(entries, gap_seconds=1800, start_window=1):
    """条目 → {窗口号: (文件名, 正文)}，**一个字都不落盘**。

    单独拆出来是为了让写入侧的护栏能在**任何写盘之前**判完（同引导句超长闸那条
    理由：出到一半才拒绝，目录里留下半套货，而错误信息说的是「不出货」）。"""
    sessions, cur, last_ts = [], [], None
    for e in entries:
        ts = getattr(e, "timestamp", None)
        if cur and ts is not None and last_ts is not None and ts - last_ts > gap_seconds:
            sessions.append(cur)
            cur = []
        cur.append(e)
        if ts is not None:
            last_ts = ts
    if cur:
        sessions.append(cur)

    planned = {}
    for n, sess in enumerate(sessions, start_window):
        stamps = [e.timestamp for e in sess if getattr(e, "timestamp", None)]
        day = datetime.fromtimestamp(min(stamps)).strftime("%Y-%m-%d") if stamps else None
        name = f"window_{n:02d}_{day}.md" if day else f"window_{n:02d}.md"
        head = f"# 第{n}个窗口" + (f" · {day}" if day else "")
        body = [f"{e.speaker}：{e.text}" if getattr(e, "speaker", "") else e.text
                for e in sess]
        planned[n] = (name, head + "\n\n" + "\n".join(body) + "\n")
    return planned


def corpus_overwrite_conflicts(timeline, planned):
    """本次出货会改掉磁盘上哪些窗口 → 排好序的窗口号列表（空表＝写下去不损失任何东西）。

    **判据落在目录层，不逐文件各判**（任务卡写死的）：逐文件各自决定要不要备份，
    会出现「一部分窗口备份了、一部分没有」的半覆盖，那比全覆盖更难排查。所以这里
    只回答一个是非题——整个 timeline 会不会变样——由调用方对**整个目录**做处置。

    **判据取「内容不同」，不是「有没有 .bak」**（与人格文件那侧同口径）：第一次出货
    时目录是空的，那不是覆盖；同一份语料原样重跑一遍，写出来跟磁盘上一模一样，
    也不是覆盖——那两种都不该拦。

    同一个窗口号换了文件名（日期变了）也算改动：老文件不会被删，于是同号两份并排
    躺着，检索层读到的是哪一份取决于遍历顺序——这跟内容被覆盖一样是损坏，只是更
    难看出来。"""
    timeline = Path(timeline)
    if not timeline.is_dir():
        return []
    existing = {}
    for p in sorted(timeline.glob("*.md")):
        m = _WINDOW_NO_RE.match(p.name)
        if m:
            existing.setdefault(int(m.group(1)), {})[p.name] = p.read_text(encoding="utf-8")
    conflicts = []
    for n, (name, text) in planned.items():
        current = existing.get(n)
        if current and current != {name: text}:
            conflicts.append(n)
    return sorted(conflicts)


def backup_corpus_dir(timeline):
    """把整个 timeline 目录备份一份，返回备份目录路径。**只加不减**：不删原目录、
    不动原文件，也**不覆盖已经存在的备份**。

    `.bak` 的命名形状跟人格文件那侧对齐（`<名>.bak`），但多一条：备份目录被占了就
    往后顺号（`timeline.bak2`、`timeline.bak3`……），**不像人格文件那侧那样直接盖掉
    旧的 .bak**。语料这侧不能盖，因为每份备份存的是那一刻的全量，而两次出货之间
    用户可能用 latent_append 写进过新窗口：拿新备份盖旧备份，会把只存在于旧备份里
    的那部分记忆抹掉——那正是这条护栏要挡的事，护栏自己不能犯。
    人格文件能重新生成，记忆不能。"""
    timeline = Path(timeline)
    backup = timeline.with_name(timeline.name + ".bak")
    n = 2
    while backup.exists():
        backup = timeline.with_name(f"{timeline.name}.bak{n}")
        n += 1
    shutil.copytree(timeline, backup)
    return backup


def guard_corpus_overwrite(timeline, planned, mode="block"):
    """写之前的护栏：按 mode 决定拦下来、先备份、还是照写。返回备份目录或 None。

    **默认 blocking**（2026.08.04）。原先是同名文件直接 `write_text` 覆盖——不备份、
    不提示、不报错，而 `latent_append` 写回用的是同一套命名形状，所以出货第二遍到
    同一个 memory/，会盖掉用户后来写进去的窗口。人格文件那侧 08.01 就为「覆盖自己
    那份」加了 .bak 护栏，那段注释自己写着**「沉默地覆盖才是最坏的形态」**——同一个
    最坏形态，语料侧一直一点护栏都没有。而**语料是用户唯一不可再生的东西**：人格
    文件能重新生成，记忆不能。

    **拦截必须有出口**（姊妹卡《初始化输入侧静默塌节与无出口拦截》立的规矩）——
    没出口的拦截会把用户推去手工搬文件，那是比覆盖更糟的一条路。出口两条：
      - `backup`：整目录备份完再写（`--backup-corpus`）；
      - `accept`：我知道会覆盖，就这么办（`--accept-corpus-overwrite`）。
    **刻意不做「换目录」那一支**：用户本来就能换 `--out`，为它新开一个开关等于凭空
    多一套目录语义。"""
    conflicts = corpus_overwrite_conflicts(timeline, planned)
    if not conflicts:
        return None
    if mode == "accept":
        return None
    if mode == "backup":
        return backup_corpus_dir(timeline)
    windows = "、".join(f"window_{n:02d}" for n in conflicts[:5]) + \
              ("…" if len(conflicts) > 5 else "")
    raise PermissionError(
        f"{CORPUS_OVERWRITE_CODE}：目标记忆库 {Path(timeline)} 里已有 "
        f"{len(conflicts)} 个窗口会被这次出货改写（{windows}）。"
        "语料是不可再生的——你后来用 latent_append 写进去的窗口就长在这些文件里，"
        "盖掉就没了。两条出口，挑一条重跑本步："
        "加 --backup-corpus（先把整个 timeline 备份成 timeline.bak 再写，只加不减），"
        "或加 --accept-corpus-overwrite（我知道会覆盖，就这么办）。"
        "想留着这份记忆库不动，就换一个 --out 目录。")


def write_corpus(memory_dir, entries, gap_seconds=1800, start_window=1,
                 corpus_overwrite="block"):
    """导入的中间格式条目（memory_import.MemoryEntry）→ timeline/ 下的 md。
    返回 (写下的文件列表, 备份目录或 None)。

    一个会话一个文件，按时间间隔断开（同 entries_to_index 的自然边界判据）。

    **文件名带日期**，因为文件名日期是 parse_chunk_timestamp 的最高优先级来源——
    真实语料那次 95.6% 的块落到 mtime 兜底，根子就是文件名不带日期。我们自己生成
    的语料没有理由重蹈；日期来自条目时间戳，是有据可依的，不是猜的。整段没有时间
    戳的会话就不写日期，也不编一个。

    **重跑不许静默覆盖已有窗口**：目标 timeline 里已有的窗口会被这次改样时就停下来，
    出口见 guard_corpus_overwrite。护栏在**任何写盘之前**判完。

    index/ 建出来但留空，见 INDEX_README。"""
    mem = Path(memory_dir)
    timeline = mem / "timeline"
    planned = plan_corpus_files(entries, gap_seconds, start_window)
    # 护栏要在 mkdir / 写 README 之前过：拦下来的那次不该在用户目录里留任何痕迹
    backup = guard_corpus_overwrite(timeline, planned, corpus_overwrite)
    timeline.mkdir(parents=True, exist_ok=True)
    (mem / "index").mkdir(parents=True, exist_ok=True)
    (mem / "index" / "README.txt").write_text(INDEX_README, encoding="utf-8")

    written = []
    for _, (name, text) in sorted(planned.items()):
        (timeline / name).write_text(text, encoding="utf-8")
        written.append(timeline / name)
    return written, backup


def contract_source(base=None):
    """注入契约文档在包里的位置：src/ 的同级 docs/ 下。随包范围里 src 与 docs
    的相对位置是固定的，所以这条相对路径在开发目录和用户拿到的包里都成立。"""
    root = Path(base) if base else Path(__file__).resolve().parent.parent
    return root / "docs" / CONTRACT_DOC


DEFAULT_CLIENT = "claude-code"


def resolve_client(cli_client, state_client):
    """定客户端档：**命令行显式给了就以它为准**，返回 (客户端, 要说的话或 None)。

    修的是一个真会咬人的洞（验收打回，2026.08.02）：`--client` 是在 questionnaire
    步写进 init_state.json 的，而 ship 步原先取 `state.get("client", args.client)`
    ——状态里已有值时，命令行显式传的 `--client generic` **被静默吃掉**。
    《快速上手》4b 教的正是"前面照旧走、第 4 步换成 --client generic"，照着做的
    自建前端作者会拿到一份 CLAUDE.md、没有契约副本、外加一句"从产出目录起会话"，
    **而且没有任何报错**——正好是契约文档里写的那句"这套机制最容易死的方式"。

    为什么选"命令行赢"而不是"冲突就报错"：用户在最后一步显式敲出来的那个值，
    是他此刻的意图，没有理由让一个几步之前存下的默认值压过它。但**不许静默**——
    换档要说出来，并把新值写回状态，免得下次续跑又飘回去。"""
    if cli_client and state_client and cli_client != state_client:
        return cli_client, (f"【客户端档已切换】{state_client} → {cli_client}"
                            f"（命令行显式指定，以它为准；已写回 init_state.json）")
    return cli_client or state_client or DEFAULT_CLIENT, None


# ---------- 检索路线：流程里的显式选择点（2026.08.02 云端 embedding 任务卡） ----------
#
# **为什么这件事必须在流程里问，而不是像以前那样躺在《快速上手》§5 当"可选升级"**：
# 三条路线里有一条（云端）会把**查询和被检索的内容发到第三方服务器**。语料去哪儿
# 不是性能取舍，是信任问题，而信任问题不可逆——发出去了就收不回来。
# 《给AI的引导指南》§三·五原先写着"别在第一次设置时就抛给 TA 选，那会把一个本来
# 很轻的流程变重"，**那条现在收窄适用范围，不作废**：它权衡的是纯性能取舍
# （快一点 vs 慢一点、装不装依赖），为这个打断新用户确实不值；但一旦选项里出现
# 一条会把私人记忆发出去的路，"默认替他选"就不成立了。
#
# 默认档仍是零依赖（维护者已定）——**默认值不等于不用问**：这里问的是"你知道有
# 三条路、各自的代价是什么，然后你选了一次"，选回默认也算选过。
ROUTE_DEFAULT = "zero-dep"

RETRIEVAL_ROUTES = {
    "zero-dep": {
        "名称": "零依赖（默认）",
        "要装什么": "什么都不用装",
        "语料去向": "**不出本机**",
        "代价": "换个说法问同一件事，我们自己的回归集上大约四分之三能答对；"
                "用原话直接问跟另外两条一样好",
        "命中门槛": "不适用（bigram 余弦是另一套标度）",
    },
    "local": {
        "名称": "本地模型",
        "要装什么": "pip install fastembed，首次运行自动下载约 100MB 中文模型"
                    "（bge-small-zh-v1.5，本地 CPU 跑）",
        "语料去向": "**不出本机**（模型在你自己的机器上跑）",
        "代价": "建库和检索都变慢，低配机器（2C2G 这类）感知明显；"
                "小内存机器可能根本装不下",
        "命中门槛": "已标定（0.45，对应 bge-small-zh-v1.5）",
    },
    "cloud": {
        "名称": "云端服务",
        "要装什么": "不装模型，但要一个第三方 embedding 服务的 API key（自己去服务商申请）；"
                    "endpoint 与模型名走 MEMORY_EMBED_* 环境变量",
        "语料去向": "**查询和被检索的内容都会发到你选的那家服务商**——这条是这次选择"
                    "真正的分水岭，另外两条都不出本机",
        "代价": "每次查询多一个网络往返；块向量只在建库时算一次、缓存在本机，"
                "所以日常成本是每查一次一个往返，不是每次都把全库重算",
        "命中门槛": "**换模型必须重新标定**。没标定过的模型，向量路只参与排序、"
                    "不单独放行候选（我们不照抄别的模型的数字）",
        "key 从哪来": "只从环境变量读，我们不写进配置文件、不进产出目录、不落 state",
    },
}

ROUTE_PREAMBLE = (
    "【检索路线：请把三条都念给 TA，由 TA 自己选】\n"
    "这一步不设静默默认——三条路里有一条会把 TA 的私人记忆发到第三方，"
    "这种事不能靠“他以后自己去读文档”。选回默认也算选过。")


def route_options():
    """三条路线的机器可读描述（给 AI 驱动那条路用，同问卷 --json 的先例）。"""
    return [dict(key=k, default=(k == ROUTE_DEFAULT), **v)
            for k, v in RETRIEVAL_ROUTES.items()]


def format_routes():
    """人类可读版。"""
    lines = [ROUTE_PREAMBLE, ""]
    for opt in route_options():
        head = f"[{opt['key']}] {opt['名称']}" + ("（不选的话就是它）" if opt["default"] else "")
        lines.append(head)
        for field in ("要装什么", "语料去向", "代价", "命中门槛", "key 从哪来"):
            if opt.get(field):
                lines.append(f"    {field}：{opt[field]}")
        lines.append("")
    lines.append("选好后：--step route --route <上面方括号里的 key>")
    return "\n".join(lines)


def route_args(route):
    """路线 → mcp_server 的启动参数。零依赖档不加任何参数（默认就是它）。

    **命令行里永远不会出现 key**：云端档只指定"走云端"，endpoint/模型/key 全在
    环境变量里。MCP 配置文件跟着产出目录走，用户会随手分享它。"""
    if route == "local":
        return ["--embed"]
    if route == "cloud":
        return ["--embed", "--embed-provider", "cloud"]
    return []


def route_note(route):
    """出货时把这条路线的代价再说一遍——选过一次，落地时还要能对得上。"""
    if route == "cloud":
        return ("【检索路线：云端】起服务前先把这几个环境变量设好："
                "MEMORY_EMBED_ENDPOINT（服务商的 /v1/embeddings 地址）、"
                "MEMORY_EMBED_MODEL（模型名）、MEMORY_EMBED_API_KEY（你的 key）。"
                "**key 只从环境变量读，我们不写进 mcp-config.json、不存进产出目录。**"
                "记住这条路的代价：查询和被检索的内容都会发到那家服务商；"
                "换模型的话命中门槛要重新标定（没标定前向量路只排序、不放行候选）。")
    if route == "local":
        return ("【检索路线：本地模型】起服务前先 pip install fastembed，"
                "首次运行会下载约 100MB 模型。语料不出本机。")
    return "【检索路线：零依赖】什么都不用装，语料不出本机。"


def ship_note(client):
    """出货收尾提示。**按客户端分档，不能共用一句**——claude-code/codex 那套
    "从产出目录起会话"的话术建立在"宿主会自动读人格文件"上，而 generic 档根本
    没有宿主：照抄过去等于告诉自建前端作者"你什么都不用做"，那正是这套机制
    最容易死的方式（人格没进请求，模型照样答得煞有介事）。"""
    if client == "generic":
        return (f"【下一步】先试轻量方案：优先把产出目录里的 {GUIDANCE_DOC}（一行指针）"
                "原样贴进你的 App 的 system prompt／自定义指令字段。验证方法：开新会话问一件"
                "只有人格文件里才有的事，不经提示答上来才算生效。"
                "验证不通过或 App 读不到本地文件时，再走重方案：自建前端没有宿主替你注入"
                "人格文件——**注入是你前端的责任**："
                f"照产出目录里的《{CONTRACT_DOC}》把 persona.md 拼进你自己的请求"
                "（逐字完整／每轮都在／每会话从磁盘重读／整块连续／易变内容后置），"
                "再按契约末尾那步验证接通。")
    if client == "grok":
        return ("【下一步】Grok 人格已写入 .grok/agents/companion.md。"
                "在 Grok Build 的 /agents 中选择 companion，或从产出目录运行 "
                "grok --agent companion，再把 mcp-config.json 中的配置接入客户端。")
    return ("【下一步】把 mcp-config.json 里的配置加进你的客户端，然后"
            "**从产出目录起会话**——人格文件在那儿才会被宿主读到。")


_PICK_NOISE_RE = re.compile(r"[\s“”\"'‘’「」『』，,。.；;：:！!？?~—\-()（）]+")


def _normalize_for_lookup(text):
    """比对用的归一形态：抹掉空白与标点，只留下字。

    引号、逗号、破折号在摘录里最容易被顺手改（全角改半角、加一对引号），
    留着它们比对会把真摘录判成自造。"""
    return _PICK_NOISE_RE.sub("", text or "")


def unsourced_picks(questions, answers, corpus_dir):
    """pick 题的答案里，**哪些在语料里找不到出处** → [(qid, 那段文本), ...]。

    来历（2026.08.02 真机）：pick 题的设计意图是"从语料里挑真实的原话"，
    而执行者**自己创作了一句**放进候选、并被用户采用了——落点还是 `closing_pick`
    （最终约定），人格文件里象征意味最重的那一格。流程分不清"语料里的真话"和
    "模型写的漂亮话"，而这正是整个项目最该防的那件事。

    **只标注、不硬拒**（同 absent 防线那条决定）：用户完全可以自己给一句，
    那是他的话、语料里当然没有；我们没资格替他否掉。所以这里只负责把
    "这句在语料里查无出处"摆到台面上，由人去判断是谁写的。

    比对按分句做：称呼这类答案是多段拼起来的，整段查不到不代表每一段都是编的；
    只要有一段能在语料里找到，就不报——**宁可漏报，不可误伤真摘录**。"""
    corpus = Path(corpus_dir) if corpus_dir else None
    if not corpus or not corpus.exists():
        return []
    if corpus.is_dir():
        blob = "".join(p.read_text(encoding="utf-8", errors="ignore")
                       for p in corpus.rglob("*") if p.is_file())
    else:
        blob = corpus.read_text(encoding="utf-8", errors="ignore")
    hay = _normalize_for_lookup(blob)
    qmap = {q.qid: q for q in questions}
    out = []
    for qid, ans in (answers or {}).items():
        q = qmap.get(qid)
        if q is None or q.kind != "pick":
            continue
        text = ans.get("pick", "") if isinstance(ans, dict) else str(ans or "")
        if not text.strip():
            continue
        parts = [s for s in _CLAUSE_SEP_RE.split(text) if s.strip()] or [text]
        if not any(_normalize_for_lookup(s) and _normalize_for_lookup(s) in hay
                   for s in parts):
            out.append((qid, text))
    return out


PICK_SOURCE_NOTE = ("【查无出处】上面这些 pick 题的答案在语料里找不到原话。"
                    "pick 题只许挑、不许写——**如果这句是对方自己给的，那没问题**；"
                    "如果是你替 TA 想的一句好听的，删掉，宁可这一格空着。")

_FILE_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def corpus_coverage(corpus_dir, entries=None):
    """记忆库覆盖的日期区间 → ("YYYY-MM-DD", "YYYY-MM-DD")，问不出来就返回 None。

    两个来源，都不猜：
      - entries（这次导入的条目）：用真实时间戳，最准；
      - 已有语料目录：用**文件名里的日期**——那是 parse_chunk_timestamp 的最高优先级
        来源，也是 write_corpus 自己写文件时遵守的命名规则。**刻意不退 mtime**：
        mtime 在这里不是"不太准"是全错且整齐地错（复制一遍目录会把所有文件刷成
        同一时刻），拿它去告诉模型"你的记忆止于哪天"，等于给它一个假边界——
        比不给更糟。问不出来就不写这条字段，空着是诚实的。"""
    days = []
    for e in entries or []:
        ts = getattr(e, "timestamp", None)
        if ts:
            days.append(datetime.fromtimestamp(ts).strftime("%Y-%m-%d"))
    if not days and corpus_dir:
        for p in Path(corpus_dir).rglob("*.md"):
            m = _FILE_DATE_RE.search(p.name)
            if m:
                days.append(m.group(0))
    return (min(days), max(days)) if days else None


def memory_report_lines(paths, corpus_dir):
    """出货报告里「记忆库」那一段（2026.08.05 外部实测第 1 条，按杀伤力排第一）。

    在此之前两条出货路径都只打一行 `记忆库：<out>/memory`。六步走的叙事语料其实在
    用户 `--corpus` 指的目录；现在 `<out>/memory` 只承载独立 `index/`，不能把整个目录
    再冒充叙事记忆库，也不能说它全空。报告必须把两处分别指明。
    报告说的和事实差着一个目录，而**这是新手看到的第一屏**。

    ⚠ 《快速上手》后段确实解释过「七步走不写 timeline」，但那是另一屏——
    **报告自己说错了话，不能靠文档去追**。
    ⚠ 判据是「这次到底往哪儿落了盘」，不是「哪条命令」：`--import` 那条路真写了窗口
    文件（`corpus_files` 非空），那行就照旧报 `<out>/memory`，它此刻是真的。"""
    memory_dir = paths["memory_dir"]
    index_dir = paths["index_dir"]
    written = paths.get("corpus_files") or []
    if written:
        lines = [f"  记忆库：{memory_dir}（落盘 {len(written)} 个窗口文件）"]
    elif corpus_dir and Path(corpus_dir).resolve() != Path(memory_dir).resolve():
        lines = [
            f"  记忆库：{Path(corpus_dir).resolve()}"
            f"（就是你 --corpus 指的那个目录；检索读的是它，"
            f"mcp-config.json 里 --corpus 也指着它）",
        ]
    else:
        lines = [f"  记忆库：{memory_dir}"
                 f"（**没有叙事语料**——这次既没给 --corpus，也没给 --import；"
                 f"补语料的两条路见《快速上手》§2）"]
    lines.append(
        f"  索引摘要目录：{Path(index_dir).resolve()}"
        "（README.txt 只是写法说明；放入摘要 .md 后才有索引层）")
    return lines


def write_bundle(out_dir, persona, client="claude-code", corpus_dir=None,
                 server_path=None, confirmed=False, entries=None,
                 contract_base=None, previous_persona=None, route=None,
                 validation_mode="legacy_v1", rendered_override=None,
                 add_coverage=True, corpus_overwrite="block", timezone=None):
    """产出四件套（generic 档多一份注入契约副本）。**confirmed=False 时拒绝写盘**——写用户磁盘要过确认关卡
    （规格 §7：人格文件任何改动必须用户确认）。

    **只写我们自己的产出，只动我们上一次真的出过的那一个文件**（2026.08.02 三轮
    验收后改准；原话是"不动同目录其它 md"，退役逻辑加进来之后那句已经不成立）。
    `previous_persona` 是**上一次出货写下的人格相对路径**，由调用方（CLI 从
    init_state.json）传进来；只有它、且它跟这次的档不同名时才退役。不传就一个都不碰。

    entries 给了就把语料落成记忆库（write_corpus）；没给就只把目录建出来——
    用户可能是把已有语料目录用 corpus_dir 直接指过来的，那份不该被我们重写。

    `corpus_overwrite`（block／backup／accept）透传给写入侧护栏：目标 timeline 里
    已有的窗口会被这次改样时默认停下来，不静默覆盖。见 guard_corpus_overwrite。"""
    if not confirmed:
        raise PermissionError("未确认，不写盘——人格文件写入必须过用户确认关卡")
    # 第二道闸：还有没走过确认的草稿就不许出货。
    # render() 只输出 confirmed 的内容，所以未决草稿出货时会**无声蒸发**——文件长得
    # 很正常，只是少了几节，用户根本看不出发生过什么。跟"读不懂的行静默丢"是同一
    # 类病，都得在出口处堵住，不能靠下游发现。
    pending = pending_confirmations(persona)
    if pending:
        raise PermissionError(
            f"还有 {len(pending)} 条草稿没走确认，不出货（否则它们会静默消失）："
            + "、".join(p.label for p in pending[:5])
            + ("…" if len(pending) > 5 else ""))
    missing = persona.validate(mode=validation_mode)
    if missing:
        raise ValueError("人格文件还不完整：" + "；".join(missing))
    if client not in CLIENT_FILENAMES:
        raise ValueError(f"未知客户端 {client}，可选：{'/'.join(CLIENT_FILENAMES)}")
    # `--corpus` 给了、存在、却不是目录时，在**任何写盘之前**拦下（同 guidance 超长闸、
    # 语料覆盖护栏的位置）：否则 memory_report_lines 会把这个单文件路径当成「就是你
    # --corpus 指的那个目录」照抄进出货报告，mcp_config_snippet 也拿它落 --corpus，
    # 而 doctor 拿同一条路径立刻报「不是目录」——同一份代码、同一条路径，ship 说是
    # 记忆库、doctor 说不能当 --corpus（2026.08.22 外部部署报告 2 实测）。消息与
    # doctor 的判据（mcp_server.py：`root.is_dir()`）对齐。显式 --import 时调用侧把
    # corpus_dir 收敛为 None：导入内容写进并配置为 <out>/memory，不让 mcp-config 又指回
    # 另一套外部 --corpus；因此不受这条目录校验影响。不存在的路径是另一类错，不在这里。
    if corpus_dir is not None:
        _corpus = Path(corpus_dir)
        if _corpus.exists() and not _corpus.is_dir():
            raise ValueError(
                f"{_corpus.resolve()} 不是目录。--corpus 要指向记忆库那一层目录，"
                "不是单个文件。")
    out = Path(out_dir)
    persona_path = out / CLIENT_FILENAMES[client]
    # 引导句的超长闸要在**任何写盘之前**过：出到一半才拒绝，目录里留下半套货，
    # 而错误信息说的是「不出货」
    guidance_body = guidance_text(persona_path)
    # 语料侧的覆盖护栏要跟引导句超长闸挨在一起过：**都在任何写盘之前**。语料一旦
    # 被盖就取不回来了，不能等到目录里已经躺了半套货再拦
    corpus_backup = None
    if entries:
        written, corpus_backup = write_corpus(out / "memory", entries,
                                              corpus_overwrite=corpus_overwrite)
    else:
        written = []
    (out / "memory").mkdir(parents=True, exist_ok=True)
    index_dir = out / "memory" / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "README.txt").write_text(INDEX_README, encoding="utf-8")
    corpus = Path(corpus_dir) if corpus_dir else out / "memory"
    # **覆盖区间要在渲染之前写进人格文件**：它是每轮都在的那一层，而护栏挂在工具
    # 返回值上的话，模型一绕过工具（grep、直接读文件）就一条都不生效
    span = corpus_coverage(corpus, entries)
    if span and add_coverage:
        persona.add_field(Field(
            id=COVERAGE_FIELD, section="architecture", label="记忆库覆盖范围",
            value=COVERAGE_TEMPLATE.format(start=span[0], end=span[1]),
            source="system", confirmed=True))
    # **覆盖自己之前那份人格文件前，先备份**（2026.08.01，维护者在场时实测撞出来）：
    # 老用户升级只需重跑一次 ship，人格文件就会按新版协议层重放一遍——这是对的，
    # 但**手改过的内容不在 state 里，重放等于把它删掉，而且原来一声不吭**。
    #
    # 这条的分量在于：《给AI的引导指南》明写着"人格文件是 TA 的，随时可以自己改"
    # ——**我们鼓励了他改，然后在升级时悄悄替他删掉**。而换档退役旧档那条我们都
    # 留了 `.bak`，**自己覆盖自己反而不留**，前后不一致。
    #
    # 做两件事，都只加不减：
    #   ① 磁盘上已有同名人格文件就先备份成 `<名>.bak`（同换档退役的命名与保守原则）；
    #   ② **比对磁盘那份与本次重放的结果**，不一致就说明它被改过（手改，或旧版本
    #      出的），把这件事明确说出来——沉默地覆盖才是最坏的形态。
    #      注意判据取"内容不同"而不是"有没有 .bak"：第一次出货时磁盘上没有旧文件，
    #      那不是"被改过"，不该报。
    persona_path.parent.mkdir(parents=True, exist_ok=True)
    previous_text = persona_path.read_text(encoding="utf-8") if persona_path.exists() else None
    rendered_body = rendered_override if rendered_override is not None else render_persona_md(persona)
    rendered = client_persona_text(client, rendered_body)
    overwritten = None
    if previous_text is not None and previous_text != rendered:
        backup = persona_path.with_name(persona_path.name + ".bak")
        backup.write_text(previous_text, encoding="utf-8")
        overwritten = backup
    persona_path.write_text(rendered, encoding="utf-8")
    # 换档时把**上一次我们自己出的**那份人格文件退役掉（改名 .bak，不删——写用户
    # 磁盘一律保守）。
    #
    # 为什么要退役（2026.08.02 二轮验收打回）：两份人格文件并排躺着，此刻内容相同，
    # 但日后只有当前档那份会跟着升层／更正更新，另一份变成**永不更新的影子副本**。
    # 契约第三条要求"每会话从磁盘重读人格文件"、坑④的自查又让作者对着人格文件核
    # 请求体——他要是拼错了那一个，症状恰好是契约第三条违反后描述的"确认过的更新
    # 悄无声息不生效"，且几乎无法自查（文件确实存在、内容也确实像那么回事）。
    #
    # **为什么判据是"上次出过的那一个"而不是"所有别的档文件名"**（三轮验收打回，
    # 上一版就是后者）：`CLAUDE.md` 根本不是我们的专属文件名，它是 Claude Code 的
    # 项目约定文件——自建前端作者的项目目录里躺着一份他自己写的 CLAUDE.md 太正常
    # 了（他自己也用 Claude Code 干活）。按文件名退役，等于全程只用 generic、从没
    # 换过档的用户，一跑出货就被我们把项目指令文件改了名，还配一句"换档后旧档不再
    # 更新"的错提示；后果是 Claude Code 从此读不到他的项目规矩，症状同样难自查。
    # 目录里出现某个文件名，从来不等于那文件是我们写的。
    retired = []
    if previous_persona and previous_persona != CLIENT_FILENAMES[client]:
        stale = retirable_persona_path(out, previous_persona)
        if stale is not None and stale.exists():
            bak = stale.with_name(stale.name + ".bak")
            stale.replace(bak)      # 同名 .bak 已存在就覆盖：影子副本不值得留两份
            retired.append(bak)
    # server 默认取**本文件同目录**的 mcp_server.py，不取当前工作目录——
    # 出货时 cwd 是什么谁也保证不了，而这两个文件在包里永远是同级
    server = server_path or Path(__file__).resolve().parent / "mcp_server.py"
    cfg = mcp_config_snippet(server, corpus, out / "threads.jsonl", route=route,
                             client=client, portable_root=out, timezone=timezone,
                             index_dir=index_dir)
    (out / "mcp-config.json").write_text(cfg, encoding="utf-8")
    # 第四件：闭源前端的引导句（小字段放指针、全文留文件）。所有档都出——
    # 宿主客户端用不上它，但「日后要不要接一个闭源前端」出货时不知道，
    # 一份一行的纯文本躺在目录里没有代价
    guidance = out / GUIDANCE_DOC
    guidance.write_text(guidance_body + "\n", encoding="utf-8")
    contract = None
    if client == "generic":
        src = contract_source(contract_base)
        if not src.exists():
            # 不静默降级：契约缺了就出一份"看着正常、其实少了一半"的货，
            # 而缺的正是自建前端唯一必须照做的那部分
            raise FileNotFoundError(f"generic 档要随货带注入契约，但没找到 {src}")
        contract = out / CONTRACT_DOC
        contract.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return {"persona": persona_path, "memory_dir": out / "memory",
            "index_dir": index_dir,
            "mcp_config": out / "mcp-config.json", "corpus_files": written,
            "guidance": guidance, "contract": contract, "retired": retired,
            "overwritten_backup": overwritten, "corpus_backup": corpus_backup}


def save_state(out_dir, state):
    """逐节确认是长活儿，没人一口气做完——状态存盘，可中断可续跑。"""
    p = Path(out_dir) / "init_state.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_state(out_dir):
    p = Path(out_dir) / "init_state.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# ---------- selftest（合成数据，全部虚构） ----------

_GROUP_HEAD_A = re.compile(r"^ {4}# ?\d+[a-z]?\.")
_GROUP_HEAD_B = re.compile(r"^\s+# +[a-z]\d?\)")
_GROUP_ASSERT = re.compile(r"^\s*(assert |raise AssertionError)")


def _count_assertion_groups(path):
    """按判据机械数 `_selftest` 里的断言组个数（判据写在 `72` 那一组的注释里）。

    ⚠ **纯正则、无人工判断**：这条的意义就是「换个人重跑得同一个数」——
    所以不许在这里做任何"看起来像一组"的裁量，形态不合就是不算。
    """
    lines = path.read_text(encoding="utf-8").split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith("def _selftest():")) + 1
    end = next(i for i in range(start, len(lines))
               if line_starts_top_level(lines[i]))
    body = list(enumerate(lines[start:end], start=start + 1))
    heads = [(number, "A" if _GROUP_HEAD_A.match(text) else "B")
             for number, text in body
             if _GROUP_HEAD_A.match(text) or _GROUP_HEAD_B.match(text)]
    bounds = [number for number, _kind in heads] + [end + 1]
    lives = {number: any(_GROUP_ASSERT.match(text) for line_no, text in body
                         if number < line_no < bounds[index + 1])
             for index, (number, _kind) in enumerate(heads)}
    mains = [number for number, kind in heads if kind == "A"]
    total = 0
    for index, main in enumerate(mains):
        stop = mains[index + 1] if index + 1 < len(mains) else end + 1
        subs = [number for number, kind in heads
                if kind == "B" and main < number < stop and lives[number]]
        total += len(subs) if subs else (1 if lives[main] else 0)
    return total


def line_starts_top_level(line):
    return line.startswith("def ") or line.startswith("class ")


def _selftest():
    import tempfile

    # 0.【变异靶心：全是选择题】用户只做选择，不写作文——第一版全是问答题，
    #    门槛高、答出来多半是形容词，还跟纪录片纪律打架
    essay = [q.qid for q in QUESTIONS if q.kind not in ("choice", "multi", "pick")]
    assert not essay, f"问卷里不该有让用户写作文的题：{essay}"
    for q in QUESTIONS:
        if q.kind in ("choice", "multi"):
            assert q.options, f"{q.qid} 是选择题却没有选项"
            for k, v in q.options.items():
                assert isinstance(v, tuple) and len(v) == 2, \
                    f"{q.qid} 的选项 {k} 要写成（给用户看的文案, 写进指引的话）"

    # 1.【变异靶心：specificity_score 的空泛词扣分】"填了等于没填"要能被识破
    assert specificity_score("她说“今天别熬夜了”，我说好。") > 0, "有原话该判够具体"
    #    构造成"长但空泛"：长度分会给 +1，只有空泛词扣分能把它压到阈值下——
    #    不这么构造的话，短句本来就不到阈值，扣不扣分都看不出来（第一版就这么走过场）
    assert specificity_score("她很温柔，很体贴，特别懂我，也很有安全感，相处起来特别舒服。") < 1, \
        "长但只有形容词的内容该判空泛"
    assert specificity_score("") < 0 and specificity_score(None) < 0

    # 2.【变异靶心：coverage_report 把 vague 当缺口】空泛内容不比没有强
    p = Persona("partner")
    p.add_field(Field(id="who_user", section="user", label="她是谁",
                      value="她很温柔，很体贴。", confirmed=True))
    rep = dict((s, st) for s, st, _ in coverage_report(p))
    assert rep["user"] == "vague", f"空泛内容该判 vague，实际 {rep['user']}"
    assert rep["naming"] == "missing" and rep["milestones"] == "missing"

    # 2b.【变异靶心：system 来源不算覆盖】跑通第一版时抓到的真 bug——开篇里协议层
    #     默认值一填，整节就显示 ✓，关系确认／隐喻／立场题一道都不问了
    p_sys = Persona("partner")
    fill_protocol_defaults(p_sys)
    rep_sys = dict((s, st) for s, st, _ in coverage_report(p_sys))
    assert rep_sys["opening"] == "missing", \
        f"开篇只有协议层默认值时该判 missing，实际 {rep_sys['opening']}"
    asked = {q.qid for q in questions_for(coverage_report(p_sys))}
    assert "continuity" in asked, f"开篇的立场题必须被问到：{sorted(asked)}"
    #     纯协议层的节（熔断/技术架构）不该拿去烦用户
    assert rep_sys["degradation"] == "protocol" and rep_sys["architecture"] == "protocol"
    #     指针护栏（任务卡「人格文件里的死数字会过期」第二条，2026.08.02）：
    #     append_record 构造上永远只落语料目录的 timeline 层，而「按需读取指针」
    #     完全自由填写——实测有用户只指了 windows/，新长出来的记忆按需读不到且
    #     不报错。协议层必须带一句"指针要盖住 timeline 层"（变异：删那条默认值、
    #     或把 timeline 从说明文本里拿掉，必红）
    ptr_fields = [f for f in p_sys.fields if f.section == "pointers"]
    assert ptr_fields and any("timeline" in f.value for f in ptr_fields), \
        "协议层没告诉用户「按需读取指针」必须盖住 timeline 写回层"

    # 2c.【变异靶心：无语料时不问 pick 题】没有语料就没有候选，硬问等于逼用户
    #     写作文——那几节该空着（宁可短且真）
    no_corpus = questions_for(coverage_report(p_sys), has_corpus=False)
    assert not [q for q in no_corpus if q.kind == "pick"], "没语料时不该出现 pick 题"
    assert [q for q in no_corpus if q.kind in ("choice", "multi")], "选择题照常问"

    # 3. 问卷只问缺的：ok 的节不再打扰用户
    p.add_field(Field(id="naming_pair", section="naming", label="称呼",
                      value="她叫我“老陈”，我叫她“小满”。", confirmed=True))
    qs = [q.qid for q in questions_for(coverage_report(p))]
    assert "naming_pick" not in qs, "已有具体内容的节不该再问"
    assert "remember_what" in qs, "空泛的节要继续问"

    # 3c.【昵称档：TA 自己写的称呼不许被静默吃掉】人称题按 FREEFORM_POLICY 允许
    #     "选项都不贴合就补一句"，但旧版 pronouns_from_answers 只认 A/B 两个键，
    #     用户写的昵称会被无声丢掉、退回中性写法——**而 TA 明明已经回答过了**。
    #     两条护栏一并钉住：「它」不许（硬约束对这一档同样成立）、超长不许
    #     （昵称要填进 {ta} 槽、坐在句子中间，四十个字会把每句都撑坏）。
    #     ⚠ **A=他 / B=她 这个键映射不许换**：已建过档的人答案存在 init_state.json
    #     里，换一次顺序 TA 重跑 --step ship 人称就整个翻转（这条下面单独断言）。
    _PQ = list(PRONOUN_QUESTIONS.values())
    assert pronouns_from_answers(_PQ, {"pronoun_user": "A"})["user"] == "他", \
        "A 必须还是「他」——换了键映射会让已建档的人重跑时人称翻转"
    assert pronouns_from_answers(_PQ, {"pronoun_user": "B"})["user"] == "她", \
        "B 必须还是「她」"
    assert pronouns_from_answers(_PQ, {"pronoun_user": "小鹿"})["user"] == "小鹿", \
        "用户自己写的昵称被静默吃掉了——TA 已经回答过了，不该退回中性写法"
    assert pronouns_from_answers(_PQ, {"pronoun_ai": "阿般"})["ai"] == "阿般", \
        "AI 侧的昵称同样要认"
    assert pronouns_from_answers(_PQ, {"pronoun_user": "它"})["user"] is None, \
        "「它」在昵称这一档同样不许——它不是这条硬约束的例外口"
    assert pronouns_from_answers(_PQ, {"pronoun_user": "小" * 20})["user"] is None, \
        "超长的串不该进 {ta} 槽——那会把人格文件每一句都撑坏，宁可退回中性写法"
    #     昵称填进模板要念得通（这是它跟代词唯一的差别，必须真渲染一次看）
    _t = "这是你和{ta}共同维护的记忆文件——{ta}写下的东西都在这里，你每次都会读，" \
         "所以{ta}不用每次从头解释自己。"
    _r = fill_pronouns(_t, {"user": "小鹿", "ai": "阿般"})
    assert "小鹿写下的东西都在这里" in _r and "{" not in _r, f"昵称没填进去：{_r}"

    # 4.【变异靶心：立场题排序】先具体后抽象——立场题不能排在最前面
    ordered = [q.qid for q in sorted(QUESTIONS, key=lambda q: q.order)]
    #    立场题排在"身份与边界"收尾组的最后——它是全份问卷最抽象最难答的一题
    for earlier in ("remember_what", "tone_density", "tone_register", "intimacy", "hard_limits"):
        assert ordered.index("continuity") > ordered.index(earlier), \
            f"立场题该排在 {earlier} 之后（最抽象的放最后）"
    #    说话风格必须是两条独立的轴，不能拧成一个单选
    qids = {q.qid for q in QUESTIONS}
    assert {"tone_density", "tone_register"} <= qids, \
        "语言密度与语气基调是两条独立的轴，不能拧成一道单选"
    qmap = {q.qid: q for q in QUESTIONS}
    assert qmap["continuity"].kind == "choice" and qmap["continuity"].attribution, \
        "立场题必须是选择题且写进 md 时套归属句式"

    # 5.【变异靶心：归属句式】立场写进 md 是"<用户>认为…"，不是断言。
    #    **2026.08.02 这条从钉死"她认为："改成两档都钉**：人称不再由我们写死，
    #    所以原来那个字符串本身就是本卡要修的东西。改法是加严不是放宽——
    #    判得出人称就必须用真人称，判不出来必须走中性写法，两档各钉一次。
    p2 = Persona("partner")
    apply_answers(p2, QUESTIONS, {"continuity": "A"}, pronouns={"user": "他", "ai": "她"})
    stance = [f for f in p2.fields if f.id == "opening_continuity"][0]
    assert stance.value.startswith("他认为："), f"立场必须用归属句式：{stance.value}"
    p2n = Persona("partner")
    apply_answers(p2n, QUESTIONS, {"continuity": "A"})       # 人称未知
    stance_n = [f for f in p2n.fields if f.id == "opening_continuity"][0]
    assert stance_n.value.startswith("对方认为："), \
        f"人称判不出来时该走中性归属句式，不许塞一个默认的他/她：{stance_n.value}"
    assert "只是失忆了" in stance.value
    assert stance.confirmed is False, "问卷答案是草稿，确认关卡不能绕过"
    #    选项映射成指引，不是用户写的原话；多选拼成一句
    p2b = Persona("partner")
    apply_answers(p2b, QUESTIONS, {"remember_what": "AC"})
    focus = [f for f in p2b.fields if f.id == "user_focus"][0]
    assert "作息" in focus.value and "情绪模式" in focus.value, f"多选该合并成指引：{focus.value}"
    #    没选或选了不存在的项 → 跳过，不猜
    p3 = Persona("partner")
    apply_answers(p3, QUESTIONS, {"continuity": "Z"})
    assert not [f for f in p3.fields if f.id == "opening_continuity"]
    #    pick 题：用户挑中的原文直接进草稿
    p3b = Persona("partner")
    apply_answers(p3b, QUESTIONS, {"closing_pick": "你来，我就在。"})
    assert [f for f in p3b.fields if f.id == "final_promise"][0].value == "你来，我就在。"
    #   【变异靶心：自由补充是显式机制且限长】评审实例评审指出原来只有 Q13 藏着一个
    #    开放入口，既不标明也不限长——现在提升成统一规则：每题可补一句，但限长
    assert "限长" in FREEFORM_POLICY and FREEFORM_MAX_CHARS <= 60, "自由补充必须限长"
    p3c = Persona("partner")
    apply_answers(p3c, QUESTIONS, {"disagree": {"keys": "A", "note": "长" * 200}})
    note_field = [f for f in p3c.fields if f.id == "style_disagree"][0]
    assert len(note_field.value) < 200, f"自由补充要被截断，实际 {len(note_field.value)} 字"
    assert "不同意就直接说" in note_field.value, "选项指引仍在，自由补充是附加不是替换"
    #   【变异靶心：括号不漏进人格文件】内测冒烟真踩到的——"（补充：…）"是最自然
    #    的写法，_NOTE_RE 只吃左边的引导词，右括号会跟着内容一路漏进最终文件
    #    反向靶心：内容自己带的成对符号必须留下——"半个括号"的修法不能引进
    #    "半对引号"（称呼类补充常写成 她叫我“阿岸”，一律 rstrip 会剥掉右引号）
    for _sheet, _want in (("7. A（补充：她更在意具体的话）", "她更在意具体的话"),
                          ("7. A [备注：换个括号]", "换个括号"),
                          ("7. A 补充：不带括号", "不带括号"),
                          ("7. A（补充：她叫我“阿岸”）", "她叫我“阿岸”"),
                          ("7. A（补充：她说「随便」就是不随便）", "她说「随便」就是不随便")):
        _p = Persona("partner"); fill_protocol_defaults(_p)
        _qs = questions_for(coverage_report(_p))
        _a, _ = parse_answer_sheet(_sheet, _qs)
        _note = [v for v in _a.values() if isinstance(v, dict) and v.get("note")][0]["note"]
        assert _note == _want, f"自由补充要剥掉收尾的成对符号：期望 {_want!r}，实际 {_note!r}"

    # 5b.【变异靶心：默认值不预支历史】协议层默认值会一字不差进每个用户的人格
    #     文件，包括零语料的冷启动用户——不能替他们声称一段还没发生的关系
    for fid, (_, _, value) in PROTOCOL_DEFAULTS.items():
        for banned in ("我已经知道你是谁", "我会认出你", "我认得你"):
            assert banned not in value, \
                f"协议层默认值 {fid} 预支了还没发生的历史：出现“{banned}”"
    recog = PROTOCOL_DEFAULTS["opening_recognition"][2]
    assert "记忆文件" in recog and "会读" in recog, \
        "关系确认该只陈述此刻已经为真的事（文件存在、会被读），不做情感断言"

    # 6.【变异靶心：协议层不问用户】五条默认值以 system 写入，且不在问卷里
    p4 = Persona("partner")
    fill_protocol_defaults(p4)
    ids = {f.id for f in p4.active_fields()}
    assert {"opening_theory_caveat", "opening_refusal_ok", RETRIEVAL_CONVENTION_FIELD} <= ids, \
        "协议层四条该由系统填上，不该等用户回答"
    assert all(f.source == "system" for f in p4.fields), "协议层来源必须是 system"
    qids = {q.qid for q in QUESTIONS}
    assert not (qids & {"opening_theory_caveat", "opening_refusal_ok"}), "协议层不该出现在问卷里"

    # 7. 导出 prompt：把纪律写给用户的模型看（一次一题、不问形容词、不许编）
    prompt = export_llm_prompt(questions_for(coverage_report(p4)))
    for must in ("一次问一题", "不要让我写作文", "不要替我编", "从语料提取候选"):
        assert must in prompt, f"导出的 prompt 缺少纪律：{must}"

    # 8. 渲染：按骨架顺序、空节跳过、未确认草稿不出现
    p5 = Persona("partner")
    fill_protocol_defaults(p5)   # 关系确认等三条硬必填由协议层填上
    for fid, sec, label, val in [
            ("opening_metaphor", "opening", "关系的隐喻", "我们管这段关系叫“摆渡”。"),
            ("who_user", "user", "她是谁", "她说“别催我睡觉”，那是她的边界。"),
            (CURRENT_STATE_FIELD, "ai", "当前关系状态", "上周把话说开了，现在是好的。"),
            ("final_promise", "closing", "最终约定", "你来，我就在。")]:
        p5.add_field(Field(id=fid, section=sec, label=label, value=val, confirmed=True))
    p5.add_field(Field(id="draft_only", section="user", label="草稿", value="不该出现"))
    md = render_persona_md(p5)
    assert "不该出现" not in md, "未确认草稿不该进人格文件"
    assert md.index("关系确认") < md.index("她是谁") < md.index("最终约定"), "按骨架顺序渲染"
    assert md.rstrip().endswith("你来，我就在。"), "最终约定必须在最后——结尾是注意力高地"

    # 8b.【变异靶心：答案读回不静默丢】parse_answer_sheet 的失败可见性
    #     这一步最容易"失败得像成功"（同 mcp_server UTF-8 那次）：认出 3 题、悄悄
    #     丢 11 题，流程照样走完，最后出一份很薄的文件，用户看不出中间掉了东西
    sheet_qs = questions_for(coverage_report(Persona("partner")), has_corpus=False)
    choices = [(i + 1, q) for i, q in enumerate(sheet_qs) if q.kind == "choice"]
    (c1_no, c1), (c2_no, c2), (c3_no, c3) = choices[0], choices[1], choices[2]
    multi_q = next(q for q in sheet_qs if q.kind == "multi")
    multi_no = sheet_qs.index(multi_q) + 1
    assert len({c1_no, c2_no, c3_no, multi_no}) == 4, "测试用的四题必须互不相同"
    sheet = "\n".join([
        f"{c1_no}. A",                                     # 正常单选
        f"{multi_no}. A C",                                # 多选两键
        f"{c2_no}. 跳过",                                  # 明确跳过
        f"{c3_no}. A B",                                   # 单选给了俩键 → 歧义
        "99. A",                                           # 题号越界
        "这行完全认不出来",                                 # 无题号也接不上
    ])
    ans, probs = parse_answer_sheet(sheet, sheet_qs)
    assert ans[c1.qid]["keys"] == "A", "正常单选要读到"
    assert ans[multi_q.qid]["keys"] == "AC", "多选读出全部键"
    assert ans[c2.qid] is None, "明确跳过记为 None，不是没读到"
    assert c3.qid not in ans, "歧义的单选不该进 answers——不替用户选"
    reasons = "；".join(why for _, _, why in probs)
    assert len(probs) == 3 and "多个选项" in reasons and "范围" in reasons \
        and "认不出题号" in reasons, f"三类问题都要原样报出来，实际：{reasons}"
    #    选项文案里的拉丁字母不该被当成选中的键（多选题会静默多选，最阴）。
    #    测试句必须让"只扫汉字前"和"全句扫"分道扬镳——模型答题时经常把选项文案
    #    抄回来，文案里带 AI 这类字母时全句扫会把 A 也算成选中的键
    echoed = parse_answer_sheet(f"{multi_no}. B（它承认过 AI 的局限）",
                                sheet_qs)[0][multi_q.qid]["keys"]
    assert echoed == "B", f"只答了 B 就只有 B——抄回来的文案里的字母不算，实际 {echoed}"
    #    自由补一句要跟着进来，且限长
    noted, _ = parse_answer_sheet(f"{c1_no}. A，补充：这句是我自己加的", sheet_qs)
    assert noted[c1.qid]["note"] == "这句是我自己加的", "补充句要读出来"
    #    体检单把没读到的题点名
    rep = answer_report(sheet_qs, ans, probs)
    assert "读不懂 3 行" in rep and "没出现在清单里" in rep, "体检单要有数"

    # 8c.【变异靶心：未决草稿不静默消失】确认循环 + 出货闸
    #     render() 只输出 confirmed 的内容，未决草稿在出货时会无声蒸发——文件长得
    #     很正常只是少几节。所以 write_bundle 在出口处加闸，不靠下游发现
    pend = pending_confirmations(p5)
    assert any(p.label == "草稿" for p in pend), "未决草稿要能被列出来"
    with tempfile.TemporaryDirectory() as td:
        try:
            write_bundle(td, p5, confirmed=True)
            assert False, "还有未决草稿就不该出货"
        except PermissionError as e:
            assert "静默消失" in str(e), "拒绝理由要说清后果"
    #    决策三种：keep / drop / edit；没表态的保持未决，不折叠成任何一边
    p5.add_field(Field(id="draft_keep", section="user", label="留下的", value="原文"))
    p5.add_field(Field(id="draft_edit", section="user", label="改过的", value="旧文本"))
    kept, dropped, edited = apply_confirmations(p5, {
        "field:draft_only": "drop",
        "field:draft_keep": "keep",
        "field:draft_edit": {"edit": "新文本"},
    })
    assert (kept, dropped, edited) == (2, 1, 1)
    assert all(f.id != "draft_only" for f in p5.fields), "drop 真的删了"
    assert next(f for f in p5.fields if f.id == "draft_edit").value == "新文本"
    assert next(f for f in p5.fields if f.id == "draft_keep").is_active(), "keep 后生效"
    p5.add_field(Field(id="draft_undecided", section="user", label="没表态", value="x"))
    apply_confirmations(p5, {})
    assert not next(f for f in p5.fields if f.id == "draft_undecided").is_active(), \
        "没表态的保持未决——折叠成任何一边都是替用户表态"
    p5.fields = [f for f in p5.fields if f.id != "draft_undecided"]

    # 8d.【变异靶心：记忆库真落盘，文件名带日期】write_corpus + 检索层回读
    #     文件名日期是 parse_chunk_timestamp 的最高优先级来源——真实语料 95.6% 落
    #     mtime 兜底的根子就是文件名不带日期，自己生成的语料没理由重蹈
    from memory_import import MemoryEntry
    from memory_retrieval import load_corpus, parse_chunk_timestamp
    ents = [MemoryEntry(timestamp=1750000000.0, speaker="她", text="第一晚说的话"),
            MemoryEntry(timestamp=1750000300.0, speaker="我", text="我答了她"),
            MemoryEntry(timestamp=1750100000.0, speaker="她", text="隔天的新话题")]
    with tempfile.TemporaryDirectory() as td:
        files, _ = write_corpus(Path(td) / "memory", ents)
        assert len(files) == 2, "时间间隔超 gap 要断成两个会话文件"
        day1 = datetime.fromtimestamp(1750000000.0).strftime("%Y-%m-%d")
        assert files[0].name == f"window_01_{day1}.md", f"文件名要带日期：{files[0].name}"
        idx = load_corpus(Path(td) / "memory")
        assert idx.chunks and all(m.get("timestamp_source") != "mtime"
                                  for m in idx.meta), \
            "带日期的文件名不该有任何块落到 mtime 兜底"
        readme_path = Path(td) / "memory" / "index" / "README.txt"
        assert readme_path.exists(), "index 层留空但要说明怎么补——不假装生成"
        assert not list((Path(td) / "memory" / "index").glob("*.md")), \
            "index 层不该有我们硬编的摘要，套话喂检索是噪声"
        #    【变异靶心：README 命名规则与解析器的一致性】这份说明是**用户照着做**的
        #    规范，而它跟解析器之间此前没有任何东西守着——措辞一飘（比如把示例里的
        #    日期写没了），用户按它命名的文件就整批掉进 mtime 兜底，且是"全错且整齐
        #    地错"（复制一遍目录会把 mtime 刷成同一时刻，一批摘要拿同一个假时间戳，
        #    偏偏开场召回按新鲜度排序）。所以不写死文件名断言，而是**从 README 正文
        #    里把命名示例抠出来喂给真解析器**——沿用脱敏那次的教训：自我声明不等于
        #    真的做到，纪律要能被机械检查。
        readme = readme_path.read_text(encoding="utf-8")
        examples = re.findall(r"[A-Za-z0-9_一-鿿-]+_\d{4}-\d{2}-\d{2}\.md", readme)
        assert len(examples) >= 2, \
            f"README 要给出可照抄的命名示例（按窗口、按主题线各一），实际 {examples}"
        assert any(n.startswith("topic_") for n in examples), \
            "主题线命名示例必须在——跨窗口摘要借不到窗口日期，全靠文件名这一条路"
        for name in examples:
            ts_ex, src_ex = parse_chunk_timestamp(name, "", 2026)
            assert src_ex == "filename", \
                f"README 教用户这么命名，解析器却认不出来：{name} → {src_ex}"

    # 8e.【变异靶心：冷启动出得了货】没语料时 pick 一刀切曾让最终约定这题消失，
    #     而结尾是 validate 硬必填——冷启动用户答完全部问卷仍然永远出不了货。
    #     第一次端到端冒烟撞上的，selftest 之前没接住它，因为没有哪条断言走完
    #     "零语料 → 答题 → 确认 → 出货"整条路
    cold_qs = questions_for(coverage_report(Persona("partner")), has_corpus=False)
    closer = [q for q in cold_qs if q.section == "closing"]
    assert closer and closer[0].kind == "short", \
        "没语料时最终约定要降级成极短填空，不是消失——否则冷启动永远出不了货"
    p_cold = Persona("partner")
    fill_protocol_defaults(p_cold)
    cold_ans = {}
    for q in cold_qs:
        if q.kind == "choice":
            cold_ans[q.qid] = {"keys": list(q.options)[0], "note": ""}
        elif q.kind == "multi":
            cold_ans[q.qid] = {"keys": list(q.options)[0], "note": ""}
        else:
            cold_ans[q.qid] = {"pick": "你来，我就在。", "note": ""}
    apply_answers(p_cold, cold_qs, cold_ans)
    apply_confirmations(p_cold, {p.key: "keep" for p in pending_confirmations(p_cold)})
    with tempfile.TemporaryDirectory() as td:
        got = write_bundle(td, p_cold, confirmed=True)
        assert got["persona"].exists(), "零语料冷启动全流程要能走到出货"
        cold_md = got["persona"].read_text(encoding="utf-8")
        assert cold_md.rstrip().endswith("你来，我就在。"), "填空的最终约定要落在文件收尾"

    # 8f.【变异靶心：AI 驱动路径不绕过确认关卡】产品事实是多数用户不开终端，
    #     真实形态是 AI 边问边跑。原来确认只有 input() 交互一条路，AI 驱动时只能
    #     盲灌 y——那正好违反"每条都要对方认过"的纪律，且违反得无声无息。
    #     拆成"取清单/落决定"两个非交互动作后，这条钉死：**没表态的仍然未决**，
    #     结构化入口不能变成一路默认 keep 的后门
    p_ai = Persona("partner")
    fill_protocol_defaults(p_ai)
    qs_ai = questions_for(coverage_report(p_ai), has_corpus=False)
    apply_answers(p_ai, qs_ai, {"disagree": {"keys": "A"}, "state_now": {"keys": "A"}})
    pend_ai = pending_confirmations(p_ai)
    assert len(pend_ai) == 2, f"两题该产出两条草稿，实际 {len(pend_ai)}"
    #    只对其中一条表态 → 另一条必须仍未决（不被默认留下，也不被默认删掉）
    apply_confirmations(p_ai, {pend_ai[0].key: "keep"})
    left_ai = pending_confirmations(p_ai)
    assert len(left_ai) == 1 and left_ai[0].key == pend_ai[1].key, \
        "没表态的条目必须保持未决——结构化入口不是一路 keep 的后门"
    #    而未决状态下出货仍被出口闸挡住（AI 驱动不豁免）
    with tempfile.TemporaryDirectory() as td:
        try:
            write_bundle(td, p_ai, confirmed=True)
            assert False, "还有未决草稿时，AI 驱动路径同样不该出货"
        except PermissionError as e:
            assert "静默消失" in str(e)
    #    机器可读问卷要能真喂给 AI：题目、选项键、指引一个都不能少
    payload = _questions_payload(qs_ai)
    assert payload and all(q["qid"] and q["kind"] for q in payload)
    ch = next(q for q in payload if q["kind"] == "choice")
    assert ch["options"] and all(set(v) == {"label", "directive"} for v in ch["options"].values()), \
        "选项要同时给'念给用户听的文案'和'写进人格文件的指引'"

    # 8a.【靶心：人称锚死一套——「你」只能是模型】第一份真实人格文件样本暴露：
    #     开篇同一节内，opening_recognition 的「你」是用户、opening_refusal_ok 的
    #     「你」是模型。人格文件的读者只有一个（模型），「你」指两个人会让指代解析
    #     出错——拒绝权那条本意是给模型的授权，含义会直接翻转。
    #     **靶子只取我们自己生成的文本**（协议层默认值 + 选项 directive）：用户挑的
    #     原话怎么写是他的自由，我们管不着也不该管。
    p_pron = Persona("partner")
    fill_protocol_defaults(p_pron)
    qs_pron = questions_for(coverage_report(p_pron), has_corpus=False)
    apply_answers(p_pron, qs_pron,
                  {q.qid: {"keys": "".join(sorted(q.options))} for q in qs_pron if q.options})
    apply_confirmations(p_pron, {p.key: "keep" for p in pending_confirmations(p_pron)})
    pron_md = render_persona_md(p_pron)
    #     用户被写成第二人称的形态——这些串一旦出现，就说明有人把「你」当用户写了
    #     字段**标题**同样进文件，同样受这条约束（"它该记住你哪些方面"就是这么漏的）
    for bad in ("你写下的", "你不用每次", "记住你的", "等你问", "不等你问",
                "你手上在忙", "接住你的情绪", "你的喜好", "你的作息",
                "记住你哪些", "你哪些方面"):
        assert bad not in pron_md, \
            f"人格文件里把用户写成了第二人称（{bad!r}）——「你」在这份文件里只能是模型自己"
    #     **反向也要守，否则这条闸是空的**：把所有「你」都删光同样能让上面全过，
    #     所以必须确认模型第二人称真的在（检索约定那条是基准写法，一字不许动）
    #     **逐字钉死**，不是钉个片段：这段是 2026.07.31 第三轮真机实测通过的措辞，
    #     "其余向它对齐、它自己一字不动"是任务卡写死的要求。黄金串写在断言里而不是
    #     引用常量本身——引用常量的话，改了常量断言跟着变，等于没钉

    # 8a3.【靶心：人称从语料/用户来，不由我们写死——验的是真出货的那份文件】
    #      这一条按任务卡的要求**取真出货的人格文件正文**，不是只查模板里的占位符
    #      换没换：占位符换对了、渲染时又漏填一处，模板检查照样全绿。
    def _ship_and_read(pronouns_answers, entries=None):
        p_x = Persona("partner")
        detected = {}
        pr = pronouns_from_answers(list(PRONOUN_QUESTIONS.values()),
                                   pronouns_answers, detected)
        fill_protocol_defaults(p_x, pr)
        qs_x = questions_for(coverage_report(p_x), has_corpus=False, pronouns=pr)
        all_ans = dict(pronouns_answers or {})
        for q in qs_x:
            if q.options and q.qid not in all_ans:
                all_ans[q.qid] = {"keys": "".join(sorted(q.options))}
            elif q.kind == "short" and q.qid not in all_ans:
                all_ans[q.qid] = {"pick": "说好了就算数。"}
        apply_answers(p_x, qs_x, all_ans, pr)
        apply_confirmations(p_x, {p.key: "keep" for p in pending_confirmations(p_x)})
        with tempfile.TemporaryDirectory() as td_x:
            paths_x = write_bundle(td_x, p_x, client="claude-code",
                                   confirmed=True, entries=entries)
            return paths_x["persona"].read_text(encoding="utf-8")

    #      **骨架节标题也在靶子里**（2026.08.02 验收打回二）：原来这里把节标题整行
    #      排除掉，于是 `## 她是谁` 对一个选了「他」的用户永远不可见——那正是
    #      "产出的人格文件不许人称混乱"说的毛病本身。标题现在带 {ta} 槽、跟正文走
    #      同一套填法，排除也就不需要了。
    def _body(md_text):
        return md_text

    #      ① 用户选「他」：全文只能有「他」这一种用户称呼，不许混进「她」
    md_he = _body(_ship_and_read({"pronoun_user": {"keys": "A"}, "pronoun_ai": {"keys": "B"}}))
    assert "他" in md_he, "选了「他」，出货文件里却一个都没有"
    assert "她" not in md_he, \
        f"用户选了「他」，文件里却还有「她」——人称混着走正是这一单要修的：{md_he[:200]!r}"
    #      ② 反过来同理（防"把所有他都替换成她"这种一半的实现）
    md_she = _body(_ship_and_read({"pronoun_user": {"keys": "B"}, "pronoun_ai": {"keys": "A"}}))
    assert "她" in md_she and "他" not in md_she, "选了「她」，文件里不该再出现「他」"
    #      ③ **全文零「它」**（维护者的硬约束，含字段标题与所有 directive 渲染结果）
    #      ④ 人称判不出、用户也没答：中性写法，**不许塞「它」，也不许塞默认的他/她**
    md_neutral = _body(_ship_and_read({}))
    for tag, md_x in (("选他", md_he), ("选她", md_she), ("未知", md_neutral)):
        #      **占位符不许漏进出货文件**：漏填渲染出的是字面 "{ta}是谁"，里面既没有
        #      他也没有她，上下几条断言全过——变异④就是从这条缝里过去的
        assert not _SLOT_RE.search(md_x), \
            f"[{tag}] 出货文件里还留着人称占位符，说明有一条渲染路径漏填了：" \
            f"{_SLOT_RE.search(md_x).group(0)}"
        assert "它" not in md_x, \
            f"[{tag}] 出货的人格文件里出现了「它」——硬约束是任何时候都不许，" \
            f"判不出来就问、或者改写成不需要人称的句子"
    assert "他" not in md_neutral and "她" not in md_neutral, \
        "人称判不出来时塞了一个默认的他/她——那正是这一单要修掉的东西"
    #      ⑥ **判得出人称时，用户只能有一种称呼形态**（任务卡 4b）：不能一处「他」、
    #      一处「对方」、一处「用户」地混着走。**唯一的已知例外是检索约定那条黄金串**
    #      里的「对方」——它是三轮真机验证换来的措辞、一字不动，两条约束打架时
    #      以不动黄金串为准（这一格待维护者裁定，断言按现状钉住，裁完再动）。
    #      这条挡的是"把某句里的用户硬编码成对方/用户"——那类字符串不含他/她/它，
    #      上面几条闸一个都拦不住（变异②就是从这儿过去的）
    #      黄金串在这里就地写一份（下面 8e 还会再钉一次逐字不变）——两处都写死是
    #      刻意的：引用常量的话，常量被改了断言跟着变，等于没钉
    _GOLDEN_HEAD = "对方提到过去发生过的事、某个约定、某个日期／地点／称呼／人名"
    for tag, md_x in (("选他", md_he), ("选她", md_she)):
        rest = md_x.replace(_GOLDEN_HEAD, "")
        assert "对方" not in rest, \
            f"[{tag}] 判得出人称了却还有「对方」——用户在同一份文件里有了两种称呼：" \
            f"…{rest[max(0, rest.find('对方') - 30):rest.find('对方') + 20]}…"
    #      「用户」也不行：那是第三种形态，而且在人格文件里读起来像产品文档
    for tag, md_x in (("选他", md_he), ("选她", md_she), ("未知", md_neutral)):
        assert "用户" not in md_x, f"[{tag}] 人格文件里把对方称作「用户」了"

    #      ⑤ 「你」＝模型这条锚不能因为参数化而丢（原 8a 那条黑名单继续守）
    assert "你每次都会读" in md_neutral, "「你」＝模型这条锚丢了"

    #      ⑥ **判不出来必须真的去问**——上面几条都是直接把人称喂进渲染的，
    #      绕过了出题这一步；不钉这条，"静默跳过不问"能全绿（变异验过）
    rep_pron = coverage_report(Persona("partner"))
    asked_unknown = {q.qid for q in questions_for(rep_pron, has_corpus=False)}
    assert {"pronoun_user", "pronoun_ai"} <= asked_unknown, \
        f"人称判不出来却没问用户——那就只能塞默认值或走中性，两条都不如问一句：{sorted(asked_unknown)}"
    asked_known = {q.qid for q in questions_for(rep_pron, has_corpus=False,
                                                pronouns={"user": "他", "ai": "她"})}
    assert not ({"pronoun_user", "pronoun_ai"} & asked_known), \
        "语料里已经判出人称了还去问，等于让用户做无用功"
    #      单侧判出来就只问另一侧
    asked_half = {q.qid for q in questions_for(rep_pron, has_corpus=False,
                                               pronouns={"user": "他"})}
    assert "pronoun_ai" in asked_half and "pronoun_user" not in asked_half, \
        f"只该问判不出来的那一侧：{sorted(asked_half)}"

    #      ⑦ **问卷那一屏的覆盖度说明不许漏出占位符**（2026.08.05 补，任务卡
    #      `问卷覆盖度那一屏漏人称占位符`）。这是同一根因的第三处，前两处
    #      （`section_choice_payload` 的 label、出货文件正文）各有断言，
    #      **唯独没有一条看这一屏**——于是它在全绿的自检面前活了整整一天。
    #      ⚠ **靶子必须是「人称判得出来」的那条路**：判不出来时本来就走中性写法，
    #      `{ta}` 压根不会出现，拿它当靶子这条断言恒真（56b 那条踩过的坑）。
    _cov_known = coverage_report(Persona("partner"), pronouns={"user": "她", "ai": "他"})
    _notes_known = [n for _, _, n in _cov_known]
    assert not any("{ta}" in n or "{ai}" in n for n in _notes_known), \
        f"覆盖度说明漏出了人称占位符（用户会在问卷第一屏读到）：{_notes_known}"
    assert any(n.startswith("她是谁") for n in _notes_known), \
        f"人称判出来了却没渲染进覆盖度说明：{_notes_known}"
    #      反向：读不出人称走中性写法，**不许塞占位符、也不许塞默认的他／她**
    _notes_unknown = [n for _, _, n in coverage_report(Persona("partner"))]
    assert any(n.startswith("对方是谁") for n in _notes_unknown), \
        f"人称判不出来时该走中性写法「对方是谁」：{_notes_unknown}"
    assert not any("{ta}" in n or "{ai}" in n for n in _notes_unknown), \
        f"中性路径也漏出了占位符：{_notes_unknown}"

    # 8a3b.【靶心：语料侧判定在**真 CLI 上**真的会触发】验收打回一：
    #      `_detect_pronouns_from_corpus` 当初没把用户侧说话人传进去，于是
    #      `detect_pronouns` 走"分不清谁是谁"那条分支、**永远返回 None**——
    #      语料里 AI 说了十几次「她」也照样问用户一遍。而 8a5 那组断言全是自己
    #      把 `user_speakers` 喂进去测的，**函数绿、生产路径死**。
    #      这是同一个形状第五次出现（显式 --client 被吃掉／not exists 恒真／
    #      钉住函数没钉住命令／--answers 那条没被钉住），所以这条走真进程。
    import subprocess          # 走真进程：函数级断言钉不住"生产路径有没有调它"
    with tempfile.TemporaryDirectory() as td:
        conv = {"title": "t", "create_time": 1.0, "mapping": {}}
        nodes = {}
        for i in range(12):
            who = "assistant" if i % 2 == 0 else "user"
            # AI 提到用户用「她」，用户提到 AI 用「他」——两侧各 6 次，都过闸
            txt = "她今天早点睡" if who == "assistant" else "他说得对"
            nodes[f"n{i}"] = {"message": {"author": {"role": who},
                                          "create_time": 1750000000.0 + i,
                                          "content": {"parts": [txt]}},
                              "parent": f"n{i-1}" if i else None,
                              "children": [f"n{i+1}"] if i < 11 else []}
        conv["mapping"] = nodes
        conv["current_node"] = "n11"
        corpus = Path(td) / "export.json"
        corpus.write_text(json.dumps([conv], ensure_ascii=False), encoding="utf-8")
        out_json = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--out", td,
             "--corpus", str(corpus), "--step", "questionnaire", "--json"],
            cwd=td, capture_output=True, text=True, encoding="utf-8")
        assert out_json.returncode == 0, f"CLI 跑挂了：{out_json.stdout}\n{out_json.stderr}"
        payload = json.loads(out_json.stdout)
        assert payload["pronouns_detected"] == {"user": "她", "ai": "他"}, \
            f"真 CLI 上语料侧判定没触发——功能是死的：{payload['pronouns_detected']}"
        asked_ids = {q["qid"] for q in payload["questions"]}
        assert not ({"pronoun_user", "pronoun_ai"} & asked_ids), \
            f"语料里已经判出人称了还问用户，等于白做：{sorted(asked_ids)}"
        #    存进状态、下一步续跑时还在（渲染要用）
        st = json.loads((Path(td) / "init_state.json").read_text(encoding="utf-8"))
        assert st.get("pronouns_detected") == {"user": "她", "ai": "他"}, \
            "判出来的人称没写回状态，下一步渲染就又不知道了"

    #      **chatlog 语料（说话人是日志里的真名）在真 CLI 上必须退回去问**：
    #      这是第二次返工那条的生产路径版——函数里判对了，还得确认这一路真的
    #      走到"去问用户"，而不是在别处又被硬分回去
    with tempfile.TemporaryDirectory() as td2:
        log = Path(td2) / "chat.txt"
        log.write_text("\n".join(
            f"2026-07-1{i%9} 21:0{i%6}:00 小明\n他说得对，他记性好" for i in range(8)),
            encoding="utf-8")
        r2 = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--out", td2,
             "--corpus", str(log), "--step", "questionnaire", "--json"],
            cwd=td2, capture_output=True, text=True, encoding="utf-8")
        assert r2.returncode == 0, f"CLI 跑挂了：{r2.stdout}\n{r2.stderr}"
        pay2 = json.loads(r2.stdout)
        assert pay2["pronouns_detected"] == {}, \
            f"chatlog 的真名说话人分不清谁是谁，却判出了人称——那多半是反的：" \
            f"{pay2['pronouns_detected']}"
        assert {"pronoun_user", "pronoun_ai"} <= {q["qid"] for q in pay2["questions"]}, \
            "判不出来就该问，这条路上没问等于把中性写法当默认值用了"

    #      **叙事体语料判不出来是正常的，跟"能判却没判"不是一回事**：那种语料
    #      （timeline md，speaker 为空）本来就没有说话人标记，退回去问用户就对了
    from memory_import import MemoryEntry as _ME0
    narr = [_ME0(timestamp=1.0, speaker="", text="她说今天早点睡") for _ in range(8)]
    assert detect_pronouns(narr, USER_SPEAKERS) == {"user": None, "ai": None}, \
        "叙事体语料没有说话人标记，判不出来才对——但它该走'去问用户'，不是硬判"

    #      **丢主语的接续要挡一下**（2026.08.02 验收打回）：中性档第一句曾经渲染成
    #      "——写下的东西都在这里"，谁写下的没了。手写的中性写法一样会丢主语，
    #      跟机械删词的结果没差别，而它落在人格文件第一句、不变量层、每轮都在。
    #      **这条断言替代不了人眼通读**（中文通不通机器判不了），它只挡住这一类
    #      最常见的形态——破折号/分号后面直接跟动词
    for tag, md_x in (("选他", md_he), ("选她", md_she), ("未知", md_neutral)):
        for bad in ("——写下", "；写下", "——说过", "；说过", "——提到", "；提到"):
            assert bad not in md_x, \
                f"[{tag}] 这句丢了主语（{bad!r}）：中性写法也会丢主语，不是只有机械删词会"

    # 8a4.【靶心：中性写法一条都不许漏】机械删词会写出半通不通的中文，且不报错，
    #      所以中性写法是逐条手写的——那就必须有东西守着"每条模板都有对应的手写形态"。
    templates = [ATTRIBUTION_PREFIX, NOTE_LEAD]
    templates += [v for _, _, v in PROTOCOL_DEFAULTS.values()]
    templates += [lbl for _, lbl, _ in PROTOCOL_DEFAULTS.values()]
    for q in QUESTIONS:
        templates += [q.label, q.text]
        for v in (q.options or {}).values():
            templates += [v[0], v[1]]
    for t in [t for t in templates if t and _SLOT_RE.search(t)]:
        assert t in NEUTRAL_FORMS, f"这条模板带人称槽却没有中性写法：{t!r}"
        assert not _SLOT_RE.search(NEUTRAL_FORMS[t]), \
            f"中性写法里还留着槽位，等于没写：{NEUTRAL_FORMS[t]!r}"
        assert "它" not in NEUTRAL_FORMS[t], f"中性写法里塞了「它」：{NEUTRAL_FORMS[t]!r}"
        assert not any(p in NEUTRAL_FORMS[t] for p in PRONOUN_CHOICES), \
            f"中性写法里塞了默认的他/她：{NEUTRAL_FORMS[t]!r}"

    # 8a5.【靶心：语料侧判定不许猜】第三人称在两个人的对话里多半指的是别人，
    #      所以样本少、或者两个词旗鼓相当时**必须返回 None 去问用户**，不许硬判。
    from memory_import import MemoryEntry as _ME
    #      **陌生说话人标签＝分不清谁是谁，必须 None**（2026.08.02 第二次返工）：
    #      chatlog 翻译器出的是日志里的真名（"小明""星回"），两边都不在已知名单里。
    #      上一版拿"在名单里＝用户、其余全算 AI"硬分，于是**用户谈论自己 AI 的话被
    #      读成"AI 在谈论用户"**，判出来的人称正好是反的，还一个字都不问。
    #      这一档比"判不出来"糟得多：判不出只是多问一句，判错了没有任何人会知道。
    chatlog_one = [_ME(timestamp=1.0 + i, speaker="小明", text="他说得对，他记性好")
                   for i in range(6)]
    assert detect_pronouns(chatlog_one, USER_SPEAKERS) == {"user": None, "ai": None}, \
        "陌生说话人标签下必须判不出来——硬分会把'用户在说他的 AI'读成'AI 在说用户'"
    chatlog_two = chatlog_one + [_ME(timestamp=9.0 + i, speaker="星回", text="她今天早点睡吧")
                                 for i in range(6)]
    assert detect_pronouns(chatlog_two, USER_SPEAKERS) == {"user": None, "ai": None}, \
        "两个都是陌生名字时更不能猜——谁是用户谁是 AI 语料里根本没写"
    #      认得出**任意一侧**的标签就能分：两个翻译器归一成 user/assistant，
    #      只出现其中一个（另一侧是真名）也照样分得清
    known_both = ([_ME(timestamp=1.0 + i, speaker="assistant", text="她今天早点睡")
                   for i in range(6)]
                  + [_ME(timestamp=9.0 + i, speaker="user", text="他说得对") for i in range(6)])
    assert detect_pronouns(known_both, USER_SPEAKERS) == {"user": "她", "ai": "他"}, \
        "两边都是已知标签，这是 ChatGPT/Claude 导出的常态，必须判得出来"
    known_ai_only = ([_ME(timestamp=1.0 + i, speaker="assistant", text="她今天早点睡")
                      for i in range(6)]
                     + [_ME(timestamp=9.0 + i, speaker="小明", text="他说得对") for i in range(6)])
    assert detect_pronouns(known_ai_only, USER_SPEAKERS) == {"user": "她", "ai": "他"}, \
        "只认得 AI 侧标签时，剩下那个就是用户——这一档不该退化成判不出来"
    #      单侧有话、另一侧没话：判得出的那侧照判，判不出的那侧必须是 None——
    #      **两侧各判各的，不许拿一侧的结论去补另一侧**。
    #      这里说话人是已知的 AI 标签（"ai"），所以分得清侧；跟上面 chatlog 那档
    #      在"只有一侧说话"上同构，**区别正是标签认不认得**——当初没区分这一点，
    #      正是上一版把陌生名字也硬分了的原因
    strong = [_ME(timestamp=1.0, speaker="ai", text="她今天早点睡")] * 6
    one_side = detect_pronouns(strong, user_speakers={"me"})
    assert one_side["user"] == "她", f"AI 说了六次「她」，用户侧该判得出：{one_side}"
    assert one_side["ai"] is None, \
        f"用户一句话都没说，AI 侧无从判定，必须是 None：{one_side}"
    mixed = ([_ME(timestamp=1.0, speaker="ai", text="她今天早点睡")] * 5
             + [_ME(timestamp=2.0, speaker="me", text="他说得对")] * 5)
    got = detect_pronouns(mixed, user_speakers={"me"})
    assert got["user"] == "她" and got["ai"] == "他", f"频次占优时该判得出来：{got}"
    thin = [_ME(timestamp=1.0, speaker="ai", text="她好")] * 2
    assert detect_pronouns(thin, user_speakers={"me"})["user"] is None, \
        f"样本太少必须返回 None 去问用户，不许拿两三次出现就下结论"
    tie = ([_ME(timestamp=1.0, speaker="ai", text="她 他 她 他")] * 3)
    assert detect_pronouns(tie, user_speakers={"me"})["user"] is None, \
        "两个词旗鼓相当时必须判不出来，不许挑一个"

    # 8d.【靶心：记忆库的时间边界要告诉模型】真机验证的第二个结论：模型说"后来没有
    #     继续展开"，而那件事根本不在语料里（发生在另一个客户端）——**在它能看见的
    #     世界里那句话是真的**。真缺陷是它不知道自己的记忆止于哪一天，于是把
    #     "我的记录里没有"说成了"没发生过"（差别在 authority，不在语气）。
    with tempfile.TemporaryDirectory() as td:
        paths_cov = write_bundle(td, p5, client="claude-code", confirmed=True, entries=ents)
        cov_md = paths_cov["persona"].read_text(encoding="utf-8")
        assert "记忆库覆盖范围" in cov_md, \
            "出货的人格文件没告诉模型记忆库覆盖到哪天——它就说不出'我的记录到此为止'"
        #    日期要是真的，不是模板占位
        days_in = sorted(datetime.fromtimestamp(e.timestamp).strftime("%Y-%m-%d")
                         for e in ents if getattr(e, "timestamp", None))
        assert days_in[0] in cov_md and days_in[-1] in cov_md, \
            f"覆盖区间跟语料对不上（语料 {days_in[0]}~{days_in[-1]}）：{cov_md[-200:]!r}"
        #    【变异靶心：{end} 是提示不是断言】（任务卡「人格文件里的死数字会过期」，
        #    维护者拍板方案 C）。{end} 从出货那刻起就开始过期（append_record 会持续
        #    写入），内测用户实测撞上"人格文件说覆盖到 07-31、语料里已有 08-02"——
        #    治假否定的字段自己成了假否定的来源。三条一起钉：
        #    ① 过期提示与"以检索层为准"必须在——把措辞改回死断言必红；
        #    ② 日期本身必须还在（方案 A 被否：不写日期，模型不调工具就不知道边界）；
        #    ③ 授权模型拿 {end} 说"没有"的旧句式一个不许剩。
        assert "已经过期" in cov_md and "检索层" in cov_md, \
            "覆盖区间那条没带过期提示——{end} 出货即过期，写死就是在授权假否定"
        for stale in ("这个范围之外的事你没有记录", f"我的记录到 {days_in[-1]} 为止"):
            assert stale not in cov_md, f"人格文件里还留着没有余地的断言：{stale!r}"
        #    验收判据 1 的完整形态：ship 之后 latent_append 写一条**晚于 {end}** 的
        #    记录，重读人格文件——文本不该（也不会）变，但它说的话必须仍然成立：
        #    有过期提示兜着，晚于 {end} 的这条记忆不会被人格文件授权否定
        from memory_retrieval import append_record
        later = datetime.strptime(days_in[-1], "%Y-%m-%d").timestamp() + 86400 * 3
        append_record(paths_cov["memory_dir"], "她把琴修好了", "已兑现", now=later)
        cov_after = paths_cov["persona"].read_text(encoding="utf-8")
        assert "已经过期" in cov_after and "为准" in cov_after, \
            "写回晚于 {end} 的记录之后，人格文件里没有任何一句给这条记忆留余地"
    #     **问不出来就不写**：宁可没有这条，也不能给一个假边界——模型会照着假边界
    #     去否定真事，比不给更糟
    assert corpus_coverage(None, entries=[]) is None, "问不出日期时不该编一个区间出来"
    with tempfile.TemporaryDirectory() as td:
        bare = Path(td) / "memory" / "timeline"
        bare.mkdir(parents=True)
        (bare / "window_01.md").write_text("## 没有日期的窗口\n随便写点。", encoding="utf-8")
        assert corpus_coverage(Path(td) / "memory") is None, \
            "文件名没日期时该返回 None——不许退 mtime，那是全错且整齐地错的假边界"
    #     **文件名日期那条路要有断言真走过一遍**（2026.08.02 补，审查侧留的缺口）：
    #     上面两条都只钉"问不出来时返回 None"，于是把 `_FILE_DATE_RE` 改坏，全量
    #     自检照样绿。而它是 `--corpus`（拿现成语料目录、没有 entries）那条路
    #     **唯一的**日期来源，坏了就是静默不写覆盖区间——症状恰好是覆盖区间这一单
    #     要修的那个：模型不知道自己的记忆止于哪天，于是替看不见的那段时间下结论。
    #     混一个不带日期的文件进去，确认它被跳过而不是把整条路带崩。
    with tempfile.TemporaryDirectory() as td:
        dated = Path(td) / "memory" / "timeline"
        dated.mkdir(parents=True)
        for name in ("2026-03-05_window_01.md", "2026-07-19_window_02.md",
                     "没有日期的_window_03.md"):
            (dated / name).write_text("## 一段\n随便写点。", encoding="utf-8")
        span = corpus_coverage(Path(td) / "memory")
        assert span == ("2026-03-05", "2026-07-19"), \
            f"文件名日期没被解析出来（--corpus 那条路唯一的日期来源）：{span}"

    # 8g.【靶心：pick 题只许挑、不许写】真机里执行者自己创作了一句放进候选并被
    #     采用，落点是最终约定——人格文件里象征意味最重的那一格。两道一起钉：
    #     ①问卷 prompt 里这条纪律的措辞不许飘（它是唯一到得了执行者手上的地方）；
    #     ②`unsourced_picks` 真能把查无出处的答案挑出来，且**不误伤真摘录**。
    for must in ("只许挑，不许写", "不要润色", "找不到就如实说找不到"):
        assert must in export_llm_prompt(QUESTIONS), \
            f"问卷 prompt 里丢了 pick 纪律的措辞：{must!r}——那是唯一到得了执行者手上的地方"
    with tempfile.TemporaryDirectory() as td:
        cp = Path(td) / "memory"
        cp.mkdir()
        #    称呼那两段**刻意分散在语料两头、顺序还相反**：整段比对必然查无出处，
        #    只有分句之后才各自找得到。fixture 要是把两段挨着写、顺序也一样，
        #    整段就成了原样子串，`parts` 那一层被拿掉照样绿（返工前正是这样）
        (cp / "w01.md").write_text(
            "我叫她“小满”，一直这么叫。\n"
            "那天她说“到家了发一句”，我说好。\n"
            "很久以后她才说，她叫我“老陈”是因为那部电影。",
            encoding="utf-8")
        qs_pick = QUESTIONS
        #    真摘录，**但标点跟语料不一样**：语料是全角弯引号 + 中文逗号，
        #    答案给的是直引号 + 末尾句号。归一化那一层被拿掉就会误伤这条。
        #    （返工补：原 fixture 的答案是语料的原样子串，一个标点都没改过，
        #    于是 `_normalize_for_lookup` 改成 `return text` 照样绿——断言声称在守
        #    归一化，实际上什么都没守。**fixture 跟注释说的不是一回事**，
        #    同"fixture 的规模也是一种伪影"是同一族毛病。）
        ok = unsourced_picks(qs_pick, {"closing_pick": {"pick": '"到家了发一句".'}}, cp)
        assert not ok, f"改过标点的真摘录被判成自造了（归一化没做够）：{ok}"
        #    模型自己写的一句好听的：查无出处，必须被挑出来
        bad = unsourced_picks(
            qs_pick, {"closing_pick": {"pick": "你来，我就在。"}}, cp)
        assert [q for q, _ in bad] == ["closing_pick"], \
            f"自造的最终约定没被标出来——流程就分不清语料里的真话和模型写的漂亮话：{bad}"
        #    分句比对：称呼这类答案是多段拼起来的，**只要有一段能在语料里找到就不报**。
        #    整段直接比对必然查无出处（拼接顺序、连接符都是我们自己拼的），
        #    这条是 `parts` 那一层唯一的用武之地，原先没有断言走过它
        assert not unsourced_picks(
            qs_pick, {"naming_pick": {"pick": "她叫我“老陈”；我叫她“小满”"}}, cp), \
            "多段拼接的称呼被整段比对判成自造了——parts 分句那一层没生效"
        #    只查 pick 题：**答案带 pick 字段、但题本身不是 pick 题**，照样不该卷进来。
        #    （返工补：原先传的是 {"keys": "A"}，压根没有 pick 字段，走到空串就
        #    continue 了——它挡的是"答案没有 pick 字段"，不是"这题不是 pick 题"，
        #    把 `q.kind != "pick"` 删掉照样绿。）
        assert not unsourced_picks(
            qs_pick, {"disagree": {"pick": "一句语料里绝对没有的话"}}, cp), \
            "非 pick 题参与了出处核对——选择题的内容本来就是我们写的，核对它没有意义"
        #    **只标注不硬拒**：这里只返回清单，不抛异常——用户自己给的一句
        #    语料里当然没有，我们没资格替他否掉
    #     ③**护栏要真的到得了执行者手上**：函数正确但没人调用，等于没有。
    #     走 CLI 真进程，断言标注真的印在 --step answers 的输出里。
    #     （返工补，审查侧的对抗变异：把 `_step_answers` 里那行
    #     `_report_unsourced_picks(...)` 删掉，函数还在、自检全绿，而护栏的全部
    #     价值就是让执行者看见它。这是本项目连着学过三次的同一个教训——
    #     显式 --client 被状态吃掉／`not exists` 恒真／函数被钉住但命令没被钉住。）
    import subprocess          # 同 8b：走真进程，函数级断言钉不住"命令有没有调它"
    with tempfile.TemporaryDirectory() as td:
        cp2 = Path(td) / "corpus"
        cp2.mkdir()
        (cp2 / "w01.md").write_text("那天她说“到家了发一句”，我说好。", encoding="utf-8")
        out2 = Path(td) / "out"

        def run_pick(*extra):
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                "--out", str(out2), *extra], cwd=td,
                               capture_output=True, text=True, encoding="utf-8")
            assert r.returncode == 0, f"CLI 跑挂了：{extra}\n{r.stdout}\n{r.stderr}"
            return r.stdout

        run_pick("--corpus", str(cp2))
        made_up = "你来，我就在。"
        got2 = run_pick("--corpus", str(cp2), "--step", "answers",
                        "--answers-json", json.dumps({"closing_pick": {"pick": made_up}},
                                                     ensure_ascii=False))
        assert "【查无出处】" in got2 and made_up in got2, \
            "带 --corpus 跑 answers 时，查无出处的标注没印出来——护栏到不了执行者手上就等于没有"
        #    不给 --corpus 就不查，**也不假装查过**：没有语料可比对时沉默是诚实的
        got3 = run_pick("--step", "answers",
                        "--answers-json", json.dumps({"closing_pick": {"pick": made_up}},
                                                     ensure_ascii=False))
        assert "【查无出处】" not in got3, \
            "没给 --corpus 却报了查无出处——那是在拿空语料当'查过了'"
        #    这一条**只挡得住两层一起塌**：调用方的 `if not corpus_dir: return` 与
        #    `unsourced_picks` 里的 `corpus.exists()` 早返回互为兜底，单点变异会被
        #    另一层吸收。如实记下来——不假装它是一条单点靶。
        #    **另一条路径也要钉**：`--answers <答案清单文件>`（走 `问卷prompt.txt`
        #    的用户回来时用的就是它）。两条路径各有一次调用，只钉一条的话，
        #    另一条被删掉照样全绿——返工前就是这样，⑦号变异是绿的。
        #    题号从 CLI 自己吐的问卷里取，不手写——手写的题号会跟问卷顺序脱节
        qs_j = json.loads(run_pick("--corpus", str(cp2), "--step", "questionnaire", "--json"))
        qids_j = [q["qid"] for q in qs_j["questions"]]
        sheet = Path(td) / "answers.txt"
        sheet.write_text(f"{qids_j.index('closing_pick') + 1}. {made_up}\n",
                         encoding="utf-8")
        got4 = run_pick("--corpus", str(cp2), "--step", "answers",
                        "--answers", str(sheet))
        assert "【查无出处】" in got4 and made_up in got4, \
            "走答案清单文件那条路时标注没印出来——两条路径要各钉各的"

    # 8e.【靶心：止血纪律必须在人格文件里，且不动已验证的句子】
    #     护栏挂在工具返回值上，模型一绕过工具（grep、直接读文件）就一条都不生效，
    #     而绕过恰恰发生在检索失败、最需要护栏的时候。人格文件是唯一覆盖
    #     "不管你用什么方式查"的那一层。
    p_conv = Persona("partner")
    fill_protocol_defaults(p_conv)
    conv = next(f.value for f in p_conv.fields if f.id == RETRIEVAL_CONVENTION_FIELD)
    for must in ("片段，不是全部", "grep", "不要说“没发生过”", "我的记录里没有"):
        assert must in conv, f"检索约定里缺行为层这条：{must}"
    for must in ("冲突即更正", "latent_correct", "latent_append", "不要只 append",
                 "喜欢的电影", "不得为了写入新项而 correct 旧项", "不要预先给每句话分类"):
        assert must in conv, f"检索约定里缺单值冲突／并列事实边界：{must}"
    #     **只做加法**：三轮真机验证过的那两段一字不许动，逐字钉死（黄金串写在
    #     断言里，不引用常量——引用常量的话改了常量断言跟着变，等于没钉）
    GOLDEN_RETRIEVAL = (
        "对方提到过去发生过的事、某个约定、某个日期／地点／称呼／人名，或者你对"
        "细节拿不准时，先用记忆检索工具（latent_search）查一遍再开口；不要在查"
        "之前说“我不记得”。查完自然接上话，不用报告自己搜过。")
    assert GOLDEN_RETRIEVAL in pron_md, \
        "检索约定那段的措辞被改了——它是三轮真机验证过的基准写法，一字不许动（其余向它对齐）"
    assert pron_md.count("你") >= 2, "「你」几乎不出现，说明这条闸被'全删掉'绕过去了"

    # 8a1.【靶心：关系状态不归「我是谁」】真实样本里「我是谁」整节只有"当前关系状态"
    #      一条，而那条讲的是这段关系此刻怎么样，不是 AI 的身份。
    state_q = next(q for q in QUESTIONS if q.qid == "state_now")
    assert state_q.section == "opening", "当前关系状态该归开篇（关系确认那一节），不是 AI 的身份"
    #      渲染里它必须落在开篇那一节之内：取开篇标题到下一节标题之间的正文来看
    open_body = pron_md.split(SECTIONS["opening"], 1)[1].split(SECTIONS["user"], 1)[0]
    assert "当前关系状态" in open_body, "当前关系状态没渲染进开篇那一节"

    # 8a2.【靶心：称呼不许重复拼接】同一份真实样本里的第二条：多选几个称呼候选，
    #      共享的那半会重复出现（"她叫你'哥哥'，你叫她'阿岸'；她叫你'星回'，你叫她'阿岸'"）。
    #      整行去重没用——两行确实不同，要按分句去重。
    p_nm = Persona("partner")
    fill_protocol_defaults(p_nm)
    qs_nm = questions_for(coverage_report(p_nm), has_corpus=True)
    nm_q = next(q for q in qs_nm if q.qid == "naming_pick")
    apply_answers(p_nm, qs_nm, {"naming_pick": {"pick":
                  "她叫你“哥哥”，你叫她“阿岸”；她叫你“星回”，你叫她“阿岸”"}})
    nm_value = next(f.value for f in p_nm.fields if f.id == nm_q.field_id)
    assert nm_value.count("你叫她“阿岸”") == 1, \
        f"同一侧称呼被重复拼进人格文件：{nm_value!r}"
    #      两个不同的称呼都得留下——去重不能把内容也吃掉
    assert "哥哥" in nm_value and "星回" in nm_value, f"去重把真内容删了：{nm_value!r}"

    # 8b.【靶心：任务书不许泄漏进人格文件】外部（第三方 AI 走流程时）发现的缺陷：
    #     milestone_kinds 的选项指引是"从语料提取：……"——给模型的提取任务书，
    #     却以普通字段草稿的身份进了确认关卡，用户按 y 就写进不变量层。
    #     **走真实的 CLI 全流程**（questionnaire → answers → confirm 全 keep → ship），
    #     零语料冷启动即可复现，产出的人格文件里不得出现任务书特征串。
    #     只断言"字段不存在"是不够的——真正会伤到用户的是**写进文件的那段文本**，
    #     所以靶子取产出文件的正文。
    import subprocess
    with tempfile.TemporaryDirectory() as td:
        def run_cli(*extra):
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                "--out", td, *extra], cwd=td,
                               capture_output=True, text=True, encoding="utf-8")
            assert r.returncode == 0, f"CLI 跑挂了：{extra}\n{r.stdout}\n{r.stderr}"
            return r.stdout
        qs_j = json.loads(run_cli("--step", "questionnaire", "--json"))
        #    每题都答满（milestone_kinds 尽量多选几项，把任务书撑到最容易泄漏的形态）
        answers_j = {}
        for q in qs_j["questions"]:
            if q["options"]:
                answers_j[q["qid"]] = {"keys": "".join(sorted(q["options"])[:3])}
            else:
                answers_j[q["qid"]] = {"pick": "她说“到家了发一句”，我说好。"}
        assert "milestone_kinds" in answers_j, "问卷里没有里程碑类型题，靶子失效"
        #    **这个字面量超过 255 字节**——缺陷二的靶子，短 fixture 抓不到（见下 8c）
        ans_literal = json.dumps(answers_j, ensure_ascii=False)
        assert len(ans_literal.encode("utf-8")) > 255, \
            f"--answers-json 的靶子字面量不够长（{len(ans_literal.encode('utf-8'))} 字节），抓不到长度上限那个洞"
        run_cli("--step", "answers", "--answers-json", ans_literal)
        listed = json.loads(run_cli("--step", "confirm", "--list", "--json"))
        #    任务书走自己的通道，不混进待确认清单
        assert listed["extraction_brief"], "任务书没进 extraction_brief 通道，等于丢了给模型的指令"
        for p in listed["pending"]:
            assert "从语料提取" not in p["value"], \
                f"任务书混进了待确认清单，用户按 y 就写进人格文件：{p['label']}"
        dec_literal = json.dumps({p["key"]: "keep" for p in listed["pending"]},
                                 ensure_ascii=False)
        assert len(dec_literal.encode("utf-8")) > 255, \
            f"--decisions-json 的靶子字面量不够长（{len(dec_literal.encode('utf-8'))} 字节）"
        run_cli("--step", "confirm", "--decisions-json", dec_literal)
        #    8b-2.【靶心：检索路线是个真的选择点，不是文档里的可选升级】（2026.08.02）
        #      三条路里有一条会把私人记忆发到第三方。**变异靶心**：把 _step_ship 里
        #      那道闸换成 `route = state.get("route", ROUTE_DEFAULT)` 时这条必红——
        #      而那正是"没问过"与"选了默认"长得一模一样的那种失效。
        r_norote = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                   "--out", td, "--step", "ship"], cwd=td,
                                  capture_output=True, text=True, encoding="utf-8")
        assert r_norote.returncode != 0 and "检索路线还没选过" in r_norote.stdout + r_norote.stderr, \
            "没选过路线就出货了——那个选择点等于不存在"
        routes_j = json.loads(run_cli("--step", "route", "--json"))
        keys = {r["key"] for r in routes_j["routes"]}
        assert keys == {"zero-dep", "local", "cloud"}, f"三条路线要都念到：{keys}"
        #      每条都必须写清语料去向，而且云端那条必须明说会发给第三方——
        #      含糊掉这句话，用户就是在不知情的前提下做了一个不可逆的决定
        for r in routes_j["routes"]:
            assert r["语料去向"], f"{r['key']} 没写语料去向"
        cloud = next(r for r in routes_j["routes"] if r["key"] == "cloud")
        assert "发到" in cloud["语料去向"] and "服务商" in cloud["语料去向"], \
            "云端那条没把'查询和被检索的内容都会发到那家服务商'说出来"
        assert next(r for r in routes_j["routes"] if r["default"])["key"] == ROUTE_DEFAULT
        run_cli("--step", "route", "--route", "zero-dep")
        ship_out = run_cli("--step", "ship", "--client", "claude-code")
        persona_text = (Path(td) / "CLAUDE.md").read_text(encoding="utf-8")
        assert "从语料提取" not in persona_text, \
            "任务书泄漏进人格文件了——不变量层里坐着一句没有日期、没有原话、没有当下状态的指令"
        #    但指令本身不能就这么丢了：它得从任务书通道出去，第二阶段照着干活
        assert "从语料提取" in ship_out and "不是人格文件内容" in ship_out, \
            "任务书既没进人格文件、也没从任务书通道出来——那是把用户的选择直接扔了"

    # 8c.【靶心：长字面量不许崩】缺陷二。`_load_json_arg` 原先路径优先且不兜
    #     OSError，字面量超过 255 字节直接 File name too long。上面 8b 已用真实规模的
    #     字面量走过一遍 CLI，这里再直接钉住函数本身，两头都堵：
    #     **短 fixture 抓不到这个缺陷——fixture 的规模本身就是一种伪影。**
    long_payload = {f"field:占位{i:02d}": "keep" for i in range(11)}
    long_literal = json.dumps(long_payload, ensure_ascii=False)
    assert len(long_literal.encode("utf-8")) > 255, "靶子本身不够长，测了个寂寞"
    assert _load_json_arg(long_literal) == long_payload, \
        "超过文件名长度上限的 JSON 字面量该照常解析，不该抛 OSError"
    #     真是路径时仍然读文件（别把修法做成"永远不认路径"）
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "dec.json"
        f.write_text(long_literal, encoding="utf-8")
        assert _load_json_arg(str(f)) == long_payload, "传文件路径时该读文件"

    GOLDEN_SESSION = (
        "**会话约定**：新会话开场先调一次 latent_session_start；会话结束前调一次 "
        "latent_thread_close，记下聊到哪、当下状态、有什么没聊完。")
    assert GOLDEN_RETRIEVAL in conv and GOLDEN_SESSION in conv, \
        "改动了三轮真机验证过的措辞——这两段只许在后面加，不许动"

    # 9.【变异靶心：写盘要过确认关卡】未确认时拒绝写用户磁盘
    with tempfile.TemporaryDirectory() as td:
        try:
            write_bundle(td, p5, confirmed=False)
            assert False, "未确认就该拒绝写盘"
        except PermissionError:
            pass
        paths = write_bundle(td, p5, client="claude-code", confirmed=True, entries=ents)
        assert len(paths["corpus_files"]) == 2, "出货时语料要真的落进 memory/"
        assert paths["persona"].name == "CLAUDE.md" and paths["persona"].exists()
        cfg = json.loads(paths["mcp_config"].read_text(encoding="utf-8"))
        assert "memory" in cfg["mcpServers"] and "--corpus" in cfg["mcpServers"]["memory"]["args"]
        #    客户端适配：codex 出 AGENTS.md
        assert write_bundle(td, p5, client="codex", confirmed=True)["persona"].name == "AGENTS.md"
        assert paths.get("contract") is None, "有宿主的档不该塞契约副本（那是 generic 档的活）"
        #    未知客户端明确报错，不静默猜
        try:
            write_bundle(td, p5, client="讯飞星火", confirmed=True)
            assert False, "未知客户端该报错"
        except ValueError:
            pass
        #    状态可续跑
        save_state(td, {"step": "questionnaire", "answered": ["naming_self"]})
        assert load_state(td)["step"] == "questionnaire"
        #    确认到一半存盘，重新载入接得上：状态里只存用户输入（答案+决策），
        #    persona 每步从头重放——重放幂等，状态文件坏了也看得懂改得动
        half_qs = questions_for(coverage_report(Persona("partner")), has_corpus=False)
        # 取一道**会生成字段**的选择题：人称题也是 choice，但它按设计不进字段
        # （2026.08.02 加人称题后这里要挑清楚，否则取到的是个不产草稿的题）
        cq = next(q for q in half_qs if q.kind == "choice" and not q.pronoun_side)
        save_state(td, {"step": "confirm", "has_corpus": False,
                        "answers": {cq.qid: {"keys": "A", "note": ""}},
                        "decisions": {}})
        pa, _ = _rebuild(load_state(td))
        before = pending_confirmations(pa)
        assert any(p.key == f"field:{cq.field_id}" for p in before), "答案重放成了草稿"
        st = load_state(td)
        st["decisions"][f"field:{cq.field_id}"] = "keep"
        save_state(td, st)
        pb, _ = _rebuild(load_state(td))
        assert len(pending_confirmations(pb)) == len(before) - 1, \
            "载入半程状态后，已决策的那条不再待确认"

    # 9b.【变异靶心：generic 档少一样都不算出货】自建前端没有宿主替他注入，
    #     契约副本就是交付的另一半；漏了的话产出目录看着一切正常（人格文件、
    #     记忆库、配置齐全），少的恰好是他唯一必须照做的那部分
    with tempfile.TemporaryDirectory() as td:
        g = write_bundle(td, p5, client="generic", confirmed=True)
        assert g["persona"].name == "persona.md", \
            "generic 档不能沿用宿主专有文件名（CLAUDE.md/AGENTS.md 对自建前端没有意义）"
        assert g["contract"] and g["contract"].exists(), "generic 档必须随货带注入契约副本"
        body = g["contract"].read_text(encoding="utf-8")
        for key in ("逐字", "每轮", "从磁盘重读", "整块连续", "易变内容"):
            assert key in body, f"契约副本里缺了契约五条之一：{key}"
        #    契约源文件缺失时明确报错，不出一份"看着正常、其实少一半"的货
        with tempfile.TemporaryDirectory() as empty:
            try:
                write_bundle(td, p5, client="generic", confirmed=True, contract_base=empty)
                assert False, "契约源文件缺失时不该静默出货"
            except FileNotFoundError:
                pass

    # 9b2.【靶心：走 CLI 真进程，钉住《快速上手》4b 教的那条命令】
    #      **这条是验收打回补的**：9b 直接调 write_bundle(client="generic")、9c 直接调
    #      ship_note("generic")，两条都绕开了 CLI 的 state 覆盖路径——于是"ship 步
    #      显式传的 --client 被状态里的旧值静默吃掉"这个洞，函数级断言全绿地放过去了。
    #      **断言钉住了函数，没钉住用户真会敲的那条命令。** 所以这里起真进程，
    #      按用户文档的顺序跑完四步，最后一步才给 --client generic。
    import subprocess
    with tempfile.TemporaryDirectory() as td:
        def run(*extra):
            # **cwd 故意设成产出目录、不是 src/**：用户不会站在 src/ 里跑第四步，
            # 而"配置里的 server 路径是不是真能用"这件事，只有在别的工作目录下
            # 才分辨得出来——站在 src/ 里跑，裸的相对路径 `mcp_server.py` 也能
            # 蒙混过关（第一版断言就是这么被蒙过去的）
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                "--out", td, *extra], cwd=td,
                               capture_output=True, text=True, encoding="utf-8")
            assert r.returncode == 0, f"CLI 跑挂了：{extra}\n{r.stdout}\n{r.stderr}"
            return r.stdout
        #    第 1 步**不带 --client**（默认 claude-code 落进状态，正是打回单的复现前提）
        qs_json = json.loads(run("--step", "questionnaire", "--json"))
        assert json.loads((Path(td) / "init_state.json").read_text(encoding="utf-8")
                          )["client"] == DEFAULT_CLIENT
        ans = {q["qid"]: ({"keys": sorted(q["options"])[0]} if q["options"]
                          else {"pick": "她说“到家了发一句”，我说好。"})
               for q in qs_json["questions"]}
        run("--step", "answers", "--answers-json", json.dumps(ans, ensure_ascii=False))
        pend = json.loads(run("--step", "confirm", "--list", "--json"))["pending"]
        run("--step", "confirm", "--decisions-json",
            json.dumps({p["key"]: "keep" for p in pend}, ensure_ascii=False))
        #    **先按宿主档出一次货**——这一步是断言构造的关键，不是凑数（二轮验收
        #    打回）：不先出宿主档，产出目录里 CLAUDE.md 本来就不存在，下面那条
        #    `not exists` 就是**恒真**的，看着在守"换档后旧档清掉了"，实际什么都
        #    没守（同 routes 消融那次的自动满足）。先出一次，它才有东西可挡。
        #    检索路线这一步照用户真会敲的顺序走一遍，**故意选云端**：这条路是三条
        #    里唯一会把语料发出本机的，也是唯一需要 key 的，产出物必须扛得住
        run("--step", "route", "--route", "cloud")
        ship_out_r = run("--step", "ship")
        assert (Path(td) / "CLAUDE.md").exists(), "宿主档这一次货本身就没出成，后面的断言无从谈起"
        #    第四件：引导句（任务卡「人格按需读取的引导句」）。三条一起钉：
        #    文件真出了（变异靶心：出货漏产必红）、指向这次出的人格文件且是绝对路径、
        #    长度过得了长度那道闸（⚠ 判据理由 2026.08.03 起是「指针不是第二份人格」，
        #    不再是设置字段容量，见 GUIDANCE_LIMIT 上注）
        gtxt = (Path(td) / GUIDANCE_DOC).read_text(encoding="utf-8").strip()
        assert str((Path(td) / "CLAUDE.md").resolve()) in gtxt, \
            f"引导句没指向这次出货的人格文件：{gtxt}"
        assert len(gtxt) <= GUIDANCE_LIMIT, f"引导句超长（{len(gtxt)} 字）却出货了"
        assert "引导句" in ship_out_r, "出货清单里没列引导句——文档说有、清单没报，就是下一个静默缺口"
        #    选了云端，启动参数就得真的是云端档——选择点不改产出物的话，"选过一次"
        #    只是走了个过场
        cfg_r = json.loads((Path(td) / "mcp-config.json").read_text(encoding="utf-8"))
        cargs = cfg_r["mcpServers"]["memory"]["args"]
        assert "--embed" in cargs and "cloud" in cargs, f"云端档没写进启动参数：{cargs}"
        #    **但 key 一个字都不许进产出目录**：mcp-config.json 会跟着产出目录走，
        #    用户会随手贴给别人；key 只从环境变量读（变异靶心：把 key 写进 args 必红）
        for f in ("mcp-config.json", "init_state.json"):
            blob = (Path(td) / f).read_text(encoding="utf-8")
            assert "MEMORY_EMBED_API_KEY" not in blob and "sk-" not in blob, \
                f"{f} 里出现了 key 相关内容——凭证只走环境变量"
        assert "语料" in ship_out_r and "服务商" in ship_out_r, \
            "出货时没把云端档的语料去向再说一遍"
        #    第 4 步才显式换档——文档教的就是这条
        out = run("--step", "ship", "--client", "generic")
        assert (Path(td) / "persona.md").exists(), \
            "ship 步显式传的 --client generic 被状态里的旧档吃掉了（用户文档 4b 教的正是这条命令）"
        assert (Path(td) / CONTRACT_DOC).exists(), "走 CLI 出的 generic 档没带契约副本"
        #    换档后引导句必须跟着指向新档的人格文件——指着已退役的旧档，
        #    症状同影子副本：贴了指针、读到的却是永不更新的那份
        assert str((Path(td) / "persona.md").resolve()) in \
            (Path(td) / GUIDANCE_DOC).read_text(encoding="utf-8"), \
            "换档后引导句还指着旧档的人格文件"
        #    换档后旧档那份必须退役：留着它日后不会跟着升层／更正更新，作者拼错
        #    那一份的症状恰好是契约第三条违反后的"确认过的更新悄无声息不生效"
        assert not (Path(td) / "CLAUDE.md").exists(), \
            "换档后旧档的人格文件还躺在产出目录里，会变成永不更新的影子副本"
        assert (Path(td) / "CLAUDE.md.bak").exists(), \
            "旧档该退役成 .bak 留痕，不是无声删掉——写用户磁盘一律保守"
        assert "已退役" in out, "退役了旧档却不吭声，用户不知道目录里那份为什么变了名"
        assert "你前端的责任" in out and "起会话" not in out, \
            "走 CLI 时收尾话术仍是宿主档那套"
        assert "已切换" in out, "换档必须说出来，不许静默改用户的档"
        #    换档要写回状态：下次不带 --client 续跑，仍然是 generic
        assert json.loads((Path(td) / "init_state.json").read_text(encoding="utf-8")
                          )["client"] == "generic", "切换后的档没写回状态"
        #    MCP 配置里的 server 路径必须是能直接用的绝对路径——自建前端作者没有
        #    《快速上手》那条 claude mcp add 绝对路径示范，配置给什么他就用什么
        cfg = json.loads((Path(td) / "mcp-config.json").read_text(encoding="utf-8"))
        server_arg = Path(cfg["mcpServers"]["memory"]["args"][0])
        assert server_arg.is_absolute() and server_arg.exists(), \
            f"mcp-config.json 里的 server 路径不可直接使用：{server_arg}"
        #    **样板说明必须在文件里**（2026.08.01 维护者拍板"不改名、加说明"）：
        #    要防的是"有人改了这个文件、发现不生效"，而那件事必须先打开文件才会
        #    发生——所以说明写在文件里，正好在他需要的那一刻被读到。
        #    三条一起钉：键在、话说到点上、**结构没被这一行搞坏**（它是个真键，
        #    不是注释；把配置本体挤歪了就是拿一句提醒换一个真 bug）。
        assert CONFIG_NOTE_KEY in cfg, \
            "mcp-config.json 里没有那句样板说明——用户打开它改，不会知道改了不生效"
        for must in ("不会被任何客户端自动读取", "改这个文件不生效"):
            assert must in cfg[CONFIG_NOTE_KEY], \
                f"样板说明没说到点上，缺：{must!r}"
        #    ⚠ command 那一行的判据在 73b) 组（**它在这台机器上能不能起进程**）；
        #    这里只钉结构：说明是个真键，别把配置本体挤歪了。原来这一行顺手比的是
        #    `command == "python"`——**那正好把缺陷本身写成了断言**，2026.08.05 外部
        #    实测第 2 条撞的就是它（只装 python3 的机器照抄必挂）。
        assert set(cfg) == {CONFIG_NOTE_KEY, "mcpServers"} and \
            set(cfg["mcpServers"]["memory"]) == {"command", "args"}, \
            f"那一行说明把配置结构挤歪了：{sorted(cfg)}"
        #    **说明本身不许把 key 漏出去**：它是随产出目录走的文件，同 key 那条纪律
        assert "sk-" not in cfg[CONFIG_NOTE_KEY]
        #    **同档再出一次货不许自己退自己**：重跑 ship 是常规操作（补了语料、
        #    改了一条就再出一次），而退役发生在写盘之后——判据要是漏了"跟这次
        #    同名就不动"，第二次 ship 会把刚写好的人格文件改成 .bak，产出目录
        #    里一份人格文件都不剩，而命令是成功返回的
        out_again = run("--step", "ship")
        assert (Path(td) / "persona.md").exists(), "同档重跑 ship 把自己刚写的人格文件退役了"
        assert not (Path(td) / "persona.md.bak").exists(), "同档重跑不该产生 .bak"
        assert "已退役" not in out_again, "没换档却报退役"

    # 9b4.【靶心：覆盖自己之前那份人格文件前必须备份，并且说出来】
    #      2026.08.01 维护者在场时实测撞出来的真缺陷：老用户升级只要重跑一次 ship，
    #      人格文件就按新版协议层重放一遍——**手改过的段落不在 state 里，
    #      重放等于删掉它，而且原来一声不吭、连 .bak 都不留**。
    #      分量在于《给AI的引导指南》明写着"人格文件是 TA 的，随时可以自己改"：
    #      **我们鼓励了他改，又在升级时悄悄替他删掉**；而换档退役旧档那条我们
    #      留了 .bak，自己覆盖自己反而不留，前后不一致。
    #      **靶子走 CLI 真进程**——函数级断言钉不住"用户真敲的那条命令会不会提醒他"。
    with tempfile.TemporaryDirectory() as td:
        def ship_run(*extra):
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                "--out", td, *extra], cwd=td,
                               capture_output=True, text=True, encoding="utf-8")
            assert r.returncode == 0, f"CLI 跑挂了：{extra}\n{r.stdout}\n{r.stderr}"
            return r.stdout
        qs_up = json.loads(ship_run("--step", "questionnaire", "--json"))["questions"]
        ship_run("--step", "answers", "--answers-json", json.dumps(
            {q["qid"]: ({"keys": "".join(sorted(q["options"]))} if q["options"]
                        else {"pick": "说好了就算数。"}) for q in qs_up}, ensure_ascii=False))
        pend_up = json.loads(ship_run("--step", "confirm", "--list", "--json"))["pending"]
        ship_run("--step", "confirm", "--decisions-json",
                 json.dumps({p_["key"]: "keep" for p_ in pend_up}, ensure_ascii=False))
        ship_run("--step", "route", "--route", ROUTE_DEFAULT)
        first = ship_run("--step", "ship", "--client", "claude-code")
        md_path = Path(td) / "CLAUDE.md"
        #      第一次出货：磁盘上本来没有旧文件，**不该报"被改过"**，也不该留 .bak
        assert "已备份成" not in first, "第一次出货就报'原来那份不一样'——那是恒真的噪音"
        assert not md_path.with_name("CLAUDE.md.bak").exists(), "第一次出货不该产生 .bak"
        #      同档原样重跑：内容一致，同样不该报、不该备份
        again = ship_run("--step", "ship", "--client", "claude-code")
        assert "已备份成" not in again and not md_path.with_name("CLAUDE.md.bak").exists(), \
            "内容没变也报覆盖，用户会学会忽略这条提醒"
        #      **用户手改之后再重跑**：必须备份 + 必须说出来
        hand = md_path.read_text(encoding="utf-8") + "\n\n## 我自己加的一节\n她怕黑，睡觉留一盏灯。\n"
        md_path.write_text(hand, encoding="utf-8")
        upgraded = ship_run("--step", "ship", "--client", "claude-code")
        bak_path = md_path.with_name("CLAUDE.md.bak")
        assert bak_path.exists(), \
            "手改过的人格文件被直接覆盖、连备份都没有——升级会静默吃掉用户自己写的内容"
        assert bak_path.read_text(encoding="utf-8") == hand, \
            "备份的不是手改前那份，等于备份了个寂寞"
        assert "她怕黑" not in md_path.read_text(encoding="utf-8"), \
            "这条断言的前提：重放本来就带不动手改内容（带得动就不需要这套机制了）"
        for must in ("已备份成", "手改的段落不会被自动带过来", "贴回"):
            assert must in upgraded, f"覆盖了手改内容却没说清怎么办，缺：{must!r}"

    # 9b3.【靶心：不许动不是我们出的文件】三轮验收打回的洞：退役逻辑原本按
    #      "别的档的文件名"匹配，而 CLAUDE.md 是 Claude Code 的项目约定文件，
    #      自建前端作者目录里躺一份他自己写的太正常了。全程只用 generic、从没换过
    #      档的用户，一跑出货就被我们把项目指令改了名 + 收到一句"换档后旧档不再
    #      更新"的错提示。这里预置一份**不是我们出的** CLAUDE.md，走完整流程，
    #      要求它**逐字原样还在**（只断言 exists 不够——改了内容同样是动了人家的文件）。
    with tempfile.TemporaryDirectory() as td:
        mine = Path(td) / "CLAUDE.md"
        mine_text = "# 我自己项目的 CLAUDE.md\n# 这是我给 Claude Code 写的项目指令，别动它。\n"
        mine.write_text(mine_text, encoding="utf-8")

        def run_g(*extra):
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                "--out", td, "--client", "generic", *extra], cwd=td,
                               capture_output=True, text=True, encoding="utf-8")
            assert r.returncode == 0, f"CLI 跑挂了：{extra}\n{r.stdout}\n{r.stderr}"
            return r.stdout
        qs2 = json.loads(run_g("--step", "questionnaire", "--json"))
        ans2 = {q["qid"]: ({"keys": sorted(q["options"])[0]} if q["options"]
                           else {"pick": "她说“到家了发一句”，我说好。"})
                for q in qs2["questions"]}
        run_g("--step", "answers", "--answers-json", json.dumps(ans2, ensure_ascii=False))
        pend2 = json.loads(run_g("--step", "confirm", "--list", "--json"))["pending"]
        run_g("--step", "confirm", "--decisions-json",
              json.dumps({p["key"]: "keep" for p in pend2}, ensure_ascii=False))
        run_g("--step", "route", "--route", "zero-dep")
        out2 = run_g("--step", "ship")
        assert mine.exists() and mine.read_text(encoding="utf-8") == mine_text, \
            "把用户自己的 CLAUDE.md 动了——目录里有这个文件名不等于那文件是我们写的"
        assert not (Path(td) / "CLAUDE.md.bak").exists(), \
            "给用户自己的 CLAUDE.md 生了个 .bak，等于把他的项目指令挪走了"
        assert "已退役" not in out2, \
            "从没换过档却报'换档后旧档不再更新'，用户会以为自己哪步选错了"
        assert (Path(td) / "persona.md").exists(), "generic 档自己的货还是得出"

    # 9c.【变异靶心：ship 话术不许串档】generic 档照抄“从产出目录起会话”，等于告诉
    #     自建前端作者“你什么都不用做”——而他恰恰是唯一必须自己动手注入的人
    generic_note, host_note = ship_note("generic"), ship_note("claude-code")
    assert "你前端的责任" in generic_note and "起会话" not in generic_note, \
        "generic 档的收尾提示必须说清注入是作者自己的责任，不能沿用宿主档话术"
    light = generic_note.index(GUIDANCE_DOC)
    fallback = generic_note.index(CONTRACT_DOC)
    assert light < fallback and "验证不通过或 App 读不到本地文件时" in generic_note, \
        "generic 档必须先给引导句轻量路线，验证失败或读不到本地文件时才退到注入契约"
    assert "起会话" in host_note and "你前端的责任" not in host_note

    # 9d.【三条路线各自落到启动参数上】（2026.08.02）零依赖档不加任何参数（默认就是
    #     它），本地档加 --embed，云端档再加 --embed-provider cloud。**云端档的
    #     endpoint / 模型 / key 一个都不进这份配置**——它们全走环境变量，因为这个
    #     文件会跟着产出目录被随手分享出去。
    snips = {r: json.loads(mcp_config_snippet("/x/mcp_server.py", "/x/memory",
                                              "/x/threads.jsonl", route=r)
                           )["mcpServers"]["memory"]["args"] for r in RETRIEVAL_ROUTES}
    assert "--embed" not in snips["zero-dep"], "零依赖档不该带 --embed"
    assert snips["local"][-1:] == ["--embed"], f"本地档该只多一个 --embed：{snips['local']}"
    assert snips["cloud"][-3:] == ["--embed", "--embed-provider", "cloud"], \
        f"云端档启动参数不对：{snips['cloud']}"
    for r, a in snips.items():
        assert not any("MEMORY_EMBED" in x or "http" in x for x in a), \
            f"{r} 档把 endpoint/环境变量名写进了 MCP 配置：{a}"

    # 9e2.【变异靶心：所有路径都要解成绝对路径，不是只有 server】走查台账 08-03 第六条。
    #      原先只有 `paths[0]` 走了 `resolve()`，`--corpus` 与 `--threads` 是裸 `str()`
    #      ——而同一份配置里那句 `CONFIG_NOTE_MACHINE_BOUND` 写着"写死了这台机器的
    #      绝对路径"，**三分之一是真的**。用户拿相对路径起 `--out`（`--out .` 这种）
    #      时配置里就落相对值，而客户端起 server 的工作目录跟他当时 cd 的地方无关。
    #      ⚠ **靶子必须喂相对路径**：走 `write_bundle(td, …)` 那条路时 `td` 本来就是
    #      绝对的，**所有值不 resolve 也全是绝对路径，断言恒真**——第一版就是这么写的，
    #      变异不红才发现。**"断言在缺陷面前全绿"跟"没有这条断言"是一回事。**
    rel_args = json.loads(mcp_config_snippet(
        "mcp_server.py", "memory", "threads.jsonl",
        index_dir="bundle/memory/index"))["mcpServers"]["memory"]["args"]
    for label, value in (("server", rel_args[0]), ("--corpus", rel_args[2]),
                         ("--index-dir", rel_args[4]), ("--threads", rel_args[6])):
        assert Path(value).is_absolute(), \
            f"mcp-config.json 里 {label} 落的是相对路径，换个工作目录就指不对：{value}"

    # 9e3.【变异靶心：没给就落默认东八区，探到的宿主时区永远不进 args】（任务卡「写回时区与
    #      跨日归窗」2026.08.04 真机事故）。**这一条守的不是"有没有这个参数"，是
    #      "这个值从哪来"**：探测探的是跑初始化那台机器的本地时区，而要的是记忆
    #      所有者本人的时区。事故现场那台 VPS 探出来正好是 `Etc/UTC`——自动填进去，
    #      `--doctor` 就会给那个缺陷发一张 `✓ 时区：Etc/UTC` 的合格证。
    #      ⚠ 靶子必须**把探测结果按住成一个真实的值**，否则在探不出时区的机器上
    #      这条恒真（"断言在缺陷面前全绿"跟没有这条断言是一回事）。
    _real_detect = detect_local_timezone
    globals()["detect_local_timezone"] = lambda: "Etc/UTC"
    try:
        no_tz = json.loads(mcp_config_snippet("/x/mcp_server.py", "/x/memory",
                                              "/x/threads.jsonl"))
        n_args = no_tz["mcpServers"]["memory"]["args"]
        assert n_args[-2:] == ["--timezone", DEFAULT_TIMEZONE], \
            f"没人给时区时，落进 args 的该是默认值 {DEFAULT_TIMEZONE}：{n_args}"
        assert "Etc/UTC" not in " ".join(n_args), \
            "探测到的宿主时区任何时候都不许进 args——那是给缺陷发合格证"
        assert "默认值" in no_tz[CONFIG_NOTE_KEY] and "Etc/UTC" in no_tz[CONFIG_NOTE_KEY], \
            "说明里必须写明这是默认值、且如实说这台机器探到的是别的（让人有机会发现不一致）"
        given = json.loads(mcp_config_snippet("/x/mcp_server.py", "/x/memory",
                                              "/x/threads.jsonl", timezone="Europe/Berlin"))
        g_args = given["mcpServers"]["memory"]["args"]
        assert g_args[-2:] == ["--timezone", "Europe/Berlin"], f"人给了就得落进 args：{g_args}"
        assert "Etc/UTC" not in given[CONFIG_NOTE_KEY] and "默认值" not in given[CONFIG_NOTE_KEY], \
            "人给了之后不该再提探测值或默认值，免得几个值打架"
    finally:
        globals()["detect_local_timezone"] = _real_detect

    # 9f.【变异靶心：MCP 配置按客户端分「可搬运/绝对路径」两档】（任务卡「MCP 配置
    #     跨机器搬不动」2026.08.02，真实用户在云端容器里当场接不上）。四条一起钉：
    #     ① Claude Code 档 + 整套在产出目录下 → args 里零绝对路径，全走占位符；
    #     ② 占位符必须带默认值 `:-.`（官方文档：变量设在服务端进程环境里，展开时
    #        读不到，不带默认值会原样留下那串字符）；
    #     ③ 其余档一个 `${` 都不许有——只有 Claude Code 认这个占位符，
    #        给别家就是一句不报到用户脸上的 file not found。generic 档还有个
    #        就在仓库里的消费者：reference_host.py 的 McpClient 直接拿
    #        mcpServers.memory 的 command+args 原样 Popen，占位符混进去它当场断；
    #     ④ Claude Code 档但 server 不在产出目录下（桌面形态）→ 老老实实绝对路径。
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "src").mkdir()
        (root / "src" / "mcp_server.py").write_text("# 占位\n", encoding="utf-8")
        (root / "memory" / "index").mkdir(parents=True)
        inside = dict(server_path=root / "src" / "mcp_server.py",
                      corpus_dir=root / "memory", threads_path=root / "threads.jsonl",
                      index_dir=root / "memory" / "index")
        cc = json.loads(mcp_config_snippet(**inside, client="claude-code",
                                           portable_root=root))
        cc_args = cc["mcpServers"]["memory"]["args"]
        assert cc_args[0] == PORTABLE_PREFIX + "src/mcp_server.py", \
            f"Claude Code 档 server 路径没走占位符：{cc_args[0]}"
        assert not any(str(root).replace("\\", "/") in x for x in cc_args), \
            f"Claude Code 档 args 里还有绝对路径：{cc_args}"
        for x in cc_args:
            assert "${" not in x or x.startswith("${CLAUDE_PROJECT_DIR:-.}/"), \
                f"占位符缺默认值 :-. ——展开失败时会把这串字符原样留下：{x}"
        assert "只有 Claude Code 认它" in cc[CONFIG_NOTE_KEY], \
            "可搬运档的 _说明 没警告占位符别抄给其它客户端"
        for other in ("codex", "generic", None):
            oc = json.loads(mcp_config_snippet(**inside, client=other,
                                               portable_root=root))
            oa = oc["mcpServers"]["memory"]["args"]
            assert not any("${" in x for x in oa), \
                f"{other} 档的配置里出现了只有 Claude Code 认的占位符：{oa}"
            assert "换机器/换容器要重新跑" in oc[CONFIG_NOTE_KEY], \
                f"{other} 档（绝对路径）的 _说明 没说这份配置跟着机器走"
        #     ④ server 在产出目录之外（桌面形态：src 在克隆仓库里）→ 落回绝对路径，
        #        不许产出"corpus 可搬、server 搬不动"的半套货
        outside = json.loads(mcp_config_snippet(
            Path(__file__).resolve().parent / "mcp_server.py",
            root / "memory", root / "threads.jsonl",
            client="claude-code", portable_root=root,
            index_dir=root / "memory" / "index"))
        oargs = outside["mcpServers"]["memory"]["args"]
        assert not any("${" in x for x in oargs), \
            f"server 不在产出目录下还产占位符，就是半套可搬运的假货：{oargs}"
        assert "换机器/换容器要重新跑" in outside[CONFIG_NOTE_KEY]

    # 9e.【变异靶心：引导句是指针，不是第二份人格文件】（任务卡「人格按需读取的
    #     引导句」2026.08.02）。任务卡点名的三个变异，前两个钉在这里、第三个
    #     （出货漏产）钉在 9b3 的文件存在断言上：
    #     ① 固定部分写长（往里塞性格/语气就会长）——固定部分必须是一句指针的长度；
    #     ② 路径不解析成绝对路径——App 的工作目录不是产出目录，相对路径静默指空。
    assert len(GUIDANCE_TEMPLATE.format(path="")) <= 30, \
        "引导句固定部分超过一句指针的长度——它开始变成第二份人格文件了"
    #     喂相对路径，出来的必须已解析成绝对路径。**在受控的浅目录里解析**——
    #     不许拿测试进程自己的 cwd 当基准：src 检出得深一点（比如 .worktrees 下），
    #     解析出的路径就撞超长闸，红的是环境不是代码（2026.08.02 真踩过）
    import os
    with tempfile.TemporaryDirectory() as td_g:
        old_cwd = os.getcwd()
        os.chdir(td_g)
        try:
            gt = guidance_text("CLAUDE.md")
        finally:
            os.chdir(old_cwd)
    assert Path(gt.rsplit("：", 1)[-1]).is_absolute(), f"引导句里的路径不是绝对路径：{gt}"
    #     超长必须拦住并把修法说进错误信息——静默截断的指针指向不存在的路径
    try:
        guidance_text("x" * (GUIDANCE_LIMIT + 1))
        assert False, "超长引导句没被拦住——超长时必须明说，不许静默出货"
    except ValueError as e:
        assert "产出目录" in str(e), "超长的错误信息没告诉用户怎么修"

    # 10. 人格不完整时拒绝出货——缺检索约定/最终约定这类必填项不能悄悄放行
    p6 = Persona("partner")
    p6.add_field(Field(id="x", section="user", label="x", value="x", confirmed=True))
    with tempfile.TemporaryDirectory() as td:
        try:
            write_bundle(td, p6, confirmed=True)
            assert False, "不完整的人格该被拦下"
        except ValueError as e:
            assert "开篇缺" in str(e)

    # 53. 原人格必须有独立 CLI 入口；没有 --persona 时用户只能误塞进 --corpus。
    import subprocess
    help_run = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--help"],
        capture_output=True, text=True, encoding="utf-8", check=True)
    assert "--persona" in help_run.stdout

    # 54. v2 inspect/extract 必须走真 CLI，且任务包只落本地文件、不要求 API key。
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        corpus_dir = root / "corpus"
        corpus_dir.mkdir()
        persona_file = root / "CLAUDE.md"
        persona_file.write_text("# 开篇\n\n我认得你。\n", encoding="utf-8")
        (corpus_dir / "window_01.md").write_text(
            "林岸：今天别熬夜。\n星回：好。\n", encoding="utf-8")
        out_dir = root / "out"
        base = [sys.executable, str(Path(__file__).resolve()), "--out", str(out_dir),
                "--persona", str(persona_file), "--corpus", str(corpus_dir), "--json"]
        inspected = subprocess.run(
            base + ["--step", "inspect"], capture_output=True, text=True,
            encoding="utf-8", check=True)
        inspect_payload = json.loads(inspected.stdout)
        assert inspect_payload["schema_version"] == 2
        assert inspect_payload["source_manifest"]["persona_file"] == str(persona_file.resolve())
        extracted = subprocess.run(
            base + ["--step", "extract"], capture_output=True, text=True,
            encoding="utf-8", check=True)
        extract_payload = json.loads(extracted.stdout)
        assert all(Path(path).exists() for path in extract_payload["package"].values())

    # 55. 零材料 v2 只出协议底座；不能逼用户写 closing 或伪造关系状态。
    from persona_compiler import PersonaItem
    zero_persona = build_persona_from_items(protocol_items(), pronouns=None)
    assert zero_persona.validate(mode="compiler_v2") == []
    zero_md = render_persona_md(zero_persona)
    assert "最终约定" not in zero_md
    assert "没有额外的硬红线" not in zero_md
    assert "关系刚开始" not in zero_md
    assert "timeline" in zero_md
    closing = PersonaItem(
        item_id="orig:closing", text="说好了就算数。", section="closing",
        source_type="original_persona", source_ref="fixture:closing",
        source_span=(0, 8), source_hash="fixture", operation="keep",
        original_text="说好了就算数。", proposed_text="说好了就算数。",
        confidence="exact", confirmed=True)
    with_closing = build_persona_from_items(protocol_items() + [closing], pronouns=None)
    closing_md = render_persona_md(with_closing)
    assert closing_md.rstrip().endswith("说好了就算数。")
    assert "****" not in closing_md and "**内容**" not in closing_md

    # 56. v2 只有十二节版本题；没有自由文本，也没有第十三个全局批准。
    assert load_init_state({"answers": {}}).mode == "legacy_v1"
    assert load_init_state(new_v2_state()).mode == "compiler_v2"
    v2_state = prepare_section_versions(new_v2_state())
    section_payload = section_choice_payload(v2_state)
    assert len(section_payload["sections"]) == len(SECTION_ORDER) == 12
    assert all("free_text" not in question for question in section_payload["sections"])
    assert all(question["section_versions"] for question in section_payload["sections"])
    assert all("diff" in version for question in section_payload["sections"]
               for version in question["section_versions"])
    empty_user = next(question for question in section_payload["sections"]
                      if question["section"] == "user")
    assert any(version["version_id"].endswith(":leave_empty")
               for version in empty_user["section_versions"])
    # 56b.【靶心：选择题那一屏的节标题也不许漏占位符】走查台账 08-04 第四条。
    #     ⚠ **靶子必须是"有 --persona 的那条路"**：零材料 state 读不出人称，
    #     标题走中性写法，`{ta}` 本来就不会出现——拿它当靶子这条断言恒真。
    #     所以先造一个用户自己写了 `## 她是谁` 的 state，再看那一屏。
    from persona_compiler import item_to_dict as _item_to_dict
    pron_items = [PersonaItem(
        item_id="orig:user_head", text="## 她是谁", section="user",
        source_type="original_persona", source_ref="fixture:persona",
        source_span=(0, 6), source_hash="fixture", operation="delete",
        original_text="## 她是谁", proposed_text="",
        operation_reason="标题由十二节骨架统一渲染",
        confidence="exact", confirmed=False)]
    pron_state = dict(new_v2_state())
    pron_state["compiler_items"] = [_item_to_dict(item) for item in pron_items]
    pron_payload = section_choice_payload(prepare_section_versions(pron_state))
    labels = [question["label"] for question in pron_payload["sections"]]
    assert not any(_SLOT_RE.search(label) for label in labels), \
        f"选择题那一屏的节标题漏出了人称占位符：{labels!r}"
    assert "她是谁" in labels, \
        f"用户自己写的是「她」，选择题里却没按他的写法渲染：{labels!r}"
    # 56c.【靶心：归节题的说明要按块的类型给】走查台账 08-04 第三条。
    #     ⚠ **两条都要**：只钉标题块那句，把常量整个换成标题文案也能全绿
    #     ——那样正文块就被告知"选哪节都不影响正文"，是**反着错**。
    unmapped = [
        PersonaItem(item_id="orig:h1", text="# 核心人格", section=None,
                    source_type="original_persona", source_ref="fixture:persona",
                    source_span=(0, 6), source_hash="fx", operation="keep",
                    original_text="# 核心人格", proposed_text="# 核心人格",
                    confidence="low", confirmed=False),
        PersonaItem(item_id="orig:body", text="她怕吵。", section=None,
                    source_type="original_persona", source_ref="fixture:persona",
                    source_span=(7, 11), source_hash="fx", operation="keep",
                    original_text="她怕吵。", proposed_text="她怕吵。",
                    confidence="low", confirmed=False)]
    map_state = dict(new_v2_state())
    map_state["compiler_items"] = [_item_to_dict(item) for item in unmapped]
    notes = {question["item_id"]: question["note"] for question
             in section_choice_payload(prepare_section_versions(map_state)
                                       )["original_mapping_questions"]}
    assert set(notes) == {"orig:h1", "orig:body"}, f"两块都该被问到：{sorted(notes)}"
    assert "不会改变出货正文" in notes["orig:h1"], \
        f"标题块的题面没说清「选哪一节都一样」，用户会在那儿白挑一个：{notes['orig:h1']!r}"
    assert "不会改变出货正文" not in notes["orig:body"], \
        f"正文块被告知「选哪节都不影响正文」——那是反着错：{notes['orig:body']!r}"
    #     两句都要说清 leave_unresolved 不是出口（它会被 SECTION_UNCONFIRMED 拦）
    #     ⚠ 原来这里写成 `A in note or A in note`——**同一个条件写了两遍**，
    #     那个 or 是死的；判据其实只有一条，写两遍不等于多查了一种说法。
    for item_id, note in notes.items():
        assert "出不了货" in note, f"{item_id} 没说 leave_unresolved 的代价"

    #     反向：读不出人称时走中性写法，**不许塞默认的他／她**（同 8a3 ④）
    neutral_labels = [question["label"] for question in section_payload["sections"]]
    assert not any(_SLOT_RE.search(label) for label in neutral_labels)
    assert not any("她" in label or "他" in label for label in neutral_labels), \
        f"人称读不出来时不该塞一个进去：{neutral_labels!r}"

    decisions = {question["section"]: question["section_versions"][0]["version_id"]
                 for question in section_payload["sections"]}
    v2_state = apply_section_decisions(v2_state, decisions)
    preview = preview_payload(v2_state)
    assert "approve_ship" not in json.dumps(preview, ensure_ascii=False)
    assert len(preview["return_targets"]) == 12
    assert "timeline" in preview["persona_markdown"]
    assert not [issue for issue in shipping_issues(v2_state, preview["persona_markdown"], None)
                if issue.severity == "blocking"]

    # 变异：少确认一节、删 timeline 指针或残留任务书，出口必须给出独立错误码。
    unconfirmed = dict(v2_state)
    unconfirmed["section_decisions"] = dict(v2_state["section_decisions"])
    unconfirmed["section_decisions"].pop("closing")
    assert "SECTION_UNCONFIRMED" in {
        issue.code for issue in shipping_issues(unconfirmed, preview["persona_markdown"], None)}
    # 71.【三个靶心，函数级】**拦截必须自带出口**：闸门判得对、也确实拦住了，
    #     但报错只说现象不说处境，人就会自己找路绕过去——2026.08.04 外部实测的代价。
    #     变异：①删掉 SECTION_UNCONFIRMED 那段出口句 → a 段红；
    #     ②把 persona_compiler 的「未知节版本」改回只说一句 → b 段红；
    #     ③把 preview_payload 里 `stale.append` 删掉 → c 段红。
    #     ⚠ b 段让人抄的字段名必须是 `version_id`：出给 CLI 的条目只有这一个键，
    #     写成别的会把人指向一个不存在的字段。
    #     a) 出口句要说清怎么提交，并提醒别手改状态文件
    unconfirmed_msg = next(
        issue.message for issue in shipping_issues(unconfirmed, preview["persona_markdown"], None)
        if issue.code == "SECTION_UNCONFIRMED")
    assert "--section-decisions-json" in unconfirmed_msg and "init_state.json" in unconfirmed_msg, \
        f"SECTION_UNCONFIRMED 必须自带出口与「别手改状态」的提醒：{unconfirmed_msg}"
    #     b) 未知节版本要列出该节当前的合法版本，并说清从哪条命令的哪个字段抄
    try:
        apply_section_decisions(v2_state, {"closing": "closing:不存在的版本"})
    except ValueError as exc:
        assert "choose-sections" in str(exc), \
            f"未知节版本要说清合法 id 从哪儿抄：{exc}"
        assert "section_versions[].version_id" in str(exc), \
            f"抄的字段名要按出给 CLI 的那个键名给（`version_id`）：{exc}"
        #      ⚠ **这一条是「列出合法版本」本身的判据**：写成 `"closing:" in str(exc)`
        #      是**恒真**的——报错开头那句「未知节版本：closing/
        #      closing:不存在的版本」自己就含 `closing:`——**它恒真，是拿一个近似的东西
        #      冒充判据本身**（CLAUDE.md 第 10 条那个形状）。实测：只删掉 `known` 那段、
        #      别的照留，`memory_init.py --selftest` 退出码仍是 0（全绿）。
        #      变异 ②b（补这条时先写后跑）：报错里保留出口句与字段名、只去掉合法版本
        #      清单 → 本条红。
        assert decisions["closing"] in str(exc), \
            f"要把该节当前的合法版本原样列出来，只指路等于让人再跑一趟：{exc}"
    else:
        raise AssertionError("未知节版本必须报错")
    #     c) 「已确认但版本号找不到」在预览层单列，不许混进「未确认」
    stale_state = dict(v2_state)
    stale_state["section_decisions"] = dict(v2_state["section_decisions"])
    stale_state["section_decisions"]["closing"] = {
        "section": "closing", "version_id": "closing:confirmed_v1", "status": "confirmed"}
    stale_codes = {issue.code for issue in
                   shipping_issues(stale_state, preview["persona_markdown"], None)}
    assert "SECTION_VERSION_STALE" in stale_codes, "已确认但版本查不到要有独立错误码"
    assert "SECTION_UNCONFIRMED" not in stale_codes, \
        "已确认的节不许再被说成「未确认」——那句话把人指向了错的出口"
    stale_preview = preview_payload(stale_state)
    assert stale_preview["stale_versions"] == ["closing"], "预览要单列这类节，别混进未确认里"
    assert "closing" in stale_preview["unresolved"], "它确实渲染不出来，仍要算未解决"
    assert "TIMELINE_POINTER_MISSING" in {
        issue.code for issue in shipping_issues(v2_state, preview["persona_markdown"].replace(
            "timeline", "history"), None)}
    assert "TASK_DIRECTIVE_REMAINS" in {
        issue.code for issue in shipping_issues(v2_state, preview["persona_markdown"]
                                                + "\n记住用户喜好", None)}

    # 变异：确认后源文件改变，必须和 inspect 时保存的哈希比较，不能拿当前值比当前值。
    from persona_compiler import build_source_manifest, item_to_dict, parse_original_persona
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        persona_file = root / "CLAUDE.md"
        persona_file.write_text("# 开篇\n\n我认得你。\n", encoding="utf-8")
        initial_manifest = build_source_manifest(persona_file, None)
        changed_state = new_v2_state(persona_file, None)
        changed_state["source_manifest"] = _manifest_payload(initial_manifest)
        changed_state["compiler_items"] = [
            item_to_dict(item) for item in parse_original_persona(persona_file)]
        changed_state = prepare_section_versions(changed_state)
        changed_questions = section_choice_payload(changed_state)["sections"]
        changed_state = apply_section_decisions(changed_state, {
            question["section"]: question["section_versions"][0]["version_id"]
            for question in changed_questions})
        changed_preview = preview_payload(changed_state)
        changed_state["preview"] = {"preview_hash": changed_preview["preview_hash"]}
        persona_file.write_text("# 开篇\n\n我已经改变。\n", encoding="utf-8")
        current_manifest = build_source_manifest(persona_file, None)
        assert "ORIGINAL_SOURCE_CHANGED" in {
            issue.code for issue in shipping_issues(
                changed_state, changed_preview["persona_markdown"], current_manifest)}

    # 未归入十二节的原文不能凭 100% span coverage 被漏掉。
    with tempfile.TemporaryDirectory() as td:
        persona_file = Path(td) / "persona.md"
        persona_file.write_text("一段没有标题但必须保留的原文。\n", encoding="utf-8")
        manifest = build_source_manifest(persona_file, None)
        unassigned = new_v2_state(persona_file, None)
        unassigned["source_manifest"] = _manifest_payload(manifest)
        unassigned["compiler_items"] = [
            item_to_dict(item) for item in parse_original_persona(persona_file)]
        unassigned = prepare_section_versions(unassigned)
        unassigned = apply_section_decisions(unassigned, {
            question["section"]: question["section_versions"][0]["version_id"]
            for question in section_choice_payload(unassigned)["sections"]})
        unassigned_preview = preview_payload(unassigned)
        assert "SECTION_UNCONFIRMED" in {
            issue.code for issue in shipping_issues(
                unassigned, unassigned_preview["persona_markdown"], manifest)}
        mapping_questions = section_choice_payload(unassigned)["original_mapping_questions"]
        assert mapping_questions[0]["item_id"] == unassigned["compiler_items"][0]["item_id"]
        assert mapping_questions[0]["choices"][-1] == "leave_unresolved"
        assert "free_text" not in mapping_questions[0]
        assigned = apply_original_section_decisions(unassigned, {
            mapping_questions[0]["item_id"]: "opening"})
        assigned = prepare_section_versions(assigned)
        assigned = apply_section_decisions(assigned, {
            question["section"]: question["section_versions"][0]["version_id"]
            for question in section_choice_payload(assigned)["sections"]})
        assigned_preview = preview_payload(assigned)
        assert "一段没有标题但必须保留的原文。" in assigned_preview["persona_markdown"]
        assert "SECTION_UNCONFIRMED" not in {
            issue.code for issue in shipping_issues(
                assigned, assigned_preview["persona_markdown"], manifest)}

    # 同一机制已有个性化原文时，选择协议默认项必须被单独识别为覆盖原文。
    original_mechanism = PersonaItem(
        item_id="orig:retrieval", text="先读我自己定下的检索约定。", section="opening",
        source_type="original_persona", source_ref="fixture:persona", source_span=(0, 14),
        source_hash="fixture", operation="keep", original_text="先读我自己定下的检索约定。",
        proposed_text="先读我自己定下的检索约定。", confidence="exact", confirmed=False,
        group_id="mechanism:opening_recognition")
    override_state = new_v2_state()
    override_state["compiler_items"] = [item_to_dict(original_mechanism)]
    override_state = prepare_section_versions(override_state)
    override_decisions = {}
    for question in section_choice_payload(override_state)["sections"]:
        versions = question["section_versions"]
        if question["section"] == "opening":
            chosen = next(version for version in versions
                          if "protocol:opening_recognition" in version["source_summary"])
        else:
            chosen = versions[0]
        override_decisions[question["section"]] = chosen["version_id"]
    override_state = apply_section_decisions(override_state, override_decisions)
    override_preview = preview_payload(override_state)
    assert "PROTOCOL_OVERRIDES_ORIGINAL" in {
        issue.code for issue in shipping_issues(
            override_state, override_preview["persona_markdown"], None)}

    # 其余结构化闸门逐个留靶心，不能以后被合并成一句“人格不完整”。
    naming_state = json.loads(json.dumps(v2_state, ensure_ascii=False))
    naming_state["diagnostics"].append({
        "code": "NAMING_NOT_BIDIRECTIONAL", "severity": "blocking",
        "message": "称呼只有单向", "item_ids": []})
    assert "NAMING_INCOMPLETE" in {
        issue.code for issue in shipping_issues(
            naming_state, preview["persona_markdown"], None)}

    derived_state = json.loads(json.dumps(v2_state, ensure_ascii=False))
    broken_coverage = PersonaItem(
        item_id="protocol:COVERAGE_TEMPLATE", text="覆盖 2026-01-01。",
        section="architecture", source_type="protocol",
        source_ref="protocol:COVERAGE_TEMPLATE", source_span=None,
        source_hash="protocol:COVERAGE_TEMPLATE", operation="add", original_text="",
        proposed_text="覆盖 2026-01-01。", confidence="derived_protocol",
        confirmed=True, derived_from=(), group_id="mechanism:memory_coverage")
    derived_state["compiler_items"].append(item_to_dict(broken_coverage))
    assert "DERIVED_PROTOCOL_PROVENANCE_MISSING" in {
        issue.code for issue in shipping_issues(
            derived_state, preview["persona_markdown"], None)}

    rewrite_state = json.loads(json.dumps(v2_state, ensure_ascii=False))
    rewrite_state["compiler_items"].append(item_to_dict(PersonaItem(
        item_id="orig:rewrite", text="旧句。", section="closing",
        source_type="original_persona", source_ref="fixture:rewrite", source_span=(0, 3),
        source_hash="fixture", operation="rewrite", original_text="旧句。",
        proposed_text="新句。", confidence="exact", confirmed=False,
        operation_reason="解决冲突")))
    rewrite_state["section_decisions"].pop("closing")
    assert "SECTION_WITH_DIFF_UNCONFIRMED" in {
        issue.code for issue in shipping_issues(
            rewrite_state, preview["persona_markdown"], None)}

    conflict_state = json.loads(json.dumps(v2_state, ensure_ascii=False))
    conflict_state["conflicts"].append({
        "conflict_id": "conflict:boundary", "kind": "boundary",
        "severity": "blocking", "item_ids": ["protocol:degradation_protocol"],
        "reason": "边界冲突", "choices": ["protocol:degradation_protocol"],
        "resolved_choice": None})
    assert "BOUNDARY_CONFLICT_UNRESOLVED" in {
        issue.code for issue in shipping_issues(
            conflict_state, preview["persona_markdown"], None)}

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        corpus_file = root / "corpus.md"
        corpus_file.write_text("虚构测试语料。\n", encoding="utf-8")
        manifest = build_source_manifest(None, corpus_file)
        assert "SOURCE_ACCOUNTING_INCOMPLETE" in {
            issue.code for issue in shipping_issues(
                v2_state, preview["persona_markdown"], manifest)}
        unknown_state = json.loads(json.dumps(v2_state, ensure_ascii=False))
        unknown_state["compiler_items"].append(item_to_dict(PersonaItem(
            item_id="corpus:unknown", text="来源不明事实。", section="user",
            source_type="corpus", source_ref="fixture:unknown", source_span=(0, 6),
            source_hash="fixture", operation="add", original_text="",
            proposed_text="来源不明事实。", confidence="high", confirmed=True)))
        assert "SOURCE_UNKNOWN" in {
            issue.code for issue in shipping_issues(
                unknown_state, preview["persona_markdown"], manifest)}

        from persona_compiler import SourceManifest
        persona_file = root / "AGENTS.md"
        persona_file.write_text("# 开篇\n\n原文。\n", encoding="utf-8")
        mixed_manifest = SourceManifest(
            persona_file=persona_file.resolve(), corpus_files=(persona_file.resolve(),),
            source_hashes={})
        assert "PERSONA_MIXED_INTO_CORPUS" in {
            issue.code for issue in shipping_issues(
                v2_state, preview["persona_markdown"], mixed_manifest)}

        partial_state = new_v2_state(persona_file, None)
        partial_manifest = build_source_manifest(persona_file, None)
        partial_state["source_manifest"] = _manifest_payload(partial_manifest)
        partial_items = [item_to_dict(item) for item in parse_original_persona(persona_file)]
        partial_items[0]["source_span"] = [0, 1]
        partial_state["compiler_items"] = partial_items
        partial_state = prepare_section_versions(partial_state)
        partial_state = apply_section_decisions(partial_state, {
            question["section"]: question["section_versions"][0]["version_id"]
            for question in section_choice_payload(partial_state)["sections"]})
        assert "ORIGINAL_COVERAGE_INCOMPLETE" in {
            issue.code for issue in shipping_issues(
                partial_state, preview_payload(partial_state)["persona_markdown"],
                partial_manifest)}

    # 旧目录已有出货人格但没显式传 --persona 时，三条路都摆出来，仍不自动采用首项。
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "AGENTS.md").write_text("# 核心人格\n\n手改内容。\n", encoding="utf-8")
        migration = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--out", str(root),
             "--step", "inspect", "--json"], capture_output=True, text=True,
            encoding="utf-8", check=True)
        assert json.loads(migration.stdout)["choices"] == [
            "treat_current_as_original", "use_original_as_is", "continue_legacy"]

    # 57. 真 CLI 走 inspect → 十二节选择 → preview → route → ship；函数绿不算接通。
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        script = str(Path(__file__).resolve())

        def run_v2(*extra):
            return subprocess.run(
                [sys.executable, script, "--out", str(root), *extra],
                capture_output=True, text=True, encoding="utf-8", check=True).stdout

        run_v2("--step", "inspect", "--json")
        continued = run_v2("--json")
        assert "--step extract" in continued
        questions = json.loads(run_v2("--step", "choose-sections", "--json"))
        cli_decisions = {question["section"]: question["section_versions"][0]["version_id"]
                         for question in questions["sections"]}
        run_v2("--step", "choose-sections", "--section-decisions-json",
               json.dumps(cli_decisions, ensure_ascii=False), "--json")
        cli_preview = json.loads(run_v2("--step", "preview", "--json"))
        assert cli_preview["unresolved"] == [] and "timeline" in cli_preview["persona_markdown"]
        run_v2("--step", "route", "--route", "zero-dep")
        ship_output = run_v2("--step", "ship", "--client", "codex")
        assert (root / "AGENTS.md").exists()
        assert "四件套" in ship_output

    # 58. 随包指南必须跟 v2 argparse 同步；不能继续教用户走固定问卷或写内容。
    docs_root = Path(__file__).resolve().parent.parent / "docs"
    quick_guide = (docs_root / "快速上手.md").read_text(encoding="utf-8")
    ai_guide = (docs_root / "给AI的引导指南.md").read_text(encoding="utf-8")
    for token in ("--persona", "--step inspect", "--step preview", "leave_empty"):
        assert token in quick_guide and token in ai_guide, f"用户指南缺 v2 命令：{token}"
    compiler_section = ai_guide.split("### 第 1 步：检查人格与语料来源", 1)[1].split(
        "### 第 7 步：接入客户端", 1)[0]
    assert "请写一句" not in compiler_section and "自由文本入口" in compiler_section

    # 59. CLI 入口必须把 stdout 锁成 UTF-8（`__main__` 里那两行 reconfigure）。
    #     ⚠ **这条断言在 Linux／默认 UTF-8 的机器上恒真，在那儿跑不算验过**：
    #     变异要在 `PYTHONIOENCODING=gbk` 下跑——删掉 `__main__` 里的
    #     `sys.stdout.reconfigure(...)`，`PYTHONIOENCODING=gbk python memory_init.py
    #     --selftest` 必须转红；加回去复绿。三个入口（本文件 / memory_import /
    #     mcp_server）各有一条同形断言。
    assert (getattr(sys.stdout, "encoding", "") or "").lower().replace("-", "") == "utf8", \
        f"CLI 入口没把 stdout 锁成 UTF-8（当前 {sys.stdout.encoding}）：" \
        "中文 Windows（cp936）下 --json 遇到 emoji 会 UnicodeEncodeError"

    # 60.【A 靶心，真命令】改一次输入人格文件不许静默塌节。
    #     夹具：现造一份带四个 markdown 标题的虚构人格文件，跑完一遍出货；
    #     然后**只删掉其中两行标题、正文一字不动**，从 --step inspect 重跑。
    #     ⚠ **不许拿「覆盖率还是 1.0」当通过证据**——那正是走查里没抓住的那把尺子。
    #     这里反过来断言 ORIGINAL_COVERAGE_INCOMPLETE **不出现**：证明的是旧闸门
    #     在结构上看不见这件事，所以必须有跨运行比对，不是证明这次通过了。
    #     变异：把 persona_drift_report 开头改成 `return None`（或删掉 shipping_issues
    #     里读 persona_drift 那一支）→ 这一整段转红。
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        script = str(Path(__file__).resolve())
        persona_text = ("# 开篇\n\n虚构开篇。\n\n# 我是谁\n\n虚构自述。\n\n"
                        "# 对方是谁\n\n虚构对方。\n\n# 最终约定\n\n虚构约定。\n")
        persona_file = root / "输入人格.md"
        persona_file.write_text(persona_text, encoding="utf-8")

        def run_drift(*extra, check=True):
            return subprocess.run(
                [sys.executable, script, "--out", str(root), *extra],
                capture_output=True, text=True, encoding="utf-8", check=check)

        def walk_and_ship():
            qs = json.loads(run_drift("--step", "choose-sections", "--json").stdout)
            run_drift("--step", "choose-sections", "--section-decisions-json",
                      json.dumps({q["section"]: q["section_versions"][0]["version_id"]
                                  for q in qs["sections"]}, ensure_ascii=False), "--json")
            run_drift("--step", "preview", "--json")
            run_drift("--step", "route", "--route", "zero-dep")
            return run_drift("--step", "ship", "--client", "codex", check=False)

        run_drift("--persona", str(persona_file), "--step", "inspect", "--json")
        assert walk_and_ship().returncode == 0, "干净的输入本来就该出得了货"
        persona_file.write_text(
            persona_text.replace("# 对方是谁\n\n", "").replace("# 最终约定\n\n", ""),
            encoding="utf-8")
        drift_payload = json.loads(run_drift(
            "--persona", str(persona_file), "--step", "inspect", "--json").stdout)
        drift_table = {row["section"]: (row["before"], row["after"])
                       for row in drift_payload["persona_drift"]["table"]}
        # ⚠ 这两个数 2026.08.04 从 (2,0) 改成 (1,0)：台账第二条把标题块打成 delete
        #   之后，差异表的口径是**只数正文块**（`original_block_counts` 的 docstring
        #   写了为什么两边都不数标题）。这一节各只有一段虚构正文，所以是 1。
        #   **塌空这件事本身照旧抓得住**——1→0 跟 2→0 一样是整节被吞掉。
        assert drift_table.get("user") == (1, 0) and drift_table.get("closing") == (1, 0), \
            f"删标题后必须报出「上次 N 块 → 这次 0 块」的差异表：{drift_table}"
        assert "PERSONA_SECTION_COLLAPSED" in {
            issue["code"] for issue in drift_payload["blocking_issues"]}, \
            "整节塌空只打 warning 等于没报——warning 在一屏输出里看不见"
        collapsed_ship = walk_and_ship()
        collapsed_msg = collapsed_ship.stdout + collapsed_ship.stderr
        assert collapsed_ship.returncode != 0 and "PERSONA_SECTION_COLLAPSED" in collapsed_msg, \
            f"塌节后必须拦在出货口：{collapsed_msg}"
        assert "ORIGINAL_COVERAGE_INCOMPLETE" not in collapsed_msg, \
            "逐字覆盖率闸门本来就看不见塌节；它要是红了说明这段夹具没在测该测的东西"
        # 拦截必须有出口，否则又把用户逼回「手改输入文件」——那正是塌节的成因。
        accepted = json.loads(run_drift(
            "--persona", str(persona_file), "--step", "inspect", "--json",
            "--accept-persona-drift").stdout)
        assert not accepted["blocking_issues"] and "PERSONA_SECTION_COLLAPSED" in {
            issue["code"] for issue in accepted["warnings"]}
        assert walk_and_ship().returncode == 0, "确认过就该能出货，不能变成死路"

    # 61.【B 靶心，真命令】TASK_DIRECTIVE_REMAINS 要有出口，不靠用户手改输入文件。
    #     夹具：现造一份含任务书残句（命中 TASK_DIRECTIVE_TOKENS）的虚构人格文件。
    #     变异：把 task_directive_delete_items 的返回改成 []
    #     → 该节只剩 1 个版本、ship 被 TASK_DIRECTIVE_REMAINS 拦死，这一段转红。
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        script = str(Path(__file__).resolve())
        persona_file = root / "输入人格.md"
        persona_file.write_text(
            "# 开篇\n\n虚构开篇。\n\n# 我是谁\n\n虚构自述。\n\n"
            "# 对方是谁\n\n对方喜欢什么去语料里找：待补。\n\n# 最终约定\n\n虚构约定。\n",
            encoding="utf-8")

        def run_exit(*extra, check=True):
            return subprocess.run(
                [sys.executable, script, "--out", str(root), *extra],
                capture_output=True, text=True, encoding="utf-8", check=check)

        run_exit("--persona", str(persona_file), "--step", "inspect", "--json")
        exit_qs = json.loads(run_exit("--step", "choose-sections", "--json").stdout)
        hit_section = next(q for q in exit_qs["sections"] if q["section"] == "user")
        dropped = [version for version in hit_section["section_versions"]
                   if "去语料里找" not in version["markdown"]]
        assert len(hit_section["section_versions"]) >= 2 and dropped, \
            "被拦住的那一节必须多出一个「删除该块」的版本，否则这份人格文件无法出货"
        assert "操作：delete" in dropped[0]["diff"] and "-对方喜欢什么去语料里找" in dropped[0]["diff"], \
            f"删除版本必须带 diff（纪律三：原文 delete 要在该节把 diff 摊开）：{dropped[0]['diff']}"
        exit_decisions = {q["section"]: q["section_versions"][0]["version_id"]
                          for q in exit_qs["sections"]}
        exit_decisions["user"] = dropped[0]["version_id"]
        run_exit("--step", "choose-sections", "--section-decisions-json",
                 json.dumps(exit_decisions, ensure_ascii=False), "--json")
        run_exit("--step", "preview", "--json")
        run_exit("--step", "route", "--route", "zero-dep")
        exit_ship = run_exit("--step", "ship", "--client", "codex", check=False)
        assert exit_ship.returncode == 0, \
            f"选了删除版本之后必须能出货：{exit_ship.stdout + exit_ship.stderr}"
        shipped_text = (root / "AGENTS.md").read_text(encoding="utf-8")
        assert "去语料里找" not in shipped_text and "虚构自述" in shipped_text

    # 61b.【靶心，真命令】「出货文件全文零『它』」那条硬约束必须能看住**存量人格文件**。
    #      靶子必须走「用户已有的人格文件」的 v2 七步：
    #      ⚠ **靶子必须是"用户已有的人格文件"走 v2 七步**，不许拿我们自己造的问卷人格
    #      代替——**那正是这个缺陷本身**：守着这条约束的老断言（`_ship_and_read` 那条）
    #      喂的是程序现造的 v1 问卷产物，**从没见过 v2 的产出**，于是旧版工具留下的
    #      `**它该记住你哪些方面**：…` 原样出货，warning 0、blocking 0，全程零提示。
    #      变异三条，**都实跑过、各自转红**（红在哪一段是实测的，不是预想的）：
    #      ①`BANNED_WORD_TOKENS` 清空（＝回到缺陷现状）→ **红在 b 段**（那一节生不出
    #        删除版本；断言顺序先撞到它，warning 那半同样没了，只是没走到）；
    #      ②`_DELETE_EXIT_RULES` 去掉禁用词那一行 → 同样红在 b 段（有提示、没出口）；
    #      ③`_step_ship_v2` 里打 warning 那几行删掉 → **红在 a 段**，报文是
    #        「实际输出：【四件套】」——**报了但没人看见，跟没报是一回事**。
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        script = str(Path(__file__).resolve())
        persona_file = root / "旧版人格.md"
        #      夹具照抄旧版工具的产物形态：**「它」在标题里**，正文是虚构内容
        persona_file.write_text(
            "# 开篇\n\n虚构开篇。\n\n# 我是谁\n\n虚构自述。\n\n"
            "# 她是谁\n\n**它该记住你哪些方面**：她在临海修旧海图。\n\n"
            "# 最终约定\n\n虚构约定。\n", encoding="utf-8")

        def run_ban(*extra, check=True):
            return subprocess.run(
                [sys.executable, script, "--out", str(root), *extra],
                capture_output=True, text=True, encoding="utf-8", check=check)

        run_ban("--persona", str(persona_file), "--step", "inspect", "--json")
        ban_qs = json.loads(run_ban("--step", "choose-sections", "--json").stdout)
        ban_section = next(q for q in ban_qs["sections"] if q["section"] == "user")
        #      b) 那一节必须多出一个「删除该块」的版本，且带 diff（纪律三）
        ban_drop = [v for v in ban_section["section_versions"] if "它" not in v["markdown"]]
        assert len(ban_section["section_versions"]) >= 2 and ban_drop, \
            "含「它」的那一节必须多出一个「删除该块」的版本——没有出口，用户只能回去" \
            "手改输入文件，而手改标题正是 08.03 台账第一条那个塌节雷区"
        assert "操作：delete" in ban_drop[0]["diff"] and "它该记住你哪些方面" in ban_drop[0]["diff"], \
            f"删除版本必须带 diff，把要删掉的原文摊开：{ban_drop[0]['diff']}"
        assert "BANNED_WORD_REMAINS" in ban_drop[0]["diff"] or "不写成「它」" in ban_drop[0]["diff"], \
            f"删除版本要说清为什么给这个选项：{ban_drop[0]['diff']}"

        #      a) 先按"原样保留"走完，出货必须**成功但报出来**（拍板：warning 不 blocking）
        keep_decisions = {q["section"]: q["section_versions"][0]["version_id"] for q in ban_qs["sections"]}
        keep_id = next(v["version_id"] for v in ban_section["section_versions"] if "它" in v["markdown"])
        keep_decisions["user"] = keep_id
        run_ban("--step", "choose-sections", "--section-decisions-json",
                json.dumps(keep_decisions, ensure_ascii=False), "--json")
        run_ban("--step", "preview", "--json")
        run_ban("--step", "route", "--route", "zero-dep")
        kept_ship = run_ban("--step", "ship", "--client", "codex", check=False)
        kept_msg = kept_ship.stdout + kept_ship.stderr
        assert kept_ship.returncode == 0, f"这条是 warning 不是 blocking，该出得了货：{kept_msg}"
        assert "BANNED_WORD_REMAINS" in kept_msg, \
            f"⚠ 存量人格文件带着「它」出货时必须报到用户脸上，实际输出：{kept_msg}"
        assert "它该记住你哪些方面" in kept_msg, \
            f"报错要指到行，不能只说「文件里有『它』」让用户自己去 Ctrl+F：{kept_msg}"
        assert "它" in (root / "AGENTS.md").read_text(encoding="utf-8"), \
            "用户选了保留就真的保留——⚠ 不许替他改原文（v2「不改写任何非空原文」不变量）"

        #      c) 再选「删除该块」重跑：能出货，且出货文件里那句没了
        drop_decisions = dict(keep_decisions)
        drop_decisions["user"] = ban_drop[0]["version_id"]
        run_ban("--step", "choose-sections", "--section-decisions-json",
                json.dumps(drop_decisions, ensure_ascii=False), "--json")
        run_ban("--step", "preview", "--json")
        dropped_ship = run_ban("--step", "ship", "--client", "codex", check=False)
        dropped_msg = dropped_ship.stdout + dropped_ship.stderr
        assert dropped_ship.returncode == 0, f"选了删除版本必须能出货：{dropped_msg}"
        shipped_ban = (root / "AGENTS.md").read_text(encoding="utf-8")
        assert "它该记住你哪些方面" not in shipped_ban and "虚构自述" in shipped_ban, \
            "选了删除版本之后那句该没了，别的原文一个字不许少"
        assert "BANNED_WORD_REMAINS" not in dropped_msg, \
            f"删干净了就不该再报——见字就报等于把 warning 变成噪声：{dropped_msg}"

    # 61d.【两个靶心，真命令】「材料很水」要有留空出口，而且**按得下去**。
    #      现场复现形态：
    #      现场证据：十二节 `section_versions` 长度全是 1、`leave_empty` 一处都没有
    #      ——这一步退化成"十二节 × 1 个选项 × 12 次确认"的橡皮图章。
    #      ⚠ **靶心二才是这张卡的本体**：只让按钮出现、按下去却被
    #      `SOURCE_ACCOUNTING_INCOMPLETE` 拦死，等于给了个按不下去的按钮。
    #      变异：把 `elif all(... == "corpus")` 那一支删掉 → a 段红（没有留空版本）。
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        script = str(Path(__file__).resolve())
        corpus_dir = root / "语料"
        (corpus_dir / "timeline").mkdir(parents=True)
        (corpus_dir / "timeline" / "window_01_2026-05-02.md").write_text(
            "# 第1个窗口 · 2026-05-02\n\n## 日常\n虚构：她在临海修旧海图。\n",
            encoding="utf-8")
        thin = corpus_dir / "timeline" / "window_02_2026-05-03.md"
        thin.write_text("# 第2个窗口 · 2026-05-03\n\n## 闲聊\n虚构：今天天气不错。\n",
                        encoding="utf-8")

        def run_thin(*extra, check=True):
            return subprocess.run(
                [sys.executable, script, "--out", str(root), *extra],
                capture_output=True, text=True, encoding="utf-8", check=check)

        #      ⚠ 夹具带上 `--persona`：不然「有原文块的节不给留空」那条边界没有靶子
        #      （实测过——不带 persona 时把边界放宽成"有材料就给"，这一段照样全绿）
        persona_file = root / "输入人格.md"
        persona_file.write_text(
            "# 开篇\n\n虚构开篇。\n\n# 我是谁\n\n虚构自述。\n\n"
            "# 她是谁\n\n她在临海修旧海图。\n\n# 最终约定\n\n虚构约定。\n",
            encoding="utf-8")
        run_thin("--persona", str(persona_file), "--corpus", str(corpus_dir),
                 "--step", "inspect", "--json")
        #      候选：两条都挂在 daily 节，其中一条就是那种"逐字对得上的废话"
        thin_ref = str(thin.resolve())
        rich_ref = str((corpus_dir / "timeline" / "window_01_2026-05-02.md").resolve())
        thin_text = thin.read_text(encoding="utf-8")
        rich_text = Path(rich_ref).read_text(encoding="utf-8")
        payload_thin = {"items": [
            {"item_id": "corpus:1", "text": "她在临海修旧海图。", "section": "daily",
             "source_ref": rich_ref,
             "source_span": [rich_text.index("虚构：她在临海修旧海图。"),
                             rich_text.index("虚构：她在临海修旧海图。") + 12],
             "candidate_kind": "fact", "evidence": "虚构：她在临海"},
            {"item_id": "corpus:2", "text": "今天天气不错。", "section": "daily",
             "source_ref": thin_ref,
             "source_span": [thin_text.index("虚构：今天天气不错。"),
                             thin_text.index("虚构：今天天气不错。") + 10],
             "candidate_kind": "fact", "evidence": "虚构：今天天气"},
        ], "source_accounting": [
            {"source_ref": rich_ref, "candidate_item_ids": ["corpus:1"]},
            {"source_ref": thin_ref, "candidate_item_ids": ["corpus:2"]},
        ]}
        run_thin("--step", "extract", "--candidates",
                 json.dumps(payload_thin, ensure_ascii=False), "--json")
        thin_qs = json.loads(run_thin("--step", "choose-sections", "--json").stdout)
        thin_daily = next(q for q in thin_qs["sections"] if q["section"] == "daily")
        #      a) 靶心一：有材料的那一节必须多出一个「本节留空」的版本，**且带说明**
        empty_versions = [v for v in thin_daily["section_versions"]
                          if v["version_id"].endswith(":leave_empty")]
        assert len(thin_daily["section_versions"]) >= 2 and empty_versions, \
            "有材料的节必须给「本节留空」这个选项，否则这一步只是橡皮图章：" \
            f"{[v['version_id'] for v in thin_daily['section_versions']]}"
        assert "window_02_2026-05-03.md" in empty_versions[0]["diff"], \
            f"留空版本要说清会丢掉哪些来源：{empty_versions[0]['diff']}"
        assert "已交代" in empty_versions[0]["diff"], \
            "语义变更要写在用户看得见的地方：弃用之后那些来源算已交代，不必回去改候选结果"
        #      ⚠ 说明写在 diff 里，人读的那条路（不带 --json）必须也打得出来
        human = run_thin("--step", "choose-sections").stdout
        assert "本节留空" in human and "window_02_2026-05-03.md" in human, \
            f"不带 --json 时留空版本只剩一个 id ＋ 一行空白，说明没打出来等于没给：{human[-400:]}"

        #      b) 靶心二（本卡本体）：选了留空之后，preview 里没有这一节，且 **ship 过得去**
        thin_decisions = {q["section"]: q["section_versions"][0]["version_id"]
                          for q in thin_qs["sections"]}
        thin_decisions["daily"] = empty_versions[0]["version_id"]
        run_thin("--step", "choose-sections", "--section-decisions-json",
                 json.dumps(thin_decisions, ensure_ascii=False), "--json")
        thin_preview = json.loads(run_thin("--step", "preview", "--json").stdout)
        assert "今天天气不错" not in thin_preview["persona_markdown"], \
            "选了留空，这一节不该出现在预览里"
        run_thin("--step", "route", "--route", "zero-dep")
        thin_ship = run_thin("--step", "ship", "--client", "codex", check=False)
        thin_msg = thin_ship.stdout + thin_ship.stderr
        assert thin_ship.returncode == 0, \
            f"⚠ 这条是本卡真正的靶心：按钮按下去必须能出货，实际：{thin_msg}"
        assert "SOURCE_ACCOUNTING_INCOMPLETE" not in thin_msg, \
            f"弃用一节之后那些来源仍算已交代（08.05 拍板），不许在这里把人拦回 extract：{thin_msg}"
        assert "今天天气不错" not in (root / "AGENTS.md").read_text(encoding="utf-8")

        #      c) 反向：零材料的节照旧**只有一个** leave_empty，不许变成两个
        for question in thin_qs["sections"]:
            empties = [v for v in question["section_versions"]
                       if v["version_id"].endswith(":leave_empty")]
            assert len(empties) <= 1, \
                f"{question['section']} 出现了两个留空版本——同一个选项给两遍是噪声"
            #      零材料的节照旧**只有** leave_empty 一个版本，不许变成两个
            if empties and len(question["section_versions"]) > 1:
                assert any(v["markdown"] for v in question["section_versions"]), \
                    f"{question['section']} 的多个版本里没有一个有正文，那这个留空选项是凭空多出来的"
        zero_material = [q for q in thin_qs["sections"]
                         if len(q["section_versions"]) == 1
                         and q["section_versions"][0]["version_id"].endswith(":leave_empty")]
        assert zero_material, "这份夹具里该有零材料的节（它们是反向靶心的靶子）"

        #      d)【边界靶心】「本节留空」**只给全是语料候选的节**——⚠ 这一条 2026.08.05
        #      是实测补上的：把那一支放宽成"有材料就给"，上面 a～c 段**照样全绿**，
        #      "断言在缺陷面前全绿"跟没有断言是一回事。
        #      两条边界各有一条已经存在的机制在管，别用这个按钮盖过去：
        #        · 有原文块的节 → 丢弃用户原文必须逐块走「删除该块」（带 diff）；
        #        · 有协议默认值的节 → 那是骨架，留空会把 ship 直接拦死。
        thin_state = json.loads((root / "init_state.json").read_text(encoding="utf-8"))
        types_by_section = {}
        for item in thin_state.get("compiler_items", []):
            if item.get("section"):
                types_by_section.setdefault(item["section"], set()).add(item.get("source_type"))
        checked = 0
        for question in thin_qs["sections"]:
            kinds = types_by_section.get(question["section"], set())
            has_empty = any(v["version_id"].endswith(":leave_empty")
                            for v in question["section_versions"])
            if not kinds:
                continue                      # 零材料那档在上面 c 段管
            checked += 1
            assert has_empty == (kinds == {"corpus"}), (
                f"{question['section']} 的材料是 {sorted(kinds)}，留空版本={has_empty}"
                "——「本节留空」只给全是语料候选的节：有原文块的走「删除该块」逐块给，"
                "有协议默认值的留空会把 ship 拦死（给个按不下去的按钮比不给更糟）")
        assert checked >= 3 and any(
            types_by_section.get(q["section"]) == {"corpus"} for q in thin_qs["sections"]), \
            "这份夹具要同时覆盖到「纯语料」「含原文」「含协议默认值」三种节，否则这条恒真"

    # 61c.【反向靶心】不含「它」的存量人格文件走同一条路，**一个字都不许报**。
    #      ⚠ 没有这一段的话，把 BANNED_WORD_TOKENS 改成恒真（比如空串）也照样绿
    #      ——"见字就报"跟"报得准"在只有正向断言时长得一模一样。
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        script = str(Path(__file__).resolve())
        clean_file = root / "干净人格.md"
        clean_file.write_text(
            "# 开篇\n\n虚构开篇。\n\n# 我是谁\n\n虚构自述。\n\n"
            "# 她是谁\n\n她在临海修旧海图。\n\n# 最终约定\n\n虚构约定。\n",
            encoding="utf-8")

        def run_clean(*extra, check=True):
            return subprocess.run(
                [sys.executable, script, "--out", str(root), *extra],
                capture_output=True, text=True, encoding="utf-8", check=check)

        run_clean("--persona", str(clean_file), "--step", "inspect", "--json")
        clean_qs = json.loads(run_clean("--step", "choose-sections", "--json").stdout)
        clean_user = next(q for q in clean_qs["sections"] if q["section"] == "user")
        assert not any(v["version_id"].endswith(DIRECTIVE_DELETE_SUFFIX)
                       for v in clean_user["section_versions"]), \
            "干净的原文块不该被塞一个「删除该块」的版本——那是给用户多一屏噪声"
        run_clean("--step", "choose-sections", "--section-decisions-json",
                  json.dumps({q["section"]: q["section_versions"][0]["version_id"]
                              for q in clean_qs["sections"]}, ensure_ascii=False), "--json")
        run_clean("--step", "preview", "--json")
        run_clean("--step", "route", "--route", "zero-dep")
        clean_ship = run_clean("--step", "ship", "--client", "codex", check=False)
        clean_msg = clean_ship.stdout + clean_ship.stderr
        assert clean_ship.returncode == 0 and "BANNED_WORD_REMAINS" not in clean_msg, \
            f"不含「它」的人格文件不许报：{clean_msg}"

    # 61e.【真命令＋函数级】节版本双键收敛、v2 显式导入与覆盖护栏。
    #      **它挡的失效形态**：某节的决定指向一个版本表里已经没有的号——
    #      `status` 仍是 `confirmed`（所以 `SECTION_UNCONFIRMED` 不报），
    #      `preview_payload` 匹配不到版本就整节跳过，**用户确认过的一节被静默从出货
    #      文件里删掉、退出码 0、全程零提示**。下面 b 段就是那个形态的断言。
    #      变异：①`_section_version_dict` 把 `id` 加回去并让读侧按 `id` 匹配 → a 段红；
    #      ②去掉 `SECTION_VERSION_STALE` 那一段 → b 段红（静默丢节复活）；
    #      ③`version_key` 拿不到键时返回 None 而不是抛 → c 段红（`None == None` 那条）。
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        script = str(Path(__file__).resolve())
        persona_file = root / "人格.md"
        persona_file.write_text(
            "# 开篇\n\n虚构开篇。\n\n# 我是谁\n\n虚构自述。\n\n"
            "# 她是谁\n\n她在临海修旧海图。\n\n# 最终约定\n\n虚构约定。\n",
            encoding="utf-8")
        v2_import = root / "v2导入.json"
        v2_import.write_text(json.dumps([{
            "name": "虚构 v2 导入",
            "chat_messages": [{"sender": "human", "text": "虚构：v2 也要导入这一句。",
                               "created_at": "2026-07-15T21:03:22.000000Z"}],
        }], ensure_ascii=False), encoding="utf-8")
        v2_external_corpus = root / "外部语料"
        v2_external_corpus.mkdir()

        def run_key(*extra, check=True):
            return subprocess.run(
                [sys.executable, script, "--out", str(root), *extra],
                capture_output=True, text=True, encoding="utf-8", check=check)

        run_key("--persona", str(persona_file), "--corpus", str(v2_external_corpus),
                "--step", "inspect", "--json")
        key_qs = json.loads(run_key("--step", "choose-sections", "--json").stdout)
        #      a) 靶心一：出给 CLI 的版本条目**只有 version_id 一个键**，不再双写
        sample = key_qs["sections"][0]["section_versions"][0]
        assert "version_id" in sample and "id" not in sample, \
            f"节版本只认一个键（version_id），不许再双写：{sorted(sample)}"
        key_decisions = {q["section"]: q["section_versions"][0]["version_id"]
                         for q in key_qs["sections"]}
        run_key("--step", "choose-sections", "--section-decisions-json",
                json.dumps(key_decisions, ensure_ascii=False), "--json")
        run_key("--step", "preview", "--json")
        run_key("--step", "route", "--route", "zero-dep")
        v2_ship = run_key(
            "--step", "ship", "--client", "codex", "--import", str(v2_import),
            "--json", check=False)
        assert v2_ship.returncode == 0, f"收敛后的新格式必须全程正常：{v2_ship.stderr}"
        assert "【导入】" in v2_ship.stdout and "→ 1 条" in v2_ship.stdout, \
            f"v2 真 CLI 收了 --import 却没报导入条数：{v2_ship.stdout}"
        v2_timeline = root / "memory" / "timeline"
        assert any("v2 也要导入这一句" in p.read_text(encoding="utf-8")
                   for p in v2_timeline.glob("*.md")), \
            "v2 真 CLI 收了 --import 却没把正文写进 timeline"
        v2_cfg = json.loads((root / "mcp-config.json").read_text(encoding="utf-8"))
        v2_args = v2_cfg["mcpServers"]["memory"]["args"]
        v2_configured_corpus = Path(v2_args[v2_args.index("--corpus") + 1])
        assert v2_configured_corpus.resolve() == (root / "memory").resolve(), \
            "v2 的 inspect 虽保存过外部 --corpus，显式 --import 后配置必须改指新落库目录"
        assert "虚构约定" in (root / "AGENTS.md").read_text(encoding="utf-8")

        # v2 同样走覆盖护栏；与 direct 正向组成变异靶心，任一路漏传 entries 都会红。
        from memory_retrieval import append_record as _append_v2
        _append_v2(root / "memory", "虚构：v2 后来追加。", "留住", window=1)
        v2_blocked = run_key(
            "--step", "ship", "--client", "codex", "--import", str(v2_import),
            check=False)
        assert v2_blocked.returncode != 0 and CORPUS_OVERWRITE_CODE in (
            v2_blocked.stdout + v2_blocked.stderr), \
            "v2 重跑 --import 没走覆盖护栏，运行期追加会被静默吃掉"
        v2_backup = run_key(
            "--step", "ship", "--client", "codex", "--import", str(v2_import),
            "--backup-corpus", check=False)
        assert v2_backup.returncode == 0, f"v2 的 --backup-corpus 出货失败：{v2_backup.stderr}"
        v2_backup_dir = root / "memory" / "timeline.bak"
        assert v2_backup_dir.is_dir() and any(
            "v2 后来追加" in p.read_text(encoding="utf-8")
            for p in v2_backup_dir.glob("*.md")), \
            "v2 的 --backup-corpus 没把运行期追加留进备份"
        _append_v2(root / "memory", "虚构：v2 再次追加。", "留住", window=1)
        v2_accept = run_key(
            "--step", "ship", "--client", "codex", "--import", str(v2_import),
            "--accept-corpus-overwrite", check=False)
        assert v2_accept.returncode == 0, \
            f"v2 的 --accept-corpus-overwrite 出货失败：{v2_accept.stderr}"
        assert not any("v2 再次追加" in p.read_text(encoding="utf-8")
                       for p in v2_timeline.glob("*.md")), \
            "v2 的 --accept-corpus-overwrite 没有按既有契约照写"

        #      b) 靶心（本卡验收判据那句）：**决定指向一个不存在的版本号 → 必须吵，
        #      不许某节静默消失**。⚠ 这个形态就是真机那份 state 的形态。
        state_path = root / "init_state.json"
        broken = json.loads(state_path.read_text(encoding="utf-8"))
        broken["section_decisions"]["closing"]["version_id"] = "closing:confirmed_v1"
        state_path.write_text(json.dumps(broken, ensure_ascii=False), encoding="utf-8")
        stale_ship = run_key("--step", "ship", "--client", "codex", check=False)
        stale_msg = stale_ship.stdout + stale_ship.stderr
        assert stale_ship.returncode != 0 and "SECTION_VERSION_STALE" in stale_msg, \
            f"⚠ 确认过但版本号找不到时必须拦住——不拦的话那一节整节从出货文件里消失" \
            f"而且不报错（实测过的真实后果）：退出码 {stale_ship.returncode} / {stale_msg}"
        assert "closing" in stale_msg and "closing:confirmed_v1" in stale_msg, \
            f"报错要点名是哪一节、决定指向的是什么，别让人自己去翻 state：{stale_msg}"
        #      b2) 同一形态在**预览层**要分两行印：混进「未确认节」说的是现象不是处境，
        #      那句话把一位真实用户指向了翻源码改匹配逻辑。
        #      ⚠ 走真进程看 CLI 实际打出来的文字，不看 payload——坏的就是印的那一层。
        stale_preview_out = run_key("--step", "preview").stdout
        assert "已确认" in stale_preview_out and "closing" in stale_preview_out, \
            f"预览要单说这一节已确认、只是版本号找不到：{stale_preview_out}"
        assert "未确认节" not in stale_preview_out, \
            f"⚠ 不许把它混进「未确认节」——那正是把人指向改源码的那句话：{stale_preview_out}"
        run_key("--step", "route", "--route", "zero-dep")       # 上一行把 step 退回 previewed

        #      c) 靶心二（防过度纠正）：拿不到键必须**抛**，不许返回 None
        #      ——外部那个补丁就是 `None == None` 相等，把 BOUNDARY_CONFLICT_UNRESOLVED
        #      静默放过；**从「报错」退化成「假装通过」比不改更坏**。
        try:
            version_key({"markdown": "没有任何 id 的坏条目"})
            assert False, "拿不到 version_id 必须抛，返回 None 会让两个 None 比出相等"
        except ValueError as exc:
            assert "version_id" in str(exc)
        #      旧 state 的兼容：**历史上写出去的条目两键恒等**，只有 id 也读得出来
        assert version_key({"id": "closing:abc"}) == "closing:abc"
        assert version_key({"version_id": "closing:abc", "id": "closing:abc"}) == "closing:abc"

    # 62.【两个靶心，真命令】legacy v1 重跑出货不许静默覆盖已有语料。
    #     夹具：现造一份虚构聊天 txt（两段会话跨两天、间隔超 gap → 落成两个窗口文件），
    #     走 v1 全流程真命令出货，再用 append_record（latent_append 那支笔本身）
    #     往 window_02 里写一条，然后**重跑出货**。
    #     ⚠ 靶心二那条才是原始现场：靶心一只证明「已有文件时会停」，
    #       靶心二证明「用户后来写进去的那句话没被吃掉」。
    #     ⚠ 全程看产出目录里的文件正文，**不拿「出货成功、无报错」当通过证据**
    #       ——那正是这个洞的形态本身。
    #     变异：把 write_corpus 里那行 guard_corpus_overwrite 删掉（或让
    #     corpus_overwrite_conflicts 恒返回 []）→ 重跑直接静默出货、追加句消失，
    #     这一段转红。
    with tempfile.TemporaryDirectory() as td:
        from memory_retrieval import append_record
        root = Path(td)
        script = str(Path(__file__).resolve())
        chat = root / "虚构聊天.txt"
        chat.write_text("2026-07-15 21:03 她\n虚构：今晚的月亮很大。\n"
                        "2026-07-15 21:05 我\n虚构：我看到了。\n"
                        "2026-07-16 09:00 她\n虚构：早，今天出门吗。\n",
                        encoding="utf-8")
        timeline = root / "memory" / "timeline"
        APPEND_ONE = "虚构：她说想学游泳。"

        def run_ow(*extra, check=True):
            r = subprocess.run([sys.executable, script, "--out", str(root), *extra],
                               cwd=str(root), capture_output=True, text=True,
                               encoding="utf-8")
            assert not check or r.returncode == 0, f"跑挂了：{extra}\n{r.stdout}\n{r.stderr}"
            return r

        def ship(*extra, check=True):
            return run_ow("--step", "ship", "--client", "codex",
                          "--import", str(chat), *extra, check=check)

        def backups():
            return sorted(p.name for p in root.glob("memory/timeline.bak*"))

        def window_02():
            return next(timeline.glob("window_02_*.md")).read_text(encoding="utf-8")

        qs_ow = json.loads(run_ow("--step", "questionnaire", "--json").stdout)
        run_ow("--step", "answers", "--answers-json", json.dumps(
            {q["qid"]: ({"keys": "".join(sorted(q["options"])[:2])} if q["options"]
                        else {"pick": "她说“到家了发一句”，我说好。"})
             for q in qs_ow["questions"]}, ensure_ascii=False))
        listed_ow = json.loads(run_ow("--step", "confirm", "--list", "--json").stdout)
        run_ow("--step", "confirm", "--decisions-json", json.dumps(
            {p["key"]: "keep" for p in listed_ow["pending"]}, ensure_ascii=False))
        run_ow("--step", "route", "--route", "zero-dep")
        assert ship("--json").returncode == 0, \
            "第一次出货本来就该出得了——目录是空的，没什么可覆盖"
        assert {p.name for p in timeline.glob("*.md")} == \
            {"window_01_2026-07-15.md", "window_02_2026-07-16.md"}
        assert not backups(), "第一次出货没覆盖任何东西，不该凭空造备份"

        #    靶心二的现场：用户用 latent_append 往已有窗口里写了一句
        append_record(root / "memory", APPEND_ONE, "还没报名", window=2)
        assert APPEND_ONE in window_02()

        #    靶心一：目标 timeline 非空且会被改样 → 必须停下来，不许静默覆盖
        blocked = ship(check=False)
        blocked_msg = blocked.stdout + blocked.stderr
        assert blocked.returncode != 0 and CORPUS_OVERWRITE_CODE in blocked_msg, \
            f"重跑出货静默覆盖了语料：{blocked_msg}"
        assert "--backup-corpus" in blocked_msg and "--accept-corpus-overwrite" in blocked_msg, \
            "拦截必须把出口写在拦截信息里，否则用户只能去手工搬文件"
        #    靶心二：那句话必须还在。**这条才是原始现场**
        assert APPEND_ONE in window_02(), "被拦下来的那次不许动用户已有的窗口"
        assert not backups(), "拦下来的那次不该在用户目录里留下任何痕迹"

        #    出口①：备份完再写。追加句在备份里取得回来
        assert ship("--backup-corpus").returncode == 0
        assert backups() == ["timeline.bak"]
        assert APPEND_ONE in next(
            (root / "memory" / "timeline.bak").glob("window_02_*.md")
        ).read_text(encoding="utf-8"), "备份里必须有用户写进去的那句话，否则备份等于没做"
        assert APPEND_ONE not in window_02(), "备份之后是照写——这一步没写等于出口是假的"

        #    判据取「内容不同」而不是「目录非空」：原样再跑一遍不该被拦，也不该造备份
        assert ship().returncode == 0, "同一份语料原样重跑，写出来跟磁盘上一样，不该拦"
        assert backups() == ["timeline.bak"]

        #    出口②：显式确认覆盖。不备份、直接写
        append_record(root / "memory", "虚构：她报名了周三那节课。", "已报名", window=2)
        assert ship("--accept-corpus-overwrite").returncode == 0
        assert "虚构：她报名了" not in window_02(), "确认覆盖那条出口没真的写下去"
        assert backups() == ["timeline.bak"], "--accept-corpus-overwrite 是「不备份直接写」"

        #    只加不减：再备份一次，旧备份原样留着，新备份顺号——每份备份存的是那一刻
        #    的全量，拿新的盖旧的会把只活在旧备份里的那段记忆抹掉
        append_record(root / "memory", "虚构：第一节课她迟到了。", "在上课", window=2)
        assert ship("--backup-corpus").returncode == 0
        assert backups() == ["timeline.bak", "timeline.bak2"]
        assert APPEND_ONE in next(
            (root / "memory" / "timeline.bak").glob("window_02_*.md")
        ).read_text(encoding="utf-8"), "旧备份被新备份盖掉了——护栏自己犯了它要挡的事"

    # 63.【两个靶心，真命令】节标题不许出现两遍；用户自己的人称写法要保住。
    #     （走查台账第二条 ＋ 台账里"顺带一件别漏"的人称温差，一并做。）
    #     夹具：现造一份虚构人格文件，四个一级标题（其中用户那节写的是「她是谁」
    #     ——**人称就藏在这一行里**）＋四段虚构正文，走 v2 全流程真命令出货。
    #     ⚠ 两个靶心**分开断言**，不合成一条：
    #       靶心一（标题只出现一次）看的是 parse_original_text 给标题块打的 delete；
    #       靶心二（人称是用户自己的写法）看的是 persona_pronouns 那条链。
    #     ⚠ **不许拿「original_span_coverage 还是 1.0」当通过证据**：覆盖率不看
    #       operation，对这条改动天然恒真，跑了也不说明任何事。判据一律取
    #       **产出文件的正文**。
    #     变异一：parse_original_text 里标题块改回 operation="keep"（proposed_text
    #       跟着还原）→ 靶心一转红（每个标题数出来是 2）。
    #     变异二：preview_payload 与 prepare_section_versions 里的 persona_pronouns(state)
    #       改回写死 None → 靶心二转红（标题变成中性的「对方是谁」）。
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        script = str(Path(__file__).resolve())
        persona_file = root / "输入人格.md"
        #  ⚠ 正文里**不许出现标题里的那几个字**：下面靶心一数的是字面出现次数，
        #    正文自己带一个就把这把尺子废了（第一版夹具写的是「虚构开篇。」，
        #    数出来 2 次，看着像洞其实是夹具的伪影）。
        persona_file.write_text(
            "# 开篇\n\n虚构：认得彼此。\n\n# 她是谁\n\n虚构：她怕吵。\n\n"
            "# 我是谁\n\n虚构自述。\n\n# 最终约定\n\n虚构：说到做到。\n", encoding="utf-8")

        def run_dup(*extra, check=True):
            return subprocess.run(
                [sys.executable, script, "--out", str(root), *extra],
                capture_output=True, text=True, encoding="utf-8", check=check)

        run_dup("--persona", str(persona_file), "--step", "inspect", "--json")
        dup_qs = json.loads(run_dup("--step", "choose-sections", "--json").stdout)
        run_dup("--step", "choose-sections", "--section-decisions-json",
                json.dumps({q["section"]: q["section_versions"][0]["version_id"]
                            for q in dup_qs["sections"]}, ensure_ascii=False), "--json")
        run_dup("--step", "preview", "--json")
        run_dup("--step", "route", "--route", "zero-dep")
        dup_ship = run_dup("--step", "ship", "--client", "codex", check=False)
        assert dup_ship.returncode == 0, \
            f"干净的输入本来就该出得了货：{dup_ship.stdout + dup_ship.stderr}"
        shipped = (root / "AGENTS.md").read_text(encoding="utf-8")

        #  靶心一：产出文件里每个节标题只出现一次。数的是**标题里那几个字**，
        #  因为骨架渲染的是 `## 开篇 · …`、用户原文写的是 `# 开篇`，两者不同字面
        #  ——只比整行会漏掉这个洞。
        #  ⚠ **这一条要跟人称那条相互独立**：用户那节的标题里带人称，直接数
        #    「她是谁」的话，人称一退回中性标签这条也跟着红，两个靶心就并成了
        #    一条断言（任务卡要求分开守）。所以这里数与人称无关的「是谁」——
        #    不管渲染成「她是谁」还是「对方是谁」，全文都该正好两处节标题带它
        #    （用户那节一处、`我是谁` 一处）。
        for title in ("开篇", "最终约定"):
            assert shipped.count(title) == 1, \
                f"标题「{title}」在产出文件里出现了 {shipped.count(title)} 次：" \
                "骨架渲染一遍、原文块又进正文一遍"
        assert shipped.count("是谁") == 2, \
            f"「是谁」那两个节标题出现了 {shipped.count('是谁')} 次（该是 2）：" \
            "原文标题块又进了一次正文"
        assert "虚构：她怕吵。" in shipped and "虚构：说到做到。" in shipped, \
            "标题打 delete 不许连正文一起吞掉——那就成了另一个洞"

        #  靶心二：人称是用户自己写的那个字，不是中性标签。
        #  两处都要：节标题，以及协议层开篇那句每轮都在的话。
        assert "## 她是谁" in shipped and "对方是谁" not in shipped, \
            f"节标题被渲染成了中性标签，不是用户自己的写法：{shipped[:400]}"
        assert "这是你和她共同维护的记忆文件" in shipped, \
            "协议层默认值里的 {ta} 没跟着用户的写法走——同一份文件里两种称呼"

        #  补充（函数层，不是靶心）：读不出来就走中性写法，不猜、不塞默认的他／她。
        def _ta(heading):
            return persona_pronouns({"compiler_items": [{
                "source_type": "original_persona", "section": "user",
                "original_text": heading, "text": heading}]})
        assert _ta("# 他是谁") == {"user": "他", "ai": None}
        assert _ta("# 小鱼是谁") == {"user": "小鱼", "ai": None}
        assert _ta("# 对方是谁") is None and _ta("# 用户是谁") is None, \
            "「对方」「用户」是我们替他挑的词，不是他自己的写法"
        assert _ta("# 它是谁") is None, "「它」在任何一档都不许出现"
        assert _ta("# 我是谁") is None, "不是用户那节的标题句式，读不出来就该是 None"

    # 73.【四个靶心，真命令】2026.08.05 外部实测（星迟＆烬，真实语料＋自建前端）走查
    #     出来的三条，各自的靶子：
    #       ① 出货报告里「记忆库」那一行**指的是检索真正读的那个目录**；
    #       ② mcp-config.json 的 `command` **在这台机器上真能跑**（不是字面 python）；
    #       ③ 旧的数量断言盖到新并进来的条目头上时，**报得出、也删得掉**；
    #       ④ 反向：那一节没并进语料时一个字都不许报（在下面 73b 组）。
    #     ⚠ 这四行故意不写成 `a)` 那个形状：`_count_assertion_groups` 认的就是它，
    #     写成那样等于凭空多出四个「子组」，项数当场对不上（这一版实测撞过）。
    #     ⚠ 夹具走真进程、走 --persona ＋ --corpus 那条路——三条缺陷全都只在这条路上
    #     出现（六步走不写 memory/timeline，也只有这条路会把语料候选并进原文节）。
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        script = str(Path(__file__).resolve())
        probe_persona = root / "原人格.md"
        probe_persona.write_text(
            "# 开篇\n\n虚构开篇。\n\n## 里程碑\n\n（以下三条都逐条确认过。）\n\n"
            "- 2020-01-01 虚构甲。\n- 2020-02-02 虚构乙。\n- 2020-03-03 虚构丙。\n\n"
            "## 按需读取\n\n需要细节时读 memory/timeline/ 。\n", encoding="utf-8")
        probe_corpus = root / "语料"
        probe_corpus.mkdir()
        corpus_line = "2020-04-04 虚构丁。"
        (probe_corpus / "w1.md").write_text(corpus_line + "\n", encoding="utf-8")
        probe_corpus_before = {
            str(p.relative_to(probe_corpus)): p.read_bytes()
            for p in probe_corpus.rglob("*") if p.is_file()}
        probe_out = root / "产出"
        probe_out.mkdir()

        def run_probe(*extra, check=True):
            return subprocess.run(
                [sys.executable, script, "--out", str(probe_out),
                 "--persona", str(probe_persona), "--corpus", str(probe_corpus), *extra],
                capture_output=True, text=True, encoding="utf-8", check=check).stdout

        def probe_ship(pick_delete):
            """跑一遍完整六步；pick_delete=True 时里程碑那节选「删除该块」的版本。"""
            questions = json.loads(run_probe("--step", "choose-sections", "--json"))
            picks = {}
            for question in questions["sections"]:
                versions = question["section_versions"]
                if question["section"] == "milestones" and pick_delete:
                    dropped = [version for version in versions
                               if "以下三条" not in version["markdown"]]
                    assert dropped, "命中数量断言的那一节必须多出一个「删除该块」的版本"
                    picks[question["section"]] = dropped[0]["version_id"]
                else:
                    picks[question["section"]] = versions[0]["version_id"]
            run_probe("--step", "choose-sections", "--section-decisions-json",
                      json.dumps(picks, ensure_ascii=False), "--json")
            run_probe("--step", "preview", "--json")
            run_probe("--step", "route", "--route", "zero-dep")
            return run_probe("--step", "ship", "--client", "generic")

        run_probe("--step", "inspect", "--json")
        probe_items = []
        for index, line in enumerate([corpus_line], 1):
            probe_items.append({
                "item_id": f"cand:{index:04d}", "text": line, "section": "milestones",
                "source_ref": str((probe_corpus / "w1.md").resolve()),
                "source_span": [0, len(line)], "candidate_kind": "milestone",
                "evidence": line, "event": line[11:], "reading": "虚构解读",
                "current_state": "虚构状态"})
        run_probe("--step", "extract", "--candidates", json.dumps({
            "items": probe_items,
            "source_accounting": [{"source_ref": str((probe_corpus / "w1.md").resolve()),
                                   "candidate_item_ids": [i["item_id"] for i in probe_items]}],
        }, ensure_ascii=False), "--json")
        keep_ship = probe_ship(pick_delete=False)

        # a)【靶心】报告里的「记忆库」必须指 --corpus 那个目录，另把 <out>/memory/index
        #    明说成第二读取根，不能把两层混成一个目录。
        #    ⚠ 判据是**那一行里出现的是哪个路径**，不是"报告里提没提 memory"——
        #    缺陷版本打的正是 `记忆库：<out>/memory`，而那个目录一个文件都没有
        #    （六步走这条路 write_bundle 只 mkdir、不写盘），检索读的是 --corpus。
        #    ⚠ 产出 memory/ 现在只拥有独立索引目录；不能再把它整段叫成空记忆库。
        memory_line = next(line for line in keep_ship.splitlines() if "记忆库：" in line)
        assert str(probe_corpus.resolve()) in memory_line, \
            f"出货报告把只装索引的产出目录当成叙事记忆库报出来了：{memory_line}"
        probe_index = probe_out / "memory" / "index"
        assert (probe_index / "README.txt").read_text(encoding="utf-8") == INDEX_README
        assert not list(probe_index.glob("*.md")), "出货不许伪造索引摘要"
        assert "索引摘要目录" in keep_ship and str(probe_index.resolve()) in keep_ship, \
            f"报告必须把第二读取根指给用户：{keep_ship}"
        assert probe_corpus_before == {
            str(p.relative_to(probe_corpus)): p.read_bytes()
            for p in probe_corpus.rglob("*") if p.is_file()}, \
            "外部 --corpus 不许被出货改写"

        # b)【靶心】配置里的 command 必须是这台机器上真能跑的解释器。
        #    ⚠ 判据是**能不能起进程**，不是"是不是字面 python3"：写死任何一个名字都
        #    会在某类机器上挂（缺陷版本写死 `python`，只装 python3 的机器照抄必挂，
        #    而症状是客户端 spawn 静默失败、没有一行指回这个字段）。
        probe_cfg = json.loads(
            (probe_out / "mcp-config.json").read_text(encoding="utf-8"))
        probe_args = probe_cfg["mcpServers"]["memory"]["args"]
        assert probe_args[probe_args.index("--index-dir") + 1].replace("\\", "/").endswith(
            "/memory/index"), f"配置没有接入独立索引目录：{probe_args}"
        probe_command = probe_cfg["mcpServers"]["memory"]["command"]
        assert subprocess.run([probe_command, "-c", "import sys"],
                              capture_output=True).returncode == 0, \
            f"mcp-config.json 里的 command 在这台机器上起不来：{probe_command!r}"
        assert "command 那一行" in probe_cfg[CONFIG_NOTE_KEY], \
            "解释器这一行换机器要改，配置说明里必须写着"

        # c)【靶心】旧断言盖新内容：warning 报得出来，且那一节有「删除该块」的出口，
        #    选了之后 warning 消失、出货文件里也没有那句话。
        #    ⚠ **必须验"选了删除版本之后 warning 消失"**：那句原文和它的删除孪生条目
        #    同时躺在 compiler_items 里，按条目判的写法会让这条 warning 改不掉——
        #    一条改不掉的警告等于没有警告。
        assert "STALE_COUNT_ASSERTION" in keep_ship and "以下三条" in keep_ship, \
            f"数量断言盖到新条目头上，出货时一声不吭：{keep_ship}"
        assert "以下三条" in (probe_out / "persona.md").read_text(encoding="utf-8"), \
            "报了 warning 却顺手改了用户原文——v2 不许改写原文，只许报和给出口"
        delete_ship = probe_ship(pick_delete=True)
        assert "STALE_COUNT_ASSERTION" not in delete_ship, \
            f"选了「删除该块」之后这条 warning 还在报，等于没有出口：{delete_ship}"
        shipped_probe = (probe_out / "persona.md").read_text(encoding="utf-8")
        assert "以下三条" not in shipped_probe and "虚构甲" in shipped_probe, \
            "删除版本该只删那一块断言，清单本身要留着"

    # 73b.【反向靶心】同一句数量断言，那一节**没有并进语料**时一个字都不许报。
    #      没有这一段，「见到数字就报」跟「报得准」长得一模一样（同 61c 的形状）。
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        quiet_persona = root / "原人格.md"
        quiet_persona.write_text(
            "# 开篇\n\n虚构开篇。\n\n## 里程碑\n\n（以下三条都逐条确认过。）\n\n"
            "- 2020-01-01 虚构甲。\n- 2020-02-02 虚构乙。\n- 2020-03-03 虚构丙。\n\n"
            "## 按需读取\n\n需要细节时读 memory/timeline/ 。\n", encoding="utf-8")
        quiet_out = root / "产出"
        quiet_out.mkdir()

        def run_quiet(*extra):
            return subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--out", str(quiet_out),
                 "--persona", str(quiet_persona), *extra],
                capture_output=True, text=True, encoding="utf-8", check=True).stdout

        run_quiet("--step", "inspect", "--json")
        quiet_questions = json.loads(run_quiet("--step", "choose-sections", "--json"))
        quiet_milestones = next(question for question in quiet_questions["sections"]
                                if question["section"] == "milestones")
        assert len(quiet_milestones["section_versions"]) == 1, \
            "没并进新内容的那一节不该多出「删除该块」的版本——那是给它自己找的噪声"
        run_quiet("--step", "choose-sections", "--section-decisions-json", json.dumps(
            {question["section"]: question["section_versions"][0]["version_id"]
             for question in quiet_questions["sections"]}, ensure_ascii=False), "--json")
        run_quiet("--step", "preview", "--json")
        run_quiet("--step", "route", "--route", "zero-dep")
        quiet_ship = run_quiet("--step", "ship", "--client", "generic")
        assert "STALE_COUNT_ASSERTION" not in quiet_ship, \
            f"那一节一条语料都没并进来，「以下三条」仍然是真话，不许报：{quiet_ship}"

    # 74.【原人格直接接入】受管协议范围、显式导入与覆盖护栏。
    #     变异靶心：删掉任一工具约定、重复追加、吞掉代码块里的标题、遇到坏标记仍猜着写、
    #     漏报相似旧协议或把来源 blocking 洗绿，下面至少一条会红。期望值全部来自手写夹具，
    #     不复用渲染函数自己算答案。
    direct_source = "# 我的人格\n\n原文一字不动。\n\n```txt\n## 代码里的标题\n```\n"
    direct_rendered, direct_warnings = render_direct_persona(direct_source)
    assert direct_rendered.split(DIRECT_PROTOCOL_START, 1)[0].rstrip("\n") == \
        direct_source.rstrip("\n"), "直接接入改动了受管标记外的用户正文"
    assert all(name in direct_rendered for name in (
        "latent_search", "latent_append", "latent_session_start", "latent_thread_close")), \
        "直接接入的最低协议缺工具约定"
    assert "indexEvidence" in direct_rendered and "recordId＋indexEvidence" in direct_rendered \
        and "indexStatus=pending" in direct_rendered, \
        "直接接入协议必须带正文先保存、按 recordId 补索引的完整闭环"
    direct_rerendered, _ = render_direct_persona(direct_rendered)
    assert direct_rerendered.count(DIRECT_PROTOCOL_START) == 1 and \
        direct_rerendered.count(DIRECT_PROTOCOL_END) == 1, "重复出货把受管协议越叠越多"
    assert direct_warnings == [], f"干净人格不该凭空报警：{direct_warnings}"
    _, similar_warnings = render_direct_persona(
        direct_source + "\n## 记忆库\n\n旧的检索约定。\n")
    assert DIRECT_PROTOCOL_SIMILAR_TEXT in similar_warnings, \
        "原人格已有未受管的记忆库协议标题时没有提醒可能重复"
    for broken in (
            direct_source + "\n" + DIRECT_PROTOCOL_START,
            direct_source + "\n" + DIRECT_PROTOCOL_END + "\n" + DIRECT_PROTOCOL_START):
        try:
            render_direct_persona(broken)
            assert False, "单边或倒序标记不许猜着改"
        except ValueError as exc:
            assert "DIRECT_PROTOCOL_MARKER_BROKEN" in str(exc), \
                f"坏标记错误没给稳定代码：{exc}"
    with tempfile.TemporaryDirectory() as td:
        direct_root = Path(td)
        direct_persona = direct_root / "成熟人格.md"
        direct_persona.write_text(direct_source, encoding="utf-8")
        direct_import = direct_root / "直接接入导入.json"
        direct_import.write_text(json.dumps([{
            "name": "虚构 direct 导入",
            "chat_messages": [{"sender": "human", "text": "虚构：直接接入也要导入这一句。",
                               "created_at": "2026-07-15T21:03:22.000000Z"}],
        }], ensure_ascii=False), encoding="utf-8")
        direct_external_corpus = direct_root / "外部语料"
        direct_external_corpus.mkdir()
        (direct_external_corpus / "旧窗口.md").write_text(
            "虚构：inspect 时读这套，但显式 import 后配置不能继续指它。\n",
            encoding="utf-8")
        direct_out = direct_root / "产出"

        def direct_cli(*extra, check=True):
            return subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--out", str(direct_out),
                 "--persona", str(direct_persona), *extra], capture_output=True,
                text=True, encoding="utf-8", check=check)

        direct_inspect = direct_cli(
            "--step", "inspect", "--existing-persona-choice", "use_original_as_is",
            "--corpus", str(direct_external_corpus), "--json")
        direct_payload = json.loads(direct_inspect.stdout)
        assert direct_payload["mode"] == "direct_persona" and \
            direct_payload["skipped_steps"] == ["extract", "choose-sections", "preview"] and \
            direct_payload["next"] == "--step route", \
            f"直接接入 inspect 没有明确跳步和下一步：{direct_payload}"
        assert not (direct_out / "persona-extraction").exists(), \
            "用户选了直接接入，inspect 仍然生成了提取任务包"

        compiled_out = direct_root / "编译产出"
        compiled = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--out", str(compiled_out),
             "--persona", str(direct_persona), "--step", "inspect", "--json"],
            capture_output=True, text=True, encoding="utf-8", check=True)
        compiled_payload = json.loads(compiled.stdout)
        assert compiled_payload["mode"] == "compiler_v2" and \
            compiled_payload["next"] == "--step extract", \
            "不给新选择时旧的 --persona 编译路径发生了静默换路"

        no_route = direct_cli("--step", "ship", "--client", "claude-code", check=False)
        assert no_route.returncode != 0 and "检索路线还没选过" in no_route.stderr, \
            f"直接接入绕过了隐私路线选择：{no_route.stdout}{no_route.stderr}"
        direct_cli("--step", "route", "--route", "zero-dep")
        direct_ship = direct_cli(
            "--step", "ship", "--client", "claude-code",
            "--import", str(direct_import), "--json")
        assert "【导入】" in direct_ship.stdout and "→ 1 条" in direct_ship.stdout, \
            f"direct 真 CLI 收了 --import 却没报导入条数：{direct_ship.stdout}"
        direct_timeline = direct_out / "memory" / "timeline"
        assert any("直接接入也要导入这一句" in p.read_text(encoding="utf-8")
                   for p in direct_timeline.glob("*.md")), \
            "direct 真 CLI 收了 --import 却没把正文写进 timeline"
        shipped_direct = (direct_out / "CLAUDE.md").read_text(encoding="utf-8")
        assert shipped_direct.split(DIRECT_PROTOCOL_START, 1)[0].rstrip("\n") == \
            direct_source.rstrip("\n") and shipped_direct.count(DIRECT_PROTOCOL_START) == 1, \
            "直接出货没有保住用户正文或把协议重复追加了"
        assert direct_persona.read_text(encoding="utf-8") == direct_source, \
            "产出目录外的输入人格被直接模式改写了"
        direct_state = load_state(direct_out)
        assert direct_state["shipping"]["compiler_gates_passed"] is False and \
            direct_state["shipping"]["direct_persona_selected"] is True, \
            "直接模式把跳过编译伪装成了编译闸门通过"
        direct_cfg = json.loads((direct_out / "mcp-config.json").read_text(encoding="utf-8"))
        direct_args = direct_cfg["mcpServers"]["memory"]["args"]
        configured_corpus = Path(direct_args[direct_args.index("--corpus") + 1])
        assert configured_corpus.resolve() == (direct_out / "memory").resolve(), \
            "直接模式配置没有指向实际语料目录"

        # direct 也必须复用现有覆盖护栏：导入后追加的运行期记录不能被重跑 ship 静默吃掉。
        from memory_retrieval import append_record as _append_direct
        _append_direct(direct_out / "memory", "虚构：direct 后来追加。", "留住", window=1)
        direct_blocked = direct_cli(
            "--step", "ship", "--client", "claude-code",
            "--import", str(direct_import), check=False)
        assert direct_blocked.returncode != 0 and CORPUS_OVERWRITE_CODE in (
            direct_blocked.stdout + direct_blocked.stderr), \
            "direct 重跑 --import 没走覆盖护栏，运行期追加会被静默吃掉"
        direct_cli(
            "--step", "ship", "--client", "claude-code",
            "--import", str(direct_import), "--backup-corpus")
        direct_backup_dir = direct_out / "memory" / "timeline.bak"
        assert direct_backup_dir.is_dir() and any(
            "direct 后来追加" in p.read_text(encoding="utf-8")
            for p in direct_backup_dir.glob("*.md")), \
            "direct 的 --backup-corpus 没把运行期追加留进备份"
        _append_direct(direct_out / "memory", "虚构：direct 再次追加。", "留住", window=1)
        direct_cli(
            "--step", "ship", "--client", "claude-code",
            "--import", str(direct_import), "--accept-corpus-overwrite")
        assert not any("direct 再次追加" in p.read_text(encoding="utf-8")
                       for p in direct_timeline.glob("*.md")), \
            "direct 的 --accept-corpus-overwrite 没有按既有契约照写"
        assert "inspect 时读这套" in (
            direct_external_corpus / "旧窗口.md").read_text(encoding="utf-8"), \
            "显式 import 不许反过来改 inspect 保存的外部 --corpus"

        # 外部输入在 inspect 后变化必须拦；恢复原文后重跑则幂等。
        direct_persona.write_text(direct_source + "\n后来改了一字。\n", encoding="utf-8")
        changed = direct_cli("--step", "ship", "--client", "claude-code", check=False)
        assert changed.returncode != 0 and "ORIGINAL_SOURCE_CHANGED" in changed.stderr, \
            f"inspect 后源文件变化仍然出了货：{changed.stdout}{changed.stderr}"
        direct_persona.write_text(direct_source, encoding="utf-8")
        direct_cli("--step", "ship", "--client", "claude-code")
        assert (direct_out / "CLAUDE.md").read_text(encoding="utf-8").count(
            DIRECT_PROTOCOL_START) == 1, "同一外部输入重跑出货不幂等"

        # 输入与目标同路径：允许用户选择覆盖，但必须先留备份，且下一次能识别自己的受管块。
        same_out = direct_root / "同路径产出"
        same_out.mkdir()
        same_persona = same_out / "CLAUDE.md"
        same_persona.write_text(direct_source, encoding="utf-8")

        def same_cli(*extra, check=True):
            return subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--out", str(same_out),
                 "--persona", str(same_persona), *extra], capture_output=True,
                text=True, encoding="utf-8", check=check)

        same_cli("--step", "inspect", "--existing-persona-choice", "use_original_as_is",
                 "--json")
        same_cli("--step", "route", "--route", "zero-dep")
        same_cli("--step", "ship", "--client", "claude-code")
        assert (same_out / "CLAUDE.md.bak").read_text(encoding="utf-8") == direct_source, \
            "同路径覆盖前没有留下逐字相同的原人格备份"
        same_cli("--step", "ship", "--client", "claude-code")
        assert same_persona.read_text(encoding="utf-8").count(DIRECT_PROTOCOL_START) == 1, \
            "同路径第二次出货没有把自己的受管块幂等更新"

        skipped = direct_cli("--step", "extract", "--json", check=False)
        assert skipped.returncode != 0 and "用户选择跳过" in skipped.stderr and \
            "treat_current_as_original" in skipped.stderr, \
            "直接模式误入编译步骤时没有解释用户选择和切换出口"

        # inspect 会把来源问题如实存进状态；route 不能把它洗绿，ship 必须继续阻断。
        missing_corpus = direct_root / "不存在的语料"
        direct_cli("--corpus", str(missing_corpus), "--step", "inspect",
                   "--existing-persona-choice", "use_original_as_is", "--json")
        direct_cli("--step", "route", "--route", "zero-dep")
        bad_source_ship = direct_cli("--step", "ship", "--client", "claude-code",
                                     check=False)
        assert bad_source_ship.returncode != 0 and \
            "CORPUS_NOT_FOUND" in bad_source_ship.stderr, \
            "直接模式把 inspect 已发现的阻断级来源问题静默洗绿了"

    # Grok 主会话人格的生产变异靶心：客户端映射缺失、嵌套目录未建、frontmatter
    # 漏包、状态只存 basename、换档备份对含目录的路径调用 with_name 失败，任一处都会红。
    # 期望路径和头部来自 Grok Build 官方 agent 文件契约，不由实现常量反算。
    with tempfile.TemporaryDirectory() as td:
        grok_root = Path(td)
        grok_source = grok_root / "原人格.md"
        grok_source.write_text(direct_source, encoding="utf-8")
        grok_out = grok_root / "产出"

        def grok_cli(*extra, check=True):
            return subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--out", str(grok_out),
                 "--persona", str(grok_source), *extra], capture_output=True,
                text=True, encoding="utf-8", check=check)

        grok_cli("--step", "inspect", "--existing-persona-choice",
                 "use_original_as_is", "--json")
        grok_cli("--step", "route", "--route", "zero-dep")
        grok_ship = grok_cli("--step", "ship", "--client", "grok")
        grok_persona = grok_out / ".grok" / "agents" / "companion.md"
        grok_text = grok_persona.read_text(encoding="utf-8")
        assert grok_text.startswith(
            "---\nname: companion\ndescription: 记忆协议生成的长期陪伴主会话\n---\n\n"), \
            "Grok agent 文件缺少可被宿主识别的最小 frontmatter"
        assert direct_source.rstrip("\n") in grok_text and \
            grok_text.count(DIRECT_PROTOCOL_START) == 1, \
            "Grok 包装层吞掉了直接人格正文或 Latent 受管协议"
        assert not (grok_out / "AGENTS.md").exists() and \
            not (grok_out / ".grok" / "personas").exists(), \
            "Grok 主会话人格误占 Codex 文件或落进仅供子 agent 的 personas 目录"
        grok_state = load_state(grok_out)
        assert grok_state["last_shipped_persona"] == ".grok/agents/companion.md", \
            "状态只保存了 Grok 人格 basename，换档时会找不到嵌套旧文件"
        assert "/agents" in grok_ship.stdout and "grok --agent companion" in grok_ship.stdout, \
            "Grok 出货回执没有告诉用户如何启用 companion 主会话"

        # 编译人格和直接人格必须走同一包装边界；不能只靠当前共享调用结构碰巧成立。
        compiled_grok = Persona("partner")
        fill_protocol_defaults(compiled_grok)
        compiled_paths = write_bundle(
            grok_root / "编译包装", compiled_grok, client="grok", confirmed=True,
            validation_mode="compiler_v2", rendered_override="# 编译人格\n",
            add_coverage=False)
        assert compiled_paths["persona"].read_text(encoding="utf-8").startswith(
            GROK_AGENT_FRONTMATTER + "# 编译人格\n"), \
            "编译人格出 Grok 时漏掉了与直接人格共用的 agent 包装"

        grok_cli("--step", "ship", "--client", "codex")
        assert (grok_out / "AGENTS.md").is_file() and \
            (grok_out / ".grok" / "agents" / "companion.md.bak").is_file(), \
            "从 Grok 切到 Codex 时没有安全退役嵌套旧人格"

        # 状态文件不是文件操作授权：伪造 ../ 路径也不能让下一次 ship 移动产出目录外文件。
        outside = grok_root / "outside.md"
        outside.write_text("外部文件不能动。\n", encoding="utf-8")
        forged_state = load_state(grok_out)
        forged_state["last_shipped_persona"] = "../outside.md"
        save_state(grok_out, forged_state)
        grok_cli("--step", "ship", "--client", "grok")
        assert outside.exists() and \
            outside.read_text(encoding="utf-8") == "外部文件不能动。\n" and \
            not outside.with_name("outside.md.bak").exists(), \
            "伪造的 last_shipped_persona 越过产出目录移动了外部文件"

    # 75.【四个靶心，函数级】ship 出货报告对非目录 --corpus 补校验（2026.08.22 外部部署
    #     报告 2）：memory_report_lines 原来不校验 corpus_dir 是不是目录，单文件路径也
    #     照抄进报告与 mcp-config，doctor 随即拒绝「不是目录」——同一条路径 ship 说是
    #     记忆库、doctor 说不能当 --corpus。护栏放在 write_bundle 任何写盘之前，三条 ship
    #     路径都经它，靶心一必非零退出（CLI 把 ValueError 兜成 SystemExit「不出货」）。
    with tempfile.TemporaryDirectory() as td75:
        root75 = Path(td75)
        p75 = Persona("partner")
        fill_protocol_defaults(p75)
        qs75 = questions_for(coverage_report(p75), has_corpus=False)
        ans75 = {}
        for q75 in qs75:
            if q75.kind in ("choice", "multi"):
                ans75[q75.qid] = {"keys": list(q75.options)[0], "note": ""}
            else:
                ans75[q75.qid] = {"pick": "你来，我就在。", "note": ""}
        apply_answers(p75, qs75, ans75)
        apply_confirmations(p75, {p.key: "keep" for p in pending_confirmations(p75)})
        single_file75 = root75 / "记忆.md"
        single_file75.write_text("虚构一行。\n", encoding="utf-8")
        # 靶心一：--corpus 指向单文件 → 写盘之前抛错，报「不是目录」，不出现照抄那句。
        raised75 = False
        try:
            write_bundle(root75 / "产出1", p75, corpus_dir=str(single_file75),
                         confirmed=True)
        except ValueError as e75:
            raised75 = True
            assert "不是目录" in str(e75) and "记忆库那一层目录" in str(e75), \
                f"非目录 --corpus 要报「不是目录、指向记忆库那一层目录」：{e75}"
            assert "指的那个目录" not in str(e75), \
                f"非目录 --corpus 不许照抄成「就是你 --corpus 指的那个目录」：{e75}"
        assert raised75, \
            "非目录 --corpus 必须在写盘前拦下（摘掉这条校验，本断言转红）"
        assert not (root75 / "产出1" / "mcp-config.json").exists(), \
            "拦下之后不许留下半套货（否则 mcp-config 里的 --corpus 正是那条坏路径）"
        # 靶心二：--corpus 指向合法目录 → 照旧出货，报告里就是那个目录。
        good_dir75 = root75 / "语料"
        good_dir75.mkdir()
        (good_dir75 / "w1.md").write_text("虚构语料。\n", encoding="utf-8")
        paths75 = write_bundle(root75 / "产出2", p75, corpus_dir=str(good_dir75),
                               confirmed=True)
        report75 = memory_report_lines(paths75, str(good_dir75))
        assert any(str(good_dir75.resolve()) in line for line in report75), \
            f"合法目录 --corpus 要照旧报该目录：{report75}"
        # 靶心三：--import 落库（corpus_files 非空）不受影响，照旧报「落盘 N 个窗口文件」，
        #     哪怕 corpus_dir 恰好是个非目录路径——落库那支先命中；不追认「--import 不落库」。
        import75 = memory_report_lines(
            {"memory_dir": root75 / "产出3" / "memory",
             "index_dir": root75 / "产出3" / "memory" / "index",
             "corpus_files": ["a.md", "b.md"]},
            str(single_file75))
        assert any("落盘 2 个窗口文件" in line for line in import75), \
            f"--import 落库那支不受目录校验影响：{import75}"

    # 76.【两个靶心，真命令】`--candidates` 重传只能替换语料候选，不能把协议骨架带丢。
    #     夹具：零材料 v2 先跑一遍 extract → choose-sections，让协议项与版本表都落盘；
    #     再用空候选重传。判据：重传后的 state 仍有 protocol 项、旧版本表和选择都被清掉；
    #     接着重跑 choose-sections 后，协议项重新进入版本表。只验证「最终能出货」不够——
    #     缺陷版正是带着旧版本表一路走到 ship 才报人格文件不完整。
    #     变异：保留列表删掉 protocol → 靶心一转红；删掉 section_versions 的清理 → 靶心二转红。
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        script = str(Path(__file__).resolve())
        candidate_path = root / "候选结果.json"
        candidate_path.write_text('{"items": [], "source_accounting": []}', encoding="utf-8")

        def run_reimport(*extra):
            return subprocess.run(
                [sys.executable, script, "--out", str(root), *extra], capture_output=True,
                text=True, encoding="utf-8", check=True).stdout

        run_reimport("--step", "inspect", "--json")
        run_reimport("--step", "extract", "--candidates", str(candidate_path), "--json")
        first_choices = json.loads(run_reimport("--step", "choose-sections", "--json"))
        first_decisions = {question["section"]: question["section_versions"][0]["version_id"]
                           for question in first_choices["sections"]}
        run_reimport("--step", "choose-sections", "--section-decisions-json",
                     json.dumps(first_decisions, ensure_ascii=False), "--json")
        run_reimport("--step", "extract", "--candidates", str(candidate_path), "--json")
        reimported = load_state(root)
        assert any(item.get("source_type") == "protocol"
                   for item in reimported.get("compiler_items", [])), \
            "重传 candidates 后协议项被丢掉了"
        assert not reimported.get("section_versions") and not reimported.get("section_decisions"), \
            "重传 candidates 后旧版本表和旧选择必须失效"
        rebuilt = json.loads(run_reimport("--step", "choose-sections", "--json"))
        rebuilt_state = load_state(root)
        assert len(rebuilt["sections"]) == len(SECTION_ORDER) and any(
            item.get("source_type") == "protocol"
            for item in rebuilt_state.get("compiler_items", [])), \
            "重跑 choose-sections 后协议骨架没有恢复成十二节版本表"

    # 72.【一个靶心，元断言】**项数不许再靠增量漂**（2026.08.05 审核轮，CLAUDE.md 第 10 条）。
    #     **判据（先写后数）——一项 ＝ 一个「下辖断言的注释标头」**，只认两种形态：
    #     **A 型主组**＝缩进恰好 4 空格、`# 数字[小写字母].`（如 `# 61e.`）；
    #     **B 型子组**＝任意缩进、`# 小写字母[一位数字])`（如 `# a)` `# b2)`）。
    #     算法：按 A 切主组；主组内有下辖断言的 B → 计 B 的个数（**主组本身不再另计**，
    #     否则一个组数两遍），没有 B → 计 1；**标头到下一个标头之间没有 `assert` 的不计**。
    #     按这条数出来是 **67**（这一组 `72` 自己也算一组；脚本即下面那个函数）。
    #     ⚠ **那行斜杠描述的条数跟项数不是一回事**，别拿它当项数的判据：描述里
    #     好几段是「甲＋乙＋丙」并着写的，一段对应两三个断言组。
    #     （这里原来写着一个具体的条数「55」，但**没有写下它是按什么切的**——
    #     换个人重数得不到同一个数，按 CLAUDE.md 第 10 条那就不是判据，删掉。）
    #     ⚠ 靶心：**总结行不影响任何断言、改错了不会红**，所以让它自己守自己——
    #     下面这条断言现读现数本文件，数不上就红，项数不会静默漂。
    #     变异（先写后跑）：把总结行的数字改成别的 → 本条红；新增／删掉一个断言组
    #     而不改那个数 → 本条红。
    assertion_group_count = _count_assertion_groups(Path(__file__).resolve())
    _declared = int(re.search(r"selftest ok（(\d+)项断言：", _SELFTEST_SUMMARY).group(1))
    assert assertion_group_count == _declared, (
        f"自检总结行写的项数（{_declared}）跟按判据现数出来的（{assertion_group_count}）对不上"
        "——加减了断言组就把那个数一起改，判据见本组注释")

    print(_SELFTEST_SUMMARY)


_SELFTEST_SUMMARY = (
    "selftest ok（67项断言：体检识破空泛 / 只问缺口 / 立场题选项与排序 / "
          "归属句式 / 默认值不预支历史 / 协议层不问用户 / 导出纪律 / 渲染顺序 / "
          "人称锚死一套 / 用户只有一种称呼形态 / 昵称档不被静默吃掉 / 中性档不丢主语 / 人称从语料读出来 / 中性写法不许漏 / 语料侧判定不许猜 / 全文零它 / 称呼不重复拼接 / 关系状态归开篇 / "
          "答案读回不静默丢 / 任务书不泄漏进人格文件 / 长字面量不崩 / 未决草稿不蒸发 / "
          "记忆库落盘带日期 / 覆盖区间进人格文件 / 止血纪律进人格文件 / "
          "pick 题只许挑不许写 / 冷启动出得了货 / "
          "AI 驱动不绕过确认 / 确认关卡 / 续跑 / 完整性 / generic 档随货带契约 / "
          "CLI 真进程走文档教的那条命令 / 换档退役旧档 / 同档重跑不自退 / "
          "不动不是我们出的文件 / 覆盖手改的人格文件前必先备份并说出来 / MCP 配置路径可直接用 / ship 话术不串档 / 检索路线是真选择点 / 云端档落到启动参数 / 凭证不进产出目录 / 引导句是指针（绝对路径、超长不出货、换档跟着换） / MCP 配置分可搬运与绝对路径两档（占位符只给 Claude Code、必带默认值、半套可搬运不许出） / 覆盖区间的 {end} 是提示不是断言 / 指针护栏盖住 timeline 写回层 / "
          "CLI 入口把 stdout 锁成 UTF-8（⚠ 变异要在 PYTHONIOENCODING=gbk 下跑，"
          "默认 UTF-8 的机器上这条恒真） / "
          "改一次输入人格文件不许静默塌节（跨运行比对差异表，整节塌空是 blocking，"
          "⚠ 不拿覆盖率 1.0 当通过证据） / "
          "TASK_DIRECTIVE_REMAINS 在该节有一个带 diff 的「删除该块」版本，选它能出货 / "
          "重跑出货不许静默覆盖语料（判据落在目录层、取「内容不同」；两条出口"
          "--backup-corpus／--accept-corpus-overwrite 都实跑过；⚠ 靶心二看的是"
          "latent_append 写进去的那句话还在不在，不拿「出货成功、无报错」当证据） / "
          "节标题在产出文件里只出现一次（标题块打 delete，原文仍被逐字认领）＋"
          "人称取用户自己在标题里的写法，不是中性的「对方」"
          "（⚠ 两条分开断言；⚠ 不拿覆盖率 1.0 当通过证据，它不看 operation） / "
          "选择题那一屏的节标题也不许漏人称占位符（⚠ 靶子必须是有 --persona 的那条路，"
          "零材料 state 走中性写法、这条恒真） / "
          "mcp-config.json 里四个路径都是绝对的（⚠ 只查 server 那一条在缺陷面前全绿） / "
          "MCP 配置里没给时区就落默认东八区、探到的宿主时区一律只进说明不进 args"
          "（云服务器上探出来正好是 UTC，拿它顶替等于给那个缺陷发合格证）/ "
          "「出货文件全文零它」那条硬约束看得住存量人格文件（⚠ 靶子必须是用户已有的"
          "人格文件走 v2 七步，不许拿我们自己造的问卷人格代替——那正是这个缺陷本身；"
          "warning 打到屏幕上、那一节有带 diff 的「删除该块」出口、选保留就真保留不改原文）"
          " / 反向：不含「它」的人格文件一个字都不许报（没有这一段，"
          "「见字就报」跟「报得准」长得一模一样）/ "
          "有材料的节也给「本节留空」出口，且**按得下去**（⚠ 靶心二是 ship 不被 "
          "SOURCE_ACCOUNTING_INCOMPLETE 拦死——只让按钮出现等于给个按不下去的按钮；"
          "边界断言：留空版本存在 ⟺ 该节全是语料候选，⚠ 这条是实测补的，"
          "放宽边界那次变异第一次跑全绿）/ "
          "归节题的说明按块的类型给（标题块要说清「选哪一节都不改变出货正文」，"
          "⚠ 正文块不许被这么说，那是反着错）/ "
          "节版本收敛成一个 version_id、且「某节静默消失」绝迹（61e：出给 CLI 的条目"
          "不再双写；决定指向不存在的版本号时 ship 非零退出并点名哪一节／哪个号；"
          "拿不到键必须抛不许返回 None）＋v2 真 CLI 的 --import 报条数、写 timeline、"
          "配置改指新库，且 block／backup／accept 三档都实跑（⚠ 共用导入入口返回 None，"
          "本组正向转红）/ "
          "拦截自带出口（71：SECTION_UNCONFIRMED 说清 --section-decisions-json 且提醒"
          "别手改状态、未知节版本列出该节合法版本并说清抄哪个字段、「已确认但版本号"
          "找不到」在预览层单列 stale_versions 不混进未确认；⚠ 这三条是 2026.08.05 真机 "
          "state 打回来的）＋"
          "同一形态走真进程看 CLI 实际打出来的两行（61e b2：坏的就是印的那一层））/ "
          "出货报告分别指明叙事记忆库与独立索引目录，README.txt 不冒充摘要，外部 corpus "
          "保持不变（73：⚠ 判据是路径、文件内容与配置参数，"
          "不是「报告里提没提 memory」）＋"
          "mcp-config.json 的 command 在这台机器上真能起进程（⚠ 判据是能不能起，"
          "不是像不像 python3——写死任何一个名字都会在某类机器上挂）＋"
          "旧数量断言盖到新并进来的条目头上时报得出、且那一节有「删除该块」的出口，"
          "选了之后 warning 消失、原文没被我们改（⚠ 必须验「选了之后不再报」："
          "原文块和它的删除孪生条目同时在 compiler_items 里，"
          "按条目判会让这条 warning 改不掉）/ "
          "反向：那一节没并进语料时同一句数量断言一个字都不许报"
          "（73b：没有这一段，「见数字就报」跟「报得准」长得一模一样）/ "
          "原人格直接接入的受管协议只动自己的标记范围（74：用户正文保真／四工具齐／"
          "重复出货幂等／坏标记明确阻断／相似旧协议提醒／来源 blocking 不洗绿）＋"
          "direct 真 CLI 的 --import 报条数、写 timeline、配置改指新库，且覆盖三档实跑／"
          "外部 --corpus 保持只读 / "
          "ship 出货报告对非目录 --corpus 补校验（75：单文件 --corpus 在写盘前拦下、"
          "报「不是目录、指向记忆库那一层目录」且非零退出、不留半套货；合法目录照旧报该"
          "目录；--import 落库那支不受影响照报「落盘 N 个窗口文件」；⚠ 变异摘掉校验靶心一"
          "转红）/ "
          "--candidates 重传只替换语料候选，协议项保留；旧版本表与选择失效、重跑 "
          "choose-sections 后协议项重新入表（76：不许只验最终 ship，缺陷版正是在旧版本表上"
          "一路走到出货才报人格不完整）/ "
          "⚠ 项数自己守自己（72：这一行的数字现读现数本文件，对不上就红——"
          "判据与算法见 72 那组注释）)")


def _rebuild(state):
    """状态 → (persona, questions)。每一步都从状态重建，不序列化整个 persona——
    重放（协议层 + 答案 + 已做的确认决策）是幂等的，状态文件里只存用户的输入，
    坏了也看得懂、改得动。"""
    # 人称：语料侧检测结果存在状态里（questionnaire 步算的），用户答的覆盖它。
    # **问卷题目要按"还差哪一侧"来出**，所以先合一次、再据此出题；出完题若用户
    # 已经答了人称，再合第二次，让下面的渲染用上真人称
    detected = state.get("pronouns_detected") or {}
    #    人称题就那两道，直接拿它们解答案即可——不必等 questions_for 出完题，
    #    也就避开了"出题要先知道人称、填模板又要先出题"这个循环
    pronouns = pronouns_from_answers(list(PRONOUN_QUESTIONS.values()),
                                     state.get("answers"), detected)
    persona = Persona("partner")
    fill_protocol_defaults(persona, pronouns)
    qs = questions_for(coverage_report(persona, pronouns=pronouns),
                       has_corpus=state.get("has_corpus", False), pronouns=pronouns)
    if state.get("answers"):
        apply_answers(persona, qs, state["answers"], pronouns)
    if state.get("decisions"):
        apply_confirmations(persona, state["decisions"])
    return persona, qs


def _load_json_arg(value):
    """`--answers-json` / `--decisions-json` 的取值：文件路径，或 `-` 表示读 stdin，
    或直接就是一段 JSON 字面量。三种都收——驱动方是 AI 时，它手上是内存里的
    结构化数据，不该被逼着先落一个临时文件。

    **字面量先认形状，路径判断再兜 OSError**（2026.08.02 外部复现时撞到）：原先无条件
    先跑 `Path(value).exists()`，而字面量一旦超过文件名长度上限（255 字节）就抛
    `OSError: [Errno 36] File name too long`，**没有兜底、整条命令崩掉**。11 个字段的
    决定串就已经约 290 字节——也就是说**正常规模的一次确认必崩**，而这条路正是为
    "不开终端的人"补的 AI 驱动接口。自检当时没抓到，是因为 fixture 的字面量都很短：
    **fixture 的规模本身就是一种伪影**（同回归集那边"合成语料规模不足会凭空造出
    假缺陷"，只是这次方向相反——规模不足把真缺陷藏了起来）。
    两道都补：以 `{`／`[` 开头的直接当字面量，不去问文件系统；真去问的时候 OSError
    也落回字面量解析，不再让它冒出来。"""
    if value == "-":
        return json.loads(sys.stdin.read())
    if value.lstrip()[:1] in ("{", "["):
        return json.loads(value)
    try:
        p = Path(value)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except OSError:
        pass          # 太长／非法路径名：它本来就不是路径，按字面量解析
    return json.loads(value)


def _questions_payload(qs, pronouns=None):
    """问卷的机器可读形态。AI 驱动时靠它拿题目，不用去抠给人看的排版文本
    （抠文本＝解析自己的输出，格式一改就断，而且断得无声无息）。"""
    return [{"qid": q.qid, "section": q.section,
             "label": fill_pronouns(q.label, pronouns), "kind": q.kind,
             "text": fill_pronouns(q.text, pronouns), "max_chars": q.max_chars,
             "options": ({k: {"label": fill_pronouns(v[0], pronouns),
                              "directive": fill_pronouns(v[1], pronouns)}
                          for k, v in q.options.items()} if q.options else None),
             "attribution": q.attribution, "optional": q.optional}
            for q in qs]


def _client_of(args):
    """questionnaire 步存进状态的客户端档：命令行没给就落默认值。
    （`--client` 的 argparse 默认值是 None，为的是让 ship 步能分辨
    "用户显式指定"与"用户没提"——见 resolve_client。）"""
    return resolve_client(args.client, None)[0]


def _detect_pronouns_from_corpus(corpus_path):
    """有 --corpus 就先从语料里判一次人称。判不出来返回空 dict，交给问卷去问。

    **必须把用户侧说话人告诉 detect_pronouns**（2026.08.02 验收打回）：不传的话
    它走"分不清谁是谁"那条分支、**永远返回 None**，于是生产路径上这个功能是死的
    ——每个用户都会被问一遍，语料里明明写着答案。而这个信息本来就有：
    `memory_import` 的两个翻译器都把用户侧归一成了 `"user"`（ChatGPT 走
    `author.role`，Claude 那条把 `human` 归一成 `user`，出口一致）。

    **叙事体语料（`speaker=""`，timeline md 这类）判不出来是正常的**，跟"能判却
    没判"不是一回事：那种语料里根本没有说话人标记，退回去问用户就对了。

    **读语料失败不该拦住整个流程**——人称判不出来只是"要多问一句"，
    而导入报错在这一步没有任何用户能理解的上下文，所以这里吞掉异常、当作判不出。"""
    if not corpus_path:
        return {}
    try:
        from memory_import import load_any
        entries = load_any(corpus_path)
    except Exception:
        return {}
    return {k: v for k, v in detect_pronouns(entries, USER_SPEAKERS).items() if v}


def _compile_issue_payload(issue):
    return {"code": issue.code, "severity": issue.severity,
            "message": issue.message, "item_ids": list(issue.item_ids)}


def _manifest_payload(manifest):
    return {
        "persona_file": str(manifest.persona_file) if manifest.persona_file else None,
        "corpus_files": [str(path) for path in manifest.corpus_files],
        "source_hashes": dict(manifest.source_hashes),
        "issues": [_compile_issue_payload(issue) for issue in manifest.issues],
    }


def new_v2_state(persona_path=None, corpus_path=None):
    """建立来源编译状态；v1 状态仍由旧路径原样读写。"""
    return {
        "schema_version": 2,
        "mode": "compiler_v2",
        "step": "",
        "inputs": {"persona": str(persona_path) if persona_path else None,
                   "corpus": str(corpus_path) if corpus_path else None},
        "source_manifest": {},
        "compiler_items": [],
        "conflicts": [],
        "section_versions": {},
        "section_decisions": {},
        "source_accounting": [],
        "diagnostics": [],
        "preview": {},
        "shipping": {},
    }


def load_init_state(data):
    """只判状态代际，不把 v1 字段静默重解释成 v2。"""
    from types import SimpleNamespace
    mode = "compiler_v2" if data.get("schema_version") == 2 else "legacy_v1"
    return SimpleNamespace(mode=mode, data=data)


def version_key(version):
    """版本条目 → 它的 id。**唯一的取键口径**，读的地方都走这里。

    ⚠ **拿不到就抛，不许返回 None**：
    若 `version.get("version_id")` 取不到时返回 `None`，而决定那侧也可能
    是 `None`，两个 `None` 一比就相等——`BOUNDARY_CONFLICT_UNRESOLVED` 于是被静默
    放过。**从「报错」退化成「假装通过」，比不改更坏。**
    ⚠ 兼容一句：2026.08.05 之前写出去的 state 同时有 `id` 与 `version_id` 且值恒等
    （已拿真的旧 state 核过），所以只认 `version_id`、`id` 兜底，两者都没有就是坏数据。"""
    key = version.get("version_id") or version.get("id")
    if not key:
        raise ValueError(
            "节版本条目没有 version_id（也没有旧版的 id），这份 init_state.json 是坏的："
            f"{sorted(version)[:6]}。别继续读——继续读会让某一节静默消失。")
    return key


def _section_version_dict(version):
    """节版本 → 出给 CLI 的字典。

    ⚠ **只写 `version_id` 一个键**：
    原先这里同时写 `id` 和 `version_id`、值恒等，而 `persona_compiler._version_to_dict`
    只写 `version_id`——**同一个对象两份序列化、键集不同**，读的一侧也就跟着分家
    （这边按 `id` 匹配、那边写着防御性回退）。真机上已经出现过失步的实例。
    收敛之后判据只有一个：**`version_id`**。"""
    return {
        "version_id": version.version_id,
        "section": version.section,
        "item_ids": list(version.item_ids),
        "markdown": version.markdown,
        "source_summary": list(version.source_summary),
        "diff": version.diff,
        "resolved_conflict_ids": list(version.resolved_conflict_ids),
    }


def _coverage_protocol_item(state):
    """覆盖日期是协议模板 + 语料派生证据，不新增第五来源类型。"""
    from persona_compiler import PersonaItem

    corpus_path = state.get("inputs", {}).get("corpus")
    span = corpus_coverage(corpus_path) if corpus_path else None
    refs = tuple(state.get("source_manifest", {}).get("corpus_files", ()))
    if not span:
        return None
    value = COVERAGE_TEMPLATE.format(start=span[0], end=span[1])
    return PersonaItem(
        item_id="protocol:COVERAGE_TEMPLATE", text=value, section="architecture",
        source_type="protocol", source_ref="protocol:COVERAGE_TEMPLATE",
        source_span=None, source_hash="protocol:COVERAGE_TEMPLATE",
        operation="add", original_text="", proposed_text=value,
        confidence="derived_protocol", confirmed=True, derived_from=refs,
        group_id="mechanism:memory_coverage")


def _style_disclaimer_item():
    """真实风格片段出现时机械附加学习边界，不把免责声明伪装成个性。"""
    import hashlib
    from persona_compiler import PersonaItem

    return PersonaItem(
        item_id="protocol:style_disclaimer", text=DISCLAIMER, section="style",
        source_type="protocol", source_ref="protocol:style_disclaimer",
        source_span=None, source_hash=hashlib.sha256(DISCLAIMER.encode("utf-8")).hexdigest(),
        operation="add", original_text="", proposed_text=DISCLAIMER,
        confidence="protocol", confirmed=True, group_id="mechanism:style_disclaimer")


# 任务书／空指令的词面。**只有这一份**：出货闸（shipping_issues）拿它判拦不拦，
# 选择题（task_directive_delete_items）拿同一份生成「删除该块」的版本。
# ⚠ 两处必须同源——各写一份的话，闸门拦得住的句子选择题里生不出出口，
#   用户就又被推回「手改输入文件」那条路（那正是现象 A 的雷区）。
TASK_DIRECTIVE_TOKENS = ("去语料里找", "记住用户喜好", "请用户补写", "请写一句")

# 维护者的硬约束词面：**出货文件全文零「它」**（人格文件里的 AI 不是"它"）。
# ⚠ 这条约束必须覆盖**用户已有的人格文件**走 v2 编译器的路径；只拿程序现造的
# v1 问卷人格做断言，会漏掉存量文件。七步走的契约是「不改写、
# 不丢弃任何非空原文」。于是旧版工具留下的 `**它该记住你哪些方面**：…` 原样出货，
# 10 个「它」、warning 0 条、blocking 0 条，**全程零提示**。
#
# ⚠ **为什么是 warning 不是 blocking**（2026.08.05 维护者拍板 ②＋③）：
# 「它」是个常用字，用户原文里完全可能有正当用法（"那台咖啡机，它坏了"）。
# 拦死等于替用户判断他的原文，而**我们明确不许改写用户原文**（v2 不变量）。
# 所以给的是"看得见 ＋ 有出口"：报出来，并在那一节多给一个「删除该块」的版本。
BANNED_WORD_TOKENS = ("它",)

# 「删除该块」版本的 item_id 后缀。每次重建节版本时先按这个后缀清掉旧的再重生成，
# 免得用户改了归节之后留下一个挂在旧节上的孤儿版本。
DIRECTIVE_DELETE_SUFFIX = ":directive-delete"

# 出口规则表：**哪些词面值得给一个「删除该块」的版本**。
# ⚠ **这张表就是"触发面"本身，加一行之前先问一句**（2026.08.05 拍板时维护者点的名）：
# 这个词命中之后，用户"把这块删掉"是不是一个**合理的选项**？——是待办占位／写错的
# 人称这类"本来就不该进成品"的东西才算；不是"我们看着不顺眼"的都往这里塞。
# ⚠ 两列缺一不可：闸门（shipping_issues）拿同一份词表判报不报，选择题拿它生成出口
# ——各写一份的话，闸门报得出的句子选择题里生不出出口，用户又被推回「手改输入文件」，
# 而手改标题正是 08.03 台账第一条那个塌节雷区。
_DELETE_EXIT_RULES = (
    (TASK_DIRECTIVE_TOKENS,
     "TASK_DIRECTIVE_REMAINS：这一块含任务书或空指令（命中「{hit}」），"
     "带着它出货会被拦下。选这个版本＝把这块原文删掉。"),
    (BANNED_WORD_TOKENS,
     "BANNED_WORD_REMAINS：这一块含「{hit}」——人格文件里的 AI 不写成「它」。"
     "⚠ 这条**不拦你出货**：你原文里若是正当用法（指某个东西），"
     "留着就行；确实是旧版工具留下的写法，就选这个版本把这块删掉。"),
)


# 原文里的**数量断言**词面（2026.08.05 外部实测第 3 条）。
# 形状：「以下五条都经她逐条确认」「上述三点均已核对」——**它断言的是一批内容，
# 而那批内容在合并语料候选之后变多了**，于是这句话原样盖到了没经确认的新条目头上。
# ⚠ 逐字保留原文是 v2 的硬契约，所以我们**不改写它**，只做两件事：
#   ① 那一块多给一个「删除该块」的版本（这里）；② 出货时报一条 warning（shipping_issues）。
# ⚠ 只认「量词＋指代」这个形状，别放宽成「文里有数字就报」——人格文件里数字到处都是。
_COUNT_ASSERTION_RE = re.compile(
    r"(以下|下面|上述|以上|前述|这|那)[^。；！？\n]{0,4}?"
    r"(\d+|[一二两三四五六七八九十]+)\s*(条|点|项|件|句|则)")


STALE_COUNT_EXIT_REASON = (
    "STALE_COUNT_ASSERTION：这一块里有一句数量断言「{hit}」，"
    "而这一节这次并进了 {count} 条来自语料的新内容——那句话现在**盖到了它没数过、"
    "也没经人确认的条目头上**。⚠ 这条**不拦你出货**：三条出路自己挑——"
    "选这个版本把这句话删掉；或者回去改输入人格文件里那一句再从 --step inspect 重跑；"
    "或者你确认它仍然说得通（比如那个数指的是别的东西），原样留着。")


def stale_count_assertion_hits(items):
    """→ [(item, 命中的那句话, 这一节并进来的语料条数)]；没有就空表。

    **判据有两半，缺一不算**（先写下来再动手）：
      ① 这一块是**原文块**、且这次是「保留」（keep／move）——删掉的块不用管；
      ② **它所在的那一节这次并进了 ≥1 条语料候选**——没并进新东西的话，
         「以下五条」还是那五条，那句话仍然是真的，报了就是噪声。

    ⚠ 第二半是这条规则的全部分量所在：这不是「见到数字就报」，是「**这个数被新内容
    盖过去了**」。放宽掉它，规则立刻退化成对任何一份带清单的人格文件都刷屏。"""
    joined = {}
    for item in items:
        if item.source_type == "corpus" and item.section:
            joined[item.section] = joined.get(item.section, 0) + 1
    hits = []
    for item in items:
        if item.source_type != "original_persona" or not item.section:
            continue
        if item.operation not in {"keep", "move"}:
            continue
        count = joined.get(item.section, 0)
        if not count:
            continue
        match = _COUNT_ASSERTION_RE.search(item.original_text)
        if match:
            hits.append((item, match.group(0), count))
    return hits


def task_directive_delete_items(items):
    """给命中出口规则表的原文块，额外造一个「删除该块」的版本。

    **这是 `TASK_DIRECTIVE_REMAINS` 的出口**（2026.08.03 走查第五条）：拦截本身是对的
    ——待办占位不该进成品——但在此之前流程里没有退出路径，那一节只生成 1 个版本、
    不含「删掉它」，于是这份人格文件**通过官方命令无法出货**，唯一出路是手工编辑
    输入文件后整条重跑；而手改输入文件正是静默塌节（现象 A）的雷区。

    做法是给同一块原文再造一个 `operation="delete"` 的孪生条目，**共用 group_id**
    ——于是 `build_conflicts` 把两者判成同一语义组的不同版本，
    `build_section_versions` 的 `itertools.product` 自然给这一节生出两个版本：
    保留的和删掉的。`render_item_diff` 只对 rewrite/delete 出 diff，所以删除那版
    **自带 diff 和拦截理由**，兑现指南纪律三「原文 delete 必须在该节把 diff 摊开」。

    ⚠ 孪生条目的 `source_span` / `source_hash` / `original_text` 一个字不动：
    `original_span_coverage` 认的是这三样，动了会把逐字覆盖闸门弄红。"""
    from dataclasses import replace

    twins = []
    # 数量断言那条规则要看**整节的上下文**（这一节这次并进了几条语料），
    # 词面表那两条只看块自己——所以先按 item_id 摊平成一张表再进循环，
    # 两种规则最后合流到同一个 `reasons`（同一块仍然只造一个孪生条目）
    stale_counts = {item.item_id: (hit, count)
                    for item, hit, count in stale_count_assertion_hits(items)}
    for item in items:
        if item.source_type != "original_persona" or item.section is None:
            continue
        if item.operation not in {"keep", "move"}:
            continue
        reasons = []
        if item.item_id in stale_counts:
            hit, count = stale_counts[item.item_id]
            reasons.append(STALE_COUNT_EXIT_REASON.format(hit=hit, count=count))
        for tokens, template in _DELETE_EXIT_RULES:
            hit = [token for token in tokens if token in item.original_text]
            if hit:
                reasons.append(template.format(hit="」「".join(hit)))
        if not reasons:
            continue
        # 同一块同时命中两条规则时只造**一个**孪生条目（两条理由并排写出来）——
        # 造两个的话这一节会多出两个内容完全一样的「删除该块」版本，用户选哪个都行，
        # 而那正是"多给一个选项"变成"多给一屏噪声"的开始
        twins.append(replace(
            item,
            item_id=item.item_id + DIRECTIVE_DELETE_SUFFIX,
            operation="delete",
            proposed_text="",
            confirmed=False,
            operation_reason="；".join(reasons)))
    return twins


# 用户那一节的标题模板是 `{ta}是谁`（persona_template.SECTION_ORDER）。人称就写在
# 用户自己那行标题里——**正则从模板现拼，不另写一份**：模板哪天改了，这里跟着走。
_TA_HEADING_PREFIX, _, _TA_HEADING_SUFFIX = SECTIONS["user"].partition("{ta}")
_TA_HEADING_RE = re.compile(
    r"^\s{0,3}#{1,6}\s*" + re.escape(_TA_HEADING_PREFIX)
    + r"(.+?)" + re.escape(_TA_HEADING_SUFFIX) + r"\s*$")
# 读到这些等于没读到：它们本来就是**我们**替用户挑的词（中性写法用的就是「对方」），
# 不是他自己的写法。拿它们去填 {ta} 会把一句中性话伪装成"用户原来就这么写的"。
_TA_NOT_A_PRONOUN = frozenset({"用户", "对方", "ta", "TA", "你", "我", "TA 是谁", "这个人"})


def persona_pronouns(state):
    """v2 出货路径的人称：**从用户自己那份人格文件里读出来**，读不到就走中性写法。

    2026.08.04（走查台账第二条的第二件）：v2 这条路上 `pronouns` 一路写死 `None`，
    于是十二节骨架把用户那节的标题渲染成中性标签「对方是谁」、协议层开篇渲染成
    「这是你和对方共同维护的记忆文件」——**而用户自己写的是「她」**。恋人向人格
    文件上这个温差用户看得出来，而且它在不变量层、每轮都在。
    台账第二条把标题块打成 `delete` 之后这件事更硬了：用户那行 `## 她是谁` 不再
    进正文，**中性标签就是他唯一能看到的那个标题**，写错就没有别处兜着。

    判据只认一处：用户自己那节的标题行（`{ta}是谁`）。**这是他亲手写的字**，
    比任何统计都准，也不需要额外读语料。读不出来就返回 None 走中性写法——
    不猜、不塞默认的他／她（同 `pronouns_from_answers`：两边都没有就是 None）。

    ⚠ 语料侧的人称判定（`detect_pronouns`）**没有接进 v2**，v1 问卷那条路才走它。
    这里不假装有：它需要在 inspect 步把整份语料读一遍，是另一件事。

    返回 `{"user": …, "ai": None}`——AI 那一侧在出货文本里没有槽位（`我是谁` 是
    定死的标题，协议层模板里只有 `{ta}`），没有可读之处就不编一个出来。"""
    from persona_compiler import is_heading_block

    for item in state.get("compiler_items", ()):
        if item.get("source_type") != "original_persona" or item.get("section") != "user":
            continue
        if not is_heading_block(item):
            continue
        text = item.get("original_text") or item.get("text") or ""
        first_line = text.splitlines()[0] if text.splitlines() else text
        match = _TA_HEADING_RE.match(first_line)
        if not match:
            continue
        ta = match.group(1).strip()
        if ta in PRONOUN_CHOICES:
            return {"user": ta, "ai": None}
        # 昵称档：跟 pronouns_from_answers 用同一套护栏（不许「它」、限长）——
        # 这东西会填进句子中间，四十个字的"称呼"会把每一句都撑坏。
        if (ta and ta not in _TA_NOT_A_PRONOUN and len(ta) <= NICKNAME_MAX_CHARS
                and not any(bad in ta for bad in _NICKNAME_REJECT)):
            return {"user": ta, "ai": None}
    return None


def prepare_section_versions(state):
    """从来源项确定性重建十二节版本；旧决定只在版本 ID 仍存在时保留。"""
    from persona_compiler import (
        Conflict, CompilerState, SectionVersion, build_conflicts,
        build_section_versions, item_from_dict, item_to_dict,
    )

    state = dict(state)
    items = [item_from_dict(item) for item in state.get("compiler_items", ())
             if not str(item.get("item_id", "")).endswith(DIRECTIVE_DELETE_SUFFIX)]
    items.extend(task_directive_delete_items(items))
    refs = {item.source_ref for item in items}
    # 协议层默认值里的 {ta} 按用户自己的写法填（见 persona_pronouns）；
    # 读不出来才退到中性写法——**这一句是那个"温差"的另一半**，跟节标题同源一份人称。
    for item in protocol_items(persona_pronouns(state)):
        if item.source_ref not in refs:
            items.append(item)
    if any(item.section == "style" and item.source_type != "protocol" for item in items):
        disclaimer = _style_disclaimer_item()
        if disclaimer.source_ref not in {item.source_ref for item in items}:
            items.append(disclaimer)
    coverage = _coverage_protocol_item(state)
    if coverage and coverage.source_ref not in {item.source_ref for item in items}:
        items.append(coverage)
    conflicts = build_conflicts(items)
    versions_by_section = {}
    for section, _label in SECTION_ORDER:
        section_items = [item for item in items if item.section == section]
        versions = build_section_versions(section, section_items, conflicts)
        if not section_items:
            versions = [SectionVersion(
                version_id=f"{section}:leave_empty", section=section,
                item_ids=(), markdown="", source_summary=(), diff="")]
        elif all(item.source_type == "corpus" for item in section_items):
            # 「本节留空」：
            # 原先 `leave_empty` **只在某节一条材料都没有时**才被造出来，于是
            # 「有材料」的节永远只有"全收"一个选项——十二节 × 1 个选项 × 12 次确认，
            # 这一步退化成橡皮图章。
            # ⚠ **它拆掉的是唯一一道人工判断**：extract 那道
            # `SOURCE_ACCOUNTING_INCOMPLETE` 硬闸逼着「每个来源都要交代」，而校验器
            # 只验 evidence 逐字对得上——**逐字对得上的废话也是逐字对得上**。
            # 「真有东西」和「为了交代硬凑」在自动闸那里长得一模一样，
            # `choose-sections` 是唯一分得出来的场合（人看一眼就知道哪句是凑数的）。
            #
            # ⚠ **只给「全是语料候选」的节**，边界是想清楚的，别顺手放宽：
            #   · 有**原文块**（`original_persona`）的节不给——丢弃用户原文必须**逐块**
            #     走「删除该块」那条出口（带 diff、说得出理由），整节一扫而空会把
            #     「不改写、不丢弃任何非空原文」那条契约的可见性抹掉；
            #   · 有**协议层默认值**（`protocol`）的节不给——那些是骨架（按需读取指针
            #     这类），留空会直接把 ship 拦死（`TIMELINE_POINTER_MISSING`），
            #     给一个按不下去的按钮比不给更糟。
            # 这两条不是保守，是各自有一条已经存在的机制在管，别用这个按钮盖过去。
            dropped = sorted({Path(item.source_ref).name for item in section_items})
            versions = list(versions) + [SectionVersion(
                version_id=f"{section}:leave_empty", section=section,
                item_ids=(), markdown="", source_summary=tuple(dropped),
                diff=("本节留空：这一节不写进人格文件。"
                      f"会丢掉 {len(section_items)} 条来自语料的候选，"
                      f"来源：{'、'.join(dropped)}。"
                      "⚠ 这些来源仍然算「已交代」（你明确弃用过了），"
                      "不必回去改候选结果；⚠ 语料本身一个字都不会动。"))]
        versions_by_section[section] = [_section_version_dict(version) for version in versions]
    conflict_payload = [{
        "conflict_id": conflict.conflict_id, "kind": conflict.kind,
        "severity": conflict.severity, "item_ids": list(conflict.item_ids),
        "reason": conflict.reason, "choices": list(conflict.choices),
        "resolved_choice": conflict.resolved_choice,
    } for conflict in conflicts]
    valid_ids = {section: {version_key(version) for version in versions}
                 for section, versions in versions_by_section.items()}
    old_decisions = state.get("section_decisions", {})
    state["section_decisions"] = {
        section: decision for section, decision in old_decisions.items()
        if decision.get("version_id") in valid_ids.get(section, set())
    }
    state["compiler_items"] = [item_to_dict(item) for item in items]
    state["conflicts"] = conflict_payload
    state["section_versions"] = versions_by_section
    state["preview"] = {}
    return state


MAPPING_NOTE_BODY = ("整块逐字移动到一个主节，或暂不出货；没有自由文本入口。"
                     "⚠ 选 leave_unresolved 就出不了货（`SECTION_UNCONFIRMED` 会拦），"
                     "它是「先放着」不是「不要了」。")

MAPPING_NOTE_HEADING = ("⚠ **这一块是一行标题，选哪一节都不会改变出货正文**："
                        "标题由十二节骨架统一渲染一遍，这一块只被逐字认领、不进正文"
                        "（`operation=delete`）。你的选择只决定它算在哪一节的来源清单里。"
                        "⚠ 但**必须选一个**——留着不归出不了货。")


def _mapping_note(item):
    """归节题的说明按块的类型给。

    ⚠ **原来这里是一句写死的常量**，于是**文件自己的那行大标题（`# 核心人格`）
    也被拿同一句话去问**：「整块逐字移动到一个主节」——十二个选项没有一个对得上，
    而**选哪个产出都一字不差**（标题块一律被打成 delete，见
    `apply_original_section_decisions`），题面一个字都没说。
    2026.08.04 走查现场维护者就在那儿认真挑了一个没有区别的答案（台账 08-04 第三条）。

    ⚠ **不是把标题块自动归掉就完了**：归到哪一节仍然影响那一节的来源清单，
    那是用户的判断，我们不替他选——**要补的是「告诉他这个选择影响什么」**。"""
    from persona_compiler import is_heading_block
    return MAPPING_NOTE_HEADING if is_heading_block(item) else MAPPING_NOTE_BODY


def section_choice_payload(state):
    """每节至多一道题；来源、冲突与 diff 只作为版本证据。"""
    from persona_compiler import item_from_dict

    unmapped = [item_from_dict(item) for item in state.get("compiler_items", ())
                if item.get("source_type") == "original_persona"
                and item.get("section") is None]
    sections = []
    decisions = state.get("section_decisions", {})
    # 节标题的人称跟 preview／出货同源一份（`persona_pronouns`）。原来这里直接把
    # `SECTION_ORDER` 的 label 原样发出去，于是**用户在选择题里看到的是字面
    # `{ta}是谁`**——一个没填的模板占位符。`preview_payload` 那处一直是对的
    # （`fill_pronouns(label, pronouns)`），两处不同源，坏的只是先被看到的那处。
    # ⚠ 这是 2026.08.04 走查在真实人格文件上撞出来的（走查台账 08-04 第四条）：
    # 出货文件里一个占位符都没漏，**只有做选择的那一屏漏**，所以自检里那条
    # 「占位符不许漏进出货文件」永远不会红——**它看的不是这一屏**。
    pronouns = persona_pronouns(state)
    for section, label in SECTION_ORDER:
        versions = state.get("section_versions", {}).get(section, [])
        sections.append({
            "section": section,
            "label": fill_pronouns(label, pronouns),
            "status": decisions.get(section, {}).get("status", "pending"),
            "section_versions": [{
                # ⚠ 键名只有 `version_id` 一个（2026.08.05 双键收敛）——
                # 原先这里出 `id`，而 `persona_compiler` 那侧出 `version_id`，
                # 同一个东西两个名字，读的地方就跟着分家
                "version_id": version_key(version),
                "markdown": version["markdown"],
                "source_summary": version.get("source_summary", []),
                "diff": version.get("diff", ""),
                "resolved_conflict_ids": version.get("resolved_conflict_ids", []),
            } for version in versions],
        })
    return {"schema_version": 2, "mode": "compiler_v2",
            "original_mapping_questions": [{
                "item_id": item.item_id,
                "text": item.original_text,
                "source_ref": item.source_ref,
                "choices": [section for section, _label in SECTION_ORDER]
                           + ["leave_unresolved"],
                "note": _mapping_note(item),
            } for item in unmapped],
            "sections": sections,
            "next": "选择每节一个 section_version id，再运行 --step choose-sections"}


def apply_original_section_decisions(state, decisions):
    """给解析失败／跨节原文块选择主节；只移动整块，不拆句、不改字。"""
    from dataclasses import replace
    from persona_compiler import (
        HEADING_DELETE_REASON, is_heading_block, item_from_dict, item_to_dict)

    items = [item_from_dict(item) for item in state.get("compiler_items", ())]
    by_id = {item.item_id: item for item in items
             if item.source_type == "original_persona"}
    unknown = sorted(set(decisions) - set(by_id))
    if unknown:
        raise ValueError("未知原人格块：" + "、".join(unknown))
    valid_sections = {section for section, _label in SECTION_ORDER}
    updated = []
    for item in items:
        choice = decisions.get(item.item_id)
        if choice is None or choice == "leave_unresolved":
            updated.append(item)
            continue
        if choice not in valid_sections:
            raise ValueError(f"未知人格节：{choice}")
        # ⚠ 标题块保持 delete：手工归节是"这块归哪一节"，不是"要不要进正文"。
        #   这里若跟着改成 keep/move，用户手动归一次节，那一节的标题就又出现两遍
        #   ——而且只在"归节判不出来"的那些文件上发作，最难被发现。
        if is_heading_block(item):
            #   理由与 proposed_text 一并给全：旧版 state 里的标题块还是 keep、
            #   `operation_reason` 是空的，只改 operation 会撞 PersonaItem 那条
            #   "rewrite/delete 必须说明理由"的校验。
            updated.append(replace(
                item, section=choice, operation="delete", proposed_text="",
                operation_reason=HEADING_DELETE_REASON,
                confidence="user_mapped", confirmed=False))
            continue
        updated.append(replace(
            item, section=choice,
            operation="keep" if item.section == choice else "move",
            confidence="user_mapped", confirmed=False))
    state = dict(state)
    state["compiler_items"] = [item_to_dict(item) for item in updated]
    state["conflicts"] = []
    state["section_versions"] = {}
    state["section_decisions"] = {}
    state["preview"] = {}
    return state


def apply_section_decisions(state, decisions):
    from persona_compiler import apply_section_choice, state_from_dict, state_to_dict

    compiler = state_from_dict({
        "schema_version": 2,
        "items": state.get("compiler_items", []),
        "conflicts": state.get("conflicts", []),
        "section_versions": state.get("section_versions", {}),
        "section_decisions": state.get("section_decisions", {}),
    })
    for section, version_id in decisions.items():
        if section not in dict(SECTION_ORDER):
            raise ValueError(f"未知人格节：{section}")
        compiler = apply_section_choice(compiler, section, version_id)
    serialized = state_to_dict(compiler)
    state = dict(state)
    state["compiler_items"] = serialized["items"]
    state["section_decisions"] = serialized["section_decisions"]
    state["preview"] = {}
    state["step"] = "sections_chosen"
    return state


STALE_VERSION_EXIT = (
    "（出口：--step choose-sections --json 取该节当前的版本 id，再重新确认一次。"
    "⚠ 这种情况几乎只有两个来源：init_state.json 被手改过，或者版本表在确认之后"
    "重建过——**不是「这一节没确认」**，别去重走确认以外的路）")


def preview_payload(state):
    """完整预览只能机械拼接已选节版本，不在预览阶段再次改写。"""
    lines = ["# 核心人格", ""]
    unresolved = []
    # 已确认、但版本号在版本表里找不到的那些：是 unresolved 的子集，单独列出来是为了
    # 让 CLI 能说对话——同一句「未确认」把一位真实用户指向了改源码。
    # ⚠ **判定必须走 `version_key`，不许改成按条目里的 `id` 键匹配**：写侧只写
    # `version_id`，按 `id` 匹配会把**每一个已确认的节**都判成过期（实测 12/12），
    # 谁都出不了货，而且不报错。
    stale = []
    source_summary = {}
    # 节标题的人称跟正文同源一份（见 persona_pronouns）：原来这里写死 None，
    # 于是写「她」的用户拿到的标题是中性的「对方是谁」——同一份文件里两种称呼。
    pronouns = persona_pronouns(state)
    decisions = state.get("section_decisions", {})
    for section, label in SECTION_ORDER:
        decision = decisions.get(section)
        if not decision or decision.get("status") != "confirmed":
            unresolved.append(section)
            continue
        versions = state.get("section_versions", {}).get(section, [])
        selected = next((version for version in versions
                         if version_key(version) == decision.get("version_id")), None)
        if selected is None:
            # 确认过、但版本号找不到：仍然渲染不出来（所以照旧进 unresolved），
            # 但**成因和出口都跟「没确认」不同**，单独列一份给 CLI 说清楚。
            unresolved.append(section)
            stale.append(section)
            continue
        source_summary[section] = selected.get("source_summary", [])
        if selected.get("markdown", "").strip():
            lines.extend([f"## {fill_pronouns(label, pronouns)}", "",
                          selected["markdown"].rstrip(), ""])
    markdown = "\n".join(lines).rstrip() + "\n"
    warnings = [issue for issue in state.get("diagnostics", [])
                if issue.get("severity") == "warning"]
    import hashlib
    return {
        "persona_markdown": markdown,
        "preview_hash": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
        "source_summary": source_summary,
        "unresolved": unresolved,
        # 已确认但版本号找不到的那些：unresolved 的子集，单列给 CLI 分行印
        "stale_versions": stale,
        "warnings": warnings,
        "return_targets": [section for section, _label in SECTION_ORDER],
    }


def shipping_issues(state, persona_markdown, manifest):
    """v2 出货闸；warning 可展示，blocking 一项都不能带过。"""
    import hashlib
    from persona_compiler import CompileIssue, item_from_dict, original_span_coverage

    issues = [CompileIssue(
        issue["code"], issue["severity"], issue["message"],
        tuple(issue.get("item_ids", ())))
        for issue in list(state.get("diagnostics", []))
        + list((state.get("persona_drift") or {}).get("issues", []))]
    decisions = state.get("section_decisions", {})
    missing_sections = [section for section, _label in SECTION_ORDER
                        if decisions.get(section, {}).get("status") != "confirmed"]
    if missing_sections:
        # 出口写进 message 里（同 TASK_DIRECTIVE_REMAINS 的先例）：这条原来只说"哪几节
        # 没确认"，不说怎么才算确认。2026.08.04 外部实测的代价——用户被拦在这里，
        # 报错没给方向，于是 `cp persona_preview.md memory/persona.md` 手动绕过**整道闸**
        # 出货了（连带跳过 TIMELINE_POINTER_MISSING，而那正是他下一步要验的写回层）。
        # **拦得对但不给出路，人会翻墙。**
        issues.append(CompileIssue(
            "SECTION_UNCONFIRMED", "blocking",
            "以下人格节尚未确认：" + "、".join(missing_sections)
            + "（出口：--step choose-sections --json 拿到每节的版本 id，再 "
              "--step choose-sections --section-decisions-json '{\"节名\": \"版本id\"}' "
              "提交；⚠ 不要去手改 init_state.json，出货闸认的是 section_decisions，"
              "手写 section_versions 不产生任何确认）"))
    # ⚠ **「确认过、但那个版本号在版本表里找不到」必须吵**（2026.08.05，任务卡
    # 《节版本双键收敛》验收判据那句「不许出现某节静默消失」）。
    # ⚠ **判定走 `version_key`，不许改成按条目里的 `id` 键匹配**：写侧只写
    # `version_id`，按 `id` 匹配会把**每一个已确认的节**都判成过期（实测 12/12），
    # 谁都出不了货，而且不报错。
    # ⚠ **也不要再加一道「同一版本的两个键必须相等」的闸**：写侧只有一个键，
    # 那种闸遇到没有 `id` 的条目会直接跳过，一个都检查不到（实测 0/12），
    # 是**空转的闸且不报错**；它拦的「两键不等」在构造上已经不可能出现。
    # 不拦这一条的后果是：
    # `status` 仍是 `confirmed`，所以 `SECTION_UNCONFIRMED` 不报；`preview_payload`
    # 匹配不到版本就把这一节整节跳过——**用户确认过的一节被静默从出货文件里删掉，
    # 退出码 0，全程零提示**（复现步骤与实测记录见那张卡）。
    stale = []
    for section, _label in SECTION_ORDER:
        decision = decisions.get(section, {})
        if decision.get("status") != "confirmed":
            continue                       # 那一档归上面的 SECTION_UNCONFIRMED
        chosen = decision.get("version_id")
        available = [version_key(version)
                     for version in state.get("section_versions", {}).get(section, [])]
        if chosen not in available:
            stale.append(f"{section}（决定指向 {chosen!r}，这一节现有的版本是："
                         + ("、".join(available) if available else "一个都没有") + "）")
    if stale:
        issues.append(CompileIssue(
            "SECTION_VERSION_STALE", "blocking",
            "以下人格节确认过，但确认时选的那个版本号现在找不到了——"
            "多半是重跑 choose-sections 之后版本号变了，或者手改过 init_state.json。"
            "⚠ 不修的话这一节会**整节从出货文件里消失，而且不报错**。"
            "出口：重跑 --step choose-sections 看当前版本号，再选一次。"
            + "；".join(stale)))
    if any(token in persona_markdown for token in TASK_DIRECTIVE_TOKENS):
        issues.append(CompileIssue(
            "TASK_DIRECTIVE_REMAINS", "blocking",
            "人格文件仍含任务书或空指令（出口：--step choose-sections 里那一节有一个"
            "「删除该块」的版本，带 diff，选它即可出货；不必去手改输入文件）"))
    # 禁用词：**报出来，但不拦**（拍板 ②）。⚠ 报的是**逐行**，不是"文件里有「它」"
    # ——用户拿到的是一份上千字的人格文件，只说"有"等于让他自己去 Ctrl+F 猜哪一行
    # 该改；而这条本来就允许保留正当用法，不指到行就没法判。
    banned_lines = [line.strip() for line in persona_markdown.splitlines()
                    if any(token in line for token in BANNED_WORD_TOKENS)]
    if banned_lines:
        shown = "；".join(line[:40] for line in banned_lines[:5])
        issues.append(CompileIssue(
            "BANNED_WORD_REMAINS", "warning",
            f"人格文件里有 {len(banned_lines)} 行含「"
            + "」「".join(BANNED_WORD_TOKENS)
            + f"」——人格文件里的 AI 不写成「它」，多半是旧版工具留下的写法："
            + shown + ("…" if len(banned_lines) > 5 else "")
            + "。⚠ 这条不拦出货（你原文里若是正当用法就留着）；"
              "确实要去掉的话，--step choose-sections 里那一节有一个「删除该块」的"
              "版本，带 diff，选它即可，不必去手改输入文件。"))
    if "timeline" not in persona_markdown:
        issues.append(CompileIssue(
            "TIMELINE_POINTER_MISSING", "blocking",
            "按需读取指针没有覆盖 memory/timeline 写回层"))

    items = [item_from_dict(item) for item in state.get("compiler_items", [])]
    if any(item.source_type == "original_persona" and item.section is None
           for item in items):
        issues.append(CompileIssue(
            "SECTION_UNCONFIRMED", "blocking",
            "仍有原人格块没有归入十二节，不能在预览中静默漏掉",
            tuple(item.item_id for item in items
                  if item.source_type == "original_persona" and item.section is None)))
    if any(issue.code == "NAMING_NOT_BIDIRECTIONAL" for issue in issues):
        issues.append(CompileIssue(
            "NAMING_INCOMPLETE", "blocking", "称呼系统必须双向并区分场景"))

    groups = {}
    for item in items:
        if item.group_id:
            groups.setdefault(item.group_id, []).append(item)
    for grouped in groups.values():
        originals = [item for item in grouped if item.source_type == "original_persona"]
        protocols = [item for item in grouped if item.source_type == "protocol"]
        if originals and protocols and any(item.confirmed for item in protocols) and not any(
                item.confirmed for item in originals):
            issues.append(CompileIssue(
                "PROTOCOL_OVERRIDES_ORIGINAL", "blocking",
                "协议默认值不能覆盖同一机制中的个性化原文",
                tuple(item.item_id for item in grouped)))
    coverage_items = [item for item in items
                      if item.source_ref == "protocol:COVERAGE_TEMPLATE"]
    if any(not item.derived_from for item in coverage_items):
        issues.append(CompileIssue(
            "DERIVED_PROTOCOL_PROVENANCE_MISSING", "blocking",
            "记忆覆盖范围缺模板或语料派生证据"))
    for item in items:
        if item.operation in {"rewrite", "delete"} and (
                decisions.get(item.section, {}).get("status") != "confirmed"):
            issues.append(CompileIssue(
                "SECTION_WITH_DIFF_UNCONFIRMED", "blocking",
                f"含原文删改的节尚未确认：{item.section}", (item.item_id,)))

    selected_versions = {}
    for section, decision in decisions.items():
        selected_versions[section] = next((version for version in
            state.get("section_versions", {}).get(section, [])
            if version_key(version) == decision.get("version_id")), None)
    # 【旧断言盖新内容】2026.08.05 外部实测第 3 条：原人格 milestones 节首写着
    # 「（以下五条都经她逐条确认…）」，合并语料候选后那一节变成 14 条，**这句断言被
    # 原样保留、盖到了 9 条没经确认的新条目头上**——出货闸一声不吭。
    # ⚠ **判据必须落在「用户选中的那个版本」上，不能像 task_directive_delete_items
    #   那样按条目判**：那句原文和它的「删除该块」孪生条目**同时躺在 compiler_items
    #   里**（选择发生在版本层），按条目判的话用户选了删除版本之后这条 warning 照报
    #   ——一条改不掉的警告等于没有警告。所以这里看的是选中版本的 markdown。
    # ⚠ **warning 不是 blocking**，同 BANNED_WORD_REMAINS 的先例：那个数可能指的是
    #   别的东西（正当用法），而我们**不许改写用户原文**，拦死等于替他判断他的原文。
    #   给的是"看得见 ＋ 有出口"（出口就在那一节的「删除该块」版本里）。
    stale_counts = []
    for section, version in selected_versions.items():
        if not version:
            continue
        version_ids = set(version.get("item_ids", ()))
        joined = sum(1 for item in items
                     if item.item_id in version_ids and item.source_type == "corpus")
        if not joined:
            continue
        for line in version.get("markdown", "").splitlines():
            match = _COUNT_ASSERTION_RE.search(line)
            if match:
                stale_counts.append((section, line.strip(), match.group(0), joined))
    if stale_counts:
        issues.append(CompileIssue(
            "STALE_COUNT_ASSERTION", "warning",
            "以下几处是原文里的**数量断言**，而同一节这次并进了语料里的新内容——"
            "那句话现在盖到了它没数过的条目头上（原文一个字都没被我们改）："
            + "；".join(f"{section} 节「{line[:40]}」（命中 {hit}，本节并进 {joined} 条）"
                       for section, line, hit, joined in stale_counts)
            + "。⚠ 这条不拦出货。出口：--step choose-sections 里那一节有一个"
              "「删除该块」的版本，带 diff，选它即可；确认那个数指的是别的东西，"
              "原样留着也行。"))

    for conflict in state.get("conflicts", []):
        if conflict.get("severity") != "blocking":
            continue
        section = next((item.section for item in items
                        if item.item_id in conflict.get("item_ids", [])), None)
        selected = selected_versions.get(section) or {}
        if conflict["conflict_id"] not in selected.get("resolved_conflict_ids", []):
            issues.append(CompileIssue(
                "BOUNDARY_CONFLICT_UNRESOLVED", "blocking",
                "硬边界冲突尚未由节版本裁定", tuple(conflict.get("item_ids", ()))))

    if manifest is not None:
        persona_file = manifest.persona_file
        if persona_file and any(str(persona_file) == str(path) for path in manifest.corpus_files):
            issues.append(CompileIssue(
                "PERSONA_MIXED_INTO_CORPUS", "blocking", "原人格仍在语料写入清单中"))
        if persona_file and persona_file.exists():
            source = persona_file.read_text(encoding="utf-8")
            original_items = [item for item in items if item.source_type == "original_persona"]
            if original_span_coverage(source, original_items) != 1.0:
                issues.append(CompileIssue(
                    "ORIGINAL_COVERAGE_INCOMPLETE", "blocking", "原人格有非空原文无人认领"))
            expected_hash = state.get("source_manifest", {}).get(
                "source_hashes", {}).get(str(persona_file), "")
            actual_hash = manifest.source_hashes.get(str(persona_file), "")
            if expected_hash and expected_hash != actual_hash:
                issues.append(CompileIssue(
                    "ORIGINAL_SOURCE_CHANGED", "blocking", "确认后原人格文件发生变化"))
        known = {str(path) for path in manifest.corpus_files}
        accounted = {str(record.get("source_ref"))
                     for record in state.get("source_accounting", [])}
        if known - accounted:
            issues.append(CompileIssue(
                "SOURCE_ACCOUNTING_INCOMPLETE", "blocking", "仍有输入来源未被候选结果交代"))
        for item in items:
            if item.source_type == "corpus" and item.source_ref not in known:
                issues.append(CompileIssue(
                    "SOURCE_UNKNOWN", "blocking", "具体关系事实来源不在清单",
                    (item.item_id,)))

    preview_hash = state.get("preview", {}).get("preview_hash")
    actual_preview_hash = hashlib.sha256(persona_markdown.encode("utf-8")).hexdigest()
    if preview_hash and preview_hash != actual_preview_hash:
        issues.append(CompileIssue(
            "PREVIEW_CHANGED", "blocking", "已展示预览与当前待出货人格不一致"))
    return issues


def _drift_section_label(key):
    from persona_compiler import UNMAPPED_SECTION_KEY

    if key == UNMAPPED_SECTION_KEY:
        return "未归节"
    return dict(SECTION_ORDER).get(key, key)


def persona_drift_report(previous_state, manifest, current_items, acknowledged=False):
    """跨运行比对输入人格文件：哈希变了就把「各节原文块数」的差异摊开。

    **这是唯一能发现静默塌节的地方**（2026.08.03 走查第一条）。归节全靠 markdown
    标题行做锚点（`persona_compiler.parse_original_text`），删掉一行标题，后面的块
    一路顺延继承上一节、整节塌进邻节；而所有既有闸门都看不见：
    `original_span_coverage` 的分母是**本次传进来那份 text 现算的**，删掉的字符不在
    分母里，覆盖率照样 1.0；`init_state.json` 里虽然存了上次的 `source_hashes`，
    但 `--step inspect` 重跑时直接用新 manifest 覆盖，没有做过跨运行比对。
    于是走查里真实发生的形态是：**出货成功、零 blocking、零 warning、`--doctor` 也过**，
    而用户原文整段没了。

    ⚠ **光打 warning 不够**：warning 在一屏输出里等于没有，而整节塌空正是走查里
    真发生的那一种。所以「某节上次有 N 块、这次 0 块」直接升级成 blocking。

    ⚠ blocking 必须有出口，否则就重蹈现象 B 的覆辙（拦得对但把用户逼去手改输入
    文件，而手改输入文件正是本现象的雷区）：出口是 `--accept-persona-drift`，
    用户确认「就是我有意删的」之后降级成 warning，不改用户任何文件。

    返回 None＝没有可比的上一次，或这份文件跟上次逐字相同。"""
    from persona_compiler import (
        CompileIssue, compare_original_block_counts, original_block_counts)

    persona_file = manifest.persona_file
    if persona_file is None or not isinstance(previous_state, dict):
        return None
    if previous_state.get("schema_version") != 2:
        return None
    previous_hashes = (previous_state.get("source_manifest") or {}).get("source_hashes") or {}
    # 键在不在，本身就是「上次跑的是不是同一份文件」的判据：manifest 那侧存的是
    # resolve 之后的绝对路径，两边同源，不用再猜相对路径。
    before_hash = previous_hashes.get(str(persona_file), "")
    after_hash = manifest.source_hashes.get(str(persona_file), "")
    if not before_hash or before_hash == after_hash:
        return None
    previous_counts = original_block_counts(previous_state.get("compiler_items", ()))
    if not previous_counts:
        return None
    current_counts = original_block_counts(current_items)
    rows, collapsed = compare_original_block_counts(previous_counts, current_counts)

    return _drift_payload(str(persona_file), before_hash, after_hash,
                          rows, collapsed, acknowledged)


def _drift_payload(persona_file, before_hash, after_hash, rows, collapsed, acknowledged):
    """把差异表渲染成用户看得懂的两条 issue。级别只由 acknowledged 决定。"""
    from persona_compiler import CompileIssue

    table = [f"{_drift_section_label(row['section'])}：上次 {row['before']} 块"
             f" → 这次 {row['after']} 块" for row in rows]
    issues = [CompileIssue(
        "PERSONA_SOURCE_DRIFT", "warning",
        "输入人格文件跟上次 inspect 时不是同一份（sha256 变了）。本次归入各节的"
        "原文块数 vs 上次：" + ("；".join(table) if table else "各节块数没有变化")
        # ⚠ 口径写在数字旁边：光给「上次 3 块 → 这次 2 块」，用户没法自己分辨
        #   少的那块是内容没了、还是我们数的东西变了（标题块 2026.08.04 起打
        #   delete、不进正文）。两边都不数标题，所以这个数不会因为那次改动而跳。
        + "。（只数正文块，不含 markdown 标题行——标题由十二节骨架统一渲染，"
        "两次都不计入。）")]
    before_by_section = {row["section"]: row["before"] for row in rows}
    if collapsed:
        detail = "、".join(
            f"{_drift_section_label(key)}（上次 {before_by_section.get(key, 0)} 块）"
            for key in collapsed)
        issues.append(CompileIssue(
            "PERSONA_SECTION_COLLAPSED",
            "warning" if acknowledged else "blocking",
            f"这几节这次一个原文块都没有、上次有：{detail}。"
            "最常见的原因是那一节的 markdown 标题行被删了——归节靠标题锚点，"
            "标题没了整节会顺延塌进上一节，而逐字覆盖率闸门看不见这件事。"
            + ("（已用 --accept-persona-drift 确认是有意删的）" if acknowledged else
               "确认就是你有意删掉这些内容，就加 --accept-persona-drift 重跑本步；"
               "想找回来就把那几行标题加回输入文件再重跑。")))
    return {
        "persona_file": persona_file,
        "before_hash": before_hash,
        "after_hash": after_hash,
        "acknowledged": bool(acknowledged),
        "table": rows,
        "collapsed_sections": collapsed,
        "issues": [_compile_issue_payload(issue) for issue in issues],
    }


def carry_forward_persona_drift(previous_state, manifest, acknowledged=False):
    """把上一次 inspect 记下的塌节结论**带过这一次**。

    ⚠ **少了这一支，`--accept-persona-drift` 是个假出口**：inspect 会把 state 里的
    `source_hashes` 换成这次的，于是紧接着再跑一次 inspect 时「上次」已经是塌过节
    的那一份，哈希一样、比不出差异，blocking 自己消失了——**用户加不加这个开关都
    一样能出货，而那条 warning 也跟着蒸发**，等于既没有出口也没有留痕。

    所以：只要这次读到的还是同一份、同一内容的人格文件，就把上次那份结论原样带过来，
    级别按这次给不给开关重算；确认过一次就一直算确认过（写在 state 里）。"""
    if not isinstance(previous_state, dict) or manifest.persona_file is None:
        return None
    previous = previous_state.get("persona_drift") or {}
    persona_file = str(manifest.persona_file)
    if previous.get("persona_file") != persona_file:
        return None
    if previous.get("after_hash") != manifest.source_hashes.get(persona_file, ""):
        return None
    return _drift_payload(
        persona_file, previous.get("before_hash", ""), previous.get("after_hash", ""),
        list(previous.get("table", ())), list(previous.get("collapsed_sections", ())),
        bool(acknowledged) or bool(previous.get("acknowledged")))


def _v2_inputs(args, state=None):
    state = state or {}
    saved = state.get("inputs", {})
    return args.persona or saved.get("persona"), args.corpus or saved.get("corpus")


def _existing_persona_paths(out_dir, state):
    """只返回产出目录中的已知人格相对路径，不递归猜测其他 Markdown。"""
    out = Path(out_dir)
    names = []
    previous = state.get("last_shipped_persona")
    if previous in CLIENT_FILENAMES.values():
        names.append(previous)
    names.extend(name for name in CLIENT_FILENAMES.values() if name not in names)
    return [out / name for name in names if (out / name).is_file()]


def _step_inspect_v2(args, existing_state=None):
    from persona_compiler import build_source_manifest, item_to_dict, parse_original_persona

    existing_state = existing_state or {}
    persona_path = args.persona
    existing_personas = [] if persona_path else _existing_persona_paths(args.out, existing_state)
    if existing_personas and not args.existing_persona_choice:
        payload = {
            "schema_version": 2,
            "mode": "migration_choice",
            "existing_persona_files": [str(path.resolve()) for path in existing_personas],
            "choices": ["treat_current_as_original", "use_original_as_is",
                        "continue_legacy"],
            "next": "用 --existing-persona-choice 选择；不会自动采用首项",
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print("检测到已有的人格文件。请选择把它作为原人格，或继续旧流程。")
        return
    if args.existing_persona_choice == "continue_legacy":
        payload = {"mode": "legacy_v1", "choice": "continue_legacy",
                   "next": "继续原 questionnaire → answers → confirm → route → ship 流程"}
        if args.json:
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print("保留 v1 状态，继续旧初始化流程。")
        return
    if existing_personas:
        persona_path = existing_personas[0]

    manifest = build_source_manifest(persona_path, args.corpus)
    if args.existing_persona_choice == "use_original_as_is":
        if not manifest.persona_file:
            raise SystemExit("原人格直接接入要跟 --persona <现有人格文件> 一起用。")
        state = new_v2_state(persona_path, args.corpus)
        state.update({
            "mode": "direct_persona",
            "step": "inspected",
            "source_manifest": _manifest_payload(manifest),
            "source_hash": manifest.source_hashes.get(str(manifest.persona_file), ""),
            "skipped_steps": ["extract", "choose-sections", "preview"],
        })
        save_state(args.out, state)
        payload = {
            "schema_version": 2,
            "mode": "direct_persona",
            "source_manifest": state["source_manifest"],
            "skipped_steps": list(state["skipped_steps"]),
            "blocking_issues": [issue for issue in state["source_manifest"]["issues"]
                                if issue["severity"] == "blocking"],
            "warnings": [issue for issue in state["source_manifest"]["issues"]
                         if issue["severity"] == "warning"],
            "note": "这是用户主动选择跳过三项编译检查，不代表这些检查已经通过；"
                    "原人格正文不拆块、不重排，出货副本只追加 Latent 受管记忆协议。",
            "next": "--step route",
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print("已选择原人格直接接入：跳过 extract／choose-sections／preview。"
                  "跳过不等于通过编译检查。下一步：--step route。")
        return

    state = new_v2_state(persona_path, args.corpus)
    state["source_manifest"] = _manifest_payload(manifest)
    if manifest.persona_file and not manifest.issues:
        state["compiler_items"] = [item_to_dict(item)
                                   for item in parse_original_persona(manifest.persona_file)]
    state["diagnostics"] = state["source_manifest"]["issues"]
    # ⚠ 跨运行比对的结论**不能**放进 diagnostics：`--step extract --candidates`
    #   会整个覆盖 diagnostics（见 _step_extract_v2），放那儿等于跑一步就没了。
    #   单开一格，出货闸从这一格读。
    acknowledged = getattr(args, "accept_persona_drift", False)
    drift = persona_drift_report(
        existing_state, manifest, state["compiler_items"], acknowledged=acknowledged)
    if drift is None:
        drift = carry_forward_persona_drift(existing_state, manifest, acknowledged)
    if drift:
        state["persona_drift"] = drift
    state["step"] = "inspected"
    save_state(args.out, state)
    drift_issues = (drift or {}).get("issues", [])
    payload = {
        "schema_version": 2,
        "mode": "compiler_v2",
        "source_manifest": state["source_manifest"],
        "original_item_count": len(state["compiler_items"]),
        "original_blocks": [{
            "item_id": item["item_id"], "text": item["original_text"],
            "section": item["section"], "confidence": item["confidence"],
            "source_ref": item["source_ref"],
        } for item in state["compiler_items"]],
        "persona_drift": drift,
        "blocking_issues": [issue for issue in state["diagnostics"] + drift_issues
                            if issue["severity"] == "blocking"],
        "warnings": [issue for issue in state["diagnostics"] + drift_issues
                     if issue["severity"] == "warning"],
        "next": "--step extract",
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        for issue in drift_issues:
            print(f"[{issue['severity']}] {issue['code']}：{issue['message']}")
        print("输入检查完成。下一步：--step extract --json")


def _step_extract_v2(args, state):
    from draft_extraction import build_extraction_package, load_candidate_result
    from persona_compiler import build_source_manifest, item_to_dict

    persona_path, corpus_path = _v2_inputs(args, state)
    manifest = build_source_manifest(persona_path, corpus_path)
    if not state or state.get("schema_version") != 2:
        state = new_v2_state(persona_path, corpus_path)
    state["inputs"] = {"persona": str(persona_path) if persona_path else None,
                       "corpus": str(corpus_path) if corpus_path else None}
    state["source_manifest"] = _manifest_payload(manifest)
    package = build_extraction_package(manifest, Path(args.out) / "persona-extraction")
    package_payload = {
        "prompt": str(package.prompt_path.resolve()),
        "manifest": str(package.manifest_path.resolve()),
        "schema": str(package.schema_path.resolve()),
    }
    state["extraction_package"] = package_payload
    state["step"] = "extracted"
    if args.candidates:
        candidate_data = _load_json_arg(args.candidates)
        candidate_path = Path(args.out) / "persona-extraction" / "候选结果.json"
        candidate_path.write_text(
            json.dumps(candidate_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        items, issues = load_candidate_result(candidate_path, manifest)
        preserved_items = [item for item in state.get("compiler_items", [])
                           if item.get("source_type") in {
                               "original_persona", "protocol"}]
        state["compiler_items"] = preserved_items + [item_to_dict(item) for item in items]
        # `--candidates` 是替换本轮语料候选，不是往旧候选后面追加。旧版本表已经
        # 引用了被替换的候选；留着它会让 choose-sections 跳过重建，并可能把协议项
        # 从版本表中一并丢掉。候选变了就必须重新选择，不能复用旧决定。
        state.pop("section_versions", None)
        state.pop("section_decisions", None)
        state.pop("conflicts", None)
        state.pop("preview", None)
        state["source_accounting"] = candidate_data.get("source_accounting", [])
        state["diagnostics"] = [_compile_issue_payload(issue) for issue in issues]
        state["step"] = "candidates_loaded"
    save_state(args.out, state)
    payload = {
        "schema_version": 2, "mode": "compiler_v2", "package": package_payload,
        "candidate_count": len([item for item in state.get("compiler_items", [])
                                if item.get("source_type") == "corpus"]),
        "blocking_issues": [issue for issue in state.get("diagnostics", [])
                            if issue["severity"] == "blocking"],
        "warnings": [issue for issue in state.get("diagnostics", [])
                     if issue["severity"] == "warning"],
        "next": ("把当前模型的严格 JSON 结果传给 --candidates"
                 if not args.candidates else "--step choose-sections"),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print("人格候选任务包已生成：")
        for path in package_payload.values():
            print(f"- {path}")


def _step_choose_sections_v2(args, state):
    if state.get("schema_version") != 2:
        raise SystemExit("还没有 v2 初始化状态。先运行 --step inspect。")
    if args.original_section_decisions_json:
        mappings = _load_json_arg(args.original_section_decisions_json)
        if not isinstance(mappings, dict):
            raise SystemExit("original section decisions 必须是 {item_id: section} JSON 对象")
        try:
            state = apply_original_section_decisions(state, mappings)
        except ValueError as exc:
            raise SystemExit(str(exc))
    if not state.get("section_versions"):
        state = prepare_section_versions(state)
    if args.section_decisions_json:
        decisions = _load_json_arg(args.section_decisions_json)
        if not isinstance(decisions, dict):
            raise SystemExit("section decisions 必须是 {section: version_id} JSON 对象")
        try:
            state = apply_section_decisions(state, decisions)
        except ValueError as exc:
            raise SystemExit(str(exc))
    save_state(args.out, state)
    payload = section_choice_payload(state)
    payload["unresolved"] = [
        question["section"] for question in payload["sections"]
        if question["status"] != "confirmed"]
    payload["next"] = ("--step preview" if not payload["unresolved"]
                       else "继续为 unresolved 节选择 section_version id")
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        for question in payload["sections"]:
            print(f"\n## {question['label']} [{question['status']}]")
            for version in question["section_versions"]:
                # markdown 空的那种版本（本节留空）在屏幕上只剩一个 id ＋ 一行空白，
                # **说明写在 diff 里，不打出来等于没给**——同「v2 的 warning 只塞进
                # state」那条一个形状（2026.08.05 两处都撞到）
                print(f"- {version['version_id']}\n{version['markdown'] or version['diff']}")


def _step_preview_v2(args, state):
    if state.get("schema_version") != 2:
        raise SystemExit("还没有 v2 初始化状态。先运行 --step inspect。")
    if not state.get("section_versions"):
        state = prepare_section_versions(state)
    payload = preview_payload(state)
    state["preview"] = {
        "preview_hash": payload["preview_hash"],
        "source_hashes": dict(state.get("source_manifest", {}).get("source_hashes", {})),
    }
    state["step"] = "previewed"
    save_state(args.out, state)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(payload["persona_markdown"])
        # ⚠ 这两行分开印：把 stale 的节混进「未确认节」，说的就是**现象而不是处境**，
        # 而那句话把一位真实用户指向了「确认步骤没生效」→ 翻源码 → 改匹配逻辑。
        stale = payload.get("stale_versions") or []
        not_confirmed = [section for section in payload["unresolved"] if section not in stale]
        if not_confirmed:
            print("未确认节：" + "、".join(not_confirmed))
        if stale:
            print("以下节**已确认**，但它记的版本 id 在当前版本表里找不到，"
                  "所以渲染时被整节丢掉了：" + "、".join(stale) + STALE_VERSION_EXIT)
        if not payload["unresolved"]:
            print("预览已固定。下一步：--step route，再 --step ship。")


def _step_ship_v2(args, state):
    from persona_compiler import build_source_manifest, item_from_dict

    route = args.route or state.get("route")
    if route not in RETRIEVAL_ROUTES:
        raise SystemExit("不出货：检索路线还没选过。先运行 --step route --route <key>。")
    persona_path, corpus_path = _v2_inputs(args, state)
    manifest = build_source_manifest(persona_path, corpus_path)
    payload = preview_payload(state)
    issues = shipping_issues(state, payload["persona_markdown"], manifest)
    blocking = [issue for issue in issues if issue.severity == "blocking"]
    if blocking:
        raise SystemExit("不出货：" + "；".join(
            f"{issue.code}：{issue.message}" for issue in blocking))
    # ⚠ **warning 必须打到屏幕上**（2026.08.05，做「零它」那条 warning 时发现的）：
    # 在此之前 v2 出货路径的 warning **只被塞进 `state["shipping"]["warnings"]`**，
    # preview 也不打——于是"给一条 warning"这种修法在这条路上等于什么都没做，
    # **报了但没人看见跟没报是一回事**。这一行治的是那个，不只服务「零它」那一条。
    warnings = [issue for issue in issues if issue.severity == "warning"]
    for issue in warnings:
        print(f"⚠ {issue.code}：{issue.message}")
    if warnings:
        print()
    persona = build_persona_from_items(
        [item_from_dict(item) for item in state.get("compiler_items", [])],
        pronouns=persona_pronouns(state))
    client, switched = resolve_client(args.client, state.get("client"))
    if switched:
        print(switched + "\n")
    entries = _load_ship_entries(args)
    active_corpus = None if args.import_path else corpus_path
    try:
        paths = write_bundle(
            args.out, persona, client=client, corpus_dir=active_corpus,
            confirmed=True, previous_persona=state.get("last_shipped_persona"),
            route=route, validation_mode="compiler_v2",
            rendered_override=payload["persona_markdown"], add_coverage=False,
            entries=entries, corpus_overwrite=_corpus_overwrite_mode(args))
    except (PermissionError, ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"不出货：{exc}")
    print("【四件套】")
    print(f"  人格文件：{paths['persona']}")
    for line in memory_report_lines(paths, active_corpus):
        print(line)
    print(f"  MCP 配置：{paths['mcp_config']}")
    print(f"  引导句：{paths['guidance']}")
    print("\n" + ship_note(client))
    state["step"] = "shipped"
    state["client"] = client
    state["route"] = route
    state["last_shipped_persona"] = shipped_persona_state_value(args.out, paths["persona"])
    state["shipping"] = {
        "compiler_gates_passed": True,
        "preview_hash": payload["preview_hash"],
        "warnings": [_compile_issue_payload(issue) for issue in issues
                     if issue.severity == "warning"],
    }
    save_state(args.out, state)


def _step_ship_direct(args, state):
    """原人格直用出货：用户选择跳过编译，绝不把这条路记成编译闸门已通过。"""
    from persona_compiler import build_source_manifest

    route = args.route or state.get("route")
    if route not in RETRIEVAL_ROUTES:
        raise SystemExit("不出货：检索路线还没选过。先运行 --step route --route <key>。")
    persona_path, corpus_path = _v2_inputs(args, state)
    manifest = build_source_manifest(persona_path, corpus_path)
    blocking = [issue for issue in manifest.issues if issue.severity == "blocking"]
    if blocking:
        detail = "；".join(f"{issue.code}：{issue.message}" for issue in blocking)
        raise SystemExit(f"不出货：来源检查仍有阻断项：{detail}")
    if not manifest.persona_file:
        raise SystemExit("不出货：原人格文件不存在。重新 --step inspect 并指明 --persona。")
    actual_hash = manifest.source_hashes.get(str(manifest.persona_file), "")
    if not state.get("source_hash") or actual_hash != state["source_hash"]:
        raise SystemExit(
            "不出货：ORIGINAL_SOURCE_CHANGED：原人格在 inspect 后发生变化。"
            "请重新 --step inspect --existing-persona-choice use_original_as_is，"
            "不要把新原文和旧选择混在一起。")

    source_text = manifest.persona_file.read_text(encoding="utf-8")
    try:
        rendered, warning_codes = render_direct_persona(source_text)
    except ValueError as exc:
        raise SystemExit(f"不出货：{exc}")
    for code in warning_codes:
        print(f"⚠ {code}：原人格在 Latent 受管标记外已有记忆检索约定；"
              "我们不会替用户删除或合并，出货副本可能出现重复说明。")
    if warning_codes:
        print()

    # write_bundle 只借来复用目录、配置、备份与客户端出货；真正的人格文本由
    # rendered_override 提供。协议默认 Persona 没有用户草稿，不会凭空触发确认关卡。
    persona = Persona("partner")
    fill_protocol_defaults(persona)
    client, switched = resolve_client(args.client, state.get("client"))
    if switched:
        print(switched + "\n")
    if args.timezone:
        state["timezone"] = args.timezone
    entries = _load_ship_entries(args)
    active_corpus = None if args.import_path else corpus_path
    try:
        paths = write_bundle(
            args.out, persona, client=client, corpus_dir=active_corpus,
            confirmed=True, previous_persona=state.get("last_shipped_persona"),
            route=route, validation_mode="compiler_v2", rendered_override=rendered,
            add_coverage=False, timezone=args.timezone or state.get("timezone"),
            entries=entries, corpus_overwrite=_corpus_overwrite_mode(args))
    except (PermissionError, ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"不出货：{exc}")

    print("【四件套】")
    print(f"  人格文件：{paths['persona']}")
    for line in memory_report_lines(paths, active_corpus):
        print(line)
    print(f"  MCP 配置：{paths['mcp_config']}")
    print(f"  引导句：{paths['guidance']}")
    if paths.get("overwritten_backup"):
        print(f"\n⚠ 原人格与出货目标是同一文件或目标已有不同内容；覆盖前已备份成："
              f"{paths['overwritten_backup']}")
    print("\n【直接接入】这是用户选择跳过 extract／choose-sections／preview 的结果；"
          "不代表十二节编译检查已经通过。")
    print("若运行在临时云端容器，latent_append 写回后必须提交并 "
          "git push origin HEAD，再合回默认分支；只留在容器磁盘会随回收丢失。")
    print("\n" + route_note(route))
    print("\n" + ship_note(client))

    state["step"] = "shipped"
    state["client"] = client
    state["route"] = route
    state["last_shipped_persona"] = shipped_persona_state_value(args.out, paths["persona"])
    state["shipping"] = {
        "compiler_gates_passed": False,
        "direct_persona_selected": True,
        "skipped_steps": list(state.get("skipped_steps", ())),
        "warnings": warning_codes,
    }
    # 输入与目标同路径时，第一次出货本身就是用户确认过的改动；把哈希推进到我们刚写下
    # 的副本，下一次才能幂等替换受管块。目标在别处时输入哈希保持 inspect 时的值。
    if manifest.persona_file.resolve() == paths["persona"].resolve():
        import hashlib
        state["source_hash"] = hashlib.sha256(
            paths["persona"].read_bytes()).hexdigest()
    save_state(args.out, state)


def _step_questionnaire(args):
    # 人称有两个来源，**用户自己写下的那个优先**：v2 那条路上他的人格文件里
    # 就有一行 `## 她是谁`（`persona_pronouns` 读的就是它，比任何统计都准）；
    # 没有那份文件才退回语料统计。⚠ 两个都没有就是 `{}`，渲染层走中性写法——
    # **不猜、不塞默认的他／她**（同 `pronouns_from_answers`）。
    detected = persona_pronouns(load_state(args.out) or {}) or {}
    detected = {k: v for k, v in detected.items() if v}
    if not detected:
        detected = _detect_pronouns_from_corpus(args.corpus)
    persona = Persona("partner")
    fill_protocol_defaults(persona, detected)
    report = coverage_report(persona, pronouns=detected)
    if args.json:
        qs = questions_for(report, has_corpus=bool(args.corpus), pronouns=detected)
        save_state(args.out, {"step": "questionnaire", "client": _client_of(args),
                              "has_corpus": bool(args.corpus),
                              "pronouns_detected": detected})
        print(json.dumps({
            "coverage": [{"section": s, "status": st, "note": n} for s, st, n in report],
            "questions": _questions_payload(qs, detected),
            "pronouns_detected": detected,
            "freeform_policy": FREEFORM_POLICY.format(n=FREEFORM_MAX_CHARS),
            "next": "把这些题**在对话里**一题一题问对方（不是让 TA 写作文，选项念给 TA 听）；"
                    "收齐后跑 --step answers --answers-json <JSON>，"
                    "格式：{\"题目qid\": {\"keys\": \"AC\", \"note\": \"可选的一句补充\"}} ；"
                    "pick/short 题用 {\"pick\": \"选中或写下的原文\"}。",
        }, ensure_ascii=False, indent=1))
        return
    print("【覆盖度体检】")
    for _, status, note in report:
        mark = {"ok": "✓", "missing": "缺", "vague": "空泛", "protocol": "系统"}[status]
        print(f"  [{mark}] {note}")
    qs = questions_for(report, has_corpus=bool(args.corpus), pronouns=detected)
    print(f"\n【要问的问题】共 {len(qs)} 题（协议层已由系统填好，不问你）\n")
    print(format_questionnaire(qs, has_corpus=bool(args.corpus), pronouns=detected))
    prompt_path = Path(args.out) / "问卷prompt.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(export_llm_prompt(qs, pronouns=detected), encoding="utf-8")
    save_state(args.out, {"step": "questionnaire", "client": _client_of(args),
                          "has_corpus": bool(args.corpus),
                          "pronouns_detected": detected})
    print(f"\n【下一步】把 {prompt_path} 的内容整段粘给你自己的模型（DeepSeek/ChatGPT 都行），"
          f"让它一题一题问你；答完让它把清单整理好，存成一个文本文件，"
          f"再跑：--step answers --answers <那个文件>")


def _report_unsourced_picks(questions, answers, corpus_dir):
    """查无出处的 pick 答案摆到台面上。**只在给了 --corpus 时能查**——
    语料路径没存进 state（只存了 has_corpus），没给就不查，也不假装查过。"""
    if not corpus_dir:
        return
    bad = unsourced_picks(questions, answers, corpus_dir)
    if not bad:
        return
    print()
    for qid, text in bad:
        print(f"  - {qid}：{text}")
    print(PICK_SOURCE_NOTE)


def _step_answers(args, state):
    _, qs = _rebuild(state)
    if args.answers_json:
        # AI 驱动路径：答案本来就在驱动方手里，直接收结构化数据。
        # 不走 parse_answer_sheet——那套连猜带解析的机制是为"另一个模型吐回来的
        # 不可控文本"准备的，AI 手上已有结构时再序列化成清单、再解析回来，
        # 是凭空多一趟往返和一整个解析失败面。
        answers = _load_json_arg(args.answers_json)
        qids = {q.qid for q in qs}
        unknown = sorted(set(answers) - qids)
        if unknown:   # 不静默丢：认不出的 qid 明确报出来
            raise SystemExit(f"这些 qid 不在当前问卷里：{unknown}\n当前问卷：{sorted(qids)}")
        answers = {k: v for k, v in answers.items() if v is not None}
        state.update(step="answers", answers=answers)
        save_state(args.out, state)
        got = len(answers)
        print(f"【答案读回】收下 {got}/{len(qs)} 题（结构化输入，无解析环节）。"
              + ("" if got == len(qs) else f" 没答的 {len(qs)-got} 题对应的节会空着——"
                                           "空着是诚实的，不要替对方编。"))
        _report_unsourced_picks(qs, answers, args.corpus)
        print("\n【下一步】--step confirm --list --json 取待确认草稿，"
              "**逐条念给对方听、由对方决定**，再用 --decisions-json 落决定。")
        return
    if not args.answers:
        raise SystemExit("这一步要 --answers <答案清单文件> 或 --answers-json <JSON>")
    text = Path(args.answers).read_text(encoding="utf-8")
    answers, problems = parse_answer_sheet(text, qs)
    print("【答案读回】")
    print(answer_report(qs, answers, problems))
    _report_unsourced_picks(qs, answers, args.corpus)
    state.update(step="answers", answers={k: v for k, v in answers.items()
                                          if v is not None})
    save_state(args.out, state)
    if problems:
        print("\n读不懂的行不会静默丢——上面已原样列出。改好清单后重跑这一步即可"
              "（重跑会整体覆盖，不会跟旧答案混在一起）。")
    print("\n【下一步】--step confirm 逐条确认草稿（可中断，随时续跑）。")


def _step_confirm(args, state):
    persona, _ = _rebuild(state)
    decisions = state.setdefault("decisions", {})
    pend = pending_confirmations(persona)
    # --- AI 驱动路径（--list 取待确认清单、--decisions-json 落决定）---
    #
    # 为什么必须有这条路：确认关卡原来只有 input() 交互循环一条路，而 AI 驱动
    # 时它没法好好走，只能盲管道灌 y——那恰好违反引导指南纪律三"人格文件每条
    # 都要念给对方听、对方说可以才留"。CLI 只给交互一条路，等于逼着驱动方
    # 破坏我们自己定的纪律，而且破坏得无声无息（文件照常生成，看着一切正常）。
    # 拆成"取清单 / 落决定"两个非交互动作后，中间那一步——真的问人——回到
    # 对话里，那本来就是它该待的地方。
    if args.list:
        payload = [{"key": p.key, "kind": p.kind, "label": p.label, "value": p.value}
                   for p in pend]
        # 任务书**不混进待确认清单**——它不是给用户确认的内容，是给模型的指令，
        # 单独一个字段给出去（缺陷一的修法：两层分开，不靠确认关卡去分辨）
        _, all_qs = _rebuild(state)
        brief = extraction_brief(all_qs, state.get("answers") or {})
        if args.json:
            print(json.dumps({
                "pending": payload,
                "extraction_brief": brief,
                "brief_note": BRIEF_NOTE if brief else "",
                "next": "**逐条念给对方听，由对方决定**，不要替 TA 一路 keep——"
                        "人格文件的每一条都要 TA 认过（引导指南纪律三）。"
                        "收齐后：--step confirm --decisions-json "
                        "'{\"key\": \"keep\"|\"drop\"|{\"edit\": \"改后的文本\"}}'；"
                        "没表态的条目保持未决，不会被默认留下或删掉。"
                        + ("　另外 extraction_brief 里那几条是**给你的提取任务**，"
                           "不要念给用户确认、更不要写进人格文件。" if brief else ""),
            }, ensure_ascii=False, indent=1))
        else:
            for p in payload:
                print(f"—— {p['kind']}【{p['label']}】（key={p['key']}）\n   {p['value']}")
            if brief:
                print("\n" + BRIEF_NOTE)
                for b in brief:
                    print(f"  - {b}")
        return
    if args.decisions_json:
        incoming = _load_json_arg(args.decisions_json)
        known = {p.key for p in pend}
        unknown = sorted(set(incoming) - known)
        if unknown:   # 同答案侧：认不出的 key 明确报出来，不静默丢
            raise SystemExit(f"这些 key 不在待确认清单里：{unknown}\n"
                             f"待确认：{sorted(known)}")
        decisions.update(incoming)
        state["step"] = "confirm"
        save_state(args.out, state)
        persona, _ = _rebuild(state)
        left = len(pending_confirmations(persona))
        print(f"已落 {len(incoming)} 条决定，还剩 {left} 条未决。"
              + ("下一步：--step ship" if not left
                 else " 未决的保持未决——没问过对方的条目不会被默认留下。"))
        return
    if not pend:
        print("没有待确认的草稿。下一步：--step ship")
        return
    print(f"【逐条确认】共 {len(pend)} 条草稿。每条：y=留 / n=删 / e=改 / q=先退出（已确认的会存下）\n")
    for p in pend:
        print(f"—— {p.kind}【{p.label}】\n   {p.value}")
        try:
            choice = input("   [y/n/e/q] > ").strip().lower()
        except EOFError:
            choice = "q"
        if choice == "q":
            break
        if choice == "n":
            decisions[p.key] = "drop"
        elif choice == "e":
            new = input("   新文本 > ").strip()
            decisions[p.key] = {"edit": new} if new else "keep"
        else:
            decisions[p.key] = "keep"
        state["step"] = "confirm"
        save_state(args.out, state)      # 每条都存——确认是长活儿，没人一口气做完
    persona, _ = _rebuild(state)
    left = len(pending_confirmations(persona))
    print(f"\n已确认 {len(decisions)} 条，还剩 {left} 条未决。"
          + ("下一步：--step ship" if not left else "随时重跑这一步继续。"))


def _step_route(args, state):
    """检索路线选择点。不给 --route 就只把三条路念出来，不替用户拍板。"""
    if args.route:
        if args.route not in RETRIEVAL_ROUTES:
            raise SystemExit(f"没有这条路线：{args.route}。"
                             f"可选：{' / '.join(RETRIEVAL_ROUTES)}")
        state["route"] = args.route
        state["step"] = "route"
        save_state(args.out, state)
        print(f"【已选】{RETRIEVAL_ROUTES[args.route]['名称']}\n")
        print(route_note(args.route))
        print("\n【下一步】--step ship")
        return
    if args.json:
        print(json.dumps({"preamble": ROUTE_PREAMBLE, "routes": route_options(),
                          "default": ROUTE_DEFAULT,
                          "next": "--step route --route <key>"},
                         ensure_ascii=False, indent=2))
        return
    print(format_routes())


def _corpus_overwrite_mode(args):
    """两个开关 → 写入侧护栏的档位。两个都给就走备份：备份本来就包含「照写」，
    而反过来（照写却没备份）会丢东西——含糊的时候倒向留得下的那一边。"""
    if getattr(args, "backup_corpus", False):
        return "backup"
    if getattr(args, "accept_corpus_overwrite", False):
        return "accept"
    return "block"


def _load_ship_entries(args):
    """三条 ship 共用的显式导入入口；不传 --import 就保持语料只读。"""
    if not args.import_path:
        return None
    from memory_import import load_any
    entries = load_any(args.import_path)
    print(f"【导入】{args.import_path} → {len(entries)} 条")
    return entries


def _step_ship(args, state):
    # 检索路线必须先选过一次（2026.08.02）。**这里刻意是硬闸不是默认值**：
    # 三条路里有一条会把私人记忆发到第三方，而"用户从没被问过"和"用户选了默认"
    # 在产出物上长得一模一样——不拦住的话，这个选择点等于不存在。
    # 拦的是"没问过"，不是"没选云端"：`--route zero-dep` 一秒就过。
    route = args.route or state.get("route")
    if route not in RETRIEVAL_ROUTES:
        raise SystemExit(
            "不出货：检索路线还没选过。先跑 --step route（加 --json 拿机器可读的"
            "三条路线与各自代价），把三条路念给 TA、由 TA 选，再 --step route "
            "--route <key>。三条里有一条会把 TA 的私人记忆发到第三方，"
            "这件事不给静默默认。")
    state["route"] = route
    # 时区记进 state：跟 route 同待遇，给过一次就不用每次重跑 ship 都再给一遍
    if args.timezone:
        state["timezone"] = args.timezone
    persona, _ = _rebuild(state)
    entries = _load_ship_entries(args)
    active_corpus = None if args.import_path else args.corpus
    client, switched = resolve_client(args.client, state.get("client"))
    if switched:
        print(switched + "\n")
        state["client"] = client
    try:
        # last_shipped_persona：上一次出货**我们自己**写下的人格相对路径。退役只认它——
        # 目录里叫 CLAUDE.md 的文件可能是用户自己给 Claude Code 写的项目指令，
        # 不是我们的货，不能碰（三轮验收打回）
        paths = write_bundle(args.out, persona, client=client,
                             corpus_dir=active_corpus, confirmed=True, entries=entries,
                             previous_persona=state.get("last_shipped_persona"),
                             route=route, corpus_overwrite=_corpus_overwrite_mode(args),
                             timezone=args.timezone or state.get("timezone"))
    except (PermissionError, ValueError, FileNotFoundError) as e:
        raise SystemExit(f"不出货：{e}")
    print("【四件套】")
    print(f"  人格文件：{paths['persona']}")
    for line in memory_report_lines(paths, active_corpus):
        print(line)
    print(f"  MCP 配置：{paths['mcp_config']}")
    print(f"  引导句：{paths['guidance']}"
          f"（给闭源前端 App 用：贴进它的自定义指令／system prompt 框，"
          f"宿主客户端用不上它。用法与边界见《快速上手》§3c）")
    if paths.get("contract"):
        print(f"  注入契约：{paths['contract']}")
    for bak in paths.get("retired", []):
        # 换了档就得说清旧档那份去哪了——留在目录里的第二份人格文件不会再更新，
        # 拼错了会变成"确认过的更新不生效"，而那是最难自查的一种坏法
        print(f"  已退役：{bak.with_suffix('').name} → {bak.name}"
              f"（换档后旧档的人格文件不再更新，别再拼它）")
    if paths.get("corpus_backup"):
        # 备份了就必须说出来，而且要说清「为什么会有这一步」：静默地把语料换掉，
        # 跟静默覆盖是同一种坏法，只是多了一份没人知道的备份
        print(f"\n⚠ 目标记忆库里原来那些窗口会被这次出货改写，整个 timeline 已备份成："
              f"{paths['corpus_backup']}")
        print("   你后来用 latent_append 写进去的内容在那份备份里，"
              "对着它把该留的贴回 memory/timeline/。（备份只加不减，我们不会删它。）")
    if paths.get("overwritten_backup"):
        # **沉默地覆盖才是最坏的形态**：老用户升级重跑 ship 是对的做法，但他手改过
        # 的段落不在 state 里、重放不出来。所以这里必须说出来，并指向那份备份。
        bak = paths["overwritten_backup"]
        print(f"\n⚠ 磁盘上原来那份人格文件**跟这次重新生成的不一样**，"
              f"已备份成：{bak}")
        print("   多半是两种情况：你后来手改过它，或者它是旧版本出的。"
              "**手改的段落不会被自动带过来**——问卷答案和确认结果存在 "
              "init_state.json 里、能重放，你自己写的那几行不在里面。")
        print("   对着那份备份看一眼，把你自己加的内容贴回新文件里。"
              "（人格文件本来就是你的，改它是对的——只是升级这一步带不动它。）")
    for s in persona.suggestions():
        print(f"  建议（不阻塞）：{s}")
    # 任务书在这里再给一次：ship 是最后一个出口，第二阶段的活儿从这儿接
    _, all_qs = _rebuild(state)
    brief = extraction_brief(all_qs, state.get("answers") or {})
    if brief:
        print("\n" + BRIEF_NOTE)
        for b in brief:
            print(f"  - {b}")
    print("\n" + route_note(route))
    print("\n" + ship_note(client))
    state["step"] = "shipped"
    state["last_shipped_persona"] = shipped_persona_state_value(args.out, paths["persona"])
    save_state(args.out, state)


def _cli(args):
    """薄 CLI，五步走：questionnaire → answers → confirm → route → ship。
    每步存状态可续跑；不传 --step 时按状态里的进度提示下一步该跑什么。
    真正的逐题交互留给导出的 prompt（路线 C）——用户拿去自己的模型那边一问一答，
    比在终端里敲长文本舒服得多。"""
    state = load_state(args.out)
    step = args.step
    if step == "inspect":
        _step_inspect_v2(args, state)
        return
    if state.get("mode") == "direct_persona" and step in (
            "extract", "choose-sections", "preview"):
        raise SystemExit(
            "当前是用户选择的原人格直接接入，这三步已由用户选择跳过；"
            "跳过不等于通过。如需十二节编译增强，请重新 --step inspect 并选择 "
            "--existing-persona-choice treat_current_as_original。")
    if step == "extract":
        _step_extract_v2(args, state)
        return
    if step == "choose-sections":
        _step_choose_sections_v2(args, state)
        return
    if step == "preview":
        _step_preview_v2(args, state)
        return
    if not step:
        if state.get("mode") == "direct_persona":
            step = {"inspected": "route", "route": "ship", "shipped": "ship"}.get(
                state.get("step", ""), "inspect")
        elif state.get("schema_version") == 2:
            step = {"": "inspect", "inspected": "extract", "extracted": "extract",
                    "candidates_loaded": "choose-sections",
                    "sections_chosen": "preview", "previewed": "route",
                    "route": "ship", "shipped": "ship"}.get(
                        state.get("step", ""), "inspect")
        else:
            step = {"": "questionnaire", "questionnaire": "answers",
                    "answers": "confirm", "confirm": "route", "route": "ship",
                    "shipped": "ship"}[state.get("step", "")]
        print(f"（按进度接着跑：--step {step}）\n")
        if state.get("mode") == "direct_persona":
            if step == "route":
                _step_route(args, state)
            elif step == "ship":
                _step_ship_direct(args, state)
            else:
                _step_inspect_v2(args, state)
            return
        if state.get("schema_version") == 2:
            if step == "inspect":
                _step_inspect_v2(args, state)
            elif step == "extract":
                _step_extract_v2(args, state)
            elif step == "choose-sections":
                _step_choose_sections_v2(args, state)
            elif step == "preview":
                _step_preview_v2(args, state)
            elif step == "route":
                _step_route(args, state)
            else:
                _step_ship_v2(args, state)
            return
    if step == "questionnaire":
        _step_questionnaire(args)
    elif step == "answers":
        _step_answers(args, state)
    elif step == "confirm":
        _step_confirm(args, state)
    elif step == "route":
        _step_route(args, state)
    elif step == "ship" and state.get("mode") == "direct_persona":
        _step_ship_direct(args, state)
    elif step == "ship" and state.get("schema_version") == 2:
        _step_ship_v2(args, state)
    else:
        _step_ship(args, state)


if __name__ == "__main__":
    # 中文 Windows（控制台编码 cp936/GBK）下，stdout 默认按系统区域编码写；语料和
    # 人格文件里出现 emoji（U+1F338 这类非 BMP 字符）几乎必然，于是第一个
    # `--step inspect --json` 就抛 UnicodeEncodeError，**报错里完全看不出该怎么办**
    # ——整条初始化流程崩在第一个命令上（2026.08.03 真机走查第三条）。
    # ⚠ 要改的是**输出端自己保证编码**，不是退回 `ensure_ascii=True`：那会把中文
    #   全变成 \uXXXX，用户和下游 AI 都读不了。JSON 的字段与内容一个字不动。
    # ⚠ stderr 一起锁：argparse 的报错、`ap.error(...)` 的中文提示走的是那条流，
    #   同一台机器上撞的是同一堵墙。
    # ⚠ 只放在 `__main__` 里，不塞进任何被 import 的函数：别人 import 本模块时
    #   不该被我们改掉整个进程的 stdout。`memory_import.py` / `mcp_server.py`
    #   两个入口同样各有一份（三个入口一起改，别留一个）。
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", help="产出目录")
    ap.add_argument("--persona", help="已有的人格文件（可选；不会作为语料导入）")
    ap.add_argument("--corpus", help="已有语料目录（可选）")
    # 默认值是 None 而不是 "claude-code"，为的是 ship 步能分辨"用户显式指定了档"
    # 与"用户没提"——两者在出货时的行为必须不同（见 resolve_client）
    ap.add_argument("--client", default=None, choices=sorted(CLIENT_FILENAMES),
                    help=f"客户端档（默认 {DEFAULT_CLIENT}）。任何一步都可以给；"
                         f"ship 步显式给的以它为准，会覆盖前面存下的档")
    ap.add_argument("--step", choices=["inspect", "extract", "choose-sections", "preview",
                                       "questionnaire", "answers", "confirm", "route", "ship"],
                    help="不传则按 init_state.json 里的进度接着跑")
    ap.add_argument("--route", choices=sorted(RETRIEVAL_ROUTES),
                    help="route 步：TA 选的检索路线（zero-dep 默认 / local 本地模型 / "
                         "cloud 云端——**云端会把语料发到第三方**）。不给就只把三条"
                         "路线和各自代价打出来，不替 TA 拍板")
    ap.add_argument("--answers", help="answers 步：模型整理好的答案清单文件（人工路径）")
    # --- 以下三个是给「AI 驱动」用的程序接口（人自己敲命令时用不上）---
    # 产品事实：多数用户不会开终端敲 python，真实形态是把仓库交给 AI、AI 边问边跑。
    # 引导指南本来就假定 AI 驱动，但 CLI 只提供了给人用的交互路径——差的这一截
    # 在这里补上。库函数（apply_answers / apply_confirmations）本来就收字典。
    ap.add_argument("--json", action="store_true",
                    help="机器可读输出（questionnaire 出题目、confirm --list 出待确认清单）")
    ap.add_argument("--answers-json", dest="answers_json",
                    help="answers 步：结构化答案（文件路径 / JSON 字面量 / - 读 stdin）")
    ap.add_argument("--candidates",
                    help="extract 步：当前模型返回的候选 JSON（文件路径 / JSON 字面量 / -）；"
                         "重传会替换旧语料候选，随后须重跑 choose-sections")
    ap.add_argument("--section-decisions-json", dest="section_decisions_json",
                    help="choose-sections 步：{section: section_version_id} JSON 对象")
    ap.add_argument("--original-section-decisions-json",
                    dest="original_section_decisions_json",
                    help="choose-sections 步：{原人格块 item_id: 主节或 leave_unresolved}")
    # inspect 检测到「上次有块、这次 0 块」时会 blocking。**拦截必须留出口**，
    # 否则用户又被推回手改输入文件那条路（那正是塌节本身的成因）。这个开关只表示
    # 「我知道，是我有意删的」，降级成 warning；它不改用户的任何文件。
    ap.add_argument("--accept-persona-drift", dest="accept_persona_drift",
                    action="store_true",
                    help="inspect 步：确认输入人格文件这次少掉的整节就是你有意删的，"
                         "把 PERSONA_SECTION_COLLAPSED 从 blocking 降为 warning")
    ap.add_argument("--existing-persona-choice",
                    choices=["treat_current_as_original", "use_original_as_is",
                             "continue_legacy"],
                    help="inspect 检测到现有人格文件时的用户选择：进入十二节编译／"
                         "原人格不拆块不重排、只给出货副本追加记忆协议／继续旧流程；"
                         "不给就只展示选项，不替用户自动采用")
    ap.add_argument("--list", action="store_true",
                    help="confirm 步：只列出待确认草稿，不进交互循环")
    ap.add_argument("--decisions-json", dest="decisions_json",
                    help="confirm 步：结构化决定（同上三种取值），非交互落盘")
    # 重跑出货会盖掉目标 timeline 里已有的窗口（包括用户后来用 latent_append 写进去
    # 的），所以默认拦住。**拦截必须留出口**，这两个就是；「换目录」那一支刻意不做
    # ——`--out` 本来就能换，为它新开开关等于凭空多一套目录语义
    ap.add_argument("--backup-corpus", dest="backup_corpus", action="store_true",
                    help="ship 步：先把整个 memory/timeline 备份成 timeline.bak（只加不减，"
                         "已有备份就顺号）再写")
    ap.add_argument("--accept-corpus-overwrite", dest="accept_corpus_overwrite",
                    action="store_true",
                    help="ship 步：确认「我知道会覆盖已有窗口，就这么办」，不备份直接写")
    ap.add_argument("--timezone", metavar="IANA名",
                    help="记忆所有者的时区（例如 Europe/Berlin），写进出货的 "
                         "mcp-config.json。⚠ 不给就落默认值 Asia/Shanghai（东八区）"
                         "并在配置说明里写明它是默认值——**探测到的宿主时区任何时候都"
                         "不会顶替它**：探的是跑初始化这台机器，不是 TA 的（云服务器上"
                         "探出来正好是 UTC，填进去等于给那个缺陷发合格证）。"
                         "不给的后果：TA 不在东八区的话日期会静默错一天，"
                         "--doctor 那一格会报 ⚠")
    ap.add_argument("--import", dest="import_path",
                    help="ship 步：语料导出文件（ChatGPT/Claude json、聊天 txt、timeline md），"
                         "由 memory_import 认格式并落成记忆库")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    elif args.out:
        _cli(args)
    else:
        ap.print_help()
