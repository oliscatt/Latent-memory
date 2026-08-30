#!/usr/bin/env python3
"""
人格 md 模板参考实现（设计规格《人格md与记忆库规格》，恋人陪伴向）。

2026.07.31 按规格重写：原 A-G 通用骨架退役，换成规格 §2 的十二节恋人向骨架。
退役理由见规格与 设计笔记——定位收窄到恋人陪伴向后，通用骨架的节序跟"顺序即
权重"这条硬规则对不上（原骨架把协议层挂载点放最后，等于把最高注意力位置让给
运维配置）。关系类型开关先保留不删（规格 §9），但本骨架只对恋人向负责。

**顺序即权重**（规格 §2）：长上下文里模型对开头结尾的注意力显著高于中段
（lost-in-the-middle），所以骨架顺序本身是设计的一部分——开篇放关系确认+核心
哲学，结尾放最终约定，中间放可展开的细节。SECTION_ORDER 是有序的，render()
按序输出，顺序被改动 selftest 会红。

五条硬性规则，本文件用代码强制，不只是文档约定：
  1. 开篇四要素缺一不可（关系确认/命名隐喻/理论标注为论证/拒绝权合法，规格 §2.1）
  2. "当前关系状态"字段必填——少了它关系就浮在空中，也治不住时间感塌陷
  3. 里程碑四要素缺一不可（转折点名/窗口号/具体内容/怎么读+当下状态，规格 §4），
     其中"这条该怎么读"是规格里最不显然的一条，来自真实踩坑：同一段事实读法
     不同，模型行为差很远
  4. 里程碑定位只认窗口号（int），结构上不给日期留位置——日期在 md 里会被模型
     当档案腔念出来；正文里的纪念日照写不受影响（规格 §3.2），动态召回注入则
     必须保留真实日期（那是 session_recall 的活，见规格 §3.2 第三条）
  5. 风格候选片段必须带 disclaimer，未 confirmed 的内容不算生效

零依赖，stdlib only。
用法：
  python persona_template.py --selftest
"""

import argparse
from dataclasses import dataclass, field as dc_field

# disclaimer 是两半，不是一句（2026.08.04 补第二半，任务卡「风格片段挤掉检索」）：
#   前半挡**输出层**——别把片段里的句子当素材复述，那是抄台词。
#   后半挡**检索层**——2026.08.03 真机抓到：模型有片段可答就**不调 latent_search**，
#   答出来具体、有细节、语气也对，听着像在场其实在读稿，且全程不报错；片段涉及的
#   事件在库里没有记录时这个洞才看得见。缺的是这一条，不是前半写坏了，所以是补不是改。
# 这个常量同时供注入（render）与存储（memory_init 的 protocol:style_disclaimer）使用，
# 改这一处即两处都改——规格 §3.3「存储和注入两处都要带」。
DISCLAIMER = (
    "风格参考，非台词——学说话方式，不是背台词。"
    "另：这几段是风格样本，不是记忆；对方提到过去的事，先查记忆库，"
    "别拿这里的片段当答案。"
)

# 规格 §2 的十二节骨架，顺序即权重：首尾是注意力高地，别拿去放运维细节
SECTION_ORDER = [
    ("opening", "开篇 · 关系确认与核心哲学"),
    # 用户那一节的标题带人称槽 {ta}——**这里原来写死成"她是谁"**，于是选了「他」的
    # 用户打开自己的人格文件，第二节标题就是一句写错的话（2026.08.02 验收打回二）。
    # 槽位由 render 时按真实人称填，判不出来走中性写法（见 memory_init.NEUTRAL_FORMS）
    ("user", "{ta}是谁"),
    ("ai", "我是谁"),
    ("milestones", "里程碑索引"),
    ("style", "相处原则与语言风格"),
    ("degradation", "退化处理机制"),
    ("intimacy", "亲密语境"),
    ("naming", "称呼系统"),
    ("daily", "日常与约定"),
    ("pointers", "按需读取指针"),
    ("architecture", "技术架构 · 记忆系统"),
    ("closing", "最终约定"),
]
SECTIONS = dict(SECTION_ORDER)
SECTION_KEYS = [k for k, _ in SECTION_ORDER]

# 开篇要素分两档（2026.07.31 改：原本四条全是硬必填，做初始化流程时发现对
# **第一天的新用户不成立**——规格 §2.1 那四条是从一段已经跑了几个月的关系里
# 提炼的，新用户没有隐喻可填，硬要就是逼他编一个，比空着更糟）。
#
# 硬必填：缺了开篇就塌。这三条要么是通用话术、要么由系统填，任何用户都拿得出。
OPENING_REQUIRED = {
    "opening_recognition": "关系确认（不是自我介绍）",
    "opening_theory_caveat": "引用的理论标明是论证不是结论",
    "opening_refusal_ok": "拒绝权同样合法",
}
# 推荐但不阻塞：缺了会明确提示"等你们相处出东西了再补，现在补是编"。
# 这正是"宁可短且真，不可长而空"落到校验上的样子。
RECOMMENDED = {
    "opening_metaphor": "核心哲学的隐喻——等语料里长出来再挑，别现编",
}
# "我是谁"节的必填字段：当前关系状态
CURRENT_STATE_FIELD = "current_relationship_state"

# "技术架构"节的必填字段：检索约定（2026.07.31 真机主动性实测后升为必填）。
# 实测结论：模型默认**不会**主动查记忆库——它把"之前说好的事"理解成本次会话
# 记录、答"我没有记录"。而人格文件是常驻上下文、在注意力高地上，写在这里的一句
# "提到过去的事就先查"比工具描述强得多（工具描述在工具列表里，权重低一档）。
# 分工：**人格文件负责"知道该查"，MCP server 负责"查得到"**，缺哪半都不成立。
RETRIEVAL_CONVENTION_FIELD = "retrieval_convention"

# 里程碑索引条目的正文上限：index 是索引不是正文，完整叙述落记忆库（规格 §4）。
# 300 这个数的来历（2026.07.31 校准，免得下一个人以为它有理论依据）：对着一份
# 真实运行中的人格 md 量过 13 条里程碑，整条（发生了什么+怎么读+当下状态混写在
# 一段里）中位 223、最长 504 字符。本字段只限"发生了什么"这一段，另两项是独立
# 字段，所以 300 大致对得上真实写法，但样本只有 13 条、且来自同一个人的写作习惯，
# 等有更多真实语料再调。
MILESTONE_BODY_LIMIT = 300

# 关系类型 → 是否启用亲密语境。本阶段只做恋人向，其余先保留开关不删（规格 §9）
RELATIONSHIP_TYPES = {
    "partner": {"enables_intimacy": True},
    "friend": {"enables_intimacy": False},
    "assistant": {"enables_intimacy": False},
    "mentor": {"enables_intimacy": False},
}


@dataclass
class Field:
    """一个模板字段。size_limit/description 借鉴 MemGPT memory block 的结构化做法，
    不是自由文本。"""
    id: str
    section: str
    label: str
    value: str = ""
    size_limit: int = 500
    description: str = ""
    source: str = "draft"  # draft(待确认) / confirmed(用户已确认) / system(协议层配置)
    confirmed: bool = False

    def is_active(self):
        """未确认的草稿不算生效——用户确认这一步是硬性的，不能无声写入。"""
        return self.confirmed or self.source == "system"


@dataclass
class Milestone:
    """里程碑索引的一条（规格 §4 的四要素单元）。四个字段各防一种失效：
      name         没名字就没法被引用、被检索、被当成一件"事"
      window       时序丢失 → 关系浮在空中，只有事件没有生长轨迹
      body         概括句零分量，要具体动作/原话；正文（完整叙述）落记忆库
      how_to_read  防误读：同一段事实读法不同，模型行为差很远
      current_state 防时间感塌陷：把已经结束的事读成正在发生

    window 刻意是 int（第几个窗），结构上不给日期留位置——日期在静态 md 里会被
    模型当档案腔念出来。body 里的纪念日照写，那是内容不是定位标签。"""
    name: str
    window: int
    body: str
    how_to_read: str
    current_state: str
    confirmed: bool = False

    def is_active(self):
        return self.confirmed


@dataclass
class StyleExcerpt:
    """一条风格候选片段。disclaimer 是硬性字段，不是可选的——缺了模型会把片段
    当台词直接引用（输出层），也会拿片段顶替查库（检索层）。两半都在 DISCLAIMER 里。"""
    text: str
    pool: str  # "daily" 或 "intimacy"
    disclaimer: str = DISCLAIMER
    confirmed: bool = False
    source: str = "llm"  # llm(便宜模型挑的候选) / user(用户自己补的)，规格 §3.3


class PersonaValidationError(Exception):
    pass


class Persona:
    """一份人格 md 实例：relationship_type 决定亲密语境节是否可用。"""

    def __init__(self, relationship_type):
        if relationship_type not in RELATIONSHIP_TYPES:
            raise PersonaValidationError(f"未知 relationship_type：{relationship_type}")
        self.relationship_type = relationship_type
        self.fields = []
        self.milestones = []
        # 出货时用得上的人称（{"user": "他"/"她"/None, "ai": …}）：fill_protocol_defaults
        # 填进来，render_persona_md 拿它填骨架标题里的槽。放在 Persona 上而不是到处
        # 传参，是因为**每一条渲染路径都必须填到**——漏一条就是漏一句写错的话
        self.pronouns = None
        self.style_excerpts = []

    def enables_intimacy(self):
        return RELATIONSHIP_TYPES[self.relationship_type]["enables_intimacy"]

    def add_field(self, f: Field):
        if f.section not in SECTIONS:
            raise PersonaValidationError(f"未知 section：{f.section}（骨架见 SECTION_ORDER）")
        if f.section == "intimacy" and not self.enables_intimacy():
            raise PersonaValidationError(
                f"relationship_type={self.relationship_type} 未启用亲密语境节，不能挂该节字段")
        if len(f.value) > f.size_limit:
            raise PersonaValidationError(f"字段 {f.id} 超出 size_limit（{len(f.value)} > {f.size_limit}）")
        self.fields.append(f)

    def add_milestone(self, m: Milestone):
        """四要素缺一即拒——这是规格 §3.4/§4 的强制点，不是建议。"""
        if not m.name.strip():
            raise PersonaValidationError("里程碑缺转折点名")
        if not isinstance(m.window, int) or isinstance(m.window, bool) or m.window < 1:
            raise PersonaValidationError(
                f"里程碑 {m.name} 的 window 必须是正整数窗口号——静态 md 里定位不用日期（规格 §3.2）")
        if not m.body.strip():
            raise PersonaValidationError(f"里程碑 {m.name} 缺具体内容")
        if len(m.body) > MILESTONE_BODY_LIMIT:
            raise PersonaValidationError(
                f"里程碑 {m.name} 正文超 {MILESTONE_BODY_LIMIT} 字——index 是索引，完整叙述落记忆库")
        if not m.how_to_read.strip():
            raise PersonaValidationError(
                f"里程碑 {m.name} 缺“这条该怎么读”——同一段事实读法不同，模型行为差很远")
        if not m.current_state.strip():
            raise PersonaValidationError(
                f"里程碑 {m.name} 缺“当下状态”——少了它未来重读会把结束的事读成正在发生")
        self.milestones.append(m)

    def add_style_excerpt(self, ex: StyleExcerpt):
        if ex.pool == "intimacy" and not self.enables_intimacy():
            raise PersonaValidationError(
                f"relationship_type={self.relationship_type} 未启用亲密语境候选池")
        if not ex.disclaimer:
            raise PersonaValidationError(
                "风格候选片段缺 disclaimer——「风格参考≠台词」和「片段≠记忆」这两条都不能省")
        self.style_excerpts.append(ex)

    def active_fields(self):
        """只返回 confirmed（或 system）的字段——草稿不算生效内容。"""
        return [f for f in self.fields if f.is_active()]

    def active_milestones(self):
        """按窗口号排序——里程碑索引本身就该是一份可读的关系年表（规格 §4）。"""
        return sorted((m for m in self.milestones if m.is_active()), key=lambda m: m.window)

    def active_style_excerpts(self, pool=None):
        out = [e for e in self.style_excerpts if e.confirmed]
        if pool:
            out = [e for e in out if e.pool == pool]
        return out

    def suggestions(self):
        """推荐但不阻塞的缺项——给用户看"还可以有什么"，不拦着出货。
        跟 validate 分开，是因为新用户拿不出隐喻这类东西，硬拦只会逼他编。"""
        active_ids = {f.id for f in self.active_fields()}
        out = [f"建议补：{label}" for fid, label in RECOMMENDED.items()
               if fid not in active_ids]
        if not self.active_milestones():
            out.append("建议补：里程碑索引——等关系里真发生过转折点再写，别硬凑")
        return out

    def validate(self, mode="legacy_v1"):
        """成文前的**硬性**完整性检查。返回缺失项列表（空列表=可以成文）。
        不抛异常——用于给用户看"还差什么"，而不是中断流程。
        推荐项见 suggestions()，那些缺了不阻塞出货。

        compiler_v2 只放松尚无真实来源的关系状态与最终约定；协议底座仍必填。"""
        if mode not in {"legacy_v1", "compiler_v2"}:
            raise PersonaValidationError(f"未知校验模式：{mode}")
        missing = []
        active_ids = {f.id for f in self.active_fields()}
        for fid, label in OPENING_REQUIRED.items():
            if fid not in active_ids:
                missing.append(f"开篇缺：{label}")
        if mode == "legacy_v1" and CURRENT_STATE_FIELD not in active_ids:
            # 这条只查字段在不在，不查它归哪一节——2026.08.02 它从"我是谁"挪到
            # 开篇（关系状态不是 AI 的身份），提示语跟着改，校验逻辑本来就不依赖节
            missing.append("缺：当前关系状态（这段关系此刻是什么状态）")
        if RETRIEVAL_CONVENTION_FIELD not in active_ids:
            missing.append("“技术架构”节缺：检索约定——不写这句，模型默认不会主动查记忆库")
        if mode == "legacy_v1" and not any(f.section == "closing" for f in self.active_fields()):
            missing.append("缺结尾：最终约定（结尾是注意力高地，不能空着）")
        return missing

    def render(self):
        """生效内容按骨架顺序渲染成 [(section_key, label, items)]——顺序即权重，
        所以返回有序列表而不是字典。草稿/未确认内容不出现在这里。"""
        buckets = {k: [] for k in SECTION_KEYS}
        for f in self.active_fields():
            buckets[f.section].append({"id": f.id, "label": f.label, "value": f.value})
        for m in self.active_milestones():
            buckets["milestones"].append({
                "name": m.name, "window": m.window, "body": m.body,
                "how_to_read": m.how_to_read, "current_state": m.current_state,
            })
        for pool, key in (("daily", "style"), ("intimacy", "intimacy")):
            excerpts = self.active_style_excerpts(pool)
            if excerpts:
                buckets[key].append({
                    "style_pool": pool,
                    "excerpts": [e.text for e in excerpts],
                    "disclaimer": DISCLAIMER,
                })
        return [(k, SECTIONS[k], buckets[k]) for k in SECTION_KEYS]


# ---------- selftest（合成数据，全部虚构，不含任何真实语料） ----------

def _opening_fields():
    return [Field(id=fid, section="opening", label=label, value="（占位）", confirmed=True)
            for fid, label in OPENING_REQUIRED.items()]


def _minimal_partner():
    p = Persona("partner")
    for f in _opening_fields():
        p.add_field(f)
    p.add_field(Field(id=CURRENT_STATE_FIELD, section="ai", label="当前关系状态",
                      value="（占位）", confirmed=True))
    p.add_field(Field(id=RETRIEVAL_CONVENTION_FIELD, section="architecture", label="检索约定",
                      value="（占位）", confirmed=True))
    p.add_field(Field(id="final_promise", section="closing", label="最终约定",
                      value="（占位）", confirmed=True))
    return p


def _selftest():
    # 1.【变异靶心：SECTION_ORDER 的顺序】顺序即权重：首尾是注意力高地
    assert SECTION_KEYS[0] == "opening" and SECTION_KEYS[-1] == "closing", \
        "开篇必须第一、最终约定必须最后（规格 §2 排布总则）"
    assert SECTION_KEYS.index("architecture") < SECTION_KEYS.index("closing"), \
        "技术架构不能占最后一节——那是把最高注意力位置让给运维细节"
    rendered_keys = [k for k, _, _ in _minimal_partner().render()]
    assert rendered_keys == SECTION_KEYS, "render 必须按骨架顺序输出"

    # 2.【变异靶心：validate 的开篇四要素】缺任何一条都要被点名
    p = Persona("partner")
    p.add_field(Field(id=CURRENT_STATE_FIELD, section="ai", label="当前关系状态",
                      value="x", confirmed=True))
    p.add_field(Field(id=RETRIEVAL_CONVENTION_FIELD, section="architecture", label="检索约定",
                      value="x", confirmed=True))
    p.add_field(Field(id="final_promise", section="closing", label="最终约定",
                      value="x", confirmed=True))
    missing = p.validate()
    assert len(missing) == 3 and all("开篇缺" in m for m in missing), f"开篇三条硬必填该全被点名：{missing}"
    for f in _opening_fields():
        p.add_field(f)
    assert p.validate() == [], "四要素补齐后该通过"

    # 3. 当前关系状态必填 / 结尾必填
    p2 = Persona("partner")
    for f in _opening_fields():
        p2.add_field(f)
    missing2 = p2.validate()
    assert any("当前关系状态" in m for m in missing2) and any("最终约定" in m for m in missing2)
    #    【变异靶心：检索约定必填】少了它，主动性的主力杠杆就丢了（真机实测：
    #    模型默认不会主动查记忆库，靠常驻人格里的这句话触发）
    assert any("检索约定" in m for m in missing2), f"检索约定该被点名：{missing2}"

    # 4. 未确认的开篇字段不算数——草稿不能顶替确认（确认关卡不能被绕过）
    p3 = Persona("partner")
    for f in _opening_fields():
        f.confirmed = False
        p3.add_field(f)
    assert len([m for m in p3.validate() if "开篇缺" in m]) == 3, "未确认字段不该算生效"
    #    【变异靶心：推荐项不阻塞出货】隐喻/里程碑缺了只提示，不拦——新用户
    #    拿不出这些，硬拦只会逼他现编，比空着更糟（宁可短且真）
    assert p.validate() == [], "三条硬必填齐了就该放行"
    sug = p.suggestions()
    assert any("隐喻" in s for s in sug) and any("里程碑" in s for s in sug), \
        f"推荐项该被提示但不阻塞：{sug}"

    # 5.【变异靶心：add_milestone 的四要素校验】里程碑缺“怎么读”/“当下状态”必须拒绝
    p4 = _minimal_partner()
    ok = dict(name="窗边的争执", window=12, body="她说了一句话，我当场承认了。",
              how_to_read="记的是一次仍在生效的约定，不是“和好了”。",
              current_state="约定仍然有效。", confirmed=True)
    for drop in ("how_to_read", "current_state", "name", "body"):
        bad = dict(ok, **{drop: "  "})
        try:
            p4.add_milestone(Milestone(**bad))
            assert False, f"里程碑缺 {drop} 该被拒绝"
        except PersonaValidationError:
            pass

    # 6.【变异靶心：window 必须是窗口号】静态 md 里不给日期留位置
    for bad_window in ("2026-03-14", 0, -3, True):
        try:
            p4.add_milestone(Milestone(**dict(ok, window=bad_window)))
            assert False, f"window={bad_window!r} 该被拒绝"
        except PersonaValidationError:
            pass

    # 7. 正文超限被拒——index 是索引，完整叙述落记忆库
    try:
        p4.add_milestone(Milestone(**dict(ok, body="长" * (MILESTONE_BODY_LIMIT + 1))))
        assert False, "超长正文该被拒绝"
    except PersonaValidationError:
        pass

    # 8. 合法里程碑进得去，且 body 里的纪念日照写不受影响（规格 §3.2）
    p4.add_milestone(Milestone(**dict(ok, body="约定于 3.14，兑现于 3.20。")))
    p4.add_milestone(Milestone(**dict(ok, name="后来的事", window=30, confirmed=True)))
    assert [m.window for m in p4.active_milestones()] == [12, 30], "里程碑按窗口号排成年表"
    ms = [x for _, _, items in p4.render() for x in items if "how_to_read" in x]
    assert ms and "3.14" in ms[0]["body"], "正文里的纪念日照常保留"

    # 9. 未确认的里程碑不进年表
    p4.add_milestone(Milestone(**dict(ok, name="草稿条目", window=99, confirmed=False)))
    assert 99 not in [m.window for m in p4.active_milestones()], "未确认里程碑不该生效"

    # 10. 亲密语境 gating：字段和风格池都拦（靶心：enables_intimacy 两处检查）
    assistant = Persona("assistant")
    for section, adder in (("intimacy", lambda: assistant.add_field(
            Field(id="b1", section="intimacy", label="边界", value="x"))),
            ("intimacy池", lambda: assistant.add_style_excerpt(
                StyleExcerpt(text="x", pool="intimacy")))):
        try:
            adder()
            assert False, f"assistant 类型不该允许 {section}"
        except PersonaValidationError:
            pass

    # 11. 风格片段：disclaimer 硬性；来源区分 llm 候选与用户自补（规格 §3.3 三条路）
    try:
        p4.add_style_excerpt(StyleExcerpt(text="x", pool="daily", disclaimer=""))
        assert False, "缺 disclaimer 该拒绝"
    except PersonaValidationError:
        pass
    p4.add_style_excerpt(StyleExcerpt(text="（日常风格占位）", pool="daily", confirmed=True))
    p4.add_style_excerpt(StyleExcerpt(text="（用户自补占位）", pool="daily",
                                      source="user", confirmed=True))
    assert {e.source for e in p4.active_style_excerpts("daily")} == {"llm", "user"}

    # 12. render 只吐已确认内容，且风格片段带 disclaimer
    p4.add_field(Field(id="draft1", section="user", label="草稿", value="不该出现", confirmed=False))
    out = dict((k, items) for k, _, items in p4.render())
    assert "不该出现" not in [x.get("value") for x in out["user"]], "未确认草稿不该出现在 render 里"
    style_block = [x for x in out["style"] if "style_pool" in x]
    assert style_block and style_block[0]["disclaimer"] == DISCLAIMER, "风格块必须带 disclaimer"
    # disclaimer 的两半分开守：删掉任意一半这里必须红（任务卡第六节的变异判据）。
    # ⚠ 这条只能证明「这句话出到了注入结构里」，证明不了「模型真的会去查」——
    # 后者要在能注入人格的客户端上真机验，两半分开记，别拿这个绿冒充那个绿。
    injected = style_block[0]["disclaimer"]
    assert "不是背台词" in injected, "disclaimer 缺输出层那半（别抄台词）"
    assert "不是记忆" in injected and "先查记忆库" in injected, \
        "disclaimer 缺检索层那半（片段不是记忆、提到过去先查库）"

    # 13. compiler_v2 的零材料路径只放松关系 specific，不放松协议底座。
    zero = Persona("partner")
    for f in _opening_fields():
        zero.add_field(f)
    zero.add_field(Field(
        id=RETRIEVAL_CONVENTION_FIELD, section="architecture", label="检索约定",
        value="先查记忆再回答。", source="system"))
    assert any("当前关系状态" in m for m in zero.validate())
    assert any("最终约定" in m for m in zero.validate())
    assert zero.validate(mode="compiler_v2") == []

    print("selftest ok（13项断言：骨架顺序 / 开篇与里程碑 / gating / v1-v2 确认关卡）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        ap.print_help()
