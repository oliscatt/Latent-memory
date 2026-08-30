#!/usr/bin/env python3
"""人格来源分级编译器。零依赖，使用 ``--selftest`` 自检。"""

import argparse
import difflib
import hashlib
import itertools
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from persona_template import SECTION_KEYS


SOURCE_TYPES = frozenset({"original_persona", "corpus", "protocol", "user_choice"})
OPERATIONS = frozenset({"keep", "move", "rewrite", "delete", "add"})


@dataclass(frozen=True)
class PersonaItem:
    item_id: str
    text: str
    section: str | None
    source_type: str
    source_ref: str
    source_span: tuple[int, int] | None
    source_hash: str
    operation: str
    original_text: str
    proposed_text: str
    confidence: str
    confirmed: bool
    operation_reason: str = ""
    derived_from: tuple[str, ...] = ()
    conflicts_with: tuple[str, ...] = ()
    group_id: str = ""

    def __post_init__(self):
        if self.source_type not in SOURCE_TYPES:
            raise ValueError(f"未知来源类型：{self.source_type}")
        if self.operation not in OPERATIONS:
            raise ValueError(f"未知操作：{self.operation}")
        if self.section is not None and self.section not in SECTION_KEYS:
            raise ValueError(f"未知人格节：{self.section}")
        if self.operation in {"rewrite", "delete"} and not self.operation_reason.strip():
            raise ValueError("rewrite/delete 必须说明理由")
        if self.source_span is not None:
            start, end = self.source_span
            if start < 0 or end < start:
                raise ValueError(f"非法来源区间：{self.source_span}")


def item_to_dict(item):
    """把来源条目序列化成稳定、JSON 可写的显式字段。"""
    return {
        "item_id": item.item_id,
        "text": item.text,
        "section": item.section,
        "source_type": item.source_type,
        "source_ref": item.source_ref,
        "source_span": list(item.source_span) if item.source_span is not None else None,
        "source_hash": item.source_hash,
        "operation": item.operation,
        "original_text": item.original_text,
        "proposed_text": item.proposed_text,
        "confidence": item.confidence,
        "confirmed": item.confirmed,
        "operation_reason": item.operation_reason,
        "derived_from": list(item.derived_from),
        "conflicts_with": list(item.conflicts_with),
        "group_id": item.group_id,
    }


def item_from_dict(data):
    """从 JSON 对象恢复来源条目，并重新执行枚举与区间校验。"""
    values = dict(data)
    span = values.get("source_span")
    values["source_span"] = tuple(span) if span is not None else None
    values["operation_reason"] = values.get("operation_reason", "")
    values["derived_from"] = tuple(values.get("derived_from", ()))
    values["conflicts_with"] = tuple(values.get("conflicts_with", ()))
    return PersonaItem(**values)


@dataclass(frozen=True)
class Conflict:
    conflict_id: str
    kind: str
    severity: str
    item_ids: tuple[str, ...]
    reason: str
    choices: tuple[str, ...]
    resolved_choice: str | None = None

    def __post_init__(self):
        if self.severity not in {"normal", "blocking"}:
            raise ValueError(f"未知冲突级别：{self.severity}")


@dataclass(frozen=True)
class SectionVersion:
    version_id: str
    section: str
    item_ids: tuple[str, ...]
    markdown: str
    source_summary: tuple[str, ...]
    diff: str
    resolved_conflict_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if self.section not in SECTION_KEYS:
            raise ValueError(f"未知人格节：{self.section}")


@dataclass(frozen=True)
class SectionDecision:
    section: str
    version_id: str | None
    status: str

    def __post_init__(self):
        if self.section not in SECTION_KEYS:
            raise ValueError(f"未知人格节：{self.section}")
        if self.status not in {"pending", "confirmed", "unresolved"}:
            raise ValueError(f"未知节确认状态：{self.status}")


@dataclass(frozen=True)
class CompileIssue:
    code: str
    severity: str
    message: str
    item_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if self.severity not in {"warning", "blocking"}:
            raise ValueError(f"未知问题级别：{self.severity}")


@dataclass(frozen=True)
class SourceManifest:
    persona_file: Path | None
    corpus_files: tuple[Path, ...]
    source_hashes: dict[str, str] = field(default_factory=dict)
    issues: tuple[CompileIssue, ...] = ()


@dataclass(frozen=True)
class MergeResult:
    active_items: tuple[PersonaItem, ...]
    conflicts: tuple[Conflict, ...]
    suppressed_items: tuple[PersonaItem, ...] = ()

    def active_texts(self, section):
        return [item.proposed_text for item in self.active_items
                if item.section == section and item.operation != "delete"]

    @property
    def renderable_text(self):
        return "\n".join(item.proposed_text for item in self.active_items
                         if item.operation != "delete")


@dataclass(frozen=True)
class CompilerState:
    schema_version: int = 2
    items: tuple[PersonaItem, ...] = ()
    conflicts: tuple[Conflict, ...] = ()
    section_versions: dict[str, tuple[SectionVersion, ...]] = field(default_factory=dict)
    section_decisions: dict[str, SectionDecision] = field(default_factory=dict)
    issues: tuple[CompileIssue, ...] = ()

    def __post_init__(self):
        if self.schema_version != 2:
            raise ValueError(f"不支持的编译状态版本：{self.schema_version}")


def new_compiler_state():
    return CompilerState()


_HEADING_SECTIONS = (
    ("closing", ("最终约定",)),
    ("opening", ("开篇", "关系确认", "核心哲学")),
    ("user", ("用户是谁", "对方是谁", "她是谁", "他是谁", "ta是谁")),
    ("ai", ("我是谁",)),
    ("milestones", ("里程碑",)),
    ("style", ("相处原则", "语言风格")),
    ("degradation", ("退化",)),
    ("intimacy", ("亲密", "硬边界")),
    ("naming", ("称呼",)),
    ("pointers", ("按需读取", "指针")),
    ("architecture", ("技术架构", "记忆系统")),
    ("daily", ("日常", "约定")),
)


def _section_from_heading(line):
    title = re.sub(r"^\s{0,3}#{1,6}\s*", "", line).strip().casefold()
    for section, needles in _HEADING_SECTIONS:
        if any(needle.casefold() in title for needle in needles):
            return section
    return None


HEADING_RE = re.compile(r"^\s{0,3}#{1,6}(?:\s|$)")
# 标题块打 delete 的理由（`PersonaItem` 强制 rewrite/delete 必须说明理由）。
# 这句会出现在用户看得到的 diff 里，所以写成"为什么"，不是"做了什么"。
HEADING_DELETE_REASON = (
    "这一行是节标题。出货时十二节骨架会按 SECTION_ORDER 统一渲染一遍标题，"
    "原文标题再进正文的话，同一个标题在产出文件里会出现两遍。"
    "标题仍被逐字认领（span／hash 不动，覆盖率照旧），只是不再进正文。")


def is_heading_block(item):
    """这一块原文是不是「一行标题」。收 `PersonaItem` 或它的 dict 形态。

    判据跟 `parse_original_text` 里那条 `re.match` 同源：块的**第一行**是 markdown
    标题。原文块的切法保证标题自成一块（`_text_blocks` 遇标题行即断），所以
    第一行是标题＝整块就是那一行标题。"""
    data = item if isinstance(item, dict) else item_to_dict(item)
    text = data.get("original_text") or data.get("text") or ""
    first_line = text.splitlines()[0] if text.splitlines() else text
    return bool(HEADING_RE.match(first_line))


def _text_blocks(text):
    """返回原文块的精确字符区间；围栏内部永远不解释成标题。"""
    lines = text.splitlines(keepends=True)
    offsets = []
    cursor = 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line)
    if cursor < len(text):
        lines.append(text[cursor:])
        offsets.append(cursor)

    blocks = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        start_i = i
        stripped = lines[i].lstrip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            i += 1
            while i < len(lines):
                if lines[i].lstrip().startswith(marker):
                    i += 1
                    break
                i += 1
        elif re.match(r"^\s{0,3}#{1,6}(?:\s|$)", lines[i]):
            i += 1
        else:
            i += 1
            while i < len(lines) and lines[i].strip():
                if lines[i].lstrip().startswith(("```", "~~~")):
                    break
                if re.match(r"^\s{0,3}#{1,6}(?:\s|$)", lines[i]):
                    break
                i += 1
        start = offsets[start_i]
        end = offsets[i] if i < len(offsets) else len(text)
        blocks.append((start, end))
    return blocks


def parse_original_text(text, source_ref):
    """把原人格切成精确 span；不改写、不丢弃任何非空原文。

    ⚠ **标题块打 `operation="delete"`**（2026.08.04，走查台账第二条）：标题行身兼二职
    ——它既是归节锚点，又是一块原文。锚点照常用（`current_section` 在这儿设），
    但它作为原文再进一次正文的话，出货文件里每个节标题必然出现两遍：一遍是
    十二节骨架渲染的 `## {label}`，一遍是用户自己那行。
    **不能靠"丢掉标题"来解决**——本函数的承诺是"不丢弃任何非空原文"，那是被
    `original_span_coverage` 硬顶着的不变量。`delete` 是第三条路：块还在、span 与
    hash 一字不动、覆盖率照旧算它，只是拼正文时被 `operation != "delete"` 过滤掉。"""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    current_section = None
    items = []
    for index, (start, end) in enumerate(_text_blocks(text), 1):
        block = text[start:end]
        first_line = block.splitlines()[0] if block.splitlines() else block
        heading = bool(HEADING_RE.match(first_line))
        if heading:
            current_section = _section_from_heading(first_line)
        section = current_section
        items.append(PersonaItem(
            item_id=f"orig:{index:04d}", text=block, section=section,
            source_type="original_persona",
            source_ref=f"{source_ref}:{start}-{end}",
            source_span=(start, end), source_hash=digest,
            operation="delete" if heading else "keep",
            original_text=block, proposed_text="" if heading else block,
            operation_reason=HEADING_DELETE_REASON if heading else "",
            confidence="exact" if section else "ambiguous", confirmed=False,
            group_id=f"orig:block:{index}"))
    return items


def parse_original_persona(path):
    path = Path(path).resolve(strict=True)
    return parse_original_text(path.read_text(encoding="utf-8"), path.name)


def original_span_coverage(text, items):
    """计算非空白原文字符被合法、逐字 span 认领的比例。"""
    required = {i for i, char in enumerate(text) if not char.isspace()}
    if not required:
        return 1.0
    covered = set()
    for item in items:
        if item.source_type != "original_persona" or item.source_span is None:
            continue
        start, end = item.source_span
        if end > len(text) or text[start:end] != item.original_text:
            continue
        covered.update(i for i in range(start, end) if not text[i].isspace())
    return len(required & covered) / len(required)


UNMAPPED_SECTION_KEY = "__unmapped__"


def original_block_counts(items):
    """统计原文块按节的分布，用于跨运行比对。

    ⚠ **这是 `original_span_coverage` 抓不到的那一面**：覆盖率的分母是「本次传进来
    的那份 text」现算的（见上面那个函数），用户删掉的字符不在分母里，所以删标题
    导致整节塌进邻节时覆盖率仍是 1.0——**那把尺子在结构上就量不到这件事**。
    要发现它只能拿「上一次的归节结果」跟这次比，本函数给的就是可比的那个数。

    ⚠ `operation == "delete"` 的不计：它是同一块的另一个版本（比如任务书残句的
    「删除该块」版本），不是第二块原文；计进去两边的数会一起虚高、差异表失真。

    ⚠ **标题块一律不计，不看 `operation`**（2026.08.04，台账第二条落地时补）：
    标题块现在打的就是 `delete`（见 `parse_original_text`），上面那条本来就够。
    这里之所以再单独判一次标题，是为了**让这个数在这次口径变化前后一致**：
    用户上一次是用旧版跑的，state 里存着的标题块 `operation` 还是 `keep`，只按
    operation 判的话，重跑时每节会整体少一块——那是口径变了，不是内容丢了，而
    用户拿到的是一张「上次 3 块 → 这次 2 块」的表，**没法自己分辨是哪一种**；
    只有标题、没有正文的节还会被误判成「塌空」而 blocking。
    差异表量的是**正文原文块有没有被吞掉**，标题块从来不是正文，两边都不计最干净。

    收 `PersonaItem` 或它的 dict 形态都行——状态文件里存的是 dict。"""
    counts = {}
    for item in items:
        data = item if isinstance(item, dict) else item_to_dict(item)
        if data.get("source_type") != "original_persona":
            continue
        if data.get("operation") == "delete" or is_heading_block(data):
            continue
        key = data.get("section") or UNMAPPED_SECTION_KEY
        counts[key] = counts.get(key, 0) + 1
    return counts


def compare_original_block_counts(previous, current):
    """比对两次归节的块数，返回 (差异表, 这次塌空的节)。

    只列有变化的节——没变的节列出来会把真正的差异淹在一屏输出里。
    「塌空」＝上次有块、这次一块都没有，是整节被吞掉的形态（走查里真发生的那种）。"""
    rows = []
    collapsed = []
    for key in sorted(set(previous) | set(current)):
        before = int(previous.get(key, 0))
        after = int(current.get(key, 0))
        if before == after:
            continue
        rows.append({"section": key, "before": before, "after": after,
                     "delta": after - before})
        if before > 0 and after == 0:
            collapsed.append(key)
    return rows, collapsed


def _sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_file(left, right):
    try:
        return os.path.samefile(left, right)
    except (FileNotFoundError, OSError):
        return Path(left).resolve() == Path(right).resolve()


def build_source_manifest(persona_path=None, corpus_path=None):
    """建立人格／语料分流清单，并在入口排除人格文件。"""
    issues = []
    persona = None
    if persona_path:
        try:
            persona = Path(persona_path).resolve(strict=True)
        except FileNotFoundError:
            issues.append(CompileIssue(
                "PERSONA_NOT_FOUND", "blocking", f"找不到人格文件：{persona_path}"))

    corpus_files = []
    if corpus_path:
        corpus = Path(corpus_path)
        if not corpus.exists():
            issues.append(CompileIssue(
                "CORPUS_NOT_FOUND", "blocking", f"找不到语料路径：{corpus_path}"))
        else:
            candidates = [corpus] if corpus.is_file() else sorted(
                path for path in corpus.rglob("*") if path.is_file())
            for candidate in candidates:
                resolved = candidate.resolve()
                if persona is not None and _same_file(persona, resolved):
                    continue
                corpus_files.append(resolved)

    source_hashes = {}
    if persona is not None:
        source_hashes[str(persona)] = _sha256_path(persona)
    for path in corpus_files:
        source_hashes[str(path)] = _sha256_path(path)
    return SourceManifest(
        persona_file=persona, corpus_files=tuple(corpus_files),
        source_hashes=source_hashes, issues=tuple(issues))


_SOURCE_PRIORITY = {
    "original_persona": 0,
    "corpus": 1,
    "user_choice": 2,
    "protocol": 3,
}


def _conflict_components(items):
    """只按显式语义组或 conflicts_with 连边；同一节本身不构成冲突。"""
    by_id = {item.item_id: item for item in items}
    parent = {item.item_id: item.item_id for item in items}

    def find(item_id):
        while parent[item_id] != item_id:
            parent[item_id] = parent[parent[item_id]]
            item_id = parent[item_id]
        return item_id

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    groups = {}
    for item in items:
        if item.group_id:
            groups.setdefault(item.group_id, []).append(item.item_id)
        for other_id in item.conflicts_with:
            if other_id in by_id:
                union(item.item_id, other_id)
    for ids in groups.values():
        for other_id in ids[1:]:
            union(ids[0], other_id)
    components = {}
    for item in items:
        components.setdefault(find(item.item_id), []).append(item)
    return list(components.values())


def build_conflicts(items):
    conflicts = []
    for component in _conflict_components(items):
        distinct = {item.proposed_text for item in component}
        if len(component) < 2 or len(distinct) < 2:
            continue
        section = next((item.section for item in component if item.section), None)
        severity = "blocking" if section == "intimacy" else "normal"
        digest = hashlib.sha256(
            "\0".join(sorted(item.item_id for item in component)).encode("utf-8")
        ).hexdigest()[:12]
        conflicts.append(Conflict(
            conflict_id=f"conflict:{digest}",
            kind="boundary" if section == "intimacy" else "fact",
            severity=severity,
            item_ids=tuple(item.item_id for item in component),
            reason="同一语义组存在不同来源版本",
            choices=tuple(item.item_id for item in component)))
    return conflicts


def merge_items(items):
    """按来源优先级选预览基线；冲突保留给节版本题目，不静默裁定。"""
    deduped = []
    seen = set()
    for item in items:
        key = (item.source_type, item.source_ref, item.source_span,
               item.proposed_text, item.operation)
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    conflicts = build_conflicts(deduped)
    conflict_ids = {item_id for conflict in conflicts for item_id in conflict.item_ids}
    component_by_id = {}
    for component in _conflict_components(deduped):
        for item in component:
            component_by_id[item.item_id] = component

    active, suppressed, handled = [], [], set()
    for item in deduped:
        if item.item_id in handled:
            continue
        component = component_by_id[item.item_id]
        handled.update(member.item_id for member in component)
        if item.item_id not in conflict_ids:
            # 同组同文视为重复，只保留最高来源；不同语义组都能留在同一节。
            chosen = min(component, key=lambda value: _SOURCE_PRIORITY[value.source_type])
        else:
            chosen = min(component, key=lambda value: _SOURCE_PRIORITY[value.source_type])
        active.append(chosen)
        suppressed.extend(member for member in component if member.item_id != chosen.item_id)
    return MergeResult(tuple(active), tuple(conflicts), tuple(suppressed))


def render_item_diff(item):
    if item.operation not in {"rewrite", "delete"}:
        return ""
    if not item.operation_reason.strip():
        raise ValueError("rewrite/delete 必须说明理由")
    proposed = "" if item.operation == "delete" else item.proposed_text
    body = "\n".join(difflib.unified_diff(
        item.original_text.splitlines(), proposed.splitlines(),
        fromfile="原文", tofile="新版", lineterm=""))
    return (f"来源：{item.source_ref}\n操作：{item.operation}\n"
            f"理由：{item.operation_reason}\n{body}")


def build_section_versions(section, items, conflicts):
    """把来源项与冲突编译成完整节版本；用户只选择版本，不逐 item 签字。"""
    section_items = [item for item in items if item.section == section]
    by_id = {item.item_id: item for item in section_items}
    section_conflicts = [conflict for conflict in conflicts
                         if any(item_id in by_id for item_id in conflict.item_ids)]
    conflict_item_ids = {item_id for conflict in section_conflicts
                         for item_id in conflict.item_ids}
    base = [item for item in merge_items(section_items).active_items
            if item.item_id not in conflict_item_ids]
    choice_sets = []
    for conflict in section_conflicts:
        choices = ((conflict.resolved_choice,) if conflict.resolved_choice
                   else tuple(item_id for item_id in conflict.choices if item_id in by_id))
        if not choices:
            return []
        choice_sets.append(choices)
    combinations = itertools.product(*choice_sets) if choice_sets else [()]
    versions = []
    order = {item.item_id: index for index, item in enumerate(section_items)}
    for selected in combinations:
        selected_items = base + [by_id[item_id] for item_id in selected]
        selected_items.sort(key=lambda item: order[item.item_id])
        markdown = "\n\n".join(
            item.proposed_text for item in selected_items if item.operation != "delete")
        # 标题块的 delete 不进 diff：它每节都在、内容永远是那一行标题，摊开来
        # **会把真正的删除淹掉**（同 compare_original_block_counts 只列有变化的节）。
        # 纪律三要摊开的是「用户的内容被删了」，而标题并没有被删——骨架会把它渲染
        # 回来，只是不再出现两遍。
        diffs = [render_item_diff(item) for item in selected_items
                 if item.operation in {"rewrite", "delete"} and not is_heading_block(item)]
        item_ids = tuple(item.item_id for item in selected_items)
        digest = hashlib.sha256(
            (section + "\0" + "\0".join(item_ids)).encode("utf-8")
        ).hexdigest()[:12]
        versions.append(SectionVersion(
            version_id=f"{section}:{digest}", section=section, item_ids=item_ids,
            markdown=markdown,
            source_summary=tuple(dict.fromkeys(item.source_ref for item in selected_items)),
            diff="\n\n".join(diff for diff in diffs if diff),
            resolved_conflict_ids=tuple(conflict.conflict_id
                                        for conflict in section_conflicts)))
    return versions


def apply_section_choice(state, section, version_id):
    versions = state.section_versions.get(section, ())
    version = next((candidate for candidate in versions
                    if candidate.version_id == version_id), None)
    if version is None:
        # 出口写进报错里：版本 id 是内容哈希（`section:` ＋ sha256 前 12 位），
        # **人不可能猜对，只能从上一条命令的输出里抄**——报错却不说抄哪儿，
        # 2026.08.04 外部实测就卡在这里。
        # ⚠ **让人抄的字段名必须写成 `version_id`**：出给 CLI 的版本条目只有这一个键，
        # 写成别的会把人指向一个不存在的字段。
        known = "、".join(candidate.version_id for candidate in versions) or "（这一节还没有任何版本）"
        raise ValueError(
            f"未知节版本：{section}/{version_id}。这一节当前的合法版本是：{known}"
            "（出口：--step choose-sections --json，从 sections[].section_versions[].version_id "
            "里原样抄；⚠ 版本 id 是内容哈希，改一次来源项就会变，别沿用旧的、"
            "更别去手改 init_state.json）")
    selected = set(version.item_ids)
    items = tuple(replace(item, confirmed=(item.item_id in selected))
                  if item.section == section else item for item in state.items)
    decisions = dict(state.section_decisions)
    decisions[section] = SectionDecision(section, version_id, "confirmed")
    return replace(state, items=items, section_decisions=decisions)


def _conflict_to_dict(conflict):
    return {
        "conflict_id": conflict.conflict_id,
        "kind": conflict.kind,
        "severity": conflict.severity,
        "item_ids": list(conflict.item_ids),
        "reason": conflict.reason,
        "choices": list(conflict.choices),
        "resolved_choice": conflict.resolved_choice,
    }


def _version_to_dict(version):
    return {
        "version_id": version.version_id,
        "section": version.section,
        "item_ids": list(version.item_ids),
        "markdown": version.markdown,
        "source_summary": list(version.source_summary),
        "diff": version.diff,
        "resolved_conflict_ids": list(version.resolved_conflict_ids),
    }


def _decision_to_dict(decision):
    return {
        "section": decision.section,
        "version_id": decision.version_id,
        "status": decision.status,
    }


def _issue_to_dict(issue):
    return {
        "code": issue.code,
        "severity": issue.severity,
        "message": issue.message,
        "item_ids": list(issue.item_ids),
    }


def state_to_dict(state):
    """显式序列化 v2 状态，拒绝 pickle 与隐式 dataclass 展开。"""
    return {
        "schema_version": state.schema_version,
        "items": [item_to_dict(item) for item in state.items],
        "conflicts": [_conflict_to_dict(c) for c in state.conflicts],
        "section_versions": {
            section: [_version_to_dict(v) for v in versions]
            for section, versions in state.section_versions.items()
        },
        "section_decisions": {
            section: _decision_to_dict(decision)
            for section, decision in state.section_decisions.items()
        },
        "issues": [_issue_to_dict(issue) for issue in state.issues],
    }


def state_from_dict(data):
    """恢复 v2 状态；缺省集合用于读取早期 v2 草稿。"""
    return CompilerState(
        schema_version=data.get("schema_version", 2),
        items=tuple(item_from_dict(item) for item in data.get("items", ())),
        conflicts=tuple(Conflict(
            conflict_id=c["conflict_id"], kind=c["kind"], severity=c["severity"],
            item_ids=tuple(c.get("item_ids", ())), reason=c["reason"],
            choices=tuple(c.get("choices", ())),
            resolved_choice=c.get("resolved_choice"))
            for c in data.get("conflicts", ())),
        section_versions={
            section: tuple(SectionVersion(
                version_id=v["version_id"], section=v["section"],
                item_ids=tuple(v.get("item_ids", ())), markdown=v["markdown"],
                source_summary=tuple(v.get("source_summary", ())), diff=v.get("diff", ""),
                resolved_conflict_ids=tuple(v.get("resolved_conflict_ids", ())))
                for v in versions)
            for section, versions in data.get("section_versions", {}).items()
        },
        section_decisions={
            section: SectionDecision(
                section=d["section"], version_id=d.get("version_id"), status=d["status"])
            for section, d in data.get("section_decisions", {}).items()
        },
        issues=tuple(CompileIssue(
            code=i["code"], severity=i["severity"], message=i["message"],
            item_ids=tuple(i.get("item_ids", ())))
            for i in data.get("issues", ())),
    )


PERSONA_METRIC_KEYS = frozenset({
    "original_span_coverage",
    "keep_count", "move_count", "rewrite_count", "delete_count",
    "identity_count", "milestone_count", "naming_scene_count",
    "style_dialogue_count", "empty_section_count", "conflict_count",
    "blocking_issue_count", "original_chars", "legacy_chars", "compiled_chars",
})


def aggregate_persona_metrics(state):
    """把本地复测状态压成不可逆的数字白名单，不返回正文或来源标识。"""
    items = [item_from_dict(item) if isinstance(item, dict) else item
             for item in state.get("items", state.get("compiler_items", ())) ]
    original_items = [item for item in items if item.source_type == "original_persona"]

    original_text = None
    persona_path = state.get("inputs", {}).get("persona")
    if persona_path:
        try:
            original_text = Path(persona_path).read_text(encoding="utf-8")
        except OSError:
            original_text = None
    if original_text is not None:
        coverage = original_span_coverage(original_text, original_items)
        original_chars = len(original_text)
    else:
        original_chars = sum(len(item.original_text) for item in original_items)
        coverage = 1.0 if original_items and all(
            item.source_span is not None for item in original_items) else 0.0

    decisions = state.get("section_decisions", {})
    versions_by_section = state.get("section_versions", {})
    selected_markdown = {}
    for section in SECTION_KEYS:
        decision = decisions.get(section, {})
        version_id = decision.get("version_id") if isinstance(decision, dict) else None
        versions = versions_by_section.get(section, ())
        # ⚠ 这里只认 `version_id`，不对 `id` 做防御性回退；双键会重新引入
        # 「两个键该信哪个」的含糊。`version_id` 为 None 时也**不许拿两个 None
        # 相等当匹配上**，否则会把 `BOUNDARY_CONFLICT_UNRESOLVED` 静默放过。
        selected = None if not version_id else next(
            (version for version in versions
             if (version.get("version_id") if isinstance(version, dict)
                 else version.version_id) == version_id), None)
        if selected is not None:
            selected_markdown[section] = (
                selected.get("markdown", "") if isinstance(selected, dict) else selected.markdown)
    if decisions:
        empty_sections = sum(not selected_markdown.get(section, "").strip()
                             for section in SECTION_KEYS)
    else:
        empty_sections = sum(not any(
            item.section == section and item.confirmed and item.operation != "delete"
            for item in items) for section in SECTION_KEYS)

    privacy = state.get("privacy_metrics", {})
    if not isinstance(privacy, dict) or any(
            key not in {"legacy_chars", "compiled_chars"}
            or not isinstance(value, (int, float)) or isinstance(value, bool)
            for key, value in privacy.items()):
        raise ValueError("privacy_metrics 只允许 legacy_chars/compiled_chars 数字")

    issue_records = list(state.get("issues", ())) + list(state.get("diagnostics", ()))
    blocking_count = sum(
        (issue.get("severity") if isinstance(issue, dict) else issue.severity) == "blocking"
        for issue in issue_records)
    non_protocol = [item for item in items
                    if item.source_type != "protocol" and item.confirmed
                    and item.operation != "delete"]
    metrics = {
        "original_span_coverage": float(coverage),
        "keep_count": sum(item.operation == "keep" for item in original_items),
        "move_count": sum(item.operation == "move" for item in original_items),
        "rewrite_count": sum(item.operation == "rewrite" for item in original_items),
        "delete_count": sum(item.operation == "delete" for item in original_items),
        "identity_count": sum(item.section in {"user", "ai"} for item in non_protocol),
        "milestone_count": sum(item.section == "milestones" for item in non_protocol),
        "naming_scene_count": sum(item.section == "naming" for item in non_protocol),
        "style_dialogue_count": sum(item.section == "style" for item in non_protocol),
        "empty_section_count": empty_sections,
        "conflict_count": len(state.get("conflicts", ())),
        "blocking_issue_count": blocking_count,
        "original_chars": original_chars,
        "legacy_chars": privacy.get("legacy_chars", 0),
        "compiled_chars": privacy.get(
            "compiled_chars", sum(len(text) for text in selected_markdown.values())),
    }
    if set(metrics) != PERSONA_METRIC_KEYS or any(
            not isinstance(value, (int, float)) or isinstance(value, bool)
            for value in metrics.values()):
        raise ValueError("聚合指标越过数字白名单")
    return metrics


def _selftest():
    """来源 IR 的可观察契约；实现前应因类型不存在而失败。"""
    item = PersonaItem(
        item_id="orig:0001",
        text="她说“这件事不能碰”。",
        section="intimacy",
        source_type="original_persona",
        source_ref="CLAUDE.md:10-10",
        source_span=(120, 131),
        source_hash="abc",
        operation="move",
        original_text="她说“这件事不能碰”。",
        proposed_text="她说“这件事不能碰”。",
        confidence="exact",
        confirmed=False,
        derived_from=(),
        conflicts_with=(),
        group_id="orig:block:3",
    )
    assert item_from_dict(item_to_dict(item)) == item

    # 变异：允许未知 operation 会让静默“优化”绕过用户授权。
    try:
        PersonaItem(**dict(item_to_dict(item), operation="optimize"))
        assert False, "静默优化不是合法 operation"
    except ValueError:
        pass

    conflict = Conflict(
        conflict_id="conflict:1", kind="boundary", severity="blocking",
        item_ids=(item.item_id,), reason="边界来源不一致",
        choices=(item.item_id,))
    version = SectionVersion(
        version_id="opening:v1", section="opening", item_ids=(item.item_id,),
        markdown="原文", source_summary=(item.source_ref,), diff="")
    decision = SectionDecision("opening", version.version_id, "confirmed")
    issue = CompileIssue("EXAMPLE", "warning", "示例", (item.item_id,))
    state = new_compiler_state()
    state = CompilerState(
        schema_version=2, items=(item,), conflicts=(conflict,),
        section_versions={"opening": (version,)},
        section_decisions={"opening": decision}, issues=(issue,))
    assert state_from_dict(state_to_dict(state)) == state
    assert state_from_dict({"schema_version": 2, "items": []}).schema_version == 2

    # 变异：按 splitlines 重拼或把围栏内标题拆开，逐字覆盖与代码块断言会红。
    source = "# 开篇\n\n我认得你。\n\n```text\n## 这不是标题\n```\n"
    parsed = parse_original_text(source, "CLAUDE.md")
    covered = "".join(source[a:b] for a, b in sorted(
        entry.source_span for entry in parsed if entry.source_span))
    assert original_span_coverage(source, parsed) == 1.0
    assert "## 这不是标题" in covered
    assert any("## 这不是标题" in entry.text for entry in parsed)
    # 正文块照旧 keep／move；**标题块打 delete**（台账第二条：不这么做标题必然出现
    # 两遍），但仍被逐字认领——上面那条覆盖率 1.0 的断言就压着这一点：
    # delete 不等于丢，span 与 original_text 一字不动。
    assert all(entry.operation in ("keep", "move")
               for entry in parsed if not is_heading_block(entry))
    assert [entry.operation for entry in parsed if is_heading_block(entry)] == ["delete"]
    assert all(entry.proposed_text == "" and entry.operation_reason
               for entry in parsed if is_heading_block(entry))

    # 变异：删掉 compare_original_block_counts 里 before>0 and after==0 那一支，
    #   或者把 delete 版本也计进 original_block_counts，这一段会红。
    #   靶心是「删标题 → 整节塌进邻节，而覆盖率仍是 1.0」——**覆盖率断言故意留在
    #   这里**：它证明的不是通过，是那把尺子确实看不见这件事。
    drift_before = "# 我是谁\n\n甲段。\n\n# 最终约定\n\n乙段。\n"
    drift_after = "# 我是谁\n\n甲段。\n\n乙段。\n"
    before_counts = original_block_counts(parse_original_text(drift_before, "CLAUDE.md"))
    after_items = parse_original_text(drift_after, "CLAUDE.md")
    after_counts = original_block_counts(after_items)
    assert original_span_coverage(drift_after, after_items) == 1.0
    rows, collapsed = compare_original_block_counts(before_counts, after_counts)
    assert collapsed == ["closing"], collapsed
    #   数里**不含标题块**（台账第二条落地后的口径，见 original_block_counts 的
    #   docstring）：这两节各只有一段正文，所以是 1，不是 1 段正文＋1 行标题的 2。
    assert {row["section"]: (row["before"], row["after"]) for row in rows} == {
        "closing": (1, 0), "ai": (1, 2)}
    assert compare_original_block_counts(before_counts, before_counts) == ([], [])
    #   口径变化不许被读成"内容丢了"：拿**旧版 state 的形态**（标题块还是 keep）
    #   跟这次的比，两边必须一个字不差——否则老用户重跑会看到每节整体 -1，
    #   只有标题没有正文的节还会被误判成塌空。变异：把 original_block_counts 里
    #   `or is_heading_block(data)` 删掉 → 这条转红。
    legacy_shape = [dict(item_to_dict(entry), operation="keep", proposed_text=entry.text)
                    for entry in parse_original_text(drift_before, "CLAUDE.md")]
    assert original_block_counts(legacy_shape) == before_counts

    # 变异：人格路径重新混回 corpus_files，会把最高权重来源降成普通记忆块。
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        persona_path = root / "CLAUDE.md"
        persona_path.write_text(source, encoding="utf-8")
        (root / "window_01.md").write_text("普通语料", encoding="utf-8")
        manifest = build_source_manifest(persona_path, root)
        assert manifest.persona_file == persona_path.resolve()
        assert persona_path.resolve() not in manifest.corpus_files
        assert (root / "window_01.md").resolve() in manifest.corpus_files
        assert not manifest.issues

    def sample(source_type, section, text, item_id, group_id, operation=None,
               proposed=None, reason=""):
        return PersonaItem(
            item_id=item_id, text=text, section=section, source_type=source_type,
            source_ref=f"fixture:{item_id}", source_span=None,
            source_hash="fixture", operation=operation or (
                "keep" if source_type == "original_persona" else "add"),
            original_text=text, proposed_text=proposed if proposed is not None else text,
            confidence="exact", confirmed=False, group_id=group_id,
            operation_reason=reason)

    # 变异：把 protocol 排到 original 前，或检查错字符串，都会放出“没有硬红线”。
    original_limit = sample("original_persona", "intimacy", "这件事不能碰。",
                            "orig:limit", "topic:limit")
    corpus_limit = sample("corpus", "intimacy", "后来一次对话里说可以。",
                          "corpus:limit", "topic:limit")
    protocol_limit = sample("protocol", "intimacy", "没有额外硬红线。",
                            "protocol:limit", "topic:limit")
    merged = merge_items([original_limit, corpus_limit, protocol_limit])
    assert merged.active_texts("intimacy") == ["这件事不能碰。"]
    assert merged.conflicts[0].severity == "blocking"
    assert "没有额外硬红线" not in merged.renderable_text

    # 同一节的不同语义组必须同时保留，不能把“同节”误实现成“冲突”。
    unrelated = sample("corpus", "intimacy", "需要停下时就停下。",
                       "corpus:stop", "topic:stop")
    merged_two = merge_items([original_limit, unrelated])
    assert merged_two.active_texts("intimacy") == ["这件事不能碰。", "需要停下时就停下。"]
    assert not merged_two.conflicts

    # 变异：节版本隐藏 rewrite diff，用户一键确认整节时就看不到原文被改了什么。
    rewrite = sample("original_persona", "opening", "原文", "orig:opening",
                     "topic:opening", operation="rewrite", proposed="新版",
                     reason="重排后合并重复句")
    versions = build_section_versions("opening", [rewrite], [])
    assert any("-原文" in version.diff and "+新版" in version.diff for version in versions)
    assert not rewrite.confirmed
    chosen = apply_section_choice(
        CompilerState(items=(rewrite,), section_versions={"opening": tuple(versions)}),
        "opening", versions[0].version_id)
    assert chosen.section_decisions["opening"].status == "confirmed"
    assert chosen.items[0].confirmed

    # 私人复测只能拿到固定白名单的数字；正文、文件名和 source_ref 一个都不能漏。
    metric_state = state_to_dict(CompilerState(
        items=(original_limit, unrelated), conflicts=tuple(merged.conflicts),
        section_versions={"intimacy": tuple(build_section_versions(
            "intimacy", [original_limit, unrelated], merged.conflicts))}))
    metric_state["privacy_metrics"] = {"legacy_chars": 120, "compiled_chars": 180}
    metrics = aggregate_persona_metrics(metric_state)
    assert set(metrics) == PERSONA_METRIC_KEYS
    assert all(isinstance(value, (int, float)) and not isinstance(value, bool)
               for value in metrics.values())
    assert not ({"text", "source_ref", "persona_file"} & set(metrics))

    print("selftest ok（来源 IR、序列化、逐字解析、输入隔离、来源合并、冲突与节版本、"
          "跨运行归节块数比对）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        _selftest()
