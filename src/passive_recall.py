"""自动浮现 W1～W3：只读候选、程序准入、来源证据与分层冷却。

本层复用现有检索并给候选建立可核验身份；W3 在提示组装前完成输入预筛、许可短句、
范围／来源／明确冲突、完整依赖、上下文覆盖及同 delivery 冷却。聊天模型仍负责细微
意图，W4 才负责原文切片和现场模板。候选不是曝光，返回 ``candidate`` 时宿主不得把
原始结构直接塞给聊天模型。
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import unicodedata

from memory_retrieval import _chunk_key, tokenize
from passive_metadata import source_signature
from session_recall import DEFAULT_MAX_ITEM_CHARS


WIRE_VERSION = "passive-recall-w3-v1"
POLICY_VERSION = "passive-admission-w3-v1"
TOOL_NAME = "latent_passive_recall"

_LOW_INFORMATION = {
    "好", "好的", "嗯", "嗯嗯", "哦", "噢", "呵呵", "哈哈", "收到", "可以", "行",
    "谢谢", "烦", "在吗", "hi", "hello", "ok", "yes", "no",
}
_CONTROL_COMMAND = re.compile(r"^/[A-Za-z][A-Za-z0-9_-]*(?:\s+[^\r\n]+)?$")
_QUOTED_SPANS = re.compile(r"“[^”]*”|‘[^’]*’|\"[^\"]*\"|'[^']*'")


class PassiveRecallRequestError(ValueError):
    """宿主入口参数不满足 W0 冻结的请求契约。"""


def _digest(value):
    wire = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(wire).hexdigest()


def _canonical(value):
    value = unicodedata.normalize("NFKC", value).lower()
    return "".join(ch for ch in value
                   if not ch.isspace() and not unicodedata.category(ch).startswith("P"))


def _pure_quoted_or_code(value):
    """只识别能可靠剥离的整段引用／代码；混合文本保留用户自己的表达。"""
    stripped = value.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        return True
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if lines and all(line.startswith(">") for line in lines):
        return True
    return bool(_QUOTED_SPANS.fullmatch(stripped))


def _input_gate(value, *, short_terms, index):
    """返回（是否检索、原因、是否由短句许可开闸）。不拿长度冒充信息量。"""
    stripped = value.strip()
    if not stripped or not _canonical(stripped):
        return False, "low_information", False
    if _CONTROL_COMMAND.fullmatch(stripped):
        return False, "control_command", False
    if _pure_quoted_or_code(stripped):
        return False, "quoted_or_code_only", False
    normalized = _canonical(stripped)
    licensed = any(normalized == _canonical(term) for term in short_terms)
    if licensed:
        return True, "licensed_short_trigger", True
    if normalized in _LOW_INFORMATION:
        return False, "low_information", False
    # 这里查的是现成 BM25 词表里的区分性词面，不跑排序、不写权重。英文单词和数字
    # 也要有库内证据；不能用“字符够长”把任意日志或寒暄送进全文检索。
    if index.lexical_admit(tokenize(stripped)):
        return True, "specific_user_signal", False
    return False, "low_information", False


def _source(row):
    meta = row.get("meta") or {}
    return {
        "source": meta.get("source"),
        "heading": meta.get("heading"),
        "chunkIndex": meta.get("chunk_index"),
        "layer": meta.get("layer", "timeline"),
    }


def _range(text, *, limit=None):
    stripped = text.strip()
    start = text.find(stripped) if stripped else 0
    length = len(stripped) if limit is None else min(len(stripped), int(limit))
    end = start + length
    excerpt = text[start:end]
    return {"start": start, "end": end, "signature": _digest(excerpt)}


def describe_record(row, *, visible_limit=None):
    """把检索行转成宿主可核验的记录描述；revision 不含曝光或检索时间。"""
    text = row["text"]
    source = _source(row)
    record_id = _chunk_key(text)
    source_signature = _digest(source)
    revision = _digest({"recordId": record_id, "text": text,
                        "sourceSignature": source_signature})
    return {
        "recordId": record_id,
        "revision": revision,
        "state": "active",
        "source": source,
        "sourceSignature": source_signature,
        "ranges": [_range(text, limit=visible_limit)],
    }


def _ranges_cover(required, supplied):
    """坐标并集覆盖判定；两边签名已由当前正文复核，不拿相似文本冒充范围。"""
    intervals = sorted((int(item["start"]), int(item["end"])) for item in supplied)
    for need in required:
        cursor = int(need["start"])
        target = int(need["end"])
        for start, end in intervals:
            if end <= cursor:
                continue
            if start > cursor:
                break
            cursor = max(cursor, end)
            if cursor >= target:
                break
        if cursor < target:
            return False
    return True


class PassiveRecallService:
    """复用常驻 MemoryIndex 的准入服务；只返回 W4 可继续组装的单条事件线。"""

    def __init__(self, index, max_candidates=5, metadata_reader=None):
        self.index = index
        self.max_candidates = int(max_candidates)
        self.metadata_reader = metadata_reader or (lambda: {})
        self._deliveries = {}
        self._requests = {}
        self._lock = threading.RLock()

    def set_index(self, index):
        """常驻宿主重载语料后换同一服务的只读索引引用。"""
        with self._lock:
            self.index = index

    def _record_row(self, record_id):
        matches = [(idx, text, self.index.meta[idx])
                   for idx, text in enumerate(self.index.chunks)
                   if idx not in self.index.retracted and _chunk_key(text) == record_id]
        if len(matches) != 1:
            return None
        idx, text, meta = matches[0]
        return {"id": idx, "text": text, "meta": meta}

    def _validated_metadata(self, scope):
        """只发布能逐项回到当前权威正文的 sidecar；坏引用不会被静默删掉。"""
        records = self.metadata_reader()
        if not isinstance(records, dict):
            raise ValueError("被动元数据账本 records 不是对象")
        valid, invalid, known = {}, set(), set()
        for record_id, item in records.items():
            known.add(record_id)
            if not isinstance(item, dict) or item.get("recordId") != record_id:
                invalid.add(record_id)
                continue
            trigger_terms = item.get("trigger_terms")
            short_terms = item.get("short_trigger_terms")
            if not isinstance(trigger_terms, list) or not isinstance(short_terms, list) \
                    or any(not isinstance(term, str) or not term.strip()
                           for term in trigger_terms + short_terms) \
                    or any(term not in trigger_terms for term in short_terms):
                invalid.add(record_id)
                continue
            if item.get("scope", "general") != scope:
                continue
            row = self._record_row(record_id)
            if row is None or item.get("sourceSignature") != source_signature(row["text"]):
                invalid.add(record_id)
                continue
            unsigned = {key: value for key, value in item.items() if key != "revision"}
            if item.get("revision") != _digest(unsigned):
                invalid.add(record_id)
                continue
            trigger_ranges = item.get("trigger_ranges") or []
            source_ranges = item.get("source_ranges") or []
            refs = item.get("context_refs") or []
            if not isinstance(trigger_ranges, list) or not isinstance(source_ranges, list) \
                    or not isinstance(refs, list):
                invalid.add(record_id)
                continue
            broken = False
            for part in trigger_ranges + source_ranges:
                if not isinstance(part, dict):
                    broken = True
                    break
                start, end = part.get("start"), part.get("end")
                if not isinstance(start, int) or not isinstance(end, int) \
                        or start < 0 or end > len(row["text"]) or start >= end \
                        or part.get("signature") != _digest(row["text"][start:end]):
                    broken = True
                    break
            for ref in refs:
                if not isinstance(ref, dict) or not isinstance(ref.get("source_ranges"), list):
                    broken = True
                    break
                ref_row = self._record_row(ref.get("recordId"))
                if ref_row is None or ref.get("sourceSignature") != source_signature(ref_row["text"]):
                    broken = True
                    break
                for part in ref.get("source_ranges") or ():
                    start, end = part.get("start"), part.get("end")
                    if not isinstance(start, int) or not isinstance(end, int) \
                            or start < 0 or end > len(ref_row["text"]) or start >= end \
                            or part.get("signature") != _digest(ref_row["text"][start:end]):
                        broken = True
                        break
            if broken:
                invalid.add(record_id)
            else:
                valid[record_id] = item
        return valid, invalid, known

    @staticmethod
    def _anchors(item):
        return tuple(item.get("trigger_terms") or ())

    @staticmethod
    def _explicit_conflict(user_input, item, *, unique_candidate):
        """只匹配规格冻结的完整句式；引语与未覆盖语法不猜。"""
        own = _QUOTED_SPANS.sub("", user_input)
        compact = _canonical(own)
        anchors = [_canonical(term) for term in PassiveRecallService._anchors(item)]
        if any(compact == _canonical(f"这次不是在说{term}") for term in anchors):
            return True
        if any(compact == _canonical(f"不要接{term}") for term in anchors):
            return True
        if compact == _canonical("不要接这个梗") and unique_candidate:
            return True
        if any(compact == _canonical(f"我现在认真说{term}，别开玩笑") for term in anchors):
            return True
        if any(compact == _canonical(f"今天不要{term}") for term in anchors):
            return True
        return False

    def _descriptor(self, row, item=None, *, ranges=None):
        record = describe_record(row)
        if ranges:
            record["ranges"] = [{key: part[key] for key in ("start", "end", "signature")}
                                for part in ranges]
        if item is not None:
            record["passiveRevision"] = item.get("revision")
            record["kind"] = item.get("kind", "unknown")
            record["scope"] = item.get("scope", "general")
            record["episodeId"] = item.get("episode_id")
        if item is not None or ranges:
            record["revision"] = _digest({
                "bodyRevision": record["revision"],
                "passiveRevision": item.get("revision") if item is not None else None,
                "ranges": record["ranges"],
            })
        return record

    def _candidate_dependencies(self, row, item):
        ranges = (item or {}).get("source_ranges") or None
        records = [self._descriptor(row, item, ranges=ranges)]
        if item is not None:
            for ref in item.get("context_refs") or ():
                ref_row = self._record_row(ref["recordId"])
                records.append(self._descriptor(ref_row, ranges=ref.get("source_ranges")))
        dependencies = []
        for record in records:
            dependency = {key: record[key] for key in
                          ("recordId", "revision", "state", "sourceSignature", "ranges")}
            if record.get("passiveRevision"):
                dependency["passiveRevision"] = record["passiveRevision"]
            dependencies.append(dependency)
        return records, dependencies

    def _dependencies_covered(self, dependencies, evidence):
        valid_evidence = self._valid_evidence(evidence)
        for dep in dependencies:
            matches = [item for item in valid_evidence
                       if item.get("recordId") == dep.get("recordId")]
            supplied = [part for item in matches for part in item.get("ranges", [])]
            if not _ranges_cover(dep.get("ranges", []), supplied):
                return False
        return True

    def candidate(self, request):
        if not isinstance(request, dict):
            raise PassiveRecallRequestError("请求必须是对象")
        user_input = request.get("userInput")
        turn = request.get("turn")
        if not isinstance(user_input, str):
            raise PassiveRecallRequestError("userInput 必须是字符串")
        if not isinstance(turn, dict) or not all(
                isinstance(turn.get(key), str) and turn.get(key)
                for key in ("sessionId", "turnId", "deliveryId")):
            raise PassiveRecallRequestError("turn 必须提供 sessionId／turnId／deliveryId")
        delivery_id = turn["deliveryId"]
        scope_value = request.get("scope")
        scope = "general" if scope_value is None else scope_value
        if not isinstance(scope, str) or not scope.strip():
            return {"status": "empty", "deliveryId": delivery_id,
                    "wireVersion": WIRE_VERSION, "policyVersion": POLICY_VERSION,
                    "reasonCodes": ["scope_unknown"]}
        request_key = (scope, turn["sessionId"], delivery_id)
        fingerprint = _digest(request)
        with self._lock:
            previous = self._requests.get(request_key)
        if previous is not None:
            if previous["fingerprint"] != fingerprint:
                raise PassiveRecallRequestError("同一 deliveryId 的请求内容不能变化")
            return json.loads(json.dumps(previous["response"]))

        try:
            metadata, invalid_metadata, known_metadata = self._validated_metadata(scope)
        except (OSError, UnicodeError, ValueError):
            response = {"status": "empty", "deliveryId": delivery_id,
                        "wireVersion": WIRE_VERSION, "policyVersion": POLICY_VERSION,
                        "reasonCodes": ["source_unresolved"]}
            return self._remember_request(request_key, fingerprint, response)
        short_terms = {term for item in metadata.values()
                       for term in item.get("short_trigger_terms") or ()}
        admitted, gate_reason, licensed = _input_gate(
            user_input, short_terms=short_terms, index=self.index)
        if not admitted:
            response = {"status": "empty", "deliveryId": delivery_id,
                        "wireVersion": WIRE_VERSION, "policyVersion": POLICY_VERSION,
                        "reasonCodes": [gate_reason]}
            return self._remember_request(request_key, fingerprint, response)

        rows = self.index.retrieve_candidates(user_input, topN=self.max_candidates)
        if not rows:
            response = {"status": "empty", "deliveryId": delivery_id,
                        "wireVersion": WIRE_VERSION, "policyVersion": POLICY_VERSION,
                        "reasonCodes": ["no_reliable_candidate"]}
            return self._remember_request(request_key, fingerprint, response)

        rejected = []
        eligible = []
        normalized_input = _canonical(user_input)
        for row in rows:
            if (row.get("meta") or {}).get("layer", "timeline") != "timeline":
                rejected.append("duplicate_summary")
                continue
            record_id = _chunk_key(row["text"])
            item = metadata.get(record_id)
            if record_id in invalid_metadata:
                rejected.append("source_unresolved")
                continue
            if record_id in known_metadata and item is None:
                rejected.append("scope_unknown")
                continue
            if licensed:
                if item is None or not any(
                        normalized_input == _canonical(term)
                        for term in item.get("short_trigger_terms") or ()):
                    continue
            eligible.append((row, item))

        # 显式登记且本轮精确提到的触发短语优先于仅靠相似词面撞中的旧记录；仍保留
        # 检索核心在同一层内的原顺序。这个优先级只选证据源，不推断本轮意图。
        eligible.sort(key=lambda pair: (
            0 if pair[1] is not None and any(
                _canonical(term) in normalized_input
                for term in pair[1].get("trigger_terms") or ()) else
            1 if pair[1] is not None else 2
        ))

        # 同一许可短句精确指向多个事件时，不拿检索第一名猜实体来源。
        exact_episodes = {item.get("episode_id") for _, item in eligible if item is not None
                          and any(normalized_input == _canonical(term)
                                  for term in item.get("trigger_terms") or ())}
        if len(exact_episodes) > 1:
            eligible = []
            rejected.append("ambiguous_source")
        if eligible:
            unique = sum(item is not None for _, item in eligible) == 1
            conflict = any(item is not None and self._explicit_conflict(
                user_input, item, unique_candidate=unique) for _, item in eligible)
            if conflict:
                # 完整否认命中明确候选后整轮留空；不能绕过它改塞一个相似但无元数据的
                # 旧块，那会把“冲突优先”降级成“换条记录继续猜”。
                eligible = []
                rejected.append("explicit_conflict")
        if not eligible:
            response = {"status": "empty", "deliveryId": delivery_id,
                        "wireVersion": WIRE_VERSION, "policyVersion": POLICY_VERSION,
                        "reasonCodes": list(dict.fromkeys(rejected)) or ["no_reliable_candidate"]}
            return self._remember_request(request_key, fingerprint, response)

        # 排名已由共用检索核心给出。只取一条主记录及它明确登记的必要背景；不为填数量
        # 拼接同人物的另一事件，也不把触发词表塞给宿主。
        row, item = eligible[0]
        records, dependencies = self._candidate_dependencies(row, item)
        context_evidence = request.get("contextEvidence")
        if context_evidence is not None and not isinstance(context_evidence, list):
            raise PassiveRecallRequestError("contextEvidence 必须是数组")
        if context_evidence is not None and self._dependencies_covered(
                dependencies, context_evidence):
            response = {"status": "empty", "deliveryId": delivery_id,
                        "wireVersion": WIRE_VERSION, "policyVersion": POLICY_VERSION,
                        "reasonCodes": ["already_covered"]}
            return self._remember_request(request_key, fingerprint, response)
        assembly_version = _digest(dependencies)
        with self._lock:
            # 这里只登记候选依赖，尚未登记曝光；W4 实际送进模型请求后覆盖同一条。
            self._deliveries[delivery_id] = dependencies
        reasons = [gate_reason, "visibility_unknown"] if context_evidence is None else [gate_reason]
        response = {"status": "candidate", "deliveryId": delivery_id,
                    "wireVersion": WIRE_VERSION, "policyVersion": POLICY_VERSION,
                    "assemblyVersion": assembly_version,
                    "records": records, "dependencies": dependencies,
                    "reasonCodes": reasons + ["w3_admitted_not_assembled"]}
        return self._remember_request(request_key, fingerprint, response)

    def _remember_request(self, request_key, fingerprint, response):
        with self._lock:
            self._requests[request_key] = {
                "fingerprint": fingerprint,
                "response": json.loads(json.dumps(response)),
            }
        return response

    def register_delivery(self, delivery_id, dependencies):
        """供 W4 登记实际组装依赖；W1 先提供可独立验证的窄接口。"""
        if not isinstance(delivery_id, str) or not delivery_id:
            raise ValueError("delivery_id 必须是非空字符串")
        with self._lock:
            self._deliveries[delivery_id] = json.loads(json.dumps(dependencies))

    def _current_descriptor(self, dependency):
        record_id = dependency.get("recordId")
        for idx, text in enumerate(self.index.chunks):
            if _chunk_key(text) != record_id or idx in self.index.retracted:
                continue
            row = {"text": text, "meta": self.index.meta[idx]}
            current = describe_record(row)
            if current["sourceSignature"] != dependency.get("sourceSignature"):
                continue
            ranges = dependency.get("ranges") or []
            if not ranges or any(
                    not isinstance(part, dict)
                    or not isinstance(part.get("start"), int)
                    or not isinstance(part.get("end"), int)
                    or part["start"] < 0
                    or part["end"] > len(text)
                    or part["start"] >= part["end"]
                    or part.get("signature") != _digest(text[part["start"]:part["end"]])
                    for part in ranges):
                continue
            passive_revision = dependency.get("passiveRevision")
            if passive_revision is not None:
                try:
                    item = self.metadata_reader().get(record_id)
                except (OSError, UnicodeError, ValueError, AttributeError):
                    continue
                if not isinstance(item, dict) or item.get("revision") != passive_revision \
                        or item.get("sourceSignature") != source_signature(text):
                    continue
            return current
        return None

    def _valid_evidence(self, evidence):
        valid = []
        for item in evidence or ():
            current = self._current_descriptor(item) if isinstance(item, dict) else None
            if current is None:
                continue
            text = next((text for idx, text in enumerate(self.index.chunks)
                         if idx not in self.index.retracted
                         and _chunk_key(text) == item["recordId"]), None)
            if text is None:
                continue
            ranges = item.get("ranges") or []
            if not ranges or any(
                    int(part.get("start", -1)) < 0
                    or int(part.get("end", -1)) > len(text)
                    or int(part.get("start", -1)) >= int(part.get("end", -1))
                    or part.get("signature") != _digest(text[int(part["start"]):int(part["end"])])
                    for part in ranges):
                continue
            valid.append(item)
        return valid

    def inspect(self, evidence=()):
        """只读核验已登记依赖，并按 recordId／revision／范围判断主动覆盖。"""
        valid_evidence = self._valid_evidence(evidence)
        covered, invalidated = [], []
        with self._lock:
            deliveries = list(self._deliveries.items())
        for delivery_id, dependencies in deliveries:
            if any(self._current_descriptor(dep) is None for dep in dependencies):
                invalidated.append(delivery_id)
                continue
            complete = True
            for dep in dependencies:
                matches = [item for item in valid_evidence
                           if item.get("recordId") == dep.get("recordId")
                           and item.get("revision") == dep.get("revision")
                           and item.get("sourceSignature") == dep.get("sourceSignature")]
                supplied = [part for item in matches for part in item.get("ranges", [])]
                if not _ranges_cover(dep.get("ranges", []), supplied):
                    complete = False
                    break
            if complete:
                covered.append(delivery_id)
        state = {"coveredDeliveryIds": covered,
                 "invalidatedDeliveryIds": invalidated}
        if invalidated:
            state.update({
                "statusNotice": "〔自动浮现状态〕先前资料的来源或版本已失效，请停止使用。",
                "retirementNotice": "〔自动浮现状态〕本会话的自动资料已退役，请勿再把旧片段作为有效依据。",
            })
        return state

    def search_metadata(self, rows):
        """给 latent_search 增加机器可见来源；文本输出和主动加权语义保持不变。"""
        evidence = [describe_record(row, visible_limit=DEFAULT_MAX_ITEM_CHARS) for row in rows]
        state = self.inspect(evidence)
        state["evidence"] = evidence
        state["wireVersion"] = WIRE_VERSION
        return {"passiveRecall": state}


def _selftest():
    from memory_retrieval import MemoryIndex

    index = MemoryIndex()
    index.add("纸箱飞船是我们在搬家堵门时形成的玩笑，认真抱怨时不要接梗。",
              {"source": "w01.md", "heading": "纸箱飞船", "chunk_index": 0})
    index.add("咖啡机的保险丝已经换好。",
              {"source": "w02.md", "heading": "咖啡机", "chunk_index": 0})
    index.build()
    service = PassiveRecallService(index, max_candidates=1)

    before = list(index.weights)
    candidate = service.candidate({"userInput": "纸箱飞船", "turn": {
        "sessionId": "s1", "turnId": "t1", "deliveryId": "d1"}})
    assert candidate["status"] == "candidate" and index.weights == before, \
        "R1：自动候选必须命中且权重零写入"

    # 真并发判据：自动查询先读到旧权重后停住，主动查询在它返回前完成加权；自动
    # 查询结束后主动增量仍必须在。若实现是“拍快照→retrieve→恢复”，这里会被抹掉。
    entered = threading.Event()
    release = threading.Event()
    original_vector_scores = index._vector_scores

    def delayed_vector_scores(query):
        scores = original_vector_scores(query)
        if threading.current_thread().name == "passive-w1-auto":
            entered.set()
            assert release.wait(2), "并发夹具没有按时放行"
        return scores

    index._vector_scores = delayed_vector_scores
    automatic = threading.Thread(target=index.retrieve_candidates,
                                 args=("纸箱飞船",), kwargs={"topN": 1},
                                 name="passive-w1-auto")
    automatic.start()
    assert entered.wait(2), "自动查询没有进入并发夹具"
    concurrent_before = index.weights[0]
    index.retrieve("纸箱飞船", topN=1)
    release.set()
    automatic.join(2)
    index._vector_scores = original_vector_scores
    assert not automatic.is_alive() and index.weights[0] == concurrent_before + index.weight_boost, \
        "R1：自动查询返回时不得用旧快照覆盖并发主动搜索的权重增量"

    dep = candidate["dependencies"][0]
    service.register_delivery("full", [dep])
    full_row = index.retrieve("纸箱飞船", topN=1)[0]
    meta = service.search_metadata([full_row])["passiveRecall"]
    assert index.weights[full_row["id"]] > before[full_row["id"]], \
        "R1：主动搜索该有的权重增量必须保留"
    assert {"d1", "full"}.issubset(meta["coveredDeliveryIds"]), \
        "S1：同 recordId／revision／全范围的主动结果应完整覆盖"

    partial = json.loads(json.dumps(dep))
    partial["ranges"][0]["end"] -= 1
    text = index.chunks[full_row["id"]]
    partial["ranges"][0]["signature"] = _digest(
        text[partial["ranges"][0]["start"]:partial["ranges"][0]["end"]])
    service.register_delivery("partial", [dep])
    assert "partial" not in service.inspect([partial])["coveredDeliveryIds"], \
        "S1：范围缺一字符也不能拿文本相似冒充完整覆盖"

    background = describe_record({"text": index.chunks[1], "meta": index.meta[1]})
    service.register_delivery("needs-background", [dep, background])
    assert "needs-background" not in service.inspect([dep])["coveredDeliveryIds"], \
        "S1：只有主记录、缺必要背景时不能宣称完整覆盖"

    unrelated = index.retrieve("咖啡机", topN=1)
    assert "full" not in service.search_metadata(unrelated)["passiveRecall"][
        "coveredDeliveryIds"], "S2：搜索别的事项不能整体关闭提示"

    index.retracted.add(full_row["id"])
    state = service.inspect([])
    assert {"d1", "full", "partial", "needs-background"}.issubset(
        state["invalidatedDeliveryIds"]), \
        "S3：来源撤回后旧 delivery 必须按权威状态失效"
    print("selftest 通过：W1 R1／S1-S3 只读候选与来源覆盖核心")


if __name__ == "__main__":
    _selftest()
