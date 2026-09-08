"""自动浮现 W1：共用只读候选、来源证据与依赖覆盖核心。

本层只复用现有检索并给候选建立可核验身份，不做短句预筛、意图冲突、事件线选择
或提示组装；这些分别属于 W3／W4。候选不是曝光，返回 ``candidate`` 时宿主不得把
原始结构直接塞给聊天模型。
"""

from __future__ import annotations

import hashlib
import json
import threading

from memory_retrieval import _chunk_key
from session_recall import DEFAULT_MAX_ITEM_CHARS


WIRE_VERSION = "passive-recall-w1-v1"
TOOL_NAME = "latent_passive_recall"


class PassiveRecallRequestError(ValueError):
    """宿主入口参数不满足 W0 冻结的请求契约。"""


def _digest(value):
    wire = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(wire).hexdigest()


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
    """复用常驻 MemoryIndex 的 W1 服务；交付账本留给 W4 扩展。"""

    def __init__(self, index, max_candidates=5):
        self.index = index
        self.max_candidates = int(max_candidates)
        self._deliveries = {}
        self._lock = threading.RLock()

    def set_index(self, index):
        """常驻宿主重载语料后换同一服务的只读索引引用。"""
        with self._lock:
            self.index = index

    def candidate(self, request):
        if not isinstance(request, dict):
            raise PassiveRecallRequestError("请求必须是对象")
        user_input = request.get("userInput")
        turn = request.get("turn")
        if not isinstance(user_input, str) or not user_input.strip():
            raise PassiveRecallRequestError("userInput 必须是非空字符串")
        if not isinstance(turn, dict) or not all(
                isinstance(turn.get(key), str) and turn.get(key)
                for key in ("sessionId", "turnId", "deliveryId")):
            raise PassiveRecallRequestError("turn 必须提供 sessionId／turnId／deliveryId")
        rows = self.index.retrieve_candidates(user_input, topN=self.max_candidates)
        delivery_id = turn["deliveryId"]
        if not rows:
            return {"status": "empty", "deliveryId": delivery_id,
                    "wireVersion": WIRE_VERSION, "reasonCodes": ["no_reliable_candidate"]}
        records = [describe_record(row) for row in rows]
        dependencies = [{key: record[key] for key in
                         ("recordId", "revision", "state", "sourceSignature", "ranges")}
                        for record in records]
        assembly_version = _digest(dependencies)
        with self._lock:
            # W1 只登记候选依赖，尚未登记曝光；W4 选出实际事件线后可用同一方法覆盖。
            self._deliveries[delivery_id] = dependencies
        return {"status": "candidate", "deliveryId": delivery_id,
                "wireVersion": WIRE_VERSION, "assemblyVersion": assembly_version,
                "records": records, "dependencies": dependencies,
                "reasonCodes": ["w1_candidates_only"]}

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
            if current["revision"] == dependency.get("revision") and \
                    current["sourceSignature"] == dependency.get("sourceSignature"):
                return current
        return None

    def _valid_evidence(self, evidence):
        valid = []
        for item in evidence or ():
            current = self._current_descriptor(item) if isinstance(item, dict) else None
            if current is None:
                continue
            text = next((text for idx, text in enumerate(self.index.chunks)
                         if idx not in self.index.retracted and _chunk_key(text) == item["recordId"]
                         and describe_record({"text": text, "meta": self.index.meta[idx]})[
                             "revision"] == item.get("revision")), None)
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
