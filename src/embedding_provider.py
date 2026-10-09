#!/usr/bin/env python3
"""
embedding 提供方可插拔层（任务卡"云端 embedding 作为一等检索路线"）。

**这一层要解决的是"检索层不该知道向量是谁算的"**：`memory_retrieval` 只管拿到
单位化向量，本地模型还是云端 HTTP 服务由这里决定。同 `draft_extraction.py` 的
`llm_call` 那个口子——**我们不内置任何一家的 SDK**，云端走 stdlib 的 urllib，
请求体是各家通用的 OpenAI 兼容 `/v1/embeddings` 形状。换一家只改配置，不改代码。

三条纪律，每条都有对应的断言（见 `_selftest`）：

1. **key 只走环境变量**。配置里给的是"key 存在哪个环境变量里"（变量名），不是
   key 本身；key 不进仓库、不进产出目录、不落 state、不进缓存文件，`describe()`
   与 `id()` 都不吐它。理由跟凭证不入库同一条：产出目录是用户会随手分享的东西。
2. **块向量缓存落盘，查询向量按次算**。建库时把每块的向量算一次、按内容哈希缓存
   起来，之后每次起服务只补新块；每次查询只付一个查询向量的往返。**没有这一条，
   云端档的成本模型完全是另一回事**——每次起 MCP 服务都把全库重算一遍，600 块就是
   600 次云端调用，而这件事在延迟表上根本看不见（表里量的是单查耗时）。
3. **门槛常数跟模型绑定，没量过就说没量过**。`HIT_FLOOR_BY_MODEL` 是一张
   **实测标定表**，不是默认值表。表里没有的模型返回 None＝未标定，调用方必须
   按"未标定"处理（见 memory_retrieval 里 vec_floor 那段），**不许照抄 0.45**——
   那个数跟 bge-small-zh-v1.5 的余弦标度绑死，换模型标度就变了，照抄会让
   "库里没有就说没有"无声失灵。表外的模型由用户自己量，量完填
   `MEMORY_EMBED_HIT_FLOOR`（量法见 `probe_guard.py --floors` 与《快速上手》）。

用法：
  python embedding_provider.py --selftest        # 零依赖自检（不联网、不需要 fastembed）
  python embedding_provider.py --describe        # 打印当前环境解析出来的提供方
"""

from array import array
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

# ---------- 配置：全部走环境变量，key 只给变量名 ----------

ENV_PROVIDER = "MEMORY_EMBED_PROVIDER"      # local / cloud
ENV_MODEL = "MEMORY_EMBED_MODEL"
ENV_ENDPOINT = "MEMORY_EMBED_ENDPOINT"      # 云端：完整 URL，如 https://.../v1/embeddings
ENV_KEY_NAME = "MEMORY_EMBED_API_KEY_ENV"   # **变量名**，不是 key 本身
ENV_QUERY_PREFIX = "MEMORY_EMBED_QUERY_PREFIX"
DEFAULT_KEY_ENV = "MEMORY_EMBED_API_KEY"
# 用户自己标定的命中门槛。标定表只收我们自己量过的模型，表外的模型（云端档的
# bge-m3 也在表外）一律由用户在自己的语料上量，量完从这里填进来；读法见 floor_from_env()。
ENV_HIT_FLOOR = "MEMORY_EMBED_HIT_FLOOR"

DEFAULT_LOCAL_MODEL = "BAAI/bge-small-zh-v1.5"

# bge 的中文模型卡要求 query 侧加指令、passage 侧不加。**这条按模型走，不能一刀切**：
# bge-m3 官方明确说不需要指令前缀，硬加反而是噪声。认不出来的模型不加前缀，并且
# 在 describe() 里说出来——让用户知道我们没替他猜。
BGE_ZH_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

# **实测标定表，不是默认值表**：模型 → 我们自己在真实语料上量过的命中门槛（余弦）。
# 表里没有 = 未标定，get_hit_floor 返回 None，绝不退回别的模型的数字。
#   bge-small-zh-v1.5：0.45，2026.08.01 在 602 块真实语料上复核过（依据、局限和
#     "它其实管不住 BM25 那一路"这件事，写在 memory_retrieval.EMBED_HIT_FLOOR 那段）。
# 只收我们自己复现过的数。别人量的数再可信，也是别人的语料、别人的问法；
# 用户要用表外的模型，就自己量、自己填 MEMORY_EMBED_HIT_FLOOR。
HIT_FLOOR_BY_MODEL = {
    "BAAI/bge-small-zh-v1.5": 0.45,
}

# 自检夹具：一个**永远不会进标定表**的模型名，专门用来守"未标定绝不退回 0.45"
# 这个最坏方向。别拿真实模型名当这个夹具：真实模型哪天进了标定表，用它的断言
# 就会成片假红。名字取成明显虚构的，免得有人以为它能跑。
UNCALIBRATED_FIXTURE_MODEL = "example-org/uncalibrated-test-embed"

# 云端批量大小：一次请求塞多少块。经验值——足够摊薄往返开销，又不至于撞上各家的
# 请求体上限。建库时才会用到批量，查询永远是 1 条。
CLOUD_BATCH = 32
CLOUD_TIMEOUT = 60

# 每条文本发给云端时最多带多少字（只限请求体里那一份，块正文与缓存键都不动）。
# 服务商对单条输入有长度上限，超了的常见反应是整批 400，建库就断在半路；而块是按
# 标题切的，偶尔会有一段几千字的长记录。取 2000 字：与线上 Voyage 接入
# （专属前端 voyage_provider 的 VOYAGE_TEXT_LIMIT）同一个数，两条路线给同一块算的
# 是同一段文字；块的标题和主题都在开头，截掉的尾巴对"这块讲什么"影响最小。
# 按字数截、不按 token 截：单条上限只有 512 token 的模型仍可能超，到时把它调小。
CLOUD_MAX_CHARS = 2000


def normalize(vec):
    """单位化（纯 python，不依赖 numpy——云端档可能连 numpy 都没装）。"""
    n = math.sqrt(sum(x * x for x in vec))
    return [x / n for x in vec] if n else list(vec)


def query_prefix_for(model, override=None):
    """该不该给 query 加指令前缀。override 是环境变量里的显式指定（空串＝明确不加）。"""
    if override is not None:
        return override
    m = (model or "").lower()
    if "bge" in m and ("zh" in m or "chinese" in m):
        return BGE_ZH_QUERY_PREFIX
    return ""      # 认不出来就不加，并在 describe() 里说明


def floor_from_env(env=None):
    """读 `MEMORY_EMBED_HIT_FLOOR` → (门槛, 没生效的原因)。

    - 没设或是空串：(None, None)；
    - 0 到 1 之间（不含两端）的有限数：(该数, None)；
    - 其余（读不成数、≤0、≥1、nan）：(None, 原因)。≤0 等于向量路什么都放行，
      ≥1 等于什么都不放行，都不是"一道门槛"。

    设歪了**不当 0 用、也不悄悄吞掉**：原因会写进 describe()，启动信息里就能看到
    自己那一行没生效，而不是以为开了门槛、其实一直走的是"未标定"那条路。"""
    raw = str((os.environ if env is None else env).get(ENV_HIT_FLOOR) or "").strip()
    if not raw:
        return None, None
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if 0.0 < value < 1.0:      # nan 的比较恒为假，一并拦在这里
        return value, None
    return None, f"{ENV_HIT_FLOOR}={raw} 不是 0 到 1 之间的数，没生效"


def get_hit_floor(model, env=None):
    """该模型的命中门槛：用户填的 `MEMORY_EMBED_HIT_FLOOR` 优先，其次标定表；
    两边都没有返回 None（未标定），**不给替代数字**。"""
    value, _ = floor_from_env(env)
    return value if value is not None else HIT_FLOOR_BY_MODEL.get(model)


def _floor_note(floor, env=None):
    """describe() 里那句门槛说明。三种来源各说各的，不混：标定表里的数是我们量的，
    用户填的数是用户自己的标定，两边都没有就明说未标定，并指出去哪儿填。"""
    user_floor, problem = floor_from_env(env)
    if floor is None:
        why = f"（{problem}）" if problem else ""
        return (f"命中门槛**未标定**{why}：向量路只参与排序、不单独放行；"
                f"在自己的语料上量好后填 {ENV_HIT_FLOOR}")
    if user_floor is not None:
        return f"命中门槛 {floor}（{ENV_HIT_FLOOR} 填的，是你自己的标定）"
    return f"命中门槛 {floor}（已标定）"


class EmbeddingProvider:
    """提供方基类。子类只需实现 `_embed_raw(texts)` → 未归一化的向量列表。

    计数器 `calls` / `texts_embedded` 不是调试残留，是**断言用的量具**：
    "建库算一次、查询不重算"这条只有数得出调用次数才测得了（见 memory_retrieval
    的缓存断言）。"""

    kind = "base"

    def __init__(self, model, query_prefix=None):
        self.model = model
        self.query_prefix = query_prefix_for(model, query_prefix)
        self.calls = 0              # 真正发出去的批次数
        self.texts_embedded = 0     # 真正算过向量的文本条数

    def _embed_raw(self, texts):
        raise NotImplementedError

    def embed(self, texts, is_query=False):
        """→ 单位化向量列表（list[list[float]]）。空输入不发请求。"""
        texts = list(texts)
        if not texts:
            return []
        if is_query and self.query_prefix:
            texts = [self.query_prefix + t for t in texts]
        vecs = self._embed_raw(texts)
        self.texts_embedded += len(texts)
        if len(vecs) != len(texts):
            raise RuntimeError(f"提供方返回 {len(vecs)} 个向量，与输入 {len(texts)} 条不符")
        return [normalize(v) for v in vecs]

    @property
    def id(self):
        """缓存与门槛都按它区分。**不含 key**——它会被写进缓存文件。"""
        return f"{self.kind}:{self.model}"

    def hit_floor(self):
        return get_hit_floor(self.model, getattr(self, "_env", None))

    def describe(self):
        raise NotImplementedError


class LocalProvider(EmbeddingProvider):
    """本地档：fastembed 跑 ONNX，CPU，**语料不出本机**。"""

    kind = "local"

    def __init__(self, model=DEFAULT_LOCAL_MODEL, query_prefix=None):
        super().__init__(model, query_prefix)
        self._embedder = None

    def _get(self):
        if self._embedder is None:
            from fastembed import TextEmbedding     # 可选件，用到才 import
            self._embedder = TextEmbedding(self.model)
        return self._embedder

    def _embed_raw(self, texts):
        self.calls += 1
        return [list(v) for v in self._get().embed(texts)]

    def describe(self):
        prefix = "加 bge 中文指令前缀" if self.query_prefix else "不加 query 前缀"
        return (f"本地模型 {self.model}（fastembed / 本地 CPU）；"
                f"语料不出本机；{_floor_note(self.hit_floor())}；{prefix}")


class HTTPCloudProvider(EmbeddingProvider):
    """云端档：HTTP 调第三方 embedding 服务。**语料会发到这家服务商**。

    请求体走各家通用的 OpenAI 兼容形状（`{"model":…, "input":[…]}` → `data[].embedding`），
    stdlib urllib 发出去——**不装任何一家的 SDK**，换一家只改 endpoint 与 model。

    key：构造时只收**环境变量名**，值在发请求时才从环境里读，且只存在内存里。
    `transport` 是给自检用的注入口（收 payload 字典、返回响应字典），有它就不联网——
    自检不能依赖网络和真 key，但"我们发的请求长什么样、key 有没有漏进不该去的地方"
    必须被断言走过。"""

    kind = "cloud"

    def __init__(self, endpoint, model, key_env=DEFAULT_KEY_ENV,
                 query_prefix=None, batch=CLOUD_BATCH, transport=None, env=None):
        super().__init__(model, query_prefix)
        if not endpoint:
            raise ValueError(f"云端档必须给 {ENV_ENDPOINT}（完整 URL，如 "
                             f"https://api.example.com/v1/embeddings）")
        self.endpoint = endpoint
        self.key_env = key_env or DEFAULT_KEY_ENV
        self.batch = batch
        self._transport = transport
        self._env = env if env is not None else os.environ

    @property
    def id(self):
        # 同一个模型名在不同服务商那里未必是同一个权重，缓存要按 host 分开
        from urllib.parse import urlparse
        host = urlparse(self.endpoint).netloc or "?"
        return f"{self.kind}:{host}:{self.model}"

    def _key(self):
        key = (self._env.get(self.key_env) or "").strip()
        if not key:
            raise RuntimeError(
                f"云端档要 API key，但环境变量 {self.key_env} 是空的。"
                f"key 只从环境变量读——不写进配置文件、不进产出目录，"
                f"我们也不替你保存。")
        return key

    def _post(self, payload):
        if self._transport is not None:      # 自检注入口，不联网
            return self._transport(payload)
        import urllib.request
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + self._key()},
            method="POST")
        with urllib.request.urlopen(req, timeout=CLOUD_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _embed_raw(self, texts):
        out = []
        for i in range(0, len(texts), self.batch):
            chunk = [t[:CLOUD_MAX_CHARS] for t in texts[i:i + self.batch]]   # 见 CLOUD_MAX_CHARS
            self.calls += 1
            data = self._post({"model": self.model, "input": chunk})
            try:
                rows = sorted(data["data"], key=lambda r: r.get("index", 0))
                out.extend(list(r["embedding"]) for r in rows)
            except (KeyError, TypeError) as e:
                # 不静默降级：形状不对就说清楚，别让半截响应变成一批零向量
                raise RuntimeError(
                    f"{self.endpoint} 的响应不是 OpenAI 兼容形状（缺 data[].embedding）：{e}") from e
        return out

    def describe(self):
        from urllib.parse import urlparse
        host = urlparse(self.endpoint).netloc or self.endpoint
        floor = self.hit_floor()
        cal = _floor_note(floor, self._env)
        prefix = "加 bge 中文指令前缀" if self.query_prefix else "不加 query 前缀"
        return (f"云端服务 {host} 的 {self.model}；"
                f"**查询和被检索的内容都会发到这家服务商**；"
                f"key 从环境变量 {self.key_env} 读（我们不存、不落盘）；{cal}；{prefix}")


def resolve_provider(spec=None, env=None, transport=None):
    """按 spec / 环境变量解析出提供方。

    spec 取值：`None`（看环境变量，默认 local）、`local`、`local:<模型名>`、`cloud`。
    云端档的 endpoint / model / key 变量名一律走环境变量——**命令行不收 key**，
    因为命令行会进 shell 历史、也会被写进 MCP 配置文件里跟着产出目录走。"""
    env = os.environ if env is None else env
    spec = spec or env.get(ENV_PROVIDER) or "local"
    kind, _, inline_model = spec.partition(":")
    kind = kind.strip().lower()
    model = inline_model.strip() or env.get(ENV_MODEL) or None
    prefix = env.get(ENV_QUERY_PREFIX)       # 显式空串＝明确不加前缀
    if kind == "local":
        return LocalProvider(model or DEFAULT_LOCAL_MODEL, query_prefix=prefix)
    if kind == "cloud":
        if not model:
            raise ValueError(f"云端档必须给模型名（{ENV_MODEL} 或 --embed-provider cloud:<模型>）")
        return HTTPCloudProvider(env.get(ENV_ENDPOINT), model,
                                 key_env=env.get(ENV_KEY_NAME) or DEFAULT_KEY_ENV,
                                 query_prefix=prefix, transport=transport, env=env)
    raise ValueError(f"未知 embedding 提供方 {spec!r}，可选：local / local:<模型> / cloud")


# ---------- 块向量缓存 ----------

def text_key(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class VectorCache:
    """块向量的落盘缓存：内容哈希 → 向量。**只缓存块，不缓存查询**。

    为什么按内容哈希而不是块下标：语料会长、会被重切，下标一变就全错位，而
    内容一样的块换到哪都是同一个向量。哈希碰撞的代价这里也只是一次错向量，
    sha1 够用。

    provider_id 不一致时整份作废——**不同模型的向量不能混着用**，混了不会报错，
    只会让余弦分数变成噪声（同"门槛不许照抄"是同一个坑的两面）。

    存 6 位小数：单位化向量的有效精度本来就有限，文件小一半多。
    内存里每条是 float32 的 `array('f')`（同 passive_facts.FactIndex）：1024 维一条 4 KiB，
    Python float list 要 32 KiB。读盘逐条解析、逐条转，不一次物化整份 list；写盘还原成
    同样的 6 位小数，文件逐字节不变（float32 误差 ≤ 6e-8，舍回 6 位必回到原值）。
    缓存文件是**用户产出目录里的中间物，不进仓库**（见 .gitignore）。"""

    def __init__(self, path, provider_id):
        self.path = Path(path) if path else None
        self.provider_id = provider_id
        self.vectors = {}
        self.dirty = False
        self.loaded_from_disk = False
        self._load()

    def _load(self):
        if not self.path or not self.path.exists():
            return
        try:
            provider, vectors = _parse_cache(self.path.read_text(encoding="utf-8"))
        except (ValueError, TypeError, OSError):
            return          # 缓存坏了就当没有，重算即可，不该让检索起不来
        if provider != self.provider_id:
            return          # 换了模型/服务商：整份作废
        self.vectors = vectors
        self.loaded_from_disk = True

    def get(self, text):
        return self.vectors.get(text_key(text))

    def put(self, text, vec):
        self.vectors[text_key(text)] = array("f", (round(x, 6) for x in vec))
        self.dirty = True

    def save(self):
        """逐条写进临时文件再原子替换（外部同步卡第七节）。不先拼出整份 JSON 字符串：3 万块、
        1024 维时那份字符串约 320MB，增量写入每存一次峰值就多这么多。写出的字节与
        `json.dumps({"provider": …, "vectors": …}, ensure_ascii=False)` 逐字节相同（自检第 13 项）；
        写到一半崩溃也不会留下半截文件（坏缓存会让下次起服务整库重算向量）。"""
        if not self.path or not self.dirty:
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            f.write('{"provider": ' + json.dumps(self.provider_id, ensure_ascii=False)
                    + ', "vectors": {')
            for n, (key, vec) in enumerate(self.vectors.items()):
                f.write((", " if n else "") + json.dumps(key, ensure_ascii=False) + ": "
                        + json.dumps([round(x, 6) for x in vec]))
            f.write("}}")
        os.replace(tmp, self.path)
        self.dirty = False
        return True


def _parse_cache(text):
    """缓存 JSON → (provider, {哈希: array('f')})。vectors 段逐条 raw_decode、逐条转 float32，
    同一时刻只有一条向量是 Python float list。格式不对抛 ValueError／TypeError。"""
    dec, ws = json.JSONDecoder(), json.decoder.WHITESPACE.match

    def skip(i, ch=None):
        i = ws(text, i).end()
        if ch is not None:
            if text[i:i + 1] != ch:
                raise ValueError(f"缓存第 {i} 个字符应为 {ch!r}")
            i = ws(text, i + 1).end()
        return i

    top, vectors = {}, {}
    i = skip(0, "{")
    while text[i:i + 1] != "}":
        key, i = dec.raw_decode(text, i)
        i = skip(i, ":")
        if key == "vectors" and text[i:i + 1] == "{":
            i = skip(i, "{")
            while text[i:i + 1] != "}":
                k, i = dec.raw_decode(text, i)
                v, i = dec.raw_decode(text, skip(i, ":"))
                vectors[k] = array("f", v)
                i = skip(i)
                if text[i:i + 1] == ",":
                    i = skip(i, ",")
            i = skip(i, "}")
        else:
            top[key], i = dec.raw_decode(text, i)
            i = skip(i)
        if text[i:i + 1] == ",":
            i = skip(i, ",")
    return top.get("provider"), vectors


def embed_with_cache(provider, texts, cache=None):
    """块向量：缓存里有的直接取，没有的才算，算完写回。返回单位化的 float32 `array('f')` 列表。

    这就是"建库时算一次、之后只补新块"的全部实现——**云端档的成本模型全靠它**。"""
    if cache is None:
        return [array("f", v) for v in provider.embed(texts)]
    # 命中的直接用缓存里那份：建完库缓存只留现有块的条目，与 _cvecs 共用同一份数组，不留副本
    out = [cache.get(t) for t in texts]
    todo = [i for i, v in enumerate(out) if v is None]
    if todo:
        fresh = provider.embed([texts[i] for i in todo])
        for i, v in zip(todo, fresh):
            cache.put(texts[i], v)
            # 取缓存里那份（舍到 6 位的 float32），重启后从缓存读回的是同一份，增量与全量才逐位一致
            out[i] = cache.get(texts[i])
        cache.save()
    return out


# ---------- 自检（不联网、不需要 fastembed） ----------

def _fake_transport(dim=8):
    """假的云端服务：按文本内容出一个确定性的向量。记下收到过的请求供断言用。"""
    seen = []

    def transport(payload):
        seen.append(payload)
        data = []
        for i, t in enumerate(payload["input"]):
            h = hashlib.sha1(t.encode("utf-8")).digest()
            data.append({"index": i, "embedding": [h[j % len(h)] / 255.0 for j in range(dim)]})
        return {"data": data}
    transport.seen = seen
    return transport


def _selftest():
    import tempfile

    # 1.【key 只走环境变量，且不许漏进任何产出】——这条是纪律不是优化，所以断言
    #    要覆盖所有会被别人看到的出口：id / describe / 缓存文件。
    #    ⚠ 这里的模型名兼作第 5 条的**未标定夹具**，所以用 UNCALIBRATED_FIXTURE_MODEL
    #    而不是某个真实模型名——真实模型名随时会被标定，那天这条就成假红了。
    env = {ENV_ENDPOINT: "https://api.example.com/v1/embeddings",
           ENV_MODEL: UNCALIBRATED_FIXTURE_MODEL, "MEMORY_EMBED_API_KEY": "sk-绝密-不该出现"}
    tr = _fake_transport()
    p = resolve_provider("cloud", env=env, transport=tr)
    assert isinstance(p, HTTPCloudProvider) and p.model == UNCALIBRATED_FIXTURE_MODEL
    assert "sk-绝密" not in p.id and "sk-绝密" not in p.describe(), "key 漏进了 id/describe"
    assert p.key_env == DEFAULT_KEY_ENV

    # 2.【缺 key 要明确报错，不许静默跑成一堆零向量】
    p_nokey = HTTPCloudProvider("https://api.example.com/v1/embeddings", "m", env={})
    try:
        p_nokey._key()
        raise AssertionError("缺 key 居然没报错")
    except RuntimeError as e:
        assert "环境变量" in str(e)

    # 3.【云端返回的向量是单位化的，且 batch 分批不打乱顺序】
    texts = [f"第{i}块内容各不相同" for i in range(70)]      # 70 > CLOUD_BATCH*2
    vecs = p.embed(texts)
    assert len(vecs) == 70
    assert all(abs(sum(x * x for x in v) - 1.0) < 1e-9 for v in vecs), "向量没单位化"
    assert p.calls == 3, f"70 条应分 3 批发，实际 {p.calls} 批"
    v_single = p.embed([texts[5]])[0]
    assert max(abs(a - b) for a, b in zip(v_single, vecs[5])) < 1e-9, "分批把顺序弄乱了"

    # 4.【query 前缀按模型走】：认不出来的模型不加（bge-m3 官方也说不需要），bge-zh 要加
    assert resolve_provider("cloud", env=env, transport=tr).query_prefix == ""
    assert LocalProvider(DEFAULT_LOCAL_MODEL).query_prefix == BGE_ZH_QUERY_PREFIX
    assert query_prefix_for("BAAI/bge-m3") == ""
    p2 = HTTPCloudProvider("https://x/v1/embeddings", "BAAI/bge-large-zh-v1.5",
                           transport=_fake_transport(), env={"MEMORY_EMBED_API_KEY": "k"})
    p2.embed(["问一句"], is_query=True)
    assert p2._transport.seen[-1]["input"][0].startswith(BGE_ZH_QUERY_PREFIX)

    # 5.【门槛表是标定表不是默认值表】：没量过的模型必须是 None。
    #    这条是本任务卡的硬要求——照抄 0.45 会让"库里没有就说没有"无声失灵。
    assert get_hit_floor(DEFAULT_LOCAL_MODEL) == 0.45
    assert get_hit_floor(UNCALIBRATED_FIXTURE_MODEL) is None, "未标定的模型不许有门槛数字"
    assert p.hit_floor() is None and "未标定" in p.describe()

    # 6.【缓存：建库算一次，第二次起服务一条都不重算】
    with tempfile.TemporaryDirectory() as td:
        cpath = Path(td) / ".embed_cache.json"
        tr1 = _fake_transport()
        pa = HTTPCloudProvider("https://api.example.com/v1/embeddings", "BAAI/bge-m3",
                               transport=tr1, env=env)
        c1 = VectorCache(cpath, pa.id)
        v1 = embed_with_cache(pa, texts, c1)
        assert pa.texts_embedded == 70 and cpath.exists()

        tr2 = _fake_transport()
        pb = HTTPCloudProvider("https://api.example.com/v1/embeddings", "BAAI/bge-m3",
                               transport=tr2, env=env)
        c2 = VectorCache(cpath, pb.id)
        assert c2.loaded_from_disk
        v2 = embed_with_cache(pb, texts, c2)
        assert pb.texts_embedded == 0 and pb.calls == 0, "第二次建库还在重算块向量"
        assert max(abs(a - b) for x, y in zip(v1, v2) for a, b in zip(x, y)) < 1e-5

        # 只加一块新语料：只补这一块，不重算整库
        v3 = embed_with_cache(pb, texts + ["新写进来的一段记忆"], c2)
        assert pb.texts_embedded == 1, f"补一块新语料却算了 {pb.texts_embedded} 条"
        assert len(v3) == 71

        # key 不许出现在缓存文件里（缓存跟着产出目录走，用户会随手分享）
        assert "sk-绝密" not in cpath.read_text(encoding="utf-8")

        # 换了模型/服务商 → 整份作废，不许混用两套标度的向量
        c3 = VectorCache(cpath, "cloud:other.example.com:BAAI/bge-m3")
        assert c3.vectors == {} and not c3.loaded_from_disk, "换服务商没作废旧缓存"

    # 7.【坏掉的缓存文件不许让检索起不来】
    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / ".embed_cache.json"
        bad.write_text("{不是合法 json", encoding="utf-8")
        assert VectorCache(bad, "local:x").vectors == {}

    # 7b.【内存里存 float32、磁盘格式不动】现行格式（json.dumps 默认分隔符、6 位小数）写的缓存
    #     读进来每条是 array('f')、全部命中不重算；原样写出逐字节相同。
    with tempfile.TemporaryDirectory() as td:
        old_file, new_file = Path(td) / "old.json", Path(td) / "new.json"
        pv = HTTPCloudProvider("https://api.example.com/v1/embeddings", "BAAI/bge-m3",
                               transport=_fake_transport(dim=1024), env=env)
        raw = {text_key(t): [round(x, 6) for x in v]
               for t, v in zip(texts, pv.embed(texts))}
        raw[text_key("边角值")] = [0.0, -0.0, 1e-06, -1e-06, 0.999999, -1.0, 0.5, 0.123457]
        old_file.write_text(json.dumps({"provider": pv.id, "vectors": raw}, ensure_ascii=False),
                            encoding="utf-8")
        c = VectorCache(old_file, pv.id)
        assert c.loaded_from_disk and len(c.vectors) == len(raw)
        assert all(type(v) is array and v.typecode == "f" for v in c.vectors.values())
        pv.texts_embedded = 0
        got = embed_with_cache(pv, texts, c)
        assert pv.texts_embedded == 0 and all(type(v) is array for v in got)
        assert got[3] is c.vectors[text_key(texts[3])], "命中的向量另拷了一份"
        c.path, c.dirty = new_file, True
        c.save()
        assert new_file.read_bytes() == old_file.read_bytes(), "读入再写出改了缓存文件"
        # 写盘时 put 进来的新向量也还原成 6 位小数，和旧写法同一字节
        c.put("新块", raw[text_key(texts[0])])
        c.save()
        assert json.loads(new_file.read_text(encoding="utf-8"))["vectors"][text_key("新块")] \
            == raw[text_key(texts[0])]
        # 缓存格式不对（vectors 里混进非数字）当坏缓存处理，不让检索起不来
        bad_file = Path(td) / "bad.json"
        bad_file.write_text('{"provider": "%s", "vectors": {"k": ["x"]}}' % pv.id, encoding="utf-8")
        assert VectorCache(bad_file, pv.id).vectors == {}

    # 8.【未知提供方名要报错，别默默跑成本地档】——静默降级在这里等于"用户以为
    #    自己选了云端，其实一直在本地跑"，或者反过来，两个方向都不能接受
    for bad_spec in ("openai", "", "本地"):
        try:
            resolve_provider(bad_spec or "x", env={})
            raise AssertionError(f"{bad_spec!r} 居然被接受了")
        except ValueError:
            pass

    # 9.【请求体里的超长块按 CLOUD_MAX_CHARS 截，手上的块不动】
    #    两头都要看：只看请求体，截断写成原地改也是绿的——那会把用户的记忆截短。
    sent = []
    pc = HTTPCloudProvider("https://api.example.com/v1/embeddings", "m",
                           transport=lambda pl: sent.append(list(pl["input"])) or
                           {"data": [{"index": i, "embedding": [1.0, 0.0]}
                                     for i in range(len(pl["input"]))]},
                           env={"MEMORY_EMBED_API_KEY": "k"})
    batch = ["长" * (CLOUD_MAX_CHARS + 1), "短句", "满" * CLOUD_MAX_CHARS]
    before = list(batch)
    pc._embed_raw(batch)       # 直接调这一层：embed() 开头那份 list() 拷贝会替原地修改打掩护
    assert [len(t) for t in sent[0]] == [CLOUD_MAX_CHARS, 2, CLOUD_MAX_CHARS],         f"请求体没按 CLOUD_MAX_CHARS 截：{[len(t) for t in sent[0]]}"
    assert batch == before, "截断改到了调用方手上的块正文"

    # 10.【MEMORY_EMBED_HIT_FLOOR：读得成就生效，读不成就当没设，并且说出来】
    assert floor_from_env({}) == (None, None)
    assert floor_from_env({ENV_HIT_FLOOR: " 0.58 "}) == (0.58, None)
    assert get_hit_floor(UNCALIBRATED_FIXTURE_MODEL, env={ENV_HIT_FLOOR: "0.58"}) == 0.58
    assert get_hit_floor(DEFAULT_LOCAL_MODEL, env={ENV_HIT_FLOOR: "0.5"}) == 0.5,         "用户自己量的数在标定表里的模型上同样优先"
    for bad in ("abc", "0", "-0.2", "1", "1.5", "nan", "inf"):
        value, problem = floor_from_env({ENV_HIT_FLOOR: bad})
        assert value is None and problem and bad in problem,             f"{bad!r} 不是门槛，必须当没设并说出原因（得到 {value!r}, {problem!r}）"
        assert get_hit_floor(UNCALIBRATED_FIXTURE_MODEL, env={ENV_HIT_FLOOR: bad}) is None
    assert floor_from_env({ENV_HIT_FLOOR: "  "}) == (None, None), "空白当没设，不算设歪"
    #     describe() 三种来源各说各的：用户填的、标定表的、设歪了的
    env_ok = {"MEMORY_EMBED_API_KEY": "k", ENV_HIT_FLOOR: "0.58"}
    env_bad = {"MEMORY_EMBED_API_KEY": "k", ENV_HIT_FLOOR: "0,58"}
    d_ok = HTTPCloudProvider("https://x/v1/embeddings", UNCALIBRATED_FIXTURE_MODEL,
                             transport=_fake_transport(), env=env_ok).describe()
    d_bad = HTTPCloudProvider("https://x/v1/embeddings", UNCALIBRATED_FIXTURE_MODEL,
                              transport=_fake_transport(), env=env_bad).describe()
    assert "0.58" in d_ok and "你自己的标定" in d_ok and "已标定）" not in d_ok, d_ok
    assert "未标定" in d_bad and "0,58" in d_bad and ENV_HIT_FLOOR in d_bad,         f"设歪了要在启动信息里看得见：{d_bad}"
    assert "（已标定）" in _floor_note(0.45, {})

    # 11.【bge-m3 没有预设门槛】标定表只收我们自己复现过的数；bge-m3 要用户自己量。
    #     这条守的是"表外模型别偷偷带一个数进来"——哪天往表里加一行没量过的数，这里红。
    assert get_hit_floor("BAAI/bge-m3", env={}) is None
    m3 = HTTPCloudProvider("https://api.example.com/v1/embeddings", "BAAI/bge-m3",
                           transport=_fake_transport(), env={"MEMORY_EMBED_API_KEY": "k"})
    assert m3.hit_floor() is None and ENV_HIT_FLOOR in m3.describe(),         "未标定时 describe() 要告诉用户门槛去哪儿填"
    assert set(HIT_FLOOR_BY_MODEL) == {DEFAULT_LOCAL_MODEL},         "标定表里多了一个模型：先确认是我们自己在真实语料上量过的，再改这条断言"

    # 13.【刚算出来的块向量＝缓存里那份】增量写入当场算的向量，要和重启后从缓存读回的逐位
    #     相同（外部同步卡）。变异：embed_with_cache 里 out[i] 换回 array("f", v) → 这条红
    with tempfile.TemporaryDirectory() as td13:
        p13 = HTTPCloudProvider("https://api.example.com/v1/embeddings", "m",
                                transport=_fake_transport(), env={"MEMORY_EMBED_API_KEY": "k"})
        cache13 = Path(td13) / "c.json"
        fresh13 = embed_with_cache(p13, ["甲块", "乙块"], VectorCache(cache13, p13.id))
        again13 = embed_with_cache(p13, ["甲块", "乙块"], VectorCache(cache13, p13.id))
        assert p13.texts_embedded == 2, "第二次该全走缓存"
        assert [v.tobytes() for v in fresh13] == [v.tobytes() for v in again13], \
            "当场算的向量与缓存读回的不逐位相同：增量写入与重启会差最后几位"
        #     逐条写出的缓存文件与整份 json.dumps 逐字节相同（格式不动，老版本照读）
        c13 = VectorCache(cache13, p13.id)
        ref = json.dumps({"provider": c13.provider_id, "vectors": c13.vectors}, ensure_ascii=False,
                         default=lambda a: [round(x, 6) for x in a])
        assert cache13.read_text(encoding="utf-8") == ref and not cache13.with_name("c.json.tmp").exists(), \
            "逐条写出的缓存文件与 json.dumps 不逐字节相同"

    print("selftest ok（13 项：key 不外泄 / 缺 key 报错 / 分批不乱序 / 前缀按模型 / "
          "未标定即 None / 缓存只算一次 / 坏缓存不致命 / 缓存内存存 float32、磁盘逐字节不变 / 未知档报错 / "
          "请求体里的超长块截断、手上的块不动 / MEMORY_EMBED_HIT_FLOOR 读不成就当没设并说出来 / "
          "bge-m3 没有预设门槛 / 当场算的向量＝缓存读回的那份）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--describe", action="store_true", help="打印当前环境解析出的提供方")
    ap.add_argument("--provider", help="local / local:<模型> / cloud")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    elif args.describe:
        pr = resolve_provider(args.provider)
        print(f"{pr.id}\n{pr.describe()}")
    else:
        ap.print_help()
