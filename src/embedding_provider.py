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
   "库里没有就说没有"无声失灵。

用法：
  python embedding_provider.py --selftest        # 零依赖自检（不联网、不需要 fastembed）
  python embedding_provider.py --describe        # 打印当前环境解析出来的提供方
"""

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
# 门槛覆盖口（**采纳自外部 PR #1**，`lu7899112-source`，2026.08.01）：对方指出，
# 换了后端/模型的人**只能改源码**才能填自己量出来的门槛。这条是对的——我们的
# 标定表纪律（没量过就是 None、不给替代数字）拦的是"照抄一个数"，不该顺带把
# **真的量过的人**也拦在外面。所以口子开，纪律照旧写在门口：见 hit_floor_override()。
ENV_HIT_FLOOR = "MEMORY_EMBED_HIT_FLOOR"

DEFAULT_LOCAL_MODEL = "BAAI/bge-small-zh-v1.5"

# bge 的中文模型卡要求 query 侧加指令、passage 侧不加。**这条按模型走，不能一刀切**：
# bge-m3 官方明确说不需要指令前缀，硬加反而是噪声。认不出来的模型不加前缀，并且
# 在 describe() 里说出来——让用户知道我们没替他猜。
BGE_ZH_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

# **实测标定表，不是默认值表**：模型 → 该模型上量过的可靠命中门槛（余弦）。
# 表里没有 = 未标定，get_hit_floor 返回 None，绝不退回别的模型的数字。
#   bge-small-zh-v1.5：0.45，2026.08.01 在 602 块真实语料上复核过（依据、局限和
#     "它其实管不住 BM25 那一路"这件事，写在 memory_retrieval.EMBED_HIT_FLOOR 那段）。
#   bge-m3（云端档常见选择）：0.60，**外部标定，我们未复现**——
#     出处：外部贡献者「星迟 & Ember」，公开仓库 PR #4，1144 块真实中文语料。
#     方法：走**真实 retrieve() 路径**（不是脚本拼 rank_lists）把 floor 从 0.30
#       扫到 0.70，**同时量两条曲线**——gold=1 的 20 题命中，和"编造题下向量路
#       独立放行的块数"（后者就是"floor 太松时向量路给编造递多少材料"）。
#     他们自报的两条局限照原话收：**单一语料形态**（中文原子记忆，块中位较短），
#       别的语料形态请复测再信；**absent 那一侧在他们语料上是被词面闸先漏的
#       （0/9，与 floor 无关）**——所以这次标定管住的是向量路，词面路仍敞着
#       （与 bge-small-zh 那条"管不住 BM25 一路"的已知局限同构）。
#     **选点规则（这条比数字值钱）**：两个候选点的测量值在噪声内等价时，
#       取**两个方向都退化得体**的那个。0.62 与 0.58 同属平台边缘——0.58 的 18/20
#       靠某个金标块压线，0.62 的上邻 0.64 已进衰退区——0.60 是唯一双向都有
#       一格余量的点。⚠ **我们标 0.45 那次没有这条判据**：只量了 present 一侧，
#       平台边缘和平台中间在我们眼里长得一样。
HIT_FLOOR_BY_MODEL = {
    "BAAI/bge-small-zh-v1.5": 0.45,
    "BAAI/bge-m3": 0.60,
}

# **第三类口径：外部标定**。模型 → 标定出处（人话，会原样进 describe() 给用户看）。
# 表里有 = 这个数是别人量的、我们没复现；表里没有 = 我们自己量的。先例口径是
# **收但标着，不是收了就当自己量的**——出处藏在用户看不见的注释里等于没标。
# ⚠ **出处只影响那句话怎么说，不影响这个数怎么用**，所以它单独一张表，不把
# HIT_FLOOR_BY_MODEL 的值改成元组／字典带出处：`get_hit_floor()` 必须继续返回
# 纯 float——memory_retrieval 那条向量路判据直接拿它比大小，改结构要动所有调用方。
# 顺带一个好处：将来我们自己复核过，**从这张表里删一行**就升级成「已标定」，
# 数一个字都不用改。
EXTERNAL_CALIBRATED = {
    "BAAI/bge-m3": "外部贡献者 星迟 & Ember，1144 块中文语料，PR #4",
}

# 自检夹具：一个**永远不会进标定表**的模型名，专门用来守"未标定绝不退回 0.45"
# 这个最坏方向。⚠ 别再拿真实模型名当这个夹具——bge-m3 当了一阵，2026.08.03 它
# 一被标定，两个文件里的夹具当场全红。名字取成明显虚构的，免得有人以为它能跑。
UNCALIBRATED_FIXTURE_MODEL = "example-org/uncalibrated-test-embed"

# 云端批量大小：一次请求塞多少块。经验值——足够摊薄往返开销，又不至于撞上各家的
# 请求体上限。建库时才会用到批量，查询永远是 1 条。
CLOUD_BATCH = 32
CLOUD_TIMEOUT = 60

# 单条截断（**采纳自外部 PR #1**，`lu7899112-source`，2026.08.01——对方在 2G 内存 VPS 上
# 独立实现了同一条云端路径，这一条是我们那版漏掉的）：护住服务商侧的 token 上限。
# 我们自己的块中位 352 字符，但真实语料里量到过 4386 的长块——一批全是长块就可能
# 撞上批量 token 上限，而**失效形态是整批请求报错**，建库当场中断。
# 截断只作用在**发出去的那份副本**上，本地的块正文一个字都不动。
CLOUD_TRUNC = 2000


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


def hit_floor_override(env=None):
    """`MEMORY_EMBED_HIT_FLOOR` → float 或 None（没设／设歪了）。

    **这个口子是给"自己量过"的人开的，不是给"想找个数填上"的人开的**（采纳自
    外部 PR #1，那一版把这条警告原样搬进了新注释，做法对）。余弦标度跟模型绑死，
    照抄别人的数会让"库里没有就说没有"无声失灵——所以这里只做一件事：**把它
    如实标成 override**，`describe()` 里说明这个数不是我们量的，出了偏差是用户
    自己的标定，不是我们的标定表。

    设歪了（非数字、不在 [-1, 1] 内）**当没设**，不当 0 用：一个悄悄变成 0 的门槛
    等于门槛不存在，而那正是这条最坏的失效方向。"""
    raw = (env if env is not None else os.environ).get(ENV_HIT_FLOOR)
    if raw is None or not str(raw).strip():
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    return val if -1.0 <= val <= 1.0 else None


def get_hit_floor(model, env=None):
    """该模型的实测门槛；**没标定过返回 None，不给替代数字**。
    用户自己量过、通过 `MEMORY_EMBED_HIT_FLOOR` 传进来的覆盖值优先。"""
    override = hit_floor_override(env)
    if override is not None:
        return override
    return HIT_FLOOR_BY_MODEL.get(model)


def _floor_note(floor, env=None, model=None):
    """describe() 里那句门槛说明。**三类分开说，不许混**：

    - **用户自己设的覆盖值**：标明是他设的——出了偏差那是他的标定，不是我们表里的数；
    - **外部标定**（`EXTERNAL_CALIBRATED`）：点名出处、并写明我们未复现。数字功能上
      照用、floor 真生效，只在口径上跟"我们量的"分开；
    - 剩下的才是**我们自己量过**的那一类。

    第二类是 2026.08.03 新加的（收外部 PR #4 的 bge-m3 标定）：**收但标着，不是收了
    就当自己量的**。出处必须落在用户看得见的这句话里——藏在注释里等于没标。"""
    if floor is None:
        return "命中门槛**未标定**"
    if hit_floor_override(env) is not None:
        return f"命中门槛 {floor}（**你自己设的覆盖值**，不是我们量的）"
    source = EXTERNAL_CALIBRATED.get(model)
    if source:
        return f"命中门槛 {floor}（**外部标定**：{source}；我们未复现）"
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
        return get_hit_floor(self.model)

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
                f"语料不出本机；{_floor_note(self.hit_floor(), model=self.model)}；{prefix}")


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
            # 发出去的副本按 CLOUD_TRUNC 截断，本地正文不动（见那个常量的注释）
            chunk = [t[:CLOUD_TRUNC] for t in texts[i:i + self.batch]]
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
        cal = _floor_note(floor, self._env, model=self.model)
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
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return          # 缓存坏了就当没有，重算即可，不该让检索起不来
        if data.get("provider") != self.provider_id:
            return          # 换了模型/服务商：整份作废
        self.vectors = {k: v for k, v in (data.get("vectors") or {}).items()}
        self.loaded_from_disk = True

    def get(self, text):
        return self.vectors.get(text_key(text))

    def put(self, text, vec):
        self.vectors[text_key(text)] = [round(x, 6) for x in vec]
        self.dirty = True

    def save(self):
        if not self.path or not self.dirty:
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            {"provider": self.provider_id, "vectors": self.vectors},
            ensure_ascii=False), encoding="utf-8")
        self.dirty = False
        return True


def embed_with_cache(provider, texts, cache=None):
    """块向量：缓存里有的直接取，没有的才算，算完写回。返回单位化向量列表。

    这就是"建库时算一次、之后只补新块"的全部实现——**云端档的成本模型全靠它**。"""
    if cache is None:
        return provider.embed(texts)
    out = [cache.get(t) for t in texts]
    todo = [i for i, v in enumerate(out) if v is None]
    if todo:
        fresh = provider.embed([texts[i] for i in todo])
        for i, v in zip(todo, fresh):
            out[i] = v
            cache.put(texts[i], v)
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

    # 8.【未知提供方名要报错，别默默跑成本地档】——静默降级在这里等于"用户以为
    #    自己选了云端，其实一直在本地跑"，或者反过来，两个方向都不能接受
    for bad_spec in ("openai", "", "本地"):
        try:
            resolve_provider(bad_spec or "x", env={})
            raise AssertionError(f"{bad_spec!r} 居然被接受了")
        except ValueError:
            pass

    # 9.【采纳自外部 PR #1：单条截断】护住服务商侧 token 上限。
    #    **靶子取真正发出去的请求体**——截断做在本地正文上就成了另一个 bug
    #    （把用户的记忆截短了），所以两头都断言：发出去的短了、手上的没动。
    long_env = {ENV_ENDPOINT: "https://api.example.com/v1/embeddings",
                ENV_MODEL: "BAAI/bge-m3", "MEMORY_EMBED_API_KEY": "k"}
    seen = []

    def _capture(payload):
        seen.append(payload)
        return {"data": [{"index": i, "embedding": [1.0, 0.0]}
                         for i in range(len(payload["input"]))]}

    long_text = "长" * (CLOUD_TRUNC + 500)
    texts = [long_text, "短句"]
    pc = resolve_provider("cloud", env=long_env, transport=_capture)
    #    **把 texts 本身传进去，不传副本**——传副本的话"本地正文没被改"这条
    #    就成了恒真断言（截断即使写成原地修改也测不出来，验过）
    pc.embed(texts)
    sent = seen[0]["input"]
    assert len(sent[0]) == CLOUD_TRUNC, \
        f"超长块没按 CLOUD_TRUNC 截断就发出去了，会撞服务商的 token 上限：{len(sent[0])}"
    assert sent[1] == "短句", "短块不该被动"
    assert texts[0] == long_text, "截断污染了本地正文——那是把用户的记忆截短了"
    #    如实标注：这一条**只挡得住两层一起塌**。`embed()` 开头的 `texts = list(texts)`
    #    是第一层，截断处不原地改是第二层，单点变异会被另一层吸收（两层都拆掉才红，
    #    验过）。不假装它是单点靶。

    # 10.【采纳自外部 PR #1：门槛覆盖口，但设歪了要当没设】
    #     覆盖口是给"自己量过"的人开的。**非法值绝不能悄悄变成 0**——门槛为 0
    #     等于门槛不存在，而那是这条最坏的失效方向（"库里没有就说没有"当场归零）。
    assert get_hit_floor(UNCALIBRATED_FIXTURE_MODEL, env={}) is None, "没设覆盖时，未标定模型仍该是 None"
    assert get_hit_floor(UNCALIBRATED_FIXTURE_MODEL, env={ENV_HIT_FLOOR: "0.62"}) == 0.62, \
        "自己量过的覆盖值该生效——不然只能改源码"
    for bad in ("abc", "", "  ", "1.5", "-2", "nan"):
        got = get_hit_floor(UNCALIBRATED_FIXTURE_MODEL, env={ENV_HIT_FLOOR: bad})
        assert got is None, f"非法覆盖值 {bad!r} 该当没设（得到 {got}），绝不能落成 0"
    #     覆盖值要在 describe() 里标明是用户设的，不许混进"我们标定过"
    note = _floor_note(0.62, {ENV_HIT_FLOOR: "0.62"})
    assert "你自己设的" in note and "已标定" not in note, \
        f"覆盖值被说成了我们的标定值：{note}"

    # 11.【第三类口径：外部标定】（2026.08.03，收外部 PR #4 的 bge-m3 标定 0.60）
    #     **数字功能上照用、floor 真生效**，只在口径上跟"我们量的"分开——先例是
    #     "收但标着"，不是"收了就当自己量的"。
    assert get_hit_floor("BAAI/bge-m3") == 0.60, "外部标定的数该跟自己量的一样真生效"
    #     ⚠ 靶心：**必须是纯 float**。把出处塞进表值（元组／字典）是最容易顺手做的
    #     那种"顺便重构"，而 memory_retrieval 那条向量路判据直接拿它比大小。
    assert type(get_hit_floor("BAAI/bge-m3")) is float, \
        "get_hit_floor 必须返回纯 float——出处走 EXTERNAL_CALIBRATED，不许耦进这个返回值"
    ext = HTTPCloudProvider("https://api.example.com/v1/embeddings", "BAAI/bge-m3",
                            transport=_fake_transport(), env={"MEMORY_EMBED_API_KEY": "k"})
    d_ext = ext.describe()
    #     ⚠ 靶心：出处要落在**用户看得见的那句话**里。把 _floor_note() 的第三支删掉
    #     （退回"已标定"）时这条必红——藏在注释里的出处等于没标。
    for must in ("外部标定", "星迟 & Ember", "1144 块中文语料", "PR #4", "我们未复现"):
        assert must in d_ext, f"外部标定的口径缺了「{must}」：{d_ext}"
    assert "（已标定）" not in d_ext, "外部标定不许被说成我们自己量的"
    #     三类互不串味：我们自己量的那格照旧说"已标定"，覆盖值照旧说"你自己设的"
    note_own = _floor_note(0.45, {}, model=DEFAULT_LOCAL_MODEL)
    assert "（已标定）" in note_own and "外部标定" not in note_own, \
        f"我们自己量的被说成了外部标定：{note_own}"
    note_ov = _floor_note(0.55, {ENV_HIT_FLOOR: "0.55"}, model="BAAI/bge-m3")
    assert "你自己设的" in note_ov and "外部标定" not in note_ov, \
        f"覆盖值被说成了外部标定：{note_ov}"
    assert get_hit_floor("BAAI/bge-m3", env={ENV_HIT_FLOOR: "0.55"}) == 0.55, \
        "自己量过的人给的覆盖值，在外部标定过的模型上同样该优先"

    print("selftest ok（11 项：key 不外泄 / 缺 key 报错 / 分批不乱序 / 前缀按模型 / "
          "未标定即 None / 缓存只算一次 / 坏缓存不致命 / 未知档报错 / "
          "超长块截断（发出去的截、本地的不动）/ 门槛覆盖口（设歪了当没设）/ "
          "第三类口径「外部标定」（数照用、出处进用户可见文本、返回值仍是纯 float））")


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
