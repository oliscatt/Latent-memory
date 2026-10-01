# 版本记录

这里记录公开版本对用户可见的变化。内部任务史、私人复测记录和未发布实验不写进本文件。

## 2026-10-01

### 修复

- `latent_correct` 撤回一条记录时，这条记录写入时带的索引摘要（文件名带同一 `recordId`）也一起退出检索、换窗召回与自动浮现，重启后同样生效；此前摘要仍会以 `status=current` 把旧值带回来。回执会报出跟着退出的摘要条数。`quote` 同时命中正文和它自己的摘要时按一条记录处理，不再报“命中了 2 条”。
- 开 `--passive-recall` 后，所有模型可见工具（不只是 `latent_search`）的 `structuredContent` 都带与正文相同的 `text`，出错时也一样。有的宿主（例如 Claude Code）在结果带 `structuredContent` 时只把它交给模型，此前模型只能看到一份空的交付账本，`latent_session_start`、`latent_append`、`latent_fact_backfill` 等的回执都看不到。

### 文档

- 《快速上手》“只能挂工具的前端这一类”一节：手机上拉不起 stdio 时，Android 可先走同机回环 HTTP（部署形态一之二），语料不离开手机；本机起不了服务才走公网。《给AI的引导指南》与 README 的 Operit 一行同步。
- 《快速上手》部署形态一之二：服务要常驻探活、失败就拉起（例如每 60 秒探一次 `initialize`＋`tools/list`）；补“服务端恢复不等于客户端恢复”——Operit 这类客户端在服务重启后可能丢掉整个插件运行态，要重进一次 App。两条均为外部实测、维护者未复现。
- 《自动浮现》兼容性记录新增一行：只能挂工具的手机前端（Operit、Kelivo 一类）没有“用户发完消息后、模型请求发出前”的 hook 时机，自动浮现不适用，主动检索、换窗召回、写回照常可用（机制／文档推断）。
- 《自动浮现宿主接入》新增“已知宿主差异”：DeepSeek Harness 桌面版把运行时上下文作为末尾一条 `user` 消息追加，适配器取本轮输入时要跳过；只改本步请求副本不等于已证明注入不落历史。
- README、《快速上手》《给AI的引导指南》《自动浮现》与出货时的引导句提示，把 Kelivo、Operit 这一类统一称为“只能挂工具的手机前端”：能挂载 MCP 工具，但不读你的文件（人格只能贴进它自己的设置字段），也没有“用户发完消息后、模型请求发出前”的 hook 时机。归类只看这几项能力，与是否开源无关。
- 两张 Issue 表单的“Latent 完整 commit”栏补了拿不到 git 时的填法：先用说明里的一行 Python 从原压缩包（tar.gz／zip）的归档注释取 40 位 commit；原压缩包也没有了，就写“commit 未知（tar 快照，无 git）”，并附上 RELEASES.md 最新那段的日期和下载日期。

## 2026-09-30

### 新增

- Claude Code 参考 hook `src/claude_code_hook.py`：`SessionStart` 调 `latent_session_start` 注入开场上下文；`UserPromptSubmit` 在 `LATENT_PASSIVE_RECALL=on` 时调隐藏入口，按历史保留模式注入（同一条记录每个会话一次，单轮 4000、单会话 40000 UTF-8 字节封顶，失败静默留空）。只用标准库，要求 Latent 以 Streamable HTTP 常驻。配法见《自动浮现》。
- 实验性自动浮现宿主接入能力：参考宿主支持默认关闭、明确选择临时／历史保留模式、首个模型请求前调度与预算协调；服务端仅在显式 `--passive-recall` 时开放宿主隐藏入口。候选会经过输入预筛、当前范围短句许可、来源／冲突过滤、单事件线选择、上下文覆盖判断与候选级冷却；有效原文与修订会在预算允许时组装成非诱导的现场资料，并协调主动检索、失效与两种生命周期的交付账本。自动路径不改记忆正文、索引与账本，也不增加权重。`latent_append` 可选写入经过原文范围核验的触发词、短句许可、必要引用与确定性 revision，并可按同一 `recordId` 分阶段补齐；更正和精准清理会同步失效或移除辅助账本。旧库不要求迁移。已验证的宿主见《自动浮现》兼容性记录（目前是维护者自用环境下的 Claude Code `UserPromptSubmit` hook 与自建前端），每条只证明它自己那个宿主；真实宿主兼容、真实模型自然效果与实际成本仍需逐项采集。
- 自动浮现事实模式：开了 `--passive-recall` 且配了事实库（`LATENT_PASSIVE_FACTS`，或语料目录旁 `事实库/` 下的 `全量-…`）时启用，完全替代块路径；按向量递出与用户原句最像的两条一句话事实，需要 `--embed`。同一条事实递出后冷却 12 小时，当天写入的事实次日起才浮现，撤回或被取代的记录拆出的事实不再递。冷却与事实向量缓存写在 `~/.cache/latent-passive-facts/`（`LATENT_PASSIVE_FACT_STATE_DIR` 可改），不在语料目录里。
- `latent_fact_backfill`：开 `--passive-recall` 后出现的模型可见工具，分批领块、按随附规则拆成事实交回、服务端逐条校验并记进度，`finish` 后生成全量事实库并当场生效。
- `latent_append` 可选 `facts`：工具里有 `latent_fact_backfill`（开了自动浮现）时带上，带了就追加进事实库的 `增量/`；格式不对时整次写回被拒；回执带 `factsStatus=`。
- 自动浮现准入：内置默认表，可用 `LATENT_PASSIVE_ADMISSION_CONFIG` 指定覆盖文件（词表取并集，数值与开关直接替换；文件不存在或 JSON 写坏时服务启动失败）。新增稀有词路径与热词路径（`LATENT_PASSIVE_HOTWORDS`、`LATENT_PASSIVE_HOTWORDS_OVERRIDE`、`LATENT_PASSIVE_USERDICT`，以及离线命令 `--build-userdict`、`--build-hotwords`）；ready 响应原因码可能附 `topic:`、`anchor:`、`hotword_path`。
- `LATENT_PASSIVE_MAX_PIECE_BYTES`：单条证据的字节上限，默认 800；`LATENT_PASSIVE_FACT_FLOOR`：事实模式的相似度下限，默认 0.42。
- 块路径不递当天写下的记录，原因码 `same_day_record`。
- 事实变迁链：新增 `latent_supersede`，以 `.supersessions.json` 保存新旧记录的双向关系、
  状态与登记时间。默认检索和换窗召回只返回 current；历史意图或同一显式链多节点命中时，
  整链按时间展开且不受 `topN` 截断。老语料无需迁移，未进账本的记录默认为 current。
- `latent_append` 预检：同一工具可传 `mode=preflight`，在零写入前提下复用真实写入的正文、
  证据、旧摘要与未解决项校验，返回预计落点、recordId 与索引状态；补索引模式同样支持。
- 可修复参数错误：`latent_append` 的输入错误在同一回执附“写错／写对”最小输入，帮助调用方
  按原错误类型自修正，不用从规则描述反猜 JSON。
- 工具级精准清理：新增 `latent_cleanup`，只按 `latent_append` 的稳定 `recordId` 两阶段清理
  一条误写正文；确认令牌绑定当前快照，删除前保存不可回灌的隔离副本与审计 manifest，
  未解决事项仍引用时拒绝执行。
- 未解决事项主动浮现：语料目录新增独立 `未解决.md` sidecar，支持显式新增、更新、关闭，
  并在新窗口先于历史快照和最近召回注入；旧调用默认兼容，严格复核为显式部署选项。
- `--log-file`：把本该只走 stderr 的诊断（启动横幅、被拒记录、异常堆栈）同时追加落盘一份。
  stdio 传输下 stderr 由客户端接管、常被直接丢弃，配上这个参数才有排查落点。文件按追加写，
  每次进程启动写一行带 pid 与时间的横幅——同一个文件里出现第二条横幅，即可直接读出子进程
  重启过一次。写盘失败只在终端提示一句，不影响服务启动。
- 正式发布文档：原理与架构、隐私与数据流、升级与迁移、故障排查及安全策略。
- 统一发布检查 `tests/run_release_checks.py`，脚本本身零第三方依赖；没装 jieba 时自动浮现相关自检段落打印“跳过”。
- GitHub Actions（`.github/workflows/release-checks.yml`）：在 Python 3.10、3.11、3.12 的 Linux 容器中先装 jieba，再运行全部公开自检，并保留 JSON 报告。
- `latent_search` 的 `structuredContent` 带 `text`。
- 换窗开场列出未解决事项时，提示模型“由你自己选择合适的时机提起”。
- `LATENT_FACT_BACKFILL=off`：不把 `latent_fact_backfill` 给模型（已经有全量、不想让模型自己补拆时用）。
- 自动浮现块路径的零依赖兜底：没装 jieba 时拿热词表对用户原句做子串匹配，不再整轮留空。热词表仍要在装了 jieba 的机器上用 `--build-hotwords` 生成，可以复制到零依赖机器上用；ready 响应的 `reasonCodes` 多一个 `substring_path`。装了 jieba 的部署行为不变。

### 破坏性变化

- **HTTP 下自动浮现隐藏入口 fail-closed**：走 Streamable HTTP 时必须另配 `--hook-token`（或 `MEMORY_HTTP_HOOK_TOKEN`），只有带这条 token 的请求看得见、调得动 `latent_passive_recall`；不配则对所有凭证关闭，宿主 hook 调用得到 `-32601`。`--hook-token` 与 `--token` 相同时拒绝启动。stdio 不受影响。
- **自动浮现块路径需要 jieba**：`pip install -r requirements-passive.txt`。没装时服务照常启动，块路径每轮留空；事实模式不用 jieba；主动检索、换窗召回与写回不受影响。

### 修复

- 自动浮现块路径不再递出已被取代（`latent_supersede`）的记录：热词路径与稀有词路径的 index 桥接会直接扫全库取候选，此前没排除被取代、撤回与插件隐藏的块，旧状态可能被当证据递出；撤回的块虽然递不出去，却会让整轮以 `source_unresolved` 留空。现在三者都在共用的准入处挡掉，挡掉后照常看下一个候选。
- `latent_correct` 撤回后的同词面提醒在候选超过全库 20% 时不再报告失去定位价值的精确
  条数，但仍说明存在未受影响的记录并列出最近三条定位；未超过门槛时仍报告条数。候选
  判据与排序未改，提示统一称“共享词面”，并明确不应据此逐条撤回无关记录。
- 人格初始化 `--candidates` 重传：新候选只替换上一轮的语料候选，协议骨架项保留；
  同时作废旧的十二节版本表与选择，必须重跑 `choose-sections` 再出货，
  避免带着已替换候选的版本一路走到 `ship` 才报人格文件不完整。
- 旧 `indexSummaries` 校验失败时会定位到 A／B／C 具体段，并指出证据词缺在摘要侧还是
  `text/current_state` 侧；格式、逐字证据与三段锚定规则在《快速上手》中完整列明，并与
  错误回执共用“写错／写对”样例口径。
- 索引重建失败时返回错误，下次调用重试。
- `--doctor` 增加工具 schema 顶层形状检查；检索自查优先选择含内容词的标题，只有日期或
  窗口号的弱探针失败时降为警告，避免把探针本身太弱误报成接线故障。
- 工具数量与清单统一为八个（开 `--passive-recall` 时另有模型可见的 `latent_fact_backfill`）；
  显式工具白名单须包含 `latent_supersede`、`latent_cleanup` 与 `latent_unresolved`，漏配会静默
  少工具。
- 零依赖建库的内存峰值：关系图谱改为逐块计算邻居，不再先攒全部块对再取前几名；余弦路与 BM25 共用同一份词频，只另存大小写不同的部分。检索结果不变，建库也更快。块数越多，峰值下降越明显。
- 发布检查在 macOS 上 `memory_init.py`、`e2e_smoke.py` 自检误红：默认临时目录本身六十多字，拼出的引导句撞 100 字长度闸，且 `/var` 是软链。两份自检改用短而无软链的临时目录（macOS 为 `/private/tmp`，Linux 不变，Windows 不动），产品的长度闸不放宽。

### 文档

- 新增面向用户与 AI 的《自动浮现》、面向宿主实现者的《自动浮现宿主接入》、两份低门槛
  GitHub Issue 表单及贡献指南；明确“支持 MCP 不等于支持自动浮现”、部署报告、先关闭再反馈、
  不改记忆库的边界、兼容性三档和隐藏入口不得暴露给模型的硬红线。
- 漏洞私下报告统一走 GitHub 公开仓库的私密漏洞报告（Security Advisory）。
- 《升级与迁移》补充客户端缓存旧工具表或硬编码根级必填字段时，合法的补索引调用可能在
  到达服务端前就被拦截；可用“报错措辞不来自服务端且服务端日志无请求”辨认并刷新 schema。
- 补充上游历史重写后的全新 clone 路径，并明确源码换目录时用户语料、索引、thread 与
  sidecar 原位保留；旧语料只有 timeline、索引层为 0 时提供不重复写正文的补索引步骤。
- 补充公网 IP 入站整体不可达时的 Cloudflare Tunnel 排查路线，以及 Persona 注入属于客户端
  或宿主职责、不能用“记忆能召回”反证人格文件已经注入的边界。
- 《故障排查》新增“内存不够被系统杀掉、反复重启”：怎么确认、为什么会连成重启循环、用 systemd drop-in 限制内存并止住循环。
- 《故障排查》同一节补本地 embedding 在小内存 VPS 上的做法：本机用同一模型建好向量缓存再拷上去；运行时仍要加载模型，内存紧的机器改走云端 embedding 或零依赖。
- 《升级与迁移》补“`src/` 里上游不带的文件”：替换 `src/` 或换目录重新 clone 会丢掉热词表、用户词典、覆盖表与 `latent_plugin_*.py`，且不报错；给出列出与带过去的办法。另补“另开目录升级（不停服务）”，并写明 `--doctor` 要照服务启动参数原样带上。
- 《升级与迁移》《故障排查》《自动浮现宿主接入》核对工具表处补：设了 `LATENT_FACT_BACKFILL=off` 时不列 `latent_fact_backfill`。

### 说明

- 本节是 2026-09-30 推送到公开仓库 `main` 的版本。尚未创建版本 tag，复现或报告问题时请记录完整 commit。
- CI 证明代码自检在对应环境中通过，不替代 Windows 真机、聊天客户端或真实语料验证。

## 首个正式版本之前

首个正式版本之前没有稳定 tag。需要复现某个行为时，请使用对应的完整 commit；
不要把当前文档套到别的提交上。
