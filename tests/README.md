# 发布检查

正式发布的公开判据只有一条：`src/` 下每个 `.py` 用目标 Python 运行
`--selftest`，退出码全部为 0。文件数不写死；新增 `.py` 会自动进入下一次检查。

```bash
python tests/run_release_checks.py
```

需要保留采集条件与逐文件输出时：

```bash
python tests/run_release_checks.py --json-out release-checks.json
```

报告会记录 Python、操作系统、Git 提交、单文件超时与逐文件结果。自检使用代码内置的
合成夹具，不读取用户人格、真实记忆库或 API key；CI 也不配置任何第三方凭证。

这套检查证明公开代码在该解释器下走通了自身判据，不等于某个聊天客户端已经接通，
也不等于真实语料上的检索质量达到某个分数。客户端与真实环境成色仍以 README 的状态矩阵为准。
