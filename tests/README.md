# 发布检查

正式发布的公开判据只有一条：`src/` 下每个 `.py` 用目标 Python 运行
`--selftest`，退出码全部为 0。文件数不写死；新增 `.py` 会自动进入下一次检查。

```bash
pip install -r requirements-passive.txt   # jieba，自动浮现相关自检要用
python tests/run_release_checks.py
```

检查脚本本身零第三方依赖。没装 jieba 时各文件照样跑完，但自动浮现相关的自检段落会打印
“跳过”，这部分就没有被覆盖。GitHub Actions（`.github/workflows/release-checks.yml`）在 Python 3.10／3.11／3.12 上先装 jieba 再跑。

`tests/dsh_export_test.py` 是宿主适配的夹具测试，不进上面这条正式判据（它不跑 `--selftest`），
单独跑：`python3 tests/dsh_export_test.py`。只用 `examples/host-dsh/` 里的虚构事件，无第三方依赖；
本机没有 `zstd` 时，CLI 增量那一段打印跳过，其余断言照跑。

**macOS 上 `e2e_smoke.py` 与 `memory_init.py` 两项会红，结果是 26/28，这不是出货问题。** macOS 的临时
目录 `/var/…` 是 `/private/var/…` 的软链接，这两份自检用临时目录夹具比对产出路径时，一边是别名、
一边是真实路径，字面对不上；报错文字（“引导句没指向这次出货的人格文件”“config 指向……”）看着像
出货 bug，实际出货路径是对的。Linux（CI）与 Windows 上这两项通过。

需要保留采集条件与逐文件输出时：

```bash
python tests/run_release_checks.py --json-out release-checks.json
```

报告会记录 Python、操作系统、Git 提交、单文件超时与逐文件结果。自检使用代码内置的
合成夹具，不读取用户人格、真实记忆库或 API key；CI 也不配置任何第三方凭证。

这套检查证明公开代码在该解释器下走通了自身判据，不等于某个聊天客户端已经接通，
也不等于真实语料上的检索质量达到某个分数。客户端与真实环境成色仍以 README 的状态矩阵为准。


## HTTP 诊断专项

```bash
python tests/http_diagnostics_test.py -v
```

用标准库启动真实 HTTP 子进程，使用临时虚构语料、随机回环端口及合成凭证；核对启动横幅与实际鉴权、拒绝日志和限流摘要的带时区时间／PID、敏感信息不落日志及正常请求静音。专项不会随 `src/*.py --selftest` 自动运行，需要单独执行。限流测试仅在测试子进程缩短窗口和上限，不改变服务默认设置。通过以命令退出 0 且全部断言成功为准，不代替具体宿主的连接验收。

运行 `python tests/recent_recall_dates_test.py` 可用虚构语料验证无查询换窗的日期排序、文件修改时间回退及带查询的正文命中。
