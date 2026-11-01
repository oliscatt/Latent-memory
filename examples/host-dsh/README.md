# DSH（DeepSeek Harness）宿主导出示例

DeepSeek Harness 把会话存成压缩事件流，落档需要的是通用 JSONL。这个目录是
[`docs/后台落档.md`](../../docs/后台落档.md) 里「DeepSeek Harness（DSH）会话接入」一节的
可执行版本：一个能直接用的导出器，加一份完全虚构的夹具。

## 用法

```bash
# 看有哪些会话
python3 dsh_export.py --list

# 增量导出某个项目的全部会话
python3 dsh_export.py --project <项目目录名> --out chat.jsonl --state export.state.json

# 只要某天之后
python3 dsh_export.py --project <项目目录名> --since 2026-10-30
```

输出一行一条：

```json
{"id": "…", "ts": "2026-11-01T13:07:00+08:00", "role": "user", "text": "…", "key": "…"}
```

`ts` 由事件时间归一（毫秒 epoch 自动除以 1000，ISO 字符串直接解析），默认按 `+08:00`
输出，`--tz` 可改。`key` 是 `时间+角色+正文` 的短哈希，会话被重写导致 `id` 变化时，
落档端还能靠它对齐。

## 为什么不是一个正则式脚本

| 噪声 | 处理 |
|---|---|
| 思考块、工具调用、工具结果、尝试事件 | 只认 `user/message`、`assistant/message`，且只要 `type == "text"` 的块 |
| 定时提醒、插件注入的历史召回、子代理回流 | `data.source.kind` 不是缺省、也不是 `"user"` 的用户消息一律跳过 |
| 事件重放 | 按消息 `id` 去重（可跨会话） |
| 会话被压缩重写 | 已导出条数变多才增量；变少说明重写过，从头重导 |
| 助手消息两种布局 | `data.message.content` 与 `data.content` 都认 |

## 跑测试

```bash
python3 tests/dsh_export_test.py
```

夹具 `fixture_events.jsonl` 里放了两条**注入消息**（`schedule` / `plugin`）、一条思考块、
一次工具调用与结果、一条重复 id、一条只有空白的助手消息、以及 `data.content` 布局 +
ISO 时间的那一条。测试断言：只剩 3 条真话、毫秒换算正确、第二次导出 0 新增、
会话重写变短后从头重导。本机没有 `zstd` 时，CLI 那一段会打印跳过说明。

## 范围说明

- 已验：字段布局来自 DSH 0.2.x 在 Android/Termux 上的实际会话文件；夹具测试在本机
  Python 3.14 跑通，含 `zstd -dc` 真进程。
- 未验：其它宿主版本或别的平台的 DSH 会话格式。字段对不上时，先按一条普通消息核对
  `data.content` 与 `data.message.content` 到底在哪一层。
- 回滚：本目录是新增文件，删掉即可；不涉及 `src/` 与既有测试。
- 数据：夹具与文档示例均为虚构，不含真实会话、问句或人物。
