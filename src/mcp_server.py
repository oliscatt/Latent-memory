#!/usr/bin/env python3
"""
MCP server 外壳参考实现（任务卡"MCP-server外壳"，规格 §9 未决项第二条）。

**这是最薄的一条完整链路**：九个 src 文件此前全部只被自造的合成语料验证过，
没有一次被真实客户端调用过——selftest 再严谨也测不出"客户端真调这个工具时，
参数格式对不对、返回结构好不好用"。所以先把链路打通，再往深了做。

**薄适配层，外壳里不重新实现任何逻辑**（本单硬性约束）：外壳只做三件事——
协议编解码、参数转发、调用现成函数。逻辑一旦糊在适配层里，真机测试暴露的问题
就分不清是"接口设计的问题"还是"底层库的问题"。
这条被写成可断言的形式：**外壳返回的文本逐字等于底层库函数的返回值**，谁在
适配层里重新实现格式化，selftest 立刻红（见 selftest 第 4 项）。

零依赖：不引 mcp SDK，stdlib 手写 JSON-RPC，跟本项目其余部分同风格。传输两条：
stdio（默认，宿主拉起）与 Streamable HTTP（--http，2026.08.03 加，给只认 HTTP 的
闭源前端直连，替掉 supergateway 桥——为什么与边界见 make_http_server 上面那段）。
协议按官方规格 2025-06-18 实现（initialize / notifications/initialized /
tools/list / tools/call，字段名与错误分层均照规格），**已查证不凭记忆写**。

**成色（2026.08.03 改准，此前写的是“尚未与任何真实客户端握手核对”，已过时）**：
已有一条真实客户端链路——Operit（Android）＋ proot Ubuntu，客户端按 stdio 拉起本进程，
握手、工具列表、真实调用三样都走过：当时五个业务工具全部接通，`latent_append` 写入后
**新开会话仍检索命中**。外部实测转录，本项目未复现。

⚠ **剩下的边界仍然作数，别读成“都验过了”**：Claude 桌面 App 那一格仍未连过；
超长文本怎么截断、参数容忍度这些真机问题只在一家客户端上过过手。
另有四条与宿主拉起有关的实测事实记在《快速上手》§3c「部署形态一」：
stdio 只能由宿主拉起（手动 `nohup` 必崩）、懒加载每次调用重读语料、
客户端的 `lastStartTime` 不能当跑通判据、这条路上终端里不会有请求日志
（stdio 没有端口，“HTTP 终端没收到 latent_search”在这条路上永远成立，
不是没跑通的证据）。

工具集一一对应现成能力，不新造：
  latent_search  → MemoryIndex.retrieve
  latent_session_start  → SessionRecall.on_session_start（thread 块 + 召回块 + 自查指令）
  latent_append  → memory_retrieval.append_record（正文层的笔）
  latent_correct → MemoryIndex.retract（+ 可选 append_record 写更正）
  latent_cleanup → 按稳定 recordId 两阶段清理一条自写记录（先隔离备份）
  latent_unresolved → UnresolvedStore.apply（显式维护未解决清单）
  latent_thread_close   → session_thread.close_thread + ThreadStore.append

用法：
  python mcp_server.py --selftest
  python mcp_server.py --corpus <md目录> [--threads <threads.jsonl>]   # stdio 服务
  python mcp_server.py --corpus <md目录> --http 127.0.0.1:8765 --token <口令>  # HTTP 服务
  python mcp_server.py --doctor --corpus <md目录> [--threads <threads.jsonl>]  # 部署体检
客户端配置（Claude Desktop 之类）里把上面第二条命令填成 server 启动命令即可；
只认 HTTP 的客户端（Kelivo/Operit 这类）用第三条起服务、客户端里填地址与 token
（`--http` 的参数是 `[HOST:]PORT`，**省略 HOST 只绑回环、别的机器连不到**；
公网仍要域名 + 反代管 TLS，见《快速上手》§3c 部署形态二）；
配完接不上、或者不确定 --corpus 指对了没有时，把同一行参数换成 --doctor 跑一次
（体检只读，不往语料目录写任何东西）。
"""

import argparse
import difflib
import errno
import hashlib
import hmac
import http.server
import io
import ipaddress
import json
import os
import re
import select
import socket
import sys
import threading
import time
import tempfile
import traceback
import urllib.parse
import unicodedata
from datetime import datetime, timezone as _dt_tz
from pathlib import Path

# 同目录模块，import 不触发各自的 CLI
from memory_retrieval import (MemoryIndex, load_corpus, append_record, corpus_files,
                              plan_append_record, plan_index_records,
                              query_miss_rate, miss_rate_note, annotate_block,
                              _chunk_key, INDEX_SUMMARY_TEMPLATE,
                              INDEX_SUMMARY_EXAMPLE, INDEX_SUMMARY_FORMAT_GUIDE)
from embedding_provider import resolve_provider
# 切块下界与体检的切块成色检查**共用同一个判据**，不各抄一份
from chunking_experiment import chunk_body as _chunk_body, chunk_heading
from session_recall import SessionRecall, format_recall_block, SELF_CHECK_FOOTER  # noqa: F401
from session_thread import ThreadStore, close_thread
from unresolved_state import (FILENAME as UNRESOLVED_FILENAME, UnresolvedRequestError,
                              UnresolvedStore, UnresolvedStoreError, source_record_ids,
                              validate_ops)
# 记忆所有者的时区（任务卡"写回时区与跨日归窗"）：一个进程一份，stdio 与 HTTP
# 两种起动形态共用同一个，不各自造一份换算逻辑
from time_context import (TimeContext, detect_local_timezone, parse_record_time_marker,
                          tzdb_available)

PROTOCOL_VERSION = "2025-06-18"   # 官方规格版本，已查证
SERVER_INFO = {"name": "memory-protocol", "title": "记忆协议", "version": "0.1.0"}

# 服务器级说明，随 initialize 响应回给客户端（规格 lifecycle 一节的 instructions
# 字段，客户端通常会注入进模型上下文）。2026.07.31 真机实测后补的——第一次主动性
# 测试完败：问"我们之前说好的那件事"，模型把它理解成"本次会话记录"，答"我没有
# 会话记录"，转头去查了宿主自带的记忆功能，压根没想到这里有个长期记忆库。
# 两个教训写进这段文字：①说清楚这是什么记忆（跨会话的长期库，不是本次对话）；
# ②直接堵死"我没有相关记录"这句默认话术——在挂着记忆库的情况下它就是错的。
INSTRUCTIONS = (
    "这台服务器挂着用户与你之间的**长期关系记忆库**：跨会话保存的时间线、摘要与"
    "会话收尾，不是本次对话的聊天记录。\n"
    "什么时候用：对方提到过去发生过的事、某个约定、某个日期/地点/称呼/人名，"
    "或者你对某个细节拿不准——先用 latent_search 查，再开口。\n"
    "**不要在没查之前说“我没有相关记录”“我不记得”**：挂着记忆库时这句话是错的，"
    "查一下往往就有。查完自然接上话即可，不用报告自己搜过。\n"
    "记忆库不是只读的：对话里出现值得长期记住的事——新约定、重要事件、状态变化、"
    "对方明确说要记住的——**当场用 latent_append 写进去**，不用请示，不用等会话结束。\n"
    "每次完整 latent_append 与会话收尾都顺手复核未解决清单：新出现且尚未结束的用 "
    "open；仍没结束但缺口变了用 update；明确完成、取消或不再继续才 close；没有变化传 "
    "none。不要按关键词或内容相似自动关闭。thread 只是上个窗口的历史快照，当前仍未解决"
    "什么只看清单。单独维护或失败重试用 latent_unresolved，不要重复 append 正文。\n"
    "但状态变化不总是直接追加：对方说出住址、当前职业、当前状态这类同一时刻只能有"
    "一个答案的事实，而 latent_search 查到的旧值与新说法互斥时，先用 latent_correct "
    "让旧值退出检索，再用 latent_append 写新值；不要只 append 造成并存。冲突用常识判断，"
    "不要预先给每句话分类。喜欢的电影、去过的地方、多个朋友这类并列事实应当共存，不得"
    "为了写新项而 correct 旧项。\n"
    "记错了的事也有出口：对方指出某段记忆不对或已经过时，**当场用 latent_correct "
    "撤回旧记录并写上更正**——只口头认错不改库，下次照样检索到错的。\n"
    "只有在明确要清掉一次误写正文、且手上有 latent_append 返回的 recordId 时，才用 "
    "latent_cleanup：必须先 preview，向对方展示将移除的记录，再把确认令牌逐字交回 delete；"
    "不得按相似文字猜记录，不得跳过确认。\n"
    "⚠ 但 latent_correct 撤的是你**逐字摘的那一块**，不是「这件事」：同一说法若还写在"
    "别的记录里，那些不受影响，开场自动浮现（没有 query、按时间和权重排）可能把旧说法"
    "直接端回来。改过重要的事实后，服务端会提醒库里还有几块提到同一说法——按提示用旧"
    "说法再查一次、逐条撤掉，才算干净。\n"
    "查过但确实没有的，就如实说没找到——查过之后的“没有”是诚实，查之前的“没有”才是错。\n"
    "第一次检索没找到、或返回内容明显对不上时，最多再查一次：保留原 query，把更具体的"
    "说法放进 queryVariant（补名字、物件、地点或记录里可能用过的词）；服务器会融合两种"
    "问法。第二次仍对不上就停，不要循环查到有为止。\n"
    "新会话开场先调一次 latent_session_start，会话结束前调一次 latent_thread_close。"
)

# 撤回粒度提醒：命中「同一说法」的其它块可能上百（报告方 1729 块），提示里只列
# 最近这么几条定位信息。⚠ 这是**输出预算**（别把提醒撑爆上下文），**不是判据**
# ——判据是 same_claim_survivors 里的区分性 token，与这个数无关。
_SURVIVOR_PREVIEW = 3

# JSON-RPC 标准错误码（规格"Error Handling"一节：未知工具/参数非法走协议错误）
E_METHOD_NOT_FOUND = -32601
E_INVALID_PARAMS = -32602

_UNRESOLVED_OPS_SCHEMA = {
    "type": "array", "minItems": 1,
    "description": ("显式复核当前未解决事项：open／update／close，或单独传 none。"
                    "旧客户端可省略；省略时核心写入仍成功，但回 "
                    "unresolvedStatus=not_reviewed。严格模式下缺字段才拒绝。"),
    "items": {"type": "object", "properties": {
        "action": {"type": "string", "enum": ["open", "update", "close", "none"]},
        "id": {"type": "string", "description": "update／close 的 U- 开头稳定 ID"},
        "summary": {"type": "string", "description": "open／update 后当前还缺什么；单行"},
    }, "required": ["action"]},
}

TOOLS = [
    {
        "name": "latent_search",
        "title": "检索记忆",
        "description": "检索这段关系的长期记忆库（跨会话保存的时间线与摘要，"
                       "不是本次对话的聊天记录）。**对方提到任何过去发生过的事、"
                       "约定、日期、地点、称呼、人名，或你对细节拿不准时，先查这里**——"
                       "不要在查之前说“我没有记录”“我不记得”，那在挂着记忆库时是错的。"
                       "若第一次结果不相关，最多重试一次：原问题仍放 query，更具体的说法放"
                       "queryVariant；两种问法会融合，第二次仍不相关就如实说没找到。"
                       "查完自然接上话，不用报告查过。",
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "要查什么"},
                "queryVariant": {
                    "type": "string",
                    "description": "可选的一种更具体说法；与原 query 融合。最多一个，不能循环扩写",
                },
                "topN": {"type": "integer", "description": "返回几条，默认 5"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "latent_session_start",
        "title": "换窗召回",
        "description": "新会话开场**主动**调一次（不用等对方要求）：先带回当前明确标记的"
                       "未解决事项，再带回上次会话历史快照与最近发生过的事，"
                       "并附四项故障信号自查指令。",
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "latent_append",
        "title": "写回记忆",
        "description": "把值得长期记住的事写进记忆库。**对话里出现新约定、重要事件、"
                       "状态变化，或对方明确说要记住的——当场调用，不用请示，也不用等"
                       "会话结束**。若 latent_search 里已有与新说法互斥的单值事实，必须先"
                       "latent_correct 旧值，不得只调本工具；可并列的喜好、经历与人际事实则"
                       "直接追加，不得撤旧项。写发生了什么和原话（纪录片写法，不写评语）；"
                       "current_state 必填：这件事现在的状态（约定成立/还在处理/"
                       "已解决……）——不写的话，未来重读会把它当成正在发生的事。"
                       "同批尽量提供 indexEvidence：只摘原文证据片段并标 event／feeling／"
                       "reason／state／context，服务端统一生成索引。证据不合格时正文仍保存，"
                       "回执给 recordId 和 pending 状态；之后用同一个 latent_append 只传 "
                       "recordId＋indexEvidence 补索引，不得重复正文。旧 indexSummaries 仅兼容。"
                       "完整写回建议同时给 unresolvedOps；旧客户端省略时正文仍保存，但回执会"
                       "明确标 not_reviewed。想先校验、绝不写盘时传 mode=preflight；通过后省略"
                       "mode 或改为 write 再调用，写入时仍会重新校验。",
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["write", "preflight"],
                         "description": "可选；默认 write。preflight 只校验并返回预计落点，绝不写盘"},
                "text": {"type": "string",
                         "description": "发生了什么——具体动作和原话，不是概括"},
                "current_state": {"type": "string",
                                  "description": "**必填**：这件事现在的状态（约定成立／还在处理／"
                                                 "已解决……）。不写会被拒绝；写了才分得清"
                                                 "「当时发生过」和「现在还在进行」——"
                                                 "缺了它未来重读会把旧事当成正在发生"},
                "window": {"type": "integer",
                           "description": "第几个窗口（可省略，按日期自动归窗）"},
                "recordId": {"type": "string",
                             "description": "补索引模式：此前写回返回的记录标识；与 indexEvidence 同传，不再重复 text"},
                "indexEvidence": {
                    "type": "array", "minItems": 1,
                    "description": "推荐；从同批正文逐段摘取的索引证据。服务端容忍空白、全半角和引号形态差异，最终只写回定位到的原文",
                    "items": {"type": "object", "properties": {
                        "type": {"type": "string",
                                 "enum": ["event", "feeling", "reason", "state", "context"]},
                        "quote": {"type": "string"},
                    }, "required": ["type", "quote"]},
                },
                "indexSummaries": {
                    "type": "array", "minItems": 1,
                    "description": "旧客户端兼容字段；新调用请改用 indexEvidence",
                    "items": {"type": "object", "properties": {
                        "summary": {"type": "string",
                                    "description": INDEX_SUMMARY_FORMAT_GUIDE},
                        "evidenceTerms": {"type": "array", "minItems": 1,
                                          "items": {"type": "string"}},
                    }, "required": ["summary", "evidenceTerms"]},
                },
                "unresolvedOps": _UNRESOLVED_OPS_SCHEMA,
            },
        },
    },
    {
        "name": "latent_correct",
        "title": "更正记忆",
        "description": "对方指出某段记忆**记错了或已经过时**（关系变了/搬家了/"
                       "计划改了……）时当场调用：撤回那段旧记录（检索不再返回它；"
                       "原文件与撤回原因留档，可追溯），并可同时写入更正后的记录。"
                       "quote 必须从 latent_search 返回的原文里**逐字**摘一段、"
                       "足够长能唯一定位那条记录。只口头认错不调这个工具的话，"
                       "库没变，下次照样检索到错的。⚠ 撤的是你摘的**那一块**，"
                       "不是「这件事」——同一说法若散在别的记录里不受影响；"
                       "撤回后服务端会告诉你库里还有几块提到同一说法，按提示逐条撤才干净。"
                       "处理单值事实的当前值变化时，本次先只撤旧值（省略 correction），"
                       "再单独调 latent_append 写新值；这样动作顺序可见，不会被误做成纯 append。",
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": False,
        },
        "inputSchema": {
            "type": "object",
            "properties": {
                "quote": {"type": "string",
                          "description": "要撤回的记录原文片段（逐字，不转述）"},
                "reason": {"type": "string",
                           "description": "为什么撤回——记错了/已过时/对方更正了什么"},
                "correction": {"type": "string",
                               "description": "更正后的内容（可省略：只撤不补）"},
                "current_state": {"type": "string",
                                  "description": "更正这件事现在的状态（写 correction 时必填）"},
            },
            "required": ["quote", "reason"],
        },
    },
    {
        "name": "latent_cleanup",
        "title": "精准清理误写记录",
        "description": "只清理 latent_append 自己写下、且有稳定 recordId 的一条误写正文；"
                       "同时移出它的关联索引。必须先 action=preview，向对方展示目标，"
                       "再把返回的 confirmation 逐字用于 action=delete，并填写 reason。"
                       "删除前会写可恢复隔离副本和审计 manifest。未解决清单仍引用该记录时"
                       "拒绝删除，须先 latent_unresolved update／close。不得用它模糊匹配、"
                       "批量删除或清理普通导入语料。",
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": False,
        },
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["preview", "delete"]},
                "recordId": {"type": "string",
                             "description": "latent_append 返回的 16 位小写十六进制标识"},
                "confirmation": {"type": "string",
                                 "description": "delete 模式必填：刚才 preview 返回的逐字确认令牌"},
                "reason": {"type": "string",
                           "description": "delete 模式必填：为什么清理，写入审计 manifest"},
            },
            "required": ["action", "recordId"],
        },
    },
    {
        "name": "latent_unresolved",
        "title": "维护未解决事项",
        "description": "单独新增、更新或关闭当前未解决事项；用于没有正文要写时的明确维护，"
                       "或 append／thread_close 回执为 blocked 后只重试清单操作。"
                       "关闭只删除清单行，不撤回 timeline 原始事实。"
                       "⚠ **这是只写工具，没有「查看」动作**：action 只有 open（新增一条）／"
                       "update（改一条）／close（关掉一条），**没有 list**，"
                       "且 open 是往清单里写一条、不是列出清单。"
                       "要看当前未解决什么，调 latent_session_start——它开场就先给【当前未解决】那一段。",
        "annotations": {"readOnlyHint": False, "destructiveHint": True,
                        "idempotentHint": False, "openWorldHint": False},
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["open", "update", "close"]},
                "id": {"type": "string"},
                "summary": {"type": "string"},
                "source": {"type": "string", "description": "可选来源；省略记为 manual"},
            },
            "required": ["action"],
        },
    },
    {
        "name": "latent_thread_close",
        "title": "收尾本次会话",
        "description": "会话结束前**主动**调一次：记下这次聊了什么线、当下状态、"
                       "并复核未解决清单，下个会话靠它接上。当下状态必填——不写的话，"
                       "下个会话会把已经结束的事读成正在发生。",
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
        "inputSchema": {
            "type": "object",
            "properties": {
                "window": {"type": "integer", "description": "第几个窗口（正整数）"},
                "current_state": {"type": "string", "description": "这件事现在的状态"},
                "topics": {"type": "array", "items": {"type": "string"}, "description": "聊了什么线"},
                "open_loops": {"type": "array", "items": {"type": "string"},
                               "description": "旧客户端兼容字段；不再作为当前未解决集合召回"},
                "started_at": {"type": "number", "description": "会话开始时间戳，省略则按 1 小时前算"},
                "unresolvedOps": _UNRESOLVED_OPS_SCHEMA,
            },
            "required": ["window", "current_state"],
        },
    },
]


def _utf8_text_stream(binary, write=False):
    """把二进制流包成 UTF-8 文本流，不看系统区域编码脸色（见 serve_stdio 说明）。"""
    return io.TextIOWrapper(binary, encoding="utf-8", newline="", write_through=write)


class ToolError(Exception):
    """工具执行失败（业务层），按规格回 isError:true 的正常结果，不是协议错误。"""


_APPEND_INPUT_EXAMPLES = {
    "full": (
        '{"text":"她说下周要去复查"}',
        '{"text":"她说下周要去复查","current_state":"日期还没定",'
        '"indexEvidence":[{"type":"event","quote":"她说下周要去复查"}]}'
    ),
    "evidence": (
        '{"indexEvidence":[{"type":"event","quote":"她计划下周复查"}]}',
        '{"indexEvidence":[{"type":"event","quote":"她说下周要去复查"}]}'
    ),
    "summary": (
        '{"indexSummaries":[{"summary":"只有一行","evidenceTerms":["只有一行"]}]}',
        '{"indexSummaries":[{"summary":"' + INDEX_SUMMARY_EXAMPLE.replace("\n", "\\n")
        + '","evidenceTerms":["下次想去看企鹅漫步","凉快的早上","新约定"]}]}'
    ),
    "repair": (
        '{"recordId":"0123","text":"重复正文","indexEvidence":[]}',
        '{"recordId":"0123456789abcdef","indexEvidence":['
        '{"type":"event","quote":"她说下周要去复查"}]}'
    ),
    "mode": (
        '{"mode":"dry-run","text":"她说下周要去复查","current_state":"日期还没定"}',
        '{"mode":"preflight","text":"她说下周要去复查","current_state":"日期还没定"}'
    ),
    "unresolved": (
        '{"unresolvedOps":[]}',
        '{"unresolvedOps":[{"action":"none"}]}'
    ),
    "exclusive": (
        '{"indexEvidence":[{"type":"event","quote":"原文"}],'
        '"indexSummaries":[{"summary":"旧摘要","evidenceTerms":["原文"]}]}',
        '{"indexEvidence":[{"type":"event","quote":"原文"}]}'
    ),
}


def _append_input_error(message, category="full"):
    """给模型同类最小反例与正例；所有入口共用，避免错误文案各抄一份。"""
    if "写错：" in message and "写对：" in message:
        return message
    wrong, right = _APPEND_INPUT_EXAMPLES[category]
    return f"{message}\n写错：{wrong}\n写对：{right}"


def _content_char(ch):
    return unicodedata.category(ch)[0] in {"L", "N"}


_INDEX_EVIDENCE_TYPES = ("event", "feeling", "reason", "state", "context")
_INDEX_EVIDENCE_LABELS = {
    "event": "事件", "feeling": "感受", "reason": "原因",
    "state": "当下", "context": "补充",
}
_QUOTE_PUNCTUATION = str.maketrans({
    "“": '"', "”": '"', "„": '"', "‟": '"', "‘": "'", "’": "'",
    "：": ":", "；": ";", "，": ",", "。": ".", "！": "!", "？": "?",
})


def _canonical_quote(value):
    """只折叠排版差异，并保留规范字符到原字符串位置的映射。"""
    chars, positions = [], []
    for original_pos, ch in enumerate(value):
        for normalized in unicodedata.normalize("NFKC", ch).translate(_QUOTE_PUNCTUATION):
            if normalized.isspace():
                continue
            chars.append(normalized)
            positions.append(original_pos)
    return "".join(chars), positions


def _closest_source_span(needle, sources, window=12):
    """定位失败时给出原文里最像的一段：省掉「重新 search 整段原文再抠字」那一轮
    （外部反馈 #23：为拿到能过逐字校验的证据而反复拉原文，单轮上下文能涨十万级）。"""
    best = None
    for source in sources:
        haystack, positions = _canonical_quote(source)
        matcher = difflib.SequenceMatcher(None, haystack, needle, autojunk=False)
        match = matcher.find_longest_match(0, len(haystack), 0, len(needle))
        if match.size and (best is None or match.size > best[0]):
            start = positions[match.a]
            end = positions[match.a + match.size - 1]
            best = (match.size, source[max(0, start - window):end + 1 + window].strip())
    return best[1] if best else None


def validate_index_evidence(items, source_texts):
    """把容忍排版差异的 quote 映射回原文；绝不把模型改写直接写进索引。"""
    if not isinstance(items, list) or not items:
        raise ValueError("indexEvidence 必须是非空数组")
    sources = [source for source in source_texts if isinstance(source, str) and source]
    resolved = []
    for number, item in enumerate(items, 1):
        if not isinstance(item, dict):
            raise ValueError(f"索引证据第 {number} 条必须是对象")
        kind, quote = item.get("type"), item.get("quote")
        if kind not in _INDEX_EVIDENCE_TYPES:
            raise ValueError(f"索引证据第 {number} 条 type 必须是 "
                             + "／".join(_INDEX_EVIDENCE_TYPES))
        if not isinstance(quote, str) or not quote.strip():
            raise ValueError(f"索引证据第 {number} 条 quote 不能为空")
        needle, _ = _canonical_quote(quote.strip())
        if sum(_content_char(ch) for ch in needle) < 2:
            raise ValueError(f"索引证据第 {number} 条 quote 太短；请摘至少两个内容字符的连续原文")
        original = None
        for source in sources:
            haystack, positions = _canonical_quote(source)
            pos = haystack.find(needle)
            if pos >= 0:
                original = source[positions[pos]:positions[pos + len(needle) - 1] + 1]
                break
        if original is None:
            hint = _closest_source_span(needle, sources)
            raise ValueError(f"索引证据第 {number} 条无法映射回原文：{quote}。"
                             "只允许空白、全半角和中英文引号／标点形态不同，不能同义改写"
                             + (f"。原文里最接近的一段是：{hint}"
                                "——请直接从这一段里逐字摘，不必重新检索原文"
                                if hint else ""))
        row = {"type": kind, "quote": original.strip()}
        if row not in resolved:
            resolved.append(row)
    return resolved


def render_index_evidence(items, local_date):
    """由服务端把已定位的原文证据统一渲染成一块高密度索引。"""
    grouped = {kind: [] for kind in _INDEX_EVIDENCE_TYPES}
    for item in items:
        if item["quote"] not in grouped[item["type"]]:
            grouped[item["type"]].append(item["quote"])
    lines = [f"**{local_date.replace('-', '.')}** · 原文证据索引"]
    for kind in _INDEX_EVIDENCE_TYPES:
        if grouped[kind]:
            lines.append(f"{_INDEX_EVIDENCE_LABELS[kind]}：" + "；".join(grouped[kind]))
    return "\n".join(lines)


def validate_index_summaries(items, text, current_state, local_date):
    """校验索引摘要的结构与证据锚点，返回规范化摘要文本列表。"""
    if not isinstance(items, list) or not items:
        raise ValueError("未提供 indexSummaries：本次不写索引摘要。")
    evidence_base = text.strip() + "\n" + current_state.strip()
    wanted_date = local_date.replace("-", ".")
    validated = []
    for number, item in enumerate(items, 1):
        if not isinstance(item, dict):
            raise ValueError(f"摘要第 {number} 条必须是对象")
        summary, terms = item.get("summary"), item.get("evidenceTerms")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError(f"摘要第 {number} 条 summary 不能为空；本次没有写盘。")
        summary = summary.strip()
        first = re.match(r"^\*\*(\d{4}\.\d{2}\.\d{2})\*\*：(.+)。\n", summary)
        if not first or not first.group(2).strip():
            raise ValueError(
                f"摘要第 {number} 条第 1 段（A）不匹配；期望第一行形状为 "
                "**YYYY.MM.DD**：(A)。；本次没有写盘。\n"
                f"{INDEX_SUMMARY_FORMAT_GUIDE}")
        remainder = summary[first.end():]
        second = re.match(r"^(.+)。\*\*当下：", remainder)
        if not second or not second.group(1).strip():
            raise ValueError(
                f"摘要第 {number} 条第 2 段（B）不匹配；期望第二行先写 "
                "(B)。**当下：；本次没有写盘。\n"
                f"{INDEX_SUMMARY_FORMAT_GUIDE}")
        third = re.fullmatch(r"(.+)。\*\*$", remainder[second.end():])
        if not third or not third.group(1).strip():
            raise ValueError(
                f"摘要第 {number} 条第 3 段（C）不匹配；期望结尾形状为 "
                "(C)。**；本次没有写盘。\n"
                f"{INDEX_SUMMARY_FORMAT_GUIDE}")
        if first.group(1) != wanted_date:
            raise ValueError(f"摘要第 {number} 条日期应为 {wanted_date}；本次没有写盘。")
        if not isinstance(terms, list) or not terms:
            raise ValueError(f"摘要第 {number} 条 evidenceTerms 必须是非空数组；本次没有写盘。")
        unique, out_of_bounds = [], []
        for term in terms:
            if not isinstance(term, str) or not term or not any(_content_char(c) for c in term):
                raise ValueError(f"摘要第 {number} 条 evidenceTerms 含空值或纯标点。")
            if term not in unique:
                unique.append(term)
            missing = []
            if term not in summary:
                missing.append("summary")
            if term not in evidence_base:
                missing.append("text/current_state")
            if missing:
                out_of_bounds.append(f"{term}（缺在 {' 与 '.join(missing)}）")
        segments = (first.group(2), second.group(1), third.group(1))
        unanchored = [str(position) for position, segment in enumerate(segments, 1)
                      if not any(term in segment for term in unique)]
        if out_of_bounds or unanchored:
            over = "、".join(dict.fromkeys(out_of_bounds)) or "无"
            gaps = "、".join(unanchored) or "无"
            raise ValueError(f"摘要第 {number} 条未通过证据锚点校验。越界词：{over}；"
                             f"未锚定段：{gaps}。每个摘要段都要含至少一个同时存在于摘要与同批 "
                             "text/current_state 的 evidenceTerms 证据词；两边缺任一处都算越界。")
        validated.append(summary)
    return validated


def commit_memory_files(plans, replace_func=os.replace):
    """同目录临时文件逐个替换；多文件中途失败时尽力恢复调用前快照。"""
    snapshots, temps, committed = {}, {}, []
    try:
        for plan in plans:
            path = Path(plan["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            snapshots[path] = path.read_bytes() if path.exists() else None
            fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                handle.write(plan["content"])
                handle.flush()
                os.fsync(handle.fileno())
            temps[path] = Path(name)
        for plan in plans:
            path = Path(plan["path"])
            replace_func(temps[path], path)
            committed.append(path)
        return [Path(plan["path"]) for plan in plans]
    except Exception as original:
        rollback_errors = []
        for path in reversed(committed):
            try:
                before = snapshots[path]
                if before is None:
                    path.unlink(missing_ok=True)
                else:
                    fd, name = tempfile.mkstemp(prefix=f".{path.name}.rollback.", suffix=".tmp",
                                                dir=path.parent)
                    with os.fdopen(fd, "wb") as handle:
                        handle.write(before)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(name, path)
            except Exception as rollback:
                rollback_errors.append(f"{path}：{rollback}")
        detail = f"；回滚也失败：{' | '.join(rollback_errors)}" if rollback_errors else ""
        raise OSError(f"多文件写入失败：{original}{detail}") from original
    finally:
        for temp in temps.values():
            temp.unlink(missing_ok=True)


_CLEANUP_PREVIEW_LIMIT = 280


def _atomic_write_bytes(path, content, replace_func=os.replace):
    """同目录写临时文件后替换；清理备份、manifest 与回滚共用。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        replace_func(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def commit_cleanup_changes(changes, fail_after=None, replace_func=os.replace):
    """原子化执行“改写或删除”集合；任一步失败都恢复调用前字节快照。

    `changes` 是 `{Path: bytes|None}`；None 表示删除。`fail_after` 只给自检注入故障，
    生产不传。恢复的是全部目标，不只恢复已经走到的那几个——这样调用方不用猜失败落点。
    """
    changes = {Path(path): content for path, content in changes.items()}
    snapshots = {path: (path.read_bytes() if path.exists() else None) for path in changes}
    applied = 0
    try:
        for path, content in changes.items():
            if content is None:
                path.unlink(missing_ok=False)
            else:
                _atomic_write_bytes(path, content, replace_func=replace_func)
            applied += 1
            if fail_after is not None and applied >= int(fail_after):
                raise OSError(f"故障注入：第 {applied} 个清理目标后失败")
        return list(changes)
    except Exception as original:
        rollback_errors = []
        for path, before in snapshots.items():
            try:
                if before is None:
                    path.unlink(missing_ok=True)
                else:
                    _atomic_write_bytes(path, before)
            except Exception as rollback:
                rollback_errors.append(f"{path}：{rollback}")
        detail = f"；回滚也失败：{' | '.join(rollback_errors)}" if rollback_errors else ""
        raise OSError(f"清理写入失败：{original}{detail}") from original


def _record_section_plan(path, record_id):
    """按 recordId 反解到唯一 H2 记录节；未命中时返回 ``None``。

    早期 recordId 是切块文本的哈希。删掉同一窗口的前一条记录后，原本第二条会继承 H1
    前导文本，重切块后的哈希改变；因此同时按“现文件切块”与“不带前导文本的本节切块”
    查找。真正清理 sidecar 时只移除现文件仍在使用的哈希。
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"原始记录文件无法读取：{path}（{type(exc).__name__}）") from None
    starts = [match.start() for match in re.finditer(r"(?m)^## ", text)]
    if not starts:
        raise ValueError("目标文件没有可清理的 ## 记录节；只支持 latent_append 写下的记录")
    preamble = text[:starts[0]]
    candidates = []
    for number, start in enumerate(starts):
        end = starts[number + 1] if number + 1 < len(starts) else len(text)
        section = text[start:end]
        scoped = preamble + section if number == 0 else section
        chunks = chunk_heading(scoped)
        hashes = tuple(_chunk_key(chunk) for chunk in chunks)
        bare_hashes = tuple(_chunk_key(chunk) for chunk in chunk_heading(section))
        if record_id in set(hashes) | set(bare_hashes):
            candidates.append({"start": start, "end": end, "section": section,
                               "chunks": chunks, "hashes": hashes})
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError(f"recordId 在原文件中对应 {len(candidates)} 个记录节，定位不唯一，拒绝删除")
    target = candidates[0]
    if parse_record_time_marker(target["section"]) is None:
        raise ValueError("目标没有 Latent 写回时刻标记；只支持 latent_append 自己写下的记录，"
                         "普通导入语料请继续用 latent_correct 撤回")
    remaining = text[:target["start"]] + text[target["end"]:]
    target["new_bytes"] = remaining.encode("utf-8") \
        if re.search(r"(?m)^## ", remaining) else None
    return target


def _cleanup_confirmation(record_id, snapshots):
    digest = hashlib.sha256()
    digest.update(record_id.encode("ascii"))
    for path, content in sorted(snapshots.items(), key=lambda row: str(row[0])):
        digest.update(str(Path(path).resolve()).encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return f"DELETE-{record_id}-{digest.hexdigest()[:16]}"


def _cleanup_preview_text(section):
    text = re.sub(r"<!--\s*写于\s*[^>]+-->", "", section)
    text = " ".join(text.split())
    return text if len(text) <= _CLEANUP_PREVIEW_LIMIT else text[:_CLEANUP_PREVIEW_LIMIT] + "…"


class MemoryServer:
    """协议层与业务层的接线。持有 index 与 thread store，工具处理器一律薄转发。"""

    def __init__(self, index=None, thread_store=None, search_topN=5, recall_topN=3,
                 corpus_dir=None, weights_path=None, retractions_path=None,
                 entities_path=None, loader=None, time_context=None,
                 source_dirs=None, index_dir=None, startup_notice=None,
                 require_unresolved_review=False, write_dir=None):
        # 两个 topN 分开（2026.07.31 真实语料冒烟后拆的）：显式检索是用户/模型
        # 主动问一件事，多给几条值；开场召回每次换窗都付一遍，条数要克制
        self.index = index if index is not None else MemoryIndex().build()
        self.thread_store = thread_store if thread_store is not None else ThreadStore()
        self.corpus_dir = corpus_dir
        self.unresolved_store = UnresolvedStore(
            Path(corpus_dir) / UNRESOLVED_FILENAME if corpus_dir is not None else None)
        self.require_unresolved_review = bool(require_unresolved_review)
        self.search_topN = search_topN
        # 时区：没传就退固定东八区默认值（不会跟着宿主迁移），但 --doctor 会报 WARN。
        # ⚠ 这一份要穿到三处——写回算自然日、召回标日期、检索结果标日期，
        # 少接一处就会出现"当场对、重启错"或"文件对、标签错"的半拉形态
        self.time_context = time_context if time_context is not None else TimeContext.default()
        # stdio 客户端经常丢弃 stderr；启动提醒还要随 initialize instructions 进模型上下文，
        # 否则“打印过 warning”只在我们自己的终端里成立，用户仍然一次都看不见。
        review_notice = ("未解决复核策略：严格模式；完整 latent_append 与 latent_thread_close "
                         "缺 unresolvedOps 时会在核心写盘前拒绝。" if self.require_unresolved_review
                         else "未解决复核策略：兼容模式；旧调用缺 unresolvedOps 时核心写入继续，"
                              "清单回 unresolvedStatus=not_reviewed。")
        self.instructions = (INSTRUCTIONS + "\n" + review_notice
                             + (f"\n\n{startup_notice}" if startup_notice else ""))
        self.recall = SessionRecall(self.index, topN=recall_topN, thread_store=self.thread_store,
                                    time_context=self.time_context,
                                    unresolved_store=self.unresolved_store)
        self.initialized = False
        # 写回与权重持久化（任务卡"记忆写回与权重持久化"）：
        # corpus_dir 是写回的落点，没配就明确拒写；weights_path 没配则权重只活在
        # 内存里（selftest/临时用法），配了就启动时载入、每次检索命中后落盘
        self.index_dir = index_dir
        # write_dir：这支笔自己写的正文落哪个目录。没配就退回旧行为
        # <corpus>/timeline，配了就与语料里别处（宿主客户端自动记的那一档）物理
        # 分开。它只搬**写**的落点，读仍由 --corpus 递归覆盖——所以它必须落在
        # corpus 里面，否则会出现"写得进、检索读不到"这种不报错的失败
        self.write_dir = write_dir
        # source_dirs 只管读取与 HTTP 指纹；它可以包含独立索引目录，但绝不能替代
        # corpus_dir 成为写回落点。未传时退回旧行为，只监听主语料。
        default_dirs = [path for path in (corpus_dir, index_dir) if path is not None]
        self.source_dirs = tuple(source_dirs or default_dirs)
        self.weights_path = weights_path
        # 撤回账本（错误记忆治理闭环）：配了就启动时载入、每次撤回后落盘——
        # 不落盘的话"改过来了"只活一个进程，跟权重当初同一个坑
        self.retractions_path = retractions_path
        if retractions_path is not None:
            self.index.load_retractions(retractions_path)
        if weights_path is not None:
            self.index.load_weights(weights_path)
        # 实体标注（图谱可插拔升级）：语料目录下有 .entities.json 就接上——
        # 实体边在 build 时算，接上后要重建一次索引才生效
        self.entities_path = entities_path
        if entities_path is not None and self.index.load_entities(entities_path):
            self.index.build()
        # loader：怎么从盘上重建索引（常驻 HTTP 形态的重读用）。stdio 懒加载
        # 每次调用重读语料所以用不上；不传则常驻形态检测到语料变化时明确不重读
        self.loader = loader
        # 本进程自己写过哪些语料文件（常驻 HTTP 形态的指纹记账用）。
        # 为什么要精确到路径而不是"handle 之后重算一遍指纹"：那样会把**同一个请求
        # 窗口里用户手动上传的 md 一起吞进基线**，那次上传再也不触发重读——
        # 正好是自动重读这个特性要治的静默形态（2026.08.03 外部评审指出的竞态）
        self.written_paths = set()

    def _reload_from_disk(self):
        """语料目录有文件级变化时从盘上重建索引（常驻 HTTP 形态专用）。

        治的是 supergateway 时代那条实测过的静默坑：常驻 server 只在启动时读一次
        语料，手动上传进目录的 md 一条都看不到、不报错，直到重启。能这么重建的
        前提是**内存里没有只活在内存的账**：写回落盘（append_record）、撤回落盘
        （每次 correct 后 save_retractions）、权重落盘（每次 search 后 save_weights），
        重建后按各自的 sidecar 路径重新接上即可，无损。
        SessionRecall 只换 index 引用，会话内状态（距上次召回的字数）保留。"""
        self.index = self.loader()
        if self.retractions_path is not None:
            self.index.load_retractions(self.retractions_path)
        if self.weights_path is not None:
            self.index.load_weights(self.weights_path)
        if self.entities_path is not None and self.index.load_entities(self.entities_path):
            self.index.build()
        self.recall.index = self.index

    # ---------- 七个工具：协议接线；未解决的解析与状态变化仍在独立模块 ----------

    def _tool_memory_search(self, args, now=None):
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolError("query 必填且不能为空")
        variant = args.get("queryVariant")
        if variant is not None and (not isinstance(variant, str) or not variant.strip()):
            raise ToolError("queryVariant 若提供，必须是非空字符串")
        queries = [query] + ([variant] if variant is not None else [])
        top_n = int(args.get("topN", self.search_topN))
        results = self.index.retrieve_queries(queries, topN=top_n) \
            if variant is not None else self.index.retrieve(query, topN=top_n)
        if self.weights_path is not None:
            # 用进落盘：retrieve 的副作用是命中块 +weight_boost，不落盘的话
            # server 一重启就归零——权重持久化的"存"这半就在这一行
            self.index.save_weights(self.weights_path)
        miss_rate = min(query_miss_rate(self.index, q) for q in queries)
        note = miss_rate_note(miss_rate)
        if not results:
            # 可靠命中门槛（验收反馈）：低相关不硬凑。这句要同时做两件事——
            # 说清"查过了、真没有"，并明确解锁如实回答（instructions 堵的是
            # "没查就说没记录"，查过之后的"没有"是诚实，不是那句被堵的话术）
            retry = ("你已经带 queryVariant 重试过了——第二次仍没找到，请停止检索并"
                     "如实告诉对方没找到/记不清，不要拿沾边内容凑答案。") \
                if variant is not None else \
                ("你已经查过一次；最多再试一次：保留原 query，把更具体的说法放进 "
                 "queryVariant（人名、物件、地点或当时可能用过的词）。")
            raise ToolError("没有可靠命中：记忆库里没有与这个说法词面或语义相关的记录。"
                            + retry + (" " + note if note else ""))
        # 缺失率标注（2026.08.01，第二份外部反馈标定）：**不改变返回什么**，
        # 只在结果后面附一句可核对的话。真实威胁是"库里没有却返回了五条真记忆、
        # 模型拿去圆"，而这个判断只有读到内容的模型能下——机制层负责把不确定性
        # 摆到台面上，不负责替它拒绝。
        # 拼接走 annotate_block（库函数），外壳仍然只是转发+组合调用，不自己拼字符串
        return annotate_block(format_recall_block(results, time_context=self.time_context),
                              miss_rate)

    @staticmethod
    def _session_start_failure_reason(exc):
        """把读取异常归到用户能行动的方向，不回显可能含正文的异常消息。"""
        if isinstance(exc, FileNotFoundError):
            return "文件不存在或路径错误"
        if isinstance(exc, PermissionError):
            return "文件不可读（权限不足）"
        if isinstance(exc, json.JSONDecodeError):
            return "会话线索文件不是合法 JSON"
        if isinstance(exc, UnicodeError):
            return "文件编码读取失败"
        return "文件读取失败"

    def _tool_session_start(self, args, now=None):
        try:
            block = self.recall.on_session_start(now=now)
        except (OSError, UnicodeError, json.JSONDecodeError) as first:
            reason = self._session_start_failure_reason(first)
            print(f"session_start 第一次{reason}，自动重试：{type(first).__name__}",
                  file=sys.stderr)
            try:
                block = self.recall.on_session_start(now=now)
            except (OSError, UnicodeError, json.JSONDecodeError) as second:
                second_reason = self._session_start_failure_reason(second)
                print(f"session_start 重试后仍{second_reason}：{type(second).__name__}",
                      file=sys.stderr)
                raise ToolError(
                    "session_start 读取记忆失败：自动重试 1 次后仍失败（首次 "
                    f"{type(first).__name__}，重试 {type(second).__name__}；{second_reason}）。"
                    "服务端没有返回空结果，而是执行失败。请检查 --corpus、--threads "
                    "指向的路径、文件可读性与服务端日志。") from None
        if block is None:
            raise ToolError("记忆库是空的，没有可召回的内容")
        return block

    def _unresolved_after_core(self, args, source):
        """核心写入后的清单复核；失败分类可见，但绝不反咬已经成功的正文／thread。"""
        if "unresolvedOps" not in args:
            return ("unresolvedStatus=not_reviewed：本次旧式调用没有提供 unresolvedOps，"
                    "核心写入已成功，未解决清单没有经过复核。")
        try:
            status, changed = self.unresolved_store.apply(args.get("unresolvedOps"), source)
        except UnresolvedRequestError as exc:
            return ("unresolvedStatus=blocked；失败类型=request_invalid："
                    f"{exc}。核心写入已成功；请修正参数后只用 latent_unresolved 重试，"
                    "不要重复 append 正文。")
        except UnresolvedStoreError as exc:
            return ("unresolvedStatus=blocked；失败类型=store_unavailable："
                    f"{exc}。这是服务端清单文件的问题，不是本次正文的问题；"
                    "请先修复清单，再只用 latent_unresolved 重试，不要重复 append 正文。")
        detail = f"（{'、'.join(changed)}）" if changed else ""
        return f"unresolvedStatus={status}{detail}。"

    def _require_unresolved_field(self, args):
        if self.require_unresolved_review and "unresolvedOps" not in args:
            raise ToolError(
                "服务器启用了 --require-unresolved-review：本次尚未写盘。请补 "
                "unresolvedOps；无变化也要传 [{\"action\":\"none\"}]。")

    def _preflight_unresolved(self, args):
        """只验证未解决操作能否应用；不生成 ID、不调用 apply、不写 sidecar。"""
        if "unresolvedOps" not in args:
            if self.require_unresolved_review:
                raise ToolError(_append_input_error(
                    "服务器启用了 --require-unresolved-review：预检未通过。请补 unresolvedOps；"
                    "无变化也要明确传 none。", "unresolved"))
            return "not_reviewed"
        try:
            normalized = validate_ops(args.get("unresolvedOps"))
            current = self.unresolved_store.read()
        except UnresolvedRequestError as exc:
            raise ToolError(_append_input_error(str(exc), "unresolved")) from None
        except UnresolvedStoreError as exc:
            raise ToolError(f"未解决清单无法预检：{exc}") from None
        if normalized == [{"action": "none"}]:
            return "reviewed"
        if self.unresolved_store.path is None:
            raise ToolError("未解决清单无法预检：服务器没有配置未解决清单路径")
        known = {item.id for item in current}
        for op in normalized:
            if op["action"] == "open":
                continue
            item_id = op["id"]
            if item_id not in known:
                raise ToolError(_append_input_error(
                    f"没有找到未解决事项 {item_id}", "unresolved"))
            if op["action"] == "close":
                known.remove(item_id)
        return "would_update"

    @staticmethod
    def _plan_append_indexes(args, body_plan, index_write_dir):
        """完整写回的索引计划；预检与真写共用，缺证据是合法 pending。"""
        if args.get("indexEvidence") is not None and args.get("indexSummaries") is not None:
            raise ValueError("indexEvidence 与旧字段 indexSummaries 只能选一个")
        if args.get("indexEvidence") is not None:
            evidence = validate_index_evidence(
                args.get("indexEvidence"),
                [args.get("text") or "", args.get("current_state") or ""])
            summaries = [render_index_evidence(evidence, body_plan["local_date"])]
            record_id = body_plan["record_id"]
        elif args.get("indexSummaries") is not None:
            summaries = validate_index_summaries(
                args.get("indexSummaries"), args.get("text") or "",
                args.get("current_state") or "", body_plan["local_date"])
            record_id = None
        else:
            return [], "pending"
        return (plan_index_records(
            index_write_dir, summaries, body_plan["window"], body_plan["local_date"],
            record_id=record_id), "indexed")

    def _tool_memory_append_preflight(self, args, now, index_write_dir, fallback_index):
        """复用真写计划但停在 commit 前；返回值明确声明零写入。"""
        record_id_arg = args.get("recordId")
        if record_id_arg is not None:
            if args.get("text") is not None or args.get("current_state") is not None:
                raise ToolError(_append_input_error(
                    "补索引预检只传 recordId＋indexEvidence；不要重复 text 或 current_state",
                    "repair"))
            if not isinstance(record_id_arg, str) or not re.fullmatch(r"[0-9a-f]{16}", record_id_arg):
                raise ToolError(_append_input_error(
                    "recordId 必须是此前 latent_append 返回的 16 位小写十六进制标识",
                    "repair"))
            if list(index_write_dir.glob(f"*_record_{record_id_arg}_item_*.md")):
                return (f"预检通过：preflightStatus=ready；recordId={record_id_arg} 已有索引，"
                        "indexStatus=indexed；本次零写入，无需重复补索引。")
            matches = [(chunk, meta) for chunk, meta in zip(self.index.chunks, self.index.meta)
                       if meta.get("layer", "timeline") == "timeline"
                       and _chunk_key(chunk) == record_id_arg]
            if len(matches) != 1:
                reason = "没有找到" if not matches else "找到多条"
                raise ToolError(_append_input_error(
                    f"按 recordId={record_id_arg} {reason}原始记录；"
                    "正文可能被手工改过，请先 latent_search 定位，不要猜着补索引",
                    "repair"))
            record_text, record_meta = matches[0]
            try:
                evidence = validate_index_evidence(args.get("indexEvidence"), [record_text])
                local_date = record_meta.get("local_date") or self.time_context.local_date(now)
                summary = render_index_evidence(evidence, local_date)
                plans = plan_index_records(
                    index_write_dir, [summary], record_meta.get("window") or 0,
                    local_date, record_id=record_id_arg)
            except ValueError as exc:
                raise ToolError(_append_input_error(str(exc), "evidence")) from None
            except OSError as exc:
                raise ToolError(str(exc)) from None
            return (f"预检通过：preflightStatus=ready；补索引 recordId={record_id_arg}；"
                    f"预计写入 {len(plans)} 个索引文件；indexStatus=indexed；本次零写入。")

        try:
            self._require_unresolved_field(args)
        except ToolError as exc:
            raise ToolError(_append_input_error(str(exc), "unresolved")) from None
        try:
            body_plan = plan_append_record(
                self.corpus_dir, args.get("text") or "", args.get("current_state") or "",
                window=args.get("window"), now=now, time_context=self.time_context,
                write_dir=self.write_dir)
        except ValueError as exc:
            raise ToolError(_append_input_error(str(exc), "full")) from None
        except OSError as exc:
            raise ToolError(str(exc)) from None
        category = "exclusive" if (args.get("indexEvidence") is not None
                                   and args.get("indexSummaries") is not None) \
            else ("evidence" if args.get("indexEvidence") is not None else "summary")
        try:
            index_plans, index_status = self._plan_append_indexes(
                args, body_plan, index_write_dir)
        except ValueError as exc:
            raise ToolError(_append_input_error(str(exc), category)) from None
        except OSError as exc:
            raise ToolError(str(exc)) from None
        unresolved_status = self._preflight_unresolved(args)
        pending = ("；写入后需只传 recordId＋indexEvidence 补索引"
                   if index_status == "pending" else "")
        hint = (f"；未配置 --index-dir，预计使用兼容路径 {index_write_dir}"
                if fallback_index else "")
        return (f"预检通过：preflightStatus=ready；预计写入第 {body_plan['window']} 个窗口"
                f"（{body_plan['path'].name}），recordId={body_plan['record_id']}；"
                f"预计索引文件={len(index_plans)}；indexStatus={index_status}{pending}；"
                f"unresolvedStatus={unresolved_status}{hint}；本次零写入。")

    def _tool_memory_append(self, args, now=None):
        mode = args.get("mode", "write")
        if mode not in {"write", "preflight"}:
            raise ToolError(_append_input_error(
                "mode 只能是 write 或 preflight；省略时默认 write", "mode"))
        if self.corpus_dir is None:
            # 不静默写进内存了事：内存态的"记住了"会随进程一起死，那是
            # "失败得像成功"——宁可让模型看到明确的失败原因
            raise ToolError("服务器没有配置可写的语料目录（--corpus），写不了回。"
                            "这条记忆不会被保存，请提醒用户检查 MCP 配置。")
        fallback_index = self.index_dir is None
        index_write_dir = Path(self.index_dir) if self.index_dir is not None \
            else Path(self.corpus_dir) / "index"
        if mode == "preflight":
            return self._tool_memory_append_preflight(
                args, now, index_write_dir, fallback_index)
        record_id_arg = args.get("recordId")
        if record_id_arg is not None:
            if args.get("text") is not None or args.get("current_state") is not None:
                raise ToolError(_append_input_error(
                    "补索引模式只传 recordId＋indexEvidence；不要重复 text 或 current_state",
                    "repair"))
            if not isinstance(record_id_arg, str) or not re.fullmatch(r"[0-9a-f]{16}", record_id_arg):
                raise ToolError(_append_input_error(
                    "recordId 必须是此前 latent_append 返回的 16 位小写十六进制标识",
                    "repair"))
            if list(index_write_dir.glob(f"*_record_{record_id_arg}_item_*.md")):
                return f"recordId={record_id_arg} 已有索引，未重复写入。indexStatus=indexed。"
            matches = [(chunk, meta) for chunk, meta in zip(self.index.chunks, self.index.meta)
                       if meta.get("layer", "timeline") == "timeline"
                       and _chunk_key(chunk) == record_id_arg]
            if len(matches) != 1:
                reason = "没有找到" if not matches else "找到多条"
                raise ToolError(_append_input_error(
                    f"按 recordId={record_id_arg} {reason}原始记录；"
                    "正文可能被手工改过，请先 latent_search 定位，不要猜着补索引",
                    "repair"))
            record_text, record_meta = matches[0]
            try:
                evidence = validate_index_evidence(args.get("indexEvidence"), [record_text])
                summary = render_index_evidence(
                    evidence, record_meta.get("local_date") or self.time_context.local_date(now))
                index_plans = plan_index_records(
                    index_write_dir, [summary], record_meta.get("window") or 0,
                    record_meta.get("local_date") or self.time_context.local_date(now),
                    record_id=record_id_arg)
                paths = commit_memory_files(index_plans)
            except ValueError as e:
                raise ToolError(_append_input_error(
                    f"原始正文仍在，但索引补写失败：{e}", "evidence")) from None
            except OSError as e:
                raise ToolError(f"原始正文仍在，但索引补写失败：{e}") from None
            if self.loader is not None:
                self._reload_from_disk()
            else:
                self.index = load_corpus(self.source_dirs).build()
                self.recall.index = self.index
            self.written_paths.update(str(path) for path in paths)
            hint = (f" 未配置 --index-dir，索引已落到兼容路径 {index_write_dir}；"
                    "推荐配置独立 --index-dir。") if fallback_index else ""
            return (f"已为 recordId={record_id_arg} 补写 1 条原文证据索引。"
                    f"indexStatus=indexed。{hint}")
        try:
            self._require_unresolved_field(args)
        except ToolError as exc:
            raise ToolError(_append_input_error(str(exc), "unresolved")) from None
        try:
            body_plan = plan_append_record(
                self.corpus_dir, args.get("text") or "", args.get("current_state") or "",
                window=args.get("window"), now=now, time_context=self.time_context,
                write_dir=self.write_dir)
        except ValueError as e:
            raise ToolError(_append_input_error(str(e), "full")) from None
        except OSError as e:
            raise ToolError(str(e)) from None
        try:
            index_plans, index_status = self._plan_append_indexes(
                args, body_plan, index_write_dir)
            if index_status == "pending":
                raise ValueError("未提供 indexEvidence（旧客户端可继续提供 indexSummaries）")
            paths = commit_memory_files([body_plan] + index_plans)
        except (ValueError, OSError) as e:
            # 索引摘要是检索辅助，不是原始事实。结构或锚点失败时仍保留正文与当下状态，
            # 让后续检索回到可核对的原记录；绝不把未验摘要写进 index 层。
            try:
                paths = commit_memory_files([body_plan])
            except OSError as write_error:
                raise ToolError(str(write_error))
            if self.loader is not None:
                self._reload_from_disk()
            else:
                self.index = load_corpus(self.source_dirs).build()
                self.recall.index = self.index
            self.written_paths.update(str(path) for path in paths)
            unresolved = self._unresolved_after_core(
                args, f"timeline/{paths[0].name}#record={body_plan['record_id']}")
            return (f"正文已写进第 {body_plan['window']} 个窗口（{paths[0].name}），"
                    f"recordId={body_plan['record_id']}。索引尚未写入：{e} "
                    "indexStatus=pending；不要重复正文，之后用本工具传 recordId＋indexEvidence 补齐。 "
                    + unresolved)
        # 写完立刻进内存索引并重建，本会话的 latent_search 就能查到——
        # 不然"我记下了"之后当场问它还查不到，模型会顺势说"没有记录"
        if self.loader is not None:
            self._reload_from_disk()
        else:
            self.index = load_corpus(self.source_dirs).build()
            self.recall.index = self.index
        self.written_paths.update(str(path) for path in paths)
        hint = (f" 未配置 --index-dir，摘要已落到兼容路径 {index_write_dir}；"
                "推荐配置独立 --index-dir。") if fallback_index else ""
        unresolved = self._unresolved_after_core(
            args, f"timeline/{paths[0].name}#record={body_plan['record_id']}")
        return (f"已写进第 {body_plan['window']} 个窗口（{paths[0].name}），"
                f"recordId={body_plan['record_id']}，并写入 {len(index_plans)} 条索引摘要。"
                f"indexStatus=indexed。{hint} {unresolved}")

    def _tool_memory_correct(self, args, now=None):
        correction = args.get("correction")
        if correction and not (args.get("current_state") or "").strip():
            raise ToolError("写 correction 时 current_state（当下状态）必填——"
                            "病灶迁移，同 latent_append：更正这件事现在是什么状态？")
        # 撤回先做——它同时是 quote 的校验关卡；quote 定位不到就该在写任何东西
        # 之前失败（否则更正落了盘、旧记录还在，比什么都没做更糟）
        try:
            old_idx, _ = self.index.retract(args.get("quote") or "",
                                            args.get("reason") or "", now=now)
        except ValueError as e:
            raise ToolError(str(e))
        msg = "已撤回那段旧记录：检索不会再返回它（原文件保留，撤回原因入账可追溯）。"
        if correction:
            if self.corpus_dir is None:
                # 撤回已生效但更正写不进去——明确报出来，不静默丢（同 append 的理由）
                if self.retractions_path is not None:
                    self.index.save_retractions(self.retractions_path)
                raise ToolError(msg + " 但服务器没有配置可写的语料目录（--corpus），"
                                "更正内容写不进去，请提醒用户检查 MCP 配置。")
            try:
                path, chunk_text, meta = append_record(
                    self.corpus_dir, f"【更正】{correction}",
                    args.get("current_state") or "", now=now,
                    time_context=self.time_context, write_dir=self.write_dir)
            except (ValueError, OSError) as e:
                if self.retractions_path is not None:
                    self.index.save_retractions(self.retractions_path)
                raise ToolError(msg + f" 但更正内容写入失败：{e}")
            self.index.add(chunk_text, meta)
            correction_idx = len(self.index.chunks) - 1   # 刚 append 的更正块，排除它
            self.index.build()
            self.written_paths.add(str(path))    # 同 latent_append，见 __init__ 注释
            # 追溯链补上：让账本能回答"哪条记录改了哪条"，不只是"这条被撤了"。
            # 只能在更正写完之后回填——新块的内容哈希这时才存在
            self.index.link_correction(old_idx, chunk_text)
            msg += f" 更正已写进第 {meta['window']} 个窗口（{path.name}）。"
        else:
            correction_idx = None
        # 撤回粒度提醒（岔口 2）：撤回是块级的、事实是跨块的——刚撤的只是这一块，
        # 同一说法若散在别的记录里，那些块不受影响；主动检索通常问不到它们，但开场
        # 自动浮现按时间/权重排、没有 query，旧说法可能被直接端出来。这里顺手扫一遍，
        # 命中未撤的其它块就把缺口在**用户改错的当下**说出来。
        # ⚠ 只提示、绝不自动撤：区分性词面命中 ≠ 讲同一件事，误撤没兜底（same_claim_
        # survivors 的 docstring 写死了这条边界）。exclude 传更正块，否则它含同样词面
        # 会命中自己、提示恒真＝等于没提示（坑 1）。
        survivors = self.index.same_claim_survivors(
            old_idx, exclude=() if correction_idx is None else (correction_idx,))
        if survivors:
            labels = []
            for i in survivors[:_SURVIVOR_PREVIEW]:
                m = self.index.meta[i]
                labels.append(m.get("heading") or m.get("source") or f"块{i}")
            more = "" if len(survivors) <= _SURVIVOR_PREVIEW \
                else f"（另有 {len(survivors) - _SURVIVOR_PREVIEW} 条未列出）"
            msg += (f" ⚠ 但撤回是块级的：库里还有 {len(survivors)} 块提到同一说法"
                    f"（最近：{'、'.join(labels)}{more}），它们**没被这次撤回影响**。"
                    "主动检索通常问不到它们，但开场自动浮现没有 query、按时间和权重排，"
                    "旧说法可能被直接端出来。要彻底改掉，请用旧说法再 latent_search 一次、"
                    "把还提到它的记录逐条撤掉——服务端只提示、不会替你自动撤"
                    "（词面命中不等于讲的是同一件事，撤错比给旧值更糟）。")
        if self.retractions_path is not None:
            self.index.save_retractions(self.retractions_path)
        return msg

    def _cleanup_plan(self, record_id):
        """把稳定 recordId 收敛成一份精确、可复算的删除计划；这里只读。"""
        if self.corpus_dir is None:
            raise ToolError("服务器没有配置可写的语料目录（--corpus），不能清理记录")
        if not isinstance(record_id, str) or not re.fullmatch(r"[0-9a-f]{16}", record_id):
            raise ToolError("recordId 必须是 latent_append 返回的 16 位小写十六进制标识")

        timeline_root = (Path(self.write_dir) if self.write_dir is not None
                         else Path(self.corpus_dir) / "timeline").resolve()
        if not timeline_root.is_dir():
            raise ToolError("写回目录不存在，找不到可清理记录")
        candidates = []
        for source in sorted(timeline_root.glob("window_*.md")):
            try:
                section_plan = _record_section_plan(source, record_id)
            except ValueError as exc:
                raise ToolError(str(exc)) from None
            if section_plan is not None:
                candidates.append((source, section_plan))
        if len(candidates) != 1:
            reason = "没有找到" if not candidates else f"找到 {len(candidates)} 条"
            raise ToolError(f"按 recordId={record_id} {reason}原始记录；"
                            "记录可能已清理或被手工修改，请重新检索，不要猜着删除")
        source, section_plan = candidates[0]

        index_root = Path(self.index_dir) if self.index_dir is not None \
            else Path(self.corpus_dir) / "index"
        linked_index = sorted(index_root.glob(f"*_record_{record_id}_item_*.md")) \
            if index_root.is_dir() else []
        unresolved_items = self.unresolved_store.read()
        blocker_ids = [item.id for item in unresolved_items
                       if record_id in source_record_ids([item])]

        removed_hashes = set(section_plan["hashes"])
        for path in linked_index:
            try:
                removed_hashes.update(_chunk_key(chunk) for chunk in chunk_heading(
                    path.read_text(encoding="utf-8")))
            except (OSError, UnicodeError) as exc:
                raise ToolError(f"关联索引无法读取：{path.name}（{type(exc).__name__}）") from None

        changes = {source: section_plan["new_bytes"]}
        changes.update({path: None for path in linked_index})
        for sidecar in (self.weights_path, self.retractions_path, self.entities_path):
            if sidecar is None:
                continue
            path = Path(sidecar)
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ToolError(f"清理前无法读取 {path.name}（{type(exc).__name__}），拒绝改盘") from None
            if not isinstance(data, dict):
                raise ToolError(f"清理前发现 {path.name} 不是对象结构，拒绝改盘")
            pruned = {key: value for key, value in data.items() if key not in removed_hashes}
            if len(pruned) != len(data):
                changes[path] = (json.dumps(pruned, ensure_ascii=False, indent=1) + "\n").encode("utf-8")

        snapshots = {}
        for path in changes:
            try:
                snapshots[path] = path.read_bytes()
            except OSError as exc:
                raise ToolError(f"清理前无法读取 {path.name}（{type(exc).__name__}），拒绝改盘") from None
        return {
            "record_id": record_id,
            "source": source,
            "section": section_plan["section"],
            "linked_index": linked_index,
            "blocker_ids": blocker_ids,
            "changes": changes,
            "snapshots": snapshots,
            "confirmation": _cleanup_confirmation(record_id, snapshots),
        }

    def _write_cleanup_backup(self, plan, reason, now=None):
        """先把本次会改动的全部原始字节移入隐藏隔离区，再允许碰活动文件。"""
        now = time.time() if now is None else now
        stamp = datetime.fromtimestamp(now, _dt_tz.utc).strftime("%Y%m%dT%H%M%SZ")
        root = Path(self.corpus_dir) / ".cleanup-trash"
        backup_dir = root / f"{plan['record_id']}-{stamp}"
        suffix = 1
        while backup_dir.exists():
            backup_dir = root / f"{plan['record_id']}-{stamp}-{suffix:02d}"
            suffix += 1
        backup_dir.mkdir(parents=True, exist_ok=False)
        entries = []
        try:
            corpus_root = Path(self.corpus_dir).resolve()
            for number, (path, content) in enumerate(
                    sorted(plan["snapshots"].items(), key=lambda row: str(row[0])), 1):
                backup_name = f"{number:02d}-{Path(path).name}.bak"
                _atomic_write_bytes(backup_dir / backup_name, content)
                resolved = Path(path).resolve()
                try:
                    original = str(resolved.relative_to(corpus_root)).replace("\\", "/")
                except ValueError:
                    original = str(resolved)
                entries.append({
                    "original": original,
                    "backup": backup_name,
                    "sha256": hashlib.sha256(content).hexdigest(),
                })
            manifest = {
                "recordId": plan["record_id"],
                "reason": reason,
                "deletedAtUtc": datetime.fromtimestamp(now, _dt_tz.utc).isoformat(),
                "entries": entries,
            }
            _atomic_write_bytes(
                backup_dir / "manifest.json",
                (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        except Exception as exc:
            raise ToolError(f"隔离备份写入失败，活动文件尚未改动：{type(exc).__name__}") from None
        return backup_dir

    def _tool_memory_cleanup(self, args, now=None):
        action = args.get("action")
        if action not in {"preview", "delete"}:
            raise ToolError("action 只能是 preview 或 delete")
        plan = self._cleanup_plan(args.get("recordId"))
        blockers = "、".join(plan["blocker_ids"])
        if action == "preview":
            blocker_note = (f"；当前被未解决事项引用：{blockers}，关闭或改写引用前不能删除"
                            if blockers else "；未发现未解决事项引用")
            return (f"清理预览：recordId={plan['record_id']}；来源={plan['source'].name}；"
                    f"关联索引={len(plan['linked_index'])} 个{blocker_note}。\n"
                    f"目标内容：{_cleanup_preview_text(plan['section'])}\n"
                    f"确认令牌：{plan['confirmation']}\n"
                    "确认目标无误后，再用 action=delete、同一 recordId、逐字 confirmation 和 reason。")

        confirmation = args.get("confirmation")
        reason = args.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ToolError("delete 模式 reason 必填：说明为什么清理，供审计追溯")
        if not isinstance(confirmation, str) or confirmation != plan["confirmation"]:
            raise ToolError("confirmation 不匹配：文件或关联状态可能已变化。请重新 preview，"
                            "不要沿用旧令牌或猜令牌")
        if plan["blocker_ids"]:
            raise ToolError(f"记录仍被未解决事项引用（{blockers}）；请先用 latent_unresolved "
                            "update／close，再重新 preview")

        backup_dir = self._write_cleanup_backup(plan, reason.strip(), now=now)
        try:
            changed = commit_cleanup_changes(plan["changes"])
        except OSError as exc:
            raise ToolError(f"清理失败，活动文件已尝试回滚；隔离备份保留在 {backup_dir}。{exc}") from None
        if self.loader is not None:
            self._reload_from_disk()
        else:
            self.index = load_corpus(self.source_dirs).build()
            if self.retractions_path is not None:
                self.index.load_retractions(self.retractions_path)
            if self.weights_path is not None:
                self.index.load_weights(self.weights_path)
            if self.entities_path is not None and self.index.load_entities(self.entities_path):
                self.index.build()
            self.recall.index = self.index
        self.written_paths.update(str(path) for path in changed)
        return (f"已精准清理 recordId={plan['record_id']}：正文记录 1 条、关联索引 "
                f"{len(plan['linked_index'])} 个；隔离备份与审计 manifest 位于 {backup_dir}。")

    def _tool_thread_close(self, args, now=None):
        self._require_unresolved_field(args)
        now = time.time() if now is None else now
        try:
            thread = close_thread(
                window=args.get("window"),
                started_at=float(args.get("started_at", now - 3600)),
                ended_at=now,
                topics=args.get("topics") or (),
                current_state=args.get("current_state") or "",
                open_loops=args.get("open_loops") or (),
            )
            self.thread_store.append(thread)
        except Exception as e:                      # 业务校验失败 → 工具执行错误
            raise ToolError(str(e))
        source = (f"thread/window_{thread.window:02d}@"
                  f"{datetime.fromtimestamp(thread.ended_at, _dt_tz.utc).isoformat()}")
        unresolved = self._unresolved_after_core(args, source)
        return f"已记下第 {thread.window} 个窗口的收尾，下个窗口会带回来。 {unresolved}"

    def _tool_unresolved(self, args, now=None):
        source = args.get("source") or "manual"
        op = {key: value for key, value in args.items()
              if key in {"action", "id", "summary"}}
        try:
            status, changed = self.unresolved_store.apply([op], source)
        except UnresolvedRequestError as exc:
            raise ToolError(f"unresolvedStatus=blocked；失败类型=request_invalid：{exc}")
        except UnresolvedStoreError as exc:
            raise ToolError(f"unresolvedStatus=blocked；失败类型=store_unavailable：{exc}")
        detail = f"：{'、'.join(changed)}" if changed else ""
        return f"unresolvedStatus={status}{detail}。"

    def _handlers(self):
        return {
            "latent_search": self._tool_memory_search,
            "latent_session_start": self._tool_session_start,
            "latent_append": self._tool_memory_append,
            "latent_correct": self._tool_memory_correct,
            "latent_cleanup": self._tool_memory_cleanup,
            "latent_unresolved": self._tool_unresolved,
            "latent_thread_close": self._tool_thread_close,
        }

    # ---------- 协议层 ----------

    def handle(self, msg, now=None):
        """一条 JSON-RPC 消息 → 一条响应（通知类返回 None，规格要求不回响应）。"""
        method, mid = msg.get("method"), msg.get("id")
        params = msg.get("params") or {}

        if method == "initialize":
            self.initialized = True
            return self._ok(mid, {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": SERVER_INFO,
                "instructions": self.instructions,
            })
        if method == "notifications/initialized":
            return None                              # 通知不回响应（规格：JSON-RPC 通知无 id）
        if method == "tools/list":
            return self._ok(mid, {"tools": TOOLS})
        if method == "tools/call":
            return self._call_tool(mid, params, now=now)
        if mid is None:
            return None                              # 其它通知一律忽略，不回错
        return self._err(mid, E_METHOD_NOT_FOUND, f"未知 method：{method}")

    def _call_tool(self, mid, params, now=None):
        name = params.get("name")
        handler = self._handlers().get(name)
        if handler is None:
            # 规格"Error Handling"：未知工具属协议错误，不是 isError 结果
            return self._err(mid, E_METHOD_NOT_FOUND, f"未知工具：{name}")
        args = params.get("arguments")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return self._err(mid, E_INVALID_PARAMS, "arguments 必须是对象")
        try:
            text = handler(args, now=now)
        except ToolError as e:
            # 工具执行错误：按规格回正常结果 + isError，让模型看得到失败原因
            return self._ok(mid, {"content": [{"type": "text", "text": str(e)}], "isError": True})
        except Exception as e:
            # 最终错误边界：不让工具内部异常冲破 HTTP/stdio，客户端只看见空白。
            # 不打印 e 的正文——底层异常可能夹带记忆内容或请求参数。
            print(f"工具执行异常（{name}）：{type(e).__name__}", file=sys.stderr)
            message = (f"{name} 执行失败（{type(e).__name__}）。服务端没有返回空结果，"
                       "而是工具执行异常；请查看服务端日志并检查相关文件与配置。")
            return self._ok(mid, {"content": [{"type": "text", "text": message}],
                                  "isError": True})
        return self._ok(mid, {"content": [{"type": "text", "text": text}], "isError": False})

    @staticmethod
    def _ok(mid, result):
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    @staticmethod
    def _err(mid, code, message):
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}

    def serve_stdio(self, stdin=None, stdout=None):
        """stdio 传输：一行一条 JSON（行分隔），读到 EOF 退出。
        解析不了的行直接跳过——没有 id 就没法回错，回了反而污染流。

        **必须显式按 UTF-8 收发**（2026.07.31 真机实测出来的 bug，不是洁癖）：
        MCP 规格定死 stdio 传输是 UTF-8，但 `sys.stdin` 在 Windows 上按系统区域
        编码解码（简中默认 cp936）。症状极隐蔽——不报错、不崩，中文 query 变成
        乱码后分词一个都匹配不上，BM25 与向量层分数全平，检索退化成"按加载顺序
        返回前几块"，看起来像是检索质量差，实际上根本没查。"""
        stdin = stdin or _utf8_text_stream(sys.stdin.buffer)
        stdout = stdout or _utf8_text_stream(sys.stdout.buffer, write=True)
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            resp = self.handle(msg)
            if resp is not None:
                stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
                stdout.flush()


# ---------- HTTP 传输（Streamable HTTP，2026.08.03） ----------
#
# 为什么要有这条：闭源前端（Kelivo/Operit 这类）只认 Streamable HTTP，此前用户
# 只能拿 npm 的 supergateway 把 stdio 桥一层。外部实测撞出来的四个坑全在桥上：
# ① supergateway 服务端模式**没有任何入站鉴权**——latent_search 读全库、
#    latent_append 可写入，端口被扫到记忆库就既可读又可写；
# ② 默认 SSE 输出与客户端的 Streamable HTTP 不匹配（`Transport disconnected`）；
# ③ 进程僵死（在但端口不通）要 pkill 重启；
# ④ 常驻只在启动时读一次语料，手动加的 md 静默看不到。
# 原生实现把四个一起治：鉴权内置且非回环裸跑直接拒绝起动、只说 Streamable HTTP、
# 少一个常驻进程、语料变化自动重读（_reload_from_disk）。
# 零依赖不破：http.server 是标准库。TLS 不在这做——闭源前端不认自签证书
# （两家独立实测），公网仍然要域名 + 反代（Caddy）那条路，反代顺手把 TLS 管了。
#
# 规格面（刻意做小，都是规格允许的服务器侧选择）：
#   POST 单条 JSON-RPC 消息 → 按客户端 `Accept` 决定形态：默认 application/json
#     单条响应；客户端**明确只要** text/event-stream 时包成单事件 SSE；通知 → 202 空身；
#   GET（SSE 长流）→ 给一条**只发心跳、永不发消息**的空长流（规格允许服务器在这条流上
#     什么都不发）；客户端明确只要 JSON、或 --no-sse-stream 关掉时才回 405；
#   DELETE（会话终止）→ 200 空操作，本 server 无会话态、不发 Mcp-Session-Id；
#   JSON 数组（批量）→ 400（批量已从 2025-06 规格移除）。
#
# **为什么要看 `Accept`**（2026.08.07，`HTTP传输不看Accept` 那张卡）：
# 规格在这两处都把选择权留给服务器，而此前我们两处都按自己方便挑了一种、
# 并**默认客户端会配合**——GET 恒 405（赌它会自己改用 POST）、POST 恒回
# application/json（赌它认 JSON）。两个赌注**赢面大、输了完全静默**：
# 不回退的客户端卡在等长流上，只认 SSE 响应体的客户端拿到 JSON 直接接不上，
# 而后者是 2xx，`_deny` 够不着，**服务端日志上一个字都没有**。
# ⚠ 「客户端会回退」这条我们只在一家客户端上见过一次，**一次不构成通则**。
# 改法不是去猜哪个客户端会怎样，是**不再赌**：`Accept` 本来就是客户端用来说
# 「我认什么」的，照着它给。GET 那条长流对我们一条消息都不会走
# （没有服务器主动推送的场景），它存在的意义只是「不让等它的人白等」。

_HTTP_BODY_LIMIT = 10 * 1024 * 1024     # 单请求上限：正常一条 tools/call 远小于此
# 被拒的请求最多吞多少 body 来保住 keep-alive（见 _drain）。超过就作废连接——
# 声明了超大长度的客户端多半根本没打算发完，等它等于把线程挂死
_HTTP_DRAIN_LIMIT = 64 * 1024
_HTTP_DRAIN_TIMEOUT = 5.0
# 被拒请求留痕的刷屏防线（见 _make_deny_logger）：一个窗口里最多逐条打多少行、
# 窗口有多长。绑了 0.0.0.0 就会被扫，401/404 能刷满终端、撑爆重定向到的文件
_HTTP_DENY_LOG_MAX = 60
_HTTP_DENY_LOG_WINDOW = 60.0
# GET 空长流：心跳间隔与并发上限。
# ⚠ **心跳不是可选项**：手机运营商网络与反代（nginx `proxy_read_timeout` 默认 60s）
# 会把长时间没有字节的连接静默掐掉，而「被掐掉」跟「服务端没实现」在客户端那头
# 长得一模一样——正是这张卡在治的那种分不清。
# ⚠ **上限也不是可选项**：ThreadingHTTPServer 一条连接一个线程，手机切后台/回前台
# 反复重连会把线程堆起来，而这些线程各自都在 sleep，不会自己走
_HTTP_SSE_HEARTBEAT = 15.0
_HTTP_SSE_MAX_STREAMS = 4
# 长流线程的轮询粒度：盯 socket 用，跟心跳无关。取小是为了 shutdown() 别拖
_HTTP_SSE_POLL = 0.5
# 405 的三句补注（见 _log_deny 的 note 参数）。**同一个状态码现在有三种含义**，
# 不许再糊成一句：原来那句只在第一格是对的，另外两格照抄它就是把真故障说成不是故障
_NOTE_405_FALLBACK = "客户端通常会自动改用 POST，这不是错误"
_NOTE_405_OFF = ("SSE 长流被 --no-sse-stream 关掉了；"
                 "客户端若不自动改用 POST，去掉这个开关")
_NOTE_405_BUSY = f"SSE 长流已开满（上限 {_HTTP_SSE_MAX_STREAMS} 条），这一条没给"


def _accept_set(header):
    """把 `Accept` 头拆成 media type 集合（小写、去掉参数、`q=0` 的丢掉）。

    ⚠ **没带头返回 `None`，不是空集合**：「客户端没说」与「说了但不要」
    在 `do_GET` 里会走向**相反**的分支（没说 → 给长流，说了但不要 → 405），
    合并成空集合就分不出来了。"""
    if header is None:
        return None
    out = set()
    for part in header.split(","):
        bits = part.strip().split(";")
        mt = bits[0].strip().lower()
        if not mt:
            continue
        q = 1.0
        for p in bits[1:]:
            k, _, v = p.partition("=")
            if k.strip().lower() == "q":
                try:
                    q = float(v.strip())
                except ValueError:
                    q = 1.0
        if q > 0:
            out.add(mt)
    return out


def _accept_has(types, mime):
    """`_accept_set` 的结果里认不认这个 media type（含 `*/*` 与 `type/*`）。

    `types is None`（客户端没带 `Accept`）一律返回 False——**「没说」不算「要」**，
    要不要因此给它什么，由调用点自己判断。"""
    if not types:
        return False
    return (mime in types or "*/*" in types
            or f"{mime.split('/')[0]}/*" in types)


def _log_sanitize(s, limit):
    """要写进日志的那截字符串：不可打印字符换成 `?`，超长截断。

    客户端能塞进请求行的东西不受我们控制（路径里放 `\\r\\n` 就能伪造出一整行
    假日志，放几 KB 就能把终端顶满），而这行日志的用处正是让人一眼看清，
    所以进日志前一律过这道。"""
    s = "".join(ch if ch.isprintable() else "?" for ch in str(s))
    return s if len(s) <= limit else s[:limit] + "…"


def _make_deny_logger():
    """造一个「被拒请求留痕」的记录函数（每个 HTTP server 一份自己的计数器）。

    **为什么要有它**（2026.08.04 立卡）：八种拒绝（401/403/404/405/411/413/400/501）
    的原因 `_deny()` 每条都写得很清楚，但那句话只进**响应体**，而多数 MCP 客户端
    不显示响应体，用户看到的是「Transport disconnected」之类的通用失败；服务端这边
    `log_message` 整个静音，于是**两头都不说话，八种原因在用户那里长成同一个样子**。

    ⚠ **`log_message` 的静音不掀掉**：它静音的理由（个人部署逐请求刷屏是纯噪声）
    依然成立，这里只在它旁边开一个窄口子——**非 2xx 才留痕，正常请求照旧不打**。

    刷屏防线：一个窗口里最多逐条打 `_HTTP_DENY_LOG_MAX` 行，超出的压掉。
    ⚠ **压掉的条数必须说出来**——「压掉但不说」等于用一个新的静默去修一个旧的
    静默，那就白做了。所以窗口结束时补一行「过去一分钟另有 X 条被拒未逐条列出」，
    而且是**定时器兜底**：不能靠「下一个被拒请求来的时候顺手补」，那样一阵扫描
    打完就没下文了的话，那 X 条就真的没人说了。
    ⚠ ThreadingHTTPServer 是多线程的，计数器配自己的一把小锁（不复用 index 那把，
    那把锁的是整段检索，等它等于把留痕挂在业务后面）。"""
    lock = threading.Lock()
    st = {"t0": None, "n": 0, "dropped": 0, "timer": None}

    def _settle_locked():
        """窗口翻页：清计数，返回要补打的那行（没压掉东西就 None）。调用方持锁。"""
        if st["timer"] is not None:
            st["timer"].cancel()
            st["timer"] = None
        dropped, st["dropped"] = st["dropped"], 0
        st["t0"], st["n"] = None, 0
        return f"过去一分钟另有 {dropped} 条被拒未逐条列出" if dropped else None

    def _on_timeout():
        with lock:
            line = _settle_locked()
        if line:
            print(line, file=sys.stderr)

    def emit(line):
        out = []
        with lock:
            now = time.monotonic()
            if st["t0"] is None or now - st["t0"] >= _HTTP_DENY_LOG_WINDOW:
                closing = _settle_locked()
                if closing:
                    out.append(closing)
                st["t0"], st["n"] = now, 0
            if st["n"] < _HTTP_DENY_LOG_MAX:
                st["n"] += 1
                out.append(line)
            else:
                st["dropped"] += 1
                if st["timer"] is None:
                    t = threading.Timer(
                        max(0.0, st["t0"] + _HTTP_DENY_LOG_WINDOW - now), _on_timeout)
                    t.daemon = True      # 别让这只表拖住进程退出
                    st["timer"] = t
                    t.start()
        for x in out:
            print(x, file=sys.stderr)

    return emit


def _is_loopback(host):
    """绑定地址是不是回环。`localhost` 走名字判断，其余按 IP 解析；解析不了的
    （域名、空串）一律当非回环——**判不准时往严的那边倒**：那样守卫会要求 token、
    横幅会多打一行提醒，两个方向的错都是"多提醒一次"，不是"漏掉一次"。"""
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host in ("localhost",)


def timezone_default_notice(effective_timezone, detected_timezone):
    """没显式配置时区、且宿主探测值与固定默认值不同时给出的启动提醒。

    **只提醒，不自动采用探测值**：宿主机器在哪儿，不等于记忆所有者在哪儿；直接拿
    VPS 时区当用户时区，会把“固定但可能要改”的配置换成“迁机器就悄悄变”的配置。
    探测失败或名字一致时不吵。调用方负责只在 CLI／环境变量都没给时调用。"""
    if not detected_timezone or detected_timezone == effective_timezone:
        return None
    return (f"⚠ 没配 --timezone：宿主本地时区探测为 {detected_timezone}，"
            f"实际仍使用固定默认值 {effective_timezone}。宿主时区只供核对，不会自动"
            "替代记忆所有者时区；请按本人所在时区显式加 --timezone <IANA 时区名>。")


def http_tls_notice(host):
    """非回环绑定时该多打的那一行（回环返回 None）。纯函数，起动横幅那边调。

    **为什么要有这一行**（2026.08.04 外部实测促成）：`http_bind_guard` 挡的是
    "非回环裸跑不配 token"，**它不挡"没有 TLS"**——配了 token 就照常起动，于是
    用户走在一条裸 HTTP 暴露公网的路上，token 与 `latent_search` 返回的记忆内容
    全程明文，而**没有任何人提醒过他**。那位用户自己提的建议是"检测到 token 已配
    就提示绑定地址"，方向要反过来：token 已配说明他知道要鉴权，**真正没被提醒的
    是 TLS**。

    ⚠ **只提示，不拒绝起动**：拒绝会把"临时绑 `0.0.0.0` 验证一下"这条路堵死，
    而那正是他这次定位根因的办法（怀疑过 SSE 格式、怀疑过桥接残留，最后是改绑定
    地址才连上）。
    ⚠ **不看 token**：token 与 TLS 是两件事，配了 token 也照样提醒——做成"有 token
    就不提示"等于把唯一一次提醒的机会送给最不需要它的人。"""
    if _is_loopback(host):
        return None
    return (f"⚠ 绑的是非回环地址（{host}）：这条链路没有 TLS，token 与 "
            f"latent_search 返回的记忆内容在网上都是明文，拿到 token 的人既能读"
            f"全库、也能写进去。推荐绑 127.0.0.1、公网那一跳交给反向代理"
            f"（Caddy／nginx）管 TLS；只是临时验证的话，至少上防火墙限制来源，"
            f"并把这个 token 当已经泄露过来处理。")


def http_bind_guard(host, token):
    """起动前的两道守卫：token 必须是 ASCII；非回环地址必须有 token。都是拒绝起动。

    **token 非 ASCII 也拒绝**（2026.08.03 外部实测）：HTTP 头是 latin-1，中文口令
    客户端根本发不出去（`UnicodeEncodeError`），就算发出去 `hmac.compare_digest`
    对非 ASCII str 直接抛 `TypeError` → 500。而失败形态是**服务看起来起来了、
    横幅也打了，就是连不上**——用户拿中文当口令是很可能的事，这正是本项目最怕的
    静默形态，所以在起动这一步就拦掉，跟下面绑定守卫同一个待遇。

    supergateway 那条路的头号坑就是"先跑通再说"地对公网开无鉴权端口——跑通那一刻
    记忆库已经暴露。这里把它做成出不了门的形态（同 guidance_text 超长拒绝出货的
    思路：失败要响，不静默）。回环上裸跑放行——本机 proot／反代在前的形态，
    鉴权在外层。"""
    if token:
        try:
            token.encode("ascii")
        except UnicodeEncodeError:
            raise ValueError(
                "token 只能用 ASCII 字符（字母／数字／符号）：HTTP 头按 latin-1 编码，"
                "中文之类的口令客户端根本发不出去，而服务照样起得来——失败形态是"
                "「看起来起来了、就是连不上」。换一串英文数字口令再起。")
        return
    if not _is_loopback(host):
        raise ValueError(
            f"拒绝在非回环地址（{host}）上裸跑：没配 token 的 HTTP 端口等于把记忆库"
            f"公开成既可读又可写。加 --token（或环境变量 MEMORY_HTTP_TOKEN），"
            f"或绑回 127.0.0.1 由带鉴权的反向代理转发。")


def _corpus_signature(source_dirs):
    """全部读取根的文件级指纹（路径+mtime+大小）。只看 .md——sidecar（.weights.json
    每次检索都落盘）进指纹的话每个请求都会触发假重读。

    ⚠ glob 到 stat 之间文件可能已被删（并发整理语料），**漏掉它就行、不许抛**
    （2026.08.03 外部评审：这里抛 FileNotFoundError 会变成 500）——文件没了本身
    就是一次语料变化，指纹里少一条正好让下一次比较发现它。"""
    out = []
    for p in corpus_files(source_dirs):
        try:
            st = p.stat()
        except OSError:
            continue
        out.append((str(p), st.st_mtime_ns, st.st_size))
    return tuple(out)


def make_http_server(server, host="127.0.0.1", port=8765, token=None,
                     sse_stream=True):
    """MemoryServer → 绑好端口的 ThreadingHTTPServer（不启动；.serve_forever() 是
    调用方的事——selftest 要拿着实例开线程、取真端口、shutdown）。

    `sse_stream`：GET 要不要给一条空长流（默认给，理由见上面那段注释）。
    关掉就退回「GET 恒 405」的旧行为，留给「长流反而碍事」的部署。"""
    http_bind_guard(host, token)
    lock = threading.Lock()                  # index 的增删改建都不是线程安全的：串行化
    deny_log = _make_deny_logger()           # 被拒请求留痕（带刷屏上限，见那个函数）
    state = {"sig": _corpus_signature(server.source_dirs)
             if (server.source_dirs and server.loader) else None}
    # 开着的 GET 空长流：计数（配上限用）与「服务端要停了」的信号。
    # ⚠ **`closing` 不能省**：长流线程平时蹲在 `wait(心跳间隔)` 上，没有这个信号
    # 就得等一个心跳周期才发现服务停了——selftest 里 `shutdown()` 会跟着变慢，
    # 真部署里 Ctrl-C 之后终端要挂十几秒
    sse_lock = threading.Lock()
    sse_open = {"n": 0}
    closing = threading.Event()

    class _Server(http.server.ThreadingHTTPServer):
        def shutdown(self):
            closing.set()                    # 先叫长流线程收摊，再停 accept 循环
            super().shutdown()

        def handle_error(self, request, client_address):
            """客户端中途断开不许喷 traceback（跟 log_message 静音同一个理由）。

            实测（2026.08.03 外部评审复验时量的）：客户端发到一半就走人——手机 App
            切后台、断网的常态——socketserver 默认往 stderr 打完整 Traceback，
            约 41 行一次。而 --http 的起动横幅也走 stderr，运维盯着终端会以为
            服务崩了。断连类只吞掉；**真异常保留一行摘要**，不静默成"什么都没发生"。"""
            exc = sys.exc_info()[1]
            if isinstance(exc, (ConnectionError, TimeoutError)):
                return
            print(f"HTTP 请求处理异常（{client_address[0]}）："
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)

    class _Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):           # 默认逐请求刷 stderr，个人部署只是噪声
            pass

        def _send(self, code, body=b"", ctype="application/json; charset=utf-8",
                  extra=()):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in extra:
                self.send_header(k, v)
            self.end_headers()
            if body:
                self.wfile.write(body)

        def _drain(self):
            """把没读的请求体吞掉，返回是否吞干净了。

            **不吞就是下一个请求必坏**（2026.08.03 外部实测）：HTTP/1.1 默认
            keep-alive，401 直接返回时残留的 body 会被当成下一个请求的**请求行**
            解析，于是「先探一次拿 401 → 带 token 重试」这个再普通不过的姿势，
            第二发收到 400 + HTML 错误页。手机 App 复用连接是常态，而 selftest 里
            urllib 每次新连接、发 `Connection: close`——**这一格自检天然盖不住，
            判据必须走裸 socket**（见 18b）。

            ⚠ **吞不动的时候绝不能傻等**（第一版修法自己踩的坑）：413 那格客户端
            多半根本没打算发完那么多字节，`read(n)` 会把这个线程挂死；chunked、
            长度不合法同理。所以三种情况直接判连接作废：读不动、大到不值得读、
            读超时。作废走的是显式 `Connection: close` 响应头，不是只在服务端
            改个标志——客户端得知道这条连接不能再用了。"""
            te = self.headers.get("Transfer-Encoding")
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = -1
            if te or n < 0 or n > _HTTP_DRAIN_LIMIT:
                return False
            old = self.connection.gettimeout()
            try:
                self.connection.settimeout(_HTTP_DRAIN_TIMEOUT)
                while n > 0:
                    chunk = self.rfile.read(min(n, 8192))
                    if not chunk:
                        return False
                    n -= len(chunk)
            except OSError:                  # 超时／对端断了：连接作废，不挂线程
                return False
            finally:
                try:
                    self.connection.settimeout(old)
                except OSError:
                    pass
            return True

        def _log_deny(self, code, msg, note=None):
            """被拒的请求在 stderr 上留一行（八条拒绝全走 `_deny`，所以落点只有这一处）。

            形如：`被拒 404 GET /nope 来自 127.0.0.1：没有这个端点`
            ——状态码、方法、**剥掉 query 的**路径、客户端 IP、原因短语。

            `note` 是**调用点写死的一句常量补注**，原样缀在行尾（不过原因短语那道
            截断）。⚠ **只许传常量，永远不许把客户端来的值塞进 `note`**——下面
            三条硬边界与「变量插值放分隔符之后」那条规矩对它同样作数，而 `note`
            恰恰绕过了截断这道保险。（服务端自己的配置常量算常量，
            `_NOTE_405_BUSY` 里那个上限数字就是这么来的。）
            目前唯一的调用点是 405（见 `do_GET`），但**同一个码有三种含义、
            三句补注**（`_NOTE_405_FALLBACK` / `_NOTE_405_OFF` / `_NOTE_405_BUSY`）
            ——只有第一种是「这不是错误」。

            ⚠ **三条硬边界（「为了让人能排查」而写进日志，正好是明文那张卡在防的
            事的另一个出口——日志会被重定向进文件、会被贴进 issue 找人帮看）**：

            1. **`Authorization` 的任何内容都不打**，一个字符都不打。401 只记
               「缺少或错误的 Bearer token」，不记收到的是什么——「只打前 4 位帮
               用户对一下」也不行。
            2. **不打请求体**。400「请求体不是合法的 UTF-8 JSON」时最想附上的就是
               那段 body，**恰恰最不能附**：那里面是 `latent_search` 的问句和
               `latent_append` 的正文，是用户最私密的记忆。413 同理。
            3. **路径剥掉 query string**：`_deny(404, ...)` 把整个 `self.path` 回给
               对方没问题（是他自己发的），但**写进日志就落盘了**，而有些客户端
               习惯把 token 塞 `?token=`。所以这里用 `urlsplit(self.path).path`，
               **不用 `self.path`**。

            原因短语取 `msg` 的**第一个分隔符之前**那截：各调用点把客户端来的值
            （路径、Origin、`Transfer-Encoding` 的值）都插在分隔符之后，切掉即可。
            ⚠ **以后新增 `_deny` 调用点，变量插值必须放在「：」「: 」「——」之后**，
            否则它会跟着原因短语进日志。"""
            reason = msg
            for sep in ("：", ": ", "——", "\n"):
                reason = reason.split(sep, 1)[0]
            path = urllib.parse.urlsplit(self.path).path
            ip = self.client_address[0] if self.client_address else "?"
            deny_log(f"被拒 {code} {_log_sanitize(self.command or '?', 16)} "
                     f"{_log_sanitize(path, 120)} 来自 {_log_sanitize(ip, 60)}："
                     f"{_log_sanitize(reason, 80)}"
                     + (f"（{note}）" if note else ""))

        def _deny(self, code, msg, extra=(), note=None):
            self._log_deny(code, msg, note=note)
            extra = list(extra)
            if not self._drain():
                self.close_connection = True
                extra.append(("Connection", "close"))
            self._send(code, json.dumps({"error": msg}, ensure_ascii=False)
                       .encode("utf-8"), extra=extra)

        def _gate(self):
            """路径 + 鉴权 + Origin 三道闸。过了返回 True。"""
            # 路径校验（2026.08.03 外部评审：文档教人填 /mcp，实际任何路径都收，
            # 无害但跟文档不一致）。放行根路径是给"填地址时漏了 /mcp"留的余地——
            # 那是最常见的手滑，收下比回 404 让人排查半天强
            if urllib.parse.urlsplit(self.path).path.rstrip("/") not in ("", "/mcp"):
                self._deny(404, f"没有这个端点：{self.path}——本 server 只在 /mcp 上收")
                return False
            if token:
                got = self.headers.get("Authorization") or ""
                if not hmac.compare_digest(got, f"Bearer {token}"):
                    self._deny(401, "缺少或错误的 Bearer token",
                               extra=[("WWW-Authenticate", "Bearer")])
                    return False
            # Origin 校验（规格的 DNS rebinding 防线）：App 客户端不发 Origin，
            # 发了且不是本机的只会是浏览器页面在拿本地端口当跳板
            origin = self.headers.get("Origin")
            if origin:
                h = (urllib.parse.urlsplit(origin).hostname or "").lower()
                if h not in ("localhost", "127.0.0.1", "::1"):
                    self._deny(403, f"Origin 不被信任：{origin}")
                    return False
            return True

        def _sse_idle_stream(self):
            """一条只发心跳、永不发消息的 SSE 长流，直到对端走人或服务端要停。

            **不带 Content-Length、也不 chunked**，所以这条连接是「读到关闭为止」
            的——必须显式 `close_connection`，否则 keep-alive 那套会跟它打架
            （HTTP/1.1 下没有长度又不关连接，客户端会一直等下一条响应的头）。

            **首字节 `: ok` 立刻发**：给客户端一个「流真的建起来了」的凭据。
            差这一下的话，「建好了但没消息」跟「还在连」在客户端那头分不出来，
            而分不出来正是这张卡在治的病。`:` 开头是 SSE 的注释行，
            任何合规的解析器都会安静地丢掉它，不会被当成一条消息。

            ⚠ **不能只蹲在「睡够一个心跳再写一次」上**（第一版就是那么写的）：
            对端断开之后，这条线程要等到下一次心跳写失败才发现，
            名额（`_HTTP_SSE_MAX_STREAMS`）跟着被占住一整个心跳周期。手机切后台／
            回前台反复重连时，客户端看到的就是**明明没人在用却回 405**。
            所以这里同时盯着三样：socket 可读（对端关了写端＝走人）、心跳到点、
            服务端要停——轮询粒度取心跳与 `_HTTP_SSE_POLL` 的较小者。"""
            self.close_connection = True
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            sock = self.connection
            try:
                self.wfile.write(b": ok\n\n")
                self.wfile.flush()
                last = time.monotonic()
                while not closing.is_set():
                    poll = min(_HTTP_SSE_POLL, _HTTP_SSE_HEARTBEAT)
                    ready, _, _ = select.select([sock], [], [], poll)
                    if ready:
                        # 空读＝对端关了写端，这条流没人要了，立刻把名额还回去。
                        # 读到东西反倒不算断开（不该有，但忽略掉比误判成断开安全）
                        if not sock.recv(4096):
                            break
                        continue
                    now = time.monotonic()
                    if now - last >= _HTTP_SSE_HEARTBEAT:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        last = now
            except OSError:
                pass          # 对端走了：正常收场，不是异常（同 handle_error 的理由）

        def do_GET(self):
            if not self._gate():
                return
            deny = ("本 server 不提供 SSE 长流，请用 POST（Streamable HTTP）",
                    [("Allow", "POST, DELETE")])
            if not sse_stream:
                return self._deny(405, deny[0], extra=deny[1], note=_NOTE_405_OFF)
            # ⚠ 「没带 Accept」与「带了但不要 SSE」必须分开判，别图省事合并：
            # 前者是懒客户端，没说要什么、却多半正等着长流；
            # 后者明确说了只收 JSON，给它长流反而是把一条本来好的路弄坏。
            types = _accept_set(self.headers.get("Accept"))
            if not (types is None or _accept_has(types, "text/event-stream")):
                return self._deny(405, deny[0], extra=deny[1], note=_NOTE_405_FALLBACK)
            with sse_lock:
                if sse_open["n"] >= _HTTP_SSE_MAX_STREAMS:
                    return self._deny(405, "SSE 长流已开满，请用 POST（Streamable HTTP）",
                                      extra=deny[1], note=_NOTE_405_BUSY)
                sse_open["n"] += 1
            try:
                self._sse_idle_stream()
            finally:
                with sse_lock:
                    sse_open["n"] -= 1

        def do_DELETE(self):
            if not self._gate():
                return
            self._send(200)                  # 无会话态，终止是空操作

        def do_POST(self):
            if not self._gate():
                return
            # chunked 明确不支持，**且报错要指对方向**（2026.08.03 外部实测）：
            # 原先不带 Content-Length 时 length 取 0、读到空串，报的是
            # 「请求体不是合法的 UTF-8 JSON」——把人往"我的 JSON 写错了"引，
            # 而真因是传输编码不匹配。这跟 supergateway 时代那个「默认 SSE 与
            # 客户端不匹配」是同形物：传输层没对上，报错还指错地方。
            # 501 是规格给这种情况的码（服务器不支持该功能）。
            if self.headers.get("Transfer-Encoding"):
                return self._deny(
                    501, f"不支持 Transfer-Encoding: "
                         f"{self.headers.get('Transfer-Encoding')}——本 server 只收带 "
                         f"Content-Length 的整条 JSON-RPC 消息。这是传输编码不匹配，"
                         f"不是你的 JSON 写错了；客户端若有「分块传输/流式上传」开关，"
                         f"关掉即可。")
            if self.headers.get("Content-Length") is None:
                return self._deny(411, "缺 Content-Length：本 server 只收带长度的整条消息")
            try:
                length = int(self.headers.get("Content-Length"))
                if length < 0:
                    raise ValueError
            except ValueError:
                return self._deny(400, "Content-Length 不合法")
            if length > _HTTP_BODY_LIMIT:
                return self._deny(413, "请求体超限")
            try:
                msg = json.loads(self.rfile.read(length).decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return self._deny(400, "请求体不是合法的 UTF-8 JSON")
            if isinstance(msg, list):
                return self._deny(400, "不支持 JSON-RPC 批量（2025-06 规格已移除）")
            if not isinstance(msg, dict):
                return self._deny(400, "请求体要是一条 JSON-RPC 消息对象")
            with lock:
                # 常驻形态的重读：语料目录文件级变化（手动上传/删除 md）→ 从盘重建。
                # ⚠ **指纹要在 handle 之后重算、而且只认自己写的那次变化**
                # （2026.08.03 外部评审指出的竞态）：原先 handle 后无条件刷新指纹，
                # 于是「一次 latent_append 与用户手动上传落在同一个请求窗口」时，
                # 那次上传会被自己的刷新永久吞掉、再也不触发重读——正好是这个特性
                # 要治的静默形态。改法：handle 前后各取一次，**只把 server 自己写回
                # 造成的差异并进基线**（after 与 before 的差集里属于自己写回的部分
                # 无从区分，所以退一步取更安全的一侧：若 handle 期间指纹又变了，
                # 保留 before，让下一个请求去发现并重读——宁可多重读一次，
                # 不可漏掉一次）。
                if state["sig"] is not None:
                    sig = _corpus_signature(server.source_dirs)
                    if sig != state["sig"]:
                        server._reload_from_disk()
                        state["sig"] = sig
                resp = server.handle(msg)
                if state["sig"] is not None and server.written_paths:
                    # **只折进 server 自己写的那几个文件**，其余差异留着让下一个
                    # 请求去发现——手动上传就落在"其余"里，不会被自己的刷新吞掉。
                    # 顺带：只读请求（latent_search 是绝大多数）written_paths 是空的，
                    # 这里整个跳过，不再每个请求 stat 全库两遍
                    base = {p: (m, s) for p, m, s in state["sig"]}
                    current = {p: (m, s) for p, m, s in _corpus_signature(server.source_dirs)}
                    for path in server.written_paths:
                        if path in current:
                            base[path] = current[path]
                        else:
                            # latent_cleanup 会删除活动文件；旧基线也得去掉该路径，
                            # 否则下一次请求会把自己的删除误判成外部变化、多重读一遍。
                            base.pop(path, None)
                    state["sig"] = tuple(sorted((p, m, s) for p, (m, s) in base.items()))
                    server.written_paths = set()
            if resp is None:
                return self._send(202)       # 通知：202 空身（规格）
            body = json.dumps(resp, ensure_ascii=False).encode("utf-8")
            # 规格允许服务器对 POST 回 JSON 或回 SSE，二选一。默认 JSON，
            # **只在客户端明确只要 SSE 时**才包成单事件 SSE——有的客户端解析器
            # 只会拆 `event:/data:`，收到 application/json 就接不上，而这一格
            # 是 2xx，`_deny` 够不着，服务端日志上一个字都没有。
            # ⚠ **回归护栏**：`Accept` 里同时有 application/json（规格要求客户端
            # 两个都写）、或有 `*/*`、或压根没带头，一律照旧回 JSON。已经跑通的
            # 客户端一个都不许被这条带坏——判据 2、3 守着这句话。
            types = _accept_set(self.headers.get("Accept"))
            if (_accept_has(types, "text/event-stream")
                    and not _accept_has(types, "application/json")):
                # 单事件 SSE：发完这一条就把流收了（规格说响应发完即可关流），
                # 所以照旧带 Content-Length，keep-alive 不受影响
                self._send(200, b"event: message\ndata: " + body + b"\n\n",
                           ctype="text/event-stream; charset=utf-8",
                           extra=[("Cache-Control", "no-store")])
            else:
                self._send(200, body)

    return _Server((host, port), _Handler)


# ---------- 部署体检（任务卡"部署体检命令"） ----------
#
# 挂在 mcp_server.py 上、不另开脚本文件：配 MCP 的人手里已经有这个文件的绝对路径
# （`claude mcp add` 那行就是它），体检命令跟着它走，等于零新增路径要记。
#
# **只读是硬约束，不是习惯**：体检最可能被在"看起来没接通"的时候跑，那时候人对
# 目录状态的信任最脆弱；跑一次体检往语料目录里掉一个 `.weights.json`，等于在
# 排查故障的现场留下一个新变量。所以这里一个字节都不写盘——包括 sidecar，
# 包括 embed 档的块向量缓存。这条由 selftest 的目录快照断言守着（第 14 项）。

OK, WARN, FAIL = "ok", "warn", "fail"
_DOCTOR_ICON = {OK: "✓", WARN: "⚠", FAIL: "✗"}

# 不该出现在语料目录里的 md：它们是产出目录那一层的东西。撞见就说明 --corpus
# 多指了一层（指到了产出目录本身，而不是里面的记忆库）——人格文件会被当成记忆
# 吃进库里，检索结果里冒出自己的人格设定，但不报任何错
_NOT_CORPUS_MD = {"claude.md", "agents.md", "persona.md", "readme.md",
                  "注入契约.md", "index_readme.md"}
# mtime 兜底占比超过这条线就报警：时间戳全落 mtime 时换窗召回的新鲜度排序整个失效
_MTIME_WARN_RATIO = 0.2
# 「只有标题行、没有正文」的块占到这个比例就报警——兜的是"第三方导出把每句消息
# 前面加 `##`"那类输入（见《导出格式把每句话切成一块》任务卡）。`chunk_heading`
# 有上界没有下界，这种语料建库成功、块数报得出来、检索也跑得动，只是每个块都是
# 一句话，**静默地没有检索价值**，还顺带把图谱那条平方曲线推过 MCP 启动超时。
#
# 判据**刻意不选"块长中位数低于 N 字"**：那是要拿真实语料标定的经验值（同
# MILESTONE_BODY_LIMIT 的教训），手上只有"正常"和"病态"两个极端、中间没有样本，
# 定不出来。"标题行就是整个块"是结构判据，不用标定：按小节切出来的块本该有正文。
# 这条线取"过半"是语义边界不是调出来的数——本仓三份真实 md 语料量到 0% / 3.6%
# / 11.6%，每句 `##` 的导出形态是 100%，两边离这条线都远得很。
_HEADING_ONLY_WARN_RATIO = 0.5


def retrieval_dirs(corpus_dir, index_dir=None):
    """主语料与可选人工索引目录 → 绝对读取根。"""
    roots = [Path(corpus_dir).resolve()]
    if index_dir is not None:
        index_root = Path(index_dir).resolve()
        if not index_root.exists():
            raise FileNotFoundError(f"索引摘要目录不存在：{index_root}")
        if not index_root.is_dir():
            raise NotADirectoryError(f"索引摘要路径不是目录：{index_root}")
        roots.append(index_root)
    return tuple(dict.fromkeys(roots))


def make_corpus_loader(corpus_dir, index_dir=None, embed=False, provider=None):
    """构造双根只读 loader；缓存与所有写侧状态仍只落主语料。"""
    source_dirs = retrieval_dirs(corpus_dir, index_dir)
    cache_path = Path(corpus_dir) / ".embed_cache.json" if embed else None

    def loader():
        return load_corpus(source_dirs, embed=embed, provider=provider,
                           cache_path=cache_path)

    return source_dirs, loader


def diagnose(corpus_dir, threads_path=None, embed=False, time_context=None,
             index_dir=None, require_unresolved_review=False, write_dir=None):
    """体检语料目录与接线，返回 [{level,title,detail}, ...]。**纯读，不写盘。**

    检的是"配好了没有"，不是"检索好不好"——后者归回归集（regression_set.py）。
    每一条都对应一种**不报错的失败**：指错目录、人格文件混进语料、时间戳全落
    mtime、sidecar 哈希对不上、写回落点只读、thread 没落点。这些的共同点是
    服务照常起、握手照常成功、模型照常回话，只是回得不对。"""
    out = []
    def add(level, title, detail):
        out.append({"level": level, "title": title, "detail": detail})

    # **相对路径必须解析成绝对路径**（判据 1，也是这一单的来源）：`.mcp.json` 里写的
    # 是 `--corpus corpus` 这种相对值，跑体检的人心里的问题正是"它到底去哪儿读了"。
    # 把他自己写的那个词原样回显给他，等于一个字没答——报告里必须出现真实绝对路径。
    # ⚠ 别把 .resolve() 去掉（selftest 第 15 项走真进程守着这条）
    root = Path(corpus_dir).resolve()
    if not root.exists():
        add(FAIL, "语料目录", f"{root} 不存在。服务端认的是 --corpus 传进去的这个路径，"
                              "跟目录叫什么名字无关——先确认 MCP 配置里那行路径写对了没有。")
        return out
    if not root.is_dir():
        add(FAIL, "语料目录", f"{root} 不是目录。--corpus 要指向记忆库那一层目录，不是单个文件。")
        return out

    add(OK, "未解决复核策略",
        "严格模式：缺 unresolvedOps 时核心写盘前拒绝；操作／文件失败仍保证正文或 thread 必达。"
        if require_unresolved_review else
        "兼容模式：缺 unresolvedOps 时核心写入继续，清单回 unresolvedStatus=not_reviewed。")
    unresolved_path = root / UNRESOLVED_FILENAME
    unresolved_for_doctor = []
    if not unresolved_path.exists():
        add(OK, "未解决清单", f"{unresolved_path} 尚未生成（正常：第一次 open 时创建）")
    else:
        try:
            unresolved, bad_lines = UnresolvedStore(unresolved_path).read(allow_partial=True)
            unresolved_for_doctor = unresolved
        except UnresolvedStoreError as exc:
            add(FAIL, "未解决清单", str(exc))
        else:
            total_chars = sum(len(item.summary) for item in unresolved)
            longest = max((len(item.summary) for item in unresolved), default=0)
            detail = (f"{unresolved_path}：有效 {len(unresolved)} 条，摘要共 {total_chars} 字，"
                      f"最长 {longest} 字")
            if bad_lines:
                add(FAIL, "未解决清单", detail + "；格式错误或重复 ID 行："
                    + "、".join(map(str, bad_lines)))
            else:
                add(OK, "未解决清单", detail)

    corpus_only_files = corpus_files(root)  # 递归，子目录里的 md 也算
    if not corpus_only_files:
        # 典型是指到了 src/ 或路径写岔了一格。给候选时**只看目录里有没有 md，
        # 不看目录叫什么名字**——外层目录名本来就是自由的（叫 corpus/、我的记忆/
        # 都一样读得到），拿名字猜就是在教人一个错的判据
        near = [d for d in sorted(root.parent.iterdir())
                if d.is_dir() and d != root and corpus_files(d)] if root.parent.exists() else []
        hint = (f"同级的这些目录里有 md，你要指的多半是其中之一："
                f"{'、'.join(d.name for d in near[:5])}。" if near
                else "它的同级目录里也没有——确认一下记忆库到底建在哪儿。")
        add(FAIL, "语料文件", f"{root} 下（含子目录）没找到任何 .md 语料。{hint}")
        return out
    add(OK, "语料目录", f"{root}（{len(corpus_only_files)} 个 md 文件）")

    try:
        roots = retrieval_dirs(root, index_dir)
    except (FileNotFoundError, NotADirectoryError) as e:
        add(FAIL, "索引摘要目录", str(e) + "。检查 MCP 配置里的 --index-dir；"
            "不需要独立索引层时删掉这一对参数。")
        return out
    if index_dir is not None:
        add(OK, "索引摘要目录", f"{Path(index_dir).resolve()}（doctor 只读；latent_append 可写）")
    else:
        fallback_index = root / "index"
        add(WARN, "索引摘要目录",
            f"未配 --index-dir：latent_append 会把摘要写入兼容路径 {fallback_index}；"
            "检索会随 --corpus 递归读到它。新部署推荐显式配置独立 --index-dir。")
    files = corpus_files(roots)

    stray = [p.name for p in corpus_only_files if p.name.lower() in _NOT_CORPUS_MD]
    if stray:
        add(WARN, "混进来的文件",
            f"语料里有 {'、'.join(sorted(set(stray)))}——这些是产出目录那一层的文件，"
            "不是记忆。多半是 --corpus 多指了一层（应指向里面的记忆库目录）。"
            "它们会被当成记忆检索到，但不会报任何错。")

    index = load_corpus(roots)         # embed=False：这一档不写任何缓存文件
    if not index.chunks:
        # "有 md、但一块都切不出来"是上面那道 `not files` 关卡挡不住的一档：
        # 文件全是空行/只有空白的话，语料目录看着满满当当，库却是空的，
        # 服务照样起、每次检索都空手。这条要显著地说，别让它一路往下崩在除法上
        add(FAIL, "建库", f"{len(files)} 个 md 文件里一块内容都切不出来——"
                          "文件是空的或只有空白行。这样的库起得来但查不到任何东西，"
                          "每次检索都会空手。先确认语料是不是真写进去了。")
        return out
    n_index = sum(1 for m in index.meta if m.get("layer") == "index")
    referenced_records = source_record_ids(unresolved_for_doctor)
    if referenced_records:
        current_records = {_chunk_key(chunk) for chunk, meta in zip(index.chunks, index.meta)
                           if meta.get("layer", "timeline") == "timeline"}
        orphans = sorted(referenced_records - current_records)
        if orphans:
            add(WARN, "未解决来源",
                f"有 {len(orphans)} 个 timeline recordId 已找不到（"
                f"{'、'.join(orphans[:3])}{' 等' if len(orphans) > 3 else ''}）；"
                "事项仍会浮现，但来源无法核对，请人工 update 或 close。")
        else:
            add(OK, "未解决来源", f"{len(referenced_records)} 个 timeline recordId 均可核对")
    lens = sorted(len(c) for c in index.chunks)
    median_len = lens[len(lens) // 2]
    add(OK, "建库", f"{len(index.chunks)} 块（索引层 {n_index} / 叙事层 "
                    f"{len(index.chunks) - n_index}），中位块长 {median_len} 字")

    # 切块成色：块数漂亮、分层漂亮，但每个块只有一句话——体检以前只报块数，
    # 报告里那个虚高一个数量级的数字**没有任何一行说它不正常**（首发用户就是这么
    # 错过它的）。只加诊断、不改切块行为
    # ⚠ 这里必须看**合并前**的形态。2026.08.03 切块下界落地后，无正文块在建库阶段
    # 就被向上合并掉了，拿 index.chunks 数永远是 0——**两条防线会互相吃掉**，
    # 而被吃掉的那条恰恰是唯一会告诉用户"你的导出有毛病"的那条。修法治的是症状，
    # 诊断仍要照直说病因：导出该改，不然每次建库都要靠合并兜。
    raw = []
    for f in files:
        try:
            t = Path(f).read_text(encoding="utf-8")
        except OSError:
            continue          # 实际不可达（load_corpus 在前面先读过一遍），防御性保留
        raw += chunk_heading(t, merge_bodyless=False)
    n_heading_only = sum(1 for c in raw if not _chunk_body(c))
    # 有文件读不出来时 raw 会比实际建库的块少，差值可能变负——报负数比不报更糊涂，
    # 钳到 0（合并只会让块变少，负数在语义上不存在）
    merged_n = max(0, len(raw) - len(index.chunks))
    if raw and n_heading_only > len(raw) * _HEADING_ONLY_WARN_RATIO:
        add(WARN, "切块", f"切块前 {n_heading_only}/{len(raw)} 块只有一行标题、没有正文"
            f"——已在建库时合并 {merged_n} 块，现在是 {len(index.chunks)} 块、"
            f"中位块长 {median_len} 字。**库是能用了，但语料本身该改**：去看一眼是不是"
            "**每句话前面都有 `##`**（有的第三方聊天导出插件这么干）。切块按 `## ` 认"
            "小节，所以一句话就成了一个块。合并只是兜底，导出后把这些消息前缀去掉再"
            "建库，块的边界才落在你真正的话题上。")

    # 时间范围（判据 3）：兜的是《快速上手》第 0 步那个坑的**部署侧版本**——
    # 旧导出包建出来的库，条数漂亮、块数漂亮、什么都不报错，只是**整份停在了
    # 过去某一天**。`memory_import.py --stats` 只在导入那一刻能发现它；导完之后，
    # 这里是唯一还会把这个数摆到人眼前的地方。只读 index.meta 里现成的 timestamp，
    # 不开任何文件句柄
    ts = [m["timestamp"] for m in index.meta if m.get("timestamp")]
    if ts:
        span = (f"{datetime.fromtimestamp(min(ts)):%Y-%m-%d} ~ "
                f"{datetime.fromtimestamp(max(ts)):%Y-%m-%d}")
        add(OK, "时间范围", f"{span}——盯住后面那个日期，问自己一句：跟 TA 最近一次"
                            "聊天真的是这天吗？对不上说明这份语料是旧快照，"
                            "重新导一份再建库（数字再健康也救不了停在过去的语料）。")
    else:
        add(WARN, "时间范围", "一块都没有时间戳，算不出时间范围——"
                              "换窗召回按时间新鲜度排序，这种情况下它没有任何排序依据。")
    if n_index == 0:
        add(WARN, "分层", "没有索引层（父目录名叫 index 的才算）。不算错，但命中率会低一档："
                          "索引层是每会话一条高密度摘要，专门喂检索。")

    # 时间戳成色：mtime 兜底不是"不太准"，是**方向错误**——复制目录/重新 clone 会把
    # 全目录 mtime 刷成"刚刚"，于是最没有时间依据的块拿到全库最新的日期，
    # 换窗召回按新鲜度排序时**最旧的事实冒充最新的**（2026.08.03 跨模型实测：
    # 这么一份语料下两家模型 0/6 答对当下状态）。
    srcs = {}
    for m in index.meta:
        srcs[m.get("timestamp_source")] = srcs.get(m.get("timestamp_source"), 0) + 1
    detail = "、".join(f"{k} {v} 块" for k, v in sorted(srcs.items(), key=lambda x: -x[1]))
    n_mtime = srcs.get("mtime", 0)
    ratio = n_mtime / len(index.chunks)
    # 判据二（2026.08.03 补，**与比例无关**）：只要有 mtime 档的块比全库最新的
    # 真日期块还"新"，就报——**危险的恰恰是混合且占比低的情况**：绝大多数块有日期、
    # 少数几块落 mtime，那几块的时间戳就是"今天"，稳稳压过所有真日期，
    # 而占比 5% 连比例判据的门都够不着，于是静默。比例判据看的是"有多少块没日期"，
    # 这条看的是"没日期的块会不会骑到有日期的头上"，两件事。
    real_ts = [m["timestamp"] for m in index.meta
               if m.get("timestamp") is not None and m.get("timestamp_source") != "mtime"]
    mtime_ts = [m["timestamp"] for m in index.meta
                if m.get("timestamp") is not None and m.get("timestamp_source") == "mtime"]
    outranks = bool(real_ts) and bool(mtime_ts) and max(mtime_ts) > max(real_ts)
    if ratio > _MTIME_WARN_RATIO or outranks:
        bad = sorted({m["source"] for m in index.meta
                      if m.get("timestamp_source") == "mtime"})
        add(WARN, "时间戳来源",
            f"{detail}——{ratio:.0%} 的块只能退到文件修改时间。它不是“不太准”，"
            "**是方向错误**：复制一遍目录或重新 clone，这些块的时间会被刷成“刚刚”，"
            "于是最没有日期依据的内容拿到全库最新的时间戳，"
            "**换窗召回会把最旧的事实当成你现在的状态端出来**"
            + ("（**这份语料已经是这种形态**：落 mtime 的块比所有带真日期的块都新）"
               if outranks else "")
            + "。修法是把日期写进文件名"
            f"（window_04_2026-06-17.md 这种）或标题行。落 mtime 的文件："
            f"{'、'.join(bad[:5])}{' 等' if len(bad) > 5 else ''}")
    else:
        add(OK, "时间戳来源", detail)
    if index.date_order:
        add(OK, "日期顺序", f"语料里的 m/d/y 型日期按 {index.date_order} 解析（有决定性证据）")

    # sidecar：三个都是可选的，**没有不是错**；有、但一块都对不上才是错——
    # 那说明语料被编辑过或换过目录，哈希对不上号，等于这份 sidecar 静默失效了
    for name, loader, what in ((".retractions.json", index.load_retractions, "撤回账本"),
                               (".weights.json", index.load_weights, "命中权重"),
                               (".entities.json", index.load_entities, "实体标注")):
        p = root / name
        if not p.exists():
            add(OK, name, f"没有（正常：{what}第一次用到时才生成）")
            continue
        try:
            n = loader(p)
        except (ValueError, OSError) as e:
            add(FAIL, name, f"{what}读不出来：{e}")
            continue
        if n == 0:
            add(WARN, name, f"{what}在，但没有一条对得上当前语料——按内容哈希对号入座，"
                            "语料被编辑过或换了目录就会全部失效（文件还在，等于没有）。")
            continue
        # 孤儿条目（2026.08.03 外部缺陷报告逼出来的一格）：账本条目对不上任何现存块
        # 就是死账——**部分孤儿比全孤儿隐蔽得多**：接上的那几条让上面那格报 OK，
        # 孤儿那条静默失效。撤回账本的孤儿尤其要命：撤回看着成功了、落盘了、
        # 体检全绿，重启后被撤回的旧说法照常召回。此前 append 手拼块文本与
        # 重启重切不一致就是这么漏网的（那个根因已修，这格防的是同形状的下一个）。
        cur_keys = {_chunk_key(c) for c in index.chunks}
        orphans = [k for k in json.loads(p.read_text(encoding="utf-8"))
                   if k not in cur_keys]
        if orphans:
            add(WARN, name, f"{what}接上 {n} 块，但有 {len(orphans)} 条孤儿条目"
                            f"对不上任何现存块（键：{'、'.join(orphans[:3])}"
                            f"{' 等' if len(orphans) > 3 else ''}）——这些账等于已静默"
                            f"失效；若是撤回账本，对应的旧记录会照常被召回。多半是"
                            f"块正文被编辑过、或旧版本手拼块文本留下的死账。")
        else:
            add(OK, name, f"{what}接上 {n} 块，无孤儿条目")
    index.build()                       # 实体边在 build 时算，接上后要重建一次

    # 写回落点：latent_append/latent_correct 要往这里写。只用 os.access 判，
    # 不试写——试写就破了只读
    write_root = Path(write_dir).resolve() if write_dir is not None else root / "timeline"
    probe = write_root if write_root.is_dir() else write_root.parent
    outside = False
    if write_dir is not None:
        try:
            write_root.relative_to(root)
        except ValueError:
            outside = True
    if outside:
        # 不报错的失败里最贵的一种：写盘成功、回执正常，但 --corpus 递归读不到它，
        # 于是"记下了"之后永远检索不到自己刚写的东西
        add(FAIL, "写回落点", f"--write-dir {write_root} 不在 --corpus {root} 之内："
            "写得进去，但检索只按 --corpus 递归读，刚写的记忆一条都查不回来。")
    elif os.access(probe, os.W_OK):
        # 落点要报到层：写回只进这一个目录（append_record 构造上锁死的，调用方
        # 传不了路径），人格文件里「按需读取指针」必须盖住它——用户拿这行对自己的
        # 指针，指漏了的症状是"新长出来的记忆按需读不到"，而且不报错
        hint = "" if write_dir is not None else \
            "；未配 --write-dir，用的是缺省落点 <corpus>/timeline。" \
            "要把这支笔写的记录跟语料里别处（例如宿主客户端自动记的那一档）分开，" \
            "就显式配一个 --write-dir，它必须仍落在 --corpus 之内"
        add(OK, "写回落点", f"{write_root} 可写（latent_append 按窗口号+日期"
            f"往这里加文件；人格文件的「按需读取指针」要盖住这个目录）{hint}")
    else:
        add(FAIL, "写回落点", f"{probe} 不可写——模型会说“记下了”，但每一次写回都失败。")

    # thread 落点：没配 --threads 时 latent_thread_close 只活在内存里，进程一退就没了，
    # 下个会话的开场召回接不上上一次聊到哪
    if not threads_path:
        add(WARN, "会话线索", "没配 --threads，会话收尾只在内存里、进程一退就没——"
                              "下个会话接不上“上次聊到哪”。MCP 配置里补一个 jsonl 路径。")
    else:
        tp = Path(threads_path)
        if not tp.exists():
            writable = tp.parent.exists() and os.access(tp.parent, os.W_OK)
            add(OK if writable else FAIL, "会话线索",
                f"{tp} 还没有（第一次 latent_thread_close 时创建）"
                + ("" if writable else "，但它的父目录不存在或不可写，创建会失败。"))
        else:
            try:
                threads = ThreadStore(tp).all()
            except (ValueError, OSError) as e:
                add(FAIL, "会话线索", f"{tp} 读不出来：{e}")
                threads = None
            if threads is not None:
                latest = max(threads, key=lambda t: (t.window, t.ended_at)) if threads else None
                add(OK, "会话线索", f"{tp}：{len(threads)} 条"
                                    + (f"，最新是第 {latest.window} 个窗口" if latest else ""))
                if latest and latest.open_loops:
                    add(WARN, "旧式未完线索",
                        f"最新 thread 仍有 {len(latest.open_loops)} 条 open_loops；"
                        "它们不再冒充当前未解决集合，请人工复核后显式 open。")

    if embed:
        # 明说不体检，而不是偷偷降级：embed 档下 load_corpus 会把块向量缓存
        # （.embed_cache.json）落在语料目录里，跑一次体检就落一个文件——跟只读冲突
        add(WARN, "检索路线", "体检只走零依赖档：embed 档建库会把块向量缓存"
                              "（.embed_cache.json）写进语料目录，跟“体检不落文件”冲突。"
                              "上面关于语料/时间戳/sidecar 的结论跟检索路线无关，照样作数；"
                              "embed 那一路通不通请直接起一次服务看。")

    # 时区：**没配的那一档必须报出来**（任务卡"写回时区与跨日归窗"）。它是这份报告里
    # 最典型的"不报错的失败"——服务照起、工具照返回成功，只是每天凌晨那几个小时写下的
    # 记忆被记成前一天，而错误日期会进文件名、H2、检索标签和重启后的排序信号。
    # ⚠ 报的是**实际在用的那个时区名**，不是"没配"三个字：用户要能一眼看出
    # 服务器认的是不是他自己那个时区（事故现场服务器认的是 Etc/UTC，用户在东八区）。
    # ⚠ **没配那一档仍然报 ⚠，哪怕默认值已经改成东八区**（2026.08.05 维护者拍板）：
    # 默认值对东八区以外的使用者就是错的，而且照样不报错——这一格是它唯一会露头的
    # 地方。⚠ 别把它降成 ✓：那等于让报告对着"可能不对的日期"打合格证。
    tc = time_context if time_context is not None else TimeContext.default()
    if tc.explicit:
        add(OK, "时区", f"{tc.name}——自然日按这个时区的 00:00～23:59 算，"
                        "写回、归窗、检索标签同一个口径")
    else:
        add(WARN, "时区", f"没配 --timezone，用的是默认值 {tc.name}（东八区）。"
                          "doctor 是独立进程，只看自己这条命令行上的 --timezone；"
                          "服务进程配过不算。"
                          "你就在东八区的话这条可以不管；**不在的话现在写下的记忆"
                          "日期是错的**，且不报错——文件名、记录标题、检索标签和重启后"
                          "的新鲜度信号会一起错。在 MCP 配置里补一行 "
                          "--timezone <你的 IANA 时区>（例如 Europe/Berlin）。")

    # 接线本身：握手 + 工具表 + 一次真检索。前面全绿也可能死在这一步，
    # 而这是唯一一处能证明"这份语料真能被查到"的检查
    srv = MemoryServer(index=index, thread_store=ThreadStore(threads_path), time_context=tc)
    #    ⚠ 故意不接 weights_path：接了的话下面这次检索会把权重落盘，体检就不再只读。
    #    要改这行之前先读 selftest 第 14 项——它就是守这个的。
    hs = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    tools = srv.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]
    add(OK, "MCP 接线", f"握手 {hs['result']['protocolVersion']}，工具 {len(tools)} 个："
                        + "、".join(t["name"] for t in tools))

    invalid_schemas = []
    for tool in tools:
        schema = tool.get("inputSchema")
        reasons = []
        if not isinstance(schema, dict) or schema.get("type") != "object":
            reasons.append("inputSchema 顶层不是 type: object")
        if isinstance(schema, dict):
            combinators = [name for name in ("anyOf", "oneOf") if name in schema]
            if combinators:
                reasons.append("顶层出现 " + "／".join(combinators))
        if reasons:
            invalid_schemas.append(f"{tool.get('name', '<未命名>')}（{'；'.join(reasons)}）")
    if invalid_schemas:
        add(FAIL, "工具 schema", "；".join(invalid_schemas)
            + "。兼容层可能拒绝整张工具表或丢掉参数。")
    else:
        add(OK, "工具 schema", "全部工具的 inputSchema 顶层均为 type: object，且无 anyOf/oneOf")

    probe, weak_probe = _doctor_probe(index)
    res = srv.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                      "params": {"name": "latent_search",
                                 "arguments": {"query": probe}}})["result"]
    level, detail = _doctor_probe_outcome(probe, weak_probe, res)
    add(level, "检索自查", detail)
    return out


def _doctor_probe(index):
    """优先取含内容词的标题；只有日期／窗口标题时返回弱探针标记。"""
    ranked = sorted(range(len(index.meta)),
                    key=lambda j: index.meta[j].get("timestamp") or 0, reverse=True)
    candidates = []
    for i in ranked:
        head = (index.meta[i].get("heading") or "").strip()
        if not head:
            head = next((ln.lstrip("# ").strip() for ln in index.chunks[i].splitlines()
                         if ln.strip()), "")
        if head:
            candidates.append(head[:30])
    for head in candidates:
        residue = re.sub(r"\d{4}[年./-]\d{1,2}[月./-]\d{1,2}日?", " ", head)
        residue = re.sub(r"第?\s*\d+\s*(?:窗|窗口|轮|次|段)", " ", residue)
        residue = re.sub(r"[\s·._/|｜:：\-—年月日记录对话聊天窗口]+", "", residue)
        if len(residue) >= 2:
            return head, False
    return candidates[0], True


def _doctor_probe_outcome(probe, weak_probe, result):
    """把探针结果分级；弱探针失败只说明标题缺内容词，不冒充接线失败。"""
    if not result["isError"]:
        return OK, f"拿语料自己的标题“{probe}”查得到"
    error = result["content"][0]["text"].splitlines()[0]
    if weak_probe:
        return (WARN, f"语料标题只有日期／窗口等通用词，只能拿“{probe}”试查；"
                      "这类探针缺少内容词，查不到不能据此判定接线失败。"
                      f"请另拿正文中的一句原话调用 latent_search：{error}")
    return FAIL, f"拿语料自己的标题“{probe}”去查，反而查不到：{error}"


def format_doctor_report(checks):
    """体检结果 → 给人看的报告。结论一句话放最后，别让人自己数图标。"""
    lines = ["记忆库部署体检", ""]
    for c in checks:
        lines.append(f"{_DOCTOR_ICON[c['level']]} {c['title']}：{c['detail']}")
    n_fail = sum(1 for c in checks if c["level"] == FAIL)
    n_warn = sum(1 for c in checks if c["level"] == WARN)
    lines.append("")
    if n_fail:
        lines.append(f"结论：{n_fail} 项过不去" + (f"、{n_warn} 项要注意" if n_warn else "")
                     + "。上面标 ✗ 的先修，修完再起服务。")
    elif n_warn:
        lines.append(f"结论：能用，{n_warn} 项要注意——标 ⚠ 的都是"
                     "“不报错但会悄悄变差”的那类，值得看一眼。")
    else:
        lines.append("结论：全部通过。")
    lines.append("（体检只读，没有向语料目录写入任何文件。）")
    return "\n".join(lines)


# ---------- 启动失败：给人话、给出口、让用户真的看得见（任务卡"服务端起不来时全是裸堆栈"） ----------
#
# **病在哪**（2026.08.05 维护者本人走完一遍 Operit 接入，一条一条撞出来的）：
# `mcp_server.py` 起不来时全是裸 Python 堆栈，而 stdio 形态下**客户端只显示
# 「连接失败」四个字，那段堆栈用户根本看不到**——手上没有任何下一步。
# 跟《出货闸报错不带出口》是同一个病搬到启动层，而且更严重：出货闸至少把报错
# 打在用户眼前，这里连打都打不出来。
#
# **修法（2026.08.06 维护者拍板：协议降级壳 ＋ 文件留痕）**：
# 1. **协议降级壳**——stdio 档启动失败时不直接死，改起一个「只会说这一句话」的
#    最小 server（`StartupFailureServer`）：照常回 `initialize`，人话进 `instructions`；
#    任何 `tools/call` 一律 `isError=true` 回同一句话。这是**唯一不依赖客户端实现、
#    在 Operit 和 Claude Code 上都能进用户眼**的路径——stderr 那条路已经证明进不去。
# 2. **文件留痕**——同一句话连同完整堆栈写进 `startup_log_path()`，横幅与人话都指这条路。
#
# ⚠ **降级壳不是变相放宽检查**（任务卡第五节第一条）：它**一个真工具都不提供**，
# `tools/list` 只有 `memory_startup_error` 一个诊断工具，任何调用都失败。
# 检查该拒还是拒了，改的只是**拒绝怎么说话**。谁以后想往这个壳里塞一个"能用的"
# 工具当降级服务，那就是这张卡明确禁止的事。
#
# ⚠ **这一节管不到「用户真的看见了没有」**：那要拿真客户端故意造一次失败、看它
# 界面上出现了什么（任务卡靶心二）。**我们在终端里看到了不算**——终端是我们的
# 视角，不是用户的。

# 慢加载提示的两道门槛（秒）。第一道给"是不是卡死了"一个答复，第二道让久等的人
# 知道进程还活着。⚠ 只提示，**不改加载时序**（任务卡靶心四：改成全异步/懒加载会
# 把可见的慢换成不可见的慢——握手秒回、第一次 latent_search 卡死，更糟）。
_SLOW_LOAD_NOTICE_SECONDS = (5.0, 30.0)

# 启动失败报告落盘的目录，可用环境变量顶掉（容器里 HOME 可能不可写）
STARTUP_LOG_DIR_ENV = "MEMORY_MCP_LOG_DIR"


def startup_log_path():
    """启动失败报告写哪儿：`$MEMORY_MCP_LOG_DIR`／默认 `~/.memory-mcp`。

    ⚠ **刻意不落在 `--corpus` 目录下**：第 1 条（语料/模型加载失败）里那个目录
    本身就可能是出问题的一方，而且语料目录是**会被同步、会被贴给别人看**的地方，
    堆栈里带着本机路径。"""
    base = os.environ.get(STARTUP_LOG_DIR_ENV) or (Path.home() / ".memory-mcp")
    return Path(base) / "last-startup-error.log"


def _looks_offline(exc):
    """这个异常是不是「拉不到模型」那一类（没网／下不动／库没装）。

    ⚠ **只在 embed 档的加载阶段问这句话**——分档主要靠**出错在哪个阶段**
    （加载 vs 绑端口），不靠翻异常文本，这样第 1 条与第 3 条不可能串。
    这里问的是加载阶段内部的第二层：是"拿不到模型"还是"语料本身有问题"。"""
    for e in (exc, exc.__cause__, exc.__context__):
        if e is None:
            continue
        if isinstance(e, (socket.gaierror, TimeoutError)):
            return True
        if isinstance(e, OSError) and e.errno in (
                errno.ENETUNREACH, errno.EHOSTUNREACH, errno.ECONNREFUSED,
                errno.ETIMEDOUT, errno.ENETDOWN):
            return True
        if isinstance(e, ImportError):
            # fastembed/onnxruntime 没装：跟没网是同一条出口（先联网备好这一档）
            return True
        name = type(e).__name__
        if name in ("ConnectError", "ConnectTimeout", "ReadTimeout", "ProxyError",
                    "RequestError", "TransportError", "SSLError", "HTTPError",
                    "ConnectionError", "MaxRetryError", "NewConnectionError",
                    "URLError"):
            return True
        text = str(e)
        for probe in ("Network is unreachable", "Temporary failure in name resolution",
                      "Max retries exceeded", "Failed to establish a new connection",
                      "getaddrinfo failed", "Name or service not known"):
            if probe in text:
                return True
    return False


def startup_failure_report(stage, exc, *, embed=False, http_bind=None, log_path=None):
    """把一次启动失败翻成「一句人话 ＋ 一条可照抄的出口」。

    `stage` 有三个值，对应任务卡那张表里仍然作数的失败路径，外加 `args` 解析阶段：
      - `"config"` → `args` 解析之后、建索引之前的配置失败（当前只有 `--timezone`
                     认不出 IANA 名，多见于没装 tzdata 的 Windows。2026.08.22
                     「stdio 档启动失败无出口」把它接进这条出口）
      - `"load"`  → 建索引/加载语料这一段（第 1 条在这里）
      - `"bind"`  → `--http` 绑端口这一段（第 3 条在这里）
    （表里第 4 条「语料很大时像卡死」不是失败，走 `slow_load_notice`；
    ⚠ **编号按任务卡那张表，故意不重排**——②虽已划掉也不把③④提上来，
    重排会让别处对「②」的引用悄悄指向另一条。）

    返回一段多行文本，第一行是人话、第二行是出口，末行指向落盘的完整堆栈。"""
    bind_unavailable = (
        stage == "bind" and isinstance(exc, OSError) and
        (exc.errno == errno.EADDRINUSE or getattr(exc, "winerror", None) == 10013)
    )
    if bind_unavailable:
        why = (f"要监听的地址已经被别的进程占着，或被 Windows 保留，"
               f"起不来（{http_bind}）。")
        how = ("出口：换一个端口重起（例如 --http 127.0.0.1:8766），"
               "或者先把占着这个端口的进程停掉再重起"
               "（Linux/macOS：lsof -i :<端口>；Windows：netstat -ano | findstr :<端口>）。")
    elif stage == "load" and embed and _looks_offline(exc):
        why = "这台机器上没有现成的 embedding 模型，而现在也下不下来（没网或下载被挡）。"
        how = ("出口：二选一——①连上网再起一次，让它把模型下完（只需一次，之后离线可用）；"
               "②不想下模型就重跑 python memory_init.py --step route --route zero-dep，"
               "让它重新生成客户端配置。"
               "⚠ 不要自己把配置里的 --embed 删掉：--embed 必须跟 init_state 记的检索路线一致，"
               "手删参数是隐患，正确出口是重跑 --step route。")
    elif stage == "config":
        # exc（TimeContext 抛的 ValueError）自己就写着完整人话＋出口
        # （pip install tzdata／固定偏移 UTC+08:00），已过内测判据。直接拿来用，
        # ⚠ **不在这里另写一句"更好的"措辞**——那会造出第二份口径（CLAUDE.md 第 11 条）。
        why = str(exc)
        how = None
    elif stage == "load":
        why = "加载阶段失败了；只凭这次异常无法判断是配置、依赖还是语料出了问题。"
        how = ("出口：先把同一行参数里的 --http/--threads 都留着、"
               "把命令换成 python mcp_server.py --doctor --corpus <同一个目录> 跑一次；"
               "体检只读不写盘。若体检没指出问题，再看随后落盘的完整报错，"
               "按里面的异常类型排查配置值、依赖或模型服务。")
    else:
        why = "服务端在启动阶段失败了。"
        how = ("出口：python mcp_server.py --doctor --corpus <同一个目录> 跑一次体检，"
               "再看下面那份完整报错。")
    # config 档 how 为 None（人话全在 why=str(exc) 里，不另造出口）；其余档 how 恒是字符串
    lines = [ln for ln in ("记忆协议服务端没能起来。",
                           f"原因：{why}",
                           how,
                           f"（技术细节：{type(exc).__name__}: {exc}）") if ln]
    if log_path is not None:
        lines.append(f"完整报错已经写进 {log_path}，排查时把那个文件贴出来即可。")
    return "\n".join(lines)


def write_startup_log(report, exc, path=None):
    """把人话和完整堆栈一起落盘，返回落点；写不进去就返回 None（**不许因此再崩一次**）。

    ⚠ 写失败必须吞掉：这段代码只在"已经失败了"的路径上跑，让它自己抛异常
    等于把用户仅剩的那句人话也弄丢——正是这张卡在治的形状。"""
    path = Path(path) if path is not None else startup_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(_dt_tz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# 记忆协议 MCP server 启动失败（{stamp}）\n\n{report}\n\n")
            f.write("--- 完整堆栈（给排查用，上面那几行才是给你看的）---\n")
            f.write("".join(traceback.format_exception(
                type(exc), exc, exc.__traceback__)))
        return path
    except OSError:
        return None


class StderrTee:
    """把 stderr 原样转发到终端，同时追加落盘一份。`--log-file` 的全部实现。

    治的是 issue #20 那个形状：stdio 传输下 stderr 由客户端接管，而**客户端
    经常直接丢弃**（本文件另一处注释早就写了这件事）。于是子进程崩没崩、什么时候重启的、
    堆栈是什么，全都没有落点——报告人只能靠回头翻 token 账单发现某段前缀被重复计费，
    倒推"是不是重启过"。那是成本层面的事后取证，定位不到任何东西。

    ⚠ 选择"换掉 `sys.stderr`"而不是改几十个 `print(..., file=sys.stderr)` 调用点：
    `print` 每次调用都现取 `sys.stderr`，换掉全局引用就等于全部接管，且解释器抛
    未捕获异常时也走同一个口子（堆栈一起落盘）。改调用点则必漏——本文件里光是
    `print(..., file=sys.stderr)` 就散在启动、拒绝、异常、横幅四类路径上。

    ⚠ 写盘失败一律吞掉、退回纯终端输出：这东西是排查用的旁路，绝不许因为日志写不进去
    反过来把服务弄崩（同 write_startup_log 那条约束）。"""

    def __init__(self, stream, handle):
        self._stream = stream
        self._handle = handle

    def write(self, text):
        written = self._stream.write(text)
        try:
            self._handle.write(text)
            self._handle.flush()      # 崩溃前最后几行不许留在缓冲区里
        except (ValueError, OSError):
            pass
        return written

    def flush(self):
        try:
            self._stream.flush()
        except (ValueError, OSError):
            pass
        try:
            self._handle.flush()
        except (ValueError, OSError):
            pass

    def isatty(self):
        try:
            return self._stream.isatty()
        except (ValueError, OSError):
            return False


def install_stderr_tee(path, stream=None, now=None):
    """按 `--log-file` 接上 tee，返回接好的流；接不上就原样返回、只抱怨一句。

    每次进程起来都先写一行带 pid 和时间的横幅——**这一行本身就是 issue #20 要的证据**：
    同一个 `--log-file` 里出现第二条横幅，就说明子进程确实重启过一次，不必再靠
    token 账单倒推。"""
    stream = stream or sys.stderr
    stamp = (now or datetime.now(_dt_tz.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        p = Path(path)
        if p.parent != Path(""):
            p.parent.mkdir(parents=True, exist_ok=True)
        handle = open(p, "a", encoding="utf-8")
        handle.write(f"\n===== 进程启动 pid={os.getpid()} {stamp} =====\n")
        handle.flush()
    except OSError as e:
        print(f"⚠ --log-file 写不进去（{type(e).__name__}: {e}），"
              f"这次只往终端打；服务照常起。", file=stream)
        return stream
    return StderrTee(stream, handle)


def slow_load_notice(stream=None, thresholds=_SLOW_LOAD_NOTICE_SECONDS):
    """加载慢过门槛就说一句话，返回一个「加载完了叫我」的取消函数。

    治的是任务卡那张表**第 4 条**：语料很大时握手前要先把全库读完建索引，
    这段时间里进程一声不吭，看起来跟卡死一模一样，客户端只会超时。

    ⚠ **这一轮只做到「慢的时候有话说」**（靶心四，防过度纠正）：
    不把加载改成全异步/懒加载——那样握手秒回、第一次 `latent_search` 卡死，
    是把可见的慢换成不可见的慢。真要改加载时序另开卡。

    ⚠ **这句话走 stderr，因此在 stdio 形态下用户多半看不到**——这跟降级壳
    （失败路径）不是一回事：第 4 条最终是"加载完了、能起来"，协议流那时才开始，
    没有一个既不撒谎又能提前说话的口子。别把这条读成"第 4 条也解决了可见性"。"""
    stream = stream or sys.stderr
    timers = []

    def say(text):
        print(text, file=stream)
        try:
            stream.flush()
        except (ValueError, OSError):
            pass

    first, second = thresholds
    timers.append(threading.Timer(first, say, args=(
        "正在读语料建索引……语料多的时候这一步要花点时间，进程没有卡死，请再等等。",)))
    timers.append(threading.Timer(second, say, args=(
        "还在建索引（已经超过 %g 秒）。首次启动最慢，之后会快一些；"
        "现在中断它不会损坏任何数据。" % second,)))
    for t in timers:
        t.daemon = True          # 别让这两只表拖住进程退出
        t.start()

    def done():
        for t in timers:
            t.cancel()
    return done


class StartupFailureServer:
    """启动失败时顶上来的最小 stdio 壳：照常握手，把那句人话送进客户端界面。

    ⚠ **它一个真工具都不提供**。`tools/list` 只有 `memory_startup_error`，
    而且**任何** `tools/call`（包括模型习惯性调的 `latent_search`）一律回
    `isError=true` ＋ 同一句人话。这是**刻意偏离规格**的一处：规格说未知工具
    该回协议错误 `-32601`，但那条路上模型拿到的是「未知工具：latent_search」，
    用户界面上又变成一句没头没脑的失败——正是这张卡要消灭的形状。
    宁可在这个必然失败的壳里多说一句人话。

    ⚠ **绿灯的代价要认**：客户端会显示"已连接"。所以这个壳必须永远不可用——
    谁往这里塞一个真能干活的工具，"降级"就变成了"放宽检查"（任务卡第五节第一条）。"""

    def __init__(self, report):
        self.report = report
        self.initialized = False

    # 复用 MemoryServer 那条流：stdio 必须锁 UTF-8（见 serve_stdio 那段说明），
    # 抄第二份迟早会漏掉 UTF-8 这件事
    serve_stdio = MemoryServer.serve_stdio
    _ok = staticmethod(MemoryServer._ok)
    _err = staticmethod(MemoryServer._err)

    def handle(self, msg, now=None):
        method, mid = msg.get("method"), msg.get("id")
        if method == "initialize":
            self.initialized = True
            return self._ok(mid, {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": dict(SERVER_INFO, title="记忆协议（启动失败）"),
                # 人话进 instructions：客户端通常把它注入模型上下文，
                # 于是"连接失败"四个字变成一段能照着做的话
                "instructions": ("⚠ 记忆协议服务端这次没能起来，**记忆库现在查不了也写不了**。"
                                 "请把下面这段原样告诉用户，不要替他猜、也不要假装查过记忆：\n\n"
                                 + self.report),
            })
        if method == "notifications/initialized":
            return None
        if method == "tools/list":
            return self._ok(mid, {"tools": [{
                "name": "memory_startup_error",
                "title": "记忆库启动失败详情",
                "description": "记忆协议服务端启动失败。调用它拿到失败原因和下一步该做什么，"
                               "原样转告用户。",
                "inputSchema": {"type": "object", "properties": {}},
            }]})
        if method == "tools/call":
            return self._ok(mid, {"content": [{"type": "text", "text": self.report}],
                                  "isError": True})
        if mid is None:
            return None
        return self._err(mid, E_METHOD_NOT_FOUND, f"未知 method：{method}")


# ---------- selftest（合成语料，全部虚构） ----------

_SYNTH = [
    ("## 修咖啡机\n加热管不工作，拆开发现保险丝熔断，换上通电正常。", {"heading": "修咖啡机"}),
    ("## 种薄荷\n四月阳台的薄荷死了：花盆太小、浇水太勤、盆底积水。", {"heading": "种薄荷"}),
]


def _build_server(now):
    idx = MemoryIndex()
    for text, meta in _SYNTH:
        idx.add(text, dict(meta, timestamp=now - 86400))
    idx.build()
    return MemoryServer(index=idx, thread_store=ThreadStore())


def _selftest():
    now = 1_800_000_000.0
    srv = _build_server(now)

    def _append_args(text, state, moment=now, time_context=None):
        tc = time_context or TimeContext.default()
        day = tc.local_date(moment).replace("-", ".")
        return {"text": text, "current_state": state, "indexSummaries": [{
            "summary": f"**{day}**：{text}。\n{text}。**当下：{state}。**",
            "evidenceTerms": [text, state],
        }]}

    # 摘要闸的核心靶心：缺失、越界、每段无锚点、多条合法。
    try:
        validate_index_summaries([], "柳州那晚很难过", "约定仍成立", "2026-08-20")
        assert False, "缺摘要不能进入索引层"
    except ValueError as exc:
        assert "不写索引摘要" in str(exc)
    try:
        validate_index_summaries(
            [{"summary": "只有一行", "evidenceTerms": ["只有一行"]}],
            "只有一行", "仍在处理", "2026-08-20")
        assert False, "形状错误必须拒绝"
    except ValueError as exc:
        assert "第 1 段（A）" in str(exc) and INDEX_SUMMARY_TEMPLATE in str(exc) \
            and INDEX_SUMMARY_EXAMPLE in str(exc), \
            "格式错误必须定位 A 段，并把同一模板与合法样例交还给调用方"
    malformed_segments = (
        ("**2026.08.20**：第一段。\n缺少当下标签", "第 2 段（B）"),
        ("**2026.08.20**：第一段。\n第二段。**当下：缺少结尾", "第 3 段（C）"),
    )
    for malformed, expected_segment in malformed_segments:
        try:
            validate_index_summaries(
                [{"summary": malformed, "evidenceTerms": ["第一段"]}],
                "第一段和第二段", "缺少结尾", "2026-08-20")
            assert False, f"{expected_segment} 形状错误必须拒绝"
        except ValueError as exc:
            assert expected_segment in str(exc) and "期望" in str(exc), \
                f"格式错误必须定位到 {expected_segment} 并说明期望形状：{exc}"
    valid_summary = "**2026.08.20**：柳州那晚约定直说。\n一直猜而难过。**当下：约定仍成立。**"
    valid_terms = ["柳州那晚", "约定", "直说", "一直猜", "难过", "仍成立"]
    base_text = "柳州那晚约定直说，一直猜让她难过。"
    assert len(validate_index_summaries(
        [{"summary": valid_summary, "evidenceTerms": valid_terms}] * 2,
        base_text, "约定仍成立", "2026-08-20")) == 2
    try:
        validate_index_summaries(
            [{"summary": valid_summary.replace("2026.08.20", "2026.08.19"),
              "evidenceTerms": valid_terms}],
            base_text, "约定仍成立", "2026-08-20")
        assert False, "摘要日期不是服务端本地当天时必须拒绝"
    except ValueError as exc:
        assert "日期应为 2026.08.20" in str(exc)
    for terms, expected in ((valid_terms + ["安全感"], "越界词"),
                            (["柳州那晚", "约定", "直说", "仍成立"], "未锚定段")):
        try:
            validate_index_summaries([{"summary": valid_summary, "evidenceTerms": terms}],
                                     base_text, "约定仍成立", "2026-08-20")
            assert False, "越界词或无证据锚点的摘要段必须拒绝"
        except ValueError as exc:
            assert expected in str(exc)
    try:
        validate_index_summaries(
            [{"summary": valid_summary, "evidenceTerms": valid_terms + ["旅行社"]}],
            base_text + "旅行社已确认。", "约定仍成立", "2026-08-20")
        assert False, "只在 text 中、没进 summary 的证据词必须拒绝"
    except ValueError as exc:
        assert "旅行社（缺在 summary）" in str(exc) and "两边缺任一处都算越界" in str(exc), \
            f"越界文案必须点明词缺在 summary：{exc}"
    try:
        validate_index_summaries(
            [{"summary": valid_summary.replace("一直猜", "旅行社一直猜"),
              "evidenceTerms": valid_terms + ["旅行社"]}],
            base_text, "约定仍成立", "2026-08-20")
        assert False, "只在 summary 中、没进 text/current_state 的证据词必须拒绝"
    except ValueError as exc:
        assert "旅行社（缺在 text/current_state）" in str(exc), \
            f"越界文案必须点明词缺在 text/current_state：{exc}"
    normalized_evidence = validate_index_evidence(
        [{"type": "event", "quote": '代号是 "LT-3289"'},
         {"type": "state", "quote": "测试　仍在进行"}],
        ['临时测试代号是“LT-3289”。', "测试 仍在进行"])
    assert normalized_evidence == [
        {"type": "event", "quote": "代号是“LT-3289”"},
        {"type": "state", "quote": "测试 仍在进行"},
    ], "只容忍排版差异，落索引的必须是重新定位到的原文"
    try:
        validate_index_evidence([{"type": "feeling", "quote": "心情很差"}], ["她有点失落"])
        assert False, "同义改写不得冒充原文证据"
    except ValueError as exc:
        assert "无法映射回原文" in str(exc)
    try:
        validate_index_evidence([{"type": "event", "quote": "代号是 LT-9999"}],
                                ['临时测试代号是“LT-3289”。'])
        assert False, "改写过的编号不得冒充原文证据"
    except ValueError as exc:
        assert "原文里最接近的一段是" in str(exc) and "LT-3289" in str(exc), (
            f"定位失败要回最接近的原文片段，模型才不用重拉原文：{exc}")
    with tempfile.TemporaryDirectory() as td_summary:
        root_summary = Path(td_summary)
        timeline_summary = root_summary / "timeline"
        index_summary = root_summary / "index"
        timeline_summary.mkdir()
        index_summary.mkdir()
        old_path = timeline_summary / "window_01_2026-08-20.md"
        old_path.write_text("旧内容\n", encoding="utf-8")
        plans = [{"path": old_path, "content": "新内容\n"}] + plan_index_records(
            index_summary, [valid_summary, valid_summary], 1, "2026-08-20")
        assert [plan["path"].name for plan in plans[1:]] == [
            "window_01_2026-08-20_item_01.md", "window_01_2026-08-20_item_02.md"]
        calls = []
        def _fail_second(source, target):
            calls.append(Path(target))
            if len(calls) == 2:
                raise OSError("故障注入：第二个 replace 失败")
            os.replace(source, target)
        try:
            commit_memory_files(plans, replace_func=_fail_second)
            assert False, "第二个 replace 失败必须报错"
        except OSError as exc:
            assert "故障注入" in str(exc)
        assert old_path.read_text(encoding="utf-8") == "旧内容\n"
        assert not list(index_summary.glob("*.md")), "新建摘要目标在失败后必须清掉"

    # 1. 握手：字段名与协议版本照规格（protocolVersion/capabilities/serverInfo）
    r = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": PROTOCOL_VERSION, "clientInfo": {"name": "x"}}})
    assert r["result"]["protocolVersion"] == PROTOCOL_VERSION
    assert set(r["result"]) == {"protocolVersion", "capabilities", "serverInfo", "instructions"}, \
        f"握手必须带 instructions（主动性靠它），实际字段 {sorted(r['result'])}"
    assert "tools" in r["result"]["capabilities"], "声明 tools capability，否则客户端不会列工具"
    #    【变异靶心：instructions】主动性靠它——2026.07.31 真机第一次主动性测试完败，
    #    模型把"之前说好的事"理解成本次会话记录、答"我没有记录"，转头查了宿主自带的
    #    记忆功能。规格 lifecycle 一节留了 instructions 这个口，客户端会注入进模型
    #    上下文，当初漏填了。两条必须在里面：说清是跨会话的长期库、堵死"我没有记录"
    instr = r["result"]["instructions"]
    assert "长期" in instr and "不是本次对话" in instr, "必须说清这是跨会话的长期记忆库"
    assert "我没有相关记录" in instr, "必须直接堵死“我没有记录”这句默认话术"
    assert "queryVariant" in instr and "最多再查一次" in instr and "第二次仍对不上就停" in instr, \
        "instructions 必须把单次改述的入口与停止边界一起交给宿主"
    assert all(x in instr for x in ("同一时刻只能有一个答案", "先用 latent_correct",
                                    "再用 latent_append", "不要只 append", "喜欢的电影",
                                    "不得为了写新项而 correct 旧项", "不要预先给每句话分类")), \
        "instructions 必须同时给出单值冲突的 correct→append 顺序与并列事实禁止 correct 的边界"
    assert all(x in instr for x in ("latent_unresolved", "open", "update", "close", "none",
                                    "不要按关键词", "历史快照")), \
        "instructions 必须交代未解决清单的四种判断、主观触发边界及 thread 历史定位"

    # 2b. 工具描述要写成触发条件，不是功能陈述——同样是真机反馈
    d = {t["name"]: t["description"] for t in
         srv.handle({"jsonrpc": "2.0", "id": 21, "method": "tools/list"})["result"]["tools"]}
    assert "不是本次对话" in d["latent_search"] and "我不记得" in d["latent_search"], \
        "latent_search 描述要说清记忆类型并堵死默认话术"
    assert "queryVariant" in d["latent_search"] and "最多重试一次" in d["latent_search"], \
        "latent_search 描述必须告诉宿主怎么触发受控改述融合"
    assert "必须先latent_correct 旧值" in d["latent_append"] and \
        "不得撤旧项" in d["latent_append"], \
        "latent_append 描述不能把单值冲突与并列新项都引向纯 append"
    assert "主动" in d["latent_session_start"] and "主动" in d["latent_thread_close"], \
        "两个生命周期工具要写明主动调用，不用等对方要求"
    #    initialized 是通知，不能回响应（回了客户端会当成野生响应）
    assert srv.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None, \
        "initialized 是通知，回响应会让客户端收到一条没人等的野生响应"

    # 2. tools/list：七个工具，schema 字段名照规格（name/inputSchema）
    tools = srv.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]
    assert [t["name"] for t in tools] == ["latent_search", "latent_session_start",
                                          "latent_append", "latent_correct",
                                          "latent_cleanup",
                                          "latent_unresolved",
                                          "latent_thread_close"]
    for t in tools:
        assert set(t) >= {"name", "description", "inputSchema"} \
            and t["inputSchema"]["type"] == "object" \
            and not {"anyOf", "oneOf"}.intersection(t["inputSchema"]), \
            f"{t['name']} 的 inputSchema 顶层必须是 object，且不得出现 anyOf/oneOf"
    # 2c.【工具 annotations】变异靶心：任一工具漏字段、把检索冒充只读、
    #     把更正冒充纯追加，或把本地记忆库误标成 open world，这张逐字表都会红。
    expected_annotations = {
        "latent_search": {"readOnlyHint": False, "destructiveHint": False,
                          "idempotentHint": False, "openWorldHint": False},
        "latent_session_start": {"readOnlyHint": True, "destructiveHint": False,
                                 "idempotentHint": True, "openWorldHint": False},
        "latent_append": {"readOnlyHint": False, "destructiveHint": False,
                          "idempotentHint": False, "openWorldHint": False},
        "latent_correct": {"readOnlyHint": False, "destructiveHint": True,
                           "idempotentHint": False, "openWorldHint": False},
        "latent_cleanup": {"readOnlyHint": False, "destructiveHint": True,
                           "idempotentHint": False, "openWorldHint": False},
        "latent_unresolved": {"readOnlyHint": False, "destructiveHint": True,
                              "idempotentHint": False, "openWorldHint": False},
        "latent_thread_close": {"readOnlyHint": False, "destructiveHint": False,
                                "idempotentHint": False, "openWorldHint": False},
    }
    assert {t["name"]: t.get("annotations") for t in tools} == expected_annotations, \
        "七个工具的 annotations 必须按真实副作用逐项声明"
    assert tools[6]["inputSchema"]["required"] == ["window", "current_state"], "当下状态必填要写进 schema"
    assert tools[3]["inputSchema"]["required"] == ["quote", "reason"], \
        "更正工具必填 quote+reason——没有原因的撤回不可追溯"
    append_tool = tools[2]
    append_schema = append_tool["inputSchema"]
    append_fields = {"mode", "text", "current_state", "window", "recordId", "indexEvidence",
                     "unresolvedOps"}
    root_combinators = {"anyOf", "oneOf", "allOf"}
    assert append_schema["type"] == "object" \
        and append_fields <= set(append_schema["properties"]) \
        and not root_combinators.intersection(append_schema), \
        "latent_append 必须发布无根级组合器的平面对象 schema，并保留现行字段与可选预检模式"
    assert append_schema["properties"]["mode"]["enum"] == ["write", "preflight"] \
        and "mode=preflight" in append_tool["description"] \
        and append_tool["annotations"]["readOnlyHint"] is False, \
        "预检必须留在现有写工具的平面 schema；工具整体不能冒充只读"
    assert "unresolvedOps" not in append_schema.get("required", []), \
        "兼容模式下 unresolvedOps 不得成为 latent_append 的 schema 必填字段"
    assert tools[4]["inputSchema"]["required"] == ["action", "recordId"] \
        and tools[4]["annotations"]["destructiveHint"] is True, \
        "latent_cleanup 必须暴露两阶段精确清理 schema，并明确标为破坏性工具"
    assert "unresolvedOps" in tools[6]["inputSchema"]["properties"] \
        and "unresolvedOps" not in tools[6]["inputSchema"].get("required", []), \
        "latent_thread_close 也要暴露可选 unresolvedOps，不能截断旧客户端"

    # 2b.【Kelivo PC schema 兼容】复刻 Kelivo
    # Chevey339/kelivo@04664e2d6d483c60e2f01e60e601956db81a2cd8 的
    # `_sanitizeNode()`：根级组合器会展开第一支，并先删掉原节点的 type／properties／items。
    # 这里只保留与当前工具 schema 有关的递归和白名单，判据同时覆盖三类 provider 的公共子集。
    def kelivo_sanitize_node(node):
        if isinstance(node, list):
            return [kelivo_sanitize_node(item) for item in node]
        if not isinstance(node, dict):
            return node
        sanitized = dict(node)
        sanitized.pop("$schema", None)
        for key in ("anyOf", "oneOf", "allOf", "any_of", "one_of", "all_of"):
            variants = sanitized.get(key)
            if isinstance(variants, list) and variants:
                flattened = kelivo_sanitize_node(variants[0])
                sanitized.pop(key, None)
                if isinstance(flattened, dict):
                    for replaced in ("type", "properties", "items"):
                        sanitized.pop(replaced, None)
                    sanitized.update(flattened)
        if isinstance(sanitized.get("items"), dict):
            sanitized["items"] = kelivo_sanitize_node(sanitized["items"])
        if isinstance(sanitized.get("properties"), dict):
            sanitized["properties"] = {
                key: kelivo_sanitize_node(value)
                for key, value in sanitized["properties"].items()
            }
        allowed = {"type", "description", "properties", "required", "items", "enum",
                   "additionalProperties"}
        return {key: value for key, value in sanitized.items() if key in allowed}

    def assert_kelivo_append_schema(schema):
        forwarded = kelivo_sanitize_node(schema)
        assert forwarded.get("type") == "object" \
            and append_fields <= set(forwarded.get("properties", {})), \
            "Kelivo 清洗后必须保留 latent_append 的对象类型与六个现行字段"

    assert_kelivo_append_schema(append_schema)
    mutated_schema = json.loads(json.dumps(append_schema))
    mutated_schema["anyOf"] = [
        {"required": ["text", "current_state"]},
        {"required": ["recordId", "indexEvidence"]},
    ]
    try:
        assert_kelivo_append_schema(mutated_schema)
    except AssertionError:
        pass
    else:
        raise AssertionError("变异失败：重新加回根级 anyOf 后，Kelivo 兼容断言必须转红")
    assert "recordId＋indexEvidence" in append_tool["description"] and \
        "旧 indexSummaries 仅兼容" in append_tool["description"], \
        "latent_append 必须告诉调用方如何在不重复正文的前提下补索引"
    assert INDEX_SUMMARY_FORMAT_GUIDE == append_tool["inputSchema"]["properties"][
        "indexSummaries"]["items"]["properties"]["summary"]["description"], \
        "summary 参数说明必须复用同一份格式契约"
    quickstart = (Path(__file__).resolve().parent.parent / "docs" / "快速上手.md").read_text(
        encoding="utf-8")
    quickstart_section = quickstart.split("### 手工调用 `latent_append` 的完整参数", 1)[1]
    quickstart_json = re.search(r"```json\n(.*?)\n```", quickstart_section, re.S)
    assert quickstart_json, "《快速上手》缺 latent_append 完整 JSON 样例"
    quickstart_args = json.loads(quickstart_json.group(1))
    quickstart_evidence = validate_index_evidence(
        quickstart_args["indexEvidence"],
        [quickstart_args["text"], quickstart_args["current_state"]])
    assert "企鹅漫步" in render_index_evidence(quickstart_evidence, "2026-08-23"), \
        "《快速上手》的结构化证据样例必须通过当前校验器并由服务端渲染"
    search_schema = tools[0]["inputSchema"]
    assert search_schema["required"] == ["query"] \
        and search_schema["properties"]["queryVariant"]["type"] == "string", \
        "queryVariant 必须是可选单字符串；原 query 仍是唯一必填项"

    # 3. tools/call 正常往返：结果结构照规格（content 数组 + type:text + isError）
    def call(name, args=None, mid=9):
        return srv.handle({"jsonrpc": "2.0", "id": mid, "method": "tools/call",
                           "params": {"name": name, "arguments": args or {}}}, now=now)
    res = call("latent_search", {"query": "咖啡机坏了"})["result"]
    assert res["isError"] is False and res["content"][0]["type"] == "text"
    assert "保险丝熔断" in res["content"][0]["text"]

    # 3b.【单次改述融合真 tools/call】原问法没有共享 bigram，改述补记录用词后命中。
    #     这条同时钉住 schema 背后的生产接线；只测 MemoryIndex 不足以证明 MCP 能用。
    variant_srv = MemoryServer(index=MemoryIndex(), thread_store=ThreadStore())
    variant_srv.index.add("阁楼那台天文镜最后换了目镜，配件已经齐了。", {"heading": "旧器材"})
    variant_srv.index.add("周末去河边公园散步，看见两只白鹭。", {"heading": "散步"})
    variant_srv.index.build()
    variant_call = variant_srv.handle({
        "jsonrpc": "2.0", "id": 31, "method": "tools/call",
        "params": {"name": "latent_search", "arguments": {
            "query": "看星星的设备后来弄好了吗",
            "queryVariant": "阁楼天文镜目镜配件",
        }}}, now=now)["result"]
    assert variant_call["isError"] is False and "配件已经齐了" in variant_call["content"][0]["text"], \
        f"queryVariant 必须穿过 MCP 并把词面错位的块带回来：{variant_call}"

    #     第二次仍空手就明确停；没传改述时只给一次出口。坏类型不能静默转字符串。
    empty_search_srv = MemoryServer(index=MemoryIndex().build(), thread_store=ThreadStore())

    def empty_search(args):
        return empty_search_srv.handle({
            "jsonrpc": "2.0", "id": 32, "method": "tools/call",
            "params": {"name": "latent_search", "arguments": args}}, now=now)["result"]

    first_empty = empty_search({"query": "不存在的旧事"})
    second_empty = empty_search({"query": "不存在的旧事", "queryVariant": "具体名字地点"})
    bad_variant = empty_search({"query": "不存在的旧事", "queryVariant": 7})
    assert first_empty["isError"] and "最多再试一次" in first_empty["content"][0]["text"], \
        "第一次空手要给唯一一次改述出口"
    assert second_empty["isError"] and "第二次仍没找到" in second_empty["content"][0]["text"] \
        and "停止检索" in second_empty["content"][0]["text"], \
        "带改述仍空手必须明确停止，不能诱导循环"
    assert bad_variant["isError"] and "非空字符串" in bad_variant["content"][0]["text"], \
        "queryVariant 类型错了要指名报参数错误"

    # 4.【变异靶心·薄适配层】外壳输出必须逐字等于底层库返回——谁在适配层里重新
    #    实现格式化，这条立刻红。这正是"分得清是接口问题还是底层库问题"的保证
    idx2 = _build_server(now).index
    expected_search = annotate_block(
        format_recall_block(idx2.retrieve("咖啡机坏了", topN=5)),
        query_miss_rate(idx2, "咖啡机坏了"))
    #    2026.08.01 随缺失率标注放宽了一格，但**纪律没放松**：比对的仍是"底层库
    #    函数的组合结果"（annotate_block(format_recall_block(...), 缺失率)），
    #    外壳只要自己拼一个字都会红
    assert call("latent_search", {"query": "咖啡机坏了"}, mid=10)["result"]["content"][0]["text"] \
        == expected_search, "latent_search 必须原样返回底层库函数的组合结果，外壳不许自拼"
    srv2 = _build_server(now)
    expected_start = srv2.recall.on_session_start(now=now)
    assert srv2.handle({"jsonrpc": "2.0", "id": 11, "method": "tools/call",
                        "params": {"name": "latent_session_start"}}, now=now
                       )["result"]["content"][0]["text"] == expected_start, \
        "latent_session_start 必须原样返回 on_session_start 的结果"

    # 5.【变异靶心·错误分层】未知工具→协议错误；工具内部失败→isError 结果
    unknown = call("no_such_tool")
    assert "error" in unknown and "result" not in unknown, \
        f"未知工具是协议错误、不是 isError 结果：{unknown}"
    assert unknown["error"]["code"] == E_METHOD_NOT_FOUND
    empty_srv = MemoryServer(index=MemoryIndex().build())
    r5 = empty_srv.handle({"jsonrpc": "2.0", "id": 12, "method": "tools/call",
                           "params": {"name": "latent_session_start"}}, now=now)
    assert "result" in r5 and r5["result"]["isError"] is True, \
        f"工具执行失败该回 isError 结果而不是协议错误——模型要看得到失败原因：{r5}"
    # 5b.【Kelivo 偶发空结果·读取异常只重试一次】生产变异：删掉
    #     `_tool_session_start` 的重试分支 → 这条会拿不到响应；把次数改成 0 或 2，
    #     calls 断言会红。替身只代替现场拿不到的短暂文件读取异常，协议包装与
    #     MemoryServer 都走真实实现。
    class FlakyRecall:
        def __init__(self):
            self.calls = 0

        def on_session_start(self, now=None):
            self.calls += 1
            if self.calls == 1:
                raise OSError("合成的短暂读取失败")
            return "【合成召回】第二次读取成功"

    retry_srv = MemoryServer(index=MemoryIndex().build())
    retry_srv.recall = FlakyRecall()
    try:
        retry_res = retry_srv.handle(
            {"jsonrpc": "2.0", "id": 120, "method": "tools/call",
             "params": {"name": "latent_session_start"}}, now=now)
    except OSError:
        retry_res = None
    assert retry_res is not None and retry_res["result"]["isError"] is False, \
        "session_start 第一次读取异常后该自动重试，不该让客户端拿到空响应"
    assert retry_res["result"]["content"][0]["text"] == "【合成召回】第二次读取成功", \
        "重试成功仍须逐字返回底层结果"
    assert retry_srv.recall.calls == 2, "读取异常只自动重试一次"

    # 5c.【重试仍失败必须有明确错误】生产变异：第二次异常不转 ToolError →
    #     这条拿不到协议响应；重试超过一次则 calls 断言红。
    class SequenceRecall:
        def __init__(self, outcomes):
            self.outcomes = list(outcomes)
            self.calls = 0

        def on_session_start(self, now=None):
            outcome = self.outcomes[self.calls]
            self.calls += 1
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    twice_srv = MemoryServer(index=MemoryIndex().build())
    twice_srv.recall = SequenceRecall([OSError("第一次"), OSError("第二次")])
    try:
        twice_response = twice_srv.handle(
            {"jsonrpc": "2.0", "id": 121, "method": "tools/call",
             "params": {"name": "latent_session_start"}}, now=now)
    except OSError:
        twice_response = None
    assert twice_response is not None, "连续两次读取异常也必须返回协议响应，不能空着断开"
    twice = twice_response["result"]
    assert twice["isError"] is True and twice["content"][0]["text"], \
        "重试仍失败要回非空 isError"
    assert "session_start" in twice["content"][0]["text"]
    assert "自动重试 1 次" in twice["content"][0]["text"]
    assert "OSError" in twice["content"][0]["text"] and twice_srv.recall.calls == 2

    # 5d.【错误要指对方向】JSONL 解析失败不能笼统说成路径问题。生产变异：
    #     删掉 JSONDecodeError 分类 → “不是合法 JSON”断言红。
    decode_srv = MemoryServer(index=MemoryIndex().build())
    decode_srv.recall = SequenceRecall([
        json.JSONDecodeError("合成 JSON 错误", "{", 1),
        json.JSONDecodeError("合成 JSON 错误", "{", 1),
    ])
    decoded = decode_srv.handle(
        {"jsonrpc": "2.0", "id": 122, "method": "tools/call",
         "params": {"name": "latent_session_start"}}, now=now)["result"]
    assert decoded["isError"] is True and "不是合法 JSON" in decoded["content"][0]["text"], \
        "会话线索 JSON 坏了就直说解析失败，别把人引去查限流或空库"

    # 5e. 路径不存在与权限不足分开说；两者都是用户能直接排查的文件问题。
    for exc_type, want in ((FileNotFoundError, "路径错误"), (PermissionError, "权限不足")):
        classified_srv = MemoryServer(index=MemoryIndex().build())
        classified_srv.recall = SequenceRecall([exc_type("第一次"), exc_type("第二次")])
        classified = classified_srv.handle(
            {"jsonrpc": "2.0", "id": 123, "method": "tools/call",
             "params": {"name": "latent_session_start"}}, now=now)["result"]
        assert classified["isError"] is True and want in classified["content"][0]["text"], \
            f"{exc_type.__name__} 应明确提示 {want}，实际：{classified}"

    encoding_srv = MemoryServer(index=MemoryIndex().build())
    encoding_srv.recall = SequenceRecall([
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "合成"),
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "合成"),
    ])
    encoding_result = encoding_srv.handle(
        {"jsonrpc": "2.0", "id": 124, "method": "tools/call",
         "params": {"name": "latent_session_start"}}, now=now)["result"]
    assert encoding_result["isError"] is True and \
        "编码读取失败" in encoding_result["content"][0]["text"], \
        "文件编码错误要与普通读取失败分开说"

    # 5f. 真空库是确定的业务结果，不是偶发读取失败：一次就报空库，不重试。
    none_srv = MemoryServer(index=MemoryIndex().build())
    none_srv.recall = SequenceRecall([None, "不该调用第二次"])
    none_result = none_srv.handle(
        {"jsonrpc": "2.0", "id": 125, "method": "tools/call",
         "params": {"name": "latent_session_start"}}, now=now)["result"]
    assert none_result["isError"] is True and "记忆库是空的" in none_result["content"][0]["text"]
    assert none_srv.recall.calls == 1, "真空库不该自动重试"

    # 5g.【最终错误边界】不在预期列表里的异常也不能冲破 HTTP/stdio，让客户端
    #     只看见空白。生产变异：删掉 `_call_tool` 的最终 except → 这条拿不到响应。
    unexpected_srv = MemoryServer(index=MemoryIndex().build())
    unexpected_calls = {"n": 0}

    def boom(args, now=None):
        unexpected_calls["n"] += 1
        raise RuntimeError("合成敏感正文不应外泄")

    unexpected_srv._handlers = lambda: {"latent_session_start": boom}
    try:
        unexpected = unexpected_srv.handle(
            {"jsonrpc": "2.0", "id": 126, "method": "tools/call",
             "params": {"name": "latent_session_start"}}, now=now)
    except RuntimeError:
        unexpected = None
    assert unexpected is not None and unexpected["result"]["isError"] is True, \
        "未预料异常也必须回 isError，不能断开成空响应"
    unexpected_text = unexpected["result"]["content"][0]["text"]
    assert unexpected_text and "session_start" in unexpected_text and "RuntimeError" in unexpected_text
    assert "合成敏感正文" not in unexpected_text and unexpected_calls["n"] == 1, \
        "异常正文不外泄，handler 也不自动重放"

    # 5h.【防过度重试】写工具第一次可能已经落盘，只是回响应时失败；最终兜底
    #     可以把错误说出来，但绝不能为了“自动恢复”再调用一次。
    write_srv = MemoryServer(index=MemoryIndex().build())
    write_calls = {"n": 0}

    def write_boom(args, now=None):
        write_calls["n"] += 1
        raise RuntimeError("合成的写后失败")

    write_srv._handlers = lambda: {"latent_append": write_boom}
    write_failed = write_srv.handle(
        {"jsonrpc": "2.0", "id": 127, "method": "tools/call",
         "params": {"name": "latent_append", "arguments": {"text": "合成"}}},
        now=now)["result"]
    assert write_failed["isError"] is True and write_failed["content"][0]["text"]
    assert write_calls["n"] == 1, "写工具失败只报错，不自动重放"
    #    参数非法（缺 query）同样走 isError，不是崩
    assert call("latent_search", {})["result"]["isError"] is True
    #    arguments 不是对象 → 协议错误
    bad = srv.handle({"jsonrpc": "2.0", "id": 13, "method": "tools/call",
                      "params": {"name": "latent_search", "arguments": "字符串"}})
    assert bad["error"]["code"] == E_INVALID_PARAMS

    # 6. latent_thread_close 真写进 store，且下一次 latent_session_start 能带回来（一条完整链路）
    ok = call("latent_thread_close", {"window": 7, "current_state": "花买好了，周末的事没定。",
                               "topics": ["阳台的花"], "open_loops": ["周末去哪还没定"],
                               "started_at": now - 3600})
    assert ok["result"]["isError"] is False and "第 7 个窗口" in ok["result"]["content"][0]["text"]
    assert srv.thread_store.latest().window == 7
    started = call("latent_session_start")["result"]["content"][0]["text"]
    assert started.startswith("【上次会话】") and "花买好了" in started, \
        "latent_thread_close 的历史快照该被下一次 latent_session_start 带回来"
    assert "周末去哪还没定" not in started, "旧 open_loops 不再冒充当前未解决集合"
    #    当下状态为空 → 业务校验拦下，走 isError（病灶迁移纪律穿透到协议层）
    assert call("latent_thread_close", {"window": 8, "current_state": "  "})["result"]["isError"] is True

    # 6b.【未解决兼容与严格复合场景】旧 payload 不截胡；严格只卡缺字段；
    #     参数错／清单坏都让正文必达，并给不同失败类型。
    def unresolved_call(server, name, arguments, mid=61):
        return server.handle({"jsonrpc": "2.0", "id": mid, "method": "tools/call",
                              "params": {"name": name, "arguments": arguments}}, now=now)["result"]
    def append_args(text, ops_marker=False):
        result = {"text": text, "current_state": "仍在处理。",
                  "indexEvidence": [{"type": "event", "quote": text}]}
        if ops_marker is not False:
            result["unresolvedOps"] = ops_marker
        return result
    with tempfile.TemporaryDirectory() as td_u:
        soft = MemoryServer(index=MemoryIndex().build(), thread_store=ThreadStore(),
                            corpus_dir=td_u)
        old = unresolved_call(soft, "latent_append", append_args("旧客户端正文仍能写。"))
        assert old["isError"] is False and "unresolvedStatus=not_reviewed" in old["content"][0]["text"]
        assert list((Path(td_u) / "timeline").glob("*.md")), "旧调用的正文必须必达"

    with tempfile.TemporaryDirectory() as td_u:
        strict = MemoryServer(index=MemoryIndex().build(), thread_store=ThreadStore(),
                              corpus_dir=td_u, require_unresolved_review=True)
        missing = unresolved_call(strict, "latent_append", append_args("缺字段该被严格模式拦下。"))
        assert missing["isError"] is True and not (Path(td_u) / "timeline").exists()
        opened = unresolved_call(strict, "latent_append", append_args(
            "住宿还没有确认。", [{"action": "open", "summary": "住宿还没有确认"}]))
        assert opened["isError"] is False and "unresolvedStatus=updated" in opened["content"][0]["text"]
        opening = unresolved_call(strict, "latent_session_start", {})["content"][0]["text"]
        assert opening.startswith("【当前未解决】") and "住宿还没有确认" in opening
        (Path(td_u) / UNRESOLVED_FILENAME).write_text("# 未解决\n坏行\n", encoding="utf-8")
        damaged = unresolved_call(strict, "latent_append", append_args(
            "清单坏了正文也必须写。", [{"action": "none"}]))
        assert damaged["isError"] is False and "失败类型=store_unavailable" in damaged["content"][0]["text"]
        assert "清单坏了正文也必须写" in next((Path(td_u) / "timeline").glob("*.md")).read_text(encoding="utf-8")
        (Path(td_u) / UNRESOLVED_FILENAME).write_text(
            "# 未解决\n\n<!-- 一行一条；关闭时删除整行，不写已完成记录。 -->\n", encoding="utf-8")
        invalid = unresolved_call(strict, "latent_append", append_args(
            "错 ID 也不能吞正文。", [{"action": "close", "id": "U-0000000000000000"}]))
        assert invalid["isError"] is False and "失败类型=request_invalid" in invalid["content"][0]["text"]

    # 7. 未知 method 走协议错误；未知通知（无 id）静默忽略，不回野生错误
    assert srv.handle({"jsonrpc": "2.0", "id": 14, "method": "resources/list"}
                      )["error"]["code"] == E_METHOD_NOT_FOUND
    assert srv.handle({"jsonrpc": "2.0", "method": "notifications/cancelled"}) is None

    # 8. stdio 传输往返：坏行跳过不崩，好行逐条回
    import io
    srv3 = _build_server(now)
    inp = io.StringIO('{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}\n'
                      '这不是 json\n'
                      '\n'
                      '{"jsonrpc":"2.0","method":"notifications/initialized"}\n'
                      '{"jsonrpc":"2.0","id":2,"method":"tools/list"}\n')
    out = io.StringIO()
    srv3.serve_stdio(stdin=inp, stdout=out)
    lines = [json.loads(x) for x in out.getvalue().splitlines() if x.strip()]
    assert [m["id"] for m in lines] == [1, 2], f"坏行该跳过、通知不回响应：{lines}"

    # 9.【变异靶心：stdio 必须按 UTF-8 收发】2026.07.31 真机实测出来的 bug——
    #    Windows 上 sys.stdin 按系统区域编码（简中 cp936）解码，中文 query 变乱码，
    #    分词匹配不上 → 分数全平 → 检索退化成"按加载顺序返回前几块"，不报错不崩，
    #    看起来像检索质量差，实际上根本没查。这里用真实 UTF-8 字节流走一遍全程
    assert _utf8_text_stream(io.BytesIO(b"")).encoding == "utf-8", "stdio 必须锁死 UTF-8"
    srv4 = _build_server(now)
    payload = ('{"jsonrpc":"2.0","id":1,"method":"tools/call","params":'
               '{"name":"latent_search","arguments":{"query":"薄荷"}}}\n')
    out4 = io.BytesIO()
    #    两个包装流都要留引用：被 GC 掉时 TextIOWrapper 会顺手关掉底层 BytesIO
    in_s = _utf8_text_stream(io.BytesIO(payload.encode("utf-8")))
    out_s = _utf8_text_stream(out4, write=True)
    srv4.serve_stdio(stdin=in_s, stdout=out_s)
    out_s.flush()
    got = json.loads(out4.getvalue().decode("utf-8"))
    assert got["result"]["isError"] is False and "薄荷" in got["result"]["content"][0]["text"], \
        f"中文 query 该原样穿过 stdio 并命中，实际 {got}"

    # 9.【变异靶心：写回当场可查 + 没落点明确拒写】latent_append 是记忆库自己
    #    生长的那半支笔（thread 是会话状态层，这是正文层）
    from pathlib import Path as _P
    call = lambda s, name, a, n: s.handle(
        {"jsonrpc": "2.0", "id": 77, "method": "tools/call",
         "params": {"name": name, "arguments": a}}, now=n)["result"]
    #    没配语料目录：明确 isError，不静默写进内存了事（内存态的"记住了"会随
    #    进程一起死，那是"失败得像成功"）
    r9 = call(srv, "latent_append",
              {"text": "x", "current_state": "y"}, now)
    assert r9["isError"] is True and "语料目录" in r9["content"][0]["text"]
    with tempfile.TemporaryDirectory() as td:
        fallback_root9 = _P(td) / "index"
        fallback_root9.mkdir()
        manual9 = fallback_root9 / "topic_旧主题_2026-06-01.md"
        manual9.write_text("已有人工摘要，不迁移不改写。\n", encoding="utf-8")
        prior_auto9 = fallback_root9 / "window_01_2027-01-15_item_07.md"
        prior_auto9.write_text("已有自动摘要，顺号不重排。\n", encoding="utf-8")
        existing_index9 = {p.name: p.read_bytes() for p in fallback_root9.iterdir()}
        srv9 = MemoryServer(index=MemoryIndex().build(), thread_store=ThreadStore(),
                            corpus_dir=td, weights_path=_P(td) / ".weights.json")
        #    schema 展平后两种组合仍由服务端守门：空参数必须明确失败，且不落文件。
        before_empty_append = {p.relative_to(td): p.read_bytes()
                               for p in _P(td).rglob("*") if p.is_file()}
        empty_append = call(srv9, "latent_append", {}, now)
        assert empty_append["isError"] is True and empty_append["content"][0]["text"], \
            "平面 schema 不等于放弃校验：空参数必须得到明确 isError"
        assert ({p.relative_to(td): p.read_bytes()
                 for p in _P(td).rglob("*") if p.is_file()} == before_empty_append), \
            "空参数被拒后不得新增或改动文件"
        #    缺当下状态 → isError（病灶迁移在写入口强制），且没有文件落盘
        before_missing_state = {p.relative_to(td): p.read_bytes()
                                for p in _P(td).rglob("*") if p.is_file()}
        bad = call(srv9, "latent_append", {"text": "她说了件事"}, now)
        assert bad["isError"] is True and "当下状态" in bad["content"][0]["text"]
        assert ({p.relative_to(td): p.read_bytes()
                 for p in _P(td).rglob("*") if p.is_file()} == before_missing_state), \
            "被拒的写回不该新增或改动文件"
        #    正常写回 → 落盘 + 本会话立刻可查（不重建索引就查不到）
        args9 = _append_args("约好周末去看鲸头鹳，她念叨了一个月。",
                             "约定成立，票还没买。")
        args9["window"] = 1
        ok9 = call(srv9, "latent_append", args9, now)
        assert ok9["isError"] is False and "第 1 个窗口" in ok9["content"][0]["text"]
        assert "兼容路径" in ok9["content"][0]["text"] and "--index-dir" in ok9["content"][0]["text"]
        fallback_files = list((_P(td) / "index").glob("*.md"))
        assert len(fallback_files) == 3, "未配 --index-dir 时应在 <corpus>/index/ 只新增一份"
        assert (fallback_root9 / "window_01_2027-01-15_item_08.md").is_file(), \
            "已有自动摘要要从最大顺号后新增，不重排"
        assert all((fallback_root9 / name).read_bytes() == content
                   for name, content in existing_index9.items()), \
            "兼容落点不得迁移、覆盖或重排已有 corpus/index/ 文件"
        hit = call(srv9, "latent_search", {"query": "鲸头鹳"}, now)
        assert hit["isError"] is False and "约定成立" in hit["content"][0]["text"], \
            "写回之后当场就要能查到——不然模型刚说'记下了'转头就'没有记录'"
        before_bad_index = {p.relative_to(fallback_root9): p.read_bytes()
                            for p in fallback_root9.rglob("*") if p.is_file()}
        bad_format = _append_args("她说今天想早点回家。", "还没决定几点走。")
        bad_format["indexSummaries"][0]["summary"] = "只有一行"
        format_deferred = call(srv9, "latent_append", bad_format, now)
        after_bad_index = {p.relative_to(fallback_root9): p.read_bytes()
                           for p in fallback_root9.rglob("*") if p.is_file()}
        assert format_deferred["isError"] is False and "indexStatus=pending" in format_deferred["content"][0]["text"], \
            "格式失败时必须保留正文，并明确没有写入索引层"
        assert after_bad_index == before_bad_index, "格式失败不得把未验证摘要写进索引层"
        assert call(srv9, "latent_search", {"query": "早点回家"}, now)["isError"] is False, \
            "摘要格式失败后，原始正文仍须当场可检索"
        bad_evidence = _append_args("她说今天很难过。", "现在已经缓和。")
        bad_evidence["indexSummaries"][0]["evidenceTerms"].append("安全感")
        deferred_evidence = call(srv9, "latent_append", bad_evidence, now)
        assert deferred_evidence["isError"] is False and "越界词：安全感" in deferred_evidence["content"][0]["text"], \
            "证据越界时必须拒绝索引摘要而不丢正文"
        no_summary = call(srv9, "latent_append",
                          {"text": "她说下周要去复查。", "current_state": "日期还没定。"}, now)
        assert no_summary["isError"] is False and "未提供 indexEvidence" in no_summary["content"][0]["text"] \
            and "indexStatus=pending" in no_summary["content"][0]["text"], \
            "未提供索引证据时，正文写回必须成功且说明降级原因"
        pending_id = re.search(r"recordId=([0-9a-f]{16})", no_summary["content"][0]["text"])
        assert pending_id, "pending 回执必须给稳定 recordId"
        timeline_before_repair = {p.relative_to(td): p.read_bytes()
                                  for p in (_P(td) / "timeline").rglob("*") if p.is_file()}
        index_before_missing_evidence = {p.relative_to(td): p.read_bytes()
                                         for p in fallback_root9.rglob("*") if p.is_file()}
        missing_evidence = call(srv9, "latent_append", {"recordId": pending_id.group(1)}, now)
        assert missing_evidence["isError"] is True \
            and "indexEvidence 必须是非空数组" in missing_evidence["content"][0]["text"], \
            "只传 recordId 必须明确要求补 indexEvidence"
        assert ({p.relative_to(td): p.read_bytes()
                 for p in fallback_root9.rglob("*") if p.is_file()} == index_before_missing_evidence), \
            "缺 indexEvidence 的补索引请求不得写入文件"
        repair = call(srv9, "latent_append", {
            "recordId": pending_id.group(1),
            "indexEvidence": [
                {"type": "event", "quote": "她说下周要去复查"},
                {"type": "state", "quote": "日期还没定"},
            ],
        }, now)
        timeline_after_repair = {p.relative_to(td): p.read_bytes()
                                 for p in (_P(td) / "timeline").rglob("*") if p.is_file()}
        assert repair["isError"] is False and "indexStatus=indexed" in repair["content"][0]["text"]
        assert timeline_after_repair == timeline_before_repair, \
            "按 recordId 补索引不得重复或改写正文"
        repaired_index = list(fallback_root9.glob(f"*_record_{pending_id.group(1)}_item_*.md"))
        assert repaired_index, \
            "补索引必须留下可由 recordId 识别的索引文件"
        index_before_repeat = {p.name: p.read_bytes() for p in repaired_index}
        repeated_repair = call(srv9, "latent_append", {
            "recordId": pending_id.group(1),
            "indexEvidence": [{"type": "event", "quote": "她说下周要去复查"}],
        }, now)
        index_after_repeat = {p.name: p.read_bytes() for p in
                              fallback_root9.glob(f"*_record_{pending_id.group(1)}_item_*.md")}
        assert repeated_repair["isError"] is False and "已有索引" in repeated_repair["content"][0]["text"]
        assert index_after_repeat == index_before_repeat, "重复补索引必须幂等，不能追加第二份"

    # 9w.【--write-dir 真接线：正文搬家、检索照读、清理跟着搬】
    #     判据（先写下来再数）：配了 write_dir 之后 ① <corpus>/timeline 逐字节零变化，
    #     ② 正文只出现在 write_dir 下且窗口号仍按整个语料续排，③ 同一条 latent_search
    #     查得回来（落点在 --corpus 之内，检索范围不变），④ latent_cleanup preview 认得
    #     出它。采集条件：stdlib only、无 embedding、固定 now、临时目录。
    #     变异：删掉 MemoryServer 对 plan_append_record 的 write_dir 透传 → ①② 转红；
    #     把 _tool_memory_cleanup 的 timeline_root 改回写死 <corpus>/timeline → ④ 转红。
    with tempfile.TemporaryDirectory() as td:
        corpusw = _P(td) / "corpus"
        notebook = corpusw / "vps" / "notebook" / "timeline"
        indexw = corpusw / "vps" / "notebook" / "index"
        host_timeline = corpusw / "timeline"
        host_timeline.mkdir(parents=True)
        (host_timeline / "window_01_2026-06-17.md").write_text(
            "# 第1个窗口 · 2026-06-17\n\n## 2026-06-17 记\n"
            "宿主客户端自动记的那一档。\n当下状态：还在记。\n", encoding="utf-8")
        notebook.mkdir(parents=True)
        indexw.mkdir(parents=True)
        source_dirsw, loaderw = make_corpus_loader(corpusw, indexw)
        srvw = MemoryServer(index=loaderw(), thread_store=ThreadStore(),
                            corpus_dir=corpusw, index_dir=indexw, write_dir=notebook,
                            source_dirs=source_dirsw, loader=loaderw)
        host_before = {p.name: p.read_bytes() for p in host_timeline.rglob("*") if p.is_file()}
        okw = call(srvw, "latent_append",
                   _append_args("她把电子琴修好了，说周末弹给我听。", "约定成立，还没弹。"),
                   now)
        assert okw["isError"] is False, f"配了 write_dir 的正常写回不该失败：{okw}"
        assert {p.name: p.read_bytes() for p in host_timeline.rglob("*") if p.is_file()} \
            == host_before, "配了 write_dir 之后，正文一个字节都不该再落进 <corpus>/timeline"
        written = sorted(notebook.glob("*.md"))
        assert len(written) == 1 and written[0].name.startswith("window_02_"), \
            f"正文该只落在 write_dir 下、窗口号按整个语料续排：{[p.name for p in written]}"
        hitw = call(srvw, "latent_search", {"query": "电子琴"}, now)
        assert hitw["isError"] is False and "电子琴" in hitw["content"][0]["text"], \
            f"搬走的正文仍在 --corpus 之内，检索必须照读得到：{hitw}"
        record_idw = re.search(r"recordId=([0-9a-f]{16})", okw["content"][0]["text"])
        assert record_idw, f"回执里要有 recordId：{okw['content'][0]['text']}"
        previeww = call(srvw, "latent_cleanup",
                        {"recordId": record_idw.group(1), "action": "preview"}, now)
        assert previeww["isError"] is False, \
            f"清理根没跟着 write_dir 搬时这里会说“不在 timeline 目录”：{previeww}"

    # 9a.【latent_append 预检与可修复报错】判据与采集条件：Windows 11、Python
    #     3.12.10、main@29a1b094 起分支、stdlib only。通过＝预检复用真写计划，
    #     timeline/index/未解决 sidecar/内存索引/written_paths 零变化；参数错误同时给
    #     写错与写对；把预检误接 commit 或 unresolved apply 时本段立即红。
    with tempfile.TemporaryDirectory() as td:
        root9a = _P(td)
        corpus9a, index9a = root9a / "corpus", root9a / "index"
        corpus9a.mkdir()
        index9a.mkdir()
        source_dirs9a, loader9a = make_corpus_loader(corpus9a, index9a)
        s9a = MemoryServer(
            index=loader9a(), thread_store=ThreadStore(), corpus_dir=corpus9a,
            index_dir=index9a, source_dirs=source_dirs9a, loader=loader9a)

        def snapshot9a():
            return {p.relative_to(root9a): p.read_bytes()
                    for p in root9a.rglob("*") if p.is_file()}

        before_files9a = snapshot9a()
        before_chunks9a = list(s9a.index.chunks)
        before_meta9a = [dict(row) for row in s9a.index.meta]
        before_written9a = set(s9a.written_paths)
        original_commit9a = globals()["commit_memory_files"]
        original_apply9a = s9a.unresolved_store.apply

        def forbidden_commit9a(*_args, **_kwargs):
            raise AssertionError("预检误调用 commit_memory_files")

        def forbidden_apply9a(*_args, **_kwargs):
            raise AssertionError("预检误调用 UnresolvedStore.apply")

        globals()["commit_memory_files"] = forbidden_commit9a
        s9a.unresolved_store.apply = forbidden_apply9a
        indexed_args9a = {
            "mode": "preflight",
            "text": "她说下周要去复查。",
            "current_state": "日期还没定。",
            "indexEvidence": [
                {"type": "event", "quote": "她说下周要去复查"},
                {"type": "state", "quote": "日期还没定"},
            ],
            "unresolvedOps": [{"action": "none"}],
        }
        try:
            ready9a = call(s9a, "latent_append", indexed_args9a, now)
        finally:
            globals()["commit_memory_files"] = original_commit9a
            s9a.unresolved_store.apply = original_apply9a
        ready_text9a = ready9a["content"][0]["text"]
        assert ready9a["isError"] is False and "preflightStatus=ready" in ready_text9a \
            and "indexStatus=indexed" in ready_text9a and "本次零写入" in ready_text9a
        ready_id9a = re.search(r"recordId=([0-9a-f]{16})", ready_text9a).group(1)
        assert snapshot9a() == before_files9a \
            and s9a.index.chunks == before_chunks9a and s9a.index.meta == before_meta9a \
            and s9a.written_paths == before_written9a, \
            "预检必须连临时文件、未解决清单、内存索引与 written_paths 都不改"

        write_args9a = dict(indexed_args9a)
        write_args9a.pop("mode")
        written9a = call(s9a, "latent_append", write_args9a, now)
        assert written9a["isError"] is False and f"recordId={ready_id9a}" in written9a["content"][0]["text"], \
            "同参数的预检与真写必须共用计划，预计 recordId 不能漂"

        pending_args9a = {
            "mode": "preflight", "text": "她改约到星期五。",
            "current_state": "对方还没回复。", "unresolvedOps": [{"action": "none"}],
        }
        before_pending_preflight9a = snapshot9a()
        pending_preview9a = call(s9a, "latent_append", pending_args9a, now + 1)
        assert pending_preview9a["isError"] is False \
            and "indexStatus=pending" in pending_preview9a["content"][0]["text"] \
            and "补索引" in pending_preview9a["content"][0]["text"]
        assert snapshot9a() == before_pending_preflight9a, "pending 预检也必须零写入"
        pending_write_args9a = dict(pending_args9a)
        pending_write_args9a.pop("mode")
        pending_written9a = call(s9a, "latent_append", pending_write_args9a, now + 1)
        pending_id9a = re.search(
            r"recordId=([0-9a-f]{16})", pending_written9a["content"][0]["text"]).group(1)

        repair_args9a = {
            "mode": "preflight", "recordId": pending_id9a,
            "indexEvidence": [{"type": "event", "quote": "她改约到星期五"}],
        }
        before_repair_preflight9a = snapshot9a()
        repair_preview9a = call(s9a, "latent_append", repair_args9a, now + 1)
        assert repair_preview9a["isError"] is False \
            and "补索引" in repair_preview9a["content"][0]["text"] \
            and "preflightStatus=ready" in repair_preview9a["content"][0]["text"]
        assert snapshot9a() == before_repair_preflight9a, "补索引预检不得写 index"
        unknown_preview9a = call(s9a, "latent_append", {
            "mode": "preflight", "recordId": "0000000000000000",
            "indexEvidence": [{"type": "event", "quote": "不存在"}]}, now + 1)
        assert unknown_preview9a["isError"] is True \
            and all(label in unknown_preview9a["content"][0]["text"]
                    for label in ("没有找到", "写错：", "写对："))
        repair_write_args9a = dict(repair_args9a)
        repair_write_args9a.pop("mode")
        assert call(s9a, "latent_append", repair_write_args9a, now + 1)["isError"] is False
        indexed_preview9a = call(s9a, "latent_append", repair_args9a, now + 1)
        assert indexed_preview9a["isError"] is False \
            and "已有索引" in indexed_preview9a["content"][0]["text"] \
            and "无需重复" in indexed_preview9a["content"][0]["text"]

        bad_calls9a = [
            {"mode": "preflight", "text": "缺状态"},
            {"mode": "preflight", "text": "她说下周要去复查。",
             "current_state": "日期还没定。",
             "indexEvidence": [{"type": "event", "quote": "她计划下周复查"}]},
            {"mode": "preflight", "text": "只有一行", "current_state": "仍在处理",
             "indexSummaries": [{"summary": "只有一行", "evidenceTerms": ["只有一行"]}]},
            {"mode": "preflight", "text": "原文", "current_state": "有效",
             "indexEvidence": [{"type": "event", "quote": "原文"}],
             "indexSummaries": [{"summary": "旧摘要", "evidenceTerms": ["原文"]}]},
        ]
        before_bad_calls9a = snapshot9a()
        for bad_args9a in bad_calls9a:
            bad_result9a = call(s9a, "latent_append", bad_args9a, now + 2)
            bad_text9a = bad_result9a["content"][0]["text"]
            assert bad_result9a["isError"] is True \
                and "写错：" in bad_text9a and "写对：" in bad_text9a, bad_text9a
        strict9a = MemoryServer(
            index=loader9a(), thread_store=ThreadStore(), corpus_dir=corpus9a,
            index_dir=index9a, source_dirs=source_dirs9a, loader=loader9a,
            require_unresolved_review=True)
        strict_bad9a = call(strict9a, "latent_append", {
            "mode": "preflight", "text": "严格模式", "current_state": "待复核"}, now + 2)
        assert strict_bad9a["isError"] is True \
            and all(label in strict_bad9a["content"][0]["text"]
                    for label in ("unresolvedOps", "写错：", "写对："))
        assert snapshot9a() == before_bad_calls9a, "所有参数错误都必须在零写入状态返回"

    # 9b.【工具级精准清理】判据与采集条件：stdlib 服务端、临时双层语料，先用当前
    #     latent_append 在同一窗口写两条（目标有 recordId 关联索引、相邻记录保留），
    #     再覆盖 preview 零写入、旧令牌失效、未解决引用阻断、sidecar 去键、隔离备份
    #     不回灌，以及无索引 pending 记录也能按 ID 精确清理。通过＝每条断言均成立。
    with tempfile.TemporaryDirectory() as td:
        corpus9b = _P(td) / "corpus"
        index9b = _P(td) / "index"
        corpus9b.mkdir()
        index9b.mkdir()
        source_dirs9b, loader9b = make_corpus_loader(corpus9b, index9b)
        weights9b = corpus9b / ".weights.json"
        retractions9b = corpus9b / ".retractions.json"
        entities9b = corpus9b / ".entities.json"
        s9b = MemoryServer(
            index=loader9b(), thread_store=ThreadStore(), corpus_dir=corpus9b,
            index_dir=index9b, source_dirs=source_dirs9b, loader=loader9b,
            weights_path=weights9b, retractions_path=retractions9b,
            entities_path=entities9b)
        first9b = call(s9b, "latent_append", {
            "text": "误写：把蓝色雨伞记成了绿色雨伞。", "current_state": "刚写入，等待核对。",
            "indexEvidence": [
                {"type": "event", "quote": "把蓝色雨伞记成了绿色雨伞"},
                {"type": "state", "quote": "刚写入，等待核对"},
            ]}, now)
        second9b = call(s9b, "latent_append", {
            "text": "相邻正确记录：门票放在书桌抽屉。", "current_state": "仍然有效。",
            "indexEvidence": [
                {"type": "event", "quote": "门票放在书桌抽屉"},
                {"type": "state", "quote": "仍然有效"},
            ]}, now + 1)
        rid9b = re.search(r"recordId=([0-9a-f]{16})", first9b["content"][0]["text"]).group(1)
        keep9b = re.search(r"recordId=([0-9a-f]{16})", second9b["content"][0]["text"]).group(1)
        target_path9b = next((corpus9b / "timeline").glob("*.md"))
        target_section9b = _record_section_plan(target_path9b, rid9b)
        target_hash9b = next(iter(target_section9b["hashes"]))
        weights9b.write_text(json.dumps({target_hash9b: 2.5}, ensure_ascii=False), encoding="utf-8")
        retractions9b.write_text(json.dumps({target_hash9b: {"reason": "测试"}}, ensure_ascii=False),
                                 encoding="utf-8")
        entities9b.write_text(json.dumps({target_hash9b: ["雨伞"]}, ensure_ascii=False),
                              encoding="utf-8")

        before_preview9b = {p.relative_to(td): p.read_bytes()
                            for p in _P(td).rglob("*") if p.is_file()}
        preview9b = call(s9b, "latent_cleanup", {"action": "preview", "recordId": rid9b}, now)
        after_preview9b = {p.relative_to(td): p.read_bytes()
                           for p in _P(td).rglob("*") if p.is_file()}
        assert preview9b["isError"] is False and "绿色雨伞" in preview9b["content"][0]["text"]
        assert before_preview9b == after_preview9b, "preview 必须纯读，连隔离目录都不能提前创建"
        token9b = re.search(r"确认令牌：(DELETE-[0-9a-f]{16}-[0-9a-f]{16})",
                            preview9b["content"][0]["text"]).group(1)
        wrong_token9b = call(s9b, "latent_cleanup", {
            "action": "delete", "recordId": rid9b, "confirmation": "DELETE-wrong",
            "reason": "清理误写"}, now)
        unknown9b = call(s9b, "latent_cleanup", {
            "action": "preview", "recordId": "0000000000000000"}, now)
        assert wrong_token9b["isError"] is True and "confirmation 不匹配" in wrong_token9b["content"][0]["text"] \
            and unknown9b["isError"] is True and "没有找到" in unknown9b["content"][0]["text"], \
            "错误确认令牌与未知 recordId 都必须明确失败"
        assert not (corpus9b / ".cleanup-trash").exists(), \
            "确认失败不得提前创建隔离目录或改动活动文件"
        no_reason9b = call(s9b, "latent_cleanup", {
            "action": "delete", "recordId": rid9b, "confirmation": token9b}, now)
        assert no_reason9b["isError"] is True and "reason 必填" in no_reason9b["content"][0]["text"]

        # 外部手工改动发生在 preview 与 delete 之间：即使改的只是文件尾，旧令牌也失效。
        timeline_before_touch9b = target_path9b.read_bytes()
        target_path9b.write_bytes(timeline_before_touch9b + b"\n")
        stale9b = call(s9b, "latent_cleanup", {
            "action": "delete", "recordId": rid9b, "confirmation": token9b,
            "reason": "清理误写"}, now)
        assert stale9b["isError"] is True and "重新 preview" in stale9b["content"][0]["text"], \
            "preview 后文件变化必须令旧 confirmation 失效"
        target_path9b.write_bytes(timeline_before_touch9b)

        _, opened9b = s9b.unresolved_store.apply(
            [{"action": "open", "summary": "先确认雨伞颜色"}],
            f"timeline/{target_path9b.name}#record={rid9b}")
        blocked_preview9b = call(s9b, "latent_cleanup",
                                 {"action": "preview", "recordId": rid9b}, now)
        blocked_token9b = re.search(r"确认令牌：(DELETE-[0-9a-f]{16}-[0-9a-f]{16})",
                                    blocked_preview9b["content"][0]["text"]).group(1)
        blocked9b = call(s9b, "latent_cleanup", {
            "action": "delete", "recordId": rid9b, "confirmation": blocked_token9b,
            "reason": "清理误写"}, now)
        assert blocked9b["isError"] is True and opened9b[0] in blocked9b["content"][0]["text"], \
            "未解决事项仍引用 recordId 时必须点名阻断，不能自动关单"
        s9b.unresolved_store.apply([{"action": "close", "id": opened9b[0]}], "manual")
        ready9b = call(s9b, "latent_cleanup", {"action": "preview", "recordId": rid9b}, now)
        ready_token9b = re.search(r"确认令牌：(DELETE-[0-9a-f]{16}-[0-9a-f]{16})",
                                  ready9b["content"][0]["text"]).group(1)
        deleted9b = call(s9b, "latent_cleanup", {
            "action": "delete", "recordId": rid9b, "confirmation": ready_token9b,
            "reason": "颜色属于误写"}, now)
        assert deleted9b["isError"] is False, deleted9b
        active_text9b = target_path9b.read_text(encoding="utf-8")
        assert "绿色雨伞" not in active_text9b and "门票放在书桌抽屉" in active_text9b, \
            "只移除目标 H2 记录节，相邻记录必须逐字保留"
        assert not list(index9b.glob(f"*_record_{rid9b}_item_*.md")) \
            and list(index9b.glob(f"*_record_{keep9b}_item_*.md")), \
            "只移除目标 recordId 的关联索引"
        for sidecar9b in (weights9b, retractions9b, entities9b):
            assert target_hash9b not in json.loads(sidecar9b.read_text(encoding="utf-8")), \
                f"{sidecar9b.name} 必须同步去掉被清记录的内容哈希"
        trash9b = corpus9b / ".cleanup-trash"
        backup_dirs9b = [p for p in trash9b.iterdir() if p.is_dir()]
        assert len(backup_dirs9b) == 1 and (backup_dirs9b[0] / "manifest.json").is_file()
        assert list(backup_dirs9b[0].glob("*.bak")) and not list(backup_dirs9b[0].glob("*.md")), \
            "隔离副本必须可恢复，但扩展名不能被 Markdown 语料扫描器重新载入"
        reloaded9b = load_corpus(source_dirs9b)
        assert all("绿色雨伞" not in chunk for chunk in reloaded9b.chunks) \
            and any("门票放在书桌抽屉" in chunk for chunk in reloaded9b.chunks), \
            "重载后误写不能从隔离备份回灌，相邻正确记录仍可检索"

        # 同一窗口先删靠前记录后，后续记录会从第二节变成首节、继承 H1 前导文本；
        # 旧 recordId 仍必须能定位到它，不能只剩检索可见、cleanup 却失效。
        keep_preview9b = call(s9b, "latent_cleanup", {"action": "preview", "recordId": keep9b}, now + 1)
        assert keep_preview9b["isError"] is False and "门票放在书桌抽屉" in keep_preview9b["content"][0]["text"], \
            "先删同窗口前一条后，后续记录的旧 recordId 仍必须能 preview"
        keep_token9b = re.search(r"确认令牌：(DELETE-[0-9a-f]{16}-[0-9a-f]{16})",
                                 keep_preview9b["content"][0]["text"]).group(1)
        keep_deleted9b = call(s9b, "latent_cleanup", {
            "action": "delete", "recordId": keep9b, "confirmation": keep_token9b,
            "reason": "相邻记录也需清理"}, now + 1)
        assert keep_deleted9b["isError"] is False and not target_path9b.exists(), \
            "后续记录按旧 recordId 清理后，空窗口文件应一并移除"

        pending9b = call(s9b, "latent_append", {
            "text": "另一条误写：会议被记到了星期二。", "current_state": "等待清理。"}, now + 2)
        pending_rid9b = re.search(r"recordId=([0-9a-f]{16})", pending9b["content"][0]["text"]).group(1)
        assert "indexStatus=pending" in pending9b["content"][0]["text"] \
            and not list(index9b.glob(f"*_record_{pending_rid9b}_item_*.md"))
        pending_preview9b = call(s9b, "latent_cleanup", {
            "action": "preview", "recordId": pending_rid9b}, now + 2)
        pending_token9b = re.search(r"确认令牌：(DELETE-[0-9a-f]{16}-[0-9a-f]{16})",
                                    pending_preview9b["content"][0]["text"]).group(1)
        pending_delete9b = call(s9b, "latent_cleanup", {
            "action": "delete", "recordId": pending_rid9b, "confirmation": pending_token9b,
            "reason": "星期写错"}, now + 2)
        assert pending_delete9b["isError"] is False and "关联索引 0 个" in pending_delete9b["content"][0]["text"], \
            "没有索引的 pending 正文也必须能按稳定 recordId 清理"

    with tempfile.TemporaryDirectory() as td:
        rollback_a9b, rollback_b9b = _P(td) / "a.md", _P(td) / "b.md"
        rollback_a9b.write_bytes(b"A-before")
        rollback_b9b.write_bytes(b"B-before")
        try:
            commit_cleanup_changes({rollback_a9b: None, rollback_b9b: None}, fail_after=1)
            assert False, "故障注入必须触发"
        except OSError:
            pass
        assert rollback_a9b.read_bytes() == b"A-before" \
            and rollback_b9b.read_bytes() == b"B-before", \
            "多目标清理中途失败必须恢复全部调用前字节"

    # 10.【变异靶心：用进撑过重启】权重持久化接线——同一份语料+权重文件，
    #     新起一个 server（模拟客户端重启 stdio 进程），命中过的块权重仍在
    with tempfile.TemporaryDirectory() as td:
        wp = _P(td) / ".weights.json"
        mk = lambda: MemoryServer(index=_build_server(now).index,
                                  thread_store=ThreadStore(),
                                  corpus_dir=td, weights_path=wp)
        s_a = mk()
        call(s_a, "latent_search", {"query": "薄荷"}, now)     # 命中 → 权重落盘
        s_b = mk()                                             # "重启"
        i = next(i for i, c in enumerate(s_b.index.chunks) if "薄荷" in c)
        assert s_b.index.weights[i] > 1.0, \
            "重启后命中过的块权重该还在——不落盘的话用进废退在生产形态下等于没有"

    # 11.【可靠命中门槛】零信号 query → isError + 明确"没有可靠命中"，且文案要
    #     解锁如实回答（instructions 堵的是"没查就说没记录"，查过之后的"没有"
    #     是诚实——两句话必须能共存，不然模型会在两条指令之间僵住）
    miss = call(_build_server(now), "latent_search", {"query": "量子对撞机的运行日志"}, now)
    assert miss["isError"] is True and "没有可靠命中" in miss["content"][0]["text"]
    assert "如实" in miss["content"][0]["text"], "没找到时要明确解锁'如实说没有'"

    # 12.【错误记忆治理闭环：撤回→更正→重启仍生效】
    with tempfile.TemporaryDirectory() as td:
        rp = _P(td) / ".retractions.json"
        mk12 = lambda extra: MemoryServer(
            index=extra, thread_store=ThreadStore(),
            corpus_dir=td, weights_path=_P(td) / ".weights.json",
            retractions_path=rp)
        s12 = mk12(_build_server(now).index)
        #    写 correction 但缺当下状态 → 拒（病灶迁移，更正也不豁免）
        bad12 = call(s12, "latent_correct",
                     {"quote": "保险丝熔断", "reason": "x", "correction": "y"}, now)
        assert bad12["isError"] is True and "当下状态" in bad12["content"][0]["text"]
        #    quote 定位不到 → 明确报错，不猜
        miss12 = call(s12, "latent_correct", {"quote": "烤箱", "reason": "x"}, now)
        assert miss12["isError"] is True
        #    正常更正：撤回旧记录 + 写入更正 → 旧的查不到、新的当场可查
        ok12 = call(s12, "latent_correct",
                    {"quote": "保险丝熔断", "reason": "维修方案已过时",
                     "correction": "咖啡机上月整机换新了，旧机的维修记录不再适用。",
                     "current_state": "新机运行正常。"}, now)
        assert ok12["isError"] is False and "已撤回" in ok12["content"][0]["text"]
        #    【撤回粒度提醒·0 命中 = 文本不变 + 变异靶心（坑 1）】薄荷块与咖啡机块
        #    不共享任何区分性词面 → 没有别的块提到同一说法，提示一个字都不该冒出来。
        #    ⚠ 变异：把 same_claim_survivors 里的 exclude 去掉，更正块正文含"咖啡机"
        #    会命中自己 → 提示恒真 → 这条断言转红。这就是"恒真的提示等于没提示"。
        assert "同一说法" not in ok12["content"][0]["text"], \
            "无其它块提到时不该有提醒（变异靶心：去掉 exclude 更正块自命中→此条转红）"
        after = call(s12, "latent_search", {"query": "咖啡机"}, now)
        assert after["isError"] is False
        assert "保险丝熔断" not in after["content"][0]["text"] and \
               "整机换新" in after["content"][0]["text"], "旧的退出检索、更正当场可查"
        #    【变异靶心：撤回落盘】"重启"（语料从盘上重读 + 账本重载）后撤回仍生效
        s12b = mk12(load_corpus(td))     # 更正记录在盘上；合成旧块不在盘上没关系
        assert s12b.index.retraction_log, "重启后撤回账本该从盘上回来"
        assert json.loads(rp.read_text(encoding="utf-8")), "账本文件要真在盘上（可追溯）"

    # 12d.【撤回粒度提醒·命中多块：撤回是块级、事实是跨块的】旧名散在 3 块，用户只
    #      逐字摘了其中一块 → 撤回成功后，返回文本要点出"库里还有 N 块提到同一说法"。
    #      采集条件（写在断言旁）：合成 20 块 → 更正后 21 块，_distinctive_df()=
    #      max(3,21//5)=4；旧名的区分性 token 命中 4 块（3 史 + 1 更正块），df=4≤4 才算
    #      区分性——⚠ 块数不够会掉进常数支 3、旧名被判成"太常见"而漏报（实施计划坑 2/
    #      自检节点名）。⚠ 更正块含旧名，靠 exclude 排除，不排会命中自己（坑 1）。
    with tempfile.TemporaryDirectory() as td12d:
        idx12d = MemoryIndex()
        idx12d.add("## 遛狗\n那只边牧旺财今天在公园撒欢跑了一下午。", {"heading": "遛狗", "timestamp": now - 300})   # 0
        idx12d.add("## 体检\n旺财上周体检结果一切合格。", {"heading": "体检", "timestamp": now - 200})               # 1
        idx12d.add("## 喂食\n旺财这几天胃口特别好。", {"heading": "喂食", "timestamp": now - 100})                   # 2
        _tails12d = ["买了牛奶", "修好单车", "看场电影", "种盆多肉", "写封书信", "擦净窗户",
                     "煮锅绿豆", "换个灯泡", "读本小说", "叠好衣裳", "浇透花草", "补几只袜",
                     "晒晒被褥", "扫净院落", "炖排骨汤", "烤个蛋糕", "刷完油漆"]              # 灌到 N=20，功能词 df 拉高
        for tail in _tails12d:
            idx12d.add(f"## 杂记·{tail}\n今天顺手记一笔，{tail}。", {"heading": f"杂记·{tail}", "timestamp": now - 1000})
        idx12d.build()
        s12d = MemoryServer(index=idx12d, thread_store=ThreadStore(),
                            corpus_dir=td12d, retractions_path=_P(td12d) / ".retractions.json")
        ok12d = call(s12d, "latent_correct",
                     {"quote": "旺财今天在公园撒欢", "reason": "记错了名字",
                      "correction": "其实那只边牧现在叫来福，不叫旺财了。",
                      "current_state": "现在统一叫来福。"}, now)
        assert ok12d["isError"] is False, ok12d
        t12d = ok12d["content"][0]["text"]
        assert "已撤回" in t12d and "还有 2 块提到同一说法" in t12d, \
            f"撤一块后要点出另外两块仍讲同一说法（块级撤回、事实跨块），实得：{t12d}"
        assert "体检" in t12d and "喂食" in t12d, "提示要给出定位（最近若干条 heading）"
        assert "只提示" in t12d and "自动撤" in t12d, "边界要写进返回：只提示、绝不自动撤（坑 3）"

    # 12c.【同会话 append→撤回，重启仍生效·变异靶心】（2026.08.03 外部缺陷报告，
    #      真机 7232 块语料上抓到）：新建窗口文件的首块，手拼返回值与重启重切的
    #      文件态块不一致（H1 被向下并进首块），撤回账本键成孤儿，「已撤回」只活
    #      一个进程——触发的恰是最自然的用法：刚记完当场改。上面第 12 项盖不到
    #      这一格：它的旧块是合成的、不在盘上，只验账本回来了、验不了**重新对上号**。
    #      变异：append_record 改回手拼 chunk_text → 这段红
    with tempfile.TemporaryDirectory() as td:
        rp = _P(td) / ".retractions.json"
        index12c = _P(td) / "index"
        index12c.mkdir()
        mk12c = lambda: MemoryServer(index=load_corpus((td, index12c)), thread_store=ThreadStore(),
                                     corpus_dir=td, retractions_path=rp, index_dir=index12c,
                                     source_dirs=(td, index12c))
        s_a12c = mk12c()
        call(s_a12c, "latent_append",
             _append_args("她想去的展览在下周三，先记着。", "还没买票。"), now)
        ok12c = call(s_a12c, "latent_correct",
                     {"quote": "# 第1个窗口", "reason": "记错了，是下周四"}, now)
        assert ok12c["isError"] is False and "已撤回" in ok12c["content"][0]["text"]
        s_b12c = mk12c()                               # 重启：语料与账本都从盘上回来
        assert s_b12c.index.retracted, \
            "重启后撤回必须重新对上号（此前这里是空集：账本键是手拼块文本的孤儿）"
        # 本卡冻结边界不包含 latent_correct 同步撤回索引摘要；
        # 这里只保留原靶心：timeline 块的撤回账本重启后仍能对上。

    # 12b.【缺失率标注接线：两条路径都要带上】高缺失率查询无论"查到了"还是
    #      "没查到"，都该把那句可核对的话交给模型——空结果那条尤其重要，它是
    #      模型决定"如实说没找到"还是"拿沾边的记录去圆"的分水岭
    from memory_retrieval import query_miss_rate as _qmr0, MISS_RATE_FLAG as _F0
    srv12b = _build_server(now)
    hit12b = call(srv12b, "latent_search", {"query": "咖啡机 保险丝 熔断 通电"}, now)
    miss12b = call(srv12b, "latent_search", {"query": "量子对撞机的运行日志"}, now)
    assert miss12b["isError"] is True and "核对提示" in miss12b["content"][0]["text"], \
        "空结果那条必须带缺失率标注——它是'如实说没找到'与'拿沾边记录去圆'的分水岭"
    #      **最要紧的一条：高缺失率不许硬拒**。这是本信号定性（专名缺席检测器，
    #      不是事件存在性检测器）直接推出来的纪律——它的误杀全部落在"抽象归纳式
    #      提问"那一档，而那是陪伴场景最有价值的一类问题。有人把标注改成拒绝返回，
    #      行为上是"专门惩罚最该被答好的提问"，而在此之前没有任何断言守着这件事
    #      （变异检查抓出来的缺口）
    mixed = "咖啡机 量子对撞机 报税"          # 高缺失率，但确实有真命中
    assert _qmr0(srv12b.index, mixed) >= _F0, "测试前提：这条要真的触发标注"
    r_mixed = call(srv12b, "latent_search", {"query": mixed}, now)
    assert r_mixed["isError"] is False, \
        "高缺失率查询**不许硬拒**——它只该带标注，判断权留给读得到内容的模型"
    assert "保险丝" in r_mixed["content"][0]["text"], "真命中必须照常返回"
    assert "核对提示" in r_mixed["content"][0]["text"], "同时要带上那句可核对的标注"

    #      有结果时按缺失率决定带不带，不硬加噪声
    from memory_retrieval import query_miss_rate as _qmr, MISS_RATE_FLAG as _F
    if _qmr(srv12b.index, "咖啡机 保险丝 熔断 通电") < _F:
        assert "核对提示" not in hit12b["content"][0]["text"], "低缺失率不该加标注"

    # 13.【图谱实体可插拔·接线靶心（变异：__init__ 不接 entities_path 必红）】
    #     语料目录下有 .entities.json 时 server 要接上并重建——换了说法的关联块
    #     经图谱进结果
    from memory_retrieval import _chunk_key
    def mk13():
        i = MemoryIndex()
        i.add("## 山顶的约定\n那晚在山顶聊到以后，说好要买一台能看土星的家伙。")
        i.add("## 到货\n快递终于送来了，装在阳台，晚上迫不及待试了试。")
        return i.build()
    with tempfile.TemporaryDirectory() as td:
        ep = _P(td) / ".entities.json"
        idx13 = mk13()
        ep.write_text(json.dumps({_chunk_key(idx13.chunks[0]): ["望远镜"],
                                  _chunk_key(idx13.chunks[1]): ["望远镜"]},
                                 ensure_ascii=False), encoding="utf-8")
        s13 = MemoryServer(index=mk13(), thread_store=ThreadStore(), entities_path=ep)
        r13 = call(s13, "latent_search", {"query": "山顶 土星"}, now)
        assert r13["isError"] is False and "阳台" in r13["content"][0]["text"], \
            "server 接上 .entities.json 后，换了说法的关联块该被图谱带回"

    # 14.【部署体检·靶心是"只读"】体检最常在"看起来没接通"的时候跑，那时候往
    #     语料目录里掉一个文件，等于在排查现场留下新变量。跑前跑后整棵目录树的
    #     快照（相对路径 + 大小 + mtime_ns）必须逐字相等——**顺手接一条
    #     weights_path 进 diagnose 里的 server，这条立刻红**，那是最容易被写出来的
    #     sidecar（检索命中就落盘）。
    #     靶子目录**故意叫 corpus/、不叫 memory/**：外层目录名是自由的，服务端认的
    #     是 --corpus 指向哪儿；判定必须靠目录内容，一旦有人拿名字做判断，这里就红
    def _snapshot(root):
        return {str(p.relative_to(root)): (p.is_dir(), p.stat().st_size if p.is_file() else 0,
                                           p.stat().st_mtime_ns)
                for p in sorted(_P(root).rglob("*"))}

    with tempfile.TemporaryDirectory() as td:
        corpus = _P(td) / "corpus"
        (corpus / "timeline").mkdir(parents=True)
        (corpus / "index").mkdir(parents=True)
        (corpus / "timeline" / "window_04_2026-06-17.md").write_text(
            "## 修咖啡机\n加热管不工作，拆开发现保险丝熔断，换上通电正常。\n",
            encoding="utf-8")
        (corpus / "index" / "window_04.md").write_text(
            "## 第4窗摘要\n修好了咖啡机。\n", encoding="utf-8")
        (corpus / UNRESOLVED_FILENAME).write_text(
            "# 未解决\n\n<!-- 一行一条；关闭时删除整行，不写已完成记录。 -->\n",
            encoding="utf-8")
        before = _snapshot(corpus)
        checks = diagnose(corpus, threads_path=_P(td) / "threads.jsonl")
        report = format_doctor_report(checks)
        assert _snapshot(corpus) == before, \
            "体检必须只读：跑完语料目录里多/少/改了东西（最常见的是 .weights.json）"
        assert not list(corpus.rglob(".*")), \
            f"体检不许留 sidecar：{[p.name for p in corpus.rglob('.*')]}"
        assert all(c["level"] != FAIL for c in checks), f"这份语料该全过：{checks}"
        by_title = {c["title"]: c for c in checks}
        assert "兼容模式" in by_title["未解决复核策略"]["detail"]
        assert "有效 0 条" in by_title["未解决清单"]["detail"]
        assert all(UNRESOLVED_FILENAME not in meta.get("source", "")
                   for meta in load_corpus(corpus).meta), "未解决.md 不得混进语料"
        strict_checks = diagnose(corpus, threads_path=_P(td) / "threads.jsonl",
                                 require_unresolved_review=True)
        assert "严格模式" in {c["title"]: c for c in strict_checks}["未解决复核策略"]["detail"]
        by = {c["title"]: c for c in checks}
        assert (by["索引摘要目录"]["level"] == WARN
                and str((corpus / "index").resolve()) in by["索引摘要目录"]["detail"]
                and "推荐" in by["索引摘要目录"]["detail"]), \
            "doctor 未配 --index-dir 时要说清兼容落点与推荐形态"
        assert "索引层 1" in by["建库"]["detail"], "index/ 那层要被认出来"
        assert by["时间戳来源"]["level"] == OK and "mtime" not in by["时间戳来源"]["detail"], \
            "文件名带日期 + 邻层继承，不该有块落 mtime"
        assert by["检索自查"]["level"] == OK, "拿语料自己的标题该查得到"
        assert "没有向语料目录写入任何文件" in report

        # 14b.【独立索引目录·README 不冒充索引】外部 --corpus 不许被出货改写，
        #      所以人工摘要要从第二个只读根接进来。README 故意用 .txt：只有它时
        #      仍该报警；放入真实摘要 .md 后才算有索引层。
        corpus_no_index = _P(td) / "corpus-no-index"
        (corpus_no_index / "timeline").mkdir(parents=True)
        (corpus_no_index / "timeline" / "window_04_2026-06-17.md").write_text(
            "## 修咖啡机\n加热管不工作，拆开发现保险丝熔断，换上通电正常。\n",
            encoding="utf-8")
        index_dir = _P(td) / "bundle" / "memory" / "index"
        index_dir.mkdir(parents=True)
        (index_dir / "README.txt").write_text(
            "这只是写法说明，不是索引摘要。\n", encoding="utf-8")
        before_corpus = _snapshot(corpus_no_index)
        before_index = _snapshot(index_dir)
        without_summary = diagnose(
            corpus_no_index, threads_path=_P(td) / "threads.jsonl", index_dir=index_dir)
        assert _snapshot(corpus_no_index) == before_corpus and \
            _snapshot(index_dir) == before_index, \
            "体检必须同时只读主语料与独立索引目录"
        assert any(c["title"] == "分层" and c["level"] == WARN
                   for c in without_summary), "README.txt 不许冒充索引层"

        (index_dir / "window_04_2026-06-17.md").write_text(
            "## 第4窗摘要\n咖啡机保险丝已经换好。\n", encoding="utf-8")
        with_summary = diagnose(
            corpus_no_index, threads_path=_P(td) / "threads.jsonl", index_dir=index_dir)
        by_with_index = {c["title"]: c for c in with_summary}
        assert "索引层 1" in by_with_index["建库"]["detail"], \
            f"第二根里的摘要没有被认成索引层：{with_summary}"
        assert not any(c["title"] == "分层" and c["level"] == WARN
                       for c in with_summary), "真实摘要进第二根后应消除缺层警告"

        assert retrieval_dirs(corpus_no_index, corpus_no_index) == \
            (corpus_no_index.resolve(),), "同一个目录传两次不许重复建块"
        missing_index = (_P(td) / "missing-index").resolve()
        bad_index = diagnose(corpus_no_index, index_dir=missing_index)
        assert bad_index[-1]["level"] == FAIL and \
            bad_index[-1]["title"] == "索引摘要目录" and \
            str(missing_index) in bad_index[-1]["detail"], \
            f"显式 --index-dir 不存在时必须指名绝对路径：{bad_index}"
        index_file = _P(td) / "not-a-directory.txt"
        index_file.write_text("不是目录。\n", encoding="utf-8")
        bad_index_file = diagnose(corpus_no_index, index_dir=index_file)
        assert bad_index_file[-1]["level"] == FAIL and \
            "不是目录" in bad_index_file[-1]["detail"], \
            f"显式 --index-dir 指向文件时必须说清：{bad_index_file}"

        captured_loader = {}
        original_load_corpus = globals()["load_corpus"]

        def capture_load_corpus(paths, **kwargs):
            captured_loader["paths"] = tuple(paths)
            captured_loader.update(kwargs)
            return "合成索引"

        globals()["load_corpus"] = capture_load_corpus
        try:
            source_dirs, loader14b = make_corpus_loader(
                corpus_no_index, index_dir=index_dir, embed=True,
                provider="合成 embedding 提供方")
            assert loader14b() == "合成索引"
        finally:
            globals()["load_corpus"] = original_load_corpus
        assert source_dirs == (corpus_no_index.resolve(), index_dir.resolve())
        assert captured_loader["cache_path"] == \
            corpus_no_index / ".embed_cache.json", \
            f"双根加载不能让 embedding 缓存漂到索引目录：{captured_loader}"
        assert captured_loader["provider"] == "合成 embedding 提供方"

        #    指错目录的三种典型都要给出**能照着改**的话，不是"失败"两个字
        gone = diagnose(_P(td) / "根本没有这个目录")
        assert gone[0]["level"] == FAIL and "--corpus" in gone[0]["detail"]
        #    指到了旁边一个没有语料的目录（典型是 src/）：报失败之外还要**按内容**
        #    把真正的候选找出来。这里的靶子目录叫 corpus/ 不叫 memory/，所以
        #    谁把候选判据写成"目录名叫 memory/"，这条立刻红
        (_P(td) / "src").mkdir()
        astray = diagnose(_P(td) / "src")
        assert astray[-1]["level"] == FAIL and "corpus" in astray[-1]["detail"], \
            f"该按内容指出“你要指的多半是 corpus/”：{astray[-1]}"
        #    指到了产出目录本身：人格文件被当成记忆吃进库，不报任何错——这条只能靠体检
        (corpus / "CLAUDE.md").write_text("# 人格文件\n你是……\n", encoding="utf-8")
        stray = {c["title"]: c for c in diagnose(corpus)}
        assert stray["混进来的文件"]["level"] == WARN and \
            "CLAUDE.md" in stray["混进来的文件"]["detail"], "人格文件混进语料要报出来"

    #     mtime 兜底的报警：文件名与正文都不带日期时，整批块拿到同一个假时间
    with tempfile.TemporaryDirectory() as td:
        c2 = _P(td) / "corpus"
        c2.mkdir()
        (c2 / "随手记.md").write_text("## 没写日期\n聊了点别的。\n", encoding="utf-8")
        m = {c["title"]: c for c in diagnose(c2)}
        assert m["时间戳来源"]["level"] == WARN and "mtime" in m["时间戳来源"]["detail"]
        assert m["会话线索"]["level"] == WARN, "没配 --threads 要提醒收尾不过夜"
        #     sidecar 在、但一块都对不上 = 静默失效（换过目录/改过语料），要报出来
        (c2 / ".weights.json").write_text('{"deadbeef": 2.0}', encoding="utf-8")
        w = {c["title"]: c for c in diagnose(c2)}
        assert w[".weights.json"]["level"] == WARN and "对得上" in w[".weights.json"]["detail"]
        #     部分孤儿比全孤儿隐蔽得多（2026.08.03 外部缺陷报告逼出来的一格）：
        #     接上的那几条让这格报 OK，孤儿那条静默失效——若是撤回账本，被撤回的
        #     旧说法照常召回。变异：把孤儿检查删掉（n>0 直接报 OK）→ 这条红
        from memory_retrieval import _chunk_key as _ck14
        good_key = _ck14(load_corpus(c2).chunks[0])
        (c2 / ".weights.json").write_text(
            json.dumps({good_key: 2.0, "deadbeef00000000": 3.0}), encoding="utf-8")
        o14 = {c["title"]: c for c in diagnose(c2)}
        assert o14[".weights.json"]["level"] == WARN and "孤儿" in o14[".weights.json"]["detail"], \
            "账本部分对得上时，孤儿条目必须点名报出来——OK 不许挡住死账"

    # 15.【部署体检·走真进程，从相对路径 cwd 起】上面第 14 项全是函数级断言，
    #     它有两个够不着的地方，而返工的三条缺陷恰好都藏在那里：
    #       ① 函数级断言喂的是 tempfile 给的**绝对**路径，于是"相对路径有没有被
    #          解析开"永远测不到——而 `.mcp.json` 里写的正是 `--corpus corpus`
    #          这种相对值，这一单的来源就是它；
    #       ② `__main__` 那段分派（缺 --corpus 的报错、按 FAIL 决定的退出码）
    #          一条断言都盖不到，全在裸奔。
    #     所以这一项起真进程、传相对值、断言 stdout 与退出码。
    #     **断的是 str(corpus.resolve()) 在不在输出里，不是字符串 "corpus" 在不在**
    #     ——后者用户本来就知道，回显给他等于一个字没答。
    #     变异：去掉 diagnose 里的 .resolve() / 把块数写死 / 空目录也报成功 → 各自红
    import subprocess
    here = _P(__file__).resolve().parent

    def run_doctor(cwd, *argv):
        p = subprocess.run([sys.executable, str(here / "mcp_server.py"), "--doctor", *argv],
                           cwd=str(cwd), capture_output=True, text=True, encoding="utf-8")
        return p.returncode, p.stdout + p.stderr

    with tempfile.TemporaryDirectory() as td:
        corpus = _P(td) / "corpus"
        (corpus / "timeline").mkdir(parents=True)
        (corpus / "index").mkdir(parents=True)
        (corpus / "timeline" / "window_04_2026-06-17.md").write_text(
            "## 修咖啡机\n加热管不工作，拆开发现保险丝熔断。\n", encoding="utf-8")
        (corpus / "index" / "window_04.md").write_text(
            "## 第4窗摘要\n修好了咖啡机。\n", encoding="utf-8")
        before = _snapshot(corpus)
        #    从父目录起、传相对值——**照 .mcp.json 里那行的形态跑**
        code, out = run_doctor(td, "--corpus", "corpus")
        assert code == 0, f"这份语料该全过，退出码 {code}：{out}"
        assert str(corpus.resolve()) in out, \
            f"报告里必须出现真实绝对路径（相对路径要解析开），实际输出：{out}"
        assert "建库：2 块" in out and "索引层 1" in out, f"块数与分层计数要在输出里：{out}"
        assert "✓ 工具 schema" in out and "无 anyOf/oneOf" in out, \
            f"doctor 必须逐工具检查 schema 顶层形状：{out}"
        #    断在“建库：”那一行上，别只断“2 块”——时间戳来源那行也带块数，
        #    松着断的话把块数写死成常量的变异会从那儿溜过去（实测溜过一次）
        #    块数要真的数出来：**同一次 selftest 里再跑一份块数不同的语料**，
        #    不然把它写死成常量的变异测不出来（第一份恰好就是那个常量）
        bigger = _P(td) / "另一份"
        bigger.mkdir()
        (bigger / "window_05_2026-06-20.md").write_text(
            "## 换纱窗\n阳台的纱窗破了个洞，量好尺寸重新装了一扇。\n\n"
            "## 修水龙头\n厨房水龙头滴水，换掉里面的胶垫就好了。\n\n"
            "## 装晾衣杆\n阳台加了一根晾衣杆，位置挑在采光最好的一侧。\n",
            encoding="utf-8")
        code2, out2 = run_doctor(td, "--corpus", "另一份")
        assert code2 == 0 and "建库：3 块" in out2, f"块数要真数出来，不是写死的：{out2}"
        #    最新标题只有日期时，必须继续找较旧但有内容词的标题，不能拿日期冒充探针。
        probe_index = MemoryIndex()
        probe_index.add("## 修咖啡机\n换掉保险丝。",
                        {"heading": "修咖啡机", "timestamp": 1})
        probe_index.add("## 2026-08-29 记\n那天整理了房间。",
                        {"heading": "2026-08-29 记", "timestamp": 2})
        probe_index.build()
        assert _doctor_probe(probe_index) == ("修咖啡机", False), \
            "doctor 应优先挑含内容词的标题，不按最新时间盲取纯日期标题"
        date_probe_index = MemoryIndex()
        date_probe_index.add("## 2026-08-29 记\n那天整理了房间。",
                             {"heading": "2026-08-29 记", "timestamp": 2})
        date_probe_index.build()
        assert _doctor_probe(date_probe_index) == ("2026-08-29 记", True), \
            "只有纯日期标题时必须标成弱探针，供失败结果降级 WARN"
        failed_probe = {"isError": True, "content": [{"text": "没有可靠命中\n请换个问法"}]}
        weak_level, weak_detail = _doctor_probe_outcome(
            "2026-08-29 记", True, failed_probe)
        assert weak_level == WARN and "标题只有日期／窗口等通用词" in weak_detail \
            and "正文中的一句原话" in weak_detail, \
            f"纯日期弱探针失败必须降级 WARN 并给人工复查出口：{weak_detail}"
        strong_level, strong_detail = _doctor_probe_outcome(
            "修咖啡机", False, failed_probe)
        assert strong_level == FAIL and "反而查不到" in strong_detail, \
            "含内容词的强探针失败仍必须判 FAIL"
        assert "时间范围" in out and "2026-06-17" in out, \
            f"时间范围（最早/最晚）要在输出里——旧快照那个坑靠它：{out}"
        assert "filename" in out, f"时间戳来源分布要在输出里：{out}"
        assert _snapshot(corpus) == before, "真进程跑一遍同样不许写盘"

        #    真 CLI 接第二根：两边都只读，且参数必须真的进 doctor，不是只在函数层可用。
        independent_index = _P(td) / "bundle" / "memory" / "index"
        independent_index.mkdir(parents=True)
        (independent_index / "window_04_2026-06-17.md").write_text(
            "## 第4窗摘要\n咖啡机保险丝已经换好。\n", encoding="utf-8")
        before_main = _snapshot(corpus)
        before_independent = _snapshot(independent_index)
        code_index, out_index = run_doctor(
            td, "--corpus", "corpus", "--index-dir", str(independent_index))
        assert code_index == 0 and "索引摘要目录" in out_index and \
            "索引层 2" in out_index, \
            f"真 CLI 没把独立索引目录接进 doctor：{code_index} / {out_index}"
        assert _snapshot(corpus) == before_main and \
            _snapshot(independent_index) == before_independent, \
            "真 CLI doctor 必须同时只读主语料与独立索引目录"

        #    空目录：显著提示 + 退出码非零（自动化靠它，人靠那句话）
        empty = _P(td) / "空目录"
        empty.mkdir()
        code, out = run_doctor(td, "--corpus", "空目录")
        assert code != 0, f"空目录必须非零退出，实际 {code}：{out}"
        assert "✗" in out and "没找到任何 .md 语料" in out, f"提示要显著：{out}"
        assert str(empty.resolve()) in out, "失败路径同样要回答“读的是哪儿”"

        #    有 md、但切不出块（全空行）：不许崩成 traceback——正在排查故障的人
        #    要的是一句看得懂的话。旧版在这里 ZeroDivisionError
        blank = _P(td) / "空白语料"
        blank.mkdir()
        (blank / "a.md").write_text("\n   \n\n", encoding="utf-8")
        code, out = run_doctor(td, "--corpus", "空白语料")
        assert code != 0, f"切不出块的语料要非零退出，实际 {code}：{out}"
        assert "Traceback" not in out, f"不许崩，要出话：{out}"
        assert "一块内容都切不出来" in out and "✗" in out, f"要显著地说“一块都没有”：{out}"

        #    切块成色：正常语料不许报警，"每句一个 `##`"的导出必须报警。
        #    两个方向都断——只断报警那一半的话，"永远报警"的变异测不出来
        assert "中位块长" in out2 and "⚠ 切块" not in out2, \
            f"正常语料要报中位块长、且不该报切块警告：{out2}"
        每句一块 = _P(td) / "每句一块"
        每句一块.mkdir()
        (每句一块 / "chat_2026-06-21.md").write_text(
            "\n\n".join(["## 好的，我看一下", "## 这个改完了吗", "## 嗯",
                         "## 明天上午十点开会", "## 收到"]), encoding="utf-8")
        code, out3 = run_doctor(td, "--corpus", "每句一块")
        assert "⚠ 切块" in out3 and "只有一行标题" in out3, \
            f"每句话一个 `##` 的导出要被体检拦下来吭一声：{out3}"
        assert "`##`" in out3, f"要告诉人去看哪儿（语料里的 `##`）：{out3}"
        #    变异：去掉 _chunk_body 的首行判断（整块都当正文）/ 把阈值放到 1.0
        #    / 只报块数不报块长 → 各自红

        #    【mtime 骑到真日期头上·变异靶心】（2026.08.03 补）比例判据看的是
        #    "有多少块没日期"，这条看的是"没日期的块会不会骑到有日期的头上"。
        #    **危险的恰恰是占比低的情况**：绝大多数块有日期、少数几块落 mtime，
        #    那几块的时间戳就是"刚刚"，稳稳压过所有真日期，而占比连 20% 的门都
        #    够不着 → 静默。这里造的正是那种语料：4 块带日期、1 块落 mtime（20%
        #    不过阈值），必须靠新判据报出来。
        混合 = _P(td) / "混合"
        混合.mkdir()
        for d in ("2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04"):
            (混合 / f"window_01_{d}.md").write_text(
                f"# 窗口 · {d}\n\n那天做了些事，记一笔留着以后看。", encoding="utf-8")
        (混合 / "没有日期的备忘.md").write_text(
            "这条没有任何日期依据，落盘时间就是刚刚。", encoding="utf-8")
        code, out_mix = run_doctor(td, "--corpus", "混合")
        assert "⚠ 时间戳来源" in out_mix, \
            f"少数几块落 mtime 却比所有真日期都新时必须报警（比例判据够不着这一格）：{out_mix}"
        assert "最旧的事实当成你现在的状态" in out_mix or "方向错误" in out_mix, \
            f"WARN 正文要说清后果是方向错误，不是“没有排序信号”：{out_mix}"
        #    变异：把 outranks 那一支删掉（只留比例判据）→ 这条红

        #    缺 --corpus：argparse 拦下，同样是真进程才盖得到的一格
        code, out = run_doctor(td)
        assert code != 0 and "--corpus" in out, f"缺 --corpus 要被拦下：{code} {out}"

    # ⚠ 下面 18 系列会**故意**撞出四种拒绝，而被拒请求 2026.08.04 起会在服务端留一行
    #    （18g 那张卡），于是这些行会跑到跑自检的人眼前。**先说一句它们不是出错**——
    #    否则用户看到一屏「被拒 …」根本分不清自检过没过（真机第一次跑就被这样问了）。
    #    ⚠ **刻意不做成「自检期间把它们吞掉」**：这张卡治的就是「压掉但不说」，
    #    在它自己的自检里加一个静默开关是反过来走；而且真有一天拒绝在不该触发时触发了，
    #    看得见比看不见强。所以是加一句说明，不是加一道闸。
    print("⚠ 下面几行「被拒 …」是自检故意撞出来的（18 系列要挨个验四种拒绝该不该挡住），"
          "不是出错——以最后一行 selftest ok 为准。", file=sys.stderr)

    # 18.【HTTP 传输·真端口往返】（2026.08.03「连不上 MCP 只好上 supergateway」
    #     那单）：supergateway 的四个实测坑逐个变成这里的一格——无鉴权（→401 两格
    #     ＋非回环裸跑拒绝）、传输不匹配（→握手/工具表/UTF-8 真 HTTP 往返）、
    #     常驻不重读语料（→手动放 md 下一个请求就查到）；外加 Origin 防线、
    #     通知 202、GET 405、批量 400 三条规格面。
    import urllib.request
    import urllib.error
    try:
        http_bind_guard("0.0.0.0", None)
        assert False, "非回环 + 无 token 必须拒绝起动——跑通的那一刻记忆库就暴露了"
    except ValueError as e18:
        assert "token" in str(e18), "拒绝的报错要指向修法（--token / 绑回环）"
    http_bind_guard("127.0.0.1", None)       # 回环裸跑放行：反代在前的形态
    http_bind_guard("0.0.0.0", "s3cret")     # 配了 token 的公开绑定放行
    with tempfile.TemporaryDirectory() as td:
        (_P(td) / "timeline").mkdir()
        (_P(td) / "timeline" / "window_01_2026-08-01.md").write_text(
            "# 第1个窗口 · 2026-08-01\n\n## 2026-08-01 记\n试养了一盆绿萝。\n"
            "当下状态：长势正常。\n", encoding="utf-8")
        index18 = _P(td) / "index"
        index18.mkdir()
        loader18 = lambda: load_corpus((td, index18))
        srv18 = MemoryServer(index=loader18(), thread_store=ThreadStore(),
                             corpus_dir=td,
                             retractions_path=_P(td) / ".retractions.json",
                             loader=loader18, index_dir=index18,
                             source_dirs=(td, index18))
        httpd = make_http_server(srv18, host="127.0.0.1", port=0, token="s3cret")
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        base18 = f"http://127.0.0.1:{httpd.server_address[1]}/mcp"

        def post18(payload, tok="s3cret", origin=None, method="POST", accept=None):
            req = urllib.request.Request(
                base18, data=(json.dumps(payload, ensure_ascii=False).encode("utf-8")
                              if payload is not None else None), method=method)
            if tok:
                req.add_header("Authorization", f"Bearer {tok}")
            if origin:
                req.add_header("Origin", origin)
            if accept:
                req.add_header("Accept", accept)
            req.add_header("Content-Type", "application/json")
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    return r.status, r.read().decode("utf-8")
            except urllib.error.HTTPError as e:
                return e.code, e.read().decode("utf-8")

        lst18 = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        #     鉴权两格（变异：把 _gate 的 token 检查删掉 → 这两条红）
        assert post18(lst18, tok=None)[0] == 401, "无 token 必须 401"
        assert post18(lst18, tok="wrong")[0] == 401, "错 token 必须 401"
        #     Origin 防线（DNS rebinding：浏览器页面拿本地端口当跳板）
        assert post18(lst18, origin="https://evil.example")[0] == 403
        #     握手 → 工具表 → 通知 202
        code18, body18 = post18({"jsonrpc": "2.0", "id": 2, "method": "initialize",
                                 "params": {}})
        assert code18 == 200 and "protocolVersion" in body18
        code18, body18 = post18(lst18)
        assert code18 == 200 and "latent_search" in body18
        code18, body18 = post18({"jsonrpc": "2.0",
                                 "method": "notifications/initialized"})
        assert code18 == 202 and not body18, "通知回 202 空身，不回 JSON-RPC 响应"
        #     中文 UTF-8 真 HTTP 往返：写回 → 当场检索命中（同 stdio 第 9 项的病灶）
        code18, body18 = post18({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                 "params": {"name": "latent_append",
                                            "arguments": _append_args("绿萝换了大一号的盆。",
                                                                      "适应中。",
                                                                      moment=time.time())}})
        assert code18 == 200 and "已写进" in body18
        code18, body18 = post18({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                                 "params": {"name": "latent_search",
                                            "arguments": {"query": "绿萝换盆"}}})
        assert code18 == 200 and "大一号的盆" in body18, \
            f"中文该原样穿过 HTTP 并命中：{body18}"
        #     本轮新增错误边界也要走真 HTTP，不拿进程内 handle 代替传输结果：
        #     未预料异常仍回 200 JSON-RPC，content 非空且 isError=true。
        handlers18 = srv18._handlers

        def http_boom18(args, now=None):
            raise RuntimeError("合成敏感正文不应外泄")

        srv18._handlers = lambda: {"latent_session_start": http_boom18}
        try:
            code18, body18 = post18(
                {"jsonrpc": "2.0", "id": 41, "method": "tools/call",
                 "params": {"name": "latent_session_start"}})
        finally:
            srv18._handlers = handlers18
        error18 = json.loads(body18)["result"]
        assert code18 == 200 and error18["isError"] is True, \
            f"工具异常要穿过真 HTTP 变成 isError：{body18}"
        assert error18["content"][0]["text"] and "RuntimeError" in error18["content"][0]["text"]
        assert "合成敏感正文" not in body18, "异常正文不许经 HTTP 泄露"
        #     常驻自动重读（变异：把 do_POST 里的指纹检查删掉 → 这条红）：
        #     手动上传 md 是 supergateway 时代实测过的静默坑——常驻只在启动读一次，
        #     新语料一条都查不到且不报错
        (_P(td) / "timeline" / "window_09_2026-08-02.md").write_text(
            "# 第9个窗口 · 2026-08-02\n\n## 2026-08-02 记\n手动上传：入手了一台"
            "折叠望远镜。\n当下状态：还没开箱。\n", encoding="utf-8")
        code18, body18 = post18({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                                 "params": {"name": "latent_search",
                                            "arguments": {"query": "望远镜"}}})
        assert code18 == 200 and "折叠望远镜" in body18, \
            "语料目录手动加的 md 必须下一个请求就查得到——不重读是静默的"
        #     规格面三格：GET／DELETE 空操作／批量已移除
        #     ⚠ GET 那格 2026.08.07 从「恒 405」改成「按 Accept 分流」，判据跟着换：
        #     **明确只要 JSON** 才 405（判据 7）；要 SSE、或压根没带 Accept 的
        #     给一条空长流——那半在 18h，**不能在这儿用 urllib 验**，
        #     `r.read()` 会一直读到流关闭，等于把自检挂死
        assert post18(None, method="GET", accept="application/json")[0] == 405
        assert post18(None, method="DELETE")[0] == 200
        code18, body18 = post18([lst18])
        assert code18 == 400 and "批量" in body18
        #     路径：文档教人填 /mcp，别的端点要明确 404（根路径放行是手滑余地）
        assert post18(lst18)[0] == 200
        req_bad = urllib.request.Request(
            f"http://127.0.0.1:{httpd.server_address[1]}/nope",
            data=json.dumps(lst18).encode(), method="POST")
        req_bad.add_header("Authorization", "Bearer s3cret")
        try:
            urllib.request.urlopen(req_bad, timeout=10)
            assert False, "未知端点要 404"
        except urllib.error.HTTPError as e:
            assert e.code == 404

        # 18b.【keep-alive 下被拒的请求必须吞掉 body·变异靶心】（2026.08.03 外部实测）
        #     ⚠ **这一格只能走裸 socket**：上面 urllib 每次新连接、发 Connection: close，
        #     所以它天然盖不住——而手机 App 复用连接是常态，「先探一次拿 401 →
        #     带 token 重试」这个最普通的姿势就撞这个坑。不吞 body 的话残留会被
        #     当成下一个请求的**请求行**，第二发收到 400 + HTML 错误页。
        #     变异：把 _deny 里的 _drain() 去掉 → 这条红
        import socket as _socket
        raw = json.dumps(lst18).encode()

        def _recv_http(sock):
            """收完整一条响应——**必须按 Content-Length 收满**：单发 recv 常常只
            拿到头部，body 还在路上，据此断言等于随机红绿。"""
            buf = b""
            while b"\r\n\r\n" not in buf:
                d = sock.recv(65536)
                if not d:
                    return buf.decode("utf-8", "replace")
                buf += d
            head, _, body = buf.partition(b"\r\n\r\n")
            want = 0
            for ln in head.decode("latin-1").split("\r\n")[1:]:
                if ln.lower().startswith("content-length:"):
                    want = int(ln.split(":", 1)[1])
            while len(body) < want:
                d = sock.recv(65536)
                if not d:
                    break
                body += d
            return (head + b"\r\n\r\n" + body).decode("utf-8", "replace")

        def _raw_post(sock, tok=None, extra_hdr=""):
            h = (f"POST /mcp HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                 f"Content-Length: {len(raw)}\r\n")
            if tok:
                h += f"Authorization: Bearer {tok}\r\n"
            sock.sendall((h + extra_hdr + "\r\n").encode() + raw)
            return _recv_http(sock)

        sk = _socket.create_connection(("127.0.0.1", httpd.server_address[1]))
        sk.settimeout(10)
        first = _raw_post(sk)                       # 无 token → 401
        assert "401" in first.split("\r\n")[0], first.split("\r\n")[0]
        second = _raw_post(sk, tok="s3cret")        # 同一条连接重试 → 必须是 200
        assert "200" in second.split("\r\n")[0], \
            f"401 之后同连接重试必须正常——被拒的请求没吞掉 body：{second.splitlines()[0]}"
        assert "latent_search" in second, "重试那发要真的拿到工具表"
        sk.close()
        #     413 分支同形（超限也要吞）：谎报一个超大 Content-Length 但不发那么多，
        #     server 该在读之前就拒掉，且连接不许被污染
        sk = _socket.create_connection(("127.0.0.1", httpd.server_address[1]))
        sk.settimeout(10)
        sk.sendall((f"POST /mcp HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer s3cret\r\n"
                    f"Content-Length: {_HTTP_BODY_LIMIT + 1}\r\n\r\n").encode())
        #     ⚠ 收响应一律走 `_recv_http`，不许 `sk.recv()` 一发了事——理由见 18c
        assert "413" in _recv_http(sk).split("\r\n")[0]
        sk.close()

        # 18c.【chunked 明确不支持，且报错指对方向·变异靶心】（2026.08.03 外部实测）
        #     原先走到"请求体不是合法的 UTF-8 JSON"，把人往「我 JSON 写错了」引，
        #     真因是传输编码不匹配——跟 supergateway 时代「默认 SSE 与客户端不匹配」
        #     同形物。变异：把 Transfer-Encoding 那一支删掉 → 这条红
        #     ⚠ **必须走 `_recv_http` 按 Content-Length 收满**（2026.08.04「自检偶发红」
        #     那单的根因）：`_send` 先冲响应头、再写 body 是两段，单发 `recv` 常常只
        #     拿到头——于是查状态行的断言恒绿、查 body 的那条随机红，看起来就是
        #     「偶发红、复跑即绿、与本轮改动无关」。同日撞了四次都记成噪音。
        #     ⚠ 这条规矩 18b 的 `_recv_http` docstring 里早写着，这里当时没照着用。
        sk = _socket.create_connection(("127.0.0.1", httpd.server_address[1]))
        sk.settimeout(10)
        sk.sendall((f"POST /mcp HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer s3cret\r\n"
                    f"Content-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n"
                    ).encode() + f"{len(raw):x}\r\n".encode() + raw + b"\r\n0\r\n\r\n")
        ck = _recv_http(sk)
        assert "501" in ck.split("\r\n")[0], f"chunked 要回 501 不是 400：{ck.splitlines()[0]}"
        assert "Transfer-Encoding" in ck and "JSON 写错" in ck, \
            "报错必须指向传输编码，并明说不是 JSON 的问题——指错方向就是这条的病灶"
        sk.close()
        httpd.shutdown()

    # 18d.【非 ASCII token 拒绝起动·变异靶心】（2026.08.03 外部实测）：中文口令
    #     server 起得来、横幅照打，但 HTTP 头是 latin-1，客户端发不出去；
    #     compare_digest 对非 ASCII str 直接抛 TypeError → 500。失败形态是
    #     「看起来起来了、就是连不上」，所以在起动这一步拦。
    #     变异：把 token.encode("ascii") 那一支删掉 → 这条红
    try:
        http_bind_guard("127.0.0.1", "口令abc")
        assert False, "非 ASCII token 必须拒绝起动——不然失败形态是静默的"
    except ValueError as e18b:
        assert "ASCII" in str(e18b), "报错要说清是 ASCII 的问题并给修法"
    http_bind_guard("127.0.0.1", "s3cret-2026")     # ASCII 照常放行

    # 18f.【非回环绑定多打一行 TLS 提醒·真进程，两个绑法都跑】（2026.08.04 外部实测
    #     促成）：`http_bind_guard` 挡的是"非回环裸跑不配 token"，**不挡"没有 TLS"**，
    #     于是配了 token 的用户走在裸 HTTP 暴露公网的路上、全程没人提醒过他。
    #     ⚠ **必须起真进程**：提醒是在 `__main__` 里打的，只断纯函数就是"函数绿、
    #     生产路径死"那张脸——把 `__main__` 里那两行删掉，纯函数断言一条都不会红。
    #     ⚠ **两个绑法都要跑**：只跑非回环那次证明不了"回环时不吵"。
    #     ⚠ **非回环那次带着 token 跑**：提醒不许看 token（token 与 TLS 是两件事），
    #     做成"有 token 就不提示"这一格要红。
    #     ⚠ 两次都断"横幅在"：否则进程压根没起来也能让"不含 TLS"那条恒真（假绿）。
    #     变异：删掉 `__main__` 里打 notice 那两行 / 让 http_tls_notice 恒返回 None /
    #           改成 token 已配就不打 → 各自红
    def run_banner(corpus_dir, bind, *extra):
        """起真进程跑 `--http`，收横幅那几行，然后掐掉。返回 stderr 文本。"""
        proc = subprocess.Popen(
            [sys.executable, str(here / "mcp_server.py"), "--corpus", str(corpus_dir),
             "--http", bind, *extra],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
        got = []
        pump = threading.Thread(target=lambda: got.extend(proc.stderr), daemon=True)
        pump.start()
        deadline = time.time() + 20
        while time.time() < deadline and not any("语料变化自动重读" in x for x in got):
            time.sleep(0.05)
        time.sleep(0.5)          # 提醒行紧跟在横幅后面，给它落地的时间
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        pump.join(timeout=5)
        return "".join(got)

    with tempfile.TemporaryDirectory() as td:
        (_P(td) / "timeline").mkdir()
        (_P(td) / "timeline" / "window_01_2026-08-01.md").write_text(
            "# 第1个窗口 · 2026-08-01\n\n## 2026-08-01 记\n试养了一盆绿萝。\n"
            "当下状态：长势正常。\n", encoding="utf-8")
        out_pub = run_banner(td, "0.0.0.0:0", "--token", "s3cret-2026")
        assert "Streamable HTTP 服务在" in out_pub, \
            f"非回环 + token 必须照常起动（只提示、不拒绝）：{out_pub}"
        assert "TLS" in out_pub, \
            f"绑非回环要多打一行 TLS 提醒——配了 token 也照打：{out_pub}"
        assert "127.0.0.1" in out_pub and "反向代理" in out_pub, \
            f"提醒里要给出修法（绑回环 + 反代管 TLS），不能只说一句「有风险」：{out_pub}"
        out_lo = run_banner(td, "127.0.0.1:0")
        assert "Streamable HTTP 服务在" in out_lo, \
            f"回环裸跑照常起动（反代在前的形态）：{out_lo}"
        assert "TLS" not in out_lo, f"回环上不许吵这一句：{out_lo}"

    # 18i.【独立索引目录也要触发 HTTP 自动重读】主语料没有变化，只往第二根放入
    #      摘要；下一次请求必须重建并命中。只测 loader 能读两根不够：旧指纹仍只盯
    #      corpus 时，服务会安静地永远看不见这次更新。
    with tempfile.TemporaryDirectory() as td18i:
        root18i = _P(td18i)
        corpus18i = root18i / "corpus"
        index18i = root18i / "bundle" / "memory" / "index"
        (corpus18i / "timeline").mkdir(parents=True)
        index18i.mkdir(parents=True)
        (corpus18i / "timeline" / "window_01_2026-08-01.md").write_text(
            "## 咖啡机旧记录\n一开始怀疑加热管。\n", encoding="utf-8")
        source18i, loader18i = make_corpus_loader(corpus18i, index_dir=index18i)
        srv18i = MemoryServer(index=loader18i(), thread_store=ThreadStore(),
                              corpus_dir=corpus18i, source_dirs=source18i,
                              loader=loader18i)
        httpd18i = make_http_server(srv18i, host="127.0.0.1", port=0)
        threading.Thread(target=httpd18i.serve_forever, daemon=True).start()

        def call18i(query):
            payload = json.dumps({
                "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": "latent_search", "arguments": {"query": query}},
            }, ensure_ascii=False).encode("utf-8")
            request = urllib.request.Request(
                f"http://127.0.0.1:{httpd18i.server_address[1]}/mcp",
                data=payload, headers={"Content-Type": "application/json",
                                       "Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.read().decode("utf-8")

        before18i = call18i("咖啡机 加热管 保险丝")
        assert "不是加热管" not in before18i, "摘要还没落盘前不许凭空命中"
        (index18i / "topic_咖啡机_2026-08-09.md").write_text(
            "## 咖啡机主题线\n最后确认故障是保险丝，不是加热管。\n",
            encoding="utf-8")
        after18i = call18i("咖啡机 加热管 保险丝")
        assert "不是加热管" in after18i, \
            f"第二根变化后下一请求必须自动重读：{after18i}"
        httpd18i.shutdown()

    # 18e.【自动重读不许吞掉同窗口的手动上传·变异靶心】（2026.08.03 外部评审的竞态）：
    #     原先 handle 之后无条件重算指纹，于是一次 latent_append 与用户手动上传
    #     落在同一个请求窗口时，那次上传被自己的刷新吞掉、**再也不触发重读**——
    #     正好是这个特性要治的静默形态。改法是只把 server 自己写过的路径折进基线。
    #     变异：把 written_paths 那套换回 state["sig"] = _corpus_signature(...) → 这条红
    with tempfile.TemporaryDirectory() as td:
        (_P(td) / "timeline").mkdir()
        (_P(td) / "timeline" / "window_01_2026-08-01.md").write_text(
            "# 第1个窗口 · 2026-08-01\n\n## 2026-08-01 记\n养了绿萝。\n"
            "当下状态：正常。\n", encoding="utf-8")
        loader18e = lambda: load_corpus(td)
        srv18e = MemoryServer(index=loader18e(), thread_store=ThreadStore(),
                              corpus_dir=td, loader=loader18e)
        httpd18e = make_http_server(srv18e, host="127.0.0.1", port=0, token="s3cret")
        threading.Thread(target=httpd18e.serve_forever, daemon=True).start()
        b18e = f"http://127.0.0.1:{httpd18e.server_address[1]}/mcp"

        def call18e(payload):
            rq = urllib.request.Request(b18e, data=json.dumps(payload).encode(),
                                        method="POST")
            rq.add_header("Authorization", "Bearer s3cret")
            with urllib.request.urlopen(rq, timeout=10) as r:
                return r.read().decode("utf-8")

        # ⚠ **上传必须落在 handle 期间**，这才是竞态窗口：写在请求之前的话，
        #   do_POST 的前置指纹检查当场就逮住它，反而测不到这一格（第一版夹具
        #   就这么造错了，变异不红——夹具错了跟钉子失效长得一模一样）。
        real_handle, done18e = srv18e.handle, []

        def handle_then_upload(msg, now=None):
            r = real_handle(msg, now=now)
            if not done18e:                  # 只插一次
                done18e.append(1)
                (_P(td) / "timeline" / "window_09_2026-08-02.md").write_text(
                    "# 第9个窗口 · 2026-08-02\n\n## 2026-08-02 记\n"
                    "手动上传：买了折叠望远镜。\n当下状态：还没开箱。\n",
                    encoding="utf-8")
            return r

        srv18e.handle = handle_then_upload
        call18e({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                 "params": {"name": "latent_append",
                            "arguments": {"text": "顺手记一条别的。",
                                          "current_state": "无。"}}})
        srv18e.handle = real_handle
        got18e = call18e({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                          "params": {"name": "latent_search",
                                     "arguments": {"query": "望远镜"}}})
        assert "折叠望远镜" in got18e, \
            "同窗口里的手动上传不许被自己的写回吞掉——吞掉就再也不重读了（静默）"
        httpd18e.shutdown()

    # 18g.【被拒的请求在服务端要留一行·四个靶心】（2026.08.04 立卡）：八种拒绝的
    #     原因 `_deny` 每条都写清楚了，但那句话只进**响应体**，多数 MCP 客户端不显示；
    #     服务端 `log_message` 整个静音，于是**两头都不说话**，八种原因在用户那里
    #     长成同一个样子（「连不上」）。落点只有 `_deny` 一处（八条全走它）。
    #     ⚠ **靶心二必须跟靶心一一起跑**：只跑靶心一的话，「正确实现」与「把
    #     `log_message` 的静音整个掀掉」在证据上长得一模一样。
    #     ⚠ 取证靠把 `sys.stderr` 换成 StringIO（`print(..., file=sys.stderr)` 每次
    #     现查属性，替换全局有效）。这一格在进程内起 server，**不走起动横幅**，
    #     所以捕到的行就是请求行本身；靶心二据此断"一行都没有"。
    #     变异：① 删掉 `_deny` 里那行 `self._log_deny(...)` → 靶心一红；
    #           ② 把 `_log_deny` 里的 `urlsplit(...).path` 换回 `self.path` → 靶心三红；
    #           ③ 删掉 `_settle_locked` 里返回「另有 X 条」那一行（只保留上限压制）
    #              → 靶心四红。
    import socket as _socket18g
    with tempfile.TemporaryDirectory() as td:
        (_P(td) / "timeline").mkdir()
        (_P(td) / "timeline" / "window_01_2026-08-01.md").write_text(
            "# 第1个窗口 · 2026-08-01\n\n## 2026-08-01 记\n试养了一盆绿萝。\n"
            "当下状态：长势正常。\n", encoding="utf-8")
        loader18g = lambda: load_corpus(td)
        srv18g = MemoryServer(index=loader18g(), thread_store=ThreadStore(),
                              corpus_dir=td,
                              retractions_path=_P(td) / ".retractions.json",
                              loader=loader18g)
        httpd18g = make_http_server(srv18g, host="127.0.0.1", port=0, token="s3cret")
        threading.Thread(target=httpd18g.serve_forever, daemon=True).start()
        port18g = httpd18g.server_address[1]
        b18g = f"http://127.0.0.1:{port18g}"

        def hit18g(path="/mcp", tok="s3cret", origin=None, method="POST", body=None,
                   accept=None):
            req = urllib.request.Request(b18g + path, data=body, method=method)
            if tok:
                req.add_header("Authorization", f"Bearer {tok}")
            if origin:
                req.add_header("Origin", origin)
            if accept:
                req.add_header("Accept", accept)
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    return r.status
            except urllib.error.HTTPError as e:
                e.read()
                return e.code

        def cap18g(fn):
            """跑一段，返回它在 stderr 上留下的全部文本。"""
            buf, old = io.StringIO(), sys.stderr
            sys.stderr = buf
            try:
                fn()
            finally:
                sys.stderr = old
            return buf.getvalue()

        good18g = json.dumps({"jsonrpc": "2.0", "id": 1,
                              "method": "tools/list"}).encode("utf-8")

        #     靶心一（要说话）：401/403/404/405 各发一次 → **各出现一行**，
        #     且那行同时含状态码、方法、路径、客户端 IP。
        #     ⚠ 判的是四条各自出现，不是"stderr 非空"。
        def _four():
            assert hit18g(tok="wrong", body=good18g) == 401
            assert hit18g(origin="https://evil.example", body=good18g) == 403
            assert hit18g(path="/nope", method="GET") == 404
            # ⚠ 要 405 就得**明确说只收 JSON**：不带 Accept 的 GET 现在拿到的是
            # 一条空长流（2026.08.07 起），urllib 会挂在那儿读到天荒地老
            assert hit18g(method="GET", accept="application/json") == 405

        out18g = cap18g(_four)
        for code18g, meth18g, path18g in (("401", "POST", "/mcp"), ("403", "POST", "/mcp"),
                                          ("404", "GET", "/nope"), ("405", "GET", "/mcp")):
            hits = [ln for ln in out18g.splitlines()
                    if ln.startswith(f"被拒 {code18g} ")]
            assert len(hits) == 1, f"{code18g} 要在 stderr 上留且只留一行：{out18g!r}"
            assert meth18g in hits[0] and f" {path18g} " in hits[0] \
                and "127.0.0.1" in hits[0], \
                f"那一行要同时含方法、路径、客户端 IP：{hits[0]!r}"

        #     靶心二（不许多说）：一次正常往返 → stderr 上**不出现任何请求行**。
        #     断的是"一个字符都没有"：`log_message` 的静音一旦被掀掉，
        #     这里会冒出 `"POST /mcp HTTP/1.1" 200 -` 那种访问日志。
        def _ok():
            assert hit18g(body=json.dumps(
                {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                 "params": {"name": "latent_search",
                            "arguments": {"query": "绿萝"}}}).encode("utf-8")) == 200, \
                "正常的 tools/call 该 200——它不 200 的话下面那条恒真（假绿）"

        ok18g = cap18g(_ok)
        assert ok18g == "", \
            f"正常请求不许留痕——`log_message` 的静音不许被掀掉：{ok18g!r}"

        #     靶心三（不许说漏嘴）：三条硬边界各钉一枚哨兵。
        def _leak():
            assert hit18g(tok="wrong-SENTINELA", body=good18g) == 401
            assert hit18g(path="/nope?token=SENTINELB", method="GET") == 404
            assert hit18g(body=b"{not json SENTINELC") == 400

        leak18g = cap18g(_leak)
        assert "SENTINELA" not in leak18g, f"Authorization 一个字符都不许进日志：{leak18g!r}"
        assert "SENTINELB" not in leak18g, \
            f"路径要剥掉 query（有客户端把 token 塞 ?token=）：{leak18g!r}"
        assert "SENTINELC" not in leak18g, \
            f"请求体不许进日志（里面是 latent_search 的问句／latent_append 的正文）：{leak18g!r}"
        n404 = [ln for ln in leak18g.splitlines() if ln.startswith("被拒 404 ")]
        assert len(n404) == 1 and " /nope " in n404[0], \
            f"剥完 query 那行的路径要正好是 /nope：{n404!r}"

        #     靶心四（上限不静默）：一个窗口内连发 200 条被拒 → 逐条行 ≤ 60，
        #     **且出现「另有 140 条」那一行**（数目对得上）。压掉不说等于用一个
        #     新的静默去修一个旧的静默。
        #     ⚠ 窗口临时缩短到 5 秒，否则要等一分钟；同时**断这 200 发确实落在
        #     一个窗口里**——夹具太慢导致窗口翻页，跟"上限失效"长得一样。
        old_win = _HTTP_DENY_LOG_WINDOW
        globals()["_HTTP_DENY_LOG_WINDOW"] = 5.0
        try:
            buf18g, old_err = io.StringIO(), sys.stderr
            sys.stderr = buf18g
            try:
                t018g = time.monotonic()

                def burst_one():
                    """一条 keep-alive 连接上连发 20 个 404。
                    ⚠ **10 条连接并发发**：单连接串行跑 200 发要 8 秒多
                    （每发都吃一次 delayed-ACK，回环上实测 ~43ms），会跨出窗口。"""
                    sk = _socket18g.create_connection(("127.0.0.1", port18g))
                    sk.settimeout(10)
                    for _ in range(20):
                        sk.sendall(b"GET /nope HTTP/1.1\r\nHost: x\r\n\r\n")
                        _recv_http(sk)    # 按 Content-Length 收满，理由见 18b
                    sk.close()

                ths18g = [threading.Thread(target=burst_one) for _ in range(10)]
                for t in ths18g:
                    t.start()
                for t in ths18g:
                    t.join(timeout=20)
                burst18g = time.monotonic() - t018g
                # 溢出那行由定时器在窗口末尾补打，等它
                dead18g = time.monotonic() + 12
                while time.monotonic() < dead18g and "另有" not in buf18g.getvalue():
                    time.sleep(0.05)
            finally:
                sys.stderr = old_err
            burst_out = buf18g.getvalue()
        finally:
            globals()["_HTTP_DENY_LOG_WINDOW"] = old_win
        assert burst18g < 4.0, \
            f"夹具太慢：这 200 发没落在同一个 5 秒窗口里（用了 {burst18g:.1f}s），" \
            f"下面的 60／140 不作数——先修夹具，别当成上限失效"
        each18g = [ln for ln in burst_out.splitlines() if ln.startswith("被拒 ")]
        assert len(each18g) == _HTTP_DENY_LOG_MAX, \
            f"逐条行要正好压到上限 {_HTTP_DENY_LOG_MAX} 行，实得 {len(each18g)}"
        assert f"过去一分钟另有 {200 - _HTTP_DENY_LOG_MAX} 条被拒未逐条列出" in burst_out, \
            f"压掉的条数必须说出来，且数目对得上：{burst_out!r}"
        httpd18g.shutdown()

    # 18h.【客户端拿 405 不回退就卡死 → 按 Accept 给它要的形态】（2026.08.07 立卡）。
    #      两条独立的卡点，此前**在用户那头长成同一个样子**（「连不上」）：
    #      ① GET 建长流、收到 405 却不改用 POST 的客户端会一直等；② 只认 SSE 响应体的客户端
    #      收到 application/json 接不上——而 ② 是 2xx，`_deny` 够不着，
    #      **服务端日志上一个字都没有**，比 ① 还阴。
    #      ⚠ **这一格全程不许用 urllib**：长流没有 Content-Length，`r.read()`
    #      会一直读到流关闭，自检当场挂死。走 http.client 并按字节数读。
    #      ⚠ **真机那半量不到**：手上没有那个卡死的客户端，也没有 Kelivo／Operit
    #      环境。这十条绿只证明**服务端形态已具备**，不证明「哪个客户端接上了」。
    #      变异：① 把 do_POST 的 Accept 分支删掉 → 判据 1 红；
    #            ② 把那个分支改成「只要有 text/event-stream 就回 SSE」→ 判据 2 红
    #               （这是最容易写出来的错版本，它会把已经跑通的客户端带坏）；
    #            ③ 把 do_GET 里「没带 Accept 也给流」那一支删掉 → 判据 5 红；
    #            ④ 把 _HTTP_SSE_MAX_STREAMS 那道闸删掉 → 判据 8 红。
    import http.client as _httpc18h
    with tempfile.TemporaryDirectory() as td18h:
        c18h = _P(td18h)
        (c18h / "a.md").write_text("## 一\n试养了一盆绿萝。\n", encoding="utf-8")
        loader18h = lambda: load_corpus(c18h)

        def _mk18h(sse_stream=True):
            srv = MemoryServer(index=loader18h(), thread_store=ThreadStore(),
                               corpus_dir=c18h, loader=loader18h)
            httpd = make_http_server(srv, host="127.0.0.1", port=0,
                                     sse_stream=sse_stream)
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            return httpd, httpd.server_address[1]

        def _conn18h(port, method, accept, body=None, path="/mcp"):
            """发一发、把响应对象原样交出去（**不读 body**，长流要按字节读）。"""
            conn = _httpc18h.HTTPConnection("127.0.0.1", port, timeout=10)
            headers = {"Content-Type": "application/json"}
            if accept is not None:
                headers["Accept"] = accept
            conn.request(method, path, body=body, headers=headers)
            return conn, conn.getresponse()

        httpd18h, port18h = _mk18h()
        lst18h = json.dumps({"jsonrpc": "2.0", "id": 1,
                             "method": "tools/list"}).encode("utf-8")
        try:
            # 判据 1：只要 SSE 的 POST → 单事件 SSE，且**能解回原来那条响应**
            #         （只断 Content-Type 不够：包错格式一样是接不上）
            cn, r = _conn18h(port18h, "POST", "text/event-stream", body=lst18h)
            assert r.status == 200 and \
                r.getheader("Content-Type", "").startswith("text/event-stream"), \
                f"只要 SSE 的 POST 得回 SSE：{r.status} {r.getheader('Content-Type')}"
            raw18h = r.read().decode("utf-8")
            cn.close()
            assert raw18h.startswith("event: message\ndata: ") and raw18h.endswith("\n\n"), \
                f"SSE 帧形状不对：{raw18h!r}"
            assert "latent_search" in json.dumps(
                json.loads(raw18h.split("data: ", 1)[1].strip()), ensure_ascii=False), \
                f"data: 那行要是原样那条 JSON-RPC 响应：{raw18h!r}"

            # 判据 2（回归护栏，**这一条比 1 重要**）：规格要求客户端两个都写，
            #         那种客户端**现在是跑通的**，不许被上面那条带坏
            for acc18h in ("application/json, text/event-stream",
                           "text/event-stream, application/json;q=0.9",
                           "*/*"):
                cn, r = _conn18h(port18h, "POST", acc18h, body=lst18h)
                got18h = r.getheader("Content-Type", "")
                r.read()
                cn.close()
                assert got18h.startswith("application/json"), \
                    f"Accept={acc18h!r} 时必须照旧回 JSON，实得 {got18h!r}"
            # 判据 3：压根没带 Accept → 照旧回 JSON
            cn, r = _conn18h(port18h, "POST", None, body=lst18h)
            got18h = r.getheader("Content-Type", "")
            r.read()
            cn.close()
            assert got18h.startswith("application/json"), \
                f"没带 Accept 的 POST 照旧回 JSON，实得 {got18h!r}"

            # 判据 4／5：GET 空长流——要 SSE 的、以及**没带 Accept 的懒客户端**
            for acc18h in ("text/event-stream", None):
                cn, r = _conn18h(port18h, "GET", acc18h)
                assert r.status == 200 and \
                    r.getheader("Content-Type", "").startswith("text/event-stream"), \
                    f"Accept={acc18h!r} 的 GET 要给长流：{r.status}"
                assert r.read(6) == b": ok\n\n", "首字节要立刻给「流建起来了」的凭据"
                cn.close()

            # 判据 6：心跳真的发（把间隔临时调小；handler 每轮现读全局，所以生效）。
            # ⚠ **必须 finally 还原**：漏还原的话后面几条会跟着变慢，且没人会红
            old_hb18h = _HTTP_SSE_HEARTBEAT
            globals()["_HTTP_SSE_HEARTBEAT"] = 0.2
            try:
                cn, r = _conn18h(port18h, "GET", "text/event-stream")
                assert r.read(6) == b": ok\n\n"
                assert r.read(13) == b": keepalive\n\n", \
                    "没有心跳的话，反代和运营商网络会把这条流静默掐掉"
                cn.close()
            finally:
                globals()["_HTTP_SSE_HEARTBEAT"] = old_hb18h

            # 判据 8：开满之后第 N+1 条要 405，不许无限堆线程。
            # ⚠ **响应对象也得攥住，不能只留 conn**（写第一版时在这儿栽了一次，
            # 而且症状看着像产品缺陷）：`Connection: close` 的响应 `will_close`
            # 为真，上一轮那个 HTTPResponse 一被 GC 就把 socket 关了，服务端当场
            # 把名额还回去 → 第 5 条拿到 200，看起来像"闸没生效"。
            # 顺带这也**正面证明了名额是即时释放的**（select 那半在干活）。
            held18h = []
            try:
                for _ in range(_HTTP_SSE_MAX_STREAMS):
                    cn, r = _conn18h(port18h, "GET", "text/event-stream")
                    assert r.read(6) == b": ok\n\n"
                    held18h.append((cn, r))
                cn, r = _conn18h(port18h, "GET", "text/event-stream")
                assert r.status == 405, f"长流开满之后要回 405，实得 {r.status}"
                r.read()
                cn.close()
            finally:
                for cn, _r18h in held18h:
                    cn.close()
        finally:
            httpd18h.shutdown()

        # 判据 9：--no-sse-stream 起的 server 退回旧行为（恒 405）
        httpd18h2, port18h2 = _mk18h(sse_stream=False)
        try:
            cn, r = _conn18h(port18h2, "GET", "text/event-stream")
            assert r.status == 405, f"--no-sse-stream 下 GET 恒 405，实得 {r.status}"
            r.read()
            cn.close()
        finally:
            httpd18h2.shutdown()

    # 19. CLI 入口必须把 stdout 锁成 UTF-8（`--doctor` 打的是中文报告）。第 8 项
    #     锁的是 stdio 传输那条流（serve_stdio 自己包 UTF-8），管不到 `print`。
    #     ⚠ **在 Linux／默认 UTF-8 的机器上这条恒真，在那儿跑不算验过**：变异要在
    #     `PYTHONIOENCODING=gbk` 下跑——删掉 `__main__` 里的
    #     `sys.stdout.reconfigure(...)` 必须转红，加回去复绿。
    assert (getattr(sys.stdout, "encoding", "") or "").lower().replace("-", "") == "utf8", \
        f"CLI 入口没把 stdout 锁成 UTF-8（当前 {sys.stdout.encoding}）：" \
        "中文 Windows（cp936）下 --doctor 遇到 emoji 会 UnicodeEncodeError"

    # 20.【时区穿到进程级·靶心】（2026.08.04 手机 Connector 真机事故，任务卡
    #     「写回时区与跨日归窗」）。**前面 memory_retrieval／session_recall 的断言
    #     喂的都是直接构造的 TimeContext，够不着"CLI 那个值有没有真的接上"**——
    #     而事故现场坏的正是接线那一层（服务器压根没有这个概念）。所以这一项走
    #     真进程，从 `--timezone` 一路断到写出来的文件名、H2 与召回标签。
    #     四格矩阵：UTC 宿主／东八区用户 × 当场／重启。
    from time_context import tzdb_available as _tzdb
    east8_20 = "Asia/Shanghai" if _tzdb() else "UTC+08:00"    # 采集条件见 time_context
    epoch20 = datetime(2026, 8, 3, 18, 5, 0, tzinfo=_dt_tz.utc).timestamp()   # 东八区 08-04 02:05
    with tempfile.TemporaryDirectory() as td20:
        corpus20 = _P(td20) / "corpus"
        (corpus20 / "timeline").mkdir(parents=True)
        index20 = _P(td20) / "index"
        index20.mkdir()
        srv20 = MemoryServer(index=MemoryIndex().build(), thread_store=ThreadStore(),
                             corpus_dir=corpus20, time_context=TimeContext(east8_20),
                             index_dir=index20, source_dirs=(corpus20, index20))
        r20 = srv20.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                            "params": {"name": "latent_append",
                                       "arguments": _append_args("凌晨两点记下的一件事。",
                                                                 "记完了。", epoch20,
                                                                 TimeContext(east8_20))}},
                           now=epoch20)
        assert not r20["result"]["isError"], r20
        assert "2026-08-04" in r20["result"]["content"][0]["text"], \
            f"回执里的文件名该是用户自然日：{r20['result']['content'][0]['text']}"
        #    a) 当场：latent_search 标的日期就是用户自然日（宿主在 UTC 也一样）
        s20 = srv20.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                            "params": {"name": "latent_search",
                                       "arguments": {"query": "凌晨两点记下的"}}})
        assert "[2026.08.04" in s20["result"]["content"][0]["text"], \
            f"当场检索的标签该是 08-04：{s20['result']['content'][0]['text']}"
        #    b) 重启：换个新进程态的 server 从盘上重建，四处日期仍要一致
        srv20b = MemoryServer(index=load_corpus((corpus20, index20)), thread_store=ThreadStore(),
                              corpus_dir=corpus20, time_context=TimeContext(east8_20),
                              index_dir=index20, source_dirs=(corpus20, index20))
        b20 = srv20b.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                             "params": {"name": "latent_session_start", "arguments": {}}})
        assert "[2026.08.04" in b20["result"]["content"][0]["text"], \
            f"重启后的开场召回该还是 08-04：{b20['result']['content'][0]['text']}"
        #    c) 走真进程：--timezone 非法值必须非零退出，不许静默退 UTC
        code20, out20 = run_doctor(td20, "--corpus", "corpus", "--timezone", "Asia/Shenzhen")
        assert code20 != 0 and "Asia/Shenzhen" in out20, \
            f"非法时区该报错退出（静默退 UTC 就是把这张卡的缺陷留回去）：{code20} / {out20}"
        #    d) 走真进程：没配 --timezone 时 --doctor 必须报 ⚠ 并打出实际在用的时区
        code20b, out20b = run_doctor(td20, "--corpus", "corpus")
        assert "⚠ 时区" in out20b and "没配 --timezone" in out20b, \
            f"没配时区要报 ⚠，这是那次事故唯一会露头的地方：{out20b}"
        assert "doctor 是独立进程" in out20b and "服务进程配过不算" in out20b, \
            f"时区 WARN 必须说明 doctor 只看自己命令行：{out20b}"
        #    ⚠ 默认值改成东八区之后（2026.08.05 拍板），这一格**不许降成 ✓**：
        #    对东八区以外的使用者它就是错的，降成 ✓ 等于给可能错的日期发合格证
        assert "Asia/Shanghai" in out20b or "UTC+08:00" in out20b, \
            f"没配时也要打出实际在用的默认时区，不能只说「没配」：{out20b}"
        assert code20b == 0, "⚠ 不算失败（能用，只是会悄悄变差），退出码仍该是 0"
        #    e) 配了就该是 ✓ 且报出名字——用户要能一眼认出服务器认的是不是他那个时区
        code20c, out20c = run_doctor(td20, "--corpus", "corpus", "--timezone", east8_20)
        assert f"✓ 时区：{east8_20}" in out20c and code20c == 0, \
            f"配了时区该报 ✓ 并写出名字：{out20c}"

        #    f) 没显式配置且宿主探测值不同：两边的值必须同时露头；同值／探测失败不吵。
        #       stdio 常丢 stderr，所以同一句还必须进入 initialize instructions。
        tz_notice20 = timezone_default_notice("Asia/Shanghai", "UTC")
        assert tz_notice20 and "UTC" in tz_notice20 and "Asia/Shanghai" in tz_notice20, \
            f"时区差异提醒要把探测值与生效值都打出来：{tz_notice20!r}"
        assert timezone_default_notice("Asia/Shanghai", "Asia/Shanghai") is None
        assert timezone_default_notice("Asia/Shanghai", None) is None
        srv20c = MemoryServer(index=MemoryIndex().build(), startup_notice=tz_notice20)
        init20c = srv20c.handle({"jsonrpc": "2.0", "id": 4, "method": "initialize",
                                 "params": {}})
        assert tz_notice20 in init20c["result"]["instructions"], \
            "stdio 客户端看不到 stderr 时，时区提醒仍要随 initialize instructions 送进去"

    # 21.【启动失败要给人话、给出口，stdio 那条路上还要从协议里说得出来】
    #     （2026.08.06，任务卡「服务端起不来时全是裸堆栈」）。
    #     ⚠ **必须起真进程**：这几条全在 `__main__` 的分派里，纯函数断言盖不到——
    #     把 `__main__` 里的 try/except 拆掉，只断 `startup_failure_report` 照样全绿，
    #     那正是"函数绿、生产路径死"那张脸（同 18f 的理由）。
    #     ⚠ **三条失败各造一次且互不串**：每一格既断"自己那句话在"，也断
    #     "另外两条的话不在"——只断前者的话，一个把三条都印一遍的实现也是绿的。
    #     ⚠ **这一项验的是「协议里说得出来」，不是「用户界面上看得见」**：
    #     靶心二那半要拿真客户端（Operit／Claude Code）故意造一次失败、看它界面上
    #     出现了什么，**我们在终端里看到了不算**。别把这一项读成靶心二已经验过。
    #     变异：删掉 __main__ 里的 try/except（回到裸堆栈）/ 让 _startup_failed 直接
    #           sys.exit 不起降级壳 / 三条话术合并成一句 → 各自红
    #     【本轮补的未知加载异常靶心】`_looks_offline()` 认不出的异常不能再被
    #     自信地说成「语料／--corpus 有问题」。2026.08.06 真机夹具用中文 API key
    #     撞出 UnicodeEncodeError，语料完全没坏，旧兜底却把人引去查语料。
    unknown_load21 = startup_failure_report(
        "load", UnicodeEncodeError("latin-1", "中", 0, 1, "合成配置编码错误"),
        embed=True)
    assert "语料读不进来" not in unknown_load21 and "--corpus 指的目录不对" not in unknown_load21, \
        f"未知加载异常不许再冒充已确诊的语料错误：{unknown_load21!r}"
    assert "无法判断" in unknown_load21 and "配置" in unknown_load21 and \
        "依赖" in unknown_load21 and "语料" in unknown_load21, \
        f"未知加载异常要如实列出未决方向，不许只剩一句空泛失败：{unknown_load21!r}"
    assert "--doctor" in unknown_load21 and "完整报错" in unknown_load21, \
        f"中性提示仍要给可行动出口：先体检，再看完整报错：{unknown_load21!r}"

    _MARK21 = {                       # 三条各自的指纹（判据先写下来，再开始数）
        1: "embedding 模型",          # 表里第 1 条：--embed 档，模型没下过 ＋ 没网
        3: "被别的进程占着",           # 表里第 3 条：--http 端口被占
        4: "正在读语料建索引",         # 表里第 4 条：语料大、加载慢（不是失败，只要求有话说）
    }
    # ⚠ 编号照任务卡那张表，**故意不重排**（②已划掉也不把③④提上来）：
    # 重排会让别处对「②」的引用悄悄指向另一条。

    def _only21(text, want):
        """断言 `want` 那条的指纹在、另外两条的不在。"""
        assert _MARK21[want] in text, f"第 {want} 条要说出自己那句人话：{text!r}"
        for other, mark in _MARK21.items():
            if other != want:
                assert mark not in text, \
                    f"第 {want} 条串出了第 {other} 条的话（「{mark}」）：{text!r}"

    import socket as _socket21
    with tempfile.TemporaryDirectory() as td21:
        corpus21 = _P(td21) / "corpus"
        (corpus21 / "timeline").mkdir(parents=True)
        (corpus21 / "timeline" / "window_01_2026-06-01.md").write_text(
            "## 起不来的那天\n服务端一句话都不说，只有一段堆栈。\n", encoding="utf-8")
        logdir21 = _P(td21) / "logs"

        def run_startup21(*argv, stdin_text=None, env_extra=None, timeout=90):
            """起真进程，返回 (退出码, stdout, stderr)。"""
            env21 = dict(os.environ, MEMORY_MCP_LOG_DIR=str(logdir21))
            env21.update(env_extra or {})
            p21 = subprocess.run(
                [sys.executable, str(here / "mcp_server.py"),
                 "--corpus", str(corpus21), *argv],
                input=stdin_text, capture_output=True, text=True,
                encoding="utf-8", env=env21, timeout=timeout)
            return p21.returncode, p21.stdout, p21.stderr

        #    a) 表里第 3 条：`--http` 端口被占。**真占一个端口**，不是模拟异常。
        squat21 = _socket21.socket()
        # ⚠ 不许给占位 socket 开 SO_REUSEADDR：Windows 会允许
        # ThreadingHTTPServer 再绑同一端口，子进程真启动后卡在 serve_forever，
        # 夹具最终只会报 90 秒超时，根本没走到「端口被占」错误出口。
        squat21.bind(("127.0.0.1", 0))
        squat21.listen(1)
        busy_port21 = squat21.getsockname()[1]
        try:
            code21a, _, err21a = run_startup21("--http", f"127.0.0.1:{busy_port21}")
        finally:
            squat21.close()
        assert code21a != 0, "端口被占该非零退出（能起来才是问题）"
        assert "Traceback" not in err21a, \
            f"第 3 条不许再出现裸堆栈——这正是本卡要消灭的东西：{err21a!r}"
        _only21(err21a, 3)
        assert "lsof" in err21a and "--http 127.0.0.1:8766" in err21a, \
            f"第 3 条的出口要可照抄（换端口／查占用进程）：{err21a!r}"

        #    b) 表里第 1 条：`--embed` 拿不到模型。**用一个真的连不上的 endpoint**
        #       （刚关掉的本机端口 → ECONNREFUSED），不看这台机器装没装 fastembed——
        #       靠"本机恰好没装某个包"的夹具，等哪天装上了就会静默变绿。
        dead21 = _socket21.socket()
        dead21.bind(("127.0.0.1", 0))
        dead_port21 = dead21.getsockname()[1]
        dead21.close()
        embed_env21 = {"MEMORY_EMBED_ENDPOINT": f"http://127.0.0.1:{dead_port21}/v1/embeddings",
                       "MEMORY_EMBED_API_KEY": "x-selftest-not-a-real-key"}
        code21b, _, err21b = run_startup21(
            "--embed", "--embed-provider", "cloud:selftest-model",
            "--http", "127.0.0.1:0", env_extra=embed_env21)
        assert code21b != 0, "拿不到模型该非零退出（不许悄悄降级成不带向量那档）"
        assert "Traceback" not in err21b, f"第 1 条不许再出现裸堆栈：{err21b!r}"
        _only21(err21b, 1)
        assert "--step route --route zero-dep" in err21b, \
            f"第 1 条的出口要写清正确出口是重跑 --step route：{err21b!r}"
        assert "不要自己把配置里的 --embed 删掉" in err21b, \
            f"第 1 条必须挡住「手删 --embed」那条隐患（配置要与 init_state 记的路线一致）：{err21b!r}"

        #    c) 落盘：同一句话连完整堆栈写进 MEMORY_MCP_LOG_DIR 指的文件
        log21 = logdir21 / "last-startup-error.log"
        assert log21.exists(), f"启动失败要留一份可贴出来的报告：{logdir21}"
        text21 = log21.read_text(encoding="utf-8")
        assert "记忆协议服务端没能起来。" in text21 and "Traceback" in text21, \
            f"报告里人话与完整堆栈都要有（人话给用户，堆栈给排查）：{text21[:400]!r}"

        #    d) 【靶心二·代码那半】stdio 档失败时**不是死掉，而是起降级壳**，
        #       那段人话从协议里回出去（instructions ＋ 任何 tools/call 的 isError）。
        #       ⚠ 这只证明"协议里说得出来"；"用户界面上看得见"要真机验。
        req21 = "\n".join(json.dumps(m, ensure_ascii=False) for m in (
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": PROTOCOL_VERSION, "clientInfo": {"name": "x"}}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
             "params": {"name": "latent_search", "arguments": {"query": "随便查点什么"}}},
        )) + "\n"
        code21d, out21d, _ = run_startup21(
            "--embed", "--embed-provider", "cloud:selftest-model",
            stdin_text=req21, env_extra=embed_env21)
        resp21 = [json.loads(x) for x in out21d.splitlines() if x.strip()]
        assert len(resp21) == 3, \
            f"降级壳也要照常按协议逐条回（否则客户端还是只有「连接失败」四个字）：{out21d!r}"
        init21, list21, call21 = resp21
        assert init21["result"]["protocolVersion"] == PROTOCOL_VERSION, init21
        _only21(init21["result"]["instructions"], 1)
        assert "--step route --route zero-dep" in init21["result"]["instructions"], \
            "人话要进 instructions——客户端把它注入模型上下文，用户才看得到下一步"
        names21 = [t["name"] for t in list21["result"]["tools"]]
        assert names21 == ["memory_startup_error"], \
            f"⚠ 降级壳一个真工具都不许提供（提供了就是变相放宽检查）：{names21}"
        #       ⚠ 模型习惯性调 latent_search：那也必须回人话，不是「未知工具」
        assert call21["result"]["isError"] is True, \
            f"降级壳里任何调用都必须失败：{call21}"
        _only21(call21["result"]["content"][0]["text"], 1)
        assert code21d != 0, "壳退出时仍要非零——它不是一个能干活的服务"

        #    e) 表里第 4 条：慢的时候有话说（靶心四：**只提示，不动加载时序**）。
        #       ⚠ 这条走函数级夹具而不是真进程：真造一份"大到会慢"的语料，
        #       跑一次要按分钟计，且多大会咬本来就没量过（任务卡第一节末那条）。
        buf21, old21 = io.StringIO(), sys.stderr
        sys.stderr = buf21
        try:
            done21 = slow_load_notice(thresholds=(0.05, 0.15))
            time.sleep(0.4)
            done21()
        finally:
            sys.stderr = old21
        _only21(buf21.getvalue(), 4)
        assert "没有卡死" in buf21.getvalue(), \
            f"第 4 条要正面答「不是卡死」——客户端那头看到的就是像卡死：{buf21.getvalue()!r}"
        #       不慢就不许出声（一进门就刷一行提示，是拿噪声冒充修好了）
        buf21b, old21b = io.StringIO(), sys.stderr
        sys.stderr = buf21b
        try:
            slow_load_notice(stream=buf21b, thresholds=(5.0, 30.0))()
            time.sleep(0.2)
        finally:
            sys.stderr = old21b
        assert buf21b.getvalue() == "", \
            f"加载够快时一个字都不许打：{buf21b.getvalue()!r}"

    # 22.【`被拒 405 GET` 那行要说清它到底是不是故障·三句补注各归各位】
    #     （2026.08.06 立，2026.08.07 按 `HTTP传输不看Accept` 那张卡扩成三句）。
    #     起因：Operit 连 Streamable HTTP 时先发 GET 建 SSE、收 405 后自动回退到 POST，
    #     **连接其实是成功的**，但日志上那行跟真失败长得一模一样——实测里差一点
    #     把一条通的路判成不通，所以在那行后面补了一句「这不是错误」。
    #     ⚠ **2026.08.07 起那句话不能单独存在了**：GET 默认给空长流之后，还会 405
    #     的只剩三种情形，而**只有第一种**是「不是错误」——
    #       ① 客户端明确只要 JSON（它本来就没想建长流）→ `_NOTE_405_FALLBACK`；
    #       ② `--no-sse-stream` 关着 → `_NOTE_405_OFF`（客户端若不回退就真的卡死）；
    #       ③ 长流开满 → `_NOTE_405_BUSY`。
    #     照抄第一句到②③上，就是**拿一个近似的东西冒充判据**（CLAUDE.md 第 10 条）：
    #     把真故障说成不是故障，而且不会报错。
    #     ⚠ **补注只准缀在 405 那一行**：401/403/404/411/413 那几条已经各说各的
    #     原因、工作正常，不许被这句话糊上（所以下面同时断 404 那行三句都没有）。
    #     变异：把 note 去掉 / 把它挂到 _log_deny 所有行上 / 三格共用同一句 → 各自红
    import http.client as _httpc22
    _ALL22 = (_NOTE_405_FALLBACK, _NOTE_405_OFF, _NOTE_405_BUSY)

    def _cap22(fn):
        """跑一段，返回它在 stderr 上留下的全部文本（同 18g 的取证手法）。"""
        buf, old = io.StringIO(), sys.stderr
        sys.stderr = buf
        try:
            fn()
            time.sleep(0.3)
        finally:
            sys.stderr = old
        return buf.getvalue()

    def _get22(port, path="/mcp", accept="application/json", read=True):
        c = _httpc22.HTTPConnection("127.0.0.1", port, timeout=10)
        c.request("GET", path, headers={"Accept": accept} if accept else {})
        r = c.getresponse()
        if read:
            r.read()
            c.close()
            return None
        return c, r

    def _lines22(text, code):
        return [ln for ln in text.splitlines() if ln.startswith(f"被拒 {code} ")]

    with tempfile.TemporaryDirectory() as td22:
        corpus22 = _P(td22)
        (corpus22 / "a.md").write_text("## 一\n随便一段话。\n", encoding="utf-8")
        loader22 = lambda: load_corpus(corpus22)

        def _mk22(sse_stream=True):
            srv = MemoryServer(index=loader22(), thread_store=ThreadStore(),
                               corpus_dir=corpus22, loader=loader22)
            httpd = make_http_server(srv, host="127.0.0.1", port=0,
                                     sse_stream=sse_stream)
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            return httpd, httpd.server_address[1]

        # ① 客户端明确只要 JSON：这一格「不是错误」是对的；404 那行不许被糊上
        httpd22, port22 = _mk22()
        try:
            out22 = _cap22(lambda: (_get22(port22), _get22(port22, path="/nope")))
        finally:
            httpd22.shutdown()
        got405, got404 = _lines22(out22, 405), _lines22(out22, 404)
        assert len(got405) == 1 and _NOTE_405_FALLBACK in got405[0], \
            f"客户端只要 JSON 那格，405 要说「这不是错误」：{got405!r}"
        assert len(got404) == 1 and not any(n in got404[0] for n in _ALL22), \
            f"这三句只属于 405，别糊到别的被拒行上（404 是真的没这个端点）：{got404!r}"

        # ② --no-sse-stream 关着：**不许再说「这不是错误」**，要说出口
        httpd22b, port22b = _mk22(sse_stream=False)
        try:
            out22b = _cap22(lambda: _get22(port22b, accept="text/event-stream"))
        finally:
            httpd22b.shutdown()
        got405b = _lines22(out22b, 405)
        assert len(got405b) == 1 and _NOTE_405_OFF in got405b[0], \
            f"关掉长流那格要指向出口（去掉开关）：{got405b!r}"
        assert _NOTE_405_FALLBACK not in got405b[0], \
            f"这一格照抄「这不是错误」就是把真卡死说成不是故障：{got405b!r}"

        # ③ 长流开满：同样不许说「这不是错误」
        httpd22c, port22c = _mk22()
        held22 = []
        try:
            for _ in range(_HTTP_SSE_MAX_STREAMS):
                c22, r22 = _get22(port22c, accept="text/event-stream", read=False)
                assert r22.read(6) == b": ok\n\n"
                held22.append((c22, r22))   # ⚠ 响应也要攥住，理由见 18h 判据 8
            out22c = _cap22(lambda: _get22(port22c, accept="text/event-stream"))
        finally:
            for c22, _r22 in held22:
                c22.close()
            httpd22c.shutdown()
        got405c = _lines22(out22c, 405)
        assert len(got405c) == 1 and _NOTE_405_BUSY in got405c[0], \
            f"开满那格要说清是开满了：{got405c!r}"
        assert _NOTE_405_FALLBACK not in got405c[0], \
            f"这一格照抄「这不是错误」同样是冒充判据：{got405c!r}"

    # 23.【config 档失败（当前是 --timezone 认不出）也要走那条降级壳·真进程】
    #     （2026.08.22，任务卡「Windows缺tzdata时stdio档启动无出口」）。
    #     缺口（第二节第 3b 条实测）：`--timezone` 解析在 `elif args.corpus:` **之前**，
    #     从前失败走 ap.error → stderr → stdio 客户端把它丢掉，stdout 全空、
    #     last-startup-error.log 也没写——三样保命机制一样没走到。修法是把
    #     `_startup_failed` 提到 args 解析之后，让 config 失败跟 load／bind 同一条出口。
    #     ⚠ **必须起真进程**：这条同 21，全在 __main__ 分派里，纯函数断言盖不到。
    #     ⚠ 采集条件：无 tzdata 的机器上直接用真靶心 Asia/Shanghai；装了 tzdata 的机器
    #        上这个名认得出、判据恒真，改用一个必然认不出的名 Asia/Nowhere 走**同一条
    #        ZoneInfoNotFound → ValueError → _startup_failed("config")** 分支
    #        （照 time_context 自检里 east8 那处同款写法）——结论一个字不变，两种机器都真走过。
    #     变异（判据第 2 条，维护者点名要留）：把 `_startup_failed("config", e)` 改回
    #        `ap.error(str(e))` → 下面 a) 必须转红（stdout 变空）。不转红等于没测。
    #     反向约束：同样的失败加上 --http 不许走 stdio 壳（b)：stdout 仍空、非零退出）。
    bad_tz23 = "Asia/Shanghai" if not tzdb_available() else "Asia/Nowhere"
    with tempfile.TemporaryDirectory() as td23:
        corpus23 = _P(td23) / "corpus"
        (corpus23 / "timeline").mkdir(parents=True)
        (corpus23 / "timeline" / "window_01_2026-06-01.md").write_text(
            "## 起不来的那天\n配错了时区，服务端从前一个字都不说。\n", encoding="utf-8")
        logdir23 = _P(td23) / "logs"

        def run_cfg23(*argv, stdin_text=None):
            """起真进程：--corpus <上面那个> --timezone <认不出的名> ＋ 可选 argv。"""
            env23 = dict(os.environ, MEMORY_MCP_LOG_DIR=str(logdir23))
            p23 = subprocess.run(
                [sys.executable, str(here / "mcp_server.py"),
                 "--corpus", str(corpus23), "--timezone", bad_tz23, *argv],
                input=stdin_text, capture_output=True, text=True,
                encoding="utf-8", env=env23, timeout=90)
            return p23.returncode, p23.stdout, p23.stderr

        init23 = json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": PROTOCOL_VERSION,
                        "clientInfo": {"name": "x"}}}, ensure_ascii=False) + "\n"

        #    a) stdio 档（靶心）：喂一条 initialize，stdout 必须从协议里回出人话
        code23a, out23a, _ = run_cfg23(stdin_text=init23)
        resp23 = [json.loads(x) for x in out23a.splitlines() if x.strip()]
        assert len(resp23) == 1 and resp23[0].get("result", {}).get("instructions"), \
            f"config 失败的 stdio 档也要按协议回、把人话塞进 instructions：{out23a!r}"
        instr23 = resp23[0]["result"]["instructions"]
        assert "记忆协议服务端没能起来" in instr23 and "tzdata" in instr23, \
            f"人话要带上 exc 自己那段出口（pip install tzdata），不另造措辞：{instr23!r}"
        assert code23a != 0, "降级壳退出仍要非零——它不是能干活的服务"
        log23 = logdir23 / "last-startup-error.log"
        assert log23.exists(), f"config 失败也要落盘 last-startup-error.log：{logdir23}"

        #    b) 反向：同一个失败加 --http → 不许走 stdio 壳，stdout 空、stderr 给人话、非零
        code23b, out23b, err23b = run_cfg23("--http", "127.0.0.1:0", stdin_text=init23)
        assert out23b.strip() == "", \
            f"--http 档 stdout 不许出协议流（这档 stdout 是给人读的横幅，套壳会破坏它）：{out23b!r}"
        assert code23b != 0 and "记忆协议服务端没能起来" in err23b, \
            f"--http 档失败要在 stderr 给人话并非零退出：{err23b!r}"

    # 24.【--log-file：stderr 那条路要留得下痕，且重启看得出来】
    #     （2026.09.03，issue #20）。缺口：stdio 形态下 stderr 由客户端接管、
    #     经常被直接丢弃，子进程崩没崩／何时重启没有任何落点，报告人只能靠回头翻
    #     token 账单发现某段前缀被重复计费、倒推"是不是重启过"。
    #     ⚠ **必须起真进程**：tee 装在 `__main__` 的 args 解析之后，纯函数断言盖不到。
    #     ⚠ 判据分两条，b) 才是这张卡真正要的东西：
    #        a) 配了 --log-file，本该只走 stderr 的诊断（这里借启动失败那段人话）落进文件；
    #        b) **同一个文件跑第二次 → 两条进程横幅**，即"重启过"可以被直接读出来。
    #     变异（照 23 的规矩留）：把 `sys.stderr = install_stderr_tee(...)` 那行删掉
    #        → a) 必转红（文件里只剩横幅、没有那段人话）。不转红等于没测。
    with tempfile.TemporaryDirectory() as td24:
        corpus24 = _P(td24) / "corpus"
        (corpus24 / "timeline").mkdir(parents=True)
        (corpus24 / "timeline" / "window_01_2026-06-01.md").write_text(
            "## 留痕那天\n从前 stderr 掉进黑洞，这次要落盘。\n", encoding="utf-8")
        log24 = _P(td24) / "logs" / "mcp.log"      # 父目录故意不存在：顺带验它会自建

        def run_log24():
            """起真进程，走一条必然失败的启动（认不出的时区），看它往日志里留了什么。"""
            p24 = subprocess.run(
                [sys.executable, str(here / "mcp_server.py"),
                 "--corpus", str(corpus24), "--timezone", "Asia/Nowhere",
                 "--log-file", str(log24), "--http", "127.0.0.1:0"],
                capture_output=True, text=True, encoding="utf-8",
                env=dict(os.environ, MEMORY_MCP_LOG_DIR=str(_P(td24) / "startuplog")),
                timeout=90)
            return p24.returncode

        run_log24()
        assert log24.exists(), f"配了 --log-file 就要把文件建出来（父目录也要自建）：{log24}"
        body24a = log24.read_text(encoding="utf-8")
        #    a) 靶心：本该只走 stderr 的那段人话，确实落进了文件
        assert "记忆协议服务端没能起来" in body24a, \
            f"stderr 上的诊断要同时落盘，不能只留横幅：{body24a!r}"
        assert "进程启动 pid=" in body24a, f"每次启动要写一行带 pid 的横幅：{body24a!r}"

        #    b) 靶心（issue #20 真正要的）：再跑一次 → 同一个文件里两条横幅＝重启看得出来
        run_log24()
        body24b = log24.read_text(encoding="utf-8")
        assert body24b.count("进程启动 pid=") == 2, \
            ("同一个 --log-file 跑两次要留下两条横幅（这正是“子进程重启过”的直接证据），"
             f"实际 {body24b.count('进程启动 pid=')} 条")
        assert body24b.startswith(body24a), "追加写：第二次不许把第一次的内容截掉"

    print("selftest ok（25项断言：握手 / 工具表 / 调用往返 / 薄适配层 / 错误分层（含 "
          "session_start 读取异常只重试一次、未预料异常不空断、写工具不重放）/ "
          "完整链路 / stdio / UTF-8 / 写回当场可查 / 写回预检（零写入、补索引预检、"
          "参数错误写错／写对对照）/ 精准清理（两阶段确认、引用阻断、"
          "隔离备份不回灌、sidecar 同步与故障回滚）/ 用进撑过重启 / "
          "无可靠命中明确说 / 撤回更正闭环 / 撤回粒度提醒（块级撤回、事实跨块：命中多块"
          "给总数＋最近定位、0 命中不冒提示且变异去 exclude 转红、只提示不自动撤）/ "
          "缺失率标注接线 / 实体标注接线 / "
          "部署体检只读 / 部署体检走真进程（相对路径解析开、空语料非零退出、"
          "每句一个 `##` 的导出要报切块警告）/ HTTP 传输真端口往返（鉴权 401、"
          "非回环裸跑拒绝、非 ASCII token 拒绝起动、Origin 403、路径 404、通知 202、"
          "只要 JSON 的 GET 回 405、批量 400、超限 413、chunked 501、"
          "keep-alive 下被拒后同连接重试、"
          "UTF-8 写回即查、语料变化自动重读、同窗口手动上传不被写回吞掉、"
          "非回环绑定的起动横幅多打一行 TLS 提醒／回环时不打——真进程两个绑法都跑、"
          "被拒的请求在服务端留一行（码／方法／剥掉 query 的路径／客户端 IP／原因，"
          "正常请求照旧不打，token 与请求体不进日志，刷屏上限 60 行且溢出条数补一行）"
          "）/ "
          "CLI 入口把 stdout 锁成 UTF-8（⚠ 变异要在 PYTHONIOENCODING=gbk 下跑，"
          "默认 UTF-8 的机器上这条恒真）/ "
          "时区穿到进程级（--timezone 一路到文件名／H2／当场检索标签／重启后开场召回，"
          "非法名非零退出、没配报 ⚠ 且打出实际时区、配了报 ✓ 并写出名字）/ "
          "启动失败给人话给出口·真进程（端口被占／拿不到模型／慢加载三条各造一次且互不串，"
          "都不再是裸堆栈，报告连堆栈落盘；stdio 档起降级壳把那段话从协议里回出去——"
          "instructions ＋ 任何 tools/call 都 isError，且一个真工具都不提供。"
          "⚠ 这验的是「协议里说得出来」，「用户界面上看得见」要真机）/ "
          "config 档失败也走同一条出口·真进程（--timezone 认不出，从前在 corpus 分支"
          "之前失败、stdout 全空；现在 stdio 档从协议里回人话＋落盘，反向断言 --http "
          "档 stdout 仍空不套壳）/ "
          "按 Accept 给客户端要的形态（只要 SSE 的 POST 回单事件 SSE 且能解回原响应，"
          "两个都写／`*/*`／没带头一律照旧回 JSON——回归护栏；GET 给只发心跳、"
          "永不发消息的空长流，没带 Accept 的懒客户端也给，心跳真的发，"
          "开满 4 条回 405，--no-sse-stream 退回恒 405。"
          "⚠ 这验的是「服务端形态已具备」，「哪个客户端接上了」要真机）/ "
          "被拒 405 GET 那行按三种原因各说各的（只要 JSON＝这不是错误、"
          "长流被关掉＝指出口、开满＝说开满），且三句都不许糊到别的被拒行上）/ "
          "--log-file 让 stderr 留得下痕·真进程（诊断人话落盘、父目录自建、"
          "每次启动一条带 pid 的横幅，同一文件跑两次＝两条横幅且追加不截断——"
          "「子进程重启过」由此可直接读出，不必再靠 token 账单倒推）")


if __name__ == "__main__":
    # 中文 Windows（cp936/GBK）下 stdout 按系统区域编码写，`--doctor` 的中文报告
    # 一遇到 emoji 就 UnicodeEncodeError（缘由与判据见 memory_init.py 同一处注释）。
    # stdio 传输那条路本来就已经自己包了 UTF-8（见 serve_stdio），这里补的是
    # **面向控制台的那部分输出**；改的是编码，不是 ensure_ascii。
    # ⚠ 只放在 `__main__` 里：被 import 时不许改掉调用方进程的 stdout。
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--doctor", action="store_true",
                    help="部署体检：查语料目录、时间戳成色、sidecar、写回落点与 MCP 接线，"
                         "只读不写盘。有一项过不去时退出码为 1")
    ap.add_argument("--corpus", help="md 语料目录")
    ap.add_argument("--index-dir",
                    help="独立索引摘要目录（推荐；未配时 latent_append 兼容写入 <corpus>/index）")
    ap.add_argument("--write-dir",
                    help="latent_append／latent_correct 的正文落点（未配时用 "
                         "<corpus>/timeline）。**必须仍在 --corpus 之内**，检索范围只按 "
                         "--corpus 递归；配它是为了把这支笔写的记录跟语料里别处"
                         "（例如宿主客户端自动记的那一档）物理分开")
    ap.add_argument("--threads", help="会话线索 jsonl 路径（省略则内存态）")
    ap.add_argument("--require-unresolved-review", action="store_true",
                    help="严格要求完整 append／thread_close 显式提供 unresolvedOps；"
                         "只卡字段缺失，清单操作失败仍不阻断正文／thread")
    ap.add_argument("--embed", action="store_true", help="用真 embedding")
    ap.add_argument("--embed-provider", dest="embed_provider",
                    help="embedding 提供方：local（默认，需 fastembed）/ local:<模型> / "
                         "cloud（云端 HTTP，endpoint 与模型走 MEMORY_EMBED_* 环境变量，"
                         "key 只从环境变量读；**语料会发到那家服务商**）")
    ap.add_argument("--http", metavar="[HOST:]PORT",
                    help="改走 Streamable HTTP（省略 HOST 默认 127.0.0.1）——给只认 "
                         "HTTP 的客户端（Kelivo/Operit 这类）直连用，不再需要 "
                         "supergateway 桥。非回环地址必须配 token，否则拒绝起动；"
                         "绑非回环还会多打一行 TLS 提醒——那条链路是裸 HTTP")
    ap.add_argument("--no-sse-stream", dest="sse_stream", action="store_false",
                    help="GET 不给 SSE 空长流、退回恒回 405 的旧行为。默认是给的："
                         "有的客户端先发 GET 建长流、收到 405 也不改用 POST，就卡在"
                         "等长流上（表现是「连不上」）。那条长流永远不发消息、只发心跳，"
                         "本 server 没有服务器主动推送的场景")
    ap.add_argument("--timezone", metavar="IANA名",
                    help="记忆所有者的时区（例如 Asia/Shanghai），也可用环境变量 "
                         "MEMORY_TIMEZONE。自然日＝这个时区的 00:00～23:59，写回归窗、"
                         "记录标题与检索标签同一个口径。不配就用默认值 Asia/Shanghai"
                         "（东八区），--doctor 会报 ⚠——⚠ 不在东八区的话一定要配，"
                         "否则写下的记忆日期会静默错一天。没有 IANA 时区数据库的机器"
                         "（Windows 裸装）可以装 tzdata，或退而写固定偏移 UTC+08:00"
                         "（⚠ 不懂夏令时）")
    ap.add_argument("--token",
                    help="HTTP 传输的 Bearer token（也可用环境变量 MEMORY_HTTP_TOKEN；"
                         "客户端侧填进 bearerToken/Authorization 头）")
    ap.add_argument("--log-file", dest="log_file", metavar="路径",
                    help="把本该只走 stderr 的诊断输出（启动横幅、拒绝记录、异常堆栈）"
                         "同时追加落盘一份。stdio 形态下 stderr 由客户端接管、经常被直接"
                         "丢弃，不配这个就等于没有任何排查落点。文件按追加写，每次进程"
                         "启动写一行带 pid 的横幅——同一个文件里出现第二条横幅即可证明"
                         "子进程重启过")
    args = ap.parse_args()

    # ⚠ 紧跟在 parse 之后：再往下就是 config／load／bind 三段，那几段的失败正是最需要
    # 落盘的一类。装晚一行，就漏掉一类。
    if args.log_file:
        sys.stderr = install_stderr_tee(args.log_file)

    # 这次运行最终会不会把 stdout 交给 MCP 协议流？只有它为真，启动失败才需要走协议
    # 降级壳；这条谓词跟失败发生在哪一行无关（任务卡「stdio 档启动失败无出口」第一节）。
    # 它同时是既有 `if args.http:` 那半判据的补全——把 selftest／doctor／没给 corpus 也算上。
    _will_serve_stdio = bool(args.corpus) and not args.http \
                        and not args.selftest and not args.doctor

    def _startup_failed(stage, exc):
        """`args` 解析之后任何启动失败的唯一出口：给人话、落盘，再按传输形态送到用户
        眼前。作用域是 `args` 解析之后的**所有**失败（config／load／bind），不再只
        覆盖 `elif args.corpus:` 之内那一段（2026.08.22「stdio 档启动失败无出口」把它提前）。

        ⚠ **别在这里加任何"那就降级起来吧"的分支**——检查该拒还是拒
        （任务卡「服务端起不来时全是裸堆栈」第五节第一条：这次的问题是失败时没给
        出路，不是失败判得太严）。"""
        path = write_startup_log(
            startup_failure_report(stage, exc, embed=args.embed,
                                   http_bind=args.http), exc)
        report = startup_failure_report(stage, exc, embed=args.embed,
                                        http_bind=args.http, log_path=path)
        print(report, file=sys.stderr)
        if not _will_serve_stdio:
            # ⚠ 只有 stdout 将成为协议流时才套降级壳。--http／--doctor／--selftest／
            # 没给 --corpus 这几条，stdout 是给人读的（横幅／体检报告／帮助），人就在
            # 终端前盯着 stderr——**绝不许给它们套壳**：往给人看的输出里吐 JSON-RPC，
            # 是把一个能用的输出变成不能用的（任务卡第一节反向约束）。照常非零退出。
            sys.exit(1)
        # stdio 档：stderr 进不了用户界面（客户端只显示「连接失败」），
        # 改由降级壳把这段话从协议里送过去（2026.08.06 维护者拍板的修法）
        StartupFailureServer(report).serve_stdio()
        sys.exit(1)

    # 时区一个进程一份，stdio／HTTP／--doctor 三条路共用（任务卡"写回时区与跨日归窗"）。
    # ⚠ 非法时区名不静默退 UTC，走上面的启动失败唯一出口：stdio 档从前它走 ap.error →
    #   stderr → 客户端丢弃，stdout 全空、last-startup-error.log 也没写，一个字到不了
    #   用户眼前（任务卡「stdio 档启动失败无出口」第二节第 3b 条实测）。
    tz_name = args.timezone or os.environ.get("MEMORY_TIMEZONE") or ""
    try:
        time_ctx = TimeContext(tz_name.strip()) if tz_name.strip() else TimeContext.default()
    except ValueError as e:
        _startup_failed("config", e)
    timezone_notice = None
    if args.corpus and not args.selftest and not args.doctor and not tz_name.strip():
        timezone_notice = timezone_default_notice(time_ctx.name, detect_local_timezone())
        if timezone_notice:
            # HTTP／手动启动的人直接在终端看；stdio 客户端可能丢 stderr，下面还会把
            # 同一句接进 initialize instructions，不能只满足“代码确实 print 过”。
            print(timezone_notice, file=sys.stderr)
    if args.selftest:
        _selftest()
    elif args.doctor:
        if not args.corpus:
            ap.error("--doctor 要跟 --corpus 一起用：体检的就是它指向的那个目录")
        checks = diagnose(args.corpus, threads_path=args.threads, embed=args.embed,
                          time_context=time_ctx, index_dir=args.index_dir,
                          require_unresolved_review=args.require_unresolved_review,
                          write_dir=args.write_dir)
        print(format_doctor_report(checks))
        # 退出码给自动化用：有 ✗ 就非零，⚠ 不算失败（那些是"能用但会悄悄变差"）
        sys.exit(1 if any(c["level"] == FAIL for c in checks) else 0)
    elif args.corpus:
        # 慢的时候要有话说（第 4 条）：只加提示，不动加载时序（靶心四）
        _load_done = slow_load_notice()
        try:
            source_dirs, loader = make_corpus_loader(
                args.corpus, index_dir=args.index_dir, embed=args.embed,
                provider=(resolve_provider(args.embed_provider)
                          if args.embed else None))
            srv = MemoryServer(index=loader(),
                               thread_store=ThreadStore(args.threads),
                               corpus_dir=args.corpus,
                               weights_path=Path(args.corpus) / ".weights.json",
                               retractions_path=Path(args.corpus) / ".retractions.json",
                               entities_path=Path(args.corpus) / ".entities.json",
                               loader=loader, time_context=time_ctx,
                               source_dirs=source_dirs, index_dir=args.index_dir,
                               write_dir=args.write_dir,
                               startup_notice=timezone_notice,
                               require_unresolved_review=args.require_unresolved_review)
        except Exception as e:                       # noqa: BLE001（裸堆栈正是本卡要消灭的）
            _load_done()
            _startup_failed("load", e)
        finally:
            _load_done()
        if args.http:
            host, _, port = args.http.rpartition(":")
            host = host or "127.0.0.1"
            token = args.token or os.environ.get("MEMORY_HTTP_TOKEN") or None
            try:
                httpd = make_http_server(srv, host=host, port=int(port), token=token,
                                         sse_stream=args.sse_stream)
            except ValueError as e:
                ap.error(str(e))
            except OSError as e:                     # 端口被占等绑定失败（第 3 条）
                _startup_failed("bind", e)
            # 起动横幅进 stderr（stdio 传输里 stdout 是协议流，这里沿用习惯）
            print(f"Streamable HTTP 服务在 http://{host}:{httpd.server_address[1]} "
                  f"（鉴权：{'Bearer token' if token else '无——仅限回环+外层反代'}；"
                  f"语料变化自动重读）", file=sys.stderr)
            # 非回环绑定多打一行 TLS 提醒：只提示、不拒绝起动，也不看有没有配 token
            notice = http_tls_notice(host)
            if notice:
                print(notice, file=sys.stderr)
            httpd.serve_forever()
        else:
            srv.serve_stdio()
    else:
        ap.print_help()
