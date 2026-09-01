#!/usr/bin/env python3
"""参考宿主（任务卡「自建前端注入契约」第二单第 4 条）。

它是注入契约的**可执行规格**：文字规则有歧义，跑得起来的一百行没有。
做四件事：读 persona → 按契约五条拼请求 → 调 OpenAI 兼容 API → 把 MCP 工具
暴露给模型。同时是跨模型试验台（契约坑④"模型异构未实测"的实测载体，DS 起步）。

刹车（内部设计资料里写死，逐字执行）：**永远不超过"能验证契约"的最小形态**——
多一个产品功能（流式、多轮管理、配置系统、idle 超时……）就是在做前端，直接砍。
会话 = 进程一次运行：每次起动从磁盘重读 persona（契约三），exit/Ctrl-D 结束。
latent_session_start / latent_thread_close 都不由宿主代调——模型主不主动，正是异构实测要看的。

用法：
  python reference_host.py <产出目录>    # 目录里要有 persona.md 与 mcp-config.json
  python reference_host.py --selftest   # 不联网不要 key：契约逐条断言 + 真起一次 server
环境变量：HOST_API_KEY（必填；key 只从环境读，不写进任何文件——凭证不入库）、
  HOST_API_BASE（默认 https://api.deepseek.com/v1）、HOST_MODEL（默认 deepseek-chat）。
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

API_BASE = os.environ.get("HOST_API_BASE", "https://api.deepseek.com/v1")
MODEL = os.environ.get("HOST_MODEL", "deepseek-chat")


class McpClient:
    """按产出目录里的 mcp-config.json 起 stdio server，握手后只做转发。"""

    def __init__(self, config_path):
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))["mcpServers"]["memory"]
        # encoding 锁死 UTF-8：MCP 规格定死 stdio 是 UTF-8，Windows 默认 cp936 会
        # 静默乱码（mcp_server.py serve_stdio 的注释记着那次真机 bug，两头都要锁）
        self.proc = subprocess.Popen([cfg["command"], *cfg["args"]],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     encoding="utf-8")
        self._id = 0
        init = self._rpc("initialize", {"protocolVersion": "2025-06-18",
                                        "clientInfo": {"name": "reference-host"}})
        # instructions 拿到手就要递给模型（进易变块）——坑①说的"很多前端不递"，
        # 参考宿主不能自己就是反面教材
        self.instructions = init.get("instructions", "")
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self.tools = self._rpc("tools/list")["tools"]

    def _send(self, msg):
        self.proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

    def _rpc(self, method, params=None):
        self._id += 1
        self._send({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}})
        for line in self.proc.stdout:
            msg = json.loads(line)
            if msg.get("id") == self._id:
                if "error" in msg:
                    raise RuntimeError(msg["error"]["message"])
                return msg["result"]
        raise RuntimeError("server 进程退出，没等到响应")

    def call(self, name, args):
        r = self._rpc("tools/call", {"name": name, "arguments": args})
        return r["content"][0]["text"]   # isError 的文本也原样交给模型——它要看到失败原因


def read_persona(out_dir):
    """契约三：每次会话从磁盘重读。宿主一次运行就是一个会话，每次起动都走这里，不缓存。"""
    return (Path(out_dir) / "persona.md").read_text(encoding="utf-8")


def build_request(persona, volatile, history, tools):
    """契约的可执行形态，五条都落在这一个函数里：
    一/四：persona 逐字整块独占第一条消息，宿主一个字不掺进块内；
    二：每一轮都从这里重拼，persona 是固定前缀，不跟历史一起滚动、不被截断；
    五：易变内容（server 指引、当前时间）排在 persona 之后；工具清单走 API 的
        tools 字段，不占消息正文。"""
    return {"model": MODEL,
            "messages": [{"role": "system", "content": persona},
                         {"role": "system", "content": volatile}] + history,
            "tools": [{"type": "function",
                       "function": {"name": t["name"], "description": t["description"],
                                    "parameters": t["inputSchema"]}} for t in tools]}


def chat(payload):
    req = urllib.request.Request(
        API_BASE.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + os.environ["HOST_API_KEY"]})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]


def run_turn(mcp, persona, history, transport=chat):
    """一轮对话：模型要用工具就转发给 server，直到它给出文字回答。"""
    while True:
        volatile = mcp.instructions + "\n当前时间：" + time.strftime("%Y-%m-%d %H:%M")
        msg = transport(build_request(persona, volatile, history, mcp.tools))
        history.append(msg)
        if not msg.get("tool_calls"):
            return msg.get("content") or ""
        for c in msg["tool_calls"]:
            print(f"  [工具] {c['function']['name']}", file=sys.stderr)
            try:
                text = mcp.call(c["function"]["name"],
                                json.loads(c["function"]["arguments"] or "{}"))
            except Exception as e:   # 工具坏了如实告诉模型，不吞掉装没事
                text = f"工具调用失败：{e}"
            history.append({"role": "tool", "tool_call_id": c["id"], "content": text})


def main(out_dir):
    if not os.environ.get("HOST_API_KEY"):
        sys.exit("先设 HOST_API_KEY 环境变量（key 只从环境读，不写进任何文件）。")
    persona = read_persona(out_dir)
    mcp = McpClient(Path(out_dir) / "mcp-config.json")
    print(f"参考宿主就绪：{MODEL} @ {API_BASE}，工具 {len(mcp.tools)} 个。exit / Ctrl-D 结束。")
    history = []
    while True:
        try:
            line = input("你> ").strip()
        except EOFError:
            break
        if not line or line == "exit":
            break
        history.append({"role": "user", "content": line})
        print("模型> " + run_turn(mcp, persona, history))
    mcp.proc.terminate()


# ---------- selftest（不联网、不要 key；契约五条逐条断言 + 真起一次 server） ----------

def _selftest():
    import tempfile
    persona = "# 人格\n开篇立场。\n## 检索约定\n先查再答。\n最终约定收尾。\n"
    tools = [{"name": "t", "description": "d", "inputSchema": {"type": "object"}}]
    long_hist = [{"role": "user", "content": f"第 {i} 句"} for i in range(30)]
    req1 = build_request(persona, "指引甲\n当前时间：轮一", [{"role": "user", "content": "嗨"}], tools)
    req30 = build_request(persona, "指引甲\n当前时间：轮三十", long_hist, tools)

    # 契约一/四：第一条消息逐字 == 磁盘内容，块内没有任何宿主内容（diff 为空即整块）
    assert req1["messages"][0] == {"role": "system", "content": persona}, \
        "人格块必须逐字独占第一条消息，宿主内容一个字不许掺进去"
    # 契约二：第 1 轮与第 30 轮的人格块完全相同——不跟历史滚动、不被截断
    assert req30["messages"][0] == req1["messages"][0], "第 30 轮的人格块必须和第 1 轮一模一样"
    # 契约五：序列化后人格块起始偏移量每轮相同（判据原文），且易变内容在人格块之后
    needle = json.dumps(persona, ensure_ascii=False)[1:-1]
    blob1, blob30 = (json.dumps(r, ensure_ascii=False) for r in (req1, req30))
    assert blob1.index(needle) == blob30.index(needle), \
        "人格块起始偏移随轮次漂移＝有易变内容跑到它前面去了"
    assert blob1.index(needle) < blob1.index("当前时间"), "易变内容必须在人格块之后"

    # 契约三：改磁盘上的文件，下一个"会话"（下一次读）必须立刻生效——缓存旧版即违反
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "persona.md"
        p.write_text("v1", encoding="utf-8")
        assert read_persona(td) == "v1"
        p.write_text("v2（用户升层后的新版）", encoding="utf-8")
        assert read_persona(td).startswith("v2"), "改盘上的 persona 必须立刻生效"

    # MCP 桥：按 mcp-config.json 真起一个 server 进程——握手、instructions、
    # 工具表、一次中文真检索（UTF-8 两头锁死的证据）
    with tempfile.TemporaryDirectory() as td:
        corpus = Path(td) / "corpus"
        corpus.mkdir()
        (corpus / "w.md").write_text("## 修咖啡机\n保险丝熔断，换上通电正常。", encoding="utf-8")
        cfgp = Path(td) / "mcp-config.json"
        server = Path(__file__).resolve().parent / "mcp_server.py"
        cfgp.write_text(json.dumps({"mcpServers": {"memory": {
            "command": sys.executable,
            "args": [str(server), "--corpus", str(corpus)]}}}), encoding="utf-8")
        mcp = McpClient(cfgp)
        try:
            assert [t["name"] for t in mcp.tools] == ["latent_search", "latent_session_start",
                                                      "latent_append", "latent_correct",
                                                      "latent_cleanup",
                                                      "latent_unresolved",
                                                      "latent_thread_close"]
            assert "长期" in mcp.instructions, "instructions 要拿到手——坑①的宿主侧责任"
            assert "保险丝熔断" in mcp.call("latent_search", {"query": "咖啡机"})

            # 工具调用回路（假 API，不联网）：先要工具、再作答；tool 结果要回进
            # history，且回路里每一轮的请求都重走 build_request（契约二穿透真回路）
            seen = []
            replies = [{"role": "assistant", "content": None, "tool_calls": [
                            {"id": "c1", "type": "function",
                             "function": {"name": "latent_search",
                                          "arguments": '{"query": "咖啡机"}'}}]},
                       {"role": "assistant", "content": "查到了。"}]
            fake = lambda payload: (seen.append(payload), replies.pop(0))[1]
            hist = [{"role": "user", "content": "咖啡机怎么修的？"}]
            assert run_turn(mcp, persona, hist, transport=fake) == "查到了。"
            assert any(m.get("role") == "tool" and "保险丝熔断" in m["content"] for m in hist), \
                "工具结果必须以 tool 消息回进 history，模型才看得到"
            assert all(s["messages"][0]["content"] == persona for s in seen), \
                "回路里每一轮请求的第一条消息都必须是逐字 persona（契约二）"
        finally:
            mcp.proc.terminate()
    print("selftest 通过：契约五条 + MCP 桥 + 工具回路")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        sys.exit(__doc__)
