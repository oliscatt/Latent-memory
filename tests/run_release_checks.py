#!/usr/bin/env python3
"""公仓发布检查入口：逐个运行 ``src/*.py --selftest``，不安装第三方依赖。"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def source_files():
    """以目录现状为准，不写死数量；新增文件会自动进入下一次检查。"""
    return sorted(SRC.glob("*.py"), key=lambda path: path.name)


def git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def run_one(path, timeout):
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    started = time.monotonic()
    try:
        result = subprocess.run(
            [sys.executable, str(path), "--selftest"], cwd=SRC, env=env,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, check=False,
        )
        return {
            "file": path.name,
            "status": "passed" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "duration_seconds": round(time.monotonic() - started, 3),
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "file": path.name,
            "status": "timeout",
            "returncode": None,
            "duration_seconds": round(time.monotonic() - started, 3),
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description="运行 Latent 公仓全部零依赖自检")
    parser.add_argument("--timeout", type=int, default=180,
                        help="单个文件的超时秒数，默认 180")
    parser.add_argument("--json-out", help="把完整结果写入 JSON；可供 CI 留作构建产物")
    parser.add_argument("--list", action="store_true", help="只列出将运行的文件")
    args = parser.parse_args(argv)

    files = source_files()
    if not files:
        print("失败：src/ 下没有可检查的 .py 文件", file=sys.stderr)
        return 1
    if args.list:
        for path in files:
            print(path.name)
        return 0

    results = []
    for path in files:
        print(f"[运行] {path.name}", flush=True)
        item = run_one(path, args.timeout)
        results.append(item)
        print(f"[{item['status']}] {path.name}  {item['duration_seconds']:.3f}s",
              flush=True)
        if item["status"] != "passed":
            if item["stdout"]:
                print("--- stdout ---", file=sys.stderr)
                print(item["stdout"], file=sys.stderr)
            if item["stderr"]:
                print("--- stderr ---", file=sys.stderr)
                print(item["stderr"], file=sys.stderr)

    report = {
        "schema_version": 1,
        "criterion": "src/ 下每个 .py 以当前解释器运行 --selftest，退出码均为 0",
        "collection": {
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
            "git_commit": git_commit(),
            "file_count": len(files),
            "per_file_timeout_seconds": args.timeout,
        },
        "summary": {
            "passed": sum(item["status"] == "passed" for item in results),
            "failed": sum(item["status"] == "failed" for item in results),
            "timeout": sum(item["status"] == "timeout" for item in results),
        },
        "results": results,
    }
    if args.json_out:
        target = Path(args.json_out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")

    ok = report["summary"]["passed"] == len(files)
    print(f"\n结果：{report['summary']['passed']}/{len(files)} 通过")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
