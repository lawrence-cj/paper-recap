#!/usr/bin/env python3
"""Run the Paper Recap repository's canonical content validation."""

from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[4]
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_site.py"

if not BUILD_SCRIPT.exists():
    print(f"找不到构建脚本：{BUILD_SCRIPT}", file=sys.stderr)
    raise SystemExit(2)

raise SystemExit(subprocess.call([sys.executable, str(BUILD_SCRIPT), "--check"], cwd=REPO_ROOT))
