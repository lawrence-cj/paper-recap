#!/usr/bin/env python3
"""Validate paper notes and build the dependency-free static site."""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTENT_DIR = ROOT / "content" / "papers"
DIST_DIR = ROOT / "dist"
REQUIRED_FIELDS = {
    "title", "paper_url", "authors", "venue", "published", "read_date",
    "status", "rating", "tags", "one_liner",
}
REQUIRED_SECTIONS = (
    "研究问题", "核心方法", "关键发现", "我的提问",
    "局限与疑问", "我的判断", "下次只看这些",
)


class ContentError(ValueError):
    pass


def parse_scalar(raw: str):
    raw = raw.strip()
    if not raw:
        return ""
    if raw.startswith("[") and raw.endswith("]"):
        try:
            return ast.literal_eval(raw)
        except (SyntaxError, ValueError) as exc:
            raise ContentError(f"无法解析列表 {raw!r}") from exc
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    return raw


def parse_note(path: Path) -> dict:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if not match:
        raise ContentError("必须以 --- 包围的元数据开头")

    metadata = {}
    for line_number, line in enumerate(match.group(1).splitlines(), start=2):
        if not line.strip():
            continue
        if ":" not in line:
            raise ContentError(f"第 {line_number} 行不是 key: value 格式")
        key, raw_value = line.split(":", 1)
        metadata[key.strip()] = parse_scalar(raw_value)

    missing = sorted(REQUIRED_FIELDS - metadata.keys())
    if missing:
        raise ContentError(f"缺少字段：{', '.join(missing)}")
    if not isinstance(metadata["tags"], list) or not metadata["tags"]:
        raise ContentError("tags 必须是至少含一个主题的列表")
    if not all(isinstance(tag, str) and tag.strip() for tag in metadata["tags"]):
        raise ContentError("tags 中的每一项都必须是非空文本")
    if not isinstance(metadata["rating"], int) or not 1 <= metadata["rating"] <= 5:
        raise ContentError("rating 必须是 1 到 5 的整数")
    try:
        datetime.strptime(str(metadata["read_date"]), "%Y-%m-%d")
    except ValueError as exc:
        raise ContentError("read_date 必须使用 YYYY-MM-DD") from exc
    if not str(metadata["title"]).strip() or not str(metadata["one_liner"]).strip():
        raise ContentError("title 和 one_liner 不能为空")

    body = match.group(2).strip()
    if body.count("$$") % 2:
        raise ContentError("独立公式的 $$ 分隔符没有成对闭合")
    if body.count("\\[") != body.count("\\]"):
        raise ContentError("独立公式的 \\[ 与 \\] 分隔符没有成对闭合")
    headings = set(re.findall(r"^##\s+(.+?)\s*$", body, re.MULTILINE))
    missing_sections = [section for section in REQUIRED_SECTIONS if section not in headings]
    if missing_sections:
        raise ContentError(f"缺少章节：{', '.join(missing_sections)}")

    metadata["slug"] = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", path.stem)
    metadata["body"] = body
    metadata["source_file"] = path.name
    return metadata


def load_papers() -> list[dict]:
    papers, errors, seen_slugs = [], [], set()
    for path in sorted(CONTENT_DIR.glob("*.md")):
        try:
            paper = parse_note(path)
            if paper["slug"] in seen_slugs:
                raise ContentError(f"slug 重复：{paper['slug']}")
            seen_slugs.add(paper["slug"])
            papers.append(paper)
        except ContentError as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")
    if errors:
        raise ContentError("\n".join(errors))
    if not papers:
        raise ContentError("content/papers 中没有阅读记录")
    return sorted(papers, key=lambda item: (item["read_date"], item["title"]), reverse=True)


def write_data(path: Path, papers: list[dict]) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "papers": papers,
    }
    json_text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    path.write_text(f"window.PAPER_RECAP_DATA = {json_text};\n", encoding="utf-8")


def build(papers: list[dict]) -> None:
    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    (DIST_DIR / "assets").mkdir(parents=True)
    for filename in ("index.html", "404.html", ".nojekyll"):
        shutil.copy2(ROOT / filename, DIST_DIR / filename)
    for filename in ("styles.css", "app.js"):
        shutil.copy2(ROOT / "assets" / filename, DIST_DIR / "assets" / filename)
    write_data(DIST_DIR / "assets" / "papers.js", papers)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="仅校验内容，不创建 dist")
    args = parser.parse_args()
    try:
        papers = load_papers()
        if not args.check:
            build(papers)
    except ContentError as exc:
        print(f"内容校验失败：\n{exc}", file=sys.stderr)
        return 1
    action = "校验通过" if args.check else "构建完成"
    print(f"{action}：{len(papers)} 篇阅读记录")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
