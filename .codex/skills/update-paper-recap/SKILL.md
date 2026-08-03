---
name: update-paper-recap
description: Add, revise, validate, or publish paper-reading notes in the Paper Recap GitHub Pages site. Use when the user says “更新到 Paper Recap”, supplies a paper summary or Agent Q&A to archive, wants to edit an existing recap, or asks to refresh/deploy the personal paper-reading website.
---

# Update Paper Recap

Turn raw reading notes and Agent conversations into compact, durable recap entries. Optimize for the user's future recall, not for reproducing the paper.

## Workflow

1. Locate the repository root by finding `scripts/build_site.py` and `content/papers/`. If absent, clone `git@github.com:lawrence-cj/paper-recap.git` into a user-approved or obvious project location, then run `python3 scripts/install_skill.py` from that clone.
2. Treat every record as public. Stop before writing if the input appears to contain company-confidential, unpublished, credential, personal, or otherwise sensitive material; ask the user for a sanitized version.
3. Inspect `git status --short --branch`. Never mix unrelated changes into the recap commit. When clean, update from `origin/main` with `git pull --ff-only` before editing.
4. Identify the paper from the supplied title, URL, PDF, or context. Browse authoritative sources only when identity, authors, venue, year, or claims require verification. Never invent missing bibliographic facts.
5. Read `references/note-format.md` before creating or substantially restructuring an entry.
6. Search `content/papers/` by title, URL, and slug. Update the existing record instead of creating a duplicate.
7. Create or edit exactly one Markdown file in `content/papers/`. Name new files `YYYY-MM-DD-short-kebab-slug.md`, using the read date and a stable English slug.
8. Preserve the user's own judgment and disagreements. Condense Agent answers, remove conversational filler, and mark unknowable gaps as `待补充`; do not silently turn inference into fact.
9. Preserve important equations in LaTeX. Use `$...$` for inline math and `$$...$$` for display math; never replace formulas with vague prose or screenshots.
10. Run `python3 .codex/skills/update-paper-recap/scripts/validate_update.py`. Fix every reported error.
11. Treat “更新到 Paper Recap” as authorization to commit this recap file and push it for publication unless the user says local-only. Stage only the intended note, commit with `content: add <paper slug>` or `content: update <paper slug>`, run `git pull --rebase origin main`, revalidate, then `git push origin main`.
12. If the push loses a concurrent race, pull with rebase and retry once. Stop on a content conflict, preserve both versions, and ask the user to choose; never resolve conflicting judgments by guessing.

## Editing rules

- Keep `one_liner` specific and useful six months later; avoid generic praise.
- Keep two to five focused tags and reuse existing tag spellings where possible.
- Rate from 1–5 only when the user gave a rating or their judgment strongly supports one. Otherwise use `3` and state that it is a neutral placeholder in the handoff.
- Include concrete result numbers only when supplied or verified.
- Keep every formula needed to understand the method or result. Define symbols immediately around the equation and check delimiter balance.
- Put distilled Q&A under `我的提问`; keep only questions that changed understanding or future action.
- Make `下次只看这些` one to three items. This is the fastest recap path.
- Do not edit generated `dist/` or `assets/papers.js`; the build script owns them.

## User-facing intake

Accept messy notes. The user does not need to fill a template. The smallest useful input is:

```text
论文：标题或链接
我的理解：随手写的要点
重要问答：与 Agent 对话中最有价值的 Q&A
我的判断：是否可信、是否值得复现、与当前工作的关系
```

If essential meaning is missing, make a conservative draft with `待补充` instead of blocking on cosmetic details.
