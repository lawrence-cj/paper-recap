# Paper note format

Use UTF-8 Markdown with this exact top-level structure. Copy `content/TEMPLATE.md` when starting a new entry.

## Frontmatter

Required fields:

- `title`: official paper title.
- `paper_url`: canonical paper or arXiv URL; use an empty quoted string if unavailable.
- `authors`: compact author display, such as `First Author et al.`.
- `venue`: conference, journal, arXiv, or `待补充`.
- `published`: four-digit year as quoted text, or `待补充`.
- `read_date`: local date in `YYYY-MM-DD`.
- `status`: normally `待读`, `略读`, `已读`, `已精读`, or `复现中`.
- `rating`: integer from 1 to 5.
- `tags`: inline quoted list with two to five stable tags.
- `one_liner`: one standalone sentence that restores the paper's core memory.

Keep all string values quoted. Keep the frontmatter parser-compatible: one `key: value` per line and no multiline values.

## Required sections

Use these headings verbatim and in this order:

1. `## 研究问题` — problem, importance, and relevant prior limitation.
2. `## 核心方法` — mechanism and true novelty; bullets are preferred.
3. `## 关键发现` — decisive evidence and verified numbers.
4. `## 我的提问` — `### Q1：...` followed by distilled answers.
5. `## 局限与疑问` — reported limitations and the user's unresolved concerns.
6. `## 我的判断` — trust, usefulness, reproduction value, and relation to current work.
7. `## 下次只看这些` — one to three numbered memory anchors or actions.

Avoid a redundant H1 title in the body. Do not paste abstracts or long chat transcripts.

## Mathematics

Preserve equations as editable LaTeX, not images:

```markdown
The noise prediction at timestep $t$ is $\epsilon_\theta(x_t, t)$.

$$
\mathcal{L}_{\mathrm{simple}}
= \mathbb{E}_{t,x_0,\epsilon}
\left[\lVert \epsilon - \epsilon_\theta(x_t,t) \rVert_2^2\right].
$$
```

- Use `$...$` for short inline expressions.
- Put `$$` on their own lines around important display equations.
- Keep multiline LaTeX inside one `$$...$$` block.
- Define symbols in nearby prose and state why the equation matters.
- Prefer standard KaTeX-supported commands. If a paper uses custom macros, expand them into standard LaTeX.
- Do not place math delimiters inside backticks; code spans intentionally skip formula rendering.
