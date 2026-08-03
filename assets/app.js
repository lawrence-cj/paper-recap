(() => {
  const data = window.PAPER_RECAP_DATA || { generated_at: "", papers: [] };
  const state = { query: "", tag: "全部", sort: "newest" };
  let lockedScrollY = 0;
  const elements = {
    grid: document.querySelector("#paper-grid"),
    tags: document.querySelector("#tag-list"),
    search: document.querySelector("#search-input"),
    sort: document.querySelector("#sort-select"),
    count: document.querySelector("#result-count"),
    empty: document.querySelector("#empty-state"),
    dialog: document.querySelector("#paper-dialog"),
    dialogContent: document.querySelector("#dialog-content"),
    imageDialog: document.querySelector("#image-dialog"),
    imageDialogImage: document.querySelector("#image-dialog-image"),
    imageDialogCaption: document.querySelector("#image-dialog-caption"),
  };

  const escapeHtml = (value = "") => String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");

  function inlineMarkdown(text) {
    return escapeHtml(text)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.+?)\*/g, "<em>$1</em>")
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  }

  function renderMarkdown(markdown = "") {
    const lines = markdown.replace(/\r/g, "").split("\n");
    const output = [];
    let listType = null;
    let displayMath = null;
    const closeList = () => { if (listType) output.push(`</${listType}>`); listType = null; };

    for (const raw of lines) {
      const line = raw.trim();
      if (displayMath !== null) {
        displayMath.push(raw);
        if (line.endsWith("$$")) {
          output.push(`<div class="math-block">${escapeHtml(displayMath.join("\n"))}</div>`);
          displayMath = null;
        }
        continue;
      }
      if (line.startsWith("$$")) {
        closeList();
        if (line.length > 2 && line.endsWith("$$")) output.push(`<div class="math-block">${escapeHtml(line)}</div>`);
        else displayMath = [raw];
        continue;
      }
      if (!line) { closeList(); continue; }
      const image = line.match(/^!\[([^\]]+)\]\((media\/[^\s)"']+\.(?:png|jpe?g|webp))(?:\s+"([^"]+)")?\)$/i);
      if (image) {
        closeList();
        const altText = image[1];
        const imagePath = image[2];
        const caption = image[3] || "";
        output.push(`<figure class="paper-figure">
          <button class="paper-image-button" type="button" data-image-src="${escapeHtml(imagePath)}" data-image-alt="${escapeHtml(altText)}" data-image-caption="${escapeHtml(caption)}" aria-label="放大查看：${escapeHtml(altText)}">
            <img src="${escapeHtml(imagePath)}" alt="${escapeHtml(altText)}" loading="lazy" decoding="async" />
          </button>
          ${caption ? `<figcaption>${inlineMarkdown(caption)}</figcaption>` : ""}
        </figure>`);
        continue;
      }
      const heading = line.match(/^(##|###)\s+(.+)$/);
      if (heading) {
        closeList();
        const level = heading[1].length;
        output.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
        continue;
      }
      const unordered = line.match(/^[-*]\s+(.+)$/);
      if (unordered) {
        if (listType !== "ul") { closeList(); listType = "ul"; output.push("<ul>"); }
        output.push(`<li>${inlineMarkdown(unordered[1])}</li>`);
        continue;
      }
      const ordered = line.match(/^\d+[.)]\s+(.+)$/);
      if (ordered) {
        if (listType !== "ol") { closeList(); listType = "ol"; output.push("<ol>"); }
        output.push(`<li>${inlineMarkdown(ordered[1])}</li>`);
        continue;
      }
      closeList();
      if (line.startsWith("> ")) output.push(`<blockquote>${inlineMarkdown(line.slice(2))}</blockquote>`);
      else output.push(`<p>${inlineMarkdown(line)}</p>`);
    }
    if (displayMath !== null) output.push(`<pre class="math-error">${escapeHtml(displayMath.join("\n"))}</pre>`);
    closeList();
    return output.join("");
  }

  function renderMath(container) {
    if (typeof window.renderMathInElement !== "function") return;
    window.renderMathInElement(container, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false },
        { left: "\\(", right: "\\)", display: false },
        { left: "\\[", right: "\\]", display: true },
        { left: "\\begin{equation}", right: "\\end{equation}", display: true },
        { left: "\\begin{align}", right: "\\end{align}", display: true },
        { left: "\\begin{gather}", right: "\\end{gather}", display: true },
      ],
      throwOnError: false,
      strict: "warn",
    });
  }

  function formatDate(value) {
    if (!value) return "日期未知";
    return new Intl.DateTimeFormat("zh-CN", { year: "numeric", month: "short", day: "numeric" })
      .format(new Date(`${value}T00:00:00`));
  }

  function syncDialogScrollLock() {
    const root = document.documentElement;
    const body = document.body;
    const shouldLock = elements.dialog.open || elements.imageDialog.open;
    const isLocked = root.classList.contains("dialog-open");

    if (shouldLock && !isLocked) {
      lockedScrollY = window.scrollY;
      root.classList.add("dialog-open");
      body.style.top = `-${lockedScrollY}px`;
      return;
    }

    if (!shouldLock && isLocked) {
      root.classList.remove("dialog-open");
      body.style.removeProperty("top");
      window.scrollTo(0, lockedScrollY);
    }
  }

  function allTags() {
    const counts = new Map();
    data.papers.forEach((paper) => paper.tags.forEach((tag) => counts.set(tag, (counts.get(tag) || 0) + 1)));
    return [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  }

  function renderTags() {
    const tags = [["全部", data.papers.length], ...allTags()];
    elements.tags.innerHTML = tags.map(([tag, count]) => `
      <button class="tag-button ${state.tag === tag ? "active" : ""}" type="button" data-tag="${escapeHtml(tag)}">
        ${escapeHtml(tag)} · ${count}
      </button>`).join("");
    elements.tags.querySelectorAll("button").forEach((button) => button.addEventListener("click", () => {
      state.tag = button.dataset.tag;
      renderTags();
      renderPapers();
    }));
  }

  function filteredPapers() {
    const needle = state.query.trim().toLocaleLowerCase();
    const filtered = data.papers.filter((paper) => {
      const tagMatch = state.tag === "全部" || paper.tags.includes(state.tag);
      const haystack = [paper.title, paper.authors, paper.venue, paper.one_liner, paper.body, ...paper.tags].join(" ").toLocaleLowerCase();
      return tagMatch && (!needle || haystack.includes(needle));
    });
    return filtered.sort((a, b) => {
      if (state.sort === "rating") return b.rating - a.rating || b.read_date.localeCompare(a.read_date);
      if (state.sort === "title") return a.title.localeCompare(b.title);
      return b.read_date.localeCompare(a.read_date);
    });
  }

  function paperCard(paper) {
    const tags = paper.tags.slice(0, 3).map((tag) => `<span>#${escapeHtml(tag)}</span>`).join("");
    const rating = "●".repeat(paper.rating) + "○".repeat(5 - paper.rating);
    return `<article class="paper-card">
      <button class="card-button" type="button" data-slug="${escapeHtml(paper.slug)}" aria-label="打开《${escapeHtml(paper.title)}》详情">
        <div class="card-top"><span class="status">${escapeHtml(paper.status)}</span><span>${formatDate(paper.read_date)}</span></div>
        <h3>${escapeHtml(paper.title)}</h3>
        <p class="authors">${escapeHtml(paper.authors)}${paper.venue ? ` · ${escapeHtml(paper.venue)}` : ""}</p>
        <p class="one-liner">${escapeHtml(paper.one_liner)}</p>
        <div class="card-bottom"><div><div class="card-tags">${tags}</div><div class="rating" aria-label="评分 ${paper.rating} / 5">${rating}</div></div><span class="arrow" aria-hidden="true">↗</span></div>
      </button>
    </article>`;
  }

  function renderPapers() {
    const papers = filteredPapers();
    elements.grid.innerHTML = papers.map(paperCard).join("");
    elements.count.textContent = `显示 ${papers.length} / ${data.papers.length} 篇记录`;
    elements.empty.hidden = papers.length !== 0;
    elements.grid.hidden = papers.length === 0;
    elements.grid.querySelectorAll("button").forEach((button) => button.addEventListener("click", () => openPaper(button.dataset.slug)));
  }

  function openPaper(slug, updateHash = true) {
    const paper = data.papers.find((item) => item.slug === slug);
    if (!paper) return;
    elements.dialogContent.innerHTML = `
      <p class="detail-kicker">${escapeHtml(paper.status)} · ${formatDate(paper.read_date)}</p>
      <h2 id="dialog-title">${escapeHtml(paper.title)}</h2>
      <div class="detail-meta"><span>${escapeHtml(paper.authors)}</span><span>${escapeHtml(paper.venue)}</span><span>${escapeHtml(paper.published)}</span><span>评分 ${paper.rating}/5</span></div>
      <p class="detail-summary">${escapeHtml(paper.one_liner)}</p>
      <div class="detail-body">${renderMarkdown(paper.body)}</div>
      ${paper.paper_url ? `<a class="paper-link" href="${escapeHtml(paper.paper_url)}" target="_blank" rel="noopener">查看原论文 ↗</a>` : ""}`;
    renderMath(elements.dialogContent);
    if (!elements.dialog.open) elements.dialog.showModal();
    syncDialogScrollLock();
    if (updateHash) history.pushState({ slug }, "", `#paper=${encodeURIComponent(slug)}`);
  }

  function closePaper(updateHash = true) {
    if (elements.dialog.open) elements.dialog.close();
    if (updateHash && location.hash.startsWith("#paper=")) history.pushState({}, "", location.pathname + location.search);
  }

  function openImage(button) {
    elements.imageDialogImage.src = button.dataset.imageSrc;
    elements.imageDialogImage.alt = button.dataset.imageAlt || "论文图片";
    elements.imageDialogCaption.textContent = button.dataset.imageCaption || "";
    elements.imageDialogCaption.hidden = !button.dataset.imageCaption;
    elements.imageDialog.showModal();
    syncDialogScrollLock();
  }

  function openFromHash() {
    const match = location.hash.match(/^#paper=(.+)$/);
    if (match) openPaper(decodeURIComponent(match[1]), false);
    else closePaper(false);
  }

  function initStats() {
    const uniqueDays = new Set(data.papers.map((paper) => paper.read_date)).size;
    document.querySelector("#stat-papers").textContent = data.papers.length;
    document.querySelector("#stat-tags").textContent = allTags().length;
    document.querySelector("#stat-days").textContent = uniqueDays;
    document.querySelector("#last-updated").textContent = data.generated_at ? `更新于 ${formatDate(data.generated_at.slice(0, 10))}` : "";
  }

  function initTheme() {
    const saved = localStorage.getItem("paper-recap-theme");
    const preferred = matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    document.documentElement.dataset.theme = saved || preferred;
    document.querySelector("#theme-toggle").addEventListener("click", () => {
      const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      localStorage.setItem("paper-recap-theme", next);
    });
  }

  elements.search.addEventListener("input", (event) => { state.query = event.target.value; renderPapers(); });
  elements.sort.addEventListener("change", (event) => { state.sort = event.target.value; renderPapers(); });
  document.querySelector("#clear-filters").addEventListener("click", () => {
    state.query = ""; state.tag = "全部"; elements.search.value = ""; renderTags(); renderPapers();
  });
  document.querySelector("#dialog-close").addEventListener("click", () => closePaper());
  elements.dialog.addEventListener("click", (event) => { if (event.target === elements.dialog) closePaper(); });
  elements.dialog.addEventListener("close", () => {
    syncDialogScrollLock();
    if (location.hash.startsWith("#paper=")) closePaper();
  });
  elements.dialogContent.addEventListener("click", (event) => {
    const button = event.target.closest(".paper-image-button");
    if (button) openImage(button);
  });
  document.querySelector("#image-dialog-close").addEventListener("click", () => elements.imageDialog.close());
  elements.imageDialog.addEventListener("close", () => {
    elements.imageDialogImage.removeAttribute("src");
    syncDialogScrollLock();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "/" && document.activeElement !== elements.search) { event.preventDefault(); elements.search.focus(); }
  });
  addEventListener("popstate", openFromHash);

  initTheme(); initStats(); renderTags(); renderPapers(); openFromHash();
})();
