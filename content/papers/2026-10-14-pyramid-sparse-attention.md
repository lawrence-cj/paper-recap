---
title: "Block Sparse Attention with Log-Linear Complexity"
paper_url: "https://arxiv.org/abs/2609.31093"
authors: "Bohao Tang et al."
venue: "arXiv"
published: "2026"
read_date: "2026-10-14"
read_at: "2026-10-14T12:17:31+08:00"
status: "已精读"
tags: ["Language Models", "Efficient Attention", "Inference Acceleration", "Training & Optimization"]
search_terms: ["PISA", "Pyramid Sparse Attention", "ByteDance Seed", "LogSumExp", "GQA", "Exact Top-K", "Branch and Bound"]
one_liner: "PISA 用均值 Key 金字塔和逐层 LSE Top-K 降低选块开销，但摘要近似与固定预算剪枝都不保证找回全局精确 Top-K。"
---

## 研究问题

这篇 PISA 指 Pyramid Sparse Attention，由上海交大、上海创智学院与 ByteDance Seed 团队提出，arXiv 提交于 2026-09-25。它研究长上下文语言模型，与本站另一篇 [Piecewise Sparse Attention](https://junsongc.top/paper-recap/#paper=pisa)（arXiv:2602.01077，视觉扩散模型）是不同工作。

Block sparse attention 先选 Key 块，再在选中块的原始 K/V 上计算 attention。设序列长度为 $N$、head 维度为 $d$、叶块大小为 $C$、每个 query 选 $K$ 个块：全量扫描 $N/C$ 个块摘要的选块成本为 $O(N^2d/C)$；选中块上的 attention 成本为 $O(NKCd)$。固定 $K,C$ 后，后者已是线性的，前者仍是平方级。

PISA 要减少的是逐 query 扫描全部候选块的成本。原文依据见 [论文 §3](https://arxiv.org/pdf/2609.31093)。以下 Q&A 中的数值例子、排序反例与精确搜索扩展是本轮讨论推导，不是论文新增实验。

## 核心方法

![PISA 的 Key 摘要树、逐层 LSE 选块和原始 K/V attention 计算流程](media/pyramid-sparse-attention/method-overview.webp "本文原创方法示意：均值摘要建树、逐层筛选与最终 token attention。概念来源：PISA §3.2–3.3；独立绘制，未复制原论文图；© 2026 Junsong Chen，按本站 COPYRIGHT 保留权利。为简洁省略强制块与 GQA 分数汇总。")

- **先建树，所有 query 共用。** 叶层包含 $C$ 个原始 Key；每个节点存一个均值摘要，再将相邻节点二合一，分支因子 $g=2$。完整等长块的摘要等于该区域所有 Key 的均值；尾部不满块时按有效子节点递归池化。
- **query 仍逐 token 处理。** 从根向叶层搜索，每层保留至多 $K$ 个节点，再展开成至多 $2K$ 个子候选。候选数不超过 $K$ 时直接全留，跳过评分。
- **中层与叶层使用不同输入。** 中间节点对两个子节点的摘要点积做 LSE；叶节点对块内原始 Key 点积做 LSE。中层不会读取整棵子树的原始 Key。
- **最终使用原始 K/V。** 选中叶块后，各 query head 在最多 $KC$ 个原始 token 上重新计算 softmax attention；均值摘要不参与 V 的加权求和。

设 $q_t$ 为位置 $t$ 的 query，$\bar k_r$ 为子节点 $r$ 的摘要。中间候选节点 $B$ 的分数为：

$$
s_t(B)=\log\sum_{r\in\operatorname{children}(B)}
\exp\left(\frac{q_t^\top\bar k_r}{\sqrt d}\right).
$$

叶块 $B$ 的精确分数为：

$$
s_t(B)=\log\sum_{j\in B}
\exp\left(\frac{q_t^\top k_j}{\sqrt d}\right).
$$

令 $T_t$ 为选中叶块的有效 token 集合。最终 attention 使用一个覆盖所有选中 token 的共同分母：

$$
o_t=
\frac{\sum_{j\in T_t}\exp(q_t^\top k_j/\sqrt d)\,v_j}
{\sum_{j\in T_t}\exp(q_t^\top k_j/\sqrt d)}.
$$

固定 $K,C,d$ 和 head 数量后，训练/prefill 的选择成本为 $O(N\log N)$，decode 单步平均成本为 $O(\log N)$。若保持固定 attention 密度，让 $KC$ 随 $N$ 增长，不能继续沿用这个常数预算下的复杂度结论。

## 关键发现

以下数字来自 [原文表 3、表 4、表 7](https://arxiv.org/pdf/2609.31093)，只适用于论文报告的设置。

- 模型规模为 418M、1.47B、2.67B：先在 4K context 上预训练 100B tokens，再在 16K 上继续预训练 10B tokens；块大小 $C=64$，预算从 $K=8$ 增至 $K=32$。
- **选块速度不是整模型速度。** Prefill benchmark 包含摘要构建与全部选块阶段，不包含选中块上的 attention；4K–16K 时 BSA 更快，长序列时 PISA Q4 更快。

| 序列长度 | BSA 选块延迟 | PISA Q4 选块延迟 | 相对 BSA 加速 |
|---|---:|---:|---:|
| 64K | 19.14 ms | 6.70 ms | 2.86 倍 |
| 128K | 76.68 ms | 14.45 ms | 5.31 倍 |
| 256K | 312.96 ms | 31.44 ms | 9.95 倍 |

- **质量只评到 16K。** 2.67B 模型在四类 RULER 检索任务、1K–16K 长度上的平均准确率：PISA 62.80%，BSA 54.99%，Full Attention 66.24%；PISA-2 为 62.92%，因此 PISA 并非所有设置都最好。256K 只是随机 BF16 张量的选块速度测试。
- **99.46% 不等于保留了全部 attention 的 99.46%。** 固定 dense checkpoint 的选块诊断中，PISA Recall@8 为 90.95%，实际捕获总 attention mass 为 86.47%；99.46% 是相对相同预算 reference Top-8 捕获量的比值。Recall 还包含三个强制块的重合。该诊断使用 418M Full-Attention checkpoint、24 层和 100 个 FDA prompts，实际长度为 618–2001 tokens，不是长序列质量验证。
- 继续预训练后的 containment 平均分并非全面改善：2.67B 的 PISA 为 66.40%，BSA 为 68.15%（表 6 的三个任务平均）。不能把 RULER 的改善扩展为所有检索任务都提升。

## 我的提问

### Q1：一个 query 怎样逐层选 Top-K，每一步算多少东西？

下面是本轮计算示例：$N=65536,C=64,K=8,g=2,d=64$。取序列末尾 query，先忽略强制块策略；单 head、完整二叉树有 1024 个叶块。

| 整层节点数 | 每个节点覆盖 token 数 | 此 query 实际候选数 | 操作 |
|---:|---:|---:|---|
| 1、2、4、8 | 65536 至 8192 | 1、2、4、8 | 全留，无评分 |
| 16 | 4096 | 16 | 子摘要 LSE，选 8 |
| 32 | 2048 | 16 | 子摘要 LSE，选 8 |
| 64 | 1024 | 16 | 子摘要 LSE，选 8 |
| 128 | 512 | 16 | 子摘要 LSE，选 8 |
| 256 | 256 | 16 | 子摘要 LSE，选 8 |
| 512 | 128 | 16 | 子摘要 LSE，选 8 |
| 1024，叶层 | 64 | 16 | 原始 Key LSE，最终选 8 |

一次中层筛选：16 个候选各读取两个子摘要，计算 32 个长度为 $d$ 的点积；将结果组织成 $[16,2]$，沿长度为 2 的轴做 16 次 LSE，再从 16 个标量分数中选 8 个。展开后仍只有 16 个候选。

一次叶层筛选：16 个候选各读 64 个原始 Key，计算 $16\times64=1024$ 个点积；将结果组织成 $[16,64]$，做 16 次 LSE，再选出 8 块。

计算 LSE 时，可用最大值移位避免指数溢出。令 $m=\max_j z_j$：

$$
\operatorname{LSE}(z_1,\ldots,z_n)
=m+\log\sum_{j=1}^{n}\exp(z_j-m).
$$

最终 attention 对选中 8 块的 512 个 token 做 QK、共同 softmax 和 V 加权求和。按论文 selector 只传出块索引的流程，attention 会重新计算选中 token 的 QK；一个叶块 LSE 标量不能提供块内每个 token 的权重。

| 阶段 | 单 query、单 head 的 QK 点积数 |
|---|---:|
| 6 次中层路由 | $6\times16\times2=192$ |
| 叶块评分 | $16\times64=1024$ |
| 最终 attention | $8\times64=512$ |
| 合计 | 1728 |

Routing 共做 112 次 LSE、7 次 Top-K。最终还要做 softmax 和 512 个 V 向量的加权求和。实际 causal query 的可见块更少，强制路径免评分，上表不是所有 query 的固定成本，也不是运行时间测量。

### Q2：哪些计算并行，GQA 与 GPU kernel 怎么安排？

同一 query 的下一层候选依赖上一层 Top-K，所以树层之间顺序推进；同层的不同候选、不同 heads 的点积可并行。不同 query 可以独立走树，不必每层都进行全局同步。

论文采用 32 query heads / 2 KV heads，每个 GQA group 有 16 个 query heads。先逐 head 算块 LSE，再相加得到共享选块分数：

$$
u_{t,h}(B)=\sum_{h'\in H(h)}s_{t,h'}(B).
$$

其中 $H(h)$ 为使用 KV head $h$ 的 query heads。共享的是块索引，最终各 head 独立计算 attention 权重和输出。跨 head 的 LSE 相加，也不等于跨 head 归一化 attention mass 的平均。

| 阶段 | 一个 query 位置、一个 KV head 的逻辑 logits 形状 | 点积结果数 |
|---|---|---:|
| 一次中层评分 | $[16,16,2]$：heads、候选、子摘要 | 512 |
| 叶层评分 | $[16,16,64]$：heads、候选、原始 Keys | 16384 |
| 最终 attention | $[16,512]$：heads、选中 tokens | 8192 |

Prefill 中层 kernel 按 query 位置与 KV head 并行；batch 为 1、64K 位置、2 个 KV heads 时，有 $65536\times2=131072$ 个逻辑 program 任务，由 GPU 分批调度。每个任务加载该组 queries，并顺序完成中层选择。

叶层则将需要给同一个叶块评分的 query 位置聚成组，每个 tile 最多 4 个位置，这些位置不必相邻。对一个叶块，完整 tile 的矩阵形状为：

$$
Q[(4\times16),64]\;K_B^\top[64,64]
\longrightarrow Z[64,64].
$$

它产生 4096 个 logits，沿 Key 轴 LSE 后得到 64 个 head 分数，再跨 heads 求和成 4 个 query/block 分数。这一个 tile 只评分一个叶块，不是同时完成这些 query 的全部候选块。

Decode 每条序列只有一个新 query，论文把中层与叶层选择融合到一个 kernel，避免多次启动。上述张量形状和 program 数不是同时驻留的 threads/warps；真实 GPU 并行度需要代码和 profiling。

### Q3：PISA 选出的 Top-K，肯定等于全量直接选 Top-K 吗？

不保证。这里“全量直接选”指所有叶块按相同的精确 LSE 规则排名。均值摘要可能掩盖高分 token；即使将父块评分理想化为精确总贡献，固定预算剪枝也会漏掉全局最强叶块。

取 $K=1$，用叶块内部指数分数之和表示未归一化贡献；取对数不改变排名。这是单 head、忽略强制保留块策略的抽象例子，用来说明固定预算树剪枝的风险：

| 父块 | 两个叶块的贡献 | 父块总贡献 |
|---|---|---:|
| A | 6、6 | 12 |
| B | 10、1 | 11 |

按父块总贡献先保留 A，再从 A 中选出贡献为 6 的叶块；全量直接选则应选 B 中贡献为 10 的叶块。因此精确叶层评分只保证在幸存候选中选准，不能恢复已剪掉的块。

论文附录 A.1 的分析只说明，等大小子块下 normalized child-mean LSE 比均值分数在数值上更接近真实 normalized LSE；不保证跨块排序或全局 Top-K 恢复。

### Q4：把对 K 取均值，改成对 K 的每一维取 LSE，会更准吗？

这里指的是沿 token 维做元素级池化，仍得到一个与 query 无关的 $d$ 维向量：

$$
\tilde k_{B,r}=\log\sum_{j\in B}\exp(k_{j,r}),
\qquad r=1,\ldots,d.
$$

这个摘要可以提前计算、缓存，存储量与路由点积的复杂度阶数不变。问题在于，它没有普遍更准确的保证：

$$
q^\top\operatorname{mean}(K)
=\operatorname{mean}(q^\top K),
\qquad
q^\top\operatorname{LSE}_{\mathrm{token}}(K)
\ne\operatorname{LSE}_{\mathrm{token}}(q^\top K).
$$

第二式左侧的 LSE 对 K 的各坐标分别求值，右侧才是 query 打分后的块 LSE。先对 Key 坐标做 LSE 会偏向较大的坐标值，而负的 query 分量可能更关注较小的 Key 值。

本轮反例取 $d=1,q=-1$，两个子块 A 为 $[0,2]$，B 为 $[1.2,1.2]$：

| 子块评分方式 | A | B | 排名更高 |
|---|---:|---:|---|
| K 均值后点积 | -1.000 | -1.200 | A |
| K 元素级 LSE 后点积 | -2.127 | -1.893 | B |
| 精确 LSE(qK) | 0.127 | -0.507 | A |

这个例子中均值排序正确，元素级 LSE 反而排错。减去 $\log C$ 做归一化也不改变排名；如果两个父块各包含两个相同的上述子块，PISA 再对子摘要分数做 LSE 只是共同加上 $\log2$，错误排名仍存在。

反例只否定“必然更准”，不说明训练后的模型一定更差。论文 PISA-1/2 消融比较的是打分 logits 的均值、均值加半个方差与 LSE，没有验证这种元素级 K 池化替换。

### Q5：取均值，是为了让重要块在 QK 打分后仍排在前面吗？

这是希望达到的检索效果，但均值不能保证排名。它精确保留的是平均 logit，并允许提前共享摘要；平均分高不等于块内最高分高，也不等于 softmax 下贡献最大。少数高分 token 可以被大量低分 token 稀释。

需要分别记住两种误差：摘要丢失块内信息，以及树搜索提前剪枝。改进摘要可能减少前者，不能自动消除后者。

### Q6：有没有既高效，又保证全局精确 Top-K 的办法？

可以用严格上界做 branch-and-bound 搜索：只有证明区域内所有叶块都不可能进入当前 Top-K，才剪掉该区域。它保证结果准确，在有利输入上可以少算很多，但最坏仍需要全量扫描，不能固定每层只保留 $K$ 个节点。

以下是本轮从 [Ram 与 Gray 的精确内积检索球形界](https://arxiv.org/pdf/1202.6101) 推导的叶块 LSE 上界，不是 PISA 的原方法，也没有 GPU 速度实验。

设节点 $T$ 缓存中心 $c_T$ 与半径 $r_T$，满足所有区域内的 Keys 都有 $\lVert k_j-c_T\rVert_2\le r_T$。节点下任意叶块 $B$ 最多有 $C$ 个有效 Keys，由 Cauchy–Schwarz 得：

$$
S_B(q)=\log\sum_{j\in B}\exp(q^\top k_j/\sqrt d)
\le
\log C+\frac{q^\top c_T+\lVert q\rVert_2r_T}{\sqrt d}
=U_T(q).
$$

1. 精确算一批叶块，维护当前最好的 $K$ 个；第 $K$ 名分数为 $\tau$，不足 $K$ 个时设 $\tau=-\infty$。
2. 若区域上界 $U_T<\tau$，整棵子树可安全剪掉；否则继续展开，到叶层计算精确 LSE 并更新候选。
3. 所有未搜索区域的上界均低于 $\tau$ 时，结果就是全局精确 Top-K。严格不等式避免跳过可能并列的候选，最终使用一致的并列排序规则。

若跨 GQA heads 的目标分数是 LSE 相加，相应上界也可以逐 head 相加。实现中的半径与上界需保守处理浮点误差，不能因舍入低估；数学证明不自动等于 BF16/FP32 代码的严格认证。

可以先用 PISA 快速找到候选，提高 $\tau$，再用上界树检查遗漏区域。上界松或分数接近时会退化为检查全部 Keys。另一条精确路线是全量分块扫描、在线维护 Top-K：它减少中间矩阵存储，但保留全量 QK 算术量。

## 局限与疑问

- 高效近似选块不等于精确全局 Top-K；增大预算可以降低漏选风险，不能提供普遍保证。
- 检索质量只验证到 16K；长序列的随机张量速度测试不证明 256K 模型能力或端到端加速。
- 训练时固定选块索引，梯度通过选中 K/V 上的 attention，不经过离散 Top-K 决策。
- Causal prefill 强制保留首块、前块、当前块及其祖先。当前路径摘要可能含未来 Keys，但不用于排名；最终 attention 另外 mask 块内未来 token。
- 原始 KV cache 仍需保留；均值树是额外缓存。Decode 的对数复杂度是平均界，包含偶发扩容。
- BSA 与 PISA 的对比同时改变树结构、评分和跨 head 聚合；不能把质量收益全部归因于分层搜索。相同树下 PISA-1/2 的对比才隔离了评分函数。
- 元素级 LSE Key 摘要、上界认证搜索的实际收益均待实验；需要计入摘要/索引维护、动态搜索与 GPU 内存访问成本。
- 原论文 Figure 1 有助于理解，但 arXiv 仅标注 non-exclusive distribution license，未确认本站可复制图像的许可；因此本条使用独立绘制的原创示意，并链接原文。

## 我的判断

本轮关注的是选块步骤究竟保留了什么信息，以及如何解释速度与精度。PISA 的价值是降低全量扫块的成本；精确叶层 LSE 不应被理解为全局选块精确。判断它是否适合使用，需要同时看选块召回、模型质量和完整运行时间。

如果继续研究精确选择，值得比较“全量融合扫描”与“PISA 候选加上界检查”。这只是本轮讨论提出的实验方向，尚无证据证明认证树在 attention 场景下更快。替换成元素级 LSE Key 摘要也应作为独立对照实验，而非默认改进。

## 下次只看这些

1. 中层 LSE 读子摘要，叶层 LSE 读原始 Keys；最终对选中原始 K/V 做 token-level softmax。
2. 均值保留平均 QK，不保留 Top-K 排名；摘要误差与固定预算剪枝都可能漏选。
3. 精确搜索需要可靠上界与可变搜索量；论文最高 9.95 倍仅是选块加速，质量评测只到 16K。
