---
title: "CachedSearch: Training-Free Cached Exploration for Test-Time Search in Video Diffusion"
paper_url: "https://arxiv.org/abs/2607.23159"
authors: "Shreshth Saini et al."
venue: "arXiv"
published: "2026"
read_date: "2026-08-03"
status: "已精读"
rating: 3
tags: ["Video Generation", "Diffusion Models", "Test-Time Search", "Inference Acceleration"]
one_liner: "用带缓存的廉价 rollout 只负责保持候选排序，再以胜出 seed 完整重生成，将视频 best-of-N 搜索改造成 explore-cheap、commit-full。"
---

## 研究问题

Verifier-guided test-time search 会为同一 prompt 生成 $N$ 个不同 seed 的完整视频，打分后只交付其中一个；其余 $N-1$ 个候选虽然被丢弃，却都支付了完整 DiT denoising 成本。已有 training-free caching 能加速单条视频，但评价重点通常是 cached/full 同 seed 视频的像素或特征相似度，没有回答一个更贴近搜索的问题：**有损缓存是否仍能保持同一 prompt 内候选的相对排序？**

CachedSearch 的目标不是让廉价候选成为最终输出，而是让它足以预测哪个 seed 值得完整计算，从而把探索成本和交付质量解耦。

## 核心方法

- **Best-of-$N$ 基线**：对 prompt $c$ 和 seeds $s_1,\ldots,s_N$ 完整生成 $y_i=G(c,s_i)$，用 verifier $V(y_i,c)$ 选出最高分候选。完整搜索成本为 $NC_f$，其中 $C_f$ 是一次完整 rollout 的成本。
- **Adaptive transformation-vector caching**：缓存最近一次完整 DiT 调用的 transformation vector

$$
\Delta=v_\theta(x_{\mathrm{ref}})-x_{\mathrm{ref}},
$$

  跳过当前调用时用 $\hat v_\theta(x)=x+\Delta$ 近似输出。累计输入漂移

$$
a\leftarrow a+
\frac{\lVert x-x_{\mathrm{ref}}\rVert_F}
{\lVert x_{\mathrm{ref}}\rVert_F}
$$

  超过阈值 $\tau$ 后才重新执行 DiT 并刷新缓存。50-step 默认配置的前后各 5 steps 始终完整计算；CFG 的 conditional/unconditional branches 分别维护缓存状态。
- **Explore cheap**：使用 cached generator $G_\tau$ 生成并评分所有候选，只用 cached score 决定胜出 seed：

$$
\hat i=\operatorname*{arg\,max}_{i}
V\!\left(G_\tau(c,s_i),c\right).
$$

- **Commit full**：默认不交付 cached winner，而是关闭缓存，以同一个 seed 完整重生成：

$$
y_{\mathrm{final}}=G(c,s_{\hat i}).
$$

  生成过程对 prompt、seed 和配置确定，因此 recommit 得到该 seed 对应的标准 full-compute 视频；缓存只可能影响选哪个 seed，不影响最终视频的单样本质量。论文也讨论直接交付 cached winner 的 keep-draft 模式，但不建议将其作为默认方案。
- **成本与适用区间**：CachedSearch-recommit 的成本为

$$
C_{\mathrm{commit}}=NC_c+C_f,
$$

  其中 $C_c$ 是一次 cached rollout 的成本。它比完整 best-of-$N$ 更便宜的条件是

$$
N>\frac{1}{1-C_c/C_f}.
$$

  默认实验中 $C_c/C_f\approx0.51$，因此 $N=2$ 基本不省钱，从 $N\geq4$ 才有明确收益。

## 关键发现

- 在 Wan2.1-T2V-1.3B、50 steps、GH200 上，单条视频从 **68.3 s 降至 34.7 s**，约 $1.97\times$ 加速。VBench 中 cached/full 的 prompt 内候选排序达到 median Spearman $\rho=0.905$，top-1 seed agreement 为 **72%**；VBench-2.0 上仍有 $\rho=0.881$、69% top-1 agreement 和 89.3% mean per-prompt gain capture。
- 在 $N=8$ 时，recommit 用约 **346 s**，而完整 best-of-8 用约 **547 s**；它以 **63.3% 成本保留 94.7% 的总体搜索增益**。这里 94.7% 是 ratio-of-means；逐 prompt 先算比例再平均的 capture 约为 90.1%，两者不能混用。
- 在接近完整 best-of-4 的预算下，它可以搜索 8 个候选，reward gain 高 **38%**。扩到 $N=16/32$ 后，capture 接近 95%，而相对成本进一步下降，说明 recommit 的一次完整生成开销会被更宽的搜索摊薄。
- 排序错误具有一定 self-limiting 特性：缓存最容易打乱近似并列的候选，而这些错误的 regret 通常较小。VBench 上 72% prompts 的 regret 为 0，mean regret 为 0.056，远低于随机选择的 0.79；score spread 与 ranking fidelity 的相关系数为 $+0.31$。
- 在相同约 $2\times$ exploration speedup 下，直接将 denoising 从 50 steps 截断到 25 steps 只保留 **72.6%** 搜索增益，而 caching 保留 **90.1%**，且截断的 regret 约高 6 倍。原因是少步采样来自不同分布，而 caching 仍是对相同 seed trajectory 的近似扰动。
- 方法在 Wan、CogVideoX、Hunyuan 和 LTX 六个模型上测试，但阈值必须按架构校准。将 caching 与 mid-trajectory pruning 组合后，exploration speedup 达到 **$3.11\times$**，仍保留 88.6% gain，说明减少单候选成本和提前淘汰候选可以叠加。

## 我的提问

### Q1：Cached rollout 和完整视频并不完全一致，为什么仍能用于搜索？

搜索真正需要的是 $\operatorname*{arg\,max}$ 决策，而不是逐像素复现。只要 cache-induced score noise 小于候选之间有意义的分差，廉价 rollout 就能指向相同或近似等价的 seed。论文因此使用 prompt 内 Spearman、top-1 agreement、regret 和 gain capture，而不把 LPIPS 当作核心成功标准。

### Q2：为什么最终还要完整重跑一次，而不直接输出 cached winner？

Aggressive caching 会系统性削弱运动：$\tau=0.10$ 时 mean optical flow 下降 3.1%，$\tau=0.20$ 时下降 8.0%。ImageReward 和 VideoScore 反而可能偏爱运动较弱、静态帧更清晰的结果。Recommit 能移除最终交付视频中的缓存 artifact；但 cached preview 的 bias 仍可能影响 seed selection，因此还需要 motion-aware verifier 和直接的 temporal audit。

### Q3：对 VideoCoCo 最直接的借鉴是什么？

可以把 VideoCoCo 扩展为多方案的 **simulation-aware cheap explore, full-quality commit**：为同一 prompt 生成多个 Blender programs 或多个编辑 seed，在 Blender draft/cached editor 阶段用 prompt alignment、轨迹、光流和事件时序共同筛选，只对赢家运行完整视频编辑。Blender trajectory 不仅作为生成 condition，也可成为 motion/physics verifier 的参照，恰好补足 CachedSearch 的 verifier motion bias。这是基于两篇工作的研究设想，不是 CachedSearch 已验证的结论。

### Q4：代码是否已经公开？

截至 2026-08-03，arXiv 页面没有官方代码链接，正文仍表述为将发布 caching wrapper、search harness、评测脚本、prompts、seeds 和逐候选分数；完整复现材料仍待补充。

## 局限与疑问

- 核心 verifier 是对 8 个抽样帧取平均的 ImageReward，无法可靠评价动态过程；VideoScore 也共享对 motion-dampened cached outputs 的偏好。论文的 direct temporal metrics 揭示了问题，但尚未提供可靠的运动感知选择器或充分的人类偏好验证。
- Recommit 保证最终输出是标准 full-compute sample，却不能撤销探索阶段已经发生的错误排序。对精细机械运动、非写实风格和 articulated manipulation，缓存扰动可能改变 verifier-relevant dynamics。
- 固定 $\tau$ 的跨架构泛化不成立：LTX-Video 在 $\tau=0.10$ 时只保留 67.6% gain，CogVideoX 和 Hunyuan 也明显弱于 Wan；每个新的 model/verifier 组合都需要 paired cached/full calibration。
- 论文主要研究 best-of-$N$ seed selection。验证中间状态、改变 trajectory、进行 evolutionary search 或使用非确定性 pipeline 时，是否仍能安全 recommit 需要单独验证。
- 94.7% 与约 90.1% capture 来自不同聚合方式；阅读或复现时应同时报告 ratio-of-means、mean per-prompt capture、置信区间和 regret，避免只引用更好看的数字。

## 我的判断

方法本身很简单，真正有价值的是把近似计算的评价目标从“廉价输出是否像昂贵输出”改成“是否保住下游真正使用的选择决策”。Seed-matched paired evaluation、ranking fidelity、regret 和 break-even cost model 都很适合迁移到其他 multi-fidelity generation pipeline。

对当前 VideoCoCo 方向，最值得做的不是机械接入 EasyCache，而是搜索多个可执行物理方案，用 Blender 轨迹构造 motion/physics-aware verifier，再对赢家执行 full-quality video editing。这个组合既能扩大物理 reasoning space，也能针对 CachedSearch 最明显的 motion bias，具有形成独立研究问题的潜力。

但论文仍是很新的 arXiv 工作，代码尚未发布，跨架构阈值需要校准，且最终收益依赖 verifier 是否真的对应人类在意的视频质量。当前适合借鉴其系统范式和评测方法，再做小规模 paired pilot，不适合直接把默认 $\tau$ 和 headline capture 当作可迁移结论。

## 下次只看这些

1. 廉价阶段不必复现最终输出，只需保持候选排序；赢家再以同 seed 完整 recommit。
2. 默认 best-of-8 以 63.3% 成本保留 94.7% ratio-of-means gain，但 motion bias 和约 90.1% mean per-prompt capture 必须一起看。
3. 对 VideoCoCo 可尝试“多 Blender 物理方案探索 + trajectory-aware verifier + full-quality commit”。
