---
title: "End-to-End Training for Autoregressive Video Diffusion via Self-Resampling"
paper_url: "https://arxiv.org/abs/2512.15702"
authors: "Yuwei Guo et al."
venue: "arXiv"
published: "2025"
read_date: "2026-08-10"
read_at: "2026-08-10T18:29:03+08:00"
status: "已精读"
tags: ["Video Diffusion", "Autoregressive Generation", "Exposure Bias", "Long Video"]
one_liner: "Resampling Forcing 先 stop-gradient 地把 GT 历史重采样成带当前模型误差的 $t=0$ context，再以全部 GT chunk 为 target 做一次并行的标准 flow-matching 更新。"
---

## 研究问题

自回归视频 diffusion 在 teacher forcing 训练中看到干净真实历史，推理时却只能依赖自身生成且逐步累积误差的历史。简单高斯加噪不能复现模型特有的结构错误；从纯噪声完整 self-rollout 又昂贵，并常需 teacher、discriminator 或 distribution matching。论文希望用真实视频的 paired supervision，直接训练一个对自身历史错误稳健的严格 causal 模型，同时控制长历史 attention 成本。

## 核心方法

![Long-video comparison for Resampling Forcing](media/resampling-forcing/method-overview.webp "论文 Figure 2：Teacher Forcing、短 teacher 蒸馏的 Self Forcing 与原生 15 秒训练的 Resampling Forcing 长视频对比。来源：End-to-End Training for Autoregressive Video Diffusion（Yuwei Guo et al.），CC BY 4.0。")

- **第一遍：autoregressive self-resampling。** 每个 iteration 先在 chunk loop 外采样一个共享的 resampling timestep $t_s$：

$$
\operatorname{logit}(t_s)\sim\mathcal N(0,1),\qquad
t_s\leftarrow\frac{s t_s}{1+(s-1)t_s},\quad s=0.6.
$$

  这里 $s=0.6$ 是改变 $t_s$ 分布的 shift factor，不是固定噪声强度。所有 chunk 共用同一个 $t_s$，但各自使用独立噪声：

$$
x^i_{t_s}=(1-t_s)x^i+t_s\epsilon^i.
$$

  再用当前 online model 按严格 AR 顺序，从 $t_s$ 走回 $0$：

$$
\widetilde x^i
=x^i_{t_s}+\int_{t_s}^{0}
v_\theta(x^i_t,\widetilde x^{<i},t,c)\,dt.
$$

  因而 $\widetilde x^2$ 条件于 $\widetilde x^1$，$\widetilde x^3$ 条件于 $\widetilde x^{1:2}$，同时包含 chunk 内 solver/model error 和跨 chunk 累积误差。整个 history rollout 使用 `no_grad`；实验为省算力只用一次 Euler step，目标不是高质量采样，而是便宜地产生与 GT 语义对齐、带当前模型错误的 noise-free history。

- **第二遍：所有 chunk 一次并行做 flow matching。** 从原始 GT 再为每个 chunk 独立采样训练 timestep $t_i$：

$$
z^i=(1-t_i)x^i+t_i\epsilon^{\prime i},\qquad
\mathcal L_{\mathrm{FM}}
=\frac1N\sum_{i=1}^{N}
\left\lVert
v_\theta(z^i,\widetilde x^{<i},t_i,c)
-(\epsilon^{\prime i}-x^i)
\right\rVert_2^2.
$$

  模型一次接收 noisy-query sequence $z^{1:N}$ 与 detached history sequence $\widetilde x^{1:N}$。block-causal mask 使 query $z^i$ 只能在本 chunk 内 self-attend，并读取严格更早的 $\widetilde x^{<i}$；它看不到其他 noisy query，也看不到自己的 $\widetilde x^i$，因此不会泄漏答案。

| 当前训练 query | 可见的 resampled context | 是否计算 FM loss |
|---|---|---|
| $z^1$ | 无 | 是 |
| $z^2$ | $\widetilde x^1$ | 是 |
| $z^3$ | $\widetilde x^{1:2}$ | 是 |
| $z^N$ | $\widetilde x^{1:N-1}$ | 是 |

  这等价于把 $N$ 个条件不同的 next-chunk 训练样本，通过同一个稀疏 attention mask 合并到一次 transformer forward；顺序发生在第一遍 history 构造，第二遍监督 forward 才是并行的。每个 $\widetilde x^i$ 只给未来 chunk 当 context，每个 $z^i$ 都贡献 loss。

- **History Routing，而非固定 window。** Dense 版本读取全部合法历史。长视频版本对每个 head、每个 query token，从所有 $j<i$ 的 chunk 中动态选择 top-$k$：

$$
\Omega(q_i)=\operatorname{TopK}_{j<i}
\left(q_i^\top\phi(K_j)\right),
$$

  其中 $\phi$ 是 frame/chunk key 的 mean pooling。实验取 $k=5$；这不是“最近 5 个”的 sliding window，而可能同时选择第一帧、近期帧和其他语义相关历史。

## 关键发现

- 基于 Wan2.1-1.3B，autoregressive unit 为 3 个 latent frames。训练依次为：5 秒 teacher-forcing warm-up 10K steps、5 秒 RF 15K、15 秒（249 RGB frames）RF 5K，再在 15 秒视频上开启 routing 微调 1.5K。
- 它没有 few-step distillation：history 构造的一步 Euler 只是错误模拟；正式推理仍使用 32-step Euler、CFG 5.0。
- 15 秒总体对比中，autoregressive resampling 的 temporal/visual/text 分数为 **90.46/64.25/25.26**，优于 parallel resampling 的 **88.01/62.51/24.51** 和 noise augmentation 的 **87.15/61.90/21.44**。
- Dense RF 在 $0$–$5$、$5$–$10$、$10$–$15$ 秒三个区间的 temporal score 为 **91.20/90.44/89.74**；Self Forcing 为 **90.03/84.27/84.26**，长段稳定性优势更明显。但 RF 的前两段 visual score 低于 Self Forcing，收益并非所有质量维度全面占优。
- top-5 of 20 history 达到 **75% history sparsity**，质量接近 dense attention；top-1 routing 也比同等稀疏度的 window-size-1 更能保持外观一致性。

## 我的提问

### Q1：构造 history cache 时，不同 chunk 用相同还是不同的 timestep？

同一个 batch sequence 的伪代码只采样一次 $t_s$，所以所有 chunk 共用 $t_s$，但使用不同 $\epsilon^i$，并按 $\widetilde x^1\rightarrow\widetilde x^2\rightarrow\cdots$ 自回归地产生 history。第二遍 FM 训练则调用 `sample_t(N)`，每个 query chunk 可以有不同的 $t_i$。

### Q2：怎样把 context 不同的所有 chunk 拼在一次训练里？

它维护两条 sequence：$z^{1:N}$ 是 noisy GT query，$\widetilde x^{1:N}$ 是 resampled $X_0$ history。自定义 mask 只开放“$z^i$ 的本 chunk tokens”和“$\widetilde x^{<i}$ 的历史 tokens”，因此一次 forward 可同时输出所有 $v^i$，再对全部 chunk 求平均 FM loss。这与 LLM 用 causal mask 一次训练所有 next-token prediction 类似，但这里 current query 与 clean-ish history 是两条不同输入流。

### Q3：context 会使用固定 sliding window 吗？

核心 RF 的 dense mask 不截断，chunk $i$ 可读取全部 $\widetilde x^{<i}$。可选的 History Routing 从全部过去动态选 top-$k$，不是固定最近邻 window；历史不足 $k$ 时读取全部过去。

### Q4：它是否需要先训练 few-step model，或者使用 DMD？

不需要。Teacher-forcing warm-up 只是用 GT history 训练普通 causal flow-matching velocity field，使 online model 先具备基本生成能力。RF 之后仍用 paired GT flow-matching target；无 teacher score、fake-score network 或 DMD。训练中一步走到 $X_0$ 是故意粗糙的 context corruption generator，最终模型仍按 32 steps 推理。

### Q5：它与 In-Context Forcing 的真正区别是什么？

两者都让历史不再完美，但对象不同：RF 把 GT 历史扰动后 resample 回 $t=0$，用“clean-looking 但带模型误差”的 context 做 teacher-free、32-step FM 训练；ICF 让历史 chunk 真正停留在互不相同的中间 noise states，用 DMD 训练 4-step student，并把 staircase schedule 转化为 wavefront 并行推理。RF 是 **error simulation / robust causal training**，ICF 是 **diffusion-state alignment / few-step parallel inference**。

## 局限与疑问

- 第一遍 detached history 仍是按 chunk 顺序执行的 online-model rollout；无外部 teacher 不等于训练便宜，长 sequence 会增加显著训练 wall time。
- 训练用从 GT 锚定的 $x_{t_s}$ 和一步 Euler 构造 error proxy，而真实推理从纯噪声出发并走 32 steps；这种误差分布近似是否覆盖最危险的 rollout failure mode，仍缺直接测量。
- 主要验证在 Wan2.1-1.3B、最长原生 15 秒；更大模型、更长 native training 和交互式 world-model setting 尚待验证。
- History Routing 的 mean-pooled key 可能漏掉短促但关键的局部事件；top-$k$ 选择本身也没有学习显式的长期状态表示。
- 与 ICF 的指标不能直接横比：模型训练目标、sampling steps、视频长度和评测协议不同。

## 我的判断

RF 最值得学习的是把“模拟推理错误”与“稳定的 paired supervision”拆成两遍：第一遍只负责制造会随 online model 演化的错误 context，第二遍仍保留简单可靠的 FM target。它比手工 history noise 更贴近 AR error，又避免 DMD/teacher；代价是顺序 self-resampling 的训练计算，以及最终仍为 32-step inference。

若复现，应依次测：teacher forcing、noise augmentation、parallel resampling、AR resampling，再单独打开 routing；同时记录 history-rollout wall time，并比较一步 proxy 与真实 32-step rollout 的误差统计，避免只看最终 VBench。

## 下次只看这些

1. 同一 $t_s$ 顺序造 detached imperfect $X_0$ history；每 chunk 独立 $t_i$ 并行做 GT flow matching。
2. 一步 history resampling 不等于 few-step model；正式推理仍是 32 steps、无 DMD。
3. Dense 看全部过去，长视频用动态 top-5 routing 而非固定 sliding window。
