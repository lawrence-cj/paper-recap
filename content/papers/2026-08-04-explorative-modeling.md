---
title: "Explorative Modeling: Unlocking a Third Pretraining Axis and End-to-End Generation"
paper_url: "https://arxiv.org/abs/2607.27372"
authors: "Alexi Gladstone, Heng Ji, Yilun Du"
venue: "arXiv"
published: "2026"
read_date: "2026-08-04"
read_at: "2026-08-04T19:43:43+08:00"
status: "已精读"
tags: ["Generative Modeling", "Diffusion Models", "Scaling Laws", "Video Generation"]
one_liner: "训练时为同一数据探索 K 个 latent/noise，只反传最匹配者，把额外训练算力换成更好的 data–latent coupling；但它只保证 K 个候选中有一个匹配，不保证推理时任意单 noise 都与 prompt 匹配。"
---

## 研究问题

直接用 reconstruction loss 从随机 latent 一步回归多模态数据时，同一输入附近会被不同有效目标反复拉动，平方误差的最优解容易落在模式之间的模糊平均。Diffusion、flow 和 autoregressive model 通过把 generation 拆成许多条件更充分、近似单模态的小步骤来避免这种 mode blurring，但训练通常只学习一步，推理却递归几十到数百步，产生 exposure bias、较高延迟和 train–inference mismatch。

论文提出不拆 generation，而是拆 training loop：每次训练让模型探索多个 latent–data couplings，只强化当前最匹配的一组。作者把探索量 $K$ 解释为继参数量和数据量之后的“第三个 pretraining axis”，并进一步尝试用它承担原本由多步生成提供的 generative expressivity。

## 核心方法

- **Forward XM：固定数据，搜索 generation**。对数据样本 $x$，从模型采样 $K$ 个候选 $\hat y_i\sim G_\theta$，只对 reconstruction loss 最小者反向传播：

$$
\mathcal L_{\mathrm{Forward}}(\theta)
=\min_{i\in\{1,\ldots,K\}}J(\hat y_i,x).
$$

  每条真实数据都会拉近至少一个 generation，因此 Forward XM 偏向覆盖完整数据分布，即 mass-covering/recall；代价是 $K$ 次 generation forward。
- **Reverse XM：固定 generation，搜索数据**。对一个 $\hat y\sim G_\theta$，从 $K$ 个真实目标中选最近者：

$$
\mathcal L_{\mathrm{Reverse}}(\theta)
=\min_{i\in\{1,\ldots,K\}}J(\hat y,x_i).
$$

  它只需一次模型 forward，扩大 $K$ 主要增加相似度计算，但偏向 precision/mode-seeking；若没有 entropy bonus 或 coverage constraint，模型可能塌缩到少量模式。主论文的大部分实验采用 Forward XM。
- **加入 diffusion/flow**：对同一数据 $x$、condition $c$ 和 timestep $t$，采样 $K$ 个不同噪声 $\epsilon_i$，构造 $K$ 个 $x_t^{(i)}$，计算各自的 velocity/flow-matching loss，只反传最小的一项：

$$
\ell_i=\left\lVert
v_\theta(x_t^{(i)},t,c)-v_{\mathrm{target}}^{(i)}
\right\rVert_2^2,
\qquad
\mathcal L=\min_i\ell_i.
$$

  它不是先完整生成 $K$ 个图像或视频；同一 training example 内只有 noise 不同。其作用是寻找一条当前模型已经较容易映射到该数据的 noise trajectory，再强化这条 coupling。
- **只增加训练成本**：推理仍从原 prior 采一个 noise，并运行原生成流程，不需要 best-of-$K$。对 Transformer，额外 candidate 只有 forward、没有 backward；FLOP-efficient 实现中 XM-$K$ 约等于 $(K+2)/3$ 个普通训练 step，即 $K=2/3/5$ 约为 $1.33/1.67/2.33\times$ step FLOPs。
- **两种显存策略**：可把 $K$ 展开到 batch dimension，同时保存全部 activation，只反传 winner；也可先 no-grad 运行所有 candidates，再带梯度重跑 winner，以一次额外 forward 换取接近普通训练的 activation memory。
- **End-to-end XM**：作者将“训练中模型实际采样路径与推理路径相同”定义为 end-to-end。Standalone XM 可以直接从 noise 一次或少数几次生成；训练期仍使用 GT 进行 best-of-$K$ winner selection，而推理只抽一个 latent，这是该定义没有消除的 selection mismatch。

## 关键发现

- 在 ImageNet $256\times256$ 的 RAE recipe 上，XM-2 将 unguided gFID 从 **1.55 降到 1.43**，$\mathrm{FDr}^{6}$ 从 **4.42 降到 3.91**。论文所称 **6.2× sample efficiency、4.1× FLOP efficiency**，指达到 baseline 最佳质量所需的数据和训练 FLOPs，不是每步训练或 inference 自动便宜相同比例。
- 固定模型大小、只增加 $K$ 时，图像 FID、视频 FVD 和 masked-diffusion language model 的 perplexity–entropy frontier 都单调改善；作者测得 XM-5 的收益随模型规模从 **13% 增至 23%**，随数据规模从 **7% 增至 36%**。
- 视频实验使用 Something-Something V2、$128\times128$、10 帧，并把第 0、1、9 帧作为 condition。XDiffusion 的 FVD 从 $K=1$ 的 **36.9** 降到 $K=12$ 的 **30.0**；10-step XJumpy 从 $K=1$ 的 **26.9** 降到 $K=8$ 的 **21.2**。这是强条件、小规模 video world model，不是开放式高分辨率 T2V。
- 越接近 end-to-end、generation steps 越少的 XJumpy，从探索中获益越大；随着 $K$ 增加，最优 jump 数下降，支持“training exploration 可以部分替代 generation factorization”的主张。
- 在低模态控制任务中，XM-10 Explorative Policy 用 **1 NFE** 匹配 100-step Diffusion Policy；Maze2D world model 平均 **2.3 NFE、score 130.0**，对比 Diffuser 的 **192 NFE、127.2**。这尚未证明开放式图像或视频可以稳定单步生成。

## 我的提问

### Q1：推理时随机 noise 与 prompt 不匹配怎么办？

Conditional model 实际学习的是 $G_\theta(z,c)$，理想情况下不存在全局的“狗 noise”：prompt $c$ 决定猫或狗，$z$ 只决定当前条件下的姿势、视角、背景和运动。例如同一 $z_1$ 可在“猫”条件下表示趴着的猫，在“狗”条件下表示趴着的狗。

但论文目标没有严格保证这种分工。Hard-min 只保证：对训练样本 $(x,c)$，从 $K$ 个 noise 中至少能找到一个低 loss candidate；它不保证 prior 中每个 $z$ 在该 $c$ 下都生成合理结果。Forward XM 是 data-covering，不等于 latent-covering。训练优化 $K$-candidate mixture，推理却只从 $G_\theta(z,c)$ 抽一个样本，有限 $K$ 下仍有 Jensen gap 和潜在 dead-latent/prior-hole 风险。

ImageNet class condition 会反复看到同一类别，论文视频又由首尾帧强约束，condition 容易压住 noise 语义；开放式 T2V 的 prompt 几乎不重复、剩余模态极多，noise dominance、prompt leakage 与单 seed 不稳定更可能成为核心问题。因此论文结果不能直接回答这一质疑。

### Q2：怎样保留 exploration，又避免只有幸运 noise 得到训练？

一个保守的待验证改法是混合普通 objective 与 winner objective：

$$
\mathcal L_{\mathrm{mix}}
=(1-\lambda)\frac{1}{K}\sum_{i=1}^{K}\ell_i
+\lambda\min_i\ell_i.
$$

$\lambda=0$ 是让所有 noise 接受训练的普通 loss，$\lambda=1$ 是论文 hard-min XM。也可用 soft-min，让 winner 权重大但其他 candidates 仍有梯度。这些是针对开放式条件生成的研究建议，不是论文已经验证的结论；实验必须额外检查 candidate win rate、noise norm、prompt consistency、随机单 seed 的 worst/mean quality 和 diversity。

### Q3：“第三个 scaling axis”有多可信？

Best-of-$K$、winner-take-all、Multiple Choice Learning 和 IMLE 都不是新机制；作者也明确把贡献放在 generative expressivity 的统一解释、与 generation factorization 的关系，以及随规模增长的 compute-matched 实验上。跨图像、视频和离散语言的单调趋势值得重视，但论文是刚发布的 arXiv v1，最大实验仍比 foundation-model training 小约四个数量级。“越大模型越需要 exploration”目前是有证据支持的趋势，尚不是经过大模型预训练验证的 scaling law。

### Q4：对当前 video generation 工作最直接的借鉴是什么？

最小改动是在现有 video flow-matching 训练中保持相同 GT video、condition 和 timestep，只将 noise 扩展为 $K=1/2/3/5$，比较相同 training steps 与相同总 FLOPs。优先放在 few-step、distillation、I2V 或强条件 video editing：这些任务剩余模态较少，而论文显示越接近 end-to-end 的模型越可能从 exploration 获益。

对开放式 T2V，不应只看 FVD。必须报告随机单 seed 的 prompt alignment、motion、temporal consistency、precision/recall/diversity，以及 winner noise 使用分布；只有 single-sample generation 也持续改善，才能说明 XM 改善了模型分布，而不是只提高“训练时总能找到一个幸运 noise”的概率。

## 局限与疑问

- Best-of-$K$ 是已有技术，论文真正的新意主要是理论 framing 和 scaling 论证；“第三个 axis”仍需独立复现与大规模验证。
- Forward XM 训练成本随 $K$ 近似线性增长；高模态数据可能需要很大的 $K$，使 fully end-to-end open-domain image/video generation 不现实。
- Reverse XM 计算更便宜，却天然 mode-seeking，需要 entropy 或 coverage constraint；连续 condition 下也难以为每个 generation 找到一组真正同条件的有效 targets。
- Hard-min 只给 winner 梯度，可能产生 latent starvation、候选使用不均和 prior holes；论文主要使用 FID/FVD 等整体分布指标，没有充分分析随机单 seed 的条件一致性尾部风险。
- 视频证据局限在 $128^2$、10 帧、首尾帧强条件的 SSv2；不能外推到现代高分辨率、长视频、开放式 T2V/I2V。
- 论文对 end-to-end 的定义聚焦 model generation path 相同，但训练期 oracle best-of-$K$ selection 在 inference 不存在，因此并未消除所有训练—推理差异。
- 代码 release 不完整：主仓库公开了部分 image/video 训练代码，但 RAE、MDLM、control tasks 和 Reverse-XM language 的关键实现仍标注待发布。

## 我的判断

这是一篇机制简单、问题 framing 很强、值得做低成本 pilot 的论文。它把 latent–data coupling 从随机配对变成随模型共同演化的训练期搜索，并用 compute-matched 结果说明：在部分任务上，把算力花在探索不同 couplings 比单纯延长训练更有效。作为现有 diffusion/flow recipe 的 augmentation，$K=2/3$ 几乎不需要改架构，也不增加 inference 成本，复现门槛低。

但当前证据不足以接受“foundation generative model 的第三个 scaling axis”或“开放式生成将走向单步 XM”这两个强结论。对当前视频方向，真正决定价值的不是 best-of-$K$ training loss 能否下降，而是随机单 noise 推理是否仍与 prompt 匹配、是否覆盖所有 latent region，以及 equal-FLOPs 下是否持续优于普通训练。

建议先在小型 few-step/强条件 video model 上做 $K=1/2/3/5$ 与 mixed-loss/soft-min 对比；若 single-seed quality、diversity 和训练效率同时改善，再扩大模型和视频分辨率。不要直接从大规模开放式 T2V 或纯 hard-min $K\gg1$ 开始。

## 下次只看这些

1. XM = 训练时对同一数据探索 $K$ 个 latent/noise，只反传最匹配者；增加训练成本，不增加 inference 成本。
2. Forward XM 保证 data coverage，不保证每个 noise 与每个 prompt 都匹配；开放式 T2V 的 single-seed latent coverage 是最大未解问题。
3. 最值得做的是 few-step/强条件 video 上的 $K=2/3$ equal-FLOPs pilot，并同时比较 hard-min、soft-min 和 ordinary-loss mixture。
