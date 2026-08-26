---
title: "Continuous Adversarial MeanFlow Transfer"
paper_url: "https://arxiv.org/abs/2608.19540"
authors: "Yara Bahram et al."
venue: "arXiv"
published: "2026"
read_date: "2026-08-26"
read_at: "2026-08-26T17:43:50+08:00"
status: "已精读"
tags: ["Diffusion Models", "MeanFlow", "Domain Adaptation", "Few-Step Sampling"]
one_liner: "MF-T 把预测 x/ε/v/u 的异构 ImageNet 模型统一改造成目标域少步 MeanFlow，CAMF 再以真实和预测区间端点间的势能平均变化做对抗后训练，在不增加推理 NFE 的前提下补回大步回归丢失的细节。"
---

## 研究问题

将预训练 diffusion 或 flow generator 迁移到数据较少的新领域后，原模型通常仍需几十到数百次网络调用；迁移解决了目标域适配，却没有解决推理速度。另一方面，现有少步加速方法往往绑定某一种输出参数化，而预训练模型可能预测干净样本 $x$、噪声 $\epsilon$、瞬时速度 $v$，或 MeanFlow 的平均速度 $u$，缺少统一的“迁移并加速”接口。

MeanFlow（MF）不预测单个时刻的瞬时速度，而预测区间 $[r,t]$ 上的平均速度：

$$
u(z_t,r,t)
=\frac{1}{t-r}\int_r^t v(z_\tau,\tau)\,d\tau.
$$

因此它能从 $z_t$ 直接做大步更新 $z_r=z_t-(t-r)u(z_t,r,t)$。但有限区间越长，平方回归越容易对多个合理终点取条件均值，表现为纹理、边缘和细节被平均；已有 CAFM 对抗目标判断的是瞬时 velocity，并不对应 MF 推理时实际执行的有限区间 transport。

## 核心方法

- **MF-T：统一 parameterization 后迁移成 MeanFlow。** 在线性路径 $z_t=(1-t)x+t\epsilon$ 上，各类 source prediction 都先转换成共同的瞬时速度：

$$
\hat v(z_t)=
\begin{cases}
\dfrac{z_t-\hat x(z_t)}{t}, & x\text{-prediction},\\
\dfrac{\hat\epsilon(z_t)-z_t}{1-t}, & \epsilon\text{-prediction},\\
\hat v(z_t), & v\text{-prediction},\\
u(z_t,t,t), & u\text{-prediction}.
\end{cases}
$$

  转换后的 source weights 用于初始化目标 MF model。对于原本只接收一个 timestep 的 $x/\epsilon/v$ model，按 DMF 的做法增加区间端点 $r$ 等条件，再在目标域用已有 iMF objective 训练。MF-T 结束时模型已经可以使用 4 NFE 采样；它并非只能四步，理论上可用不同区间划分跑 1、2、4、8 步，但论文中的迁移模型在 4 步附近更可靠。

- **只映射输出，不强行对齐 noise schedule。** 作者发现 Diff2Flow 式的 schedule alignment 在 domain shift 和低 NFE 下会使训练不稳定，尤其会放大 MF identity 中 JVP 的脆弱性；MF-T 仅保留必要的 time convention 翻转/缩放和 velocity mapping。这个选择是论文对 MF-T 最重要的经验结论。

- **CAMF：在 MF-T checkpoint 上做有限区间对抗后训练。** 从同一目标域样本 $x$ 和噪声 $\epsilon$ 构造真实路径：

$$
z_t=(1-t)x+t\epsilon,\qquad
z_r=(1-r)x+r\epsilon,\qquad 0\le r<t\le1.
$$

  MF generator 从同一个起点预测 fake endpoint：

$$
\widehat z_r=z_t-(t-r)u_\theta(z_t,r,t).
$$

  其中 $z_r$ 是 real endpoint；$r=0$ 时它就是干净图像 $x$，$r>0$ 时则是真实路径上的带噪中间状态。$\widehat z_r$ 是模型预测的 fake endpoint，两者具有相同的起点 $z_t$ 和时间区间。

- **$D_\psi$ 是 scalar potential，不是生成模型或概率输出。** 论文定义一个带时间条件的新 discriminator：

$$
D_\psi:\mathbb R^n\times[0,1]\rightarrow\mathbb R.
$$

  对单个 state-time pair，它只输出一个实数；batch 下通常为 $[B,1]$。例如输入 latent 可以是 $[B,4,32,32]$，pixel state 可以是 $[B,3,256,256]$，但输出仍是每个样本一个 scalar。这个 scalar 的绝对值没有校准后的概率或“真实性”语义，只有不同 state-time 点之间的差和导数有意义，因此训练还需要 centering penalty 防止整体平移漂移。

- **用势能的平均变化率评价 transport。** $\mathcal A$ 是函数名，不是与括号相乘；分号只用于分隔 state 和 time 参数：

$$
\mathcal A_\psi(z_t,z_r;t,r)
:=
\frac{D_\psi(z_t,t)-D_\psi(z_r,r)}{t-r}.
$$

  CAMF 分别计算：

$$
\mathcal A_{\mathrm{real}}
=\frac{D_\psi(z_t,t)-D_\psi(z_r,r)}{t-r},
\qquad
\mathcal A_{\mathrm{fake}}
=\frac{D_\psi(z_t,t)-D_\psi(\widehat z_r,r)}{t-r}.
$$

  Discriminator 通过 least-squares adversarial game 区分真实和模型 transport 的平均“势能坡度”，generator 则让 fake transport 的分布接近 real。它判断的是 MF 实际执行的有限区间移动，而不是直接给 endpoint 图片输出真假概率。

- **与 CAFM 的关系。** 当 $r\rightarrow t$ 时，区间差商退化为沿 velocity 的方向导数：

$$
\frac{D(z_t,t)-D(z_r,r)}{t-r}
\longrightarrow
\partial_zD\cdot v+\partial_tD.
$$

  因此 CAMF 严格包含 CAFM 的瞬时 criterion。$D_\psi$ 只在后训练中使用，推理时被丢弃；CAMF 不负责把模型变成四步，也不增加最终 sampling NFE。

## 关键发现

- 实验覆盖四种 ImageNet source parameterization：DiT（$\epsilon$）、SiT（$v$）、JiT（$x$）和 iMF（$u$），迁移到 ArtBench、Caltech、CUB-Birds、Food-101 和 Stanford Cars，均为 $256\times256$ class-conditional generation。
- CAMF 接在 MF-T 后，4-step 平均 FID 分别从 iMF 的 $13.13$ 降到 $7.88$、SiT 的 $11.96$ 降到 $10.30$、JiT 的 $29.14$ 降到 $22.11$、DiT 的 $47.26$ 降到 $29.32$；作者汇总为 few-step FID 平均改善约 **29%**。
- SiT 的五数据集平均 FID 在 4 NFE 时达到 **10.30**，略优于带 CFG 的 250-step fine-tuned model 的 **10.47**；DiT 在 8 NFE 时达到 **10.62**，优于 250-step model 的 **11.87**。
- “最多 125× fewer NFE”来自带 CFG 的 baseline 实际为 $250\times2=500$ 次网络调用，而 MF-T+CAMF 为 4 次。CUB 上 SiT 的 batch-32 单图延迟从约 **735 ms** 降至 **5.88 ms**，但训练更贵：MF-T 与 CAMF 两阶段约需 $3.32+6.99$ H100 GPU-hours，而普通 fine-tuning 约为 $1.54$ GPU-hours。
- CAMF 冷启动地直接作用在未经目标域 MF-T 适配的 source model 上效果较弱；它是依赖已有目标域低成本 transport 的 refinement stage，而不是独立完成迁移的方法。

## 我的提问

### Q1：MF-T 有多少真正的新意，是否只是调整 loss？

不只是换 loss：它还统一 $x/\epsilon/v/u$ 输出、继承 source weights、给单时间模型增加区间条件，并明确选择 velocity mapping 而不做 schedule alignment。但各零件大多已有来源：parameterization conversion 接近 Diff2Flow，结构改造来自 DMF，训练目标来自 iMF。严格评价，MF-T 更像有效的集成与迁移 recipe，创新主要在覆盖异构 source 的统一流程及“domain shift 下不要强行对齐 schedule”的实证；CAMF 才是更实质的方法贡献。

### Q2：$D(z,t)$ 是什么，输出有什么意义和 size？

$D_\psi$ 是 CAMF 阶段新训练的 time-conditioned discriminator potential，输入与 generator state 同尺寸的 $z$ 和一个 scalar timestep，输出每样本一个 scalar。它不输出概率、velocity 或 FID；可以把它想成定义在“状态 $\times$ 时间”空间上的可学习高度场，CAMF 使用的是两点间的平均坡度。由于只看差值，$D$ 的绝对零点没有意义。

### Q3：$\mathcal A_\psi(z_t,z_r;t,r)$ 中的 $\mathcal A$ 是否表示乘法？

不是。$\mathcal A$ 是新定义的 interval adversarial score 函数，括号表示函数调用，下标 $\psi$ 表示它依赖 discriminator 参数；分号没有运算含义，只是区分状态变量 $z_t,z_r$ 与时间变量 $t,r$。

### Q4：real 和 fake 分别是什么？

二者共享同一个带噪起点 $z_t$。Real 是由同一对 $(x,\epsilon)$ 解析构造的真实低噪终点 $z_r$；fake 是 MF 用平均速度从 $z_t$ 推出的 $\widehat z_r$。因此 CAMF 比较的是 $z_t\rightarrow z_r$ 与 $z_t\rightarrow\widehat z_r$ 两条区间 transport，而不只是两张独立图片。

### Q5：CAMF 在整篇论文中到底起什么作用？

MF-T 已经完成目标域迁移和四步化；CAMF 只负责质量 refinement。论文的解释是：有限区间平方回归会倾向条件均值，CAMF 通过分布级对抗监督允许模型产生清晰、合理的终点，从而补回纹理与边缘。实验中的 FID/FDD 和样例支持 CAMF 有明显增益，但论文没有单独做频率或纹理分析，所以“恢复细节”是合理的机制解释，不是被直接隔离验证的因果结论。

## 局限与疑问

- 只验证了 ImageNet 初始化的 $256\times256$ class-conditional image generation；高分辨率、text-to-image 和 video 尚未验证。
- 论文所说的 limited data 是相对 ImageNet 而言，实验使用完整目标数据集，没有严格的 $K$-shot subsampling 结果。
- 4-step 质量可靠，但在目标域数据有限时仍未恢复 data-rich MeanFlow 的高质量 1–2 step generation。
- 主实验的强基线主要是 standard fine-tuning 和 AFM；与更多近期 distillation、domain adaptation pipeline 的同协议横向比较不足。
- CAMF 的对抗阶段需要 discriminator warm-up 和多次 discriminator update，显著增加训练成本；headline acceleration 主要是推理收益。
- 论文按最佳 4-step FID checkpoint 报告结果。“CAMF 恢复细节”的叙述主要来自分布指标和视觉样例，缺少独立的纹理、频率或 mode-coverage 分析。

## 我的判断

最准确的定位是：MF-T 是一个覆盖异构预训练模型的实用 accelerate-and-adapt recipe，CAMF 是与 MeanFlow 有限区间推理算子匹配的 adversarial refinement。前者的单项组件创新有限，但把四类 parameterization 放进同一迁移协议、并指出 schedule alignment 在 domain shift 下会失败，具有工程价值；后者从 infinitesimal potential derivative 推广到 finite-difference potential slope，数学形式简洁，也与实际少步 transport 更一致。

实验足以支持“MF-T 后再做 CAMF 能显著改善 4-step FID/FDD，且不增加推理 NFE”，但还不足以证明它可直接泛化到大规模文生图、视频或严格 few-shot 场景。如果复现，应把 MF-T 与 CAMF 分阶段报告，并同时记录训练 GPU-hours、1/2/4-step 质量和实际 wall-clock，避免只复述 125× NFE。

## 下次只看这些

1. MF-T 后模型已经是 4-step MeanFlow；CAMF 只做质量增强，不改变 sampler 或 NFE。
2. CAMF 的关键是比较同一起点下 real endpoint $z_r$ 与 fake endpoint $\widehat z_r$ 的势能平均变化率；$D$ 每样本只输出一个无绝对语义的 scalar potential。
3. MF-T 偏集成创新，CAMF 才是主要方法创新；实证强在 4-step、四类 parameterization，外推到 1-step、text-to-image 和 video 仍待验证。
