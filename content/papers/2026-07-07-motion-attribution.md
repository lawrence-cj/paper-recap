---
title: "Motion Attribution for Video Generation"
paper_url: "https://arxiv.org/abs/2601.08828"
authors: "Xindi Wu et al."
venue: "arXiv"
published: "2026"
read_date: "2026-07-07"
status: "已精读"
tags: ["Video Generation", "Data Selection", "Motion", "Influence Functions"]
one_liner: "Motive 用运动区域加权的 diffusion gradient 衡量训练视频对目标运动 query 的影响，从海量数据中挑出少量真正能改善运动的样本。"
---

## 研究问题

视频微调数据中，大量样本的运动弱、背景占主导或与目标 motion distribution 无关。随机抽样浪费训练预算，按 optical-flow magnitude 又无法判断某条视频是否真的会推动模型学到 query 所需运动；论文研究训练样本级 motion attribution。

## 核心方法

![Motive motion attribution pipeline](media/motion-attribution/method-overview.webp "论文 Figure 1：Motive 从运动检测与 loss mask 生成 motion gradient，再通过投影和 influence matrix 排序训练数据。来源：Motion Attribution for Video Generation（Xindi Wu et al.），CC BY 4.0。")

- 用 AllTracker 得到稠密轨迹/光流并构造 motion mask，使归因聚焦于运动区域而非静态背景。
- 对候选训练视频和目标 query 视频计算 motion-weighted diffusion loss gradient。
- 将约 1.419B 维参数梯度用 Fastfood 随机投影压缩到 512 维，并归一化。
- 用 query gradient 与候选 gradient 的 cosine similarity 近似 influence，选择最高分的 10% 数据：

$$
I(x,q)
\approx
\frac{(Pg_x)^\top(Pg_q)}{\lVert Pg_x\rVert_2\lVert Pg_q\rVert_2},
$$

其中 $g_x=\nabla_\theta\mathcal L_{\mathrm{motion}}(x)$，$P$ 是 Fastfood projection。实现固定一个 timestep $t=751$ 和噪声以控制成本。

## 关键发现

- Wan2.1 Dynamic Degree：base **39.6**，full fine-tune **42.0**，random 10% **41.3**，无 mask gradient **43.8**，Motive 10% **47.6**。
- Wan2.2：base **42.0**，full **45.3**，random **41.6**，无 mask **43.8**，Motive **48.3**。
- 人类偏好相对 base 为 **74.1%**；random 10% 为 58.9%，full data 为 53.1%。结果支持精选 10% 在运动上优于全量微调，但背景/成像质量可能略有退化。

## 我的提问

### Q1：它是新训练 loss 还是数据选择算法？

核心是训练前的数据选择。motion-weighted loss 用来生成每条数据的归因 embedding，最终 fine-tuning 仍可沿用基础模型的原训练目标。不要把 attribution 阶段与模型微调阶段混为一谈。

### Q2：为什么不能只按 motion magnitude 排序？

大光流可能来自镜头抖动，也未必与目标 query 的运动类型一致。gradient similarity 衡量的是“在当前模型上，这个样本会把参数推向与目标 motion 相似的方向”，更接近训练作用而非表面运动量。

## 局限与疑问

- optical tracker 在遮挡、快速运动和非刚体变化上会失败，camera motion 也可能污染 mask。
- 单 timestep、单噪声、512 维投影以及省略 Hessian 都是较强近似。
- 归因计算本身昂贵，且结果依赖当前 checkpoint 与 query set。
- 论文主要证明微调数据选择，不足以声称可解释预训练数据的完整因果贡献。

## 我的判断

这是少量高价值视频筛选的强基线，尤其适合有明确 motion target 的后训练。实际落地应先做 camera-motion 抑制和 tracker 置信度过滤，并检查运动提升是否牺牲背景稳定与画质。

## 下次只看这些

1. 运动 mask 后的 diffusion gradient 才是样本表示，不是直接用光流大小。
2. Fastfood 把超高维梯度压到 512 维，再做 cosine influence。
3. 10% 数据运动收益突出，但归因近似、成本和画质副作用要复核。
