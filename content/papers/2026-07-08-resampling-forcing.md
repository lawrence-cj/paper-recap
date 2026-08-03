---
title: "End-to-End Training for Autoregressive Video Diffusion via Self-Resampling"
paper_url: "https://arxiv.org/abs/2512.15702"
authors: "Yuwei Guo et al."
venue: "arXiv"
published: "2025"
read_date: "2026-07-08"
status: "已精读"
rating: 3
tags: ["Video Diffusion", "Autoregressive Generation", "Exposure Bias", "Long Video"]
one_liner: "Resampling Forcing 先用模型自身把真实历史扰动成推理期风格的错误历史，再 stop-gradient 地用标准 flow matching 学会在这种历史上继续生成。"
---

## 研究问题

自回归视频 diffusion 训练时通常看到干净真实历史，推理时却只能条件于自己生成且不断累积误差的历史，形成 teacher-forcing exposure bias。简单加噪不能复现模型特有的结构错误；完整从纯噪声 rollout 又昂贵且常需对抗或分布匹配目标。

## 核心方法

- 第一遍 **self-resampling**：从真实历史出发，在噪声强度 $s=0.6$ 处加入扰动，按真实自回归顺序用一次 Euler 更新生成“像模型自己犯错”的历史。
- 将该 dirty history detach，阻断穿过第一遍 rollout 的梯度。
- 第二遍在 dirty history 条件下，用正常 paired flow-matching 目标预测干净 ground-truth future：

$$
\widetilde h=\operatorname{stopgrad}\!\left(R_\theta(h_{\mathrm{gt}},\epsilon;s)\right),
$$

$$
\mathcal L_{\mathrm{RF}}
=\mathbb E_{x,t,\epsilon}
\left[\left\lVert v_\theta(x_t,t\mid \widetilde h)-v^*(x_t,t)\right\rVert_2^2\right].
$$

- **History Routing**：对历史帧 key 做 mean-pooling，由当前 query 计算相关性，按 head/token 选 top-k 历史块；20 帧历史取 top-5 时达到约 75% history sparsity。

## 关键发现

- Wan2.1-1.3B 设置：teacher-forcing warmup 10k steps，RF 训练 15k（5 秒）+ 5k（15 秒），History Routing 再训练 1.5k；推理使用 Euler 32 steps、CFG 5。
- 自回归 resampling 的 temporal score **90.46**，高于 noise augmentation 的 **87.15** 和 parallel resampling 的 **88.01**。
- 记录中的 visual/text 指标为 **64.25/25.26**，说明时间一致性提升最明确，视觉/文本维度并非全面占优。

## 我的提问

### Q1：它和 Self-Forcing 的本质区别是什么？

Self-Forcing 从纯噪声采样模型自己的完整分布，再用 DMD/SiD/GAN 类 distribution matching 训练；Resampling Forcing 从 ground truth 锚定的中间噪声构造有模型特征的错误历史，detach 后仍使用 paired flow matching。后者目标更简单，但训练仍需额外自采样计算。

### Q2：为什么 parallel resampling 不够？

真实推理错误是顺序传播的：第 $i$ 个 chunk 的错误会改变第 $i+1$ 个 chunk。所有历史块并行重采样无法模拟这种因果累积，因此 temporal score 低于严格 AR resampling。

## 局限与疑问

- self-resampling 增加训练前向与顺序 rollout 成本，“无外部 teacher”不等于训练便宜。
- 主要验证在 Wan2.1-1.3B 和最长约 15 秒，规模、长度与多任务泛化有限。
- History Routing 的 pooled key 近似可能漏掉短促但关键的历史事件。
- 自动视频 benchmark 对长期因果和细微视觉退化仍不充分。

## 我的判断

这是针对 AR video exposure bias 很直接的训练配方：用模型自身错误替代手工高斯噪声，同时保留稳定的监督目标。若复现，先只验证 self-resampling，再单独加入 routing，避免混淆训练收益和 attention 加速收益。

## 下次只看这些

1. 第一遍造 dirty history 并 detach，第二遍才做 paired flow matching。
2. 严格 AR resampling 比并行加噪更接近推理期误差传播。
3. 时间指标强，但训练成本、长视频规模和视觉质量仍需独立验证。
