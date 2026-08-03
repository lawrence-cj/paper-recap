---
title: "Why Larger Models Learn More: Effects of Capacity, Interference, and Rare-Task Retention"
paper_url: "https://arxiv.org/abs/2605.29548"
authors: "Jing Huang et al."
venue: "arXiv"
published: "2026"
read_date: "2026-06-26"
status: "已读"
rating: 3
tags: ["Scaling Laws", "Data Difficulty", "Curriculum Learning", "Video Training"]
one_liner: "数据难度不是样本的固定属性：大模型能从小模型视为噪声的困难样本中继续学习，因此选数应观察随容量和训练进程变化的可学习性。"
---

## 研究问题

训练数据常按 loss 或某个静态 difficulty score 排序，但高 loss 可能同时表示“信息丰富但尚未学会”与“标注错误或不可学习”。论文关注模型容量如何改变样本的可学习范围，以及为何大模型能从更多困难数据中获益。

## 核心方法

![Model scaling and learnable distribution regions](media/why-larger-models-learn-more/method-overview.webp "论文 Figure 1：数据扩展能触及的 loss 区域与必须依靠模型容量扩展才能学习的区域。来源：Why Larger Models Learn More（Jing Huang et al.），CC BY 4.0。")

- 将难度视为模型、训练阶段和数据共同决定的量，而不是数据集的永久标签。
- 比较不同容量模型在不同数据 bucket 上的 loss 水平、下降速度与泛化收益。
- 区分三类样本：快速收敛的 easy、下降较慢但持续改善的 hard-but-learnable、长期高损失且无改善的 noisy/unlearnable。

对样本 bucket $b$，比绝对 loss 更有用的是归一化学习斜率：

$$
s_b=-\frac{\Delta\,\widetilde{\mathcal L}_b}{\Delta\log n},
$$

其中 $\widetilde{\mathcal L}_b$ 是消除 timestep、SNR 或任务固有尺度后的验证 loss，$n$ 是训练步数。高 loss 且 $s_b>0$ 才更像“困难但正在学习”。

## 关键发现

- 大容量模型扩大 hard-but-learnable 区域，因此同一批数据对小模型可能接近噪声，对大模型却仍有可提取结构。
- 单次高 loss 排名无法可靠区分困难与脏数据；跨训练阶段的 learning curve 更关键。
- 对视频数据，可把归一化 loss 趋势与运动、时间、语义组合度及稀有度 proxy 一起使用。

## 我的提问

### Q1：如何判断视频数据是“难”而不是“坏”？

先按 timestep/SNR、分辨率、长度、condition 类型等归一化 denoising 或 validation loss，再跟踪多次 checkpoint。如果 loss 高但稳定下降，并且更大模型下降更明显，它更可能是 hard-but-learnable；若始终不降且人工检查发现字幕错配、剪辑断裂或伪影，则更像噪声。

### Q2：能否直接过采样所有高 loss 视频？

不能。应组合 $z$-score 后的 loss、下降斜率、motion/semantic complexity、稀有度与质量门控，并给每个 bucket 设置上限。否则会把坏 caption、损坏视频和极端噪声放大。

## 局限与疑问

- loss 对扩散 timestep 和参数化高度敏感，未归一化时几乎不能跨样本比较。
- proxy 容易把镜头抖动当复杂运动，把罕见伪影当稀有知识。
- 论文的容量结论不能自动给出生产数据管线的最佳阈值，仍需下游指标闭环。

## 我的判断

真正可执行的结论是：数据选择要从“静态难度表”升级为“分桶学习曲线”。对视频模型应建立一个小型、稳定的 validation replay，记录不同容量/阶段对同一 bucket 的改善，而非只导出一次 per-sample loss。

## 下次只看这些

1. 难度依赖模型容量；高 loss 不等于有价值。
2. hard-but-learnable 的标志是归一化 loss 高但持续下降。
3. 视频选数要把学习曲线与运动、语义、稀有度和质量门控结合。
