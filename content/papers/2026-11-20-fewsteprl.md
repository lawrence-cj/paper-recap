---
title: "FewStepRL: Fine-tuning Few-step Diffusion Models with Reinforcement Learning"
paper_url: "https://arxiv.org/abs/2611.08240"
authors: "Haitao Lin, Minsi Ren, Yang Li, Xingtong Ge, Zheyuan Hu, Peiyan Hu, Tailin Wu"
venue: "arXiv technical report v1"
published: "2026"
read_date: "2026-11-20"
read_at: "2026-11-20T12:11:42+08:00"
status: "已读"
tags: ["Diffusion Models", "Knowledge Distillation", "Training & Optimization", "Video Generation"]
search_terms: ["FewStepRL", "GRPO", "AWM", "DiffusionNFT", "MeanFlowNFT", "DMDNFT", "OPSD", "ReFL", "Reinforcement Learning", "AnyFlow", "Average Velocity", "Reward Hacking"]
one_liner: "把已有奖励优化目标统一拆成参考回归与奖励推动，再为平均速度少步模型构造相应预测；few-step 后做 RL 已有先例，本文贡献是系统适配与比较。"
---

## 研究问题

已经有一个能四步生成的 checkpoint，如何继续用 GRPO、NFT 等奖励优化方法提高质量，同时保留少步采样能力？

许多现有目标依赖瞬时速度或多步采样器的转移概率；MeanFlow / AnyFlow 一类模型输出区间平均速度。直接替换网络输出，会混淆两种速度的含义，也留下参考预测如何计算、哪些区间接受训练等问题。

阅读依据：[arXiv v1 原文，2026-11-05](https://arxiv.org/html/2611.08240v1)，重点为 §2、§4–6 与表 6–8。本文实验以 AnyFlow 蒸馏后的 SD3.5-Medium 和 Wan2.1-T2V-1.3B 为起点。

## 核心方法

### 贡献定位：统一分析、少步适配、共同实现下比较

这是一套构造和分析框架，并非首次提出在 few-step 模型上做 RL，也不是一个完全独立于 GRPO / NFT 的新策略优化原则。作者将几类已有目标放到相同预测空间，拆出共同梯度结构，再构造各自需要的少步预测和参考量。

![FewStepRL 框架图：目标转换到预测样本空间，梯度拆成参考回归和奖励推动，再区分少步轨迹端点与瞬时切线预测](media/fewsteprl/method-overview.png "论文 Figure 2：统一变换、梯度拆分与少步模型适配。来源：Haitao Lin et al., FewStepRL, arXiv:2611.08240v1，CC BY 4.0；原图未改。")

图源：[论文 HTML 原始图片](https://arxiv.org/html/2611.08240v1/fig_overview.png)。论文 HTML 标注 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)；图片随论文许可转载。

### 1. 区分瞬时速度与平均速度

令 $t=1$ 为噪声、$t=0$ 为数据。普通 flow 模型预测瞬时速度 $v$，少步模型预测沿轨迹的平均速度 $u$：

$$
u(x_t,t,r)=\frac{1}{t-r}\int_r^t v(x_\nu,\nu)\,d\nu,
\qquad x_r=x_t-(t-r)u(x_t,t,r).
$$

平均速度对应两个端点之间的割线，瞬时速度对应当前点的切线。上式是理想 flow map 的关系；训练网络只能近似它。policy 就是少步生成网络 $u_\theta$，不需要另接一个动作预测头。

### 2. 转到预测样本空间并拆分梯度

定义 $X[x_t,v,r]=x_t-(t-r)v$，则速度误差可换成预测点误差：

$$
\|v_a-v_b\|^2=\frac{\|X[x_t,v_a,r]-X[x_t,v_b,r]\|^2}{(t-r)^2}.
$$

这里的 $x$ 通常仍是 VAE latent，不是 RGB。省略样本下标，将当前预测记为 $y_\theta$，所讨论目标的梯度可整理为：

$$
\nabla_\theta L
=2\left(\frac{\partial y_\theta}{\partial\theta}\right)^\top
\left[W_{\rm reg}(y_\theta-y_{\rm reg})
-W_{\rm rwd}\,\omega(y_{\rm rwd}-y_{\rm base})\right].
$$

$y_{\rm reg}$ 是保持接近的参考预测；$y_{\rm rwd}$ 是奖励指向的点，常取已打分的 rollout 样本；$y_{\rm base}$ 是奖励方向的起点；$\omega$ 是有符号系数。不同方法选择不同的点和权重，部分方法 $W_{\rm reg}=0$。

令 $g=\omega(y_{\rm rwd}-y_{\rm base})$，奖励项可用停止梯度的目标实现：

$$
L_{\rm rwd}=W_{\rm rwd}\left\|y_\theta-\operatorname{sg}(y_\theta+g)\right\|^2.
$$

统一的是梯度结构，不表示各算法完全等价，也不表示原始 loss 数值相同。

### 3. 为少步模型构造这些预测量

- **直接预测端点（u-form）**：查询 $u_\theta(\tilde x_t,t,0)$，得到干净端点预测。只训练这种形式主要覆盖 $r=0$。
- **从自身估计瞬时速度（MeanFlow / v-ind）**：利用 $v=u+(t-r)du/dt$。实现用旧策略估计停止梯度的修正量 $J=(t-r)\operatorname{sg}(du_{\rm old}/dt)$，当前预测使用 $u_\theta+J$，相关参考预测共享同一个 $J$。中心方向差分每次查询最多增加两次旧模型前向；它是数值估计。
- **辅助网络估计瞬时速度（DMD / v-aux）**：在当前生成样本上训练 $v_\phi$，拟合当前生成分布的瞬时速度，构造奖励基线。省去差分，却增加第二个网络及优化器；实验每轮拟合 5 步。DMDNFT 的名称不等于 RL 阶段仍在完整执行原来的教师蒸馏目标。

### 4. 各种适配与训练流程

| 方法 | 主要区别 |
| --- | --- |
| FewStepGRPO | 随机 rollout 并保存转移，用回放评分构造 clipped surrogate；论文明确将其称为 surrogate 适配 |
| MeanFlowAWM / DMDAWM | advantage 调节匹配生成样本的方向；分别使用自身估计与辅助速度基线，无显式自回归项 |
| MeanFlowNFT / DMDNFT | 保留 NFT 的正负隐式策略分支，拆成参考回归和奖励推动；两种瞬时速度估计方式 |
| FewStepOPSD | 沿可微奖励梯度构造上升、下降目标点，再用回归更新策略 |
| ReFL | 用可微奖励梯度更新去噪预测，是重要的简单对照 |

主体流程为：旧策略少步 rollout → 打分与 advantage / 奖励梯度 → 重新加噪并抽取 $(t,r)$ → 构造预测、参考量和权重 → 更新策略 → 刷新旧策略 EMA。GRPO 使用保存的转移，区别于主要的重新加噪训练流程。多数更新不反传整条 rollout；OPSD / ReFL 还需通过奖励模型和 VAE decoder 求梯度。

训练同时覆盖 $r=t$、$r=0$ 及中间区间，避免只更新一种跳跃。实验采用四步生成、attention 上 rank-32 LoRA，并有相对固定初始少步模型的速度惩罚。不能把论文称为 KL 的速度平方误差一概视为精确分布 KL。

## 关键发现

- **图像可微奖励下，ReFL 已很强。**混合奖励实验中，GenEval2 从初始 0.363 提升至第 400 轮的 MeanFlowNFT 0.407、DMDNFT 0.412、FewStepOPSD 0.463、ReFL 0.452。ReFL 最佳 checkpoint 为第 200 轮的 0.465；与 OPSD 的 0.463 差距小于论文所述约 0.03 的评估分辨能力，不能据此判优。
- **不可微奖励也能优化，但有质量取舍。**OCR 实验初始准确率 0.498，MeanFlowNFT 第 100 轮达到 0.850，同时 GenEval2 从 0.363 降至 0.306。奖励提高不能直接当成综合质量提高。
- **视频没有显示明确的综合收益。**DMDNFT 训练奖励从 16.54 提至 17.93，独立 VBench overall 从 83.68 降至 83.38。七种方法的 VBench overall 为 83.38–83.70，多数接近初始化；缺少支持稳定综合改善的证据。
- **成本排序随模态改变。**辅助网络路线在图像中每轮更快，在视频中反而更慢；必须比较总训练成本和独立评测，不能只数少步 rollout 的 NFE。
- **增加推理步数没有稳定的整体收益。**将四步训练后的图像模型改成 8、16、32 步采样，部分样例更好，但聚合指标不支持系统性提升。

数值与结论来源：原文表 6–8、§6.3–6.5。本次未复现训练；以上是作者报告的实验。

## 我的提问

### Q1：核心创新到底是什么？主要就是各算法的统一吗？

主要是“统一分析 + 系统适配 + 共同实现下比较”。统一分析说明每种方法依赖哪些预测和参考量；适配部分再把它们构造为少步模型可计算的量。因此不仅是整理公式，但也不是一种全新的策略优化原则。

### Q2：以前有没有在已经 few-step 的模型上做 RL 的先例？

有，而且是直接相关的前作：

- [TDM-R1，2026-03](https://arxiv.org/abs/2603.07700)：在少步模型上用不可微奖励做 RL，将代理奖励学习与生成器更新分开。
- [Flow-Map GRPO，2026-07](https://arxiv.org/abs/2607.00535)：为已有确定性少步 flow-map 模型构造随机过程，以支持 GRPO 后训练。
- [MeanFlowNFT，2026-07](https://arxiv.org/abs/2607.15273)：已利用 MeanFlow 恒等式构造瞬时速度预测，将 DiffusionNFT 适配到平均速度模型。

所以“少步模型后做 RL”，以及“用恒等式连接平均与瞬时速度”，都不能记成 FewStepRL 首创。FewStepRL §2 也明确承认这些前作；其增量是将单一目标的适配扩展为多目标的共同构造和比较。

### Q3：能否理解成，在已有 few-step checkpoint 上更好地适配现有 RL 算法？

这是本次讨论确认的主要理解。更准确地说：已有 MeanFlow / AnyFlow 一类平均速度少步 checkpoint，想继续优化奖励时，这篇提供了适配几类现有目标的方法。本文也确实从 AnyFlow 蒸馏好的 checkpoint 开始 RL。

“更好适配”指目标构造更合理，且部分实验优于直接把平均速度塞进原始公式；不保证任意少步 checkpoint 开箱即用，也不保证所有任务的综合质量更好。

## 局限与疑问

- 核心讨论针对平均速度 / flow-map 参数化。任意 consistency、对抗蒸馏或固定步数 checkpoint 是否支持所需查询，必须另查。
- 自身估计与辅助网络都存在近似误差；理论恒等式不等于有限规模网络能精确恢复瞬时速度。
- 视频中只有 HPSv3 部分可微；OPSD / ReFL 的奖励梯度只覆盖 81 帧中的 21 帧，并采用逐 latent 帧解码。它们使用 48 张 H20、每轮 3072 个 rollout，其他方法使用 32 张、2048 个 rollout，不是相同样本预算。
- 视频奖励梯度还受精度和解码实现影响，不能把图像上可微奖励方法占优直接推广到视频。
- 训练奖励上涨可能伴随独立指标下降，需用 held-out 评测和早停判断收益。图像推理增加步数的观察也不能作为普遍 scaling 保证。

## 我的判断

用户在追问后形成并确认的理解：这篇适合看作已有 few-step checkpoint 的 RL 适配框架；few-step 后再做 RL 已有先例，本文价值主要在系统地整理、扩展和比较。

Agent 的阅读判断：值得借鉴的是预测量的正确构造、停止梯度的位置，以及同一实现下的成本与质量比较。图像上简单 ReFL 值得作为对照；视频上还不足以据此选择一个稳定提高综合质量的赢家。用户尚未决定复现。

## 下次只看这些

1. 已有 few-step 模型 → 奖励后训练；重点是平均速度 $u$ 与 RL 需要的瞬时速度 $v$ 不同，需正确构造预测和参考量。
2. 前作 MeanFlowNFT / Flow-Map GRPO / TDM-R1 已做少步 RL。FewStepRL 的定位是统一分析、扩展适配和系统比较。
3. 图像 ReFL 是强对照；视频 reward 上涨不等于 VBench 上涨，不能把适配成功当成综合质量改善。
