---
title: "Uncertainty DMD: Restoring Diversity in Few-Step Autoregressive Video Distillation"
paper_url: "https://arxiv.org/abs/2609.11265"
authors: "Zixuan Duan et al."
venue: "arXiv"
published: "2026"
read_date: "2026-09-17"
read_at: "2026-09-17T16:16:14+08:00"
status: "已读"
tags: ["Video Generation", "Diffusion Models", "Few-Step Sampling", "Knowledge Distillation"]
one_liner: "只扰动首个 chunk 第一次去噪的 timestep，并在历史 latent 写入 KV 前加噪；缓存写入后仍固定，训练和推理都采用这两项操作。"
---

## 研究问题

DMD 少步蒸馏后的自回归视频生成，可能出现同一 prompt 下不同 seed 的结果相似、运动减少的问题。作者将其归因于首个 chunk 的多样性损失，以及历史缓存把相似条件传给后续 chunk。本文关注的是固定 prompt 下的样本差异和时间变化。

阅读依据：[arXiv v1 原文](https://arxiv.org/html/2609.11265v1)，重点为 §3–4、Algorithm 1 和 Table 2–3。以下问答整理自本次阅读讨论，区分论文操作、实验结果与解释。

## 核心方法

### 首个 chunk 的第一次去噪：只改时间条件

保持实际输入的初始噪声 latent 不变，将第一次调用的时间条件改为：

$$
\tilde T=T-\Delta,\qquad \Delta\sim\mathcal U(0,100).
$$

$T$ 是原始起始 timestep，$\Delta$ 是随机偏移，结果限制在合法范围。不是先把 latent 去噪到 $\tilde T$，也不是直接跳过一个采样步骤。改变时间条件会改变模型的首次输出，从而影响开头的构图、姿态等。

只有首个 chunk 的第一步受这个操作影响；首个 chunk 的其余步骤和后续 chunk 都沿用原本预设的去噪 schedule。“固定 schedule”不等于每一步使用相同的 timestep。

### 写入历史缓存前：扰动 latent，再计算 KV

当前 chunk 得到 clean prediction $\hat x_j$ 后，对用于构建缓存的副本按概率加噪：

$$
\bar x_j=(1-m_j)\hat x_j+m_j(\alpha_{\tau_c}\hat x_j+\sigma_{\tau_c}\epsilon_j),
\qquad m_j\sim\mathrm{Bernoulli}(p_c),\quad \epsilon_j\sim\mathcal N(0,I).
$$

$\tau_c$ 决定噪声强度，$p_c$ 是是否加噪的概率。这里固定的是噪声水平参数，随机的是高斯噪声内容以及加不加；并未描述每个 chunk 再随机采样一个噪声强度。

Algorithm 1 用下面的前向计算更新缓存：

$$
\mathrm{KV}\leftarrow\mathrm{KV}\cup G_\theta^{\mathrm{KV}}(\bar x_j,0,\mathrm{KV}).
$$

带噪 latent 经过网络产生各层 K、V，并被追加到历史缓存。写入时的 timestep 是 0；没有额外迭代把历史 latent 去噪干净再缓存。缓存条目写入后保持固定，后续 chunk 的去噪通过 attention 使用它。

### 训练与推理采用同样的扰动规则

训练初期关闭两项扰动，warm-up 后逐渐增加启用概率：

$$
\gamma_\ell=\mathrm{clip}\left(\frac{\ell-S_w}{R},0,1\right),\qquad
p_{\mathrm{root}}=\gamma_\ell,\quad p_c=\gamma_\ell p_c^{\max}.
$$

$\ell$ 为训练迭代，$S_w$ 为 warm-up 长度，$R$ 为概率递增阶段长度。推理使用最终概率，第一项始终启用，缓存按最终概率扰动；仍用原有 DMD 目标训练。

![Uncertainty DMD 方法图：首步时间扰动、缓存写入扰动与 DMD 训练](media/uncertainty-dmd/method-overview.webp "论文 Figure 5，来源：Zixuan Duan et al., Uncertainty DMD, arXiv:2609.11265v1，CC BY 4.0；原图转为 WebP，内容未改。")

图源：[Figure 5](https://arxiv.org/html/2609.11265v1#S3.F5)；[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。操作细节以正文公式及 Algorithm 1 为准。

## 关键发现

- 教师/学生替换实验显示：只让教师生成首个 chunk，后续由学生生成，能恢复大部分多样性；只替换后续采样器效果有限。这支持首段是关键瓶颈。
- Table 2 的 Self-Forcing 设置下，VENDI-DINO 从 2.6 到 3.3，Dynamic Degree 从 25.1 到 42.4；部分美学和一致性指标略降，不能说所有质量维度都提升。
- Table 3 去掉缓存扰动后，VENDI-DINO 从 3.3 到 3.2，Dynamic Degree 从 42.4 到 37.0。该消融中缓存扰动主要帮助运动，首步 timestep 扰动对跨样本差异更关键。

## 我的提问

### Q1：加噪是随机大小，还是固定大小？

按正文公式，噪声水平 $\tau_c$ 是给定参数；每次随机采样噪声内容 $\epsilon_j$，再由 $m_j$ 决定是否启用。所谓固定强度指固定分布尺度，并不要求每份高斯样本的实际范数完全相等。训练逐渐提高的是启用概率。

### Q2：带噪 latent 变成 KV 时还有 denoise 吗？

有模型前向，用来计算各层 K、V；没有额外的历史 chunk 去噪采样。后续使用的是带噪输入编码出的历史 KV。它包含输入噪声的影响，但不等于直接给干净 KV 加一份高斯噪声。

### Q3：KV 写入后还是固定的，那和原来有什么区别？

这是本次讨论最重要的纠正。不能把方法解释成“后续去噪时缓存一直变化”。对给定历史 $x$，原方法产生 $C=f(x)$；加入扰动后，可以产生 $C_\epsilon=f(\alpha x+\sigma\epsilon)$。每次写入后的条目仍固定，但同一历史在不同随机采样下可以对应不同缓存。

因此改变的是缓存取值和训练时的历史条件分布，不是缓存是否固定。初始生成噪声本来也随机；作者的动机是蒸馏模型对这些原有噪声的响应变弱，因而尝试在历史条件上增加另一种扰动。它是否有效需要实验，不能由“加了随机量”直接推出。

### Q4：同一个开头，后续也可能变化？后面的 scene 原来是固定的吗？

同一个 clean 起始 chunk 可以因缓存扰动不同，得到不同的后续条件，因此后续有机会产生差异。但后续场景从未被数学上锁死，原方法也有当前 chunk 的生成噪声。历史只是强烈约束场景、人物和构图；目标是在连贯的前提下增加变化，并非让每段随机换场景。

### Q5：为什么训练时也要加？

训练和推理采用同样规则，让学生在带扰动的历史条件下学习继续生成，避免只在推理时突然改变条件。“减弱对确定历史的过度依赖”可以作为直观解释，不能替代对具体作用机制的验证。

### Q6：首步 timestep 随机到底是什么意思？后面的 timestep 呢？

同一份实际输入 $z$，原来调用 $G_\theta(z,T,c)$，现在调用 $G_\theta(z,T-\Delta,c)$。例如 $T=1000$ 时，条件可落在 900–1000，实际 latent 并未预先变干净。时间条件不同，输出就可能不同。

只随机扰动首个 chunk 的第一次调用。后面的 timestep 仍按原来的 schedule 走。缓存扰动是另一项独立操作，不要把两者混为每一步都随机改时间。

## 局限与疑问

- 运动指标上升不等于动作更合理，也不证明完整恢复教师分布；复杂运动的合理性仍有局限。
- 缓存加噪可能被模型忽略，也可能损害连贯性。已有消融支持所测配置的收益，不构成普遍保证。
- 本次未检查实现代码；具体 $\tau_c$、最终概率和 schedule 数值待补充，不把示例时间值当作完整复现配置。

## 我的判断

保留本次阅读中的质疑：缓存写入后仍固定，因此“固定缓存导致问题”过于粗糙。更准确的理解是，蒸馏后的历史条件相似，而学生对当前噪声的响应弱；作者尝试在首步时间条件和缓存构建处注入变化。

Agent 的阅读判断：教师/学生替换实验和组件消融值得参考；方法收益已有实验支持，但为什么特定扰动能改善运动仍需更细的验证。用户尚未给出是否复现的最终判断。

## 下次只看这些

1. 首个 chunk、第一次 denoise：latent 不变，只让 $t=T-\Delta$；其余步骤沿用原 schedule。
2. clean prediction 的缓存副本加噪，再以 timestep 0 前向计算 KV；不额外去噪，写入后不反复扰动。
3. 同一开头可对应不同缓存，但每份缓存仍固定；不要把它说成动态变化的 KV，也不要把运动增加等同于正确运动。
