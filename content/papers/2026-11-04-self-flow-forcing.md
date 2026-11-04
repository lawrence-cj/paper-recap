---
title: "Beyond Distribution Matching: Self-Supervised Representation Forcing for Few-Step Video Generation"
paper_url: "https://arxiv.org/abs/2610.35594"
authors: "Junyi Chen et al."
venue: "NeurIPS 2026"
published: "2026"
read_date: "2026-11-04"
read_at: "2026-11-04T16:20:54+08:00"
status: "已读"
tags: ["Video Generation", "Diffusion Models", "Knowledge Distillation", "Autoregressive Generation", "Training & Optimization"]
search_terms: ["SF2", "SF²", "Self Flow Forcing", "Self-Flow", "DMD", "EMA", "feature teacher", "representation alignment", "四个模型", "混合噪声"]
one_liner: "在四步 DMD 上增加学生的 EMA 特征教师，让混合噪声输入的浅层预测较干净输入的深层特征；两项 loss 分别构造输入，训练共四个主要模型，推理只用学生。"
---

## 研究问题

DMD（Distribution Matching Distillation，分布匹配蒸馏）让少步学生的生成分布接近多步教师。作者观察到，步数压缩后，文字、小物体和复杂几何结构容易退化，视频也会出现肢体跳变、物体消失和闪烁。因此提出：保留 DMD，再直接监督学生的中间特征，能否改善空间细节和运动连贯性？

这里的“缺少特征监督”指缺少显式的特征目标。原有 DMD 的梯度仍然会更新学生的中间层；不能理解成这些层原本没有训练。

阅读依据：[论文 v1 §3–5、附录 A–F](https://arxiv.org/html/2610.35594v1)、[官方项目页](https://sotamak1r.github.io/SF2-web/)、[官方仓库](https://github.com/SOTAMak1r/SF2)。arXiv 标注已被 NeurIPS 2026 接收。本次核对原文并整理问答，没有进行训练或推理复现；截至 2026-11-04，仓库 README 仍标注代码整理中、coming soon。

## 核心方法

![SF² 方法总览：复用学生生成轨迹，最终结果计算 DMD，两个中间状态经空间 mask 构造学生输入，较干净状态送入 EMA 特征教师](media/self-flow-forcing/method-overview.webp "论文 Figure 1：Self Flow Forcing 方法总览。来源：Junyi Chen 等，arXiv:2610.35594v1；CC BY 4.0；原图无损转换为 WebP，未改动图示内容。")

图源：[论文 Figure 1 原图](https://arxiv.org/html/2610.35594v1/pipeline.png)，许可：[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。图中的层标记与噪声时间步是两套概念；下文用层号 9、21 解释特征监督。

### 四个主要模型，两种教师

SF² 在 DMD 常见的三个模型实例之外，增加一个与学生架构相同、参数独立维护的 EMA 特征教师。MLP 投影头是额外的小模块，不计作完整生成模型。

| 模型 | 用途 | 更新方式 |
| --- | --- | --- |
| 学生生成器 $G_\theta$ | 生成视频，并预测目标特征 | DMD loss 与 feature loss 联合更新 |
| Real-score 教师 $s_{\mathrm{real}}$ | 提供目标分布的 score | 冻结 |
| Fake-score 网络 $s_\psi$ | 估计学生当前生成分布的 score | 独立用去噪 loss 训练 |
| 特征教师 $G_{\bar\theta}$ | 提供较干净输入的深层特征 | 对学生参数做 EMA，不接收反传梯度 |

Wan 配置中，学生、fake-score 和 EMA 教师为 1.3B，冻结 real-score 使用 Wan2.1-14B。四个模型不代表四倍相同模型的参数量，也不代表四次推理调用；推理只需要学生。

### 正常生成与 DMD 分支

本文双向和因果实验均使用 $K=4$ 步学生。同一个学生网络在不同噪声时间步上调用，生成最终结果 $\hat x_0$，并记录中间状态。论文式 5 的每一步先预测干净 latent，再构造下一时间步的带噪输入：

$$
\hat x_0^{(k)}=G_\theta(x_{t_k},t_k),\qquad
x_{t_{k-1}}=\alpha_{t_{k-1}}\hat x_0^{(k)}+\sigma_{t_{k-1}}\epsilon^{(k)}.
$$

标准 DMD 对最终生成结果重新加噪的操作保留。另采样一个用于 score 评估的时间步 $\tau$ 和高斯噪声 $\epsilon$：

$$
x_\tau^{\mathrm{DMD}}=\alpha_\tau\hat x_0+\sigma_\tau\epsilon.
$$

同一个带噪样本送入 real-score 和 fake-score 网络，由两者的 score 差向学生提供 DMD 更新信号。这里的 $\tau$ 与后面从生成轨迹中选出的 $t,s$ 用途不同。

### 特征分支：混合输入，再前向提取特征

从已经记录的生成轨迹中选两个不同状态，$t>s$，其中 $x_t$ 噪声较多，$x_s$ 噪声较少。二者对应同一段视频的相同 token 位置。用二值空间 mask $M$ 构造输入：

$$
\tilde x^{\mathrm{stu}}=M\odot x_s+(1-M)\odot x_t,\qquad
\tilde x^{\mathrm{tea}}=x_s.
$$

- $M=1$ 的位置取较低噪声状态，$M=0$ 的位置取较高噪声状态。token 都保留，也参与正常 attention。
- 将混合输入重新送入学生，将较干净状态送入 EMA 特征教师，再提取各自的 hidden states。不能把这里理解成直接拿原来普通采样前向的 feature 做比较。
- 默认所有帧共享同一张空间 mask，减少从相邻帧同一位置复制较清楚内容的捷径。mask 不改变因果 attention 的可见范围。
- 这条分支复用轨迹中的带噪状态，不使用 DMD 对最终结果重新加噪得到的 $x_\tau^{\mathrm{DMD}}$，也不在后者上叠加一次 mask 加噪。

### 浅层学生预测深层教师

以 Wan 的默认配置为例，提取学生第 9 层和 EMA 教师第 21 层的特征，形状均为 $[B,N,D]$；$N$ 为视频 token 数，$D$ 为特征维度。学生特征先经过三层 MLP 投影头 $P$，隐藏宽度为 $4D$，输入输出宽度均为 $D$。

$$
p_{b,n}=P\!\left(h_{\theta,b,n}^{(9)}(\tilde x^{\mathrm{stu}})\right),\qquad
y_{b,n}=\operatorname{sg}\!\left[h_{\bar\theta,b,n}^{(21)}(x_s)\right].
$$

$\operatorname{sg}$ 表示停止梯度。对相同位置的 token 计算余弦距离：

$$
\mathcal L_{\mathrm{rep}}=\frac{1}{BN}\sum_{b=1}^{B}\sum_{n=1}^{N}
\left[1-\frac{p_{b,n}^{\top}y_{b,n}}{\lVert p_{b,n}\rVert_2\lVert y_{b,n}\rVert_2}\right].
$$

按式 8，损失在全部视频 token 上平均，包含输入中较低和较高噪声的位置。mask 负责构造输入难度，并未将损失限制到部分位置。目标是中间特征，不是像素、latent 重建值或速度预测。

联合训练时，学生和投影头使用：

$$
\mathcal L_G=\mathcal L_{\mathrm{DMD}}+\lambda_{\mathrm{rep}}\mathcal L_{\mathrm{rep}},\qquad
\lambda_{\mathrm{rep}}=0.5.
$$

EMA 教师参数按下式更新，$\mu=0.99$：

$$
\bar\theta\leftarrow\mu\bar\theta+(1-\mu)\theta.
$$

开始的 100 步关闭 DMD、固定骨干，仅训练投影头；预热后开启联合目标与 EMA 更新。因果模型在特征分支仍逐 chunk 前向，学生、教师维护各自独立的 KV cache，最后才拼接各 chunk 特征计算损失。

## 关键发现

主表中的蒸馏模型使用相同 Wan2.1-T2V-1.3B 骨干、四步生成和 $848\times480$ 分辨率。来源：[论文 Table 2](https://arxiv.org/html/2610.35594v1#S5.SS4)。

| 设置 | 方法 | VBench 总分 ↑ | VisionReward ↑ |
| --- | --- | --- | --- |
| 双向 | CausVid | 83.19 | 5.310 |
| 双向 | SF² | 84.15 | 7.060 |
| 因果 | Self-Forcing | 83.74 | 5.820 |
| 因果 | Causal-Forcing | 84.04 | 6.326 |
| 因果 | SF² | 84.15 | 7.523 |

- 作者报告各组内相同推理吞吐与延迟。相对 Causal-Forcing，因果 VBench 总分只增加 0.11；VisionReward 的改善更明显。后者在 100 条运动丰富的提示词上计算，是自动偏好模型分数，不是人类评测胜率。
- 时间共享的空间 mask 比逐 token 独立采样更有利于保留运动；独立采样的背景一致性更高，但 Dynamic Degree 更低。作者将其解释为沿时间复制的捷径，不能只看背景一致性判断方法好坏。
- 在 HunyuanVideo-1.5 的 120 条静态场景加相机运动测试中，3D 重建后重渲染的 PSNR 为 20.64 → 21.24，Dynamic Degree 均为 90.00。这支持所测场景的几何一致性改善，不能据此证明复杂动态物理已解决。
- 新增特征分支占完整训练周期的 6.2%；该周期包含一次生成器更新和五次 critic 更新。这个比例的分母是加入 SF² 后的总周期，不能直接当作所有配置相对 DMD 的固定加时比例，也不等于显存开销。推理不保留投影头或额外教师。

## 我的提问

### Q1：就是监督中间 feature 再加 mask？和 Black Forest Labs 的 Self-Flow 是同一个思路吗？

是。顺序是先用 mask 构造高低噪声混合输入，再让学生浅层预测 EMA 教师较干净输入的深层特征。SF² §4.1 明确引用了 [BFL Self-Flow](https://arxiv.org/html/2603.06507v1#S3.SS4)。Self-Flow 在真实训练数据上做 Flow Matching 与特征预测；SF² 把这个设计接到 DMD 少步蒸馏，用学生自身生成轨迹提供状态，并适配因果 KV cache。

### Q2：到底监督哪些 feature？mask 是否也用来筛选 loss？

Wan 默认是学生第 9 层经 MLP，逐位置匹配 EMA 教师第 21 层，使用余弦距离并对所有视频 token 平均。mask 用在输入状态选择，不是 attention mask，也不是只计算被遮挡位置的 loss。就本次特征前向的直接计算路径而言，梯度经过 MLP 回到学生第 9 层及之前；原有 DMD 继续通过生成输出训练学生。教师分支停止梯度。

### Q3：学生输入为什么有高低噪声？student 不是从纯噪声一步生成吗？

本文实验是四步学生，普通生成轨迹中已有多个不同噪声程度的输入状态。特征分支从中取两个状态混合，再额外做前向；正常推理不需要人为准备混合噪声输入。

若目标是严格一步 $G(z)\to\hat x_0$，轨迹只有一个采样输入，无法原样执行文中的“选两个不同轨迹时间步”。可以设想对生成结果重新加噪来构造特征输入，但模型也必须能处理这些中间噪声状态；这是待验证的改法，不是本文已经验证的一步方案。

### Q4：正常 DMD 的教师输入也会加噪，这里还加吗？两项 loss 是各自准备输入吗？

原有 DMD 的加噪保留：最终结果 $\hat x_0$ 重新加噪后，同一个带噪样本送入 real/fake-score 网络。特征分支则直接取生成轨迹中的 $x_t,x_s$，学生接收混合输入，EMA 教师接收 $x_s$。

两项 loss 分别构造输入、共享同一次正常生成轨迹，最后联合更新学生。“各自准备输入”不代表每个网络都独立采一份噪声：DMD 的 real/fake-score 接收同一个带噪样本，特征分支取出的状态本身已带噪。

### Q5：所以是 DMD 原来的三个模型，再加一个单独的 feature teacher，共四个模型？

是，这是本轮最后确认的理解：学生生成器、冻结 real-score 教师、可训练 fake-score 网络，以及学生的 EMA 特征教师。新增教师与学生架构相同，维护单独参数；另有小 MLP 投影头。两个教师作用不同。训练维护四个主要模型实例，推理仅使用学生。

## 局限与疑问

- v1 正文 §4.1 写学生按 token 使用对应的 $t/s$ 时间步条件，附录 B 却写混合输入统一使用 $t$。时间步条件的实际实现待代码确认。
- 算法给出了混合输入的学生前向和 EMA 教师前向。作者将额外成本概括为单次教师前向，实际复现还需核对学生重算、梯度保留与 KV cache 的成本，不能把所有额外工作都理解成免费复用。
- 因果学生从 Causal-Forcing 的 ODE 回归检查点初始化，实验收益包含已有初始化流程；不能据此宣称不再需要 ODE 初始化。
- 主要实验验证四步学生；严格一步训练的状态构造、时间步支持和有效性仍待验证。
- 特征监督有效不等于已经证明 DMD 视频退化只有一个根因，也不保证获得多步教师本身缺少的概念或物理能力。当前公开代码尚不足以进行实现核对。

## 我的判断

已确认的理解：这是在 DMD 输出分布监督之外，增加一条 Self-Flow 式的自监督特征训练分支；两个 loss 的输入构造不同，两个教师也不同，训练需要四个主要模型实例。

是否复现、是否接入严格一步学生：待决定。若继续评估，先核对代码中的时间步条件、混合输入前向和开销，再判断对一步生成器的适用性；论文中的四步结果不能直接充当一步效果证据。

## 下次只看这些

1. 四个模型：学生、real-score、fake-score、学生的 EMA 特征教师；推理只用学生。
2. DMD 对最终结果重新加噪；feature loss 用两个轨迹状态混合，让学生浅层预测 EMA 教师深层，按所有视频 token 计算余弦距离。
3. 论文验证的是四步学生；先看 Figure 1、§4 和附录 A–B，再核对一步适配与时间步条件的待确认点。
