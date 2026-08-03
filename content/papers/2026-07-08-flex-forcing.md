---
title: "Flex-Forcing: Towards a Unified Autoregressive and Bidirectional Video Diffusion Model"
paper_url: "https://arxiv.org/abs/2607.03509"
authors: "Xinyin Ma et al."
venue: "ICML 2026"
published: "2026"
read_date: "2026-07-08"
status: "已精读"
rating: 3
tags: ["Video Diffusion", "Autoregressive Generation", "Bidirectional Attention", "Long Video"]
one_liner: "Flex-Forcing 让同一视频 diffusion 权重在任意 chunk 内双向、chunk 间因果，从而把生成质量、流式延迟和长视频记忆变成可调粒度。"
---

## 研究问题

双向 video diffusion 质量高但必须等待完整 clip，自回归模型可流式生成却容易累积误差。固定 chunk size 又把模型锁死在一个质量—延迟点；论文希望用同一组权重覆盖 bidirectional、strict AR 与中间 hybrid 模式。

## 核心方法

- 按视频时间与 denoising 时间对 token 做二维 flexible chunking。
- 同一 chunk 内允许 bidirectional attention，不同 chunk 之间使用 causal attention；改变 partition 即可改变推理模式。
- 训练随机采样 2–10 的 chunk size 和嵌套切分，让模型见过多种因果粒度。
- K-Projection 将缓存的 clean-history key 投影到当前噪声 timestep 的 key space，只调整可复用历史 K，减少 train–test mismatch。
- 从 bidirectional prior 初始化，通过 ODE initialization 与 self-rollout 的 asymmetric DMD/VSD 训练适应 causal generation。

对 partition $\pi(i)$，attention mask 可写成

$$
M_{ij}=\mathbf 1\!\left[\pi(j)\le \pi(i)\right],
$$

因此同一 partition 内全可见，更早 partition 可作为 causal history；一个模型通过改变 $\pi$ 在不同模式间切换。

## 关键发现

- 选定配置报告 **25.8 FPS / Total 85.07**，显示 hybrid 点可在速度与质量间取得较好折中，但不是所有指标都占优。
- 与 Self-Forcing 在相同 NFE 下，一些质量/吞吐指标提升，语义指标在部分设置下降。
- 人类视觉偏好：5 秒 **55.4%**，30 秒 **53.9%**；30 秒 prompt alignment **50.6%**，接近随机，说明长期优势并不压倒性。

## 我的提问

### Q1：它是同时训练多个模型吗？

不是。核心是同一权重在训练时随机看到不同 partition/mask，推理时通过 chunking schedule 选择模式。bidirectional 与 AR 是同一 attention 图的两个极端。

### Q2：最佳 chunk 是否固定？

不是。较大 chunk 通常有更强块内一致性但首帧等待和显存更高；较小 chunk 更流式但因果误差更多。论文最重要的观点是把 causal granularity 变成部署 knob，而不是宣布一个永远最优的 chunk。

## 局限与疑问

- 主要依赖 Wan 1.3B/14B teacher，模型家族与 teacher 偏差有限。
- 论文缺少多 seed 方差；FPS 也不能替代首帧延迟、尾延迟和显存峰值。
- chunk/search 配置很多，选择最佳点可能带来评测调参偏差。
- 编辑能力多为定性；灵活 mask 缓解但没有消除 self-rollout mismatch。

## 我的判断

它最有价值的是统一接口和部署弹性：同一 checkpoint 可按产品需求选择离线高质量或流式生成。复现时应画完整 Pareto frontier，至少同时报告 TTFF、每 chunk latency、总时长、显存和长期一致性。

## 下次只看这些

1. chunk 内双向、chunk 间因果；partition 决定推理模式。
2. K-Projection 只把缓存 clean K 映射到当前 timestep。
3. 灵活因果粒度是可调旋钮，不存在论文证明的单一全局最优 chunk。
