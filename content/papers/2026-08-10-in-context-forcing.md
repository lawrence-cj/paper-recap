---
title: "In-Context Forcing: Uncovering Context Effects in Autoregressive Video Diffusion"
paper_url: "https://arxiv.org/abs/2608.05237"
authors: "Lingxiao Yang et al."
venue: "arXiv"
published: "2026"
read_date: "2026-08-10"
read_at: "2026-08-10T17:48:44+08:00"
status: "已精读"
tags: ["Video Diffusion", "Autoregressive Generation", "Few-Step Sampling", "Exposure Bias", "Acceleration"]
one_liner: "ICF 让相邻历史 chunk 更 noisy、远处历史逐级更 clean，在 4-step self-rollout 中抑制 clean-context copy shortcut，并把同一 staircase schedule 转为跨 chunk 并行去噪。"
---

## 研究问题

Self Forcing 类 few-step autoregressive video diffusion 在训练和推理时都使用已经完全去噪的历史 chunk。虽然解决了 GT-context exposure bias，但 clean history 在当前 chunk 的早期 denoising 阶段泄漏过多局部纹理与空间对应，容易诱导模型直接复制上一帧，得到清晰但静止、重复的运动；严格等待上一 chunk 完全 clean 也阻断了跨 chunk 并行。

ICF 要解决的是：历史 context 应该处于什么 diffusion state，才能同时提供时间一致性、避免细节 shortcut，并让 few-step AR denoising 形成可并行的 wavefront。

## 核心方法

- **Progressive context schedule。** 对当前 chunk $i$ 的第 $j$ 个 denoising stage，历史不是统一 clean，也不是全部与当前同噪声，而是：

$$
p(x^i_{t_j}\mid x^{<i}_{0:T})
=p\!\left(
x^i_{t_j}
\mid x^{i-1}_{t_{j-1}},x^{i-2}_{t_{j-2}},\ldots,x^{i-k}_{t_{j-k}},\ldots
\right),
$$

  当 $j-k<0$ 时将 timestep clip 到 $t_0$。因此最近的 history 更 noisy，距离越远越 clean；早期去噪只从近邻获得 coarse semantic/motion guidance，后期才逐步获得细纹理。生成第三个 chunk 时，第二个 chunk 通常仍在较新的 noisy stage，第一个 chunk 已多走一步、更接近 clean，而不是两者都固定为 $X_0$。

- **Step-wise Rolling KV Cache。** 每个 denoising timestep 对应一个 KV cache。完成一个 chunk 的某一步后，把相应 KV 写入该层 cache，再做 bottom-up rolling update，使下一个 chunk 恰好读到上述错位的 noise ladder。前序 rollout/cache stop-gradient，可 offload CPU；论文称不开 offload 时多级 cache 也比标准训练额外增加不到 30% memory。

- **训练时统一 exit stage，避免 mixed-quality DMD output。** 对整段 sequence 只采样一次：

$$
s\sim\operatorname{Uniform}\{1,\ldots,T\}.
$$

  每个 chunk 都从 $t_T$ rollout 到同一个 $t_s$；$t_T\rightarrow t_{s+1}$ 无梯度，在 $t_s$ 的 student call 直接预测 $\hat x_0^i$ 并保留梯度。它不是总把每个 chunk 按剩余 solver stages 完整推到 $t_0$，而是在相同 exit stage 产生 clean-space estimate，使拼接后的 $\hat X_0$ 质量层级一致。随后在另一个独立 DMD timestep $\tau$ 重新加噪：

$$
x_\tau=\alpha_\tau\hat X_0+\sigma_\tau\epsilon,
$$

$$
\nabla_\theta\mathcal L_{\mathrm{DMD}}
=-\mathbb E\left[
\bigl(s_{\mathrm{real}}(x_\tau,\tau)-s_{\mathrm{fake}}(x_\tau,\tau)\bigr)
\frac{d\hat X_0}{d\theta}
\right].
$$

  $s$ 决定 self-rollout 在哪个 few-step stage 退出，$\tau$ 决定 DMD critic 在哪个 noise level 比较 real/fake，二者不是同一个 timestep。论文用 Wan2.1-T2V-14B bidirectional model 提供 DMD score，并以 causal Wan2.1-T2V-1.3B 作为 4-step student。

- **Cross-frame causal attention。** Staircase schedule 解除了“前一 chunk 必须完全 clean，后一 chunk 才能开始”的硬依赖。推理时多个 chunk 可以位于不同 denoising stages 并行推进，用 active mask 找出本轮应更新的 chunks；多级训练 cache 可在推理时折叠为一个 unified KV cache，提高 GPU utilization。

## 关键发现

- 实验模型为 1.3B、4-step，每个 autoregressive chunk 联合生成 3 个 latent frames；ODE initialization 和 DMD 只使用文本 prompts，不需要视频数据，但依赖 14B bidirectional teacher。
- 短视频 VBench：ICF 为 **84.34 total / 84.99 quality / 81.77 semantic / 72 dynamic degree**；Self Forcing 为 **83.95/84.67/81.05/63**。主要增益来自 motion 和 semantic，而不是单纯清晰度。
- Chunk-wise 吞吐从 Self Forcing 的 **17.0 FPS** 提升到 **18.4 FPS**，总时间从 4.85 秒降到 4.40 秒；frame-wise 从 **8.9 FPS** 提升到 **16.2 FPS**，总时间减少 45.1%，说明 wavefront 的系统收益在原本更串行的配置中更明显。
- 30 秒 extrapolation 的 dynamic degree 为 **86.40**，高于 Self Forcing 的 **52.51** 和 Rolling Forcing 的 **40.63**；但 30 秒来自 inference-time extrapolation，不等于原生 30 秒训练。
- Progressive schedule 直接套到 Self Forcing 权重、不额外训练时 dynamic degree 从 **0.633** 升到 **0.653**；完整 ICF 训练达到 **0.719**。这支持 schedule 本身有效，也说明训练适配仍贡献了大部分 motion gain。
- ICF 收敛需要约 1900 iterations，而 Self Forcing baseline 为 1500，额外成本主要发生在训练。

## 我的提问

### Q1：in-context length 多长？生成第三个 chunk 时，前两个处于什么状态？

公式上当前 chunk 可读取所有更早 chunk，区别是它们按时间距离映射到不同 noise levels，并在超过 denoising depth 后 clip 到 $t_0$。以第三个 chunk 为例，在当前 stage $t_j$，第二个 chunk 对应 $t_{j-1}$，第一个对应 $t_{j-2}$；越远越 clean。论文没有把核心方法定义成只看固定两个 chunk，但长视频 extrapolation 的具体 retained context/window 配置披露不够完整，记为待补充。

### Q2：训练是不是最长只有 5 秒？

论文明确说 bidirectional base 生成 5 秒 clips，并用其 ODE pairs 初始化；30 秒结果采用 Rolling Forcing 的 inference-time extrapolation，60 秒只给定性样例。但 Algorithm 1 的训练帧数 $N$ 没有在实现细节中明确列出，因此不能仅凭 chunk 数断言 DMD self-simulation 的 exact training horizon；目前更稳妥的记录是“短模型训练、长视频外推，训练 $N$ 待补充”。这与明确原生训练到 15 秒的 Resampling Forcing 不同。

### Q3：5 秒里的哪些 chunk 用于训练，哪些只是推理？

所有 rollout chunk 都参与构造 progressive context；在采样到的统一 exit stage $s$，每个 chunk 都输出一个带梯度的 $\hat x_0^i$，拼成 $\hat X_0$ 参与 DMD。更早 denoising calls 和前序 KV 只用于 self-simulation/context，使用 no-grad；不存在固定“前几个仅推理、最后一个才训练”的划分。

### Q4：统一 timestep 是固定 $t_0$，还是 $t_1$ 也可以？

训练统一的是随机 exit index $s\in\{1,\ldots,T\}$，所以可以是靠近 clean 的 $t_1$，也可以是更早 stage；不是永远固定 $t_0$。每个 chunk 在该 stage 直接预测 $\hat x_0$。DMD 之后又独立采样 $\tau$ 对 $\hat X_0$ re-noise，再交给 real/fake score networks。

### Q5：不同 chunk 使用不同 noise level，是否早已有类似 forcing？

是。Diffusion Forcing 已允许每帧独立 noise level，Rolling/rolling-denoising 方法也使用沿视频时间错位的 noise schedule。因此“不同 chunk 不同噪声”本身不是 ICF 的充分创新。更可信的新意是四件事的组合：针对 clean-context copy shortcut 设计“近邻更 noisy、远处更 clean”的方向性 schedule；让 DMD 输出在同一 exit stage 保持同质；用 step-wise rolling KV 精确复现训练/推理状态；再把该 schedule 变成 cross-frame wavefront parallelism。

### Q6：它与 Resampling Forcing 的区别是什么？

Resampling Forcing 从 GT 的共享 $t_s$ 出发，把每个历史 chunk 自回归 resample 回 $t=0$，得到 clean-looking 但有模型误差的 detached context，再以 per-chunk $t_i$ 做普通 GT flow matching；它无 DMD、正式推理 32 steps，并原生训练到 15 秒。ICF 的 context 则真的停留在不同中间 noise states，使用 14B teacher 做 4-step DMD，并在推理时让多个 chunks 以 staircase schedule 同时去噪。

一句话区分：Resampling Forcing 改变的是“如何制造 realistic imperfect history”；ICF 改变的是“few-step rollout 中 history 应暴露在哪个 diffusion state”。两者理论上可组合：先用 RF 训练长时、teacher-free causal backbone，再训练其接受 ICF progressive states，但现有论文没有验证。

## 局限与疑问

- ICF 依赖 14B bidirectional teacher 和 DMD/fake-score machinery；“不需要视频数据”不等于训练简单或 teacher-free，bidirectional teacher 也可能把短 clip/future-aware bias传给 causal student。
- 精确训练 clip length、Algorithm 1 的 $N$ 和长视频 retained context 配置披露不足；30/60 秒展示不能替代 native long-video training。
- 不同 noise context 与 rolling schedule 不是首次提出，论文对 Diffusion Forcing/Rolling Forcing 有讨论，但当前 v1 没有比较或引用更早的 Resampling Forcing，削弱了 exposure-bias 相关工作的定位。
- 主要证据集中在 Wan2.1 1.3B、4-step 配置；对更多 NFE、模型规模、I2V/interactive control 和其他 causal architectures 的普适性未知。
- Dynamic Degree 增益很大，但长期因果、身份状态和物理一致性不能由 VBench motion 指标充分代表；需要 state-tracking 与人评验证。

## 我的判断

ICF 最值得学习的不是“给不同 chunk 加不同噪声”，而是把 context information budget 与 diffusion coarse-to-fine stage 对齐：早期遮住近邻细节以阻止复制，后期再逐步释放纹理。论文同时把这一建模假设落实成可训练的 KV state machine 和可加速的 inference wavefront，方法—系统闭环很漂亮。

但 novelty 应按组合创新而非新 forcing primitive 来评价。若复现，优先拆开验证三部分：只换 progressive inference schedule、加入 step-wise self-simulation、再加入 DMD finetuning；同时与 RF-style imperfect-$X_0$ context 做同 backbone、同 NFE、同 native length 的 controlled comparison。

## 下次只看这些

1. 当前 stage $t_j$ 读取 $t_{j-1},t_{j-2},\ldots$ 的历史：近邻 noisy、远处 clean，抑制 clean-context copy shortcut。
2. 每段 sequence 共享随机 exit $s$，所有 chunk 产出同质 $\hat X_0$；DMD re-noise timestep $\tau$ 与 $s$ 独立。
3. 真正贡献是 progressive schedule + rolling KV + cross-frame parallelism；长训练窗口仍待论文补充。
