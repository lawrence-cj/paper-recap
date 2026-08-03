---
title: "VideoCoCo: Code-as-CoT for Physically-Consistent Video Generation via an Agentic Dual-Engine System"
paper_url: "https://arxiv.org/abs/2607.27380"
authors: "Haodong Li et al."
venue: "arXiv"
published: "2026"
read_date: "2026-08-03"
status: "已精读"
rating: 3
tags: ["Video Generation", "Physical Reasoning", "Diffusion Models", "Agentic Systems"]
one_liner: "先用 Agent 编写并执行 Blender 程序，将物理过程落实为逐帧白模草稿，再让视频编辑 DiT 只负责写实化，从而把动态推理与视觉生成解耦。"
---

## 研究问题

文本 prompt 只压缩描述了“发生什么”，却没有给出对象、状态转移及逐帧因果过程；端到端 Text-to-Video 模型必须同时猜测物理演化和生成写实像素，因而容易出现单帧合理但运动、碰撞、熔化、升华或浮力过程错误。论文将这种缺失显式过程的现象称为 **Causal Opacity**，目标是在生成最终像素前先外化一个完整、可检查、可修改的时空过程。

## 核心方法

- **Executable Simulation Engine**：代码 Agent 根据 prompt $p$ 生成自包含的 Blender Python 程序 $c=A_{\mathrm{code}}(p)$，显式规定对象、物理属性和时间演化；沙箱执行得到白色黏土风格的低保真草稿视频 $d=\mathcal{B}(c)$。
- **Generative Video Engine**：instruction Agent 同时读取原 prompt 与草稿，生成只描述主体、材质、光照和镜头的编辑指令 $e=A_{\mathrm{edit}}(p,d)$；视频编辑器输出 $\hat v=G_\theta(d,e)$。草稿决定“发生什么以及何时发生”，指令决定“最终看起来怎样”。
- **VideoCoCo-3K**：对每个 prompt 生成 Blender 草稿 $d_i$ 和编辑指令 $e_i$，再调用 Seedance 2.0 teacher 产生写实目标 $y_i=G_T(d_i,e_i)$，得到 3,000 个合成三元组：

$$
\mathcal D_{\mathrm{VideoCoCo\text{-}3K}}
=\left\{(d_i,e_i,y_i)\right\}_{i=1}^{3000}.
$$

这里不是把真实拍摄视频和 Blender 视频配对，而是让闭源 teacher 把白模 source 重绘为 synthetic target。原始 prompt 和 Blender code 仅作为 metadata 保留。

- **DiT condition 注入**：VideoCoCo 复用 OmniWeaving/HunyuanVideo-1.5 的 editing 架构。Blender MP4 经视频 VAE 得到 $z_d$；训练目标 $y$ 经 VAE 得到 $z_0$，在时间步 $t$ 加噪为 $z_t$。主条件在 patchify 之前沿 channel 维 early fusion：

$$
x_{\mathrm{DiT}}=\operatorname{Concat}_{C}[z_t;z_d;m],
$$

其中 $m$ 是 condition-valid mask，完整视频编辑时全为 1。OmniWeaving 中 $z_t$ 和 $z_d$ 各有 32 个 channel，因此实际输入为 $32+32+1=65$ channels；随后由 `Conv3D PatchEmbed` 投影并 flatten 成时空 token。模型输出仍为 32-channel 的 target noise/velocity prediction。

- 除稠密 latent concat 外，草稿还会抽取 4/6/8 帧：一条路径与 edit prompt 一起进入多模态文本编码器，另一条经 vision encoder 形成 semantic tokens，并入 DiT 的条件 token stream。稠密运动主要依赖第一条 channel-concat 路径。
- Blender 和代码 Agent 均不在梯度图中，也没有对 Blender 反向传播。只训练视频编辑器，目标是标准条件去噪：

$$
\mathcal L(\theta)
=\mathbb E_{(d,e,y),t,\epsilon}
\left[\left\lVert
\epsilon-\epsilon_\theta(z_t,t,d,e)
\right\rVert_2^2\right].
$$

因此这里的 Code-as-CoT 是外置、可执行的系统级中间过程，不是视频 DiT 内部生成的一串 reasoning tokens。

## 关键发现

- 在 PhyGenBench 上，将 VideoCoCo 加到 OmniWeaving 后，平均物理一致性从 **0.475 提升到 0.558**；材料动态从 0.392 提升到 0.525，热学从 0.433 提升到 0.511。
- 在 VBench-2.0 的物理维度上，平均分从 **52.18% 提升到 77.88%**；力学达到 92.31%，热学达到 72.92%，材料为 68.42%。
- 不微调编辑器、只加入可执行草稿时，PhyGenBench 已从 0.475 提升到 0.506，说明收益不完全来自额外训练。
- LoRA 微调达到 0.558，优于全参数微调的 0.535。任务只需学习“保留白模运动并写实化”的窄技能，LoRA 更能保留基础模型已有的视觉先验。

## 我的提问

### Q1：训练 data pair 究竟如何构造，Blender 的输出是视频吗？

严格说是 $(d,e,y)$ 三元组。$d$ 是 Blender 逐帧渲染的白模 MP4，不是几张关键帧或代码 token；$e$ 是写实化指令；$y$ 是 Seedance 2.0 根据 $d,e$ 生成的写实视频。训练时读取预先生成好的三元组，不会在每个 gradient step 里运行 Blender 或 Seedance。

### Q2：Blender 输出如何作为 CoT 参与模型计算？

代码 $c$ 是 symbolic CoT，执行后的白模视频 $d$ 是 visualized CoT；视频 DiT 实际消费的是 $d$ 的 VAE latent。没有梯度穿过代码 Agent 或 Blender，也没有将 Blender code 作为文本 token 输入视频模型。“CoT”描述的是系统中的过程级中间表示，而非一种新的可微 CoT loss。

### Q3：condition 是和输入 feature concat，还是通过 cross-attention？

主路径是 **VAE latent 的 channel concat**：$[z_t;z_d;m]$ 在 `Conv3D PatchEmbed` 前融合，所以 target 与草稿在相同 $(t,h,w)$ 位置直接对齐。另有抽样帧通过多模态 text encoder 和 vision-token condition stream 提供高层语义，但这不是主要的稠密运动约束。

### Q4：训练数据开源了吗？

截至 2026-08-03，完整 VideoCoCo-3K 未公开。官方仓库只提供 8 条人工检查的 toy triplets、Agent Skills 和推理代码；完整 prompts、Blender programs、3K targets 与清洗记录没有下载链接。模型 Hugging Face 仓库当时也只有初始占位文件，README 仍标注权重正在上传。

## 局限与疑问

- Blender 草稿不等于严格的高精度物理仿真。部分熔化、破裂、透明度或覆盖变化可能由 Agent 通过程序化动画表达；如果 Agent 写错因果过程，编辑器只会把错误过程画得更逼真。
- 完整 3K 数据及清洗流程未公开；论文没有充分报告 source prompt distribution、代码生成成功率、自动修复轮数、teacher target 验收率、总延迟和成本，复现证据不足。
- 训练目标来自 Seedance 2.0，因此包含闭源 teacher 偏差，也不能验证其是否总能严格保持白模运动。
- PhyGenBench 依赖 GPT-4o judge；VBench-2.0 只评估选定物理维度。当前结果支持“物理 benchmark 提升”，不能推出综合视频质量全面领先。
- 系统增加 Agent 编程、Blender 执行和视频编辑延迟，并受 Blender 表达能力限制；湍流、复杂流体、燃烧和材料断裂仍然困难。

## 我的判断

最值得记住的不是 Blender 本身，而是“**可执行中间表示 + 生成式实现**”这一系统范式：让模拟器负责过程约束，让生成模型负责外观，可以将不可控的端到端任务拆成可检查、可编辑的阶段。代码与草稿带来的可诊断性很有价值，也可迁移到机器人轨迹、3D 场景、科学仿真和 UI 生成。

但这仍是较新的 arXiv v1，完整数据、权重和关键复现细节尚未公开。当前更适合作为系统设计启发和小规模原型方向，而不是直接相信其 SOTA 结论或投入完整复现。

## 下次只看这些

1. 核心计算是 $[z_t;z_d;m]$ 的 latent channel concat；Code-as-CoT 是外置流程，不是 DiT reasoning token。
2. 训练集是 Blender source + edit instruction + Seedance synthetic target 的 3K 三元组，完整数据尚未开源。
3. 真正启发是把“过程正确性”和“像素写实度”交给不同引擎，并用可执行草稿作为可检查的因果瓶颈。
