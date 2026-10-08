---
title: "SGF+: Decoupling Gradient Flows for Autoregressive Video Generation"
paper_url: "https://arxiv.org/abs/2610.10429"
authors: "Zihan Su et al."
venue: "arXiv"
published: "2026"
read_date: "2026-10-08"
read_at: "2026-10-08T20:51:07+08:00"
status: "已读"
tags: ["Video Generation", "Autoregressive Generation", "Streaming Video", "Diffusion Models", "Knowledge Distillation"]
search_terms: ["Video Generation", "Autoregressive Generation", "Knowledge Distillation"]
one_liner: "在 SGF 可求导历史重建上，把 t=0 的 context writer 与带噪去噪器分成两套参数，减少角色梯度冲突；完整版本生成器约从 1.4B 翻至 2.8B，推理调用次数不翻倍。"
---

## 研究问题

SGF 恢复了未来生成损失对历史 KV 写入的监督，但 clean-context encoding 与 noisy-latent denoising 仍更新同一套 DiT 权重。SGF+ 问的是：两种输入条件和计算角色，是否要求相互冲突的参数更新？

本文正式名称是 SGF+，不是 SGF++，发表于 2026-10-07。阅读依据：[论文 v1 §3–4、Appendix B–E](https://arxiv.org/html/2610.10429v1)、[官方项目页](https://zihan-su.github.io/self-gradient-forcing-plus/)、[官方代码快照 14cda9b](https://github.com/Zihan-Su/Self_Gradient_Forcing_Plus/tree/14cda9bb35f87000fbc11138a5170002d213effe)。本次核对了实现，没有运行训练或推理复现。

建议先读 [SGF review](https://junsongc.top/paper-recap/#paper=self-gradient-forcing)：两遍流程、exit 状态、causal mask 和 DMD 都沿用这一基础。本文真正新增的是角色参数分离，不是发明两次前向。

## 核心方法

![SGF+ 两遍训练方法图：第一遍无梯度生成并保存 clean/noisy 状态，第二遍用独立 context writer 与 denoiser 接受同一个 DMD loss](media/self-gradient-forcing-plus/method-overview.webp "论文 Figure 4 对应的官方训练方法图：SGF 共享参数与 SGF+ 角色参数分离。来源：Zihan-Su/Self_Gradient_Forcing_Plus，assets/method.png；仓库 Apache-2.0；转为 WebP，未改动图示内容。")

图源：[官方资产](https://github.com/Zihan-Su/Self_Gradient_Forcing_Plus/blob/14cda9bb35f87000fbc11138a5170002d213effe/assets/method.png)，许可：[Apache-2.0](https://github.com/Zihan-Su/Self_Gradient_Forcing_Plus/blob/14cda9bb35f87000fbc11138a5170002d213effe/LICENSE)。随图保留仓库许可文本。这里依据仓库资产许可转载，不将 arXiv 的 non-exclusive distribution 许可当图片再授权。

### 1. 同一个 loss 内，两种角色的梯度也可能冲突

对一个共享权重 $W$，把历史写入和当前去噪 tokens 的贡献拆开：

$$
g_W=g_C+g_D,\qquad
\operatorname{cos}(g_C,g_D)
=\frac{\langle g_C,g_D\rangle}{\|g_C\|_2\|g_D\|_2}.
$$

$g_C$ 来自 context writing，$g_D$ 来自 denoising；它们不是两项不同 loss，而是同一 DMD 目标经过两种角色的梯度。若内积为负，合并到同一权重时会部分抵消。

作者在所测 128 prompts × 4 timesteps 中，Attention 和 FFN 各 512 对角色梯度的余弦相似度均为负。平均方向的可视化报告夹角为 104.2°、106.3°。逐对余弦使用完整的所选权重梯度；t-SNE 与平均方向图使用坐标抽样表示，二者的测量口径不同。实验细节见 Q7。这是特定模型与测量的观察，不是所有视频模型必然如此的定理。

### 2. 两套参数，前向仍通过 KV 合作

将历史写入参数记为 $\theta_c$，去噪参数记为 $\theta_d$，均从同一个自回归初始化复制。第二遍计算：

$$
M_{\theta_c}=\mathcal C_{\theta_c}(\operatorname{sg}(X^\star),0;\mathcal M_{\mathrm{rec}}),
$$

$$
\hat X=\mathcal D_{\theta_d}(Z^\star,t^\star\mid M_{\theta_c};\mathcal M_{\mathrm{rec}}),
\qquad\mathcal L=\mathcal L_{\mathrm{DMD}}(\hat X).
$$

$X^\star$ 是第一遍保存的干净预测，$Z^\star$ 是选定 exit 的实际带噪输入。历史 latent 仍 detach，KV 重建保留梯度。loss 对 $\theta_c$ 的更新通过历史 KV 到达，对 $\theta_d$ 的更新通过当前去噪到达；没有额外 KV loss，也没有做梯度投影。

### 3. 每层主要模块都分开，不只加一个 linear

论文实际使用两套完整参数。官方实现以 `DualFullExpertLinear` 在同一个模型对象内保存两份权重，按 clean-memory / noisy-generation token 角色确定使用哪一份，不是学习一个动态 gating。

| 模块 | 参数分离方式 |
| --- | --- |
| 每层 self-attention 的 Q/K/V/O | 两份完整 linear 权重与对应 bias |
| 每层 text cross-attention 的 Q/K/V/O | 两份完整 linear 权重与对应 bias |
| 每层 FFN 的两个 linear | 两份完整权重 |
| Norm 的可学习参数、modulation | 各自角色参数；无参数的 norm 没有可复制权重 |
| Patch、时间、文本 embedding/projection 与输出 head | 相应模块也复制 memory 版本 |

这是模型内部模块的分离，不意味着另复制整个外部文本编码器、VAE 或 real-score teacher。两份参数从同一初始值开始，之后独立更新。[参数复制函数](https://github.com/Zihan-Su/Self_Gradient_Forcing_Plus/blob/14cda9bb35f87000fbc11138a5170002d213effe/wan/modules/causal_model.py#L878)

### 4. 各层 KV 一一对应

在第 $\ell$ 层，历史 hidden state 经 writer 的 $W_{K,c}^{(\ell)},W_{V,c}^{(\ell)}$ 形成历史 K/V；denoiser 同层的 Q 读取这些表示。当前块自己的 noisy K/V 则由 generation 参数产生。

| 历史写入，$t=0$ | 提供的历史特征 | 当前去噪，$t=t^\star$ |
| --- | --- | --- |
| Writer 第 1 层 | $K_c^{(1)},V_c^{(1)}$ → | Denoiser 第 1 层读取 |
| Writer 第 2 层 | $K_c^{(2)},V_c^{(2)}$ → | Denoiser 第 2 层读取 |
| 各层继续向下计算 | 各自这一层的 K/V → | 对应层读取 |
| Writer 第 $L$ 层 | $K_c^{(L)},V_c^{(L)}$ → | Denoiser 第 $L$ 层读取 |

不是 writer 最后产一份 KV 给所有层。推理中 cache 按层索引；第二遍训练则在同一个带 mask 的 forward 内重建各层表示。每层 KV 有到后续 loss 的路径，不保证所有 writer 参数每次都有非零梯度；Appendix E 观察到最后一层 FFN 的 context 梯度为零，因为它位于最后一层 KV 计算之后。

## 关键发现

实验使用 Wan2.1-T2V-1.3B 衍生因果学生、冻结 Wan2.1-T2V-14B teacher 和单独训练的 1.3B fake-score network，从 Causal Forcing 的 TF 初始化开始。训练 rollout 为 5 秒，使用四步去噪；60/240 秒分别在 VBench-Long/MovieGen-128 prompts 上测试。

下面摘录 Table 1 的 chunkwise 240 秒结果，分数乘 100，三者在同一设置内比较：

| 方法 | Subject | Background | Flickering | Dynamics | Aesthetics | Imaging |
| --- | --- | --- | --- | --- | --- | --- |
| SF | 94.94 | 95.21 | 93.54 | 93.90 | 55.12 | 68.38 |
| SGF | 97.72 | 96.90 | 97.03 | 56.97 | 62.54 | 71.11 |
| SGF+ | 98.21 | 97.31 | 97.57 | 56.92 | 64.74 | 71.39 |

SGF+ 在多数质量与一致性指标提高，Dynamics 并非优势。framewise 60 秒 Imaging 为 SGF 71.52、SGF+ 71.11，也说明它并非逐项都领先。原 SGF 论文与 SGF+ 的表格数值不能跨表直接视为同一次实验。

Table 2 在 chunkwise 60 秒比较只分 Attention、只分 FFN、全部参数分离：Subject 分别为 95.37、97.43、98.48，Aesthetics 为 58.49、63.49、66.63。完整分离优于两个局部方案，但这些消融也同时改变参数量，不能单凭它们排除容量增加的贡献。

### 参数、显存与计算代价

生成器参数约从 1.4B 增至 2.8B。它仍使用原来的去噪与历史写入调用安排，每种 token 只执行自己的角色参数，因此参数翻倍不等于计算量翻倍。Table 3 所测配置为：

| 指标 | SF | SGF | SGF+ |
| --- | --- | --- | --- |
| 训练峰值显存 | 86.36 GB | 97.83 GB | 98.15 GB |
| 训练稳定显存 | 86.19 GB | 70.06 GB | 73.60 GB |
| 训练每步耗时 | 10.02 s | 11.79 s | 12.76 s |
| 推理显存 | 24.85 GB | 24.85 GB | 27.96 GB |
| 81 帧推理耗时 | 4.963 s | 4.962 s | 4.969 s |

相对 SGF，训练每步约增加 8.2%；推理延迟近似不变。绝对显存与耗时只适用于所测设置，不是任意硬件、batch 或长度的保证。参数翻倍也只指生成器，不是整个 teacher/fake-score 训练系统。

### 长时间生成的证据范围

论文正式定量评测为 5、60、240 秒；Appendix D 展示四个连续 24 小时生成案例。[项目页](https://zihan-su.github.io/self-gradient-forcing-plus/)提供长视频展示。24 小时可持续生成，不等于对 24 小时所有帧做过完整语义、运动或剧情评测。

持续运行还依赖 sink + FIFO 和时间位置处理。官方 cache 配置为 framewise 4+16+1=21、chunkwise 3+6+3=12 个 latent 时间帧；它没有存住全部 24 小时历史。因此视觉稳定外推与长期事件记忆是不同能力。

## 我的提问

### Q1：第一遍只使用 noisy 分支，第二遍才用 clean 分支吗？

不是。第一遍就交替执行 generation 去噪与 memory 写历史，两者都无梯度。第二遍再用两种角色重建计算，两者都保留相应参数的梯度。

### Q2：同一个干净预测为什么要过两次 Memory？

第一次产生 KV，帮助第一遍生成后续 chunk；第二次用保存的同一份 clean latent 重建 KV，恢复参数到 KV 的计算图。第一遍的 cache 不直接用于第二遍。最后一个块的 KV 如果没有后续块，就没有后续消费者。

### Q3：记录的输入、输出与两个 timestep 怎么对应？

| 状态 | 角色 | timestep | 第二遍用途 |
| --- | --- | --- | --- |
| 记录的 $Z_B^{t^\star}$，带噪输入 | Generation | $t^\star$ | 重算 B 的预测 |
| 记录的 $X_B^\star$，干净估计 | Memory | 0 | 为 C 构建 B 的历史 KV |

“clean”指预测目标是无噪声，不代表画面正确或已经完成所有步骤。默认 `cache_mode: exit` 保存选中步骤的 $x_0$ 估计；默认代码可继续跑完 schedule，但写历史仍使用该估计。第 2 步的索引不等于 timestep 数值 2。[训练实现](https://github.com/Zihan-Su/Self_Gradient_Forcing_Plus/blob/14cda9bb35f87000fbc11138a5170002d213effe/pipeline/self_gradient_forcing_training.py#L142)

### Q4：第二遍 causal 是逐块执行吗？loss 又是什么？

第二遍一次并行 forward，clean 与 noisy tokens 按角色用参数，再用专门的 block-causal mask 交互。noisy B 可以读 clean A，不能读 clean B/C；C 使用保存的 clean B，不等待第二遍新预测的 B。

loss 沿用 DMD。冻结 teacher 与跟踪学生分布的 fake-score network 为第二遍输出提供分布匹配更新；不是对第一遍输出做 reconstruction MSE，也没有额外 KV 标签。DMD 计算时另外给输出加噪的时间，与 rollout 选中的 $t^\star$ 不同。详见 [SGF review 的 DMD 公式](https://junsongc.top/paper-recap/#paper=self-gradient-forcing)。

### Q5：是不是模型大小扩大两倍？值得吗？

是，完整参数版生成器基本翻倍。用户明确指出了这个代价。论文报告的推理 latency 几乎不变，是调用安排与每 token 激活参数没翻倍，不是多出来的权重没有成本。需要更多参数存储，训练还需对应梯度与优化器状态。

论文说也可以用两套 LoRA 实现角色分离，但本次结果来自两套完整参数；不能把 LoRA 看成已经验证的同等效果。若关注效率，还应比较相同参数预算的共享模型或不同范围的角色分离。

### Q6：和 Vidu S2 的 Self-Replay Forcing 是不是同一类方法？

用户的相似性判断成立：二者都先无梯度 rollout，再对 detach 状态建立带梯度 replay，避免保存完整采样图。区别不只在是否多了一套模型：**SGF 还没分参数时，就已经与 SRF 的 replay 输入不同。**

| 维度 | Vidu S2 SRF | SGF | SGF+ |
| --- | --- | --- | --- |
| 第二遍带噪输入 | 自生成轨迹重新加噪 | 第一遍实际记录的 exit 输入 | 同 SGF |
| replay 内历史 | 带噪过去块产生的表示 | clean 过去块在 0 编码的 KV | 同 SGF，但用 writer 参数 |
| 角色参数 | SRF 公式同一学生，未报告角色分离 | 共享 | writer/denoiser 独立 |
| 监督 | DMD + perceptual | DMD | DMD |
| 第二遍与原前向的关系 | 对重新加噪轨迹做新预测 | 恢复选定 exit 计算 | 恢复选定 exit 计算 |

SRF 的 replay 内历史表示也是可求导的，后续损失能通过带噪历史 K/V 回传；范围外继承历史仍固定。SGF 系列明确恢复 $t=0$ clean-memory writing 的路径。二者都不是完整 rollout BPTT。[Vidu S2 §2.2](https://arxiv.org/html/2609.11638v1#S2.SS2)

Vidu 的 Backbone/超分 Refiner 与 SGF+ 的 Writer/Denoiser 不同：前者分工是低分辨率生成与高分辨率细节恢复，后者分工是历史编码与当前去噪。不能因为 Vidu 有两个模块，就认定已经采用 SGF+ 的角色分离。

### Q7：梯度冲突是怎样设计实验、从共享权重里测出来的？

**固定原始 SGF 共享模型，只测梯度，不更新权重。** 官方诊断用 SGF generator checkpoint、配套 fake-score critic 和冻结 Wan2.1-T2V-14B teacher；128 个 prompts 各覆盖四个去噪 exit，共 512 个 prompt–exit 配对。不是先训练两个独立模型，再比较它们的梯度。第一遍无梯度 rollout 后，在第二遍重建的同一个 DMD loss 上做 backward。[实验配置与采集说明](https://github.com/Zihan-Su/Self_Gradient_Forcing_Plus/blob/14cda9bb35f87000fbc11138a5170002d213effe/gradient_conflict/README.md)

对共享 linear 的 $Y=HW^\top$，保存输入 $H$，并在输出处获取反向信号 $\Delta=\partial\mathcal L/\partial Y$。把 batch 与 token 维展开，再按 clean/context 与 noisy/denoising 两组 tokens 分开：

$$
g_C=\Delta_C^\top H_C,\qquad
g_D=\Delta_D^\top H_D,\qquad
\frac{\partial\mathcal L}{\partial W}=g_C+g_D.
$$

这拆的是同一 loss 对同一权重的两类使用位置的贡献。$\Delta_C$ 已包含后续预测经 attention 回传的信号，所以 clean stream 无需另设 loss。代码用 forward/output hooks 做拆分，FP32 计算完整矩阵的内积与范数，再校验两项相加与 autograd 的实际权重梯度近似一致。[`collect.py` 的拆分与校验](https://github.com/Zihan-Su/Self_Gradient_Forcing_Plus/blob/14cda9bb35f87000fbc11138a5170002d213effe/gradient_conflict/collect.py#L114)

**测量范围与统计单位需要分清：**

- 所有 30 层 self-attention 的 Q/K/V/O 权重，共 120 个矩阵；FFN 的 up/down 权重，共 60 个矩阵。不包括 bias，也不是遍历全模型所有参数。
- 对每个 prompt–exit，将同一模块族的各层权重梯度看作一个拼接向量，分别计算 Attention、FFN 的 $\operatorname{cos}(g_C,g_D)$。两个族各有 512 个值，全部为负；不能扩写成“每个层、每个参数都冲突”。[`analyze.py` 的聚合统计](https://github.com/Zihan-Su/Self_Gradient_Forcing_Plus/blob/14cda9bb35f87000fbc11138a5170002d213effe/gradient_conflict/analyze.py)
- t-SNE 展示两类梯度分布差异，负余弦才是直接的方向冲突证据。最后一层 FFN 的 context 梯度全为零，余弦与角距离没有定义；Appendix E 因此不画它的 t-SNE。

**另一个实验在 TF 初始化、SGF 训练前测。** Appendix C 使用 128 段真实视频 × 50 个随机去噪时间，context 固定为 0，共 6,400 对／模块族。平均方向角为 Attention 95.1°、FFN 105.1°，绝大多数配对余弦为负。它说明角色差异在这个初始化上已经存在；不能和主实验的 prompt rollout、512 对混为一组。[Appendix C](https://arxiv.org/html/2610.10429v1#A3)

对我们更有用的是这套诊断：先在目标模型上按角色拆梯度，检查和是否还原、负余弦比例、两种梯度的相对大小和逐层分布，再决定分哪些参数。负内积说明共享更新有抵消项，但不单独证明实际 AdamW 更新会让 loss 上升，也不保证增参后效果一定更好；仍需训练消融与相同参数预算对照。

## 局限与疑问

- **更多参数的贡献待分离**：角色梯度分析支持设计动机，但希望有同参数预算的共享模型对照，量化角色分离和容量增加各自的贡献。
- **运动与稳定性的取舍**：Dynamics 下降不能直接当失败，也不能自动当成功；应检查有效动作、运动幅度和冻结倾向。
- **24 小时只是演示范围**：没有由此证明长期剧情规划、事件记忆或任意输入下的稳定性；有限 cache 已限制直接访问的历史。
- **迁移需要实测**：Attention/FFN 的角色冲突是否出现在另一架构、损失与噪声日程中，应先测梯度。最终层某些 writer 模块没有梯度也是实现与依赖关系的一部分。
- **与 Vidu 的效果不能横比**：任务、数据、teacher、历史噪声、rollout/replay 时长及系统优化不同；SRF 参数公开不全。本次只有机制对比，没有独立复现或控制实验。

## 我的判断

用户已明确两遍 forward 的输入输出、Memory 两次调用、每层 KV 的对应关系，并提出参数基本翻倍与 Vidu SRF 相似性的疑问。用户理解了方法，但尚未决定复现或接受参数开销。

Agent 的判断：先验证 SGF 的历史梯度路径，再测角色梯度冲突，最后决定是否上完整分离。小规模实验可比较冻结历史、共享参数 SGF、角色 LoRA 和完整 SGF+，固定初始化、教师与推理预算，同时记录质量、动作与成本。LoRA 和把角色分离迁移到 SRF 都是可研究方案，不是本文已验证结论。

关于方法是否 elegant：Agent 更欣赏原始 SGF 明确选择 detach 边界、恢复所缺历史梯度的设计。SGF+ 的参数分离直观，但生成器基本翻倍，简洁程度要结合参数预算判断；“优雅”是阅读偏好，不能替代质量与成本对照。详见 [SGF 的 Q9](https://junsongc.top/paper-recap/#paper=self-gradient-forcing)。

## 下次只看这些

1. **新增设计与证据**：SGF 恢复 clean KV 的梯度，SGF+ 再分写历史／去噪参数；Q7 用同一 loss 的 token 角色拆梯度，主实验为模块族各 512 个负余弦，不是每层都负。
2. **两遍与成本**：两遍都用 Memory，第一次无梯度帮助生成，第二次有梯度接受未来 DMD；完整生成器约 1.4B → 2.8B，调用次数与计算量不随之翻倍。
3. **与 SRF 对照**：SRF 重加噪并训练带噪历史表示，SGF 重放 exit 并重建干净历史；SGF+ 在此基础上再分角色参数。长演示与长期剧情能力分开判断。
