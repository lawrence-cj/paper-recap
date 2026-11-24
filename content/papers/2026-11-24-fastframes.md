---
title: "FastFrames: Video Generation at Interactive Latency"
paper_url: "https://arxiv.org/abs/2611.23226"
authors: "Gene Chou et al."
venue: "arXiv"
published: "2026"
read_date: "2026-11-24"
read_at: "2026-11-24T14:42:45+08:00"
status: "已读"
tags: ["Video Generation", "Autoregressive Generation", "Knowledge Distillation", "Inference Acceleration", "Training & Optimization"]
search_terms: ["FastFrames", "Causal Forcing++", "CF++", "pixel decoder", "KV refinement", "KV correction", "attention output distillation", "DWD", "Decoding Without a Decoder", "B200"]
one_liner: "FastFrames 冻结两步因果 DiT，用低分辨率像素头替代 VAE 解码、用 KV 修正网络替代 clean-cache forward，再配合系统优化将 1.3B 模型的 480p 稳态响应降到单 B200 约 36 ms。"
---

## 研究问题

视频扩散蒸馏到两步后，降低去噪步数不再足以解决交互延迟：VAE 解码和为历史缓存额外执行的 DiT forward 已占据很大开销。FastFrames 优化的是用户输入到新画面计算完成的响应时间，而非批量生成帧数的平均吞吐。

依据[论文 v1](https://arxiv.org/html/2611.23226v1)、[官方实现](https://github.com/apple-aiml-research/ml-fastframes)及本次问答整理。速度和质量均为作者报告，未独立复现。实现中的 benchmark 不含文本编码、host transfer 和显示，不能直接等同于完整应用的操作到显示延迟。

## 核心方法

### 先得到两步因果生成器，再训练附加模块

- FastFrames-S：直接使用 Causal Forcing++ 的 Wan2.1-1.3B 两步 checkpoint，冻结整个 DiT。
- FastFrames-L：从 Wan2.1-14B 开始，按 CF++ 路线训练 frame-causal teacher，再做 consistency distillation 和 self-forcing DMD；得到两步生成器后冻结 DiT。
- 两者再分别训练 pixel decoder 和 KV refinement network。官方配方把 KV 训练列在 decoder 重建与对抗训练之后，但 KV 训练本身不依赖 pixel decoder 的输出。
- 稳态每次执行两次去噪、生成一个 latent frame，通常对应四张 RGB 帧；首帧按 Wan 的时间约定特殊处理。S/L 不是两套不同的加速算法。

来源：[官方训练配方](https://github.com/apple-aiml-research/ml-fastframes/blob/main/docs/training.md)。

### 从 DiT 特征直接读出像素

原路径是“DiT features → latent prediction → VAE decoder → RGB”。FastFrames 保留 latent prediction 供自回归使用，另从最后一步的 pre-head features 输出 RGB。

像素头包含三个 causal 3D ConvNeXt block：因果 $3\times3\times3$ depthwise convolution 加通道 MLP，全部在 token 网格上计算，末端 Linear 将每个 token 映射到 $4\times16\times16\times3$ 个像素值。关键是避免在逐级放大的高分辨率特征图上反复做卷积，而非单纯压缩参数量。

冻结生成器 rollout 得到特征 $h$ 和最终 latent $\hat z_0$，先模仿原 VAE 解码结果：

$$
\hat y=P_\phi(h),\qquad y^*=D_{\mathrm{VAE}}(\hat z_0).
$$

$$
L_{\mathrm{rec}}=0.5L_1+5L_{\mathrm{MSE}}+0.05L_{\mathrm{grad}}+0.05L_{\mathrm{temp}}+0.5L_{\mathrm{motion}}.
$$

空间梯度项约束边缘，时间项匹配相邻帧变化，motion 项提高运动区域的重建权重。之后使用真实视频和基于冻结 DINOv3 的空间、时间判别头加入对抗训练，DiT 仍冻结。它是适配特定生成器的像素读出头，不是可在不同模型间直接共享的通用 VAE。

### 用小网络替代 clean-cache forward

第 $i$ 帧最后一步去噪的输入仍是带噪 latent $z_t^i$。虽然输出干净预测 $\hat z_0^i$，该次前向留下的各层 $K_i^n,V_i^n$ 仍来自带噪输入，不能当作干净帧的缓存。

基线额外将 $\hat z_0^i$ 在 $t=0$ 输入同一个 DiT，得到各层 clean KV，记为 $K_i^*,V_i^*$；丢弃这次预测输出，只覆盖当前帧缓存。

FastFrames 学习替代该次前向：

$$
(\hat K_i,\hat V_i)=R_\theta(\hat z_0^i,K_i^n,V_i^n).
$$

共享 stem 提取干净 latent 特征，再由每层独立 head 修正该层 noisy KV。修正只覆盖当前帧的 cache slice，不重算全部历史。首帧仍保留真实 clean-prime forward 作为起点。

可回顾[论文 Figure 2：推理流程与耗时](https://arxiv.org/html/2611.23226v1#S1.F2)及 [Figure 6：KV 修正消融](https://arxiv.org/html/2611.23226v1#A4.F6)。已检查 arXiv 论文许可与项目页：论文采用 non-exclusive distribution license，未确认图像另有公开转载授权，因此只链接原图，不复制图片。

### 系统优化仍贡献一部分速度

持久 KV buffer、条件与 RoPE 预计算、融合 QKV、分段 KV attention 和 CUDA Graph 降低单次 DiT 开销；多卡采用 context parallelism，在分摊 attention 的同时分摊 FFN。不能把全部加速归因于两个新增网络。

## 关键发现

| 模型 | DiT 规模 | B200 数 | 480p 稳态响应延迟 | VBench Total |
| --- | --- | --- | --- | --- |
| Causal Forcing++ | 1.3B | 1 | 272 ms | 84.11 |
| FastFrames-S | 1.3B | 1 | 36 ms | 84.23 |
| FastFrames-S | 1.3B | 8 | 14 ms | 84.23 |
| FastFrames-L | 14B | 1 | 158 ms | 84.44 |
| FastFrames-L | 14B | 8 | 45 ms | 84.44 |

CF++ 基线的约 272 ms 包含 KV 更新 41.6 ms、两步去噪 83.2 ms、VAE 解码 147 ms。官方实现的 FastFrames-S 示例为两次 DiT 16.1/16.2 ms、KV 修正 2.2 ms、像素解码 0.9 ms，合计 35.4 ms，与论文约 36 ms 接近。

计时为预热后、满 KV cache 的组件耗时汇总。$\mathrm{RPS}=1000/L_{\mathrm{ms}}$；一个生成单元只算一次响应，即使对应四张 RGB 帧，也不能把 28 RPS 直接当作视频播放 FPS。

总分接近不代表每项都改善：S 的 VBench Quality 为 85.30（CF++ 85.09），Semantic 为 79.92（CF++ 80.20）。720p 时 S 单 B200 为 127 ms、L 八 B200 为 116 ms，尚未达到 50 ms 目标。来源：[推理文档](https://github.com/apple-aiml-research/ml-fastframes/blob/main/docs/inference.md)、论文 Table 1/2。

## 我的提问

### Q1：KV loss 和 attention loss 分别监督什么？

KV loss 比较缓存数值；attention loss 比较未来帧读取缓存后的结果。两者都更新修正网络，DiT 不更新。

以单层为例，定义简化的张量距离：

$$
D(X,X^*)=1-\operatorname{cos}(X,X^*)+\frac{\|X-X^*\|_2}{\|X^*\|_2+\epsilon}.
$$

$$
L_{\mathrm{KV}}=D(\hat K_i,K_i^*)+3D(\hat V_i,V_i^*).
$$

实现中 cosine 按 token/head 向量计算后平均，relative L2 是范数之比、没有平方，再对层平均。Cosine 约束方向，relative L2 同时约束幅度。

对未来帧 $j>i$ 的固定 query $Q_j$，构造两份相同的完整缓存，只将学生分支第 $i$ 帧替换为预测 KV。设老师缓存为 $C^*$，替换后的缓存为 $C^{(i)}$：

$$
A(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt d}\right)V,
$$

$$
O_j^*=A(Q_j,K_{C^*},V_{C^*}),\qquad
\hat O_j=A(Q_j,K_{C^{(i)}},V_{C^{(i)}}),
$$

$$
L_{\mathrm{attn}}=\operatorname{mean}_{\mathrm{layers},j}\frac{\|\hat O_j-O_j^*\|_2}{\|O_j^*\|_2+\epsilon},\qquad
L=L_{\mathrm{attn}}+0.1L_{\mathrm{KV}}.
$$

这里比较的是乘完 V 的 attention 输出，不只是 softmax 权重图。其他历史帧保持 clean KV，query 当前帧保留自己的 noisy KV；不是让 query 只读取被修正的一帧。

直观例子：注意力权重为 $[0.99,0.01]$ 时，V 的输出误差为 $0.99\Delta V_1+0.01\Delta V_2$。KV loss 约束两处重建，attention loss 反映哪种错误更影响实际读取；直接 KV 监督也能约束有限 query 没有关注到的位置。此例仅用于解释，不是论文实验。

### Q2：和老师第 5 帧 KV 比较，老师也需要重跑一次 t=0 吗？

需要。训练时同一个冻结 DiT 先去噪得到 $\hat z_0^5$ 和 noisy KV，再将该预测 latent 以 $t=0$ 输入 DiT，提取 clean KV 作为 target。所谓 clean 指由干净预测重新前向，不代表来自真实视频 ground truth。teacher 不需要是更大的另一套模型。

训练 rollout 记录这些目标后用于修正网络的优化；省下的是部署时每帧额外的 clean-cache DiT forward，不是生成训练标签时的计算。首帧是仍执行 clean-prime 的例外。

### Q3：未来 Q 从哪里来，是否对整段视频反向传播？

冻结的基线 rollout 会记录各帧最后去噪步骤的 post-RoPE query、noisy/clean KV 和干净 latent 的 patch features。每次选择一帧缓存修正，用下一帧和随机更远帧的固定 Q 做读取比较。

梯度路径是 $L_{\mathrm{attn}}\rightarrow\hat O_j\rightarrow(\hat K_i,\hat V_i)\rightarrow R_\theta$；老师目标和记录的 Q 不更新。这是局部 teacher-forced 蒸馏，不是将全部修正缓存连续接入未来生成后，再穿过整段视频反传。推理时不需要预知未来 Q。

上述公式与流程已核对[官方实现的 capture、attention_loss、kv_loss 和 train_rollout](https://github.com/apple-aiml-research/ml-fastframes/blob/main/fastframes/training/kv_refinement.py)。

### Q4：correction module 是不是在 CF++ 之后单独训练？

是。先完成两步因果生成器，再冻结 DiT 单独训练修正模块。S 使用现成 CF++ checkpoint；L 自己按 CF++ 路线完成 causal teacher、consistency 和 DMD。pixel decoder 的重建/对抗训练与 KV 修正训练是不同任务，不是一起重新训练整个生成器。

### Q5：S 和 L 有什么不同？

S/L 是 Small/Large，主要区别是 1.3B 与 14B 的生成器规模及取得 checkpoint 的路径。两者都执行两步去噪、每次生成一个 latent frame，并使用适配自身 DiT 的像素头和 KV 修正网络。L 的像素头将 5120 维 features 投影到 1536 维，并从 S 的 decoder checkpoint 初始化后适配。S 在单 B200 上达到交互延迟；L 的 45 ms 需要八 B200。

### Q6：和 DWD 的 pixel decode 训练是不是很像？

是。共同点是保留 latent 采样，从 DiT 最后一步的高维 hidden features 直接输出像素，而不是仅将最终 latent 线性放大。区别在于特征适配放在哪里、监督从哪里来。

| 比较项 | FastFrames | 标准 DWD |
| --- | --- | --- |
| 像素头 | causal ConvNeXt + Linear | 主要为 Linear，可选 refiner |
| DiT | 全部冻结 | 微调尾部若干 block 与输出组件 |
| 重建数据 | 冻结生成器 rollout | 真实图像/视频编码后加噪 |
| 像素标签 | 先模仿原 VAE 解码结果，后加真实视频对抗监督 | 真实像素上的 L1、LPIPS、对抗监督 |
| 保留原去噪能力 | 直接冻结 DiT | denoising head 匹配冻结原模型预测 |
| KV correction | 有，替代 clean-cache forward | 标准方法不涉及这一模块 |

标准 DWD 用 $L_{\mathrm{pix}}+\lambda L_{\mathrm{traj}}$ 联合训练，共用尾部同时接收两项梯度；并非先彻底训完去噪约束、再单独训像素。可理解为 FastFrames 把适配放在新增非线性头，DWD 则让 DiT 尾部也适应简单读出头；这是结构与训练方式的解释，不是已验证的因果结论。

重要例外：DWD 附录 B.5 的 MiniMax-H3 采用 decoder distillation，以参考模型完成原生最后一步并经原 decoder 的输出作为像素目标，且需要轻量 refiner；这个版本在监督构造上更接近 FastFrames。不能把“DWD 都用真实像素标签、都只加 Linear”推广到 H3。

详见已有的 [DWD 笔记](https://junsongc.top/paper-recap/#paper=dwd)、[DWD §3.3](https://arxiv.org/html/2611.22287v2#S3.SS3)和[附录 B.5](https://arxiv.org/html/2611.22287v2#A2.SS5)。

## 局限与疑问

- 480p、B200、满缓存稳态计算延迟有明确硬件和口径限制；显示、传输和冷启动仍需另测，720p 尚不满足目标。
- 像素头依赖具体 DiT；换生成器需要适配，不能当成通用 VAE 的直接替换件。
- KV 修正训练使用老师上下文与有限未来 query，不保证长时间自由 rollout 的误差不累积。
- “轻量”主要指运行成本。S 的 KV 网络约 346M 参数；训练还需在线 rollout 和 clean-prime 标签计算，不能理解为低成本加几个线性层。
- 两步蒸馏、像素头、KV 修正和系统优化的收益应分开评估；T2V 的 36 ms 不能直接套用于更复杂的相机/物理条件生成。
- 与 DWD 缺少统一 backbone、数据、画质与硬件条件下的直接比较，尚不能判断哪种适配方式更好。

## 我的判断

用户已确认的理解：KV 修正阶段需要老师额外跑 $t=0$，在 CF++ 两步生成器完成后单独训练；重点追问了缓存数值与 attention 输出监督的关系，并指出它与 DWD 像素读出的相似性。尚未表达复现决定或最终优劣判断。

Agent 的判断：先测少步生成中 decoder 与 clean-cache forward 的真实占比，再决定迁移价值。冻结 DiT 的像素读出和 KV 修正可以分别评估。与 DWD 对照时，最值得比较的是“冻结 DiT、增加非线性头”和“微调 DiT 尾部、使用简单读出头”，以及真实像素监督和 decoder distillation 的区别。

## 下次只看这些

1. 两步因果生成器完成后冻结 DiT；训练像素头替代 VAE，再单独训练 KV 网络替代 clean-cache forward。
2. KV loss 匹配缓存张量，attention loss 用固定未来 Q 匹配完整读取结果；训练仍需 $t=0$ 老师标签，部署才省掉该次前向。
3. 与 DWD 共用 features-to-pixels 思路，但标准训练的数据、可训练参数不同；H3 的 decoder-distillation + refiner 版本更接近。36 ms 是单 B200、480p 的稳态计算口径。
