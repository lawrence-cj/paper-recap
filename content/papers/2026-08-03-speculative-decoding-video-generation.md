---
title: "Speculative Decoding for Autoregressive Video Generation"
paper_url: "https://arxiv.org/abs/2604.17397"
authors: "Yuezhou Hu, Jintao Zhang"
venue: "arXiv"
published: "2026"
read_date: "2026-08-03"
status: "已精读"
tags: ["Video Diffusion", "Autoregressive Generation", "Speculative Decoding", "Acceleration"]
one_liner: "SDVG 让 1.3B 模型先生成 autoregressive video block，再用最差帧 ImageReward 决定直接接受还是交给 14B 重做，以放弃严格 target 分布为代价换取真正跳过大模型计算。"
---

## 研究问题

LLM speculative decoding 能让小模型先提出 token、再由大模型批量验证，并通过 rejection sampling 严格保持 target 分布；但 autoregressive video diffusion 的输出单元是连续、高维的时空 latent block，没有可直接用于 token-level verification 的离散概率分布。论文要回答的是：能否利用视频的 block-autoregressive 结构，让 1B 级模型承担多数简单 block，同时只在必要时调用 14B target，从而兼顾流式生成速度与大模型质量？

已有 T-Stitch、SRDiffusion 等方法主要在 denoising timestep 上固定切分大小模型，不能识别并纠正内容相关的坏 draft；SDVG 则把算力路由粒度移到完整视频 block。

## 核心方法

- **Drafter–target 配置**：1.3B Wan2.1 Self-Forcing drafter 与 14B Krea Realtime Video target；二者每个 block 都运行 4 个 denoising steps。
- **逐 block routing**：drafter 先生成候选 latent block，经 causal VAE 解码后，使用现成的 text-image reward model ImageReward 对每一帧评分。
- **最差帧聚合**：block 分数取所有 decoded frames 的最小值，而不是平均值：

$$
q_b=\min_i \mathcal R\!\left(f_i^{(b)},p\right),
$$

  其中 $p$ 是文本 prompt，$f_i^{(b)}$ 是第 $b$ 个 block 的第 $i$ 帧。只要 $q_b\ge\tau$ 就接受 draft，否则由 14B 从相同初始噪声重新生成；最小值用于捕捉平均分会掩盖的单帧崩坏和闪烁。
- **首 block 强制重做**：block 0 无论 reward 多高都由 target 生成，因为它决定主体、构图与风格，早期错误会通过 causal context 传播到整段视频。
- **状态维护**：drafter 始终以自己的历史输出继续生成；draft 被接受时写入 target 的后续条件状态，被拒绝时恢复 VAE decode cache，避免试生成的 draft 污染后续时序状态。
- **单阈值调速**：固定阈值 $\tau$ 是质量—速度旋钮；阈值越宽松，接受率和速度越高，但输出越偏向 drafter。

该方法没有执行传统 speculative sampling 的 target probability verification。ImageReward 只是质量代理，因此更准确的名称是 **reward-guided adaptive model routing**。

## 关键发现

- 在 1003 个 MovieGenVideoBench prompts、$832\times480$、每段 9 个 blocks 的实验中，14B target-only 的 VisionReward 为 **0.0788**，平均耗时 **97.0 秒**。
- $\tau=-0.7$ 时，SDVG 接受 73.1% 的非首 block，VisionReward 为 **0.0773**，保留 target-only 的 **98.1%**，耗时降至 **60.9 秒，即 1.59× 加速**。
- 激进设置 $\tau=-2.5$ 达到 **2.09×**，但只保留 95.7% 的 target 指标质量；仍明显好于 1.3B draft-only 的 0.0644。
- 相同接受率附近，随机 routing 明显差于 reward-guided routing，说明 ImageReward 信号确实承担了选择作用。
- 平均帧分数不如最差帧分数：平均会掩盖单帧 artifact，后者更适合捕捉视频闪烁。

这些实验支持的是“按 block 跳过部分 14B 计算可以形成有效 Pareto frontier”，而不是“输出严格等价于 14B target”。

## 我的提问

### Q1：能否引入传统 speculative decoding，并严格保持 target 分布？

如果使用随机 DDPM/SDE sampler，每个 reverse transition 通常是显式 Gaussian。小模型从 draft transition 提议 state，大模型计算 target transition 后，可以按两者密度比接受；拒绝时再从 target 与 draft 的残差分布采样。已有 continuous speculative sampling 和 Gaussian coupling 工作说明，这在数学上可以保持选定 target sampler 的分布。

但验证必须实际计算 target 的 denoising mean，而且视频 latent 极高维：大小模型的微小均值误差会在维度上累积，导致接受率下降。确定性 DDIM/ODE 或 $\sigma=0$ 的 few-step sampler也不能直接使用 Gaussian rejection sampling。因此“数学上可行”并不等于“在当前 4-step video DiT 上有系统收益”。

### Q2：video DiT 更接近 compute-bound；batch verification 近似线性变慢，draft 还有意义吗？

这正是传统策略的核心障碍。设大模型生成一个单元的成本为 $T$，小模型 draft 成本为 $D$，一次验证 $K$ 个候选的成本为 $V(K)$，平均真正推进 $A$ 个单元，则只有

$$
K D+V(K)<A T
$$

时才能加速，其中 $A\le K$。如果 compute-bound 使 $V(K)\approx KT$，即便全部接受也要额外支付 $KD$，必然慢于 target-only。LLM decoding 常因 memory bandwidth 和权重读取而能廉价验证多个 token；单个视频 latent 已经形成大 GEMM 和昂贵时空 attention，增加 batch 更接近同比增加 FLOPs。

SDVG 仍能加速，是因为 accepted block **完全不运行 target**。其近似 break-even 条件为 $aT>D+R$，其中 $a$ 是接受率，$R$ 是 router 成本。由此看，draft 的价值不在批量 target verification，而在可靠地跳过 target 或只触发局部 correction。

### Q3：这种思路只适合 autoregressive video 吗？一次直接生成 8 秒的 T2V/I2V 是否还有空间？

SDVG 的“逐 block 接受并提交”依赖 causal block 和 KV history，确实主要适用于 autoregressive video。full-clip bidirectional diffusion 没有天然 prefix，所有帧共同去噪，不能直接接受前几秒再让 target 接着生成。

但更广义的 draft-guided compute allocation 仍有空间：

- 按 denoising timestep 切换大小模型，让大模型负责决定语义与运动的关键步骤，小模型负责较容易的步骤；这会真正减少大模型 FLOPs。
- 小模型先生成完整或低分辨率 motion draft，同时预测时空 uncertainty map；target 只修复人脸、手、快速运动、遮挡和 temporal flicker 等困难 tubes。
- 动态降低简单时间段或静态区域的 temporal/spatial resolution，并保留少量全局低分辨率 tokens 维持一致性。
- 用 draft feature 预测哪些 layer、token 或 timestep 必须执行，而不是先完整解码视频再打一个总分。

I2V 可能比纯 T2V 更适合：输入图像已经锚定身份、构图和纹理，剩余不确定性主要集中在运动、遮挡与形变，因而更容易预测需要 target refinement 的区域。前提是系统具有 token packing、sparse attention 或 masked refinement；如果 14B 最终仍对全部 tokens 做 dense forward，就没有真实 FLOP 节省。

## 局限与疑问

- SDVG 不保持 target 分布；接受 draft 会引入朝 1.3B 模型的 distribution shift，不能称为 LLM 意义上的 lossless speculative decoding。
- ImageReward 按 text-image 单帧训练，不能直接判断运动自然度、长时序一致性或跨帧身份漂移；最终评测却使用另一个自动指标 VisionReward，仍缺少充分的人类偏好验证。
- 被拒绝的 block 会浪费 drafter forward 和 VAE decode；首 block 更是已知必拒却仍先 draft，系统上还有明显冗余。
- 实验只覆盖一个 Wan/Self-Forcing 大小模型组合、4-step sampler、一个 benchmark 和两张 A6000；尚不能外推到其他架构、分辨率、视频长度与硬件。
- 论文对“accepted 1.3B latent 如何正确建立 14B target 后续所需状态/KV”的实现描述偏少。不同宽度模型的逐层 KV 无法直接复制，复现时需要特别核查其 cache 构造和实际计时边界。
- 对 full-clip video，整段接受/拒绝过于粗粒度；随着视频变长，只要局部出现坏帧就可能导致整段重算，因此需要时空局部 verifier 与真正稀疏的 target correction。

## 我的判断

论文最有价值的结论不是“video diffusion 也能照搬传统 speculative decoding”，而是：**autoregressive video block 可以成为动态 inference-time compute allocation 的自然单元**。方法足够简单，实验也证明小模型生成多数 block、14B 按需兜底能得到实际 wall-clock Pareto frontier。

但从系统视角，它的收益恰恰来自不做严格 target verification。对 compute-bound video DiT，批量验证很可能接近线性增加计算，因此 exact speculative diffusion 虽然数学上漂亮，却未必比 target-only 更快。更值得继续研究的是低成本 latent verifier、风险可校准的 target-skipping，以及 full-clip I2V 中的 draft-guided 时空选择性 refinement。

当前适合作为动态大小模型协作的 baseline，而不是直接接受其“speculative decoding”命名或投入传统 LLM 式 batch verification。若要复现，第一步应测量 $V(K)/T$ 的真实 scaling、target cache 构造和 verifier 开销，而不是只看接受率。

## 下次只看这些

1. SDVG = 1.3B draft + 最差帧 ImageReward routing + 14B 按需重做；1.59× 来自跳过 target，不是廉价 batch verification。
2. compute-bound video DiT 若满足 $V(K)\approx KT$，传统 exact speculative sampling 即使全接受也会被额外 draft 成本拖慢。
3. full-clip T2V/I2V 更有希望的方向是 timestep routing 与 draft-guided 时空稀疏 refinement，而非整段 target 批量验证。
