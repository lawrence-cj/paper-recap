---
title: "Addressable Memory for Video World Models"
paper_url: "https://arxiv.org/abs/2608.07408"
authors: "Xindi Wu et al."
venue: "arXiv"
published: "2026"
read_date: "2026-08-10"
read_at: "2026-08-10T18:17:10+08:00"
status: "已精读"
tags: ["Video Generation", "World Models", "KV Cache", "Long Video"]
one_liner: "WorldTrace 将淘汰历史的 Key 先撤销 temporal RoPE、在 canonical space 压缩，再按固定 slot rank 映射到训练范围内的虚拟位置，使定长 KV cache 仍能读取远期视觉记忆。"
---

## 研究问题

自回归视频 world model 逐 chunk 生成未来，并通过 KV cache 携带历史。保留全部 KV 会让显存和 attention 成本随时间线性增长；sliding window 虽然定长，却会直接丢弃已经离开的场景。直觉上的折中是把远期历史压缩成少量 summary slots，但论文指出，长期记忆包含两个耦合问题：

- **Addressability**：历史即使仍在 cache 中，temporal RoPE 的 query-key offset 一旦超出训练范围，模型也可能无法稳定读取；把所有超长 offset 截到同一个边界位置，又会让多个 summary slots 在位置上不可区分。
- **Informativeness**：若直接平均已经施加 RoPE 的 Keys，不同时间相位会部分抵消；summary 可能被成功寻址，却已不再保留有效内容。

因此目标不是单纯“存更多历史”，而是在固定 cache 预算内同时决定 **记忆存什么** 和 **以什么位置被读取**，使长 rollout 中的场景既能连续演化，也能在离开后被重新生成。

## 核心方法

- **固定预算、推理期划分**：将训练时的 local attention window 拆成 $N_s$ 个 summary slots 和 $N_r$ 个 recent slots，满足 $N_s+N_r=L_{\mathrm{attn}}$。Recent 原样保存最近 latent frames；只有在 recent overflow 后，最老内容才被写入 summary。划分是 inference-time hyperparameter，不需要重新训练模型。
- **Slot-rank virtual position**：当前 query 位置为 $q$ 时，第 $s$ 个 summary slot 使用

$$
t_s^v=q-\left(L_{\mathrm{attn}}-1-s\right),
\qquad s=0,\ldots,N_s-1.
$$

  这样每个 summary slot 都有不同位置，且它到当前 query 的相对距离只由 slot rank 决定，始终落在训练 offset 内，不随真实 rollout 长度增长。
- **Canonical Key compression**：对真实时间为 $t_m$ 的 RoPE-rotated Key $K_{t_m}^f$，先用负角度撤销第 $f$ 个 temporal frequency 的旋转，在无时间相位的 canonical space 中平均，再旋转到 slot 的虚拟位置：

$$
K_{\mathrm{field}}^f(t_s^v)
=R(\theta_f t_s^v)
\frac{1}{M}\sum_{m=1}^{M}
R(-\theta_f t_m)K_{t_m}^f.
$$

  Re-rotate 的方向不是由内容推断，而是直接使用基础模型在目标位置 $t_s^v$ 上的正常 forward RoPE。Value 不带 RoPE，按来源帧直接平均。
- **WorldTrace-Field**：把远期历史分成连续时间 buckets，每个 slot 保存一段历史的 canonical Key/Value 均值。它适合 future attention 平滑分布在历史上的情形，主要改善长时间连续性和 scene drift。
- **WorldTrace-Landmark**：相邻 canonical Keys 的 cosine distance 出现峰值时，将该帧视为 scene entry，并把其 canonical Key 原样冻结在 landmark slot 中；读取时从这份 frozen canonical Key 一次性旋转到当前虚拟位置。它适合 attention 集中于少数旧场景的 episodic recall，也避免 BF16 下反复 unrotate/re-rotate 的累计误差。
- **Structured sparse attention 视角**：两种 writer 都是在用固定大小的投影矩阵 $P$ 近似 full-history attention。Field 的一行 $P$ 平均连续时间组；Landmark 的一行只选择一个关键帧。前者近似 smooth historical attention，后者近似 sparse recall query。
- **LoopBench**：让模型沿 ABA、ABCA、ABCDA 等路径离开并返回先前位置，以第一次生成的场景作为 reference，对齐返回位置后计算 Position-Aligned CLIP；这比只检查相邻帧平滑度更直接地测试“模型是否记得去过哪里”。

## 关键发现

- 在 MG2-1.3B 上，$N=48$ chunks 时 sliding window 的 TempSSIM 为 **0.472**、Scene Drift 为 **0.0305**；WorldTrace-Field 达到 **0.545** 和 **0.0295**，即 temporal consistency 相对提升 **15.5%**。
- LoopBench ABA 中，WorldTrace-Landmark 在 $N=16$ 时将 PAC 从 **0.723 提升到 0.864**，相对提升约 **19.5%**；在 $N=32$ 时从 **0.627 提升到 0.825**。完整 $360^\circ$ camera pan 的增益明显缩小，说明大视角变化仍然困难。
- Canonical averaging 的 phase-cancellation ablation 中，$N_s=4$ 时 LatentDiff 从 naive RoPE-space averaging 的 **0.312 降至 0.233**，下降 **25.3%**，直接支持“先 unrotate 再压缩”的必要性。
- 在带 Plücker camera conditioning 的 14B LingBot-World 上，WorldTrace-Landmark 从 $4\times$ training horizon 开始仍优于 sliding window，$4\times/6\times/8\times$ 的相对增益分别为 **8.9%/14.1%/7.3%**；WorldTrace-Field 则大体接近 baseline，说明 camera-pose prior 已覆盖部分连续性信号。
- 主实验的 per-chunk runtime 相对 sliding window 增加不超过 **6%**。但默认 Field recompute writer 会把淘汰的 canonical Keys 存到 CPU，每个 generated latent frame 增加约 **5.4 MB** host memory；真正 $O(N_s)$ 状态的 streaming writer 在 $N=48$ 时仍取得 0.538 TempSSIM，略低于 recompute writer 的 0.545，但保留了大部分收益。

## 我的提问

### Q1：Unrotate、压缩、re-rotate 在实现中到底怎么做？Re-rotate 的方向由什么决定？

每个 temporal RoPE 二维 pair 都先按自己的原始时间 $t_m$ 使用负角度旋回 canonical coordinates。多个来源 Key 只有在这个共同坐标系中才能安全平均。压缩后直接调用基础模型原本的 RoPE operator，将 canonical summary 旋转到由当前 query 和 slot rank 共同确定的 $t_s^v$；因此没有需要从内容预测的“方向”。若只保存一个 landmark，整个操作等价于把原 Key 净旋转 $\theta_f(t_s^v-t_m)$，但实现上应永久保存 canonical Key，每次 fresh rotate，避免低精度累计误差。

### Q2：最开始没有 summary，随着 summary 出现，index 为什么不会错位？

需要区分真实时间、cache slot rank 和用于 RoPE 的 virtual position。以 $L_{\mathrm{attn}}=4$、$N_s=1$、$N_r=3$ 为例：

| 当前位置 $q$ | S0 的真实来源 | S0 的 virtual position | Recent 内容 | 模型看到的位置 |
|---:|---|---:|---|---|
| 0 | 空 | — | `[0]` | `[0]` |
| 1 | 空 | — | `[0,1]` | `[0,1]` |
| 2 | 空 | — | `[0,1,2]` | `[0,1,2]` |
| 3 | frame 0 | 0 | `[1,2,3]` | `[0,1,2,3]` |
| 4 | frames 0–1 的摘要 | 1 | `[2,3,4]` | `[1,2,3,4]` |
| 5 | frames 0–2 的摘要 | 2 | `[3,4,5]` | `[2,3,4,5]` |

Summary 的 virtual position 确实每步加一，但 query 也同时加一，所以二者的相对距离始终为 3。真实旧内容被有意映射到当前窗口最左侧的“老历史邮箱”；它不再表达准确年龄，而是换取训练分布内的稳定可读性。Summary 保存 canonical Key，位置在每次 attention 时从当前 $q$ 重新计算，因此不会继承上一步的虚拟相位。

### Q3：Summary 和 recent 的数量必须在训练时确定吗？

训练只决定总的有效窗口和最大可信 offset；$N_s/N_r$ 的拆分在 rollout 前由 inference policy 决定。论文在总预算 6 下，短时 coherence 实验用 $2+4$，长时 recall 实验用 $4+2$，无需重新训练。Summary 不是额外增加的 cache：增加一个 summary slot 就要减少一个 recent slot，因此长期回忆和短期运动细节之间存在直接预算权衡。动态改变划分在原则上可行，但需要从 canonical history 重新构造 buckets 和 virtual positions；论文当前主要使用固定拆分，并把自适应策略留作未来工作。

## 局限与疑问

- 方法依赖带 temporal RoPE、固定 local window 的自回归生成器；对 bidirectional video diffusion、无 RoPE 架构或隐式状态模型不能直接套用。
- Virtual position 丢弃了历史的真实年龄。它适合“是否还能读到旧场景”，但对依赖准确时间间隔、速度或周期的动态推理可能产生错误语义。
- Field 用固定 slots 平均越来越长的历史，具体物体和局部几何必然逐渐模糊；其 mean-attention 保证只成立在 pre-softmax score 层面，不等价于完整 attention weights 或最终生成保持不变。
- Landmark 只能可靠保留不超过 $N_s$ 个场景，且依赖简单的 canonical-Key distance threshold。缓慢场景切换、同一地点的大视角变化以及更重要但无明显 feature jump 的事件都可能漏检。
- 默认 Field recompute writer 的 GPU cache 是定长的，但 CPU history 仍线性增长；部署时应优先评估 streaming writer，而不能只引用“bounded memory”。
- LoopBench 以模型自己的第一次生成为 reference，PAC/TempSSIM 能测外观回忆，却不能证明 3D geometry、object state 或物理世界状态真正一致。

## 我的判断

论文最有价值的不是一个具体 cache heuristic，而是把长期记忆拆成 **addressability** 与 **informativeness**，并指出 position assignment 和 content compression 必须联合设计。对任何带 RoPE/Fourier/pose encoding 的视频缓存，先消除位置相位、在 canonical coordinates 聚合、再映射到目标地址，是非常可迁移的工程原则。

Field 与 Landmark 的对照也说明：连续性记忆和情景回忆需要不同 writer。未来更完整的视频 world model memory 应同时包含 recent working memory、平滑的 field memory、稀疏 landmark memory，以及 geometry-aware retrieval，而不是寻找一个统一平均策略。

对当前长视频生成与 inference acceleration 方向，这篇论文与 sparse attention、KV compression 和 autoregressive forcing 直接相关，值得先复现 canonical Key cache 和 slot-rank remapping；但它仍是 training-free patch，不会改善基础模型本身的单 chunk 画质、action following 或动力学准确性。

## 下次只看这些

1. 长期记忆要同时解决“存了什么”和“还能不能被寻址”；只压缩 KV 不处理 RoPE position 不够。
2. Summary 的真实来源可以很老，但 virtual position 永远位于当前训练窗口内：canonical Key → 压缩 → 按 slot rank fresh re-rotate。
3. $N_s+N_r$ 是固定预算；Field 负责连续性，Landmark 负责具体旧场景，两者的 slot 配比决定长期回忆与短期细节的取舍。
