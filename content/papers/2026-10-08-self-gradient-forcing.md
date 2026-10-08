---
title: "Self Gradient Forcing: Native Long Video Extrapolation"
paper_url: "https://arxiv.org/abs/2607.20368"
authors: "Junhao Zhuang et al."
venue: "arXiv"
published: "2026"
read_date: "2026-10-08"
read_at: "2026-10-08T20:45:55+08:00"
status: "已读"
tags: ["Video Generation", "Autoregressive Generation", "Streaming Video", "Diffusion Models", "Knowledge Distillation"]
search_terms: ["Video Generation", "Autoregressive Generation", "Knowledge Distillation"]
one_liner: "先无梯度自回归生成并记录 exit-step 输入，再用干净历史重建可求导的 KV，让未来 DMD 损失训练历史写入；恢复的是记忆编码梯度，不是整条采样轨迹的梯度。"
---

## 研究问题

Self Forcing 让学生读自己生成的历史，缓解训练时读真实视频、推理时读生成视频的差异。但历史 KV 被 detach 后，后续损失只能训练当前块怎样读取历史，不能训练过去的画面怎样被编码成有用的 KV。共享参数仍会随去噪更新而变化，历史写入却没有来自未来预测的直接纠正。

SGF 解决这个 historical context-gradient gap：不保留完整串行 rollout 的反向图，仍让未来生成的损失更新历史编码过程。重点是“画面生成得好”与“这幅画面的历史表示有助于后续生成”是两件事。

阅读依据：[论文 v1 §3、Algorithm 1、Appendix D](https://arxiv.org/html/2607.20368v1)、[官方项目页](https://zhuang2002.github.io/SelfGradientForcing/)及 [SGF 官方代码](https://github.com/zhuang2002/Self_Gradient_Forcing/tree/ba16e1b70d537b2b9f542efc9c6ef4cef1819d8c)。下面将论文算法与 2026-10-08 的公开代码快照区别说明。本次未运行训练复现。

相关笔记：[SGF+：把历史写入和当前去噪参数分开](https://junsongc.top/paper-recap/#paper=self-gradient-forcing-plus)、[Vidu S2 的 Self-Replay Forcing](https://junsongc.top/paper-recap/#paper=vidu-s2)。SGF 本身没有新增一套独立 Memory 参数。

## 核心方法

![SGF 方法图：Self Forcing 的冻结历史 KV 与 SGF 的可求导历史重建，包含 clean/noisy 两路的因果 attention mask](media/self-gradient-forcing/method-overview.webp "论文 Figure 2 对应的官方方法图：恢复历史 KV 的跨块梯度。来源：zhuang2002/Self_Gradient_Forcing，assets/flowchat.jpg；仓库 Apache-2.0；转为 WebP，未改动图示内容。")

图源：[官方资产](https://github.com/zhuang2002/Self_Gradient_Forcing/blob/ba16e1b70d537b2b9f542efc9c6ef4cef1819d8c/assets/flowchat.jpg)，许可：[Apache-2.0](https://github.com/zhuang2002/Self_Gradient_Forcing/blob/ba16e1b70d537b2b9f542efc9c6ef4cef1819d8c/LICENSE)。随图保留仓库许可文本；论文 PDF 的发行许可与仓库资产许可分开看。

### 1. 每个块有去噪和历史写入两种调用

记第 $i$ 块的带噪输入为 $z_i^t$，网络预测的干净 latent 为 $\tilde x_i$。历史写入调用在 $t_{\mathrm{ctx}}=0$ 编码这个干净预测：

$$
\mathrm{KV}_i=\mathcal C_\theta(\tilde x_i,0\mid\mathrm{KV}_{<i}).
$$

$\mathcal C_\theta$ 是同一 DiT 的 clean-context 计算；去噪调用记为 $\mathcal D_\theta$。这里是两种角色，共用参数 $\theta$，不是两个独立模型。KV 是每层 attention 的内部特征，不是另生成的一段视频。

### 2. 第一遍：无梯度自生成，记录选定步骤

从 prompt 和随机噪声开始，按 A → B → C 串行生成。每处理完一个块就写历史 KV，后一个块才能继续生成；不是等整段视频结束再统一写 cache。

从实际少步去噪 schedule $\mathcal T=(t_1,\ldots,t_K)$ 中抽一个索引：

$$
s\sim\operatorname{Uniform}\{1,\ldots,K\},\qquad t^\star=t_s.
$$

对各块保存同一选定步骤的带噪输入和干净预测：

$$
Z_i^\star=z_i^{t^\star},\qquad X_i^\star=\hat x_{0,i}^{(s)}.
$$

$X_i^\star$ 是该步骤预测的无噪声估计，不一定是走完所有步骤后的结果。它被送入 $t=0$ 的历史编码调用，供后续块使用。所有记录与串行 cache 都不保留梯度。

### 3. 第二遍：一次并行 forward，重建可求导历史

保存的 clean 与 noisy 两路一起输入，通过专门的因果 mask 重算：

$$
M_\theta=\mathcal C_\theta(\operatorname{sg}(X^\star),0;\mathcal M_{\mathrm{rec}}),
$$

$$
\hat X=\mathcal D_\theta(Z^\star,t^\star\mid M_\theta;\mathcal M_{\mathrm{rec}}),\qquad
\mathcal L=\mathcal L_{\mathrm{DMD}}(\hat X).
$$

$\operatorname{sg}$ 表示 stop-gradient；历史 latent 固定，但重建历史的 hidden states、K/V projection 和 attention 保留梯度。未来损失可以更新历史写入：

$$
\mathcal L_B\longrightarrow K_A,V_A\longrightarrow\theta.
$$

它不会回到第一遍生成 A 的多步采样过程。第二遍也不拿自己新预测出的 B 去生成 C；C 使用保存的 $X_B^\star$ 所构建的历史表示。这让固定窗口内的重建可以并行。

### 4. 因果由可见范围决定，不要求逐块调用网络

下表忽略 sink/window 对可见历史的进一步限制。clean 和 noisy 对应同一条轨迹的同一批时间位置，但不能互相随意读取：

| Query 所在块 | 能读的 clean blocks | 能读的 noisy blocks |
| --- | --- | --- |
| clean A | A | 无 |
| clean B | A、B | 无 |
| clean C | A、B、C | 无 |
| noisy A | 无 | A |
| noisy B | A | B |
| noisy C | A、B | C |

noisy B 看不到 clean B，因此不会直接读到自己的干净答案。clean B 不能读 noisy B，也不能读未来 clean C，避免历史表示间接泄漏答案。同一个块内部允许交互，因果粒度为 block。

在每一层，所有位置上一层的 hidden state 已经就绪，可以同时计算 Q/K/V，再用 mask 控制访问；层与层仍顺序执行。并行时间位置和 causal attention 不矛盾。[Mask 实现](https://github.com/zhuang2002/Self_Gradient_Forcing/blob/ba16e1b70d537b2b9f542efc9c6ef4cef1819d8c/wan/modules/causal_model.py)

### 5. DMD 监督新预测，不拟合第一遍输出

DMD 使用冻结的 real-score teacher 和单独训练的 fake-score network；后者跟踪当前学生的生成分布。对第二遍的干净预测 $\hat X$ 另采噪声时间 $u$，加噪后让两个 score 网络预测干净结果 $R_u,F_u$。$u$ 与 rollout 的 $t^\star$ 分开采样。

官方实现以归一化的干净预测差构造梯度，再用 surrogate loss 注入生成器：

$$
g=\operatorname{Normalize}(F_u-R_u),\qquad
\mathcal L_{\mathrm{DMD}}=\frac12\operatorname{mean}
\left[\left\|\hat X-\operatorname{sg}(\hat X-g)\right\|^2\right].
$$

这里只展示主要计算，省略 CFG 等细节。目标中的 $\hat X-g$ 已 detach；它不是 paired ground-truth 视频，也不是第一遍保存的 $X^\star$。生成器更新时 score 计算不保留梯度，梯度通过第二遍 $\hat X$ 更新去噪与历史写入。没有额外的 KV loss。[DMD 实现](https://github.com/zhuang2002/Self_Gradient_Forcing/blob/ba16e1b70d537b2b9f542efc9c6ef4cef1819d8c/model/dmd.py)

## 关键发现

- 训练 rollout 窗口为 5 秒，正式评测覆盖 5、60、240 秒；60 秒用 VBench-Long prompts，240 秒用 MovieGen-128。匹配 SF/SGF 对照保持初始化、prompt、seed、sink/FIFO 和采样设置一致。
- 原论文 Table 2 的 TF 初始化、60 秒 chunkwise：subject consistency 为 0.951 → 0.982，aesthetics 为 0.582 → 0.654，dynamic degree 为 0.909 → 0.634。稳定性与运动幅度需同时检查。
- Table 3 的超过 1,900 次成对人工判断中，TF 初始化、240 秒 chunkwise 的 GSB 为 +44.9%。GSB 是 $(G-B)/(G+S+B)$，不能写成 44.9% 偏好率。
- Table 4 所测设置：SF/SGF 峰值显存 79.01/87.01 GB，稳定显存 79.01/63.73 GB，每五训练步耗时 10.39/11.71 秒；直接保留可求导串行 cache 的版本 OOM。这里不是 SGF+ Table 3 的成本口径，不能混用。
- Appendix D 与项目页报告 96 组 prompt/exit 对比的重建平均 relative L2 为 0.01409、cosine 为 0.999886。这说明第二遍近似恢复选定步骤的前向计算，不是说 loss 要拟合第一遍，也不证明所有实现下严格相等。

### 5 秒训练如何支持更长推理？

模型反复执行局部的“读历史 → 生成 → 写历史”，推理只保留 sink 加最近历史，KV 大小有界；位置编码也需要限制在配置范围内。官方 framewise 配置为 4 sink + 16 FIFO + 1 current = 21 latent 时间帧；chunkwise 为 3 + 6 + 3 = 12。单位不是解码后的 RGB 帧。

公开长视频实现存储未应用 RoPE 的 K，每次读取按保留窗口重新施加时间位置；`top_aligned` 把 sink 放在开头、近期窗口限制在配置的位置范围内。固定窗口使持续运行可行，SGF 改善反复执行时的质量；总计算和输出存储仍随长度增加。[推理脚本](https://github.com/zhuang2002/Self_Gradient_Forcing/blob/ba16e1b70d537b2b9f542efc9c6ef4cef1819d8c/scripts/infer_self_gradient_forcing.sh)

“只训练 5 秒”指这一阶段的 rollout 长度，不代表从零只学过 5 秒数据。它复用 Wan 预训练模型和 Causal Forcing 初始化，最终提示词驱动的 DMD 阶段不要求真实视频作为当前 rollout 输入。[训练与初始化](https://github.com/zhuang2002/Self_Gradient_Forcing#-download-weights)

## 我的提问

### Q1：第一遍用真实视频加噪，还是 noise + prompt？

本次讨论的 T2V self-rollout 用随机噪声和 prompt，历史来自当前模型自己生成的块。此前的模型预训练与初始化阶段另有数据需求，不应混为一谈。

### Q2：干净预测的 timestep 是多少？预测步骤和写 KV 步骤为什么不同？

带噪输入 $Z_i^\star$ 在 $t^\star$ 送入去噪角色，输出是无噪声估计 $X_i^\star$；后者在 $t=0$ 送入历史写入角色。两次收到的 latent 不同，不能理解成同一份带噪输入被随意改标为 0。第 2 个步骤也不是 timestep 数值 2。

### Q3：干净 A、B 会过两次历史编码吗？第一遍 KV 哪里用？

会。第一次无梯度写 KV，A 帮助生成 B，B 帮助生成 C；第二次用保存的同一份 clean latent 重建带梯度的 KV。第一遍 KV 已在第一遍内部用过，第二遍不直接复用它。若最后一个块之后没有下一块，它写出的 KV 就没有后续消费者。

Memory 的主要用途是生成各层内部 K/V，不是再输出视频；可解码的视频 latent 来自去噪角色。

### Q4：两遍都输入同一段视频，为什么不会偷看答案？

第二遍包含同一轨迹的 clean 与 noisy 两组状态，但 noisy B 只能读 clean A 和自身 noisy B；clean B 留给 C 使用。严格的跨角色 mask 才是关键，不能简单把两组 tokens 拼接后套普通下三角 mask。

### Q5：第二遍是依次 forward，还是一次 forward？

一次并行重建，仍使用 block-causal mask。第二遍所需历史 latent 已由第一遍提供，不等待第二遍的新输出。损失可跨块进入历史表示，但不穿越 detach 边界回到原 rollout；不是完整 BPTT。

### Q6：记录步骤怎样选？所有块是否相同？

从实际少步 schedule 的索引均匀抽取，同一 rollout 的所有块共用这个 exit；当前默认允许不同 GPU rank 独立抽取。随机覆盖不同去噪阶段，同时每次只对一个选定步骤重建带梯度的 forward。配置中的 `1000,750,500,250` 还会经 scheduler 映射，不是从全部 1,000 个时间点任意抽。

论文 Algorithm 1 描述到 exit 停止；2026-10-08 公开代码在默认 `per_rank_exit_step: true` 时会继续跑完各块 schedule，但 `cache_mode: exit` 仍选 exit 的干净估计写历史。关闭该选项时，exit 模式会在记录步骤 `break`。因此“计算跑了几步”和“哪一步结果被传给下一块”必须分开看。[对应代码](https://github.com/zhuang2002/Self_Gradient_Forcing/blob/ba16e1b70d537b2b9f542efc9c6ef4cef1819d8c/pipeline/self_gradient_forcing_training.py#L142)

### Q7：保存的输入、输出分别去哪里？

| 第一遍记录 | 第二遍角色 | timestep | 后续作用 |
| --- | --- | --- | --- |
| $Z_B^\star$，带噪输入 | 去噪 | $t^\star$ | 重算 B 的预测并接受 DMD |
| $X_B^\star$，干净估计 | 历史写入 | 0 | 重建 B 的 KV，供 C 读取 |

选中步骤改变这两组状态，也改变第一遍后续块读取的历史。第二遍继续使用选定 $t^\star$，不是再抽一个 rollout 步骤。

### Q8：与 Vidu S2 的两次前向哪里不同？

用户指出两者很像。这个观察成立，但对应的 Vidu 方法是 [Self-Replay Forcing（SRF），§2.2](https://arxiv.org/html/2609.11638v1#S2.SS2)。共同点是先无梯度生成，再对 detach 状态做带梯度 causal replay，恢复 replay 范围内的跨块梯度。

| 比较点 | Vidu S2：SRF | SGF |
| --- | --- | --- |
| 第二遍当前输入 | 自生成轨迹重新加噪 | 第一遍实际记录的 exit noisy 输入 |
| 第二遍内部历史 | 重新加噪的过去块产生的表示/KV | 保存的干净预测在 $t=0$ 重建的表示/KV |
| 前向关系 | re-noised video 的 block-causal replay | clean-history 与 noisy-target 两路重建 |
| 与第一遍对应 | 输入经过新的噪声变换，不要求还原原采样步骤 | 设计目标是恢复选定 exit 的前向计算 |
| 损失 | DMD + perceptual loss | 沿用 DMD，无新增辅助 KV loss |
| 参数 | SRF 公式使用同一学生 $f_\theta$；未报告独立 writer | 写历史与去噪共享 $\theta$ |

用 A → B 表示，SGF 的 B 损失通过 **干净 A 的编码**传回；SRF 则通过 **replay 中重新加噪 A 的表示**传回。SRF 范围外继承的历史仍 detach。SRF 也训练内部历史表示，不能说它完全没有 memory 梯度。

因此区别先在 replay 的输入和梯度路径，再在 SGF+ 新增的角色参数分离。Vidu S2 还有用于超分的 Refiner，不能把它当 SGF+ 的 context writer。SRF 的具体噪声采样、teacher 与 replay 长度公开不全；没有控制实验能说明 SGF 必然优于 SRF。

### Q9：SGF 和 SRF，哪一个更 elegant？

Agent 的阅读偏好是 **原始 SGF 在问题与改动的对应关系上更清楚**：先指出历史 KV 被 detach 导致的监督缺口，再在固定 rollout 输入下重建这一条梯度路径；不新增 loss，也不要求整条采样轨迹反传。保留实际 exit 输入，也便于检查第二遍是否近似恢复原前向。

SRF 的单一 noisy stream 更容易表达和实现：给自生成轨迹重新加噪，以 causal replay 训练。但噪声构造改变了第二遍输入，历史处于带噪条件，解释效果时还需区分跨块梯度与重加噪本身的影响。SGF 的代价则是 clean/noisy 双流及专门 mask，实现更复杂。

如果 elegant 指最少的输入流与 mask，SRF 更占优；如果指针对所缺梯度作明确修改，Agent 更偏向 SGF。这是设计判断，没有由此证明 SGF 的效果优于 SRF。SGF+ 又是另一个取舍：角色分离容易理解，但完整参数翻倍，不宜直接沿用对原始 SGF 的成本判断。

## 局限与疑问

- 原 rollout 的画面 latent 和采样路径仍 detach；恢复的是固定输入下的历史编码梯度，不是长期事件决策的完整 credit assignment。
- 有限 sink/FIFO 没有保存全部历史。分钟级视觉稳定不等于长期剧情记忆、任意内容都能无限生成，也不等于人物动作始终正确。
- Dynamic Degree 下降可能包含减少不连贯跳变，也可能包含运动减少。论文的解释不能替代动作完成度和运动真实性检查。
- exit clean estimate、最终 clean estimate、干净 KV 与 noisy KV 是不同设计选择；当前代码与论文的停止时机也有差异，复现前应固定版本和这些配置。
- 与 SRF 的比较是机制分析，不是同教师、数据、算力下的效果对照。应分别消融输入重加噪、历史噪声、KV 梯度和 rollout/replay 长度。

## 我的判断

用户通过连续追问，最终明确了：保存的干净预测过两次历史编码，第一次帮助第一遍生成后续块，第二次建立梯度；第二遍当前输入来自保存的 noisy latent，clean/noisy timestep 各有作用。还提出了与 Vidu S2 SRF 的两遍训练相似性。

Agent 的阅读判断：SGF 最值得借鉴的是 detach 边界的选择。先确认历史写入是否能接受未来损失，再考虑更复杂的缓存或参数分离。优先按相同初始化、prompt、seed 比较冻结 KV 与可求导重建，在 5/60/240 秒同时检查运动、身份、背景和画质。用户未决定复现预算与投入优先级。

## 下次只看这些

1. **两遍分工**：第一遍 noise + prompt 串行生成，保存 exit 的输入与干净估计；第二遍并行重建，clean 用 0，noisy 用 $t^\star$。
2. **梯度边界与 mask**：latent detach，重建 KV 不 detach；noisy B 读 clean A，不能读 clean B，B 的损失训练 A 的历史编码。
3. **与 SRF/SGF+ 的区别**：SRF 重加噪轨迹，SGF 重放实际 exit 并重建 clean 历史，SGF+ 再分离 writer/denoiser 参数。
