---
title: "Why Low-Precision Transformer Training Fails: An Analysis on Flash Attention"
paper_url: "https://arxiv.org/abs/2510.04212"
authors: "Haiquan Qiu, Quanming Yao"
venue: "ICLR 2026"
published: "2026"
read_date: "2026-08-28"
read_at: "2026-08-28T18:46:04+08:00"
status: "已精读"
tags: ["Transformer", "FlashAttention", "Low-Precision Training", "Numerical Stability"]
one_liner: "低精度 Attention 的危险不在单次舍入误差大小，而在重复最大值让多个未归一化权重精确等于 1，与同号 Value 形成有偏误差，再沿跨 Token、跨 Step 相似的低秩梯度方向写入参数并正反馈放大。"
---

## 研究问题

BF16 FlashAttention 训练有时会在正常运行数千步后突然 Loss Explosion，最终可能产生 Inf/NaN。经验方案如改回 Standard Attention、提高精度、QK Normalization 或 QK Clipping 能缓解问题，却没有解释为什么微小的浮点误差可以跨越大量 Token 和训练 Step，最终破坏参数更新。

论文复现一个 GPT-2 训练故障，并追问三件事：错误究竟来自 FlashAttention 的分块、前向输出还是反向公式；BF16 的 Round-to-Nearest, Ties-to-Even 为什么会在 Attention 中产生非零均值偏差；这个局部误差如何通过梯度结构累计成权重谱范数、激活和 Loss 的爆炸。苏剑林的技术文章《[低精度Attention可能存在有偏的舍入误差](https://kexue.fm/archives/11371)》提供了更直观的“两大一小”解释，本笔记结合论文当前版本与阅读问答整理。

## 核心方法

- **逐层替换高精度计算以定位故障。** 单头 Attention 写为

$$
S=\alpha QK^\top,\qquad
\bar P=\exp\bigl(S-\operatorname{rowmax}(S)\bigr),
$$

$$
\bar O=\bar PV,\qquad
O=\frac{\bar O}{\operatorname{rowsum}(\bar P)}.
$$

  其中 $\alpha=1/\sqrt d$，$\bar P$ 是未归一化 Attention 权重。论文依次关闭分块、替换单层与单头、重算反向中间量，最终把直接误差源定位到 BF16 输出 $O$，更具体地说是 $\bar O=\bar PV$ 的最终 BF16 舍入；仅让这一步或问题 Head 的 $O$ 保持 FP32 就能恢复稳定。

- **重复最大值制造两个或多个“未缩小的主项”。** 若某行 $S$ 有两个相同最大值，则对应

$$
\bar P_{t,i_1}=\bar P_{t,i_2}=\exp(0)=1.
$$

  对某个输出维度 $j$，点积成为

$$
\bar O_{t,j}
=\underbrace{V_{i_1,j}+V_{i_2,j}}_{\text{主项}}
+\underbrace{\sum_{i\ne i_1,i_2}\bar P_{t,i}V_{i,j}}_{\text{微小余项}}.
$$

  “两大”不是 $Q,K$ 或两个矩阵，而是两个相对余项更大的 Token 贡献。它们不必绝对数值很大。

- **为什么 $1\times V$ 的位模式特殊。** BF16 只有 7 个显式尾数位。BF16 数转入 FP32 累加器时，缺失的 16 个低位只能补零：

$$
1.b_1b_2\cdots b_7\quad\longrightarrow\quad
1.b_1b_2\cdots b_7\underbrace{00\cdots0}_{16\text{ bits}}.
$$

  当 $\bar P_i=1$ 时，$1\times V_i=V_i$ 精确保留这种规则零尾。两个同号、指数相近的 BF16 主项相加并发生进位归一化后，更容易形成“保留位之后恰为 $1000\cdots$”的中点结构。这里二进制 $0.1000\cdots=1/2$：若尾部小于它则向下舍入，等于它则 Ties-to-Even，大于它则向上舍入。原本极小的同号余项会激活 Sticky Bit，把 $1000\cdots$ 变成 $1001\cdots$，从“正好一半”推到“一半以上”，从而破坏向偶舍入的平衡。

- **$V$ 的符号与低秩结构把舍入噪声变成偏差。** Attention 集中时只有少数主项，其余是大量微小余项；若 $V$ 某个 Feature 上大部分同号，这些余项也会同号，舍入误差便不易正负抵消。论文的具体故障维度中，$V$ 大部分为负，BF16 输出 $O_{lp}$ 系统性地比 FP32 输出 $O_{hp}$ 更负。危险条件不是一般的“Attention 稀疏”，而是重复最大值、同号 $V$、特殊 BF16 位模式和相似梯度方向同时出现。

- **累计发生在参数更新，而不是一次前向输出。** FlashAttention 反向传播使用

$$
\delta=\operatorname{rowsum}(dO\circ O).
$$

  若 $O_{lp}=O_{hp}+\varepsilon$，则

$$
\delta_{lp}-\delta_{hp}
=\operatorname{rowsum}(dO\circ\varepsilon).
$$

  论文观察到问题 Feature 上 $dO$ 与 $\varepsilon$ 符号相关，使这个误差持续偏向同一方向。对 Query 投影权重，梯度差可写成

$$
dW^Q_{hp}-dW^Q_{lp}
=\alpha\sum_{T=1}^{N}
(\delta_{lp}-\delta_{hp})[T]
\,(PK)[T]^\top X[T].
$$

  每一项都是秩一矩阵；论文发现它们在不同 Token 和 Step 间结构相似，可近似为共同方向 $R$。若标量系数也同号，则误差不是随机游走，而是近似沿 $R$ 线性累计。参数漂移继续增大谱范数和 Logit，又会制造更多重复最大值，形成正反馈。

- **用条件式 Softmax 平移移除精确的 1。** 设行最大值 $r_m=\operatorname{rowmax}(S)$，并用容差 $\epsilon$ 统计近似最大值数量 $r_s$。只有当 $r_s>1$ 时，动态选择比最大值更大的平移量 $m$：正最大值使用 $m=\beta r_m$，负最大值使用 $m=0$；否则仍用 $m=r_m$。然后计算

$$
\bar P=\exp(S-m),\qquad \beta>1.
$$

  这样重复最大位置也满足 $\bar P_i<1$。由于 Softmax 对整行常数平移不变，精确数学中的最终 Attention 概率不变；有限精度中的乘积不再是特殊的 $1\times V=V$，并可能让极小余项下溢为零，从而避开该有偏舍入结构。论文使用 $\epsilon\approx10^{-3}$、$\beta\in[2,8]$；$\beta$ 太小可能仍舍回 1，太大则可能下溢。关键不是“乘一个小数就绝对安全”，而是条件、动态地打破多个精确 1 及其规则位模式。

## 关键发现

- 在 12 层、12 Head、隐藏维 768、上下文 1024 的 GPT-2/OpenWebText 复现中，BF16 FlashAttention 在数千步后出现突然 Loss Explosion；把 Block Size 设为整个序列仍然失败，说明 Tiling 不是根因。
- 异常首先集中在第二层 Attention：只在第二层使用 FlashAttention 足以复现失败；只把第二层换成 Standard Attention 则恢复稳定。进一步只将少数异常 Head 的 $O$ 改为 FP32，也足以稳定训练。
- 在失败前的 Step 6580–6680，$\sum_T(\delta_{lp}-\delta_{hp})[T]$ 的累计值持续为正；论文在 Step 6619 的细粒度分析中把偏差定位到 $\bar PV$ 的 BF16 最终舍入，并观察到误差跳变与 $\bar P_i=1$ 的位置重合。
- 重复最大值出现频率在 Loss 爆炸前就开始增加，约 Step 7000 已表现出领先关系，因此它是该复现故障的 Leading Indicator；但一般的低熵或稀疏 Attention 并不等价于即将 Spike。
- 条件式平移让所有触发行的 $\bar P_i<1$ 后，GPT-2S 在 BF16 下分别用 AdamW 与 Muon 训练到 600K Steps 保持稳定；更大的 GPT-2M + AdamW 训练 100K Steps 也未出现原故障。论文还报告该机制在 NVIDIA A100、RTX 4090 和 Huawei Ascend 910B 上具有一致性。

## 我的提问

### Q1：Attention 训练中的“两大”到底是什么？

它们是同一个 Query、同一个输出 Feature 上，由两个强关注 Token 贡献的主加数。两个 Score 并列最大时，未归一化权重都等于 1，主项就是 $V_{i_1,j}$ 与 $V_{i_2,j}$；其他弱关注 Token 构成大量微小余项。最终归一化概率可能接近各 $1/2$，但问题发生在归一化前的 $\bar PV$ 计算中。

### Q2：为什么规则零尾与很小的余项会改变舍入？

用只能保留一位小数的十进制类比，$1.25$ 可写作保留部分 `1.2` 加被舍弃部分 `5000...`，它正好位于 1.2 和 1.3 的中点；加上 $0.001$ 后变成 `5100...`，便必须舍到 1.3。二进制的对应关系是 `1000...` 等于半格，`1001...` 大于半格。小余项的主要作用不是贡献大小，而是充当打破舍入平局的“决定票”。

### Q3：为什么训练中会频繁出现两个最大值？

精确实数中的完全相等很少见，但 BF16 把连续值压到离散格点；数值越大，相邻 BF16 数的绝对间距越大。Attention Sink、相似 Key 和低秩表示又会让顶部 Score 接近，因此原本不同的 Logit 更容易量化成同一最大值。重复最大值可以是故障原因，也可能是参数已经开始漂移的症状；论文证明的是强相关和针对性干预有效，而不是所有重复最大值都会导致崩溃。

### Q4：数值只是稍微偏一点，为什么会累计成 Loss Spike？

累计的不是一次输出值，而是输出误差经 $\delta=\operatorname{rowsum}(dO\circ O)$ 进入梯度后，被优化器反复写进参数。若误差零均值、方向随机，它们大体抵消；本例中误差系数同号，秩一梯度矩阵又跨 Token、跨 Step 相似，因而持续沿同一低秩方向推移权重。谱范数与 Logit 增大后又提高重复最大值频率，最终越过稳定阈值。

### Q5：Loss Spike 就是 NaN，或者预测突然偏离 Target 吗？

两者不是同义词。对交叉熵 $L=-\log p(\text{target})$，有限 Loss Spike 直接表示模型给 Target 的概率骤降，通常是错误 Token 的 Logit 相对 Target 过大。若权重、激活和优化器状态继续发散，才可能进一步出现 Inf，以及由 `Inf - Inf`、`0/0` 等产生的 NaN。MaxLogit 大也不必然意味着高 Loss：如果 Target 仍是最大 Logit，Loss 反而可能很小。

### Q6：看到预警或已经 Spike 后，如何继续训练？

预警阶段优先让问题层或 Head 的 $\bar PV$ 输出保持 FP32，或实现论文的条件式 Softmax 平移。若已经发生持续 Loss Explosion，应从 `multiple-max rate`、MaxLogit、谱范数等指标尚未漂移的最后一个健康 Checkpoint 恢复完整模型、优化器与调度器状态，再应用修复；不要只回退到 Loss 跳变前一两个 Step。若参数或 Adam Moment 已含 Inf/NaN，必须回滚。降低学习率、Gradient Clipping 或跳过单个 Batch 只能作为护栏，不能消除持续的有偏舍入；论文复现本身已使用 Global Gradient Clipping，仍然失败。

## 局限与疑问

- 论文核心因果链来自一个特定 GPT-2 故障，虽然补充了不同硬件、优化器和模型尺寸实验，也在 Llama-3.1-8B 中观察到重复最大值，但尚不能推出所有架构、真实大规模训练或 FP8 的 Loss Spike 都由同一机制产生。
- 重复最大值、Attention 集中与崩溃之间仍有部分因果歧义：它可能触发有偏舍入，也可能说明模型已进入不稳定区域。健康模型同样可能存在 Attention Sink 或多个最大值，因此不能把单次 `multiple-max` 当作充分条件。
- 论文的修复针对“多个精确 1 + 同号 $V$ + 有偏 BF16 舍入”这一具体结构；固定 Offset、无条件缩放或过大的 $\beta$ 可能引入新的舍入偏差或下溢。
- 实际 FlashAttention CUDA Kernel 的 FP32 Accumulator、Fused Multiply-Add、Block 顺序和输出 Cast 细节都会改变误差。将论文的 PyTorch 复现结论迁移到生产 Kernel 前，仍应在目标硬件和实现上比较 BF16/FP32 的有符号输出差与训练轨迹。
- 更值得监控的不是泛化的“Attention 是否稀疏”，而是逐层逐 Head 的重复最大值率、Top-1/Top-2 Gap、Attention Entropy、MaxLogit、$W_Q/W_K$ 谱范数，以及 $O_{lp}-O_{hp}$ 是否出现持续的有符号偏差；这些指标的阈值仍需按模型校准。

## 我的判断

这篇论文最有价值的地方，是把“浮点误差很小所以可当噪声”的直觉拆成两个独立问题：误差是否零均值，以及梯度误差方向是否会跨样本和 Step 对齐。它给出的链条——重复最大值产生多个 $\bar P_i=1$，同号 $V$ 与 Sticky Bit 造成有偏舍入，$\delta$ 把误差送入相似低秩梯度方向，参数漂移再放大 Logit——既能解释突然 Spike，也给出了可证伪的分层诊断方法。

针对该复现故障，证据较强：高精度替换定位到单层、单 Head 和 $\bar PV$，针对性修改又跨优化器与模型尺寸阻止了爆炸。不过，生产训练中应先做可逆的精度 Canary：从 Spike 前 Checkpoint 重放相同 Batch，只把可疑层的 $\bar PV$ 保持 FP32；若重复最大值、输出有符号偏差和谱范数漂移同步消失，再考虑定制 Kernel。比起立刻修改架构，这是风险最低、最能确认根因的路径。

## 下次只看这些

1. 两个相同最大 Score 会制造两个 $\bar P_i=1$，使主项退化为规则 BF16 格点上的 $V_i$；同号小余项通过 Sticky Bit 把中点舍入持续推向一侧。
2. 真正累计的是梯度：$O$ 的有符号误差进入 $\delta$，再沿跨 Token、跨 Step 相似的低秩方向写进参数，形成“谱范数/Logit 增大 → 重复最大值更多 → 偏差更强”的正反馈。
3. 发现预警先把可疑 $\bar PV$ 改为 FP32；已经持续 Spike 或出现 NaN，则从指标尚未漂移的健康 Checkpoint 回滚，修复后再恢复完整训练状态。
