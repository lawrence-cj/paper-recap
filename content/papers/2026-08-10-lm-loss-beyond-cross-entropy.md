---
title: "除了交叉熵，LM Loss还有什么选择？"
paper_url: "https://kexue.fm/archives/11854"
authors: "苏剑林"
venue: "科学空间（技术文章）"
published: "2026"
read_date: "2026-08-10"
read_at: "2026-08-10T22:06:45+08:00"
status: "已精读"
tags: ["Language Models", "Loss Functions", "Proper Scoring Rules", "Optimization"]
one_liner: "LM Loss 必须能从单个 Token 样本无偏估计完整目标分布；所有合格选择可由凹的广义熵构造，但只有配套激活函数后才能判断优化优劣，Softmax 与交叉熵正是梯度最干净的天然配对。"
---

## 研究问题

语言模型表面上是逐 Token 分类，实质上却要学习一对多的条件分布：同一上下文可以有多个合理后继。训练语料不会显式给出真实分布 $\boldsymbol p$，每次只提供一个采样 Token $i\sim\boldsymbol p$，因此并非任意分类损失或分布距离都能用于 LM 训练。

文章要回答三个问题：哪些 Loss 既能由零散 Token 样本估计，又保证预测分布 $\boldsymbol q$ 收敛到 $\boldsymbol p$；为什么交叉熵在这些选择中格外特殊；替换 Loss 时，为什么还必须同步考虑从 Logits 到概率的激活函数。

## 核心方法

- **从“学习分布 + 允许采样”刻画 LM Loss。** 合格的总体风险必须关于目标分布 $\boldsymbol p$ 线性，从而能由单个观测 Token 给出无偏随机估计；同时，真实分布必须是其最优预测（严格版本要求最优解唯一）：

$$
L(\boldsymbol p,\boldsymbol q)
=\sum_{i=1}^n p_i S(\boldsymbol q,i)
=\mathbb E_{i\sim\boldsymbol p}[S(\boldsymbol q,i)],
\qquad
\boldsymbol p
=\underset{\boldsymbol q\in\Delta^{n-1}}{\arg\min}\,
L(\boldsymbol p,\boldsymbol q).
$$

  这里 $\boldsymbol p$ 是真实条件分布，$\boldsymbol q$ 是模型预测，$S(\boldsymbol q,i)$ 是观测到 Token $i$ 时计算的单样本 Loss。Total Variation 等距离不满足可用的采样形式，因为单样本项仍依赖未知的 $p_i$。

- **用凹的广义熵统一构造 Proper Scoring Rules。** 定义正确预测时的最小风险 $H(\boldsymbol p)=L(\boldsymbol p,\boldsymbol p)$。固定 $\boldsymbol q$ 后，$L(\boldsymbol p,\boldsymbol q)$ 是 $H(\boldsymbol p)$ 在 $\boldsymbol p=\boldsymbol q$ 处的支撑超平面，因此 $H$ 必须是凹函数。反过来，任取可微凹函数 $H$，都可构造

$$
S(\boldsymbol q,i)
=H(\boldsymbol q)
+(\boldsymbol e_i-\boldsymbol q)\cdot\nabla H(\boldsymbol q),
$$

  其中 $\boldsymbol e_i$ 是 Token $i$ 的 One-Hot 向量。这给出了可采样、能恢复真实分布的 LM Loss 的一般形式；$H$ 严格凹时对应唯一最优解。

- **经典实例。** Shannon 熵对应对数评分 $S(\boldsymbol q,i)=-\log q_i$，即交叉熵；二次熵对应 Brier Score $\lVert\boldsymbol q-\boldsymbol e_i\rVert^2$；Tsallis、球面和 Rényi 熵还给出带参数 $\alpha$ 的一族推广，后三者在 $\alpha\to1$ 时回到交叉熵。若进一步要求单样本 Loss 只依赖正确 Token 的概率 $q_i$，那么在通常正则条件下，对数评分是唯一选择（忽略正常数缩放和常数平移）。

- **不能脱离激活函数评价 Loss。** 令 $\boldsymbol q=\operatorname{softmax}(\boldsymbol z)$，交叉熵和 Brier Score 对 Logits $\boldsymbol z$ 的梯度分别为

$$
\nabla_{\boldsymbol z}S_{\mathrm{CE}}(\boldsymbol q,i)
=\boldsymbol q-\boldsymbol e_i,
$$

$$
\nabla_{\boldsymbol z}S_{\mathrm{Brier}}(\boldsymbol q,i)
=2\bigl(\operatorname{diag}(\boldsymbol q)-\boldsymbol q\boldsymbol q^\top\bigr)
(\boldsymbol q-\boldsymbol e_i).
$$

  交叉熵只在预测达到目标时梯度为零，且关于 Logits 是凸的；Brier 多乘了 Softmax Jacobian，模型在“自信但错误”时也可能梯度消失，前中期训练效率更低。不过，这种饱和性在训练后期也可能自动弱化极难或错误样本，因此潜在抗噪性更好。

- **从 Loss 反推出配套激活函数。** 若要求任意评分规则仍获得 $\nabla_{\boldsymbol z}S=\boldsymbol q-\boldsymbol e_i$ 的干净梯度，则其势函数与激活应为

$$
\Phi(\boldsymbol z)
=\max_{\boldsymbol p\in\Delta^{n-1}}
\left[\boldsymbol p\cdot\boldsymbol z+H(\boldsymbol p)\right],
\qquad
\boldsymbol q
=\underset{\boldsymbol p\in\Delta^{n-1}}{\arg\max}
\left[\boldsymbol p\cdot\boldsymbol z+H(\boldsymbol p)\right].
$$

  这正是 Fenchel–Young Loss 框架。Shannon 熵导出 Softmax；Tsallis 熵导出 Entmax-$\alpha$，其中 $\alpha=2$ 是 Sparsemax。$\alpha>1$ 产生稀疏分布，$\alpha<1$ 产生稠密分布，$\alpha\to1$ 回到 Softmax。

## 关键发现

- 交叉熵不是唯一能学习完整 Token 分布的 Loss；Brier、Tsallis、球面和 Rényi 评分都属于 Proper Scoring Rules，也能在总体风险层面以 $\boldsymbol q=\boldsymbol p$ 为最优解。
- 交叉熵的特殊性来自多重结构同时成立：可逐样本估计、局部 Loss 只依赖 $q_i$、与 Softmax 配合后梯度为 $\boldsymbol q-\boldsymbol e_i$，并且关于 Logits 保持凸性。
- “Softmax + Brier”并不是公平替换：Brier 在 Softmax 饱和区会压低错误样本梯度。更合理的比较应当把广义熵、评分规则和激活函数作为整体，例如比较“交叉熵 + Softmax”与“Tsallis Loss + Entmax”。
- 作者的实践倾向是：交叉熵学习效率更高，应作为主 Loss；平方损失一类的饱和梯度可能在训练后期提高抗噪性，值得做阶段性切换或混合实验。
- 文章提供的是理论构造与优化性质分析，没有实验证明某个替代 Loss 能提升 LLM 下游能力，因此不同方案的最终价值仍需同算力、同数据的训练实验确认。

## 我的提问

### Q1：训练标签明明是 One-Hot，为什么说 LM 学的是完整分布？

One-Hot 只是从上下文真实分布 $\boldsymbol p$ 中抽到的一次样本，不是“世界上只有这一个正确后继”的声明。只要单样本 Loss 的期望在 $\boldsymbol q=\boldsymbol p$ 时最小，大量不同语料实例的随机梯度就会共同恢复完整条件分布；交叉熵正满足这一点。

### Q2：为什么一个能比较分布的距离不一定能作为 LM Loss？

LM 训练时不知道完整 $\boldsymbol p$，所以每步 Loss 必须只靠采到的 $i$ 和模型输出 $\boldsymbol q$ 计算。以 Total Variation 为例，其采样表达含有未知比值 $q_i/p_i$，无法从单次 Token 观测获得；“分布间可计算”与“仅从目标样本可估计”是不同要求。

### Q3：能否把交叉熵直接换成概率平方误差？

统计一致性上可以，优化上却未必。Brier Score 与 Softmax 组合后会多出一个 Jacobian，使自信错误区域的梯度接近零，而且对 Logits 非凸。若要认真比较，应该使用与广义熵匹配的激活，或只在模型已稳定的后期混入 Brier，而不是从头机械替换。

### Q4：这篇文章真正建议的实验方向是什么？

不是寻找一个孤立的“更好 Loss”，而是联合实验评分规则与激活函数，并区分训练阶段。可从交叉熵基线出发，比较后期 CE/Brier 混合、Tsallis Loss + Entmax，以及不同 $\alpha$ 对抗噪、校准、长尾概率和下游任务的影响；不同 Loss 的绝对数值不可直接横向比较。

## 局限与疑问

- 全文以理论推导为主，没有 LLM 预训练或后训练的对照实验，因而“后期使用平方损失可能更鲁棒”仍是有依据的假设，而非已验证结论。
- Properness 只保证无限数据、函数空间中的总体风险最优点正确，不保证有限数据下泛化更好，也不保证深度网络参数空间更容易优化或最终能力更强。
- “只依赖 $q_i$ 时交叉熵唯一”的论证受概率单纯形约束影响，文章也明确提醒其朴素微分写法略欠严谨；结论应理解为通常正则性与局部性条件下的唯一性。
- Entmax/Sparsemax 的精确零概率可能改善稀疏性和计算，却也可能过早删除合理的长尾 Token；这对开放式生成、校准和多样性的净影响需要实证。
- 文章没有讨论词表规模下 Entmax 阈值求解的实际吞吐、分布式训练稳定性，以及替代 Loss 后 Scaling Law 和跨实验 Loss 可比性如何重建。

## 我的判断

这篇文章最有价值之处不是罗列几个替代 Loss，而是把“训练样本为何足以学习分布”“什么 Loss 具有统计一致性”“为什么 Softmax 与交叉熵总是配套”串成了同一条几何与凸分析链路。它很好地解释了交叉熵的主导地位并非只靠信息论传统，而是统计可估计性、局部性和优化友好性共同筛选的结果。

它更适合作为设计实验的理论地图，而不是某个新方法的效果证据。若要复现，优先做低风险的训练后期 Loss mixing，并同时跟踪梯度范数、噪声样本鲁棒性、概率校准、长尾召回和标准下游任务；只有在这些信号成立后，再考虑从头训练 Entmax/Tsallis 配对。

## 下次只看这些

1. LM Loss 必须写成 $\mathbb E_{i\sim\boldsymbol p}[S(\boldsymbol q,i)]$ 且在 $\boldsymbol q=\boldsymbol p$ 时最优；任意分布距离并不自动满足可采样性。
2. 任取凹广义熵 $H$，都有 $S(\boldsymbol q,i)=H(\boldsymbol q)+(\boldsymbol e_i-\boldsymbol q)\cdot\nabla H(\boldsymbol q)$；若再要求 Loss 只依赖 $q_i$，交叉熵基本唯一。
3. Loss 与激活必须成对看：Shannon/交叉熵对应 Softmax，Tsallis 对应 Entmax；Softmax + CE 的核心优势是干净梯度 $\boldsymbol q-\boldsymbol e_i$。
