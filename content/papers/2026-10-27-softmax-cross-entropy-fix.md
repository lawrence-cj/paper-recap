---
title: "A Fix for Softmax and Cross-Entropy"
paper_url: "https://arxiv.org/abs/2610.31642"
authors: "Jonathan T. Barron, Dan Gnanapragasam, Michael Tschannen"
venue: "arXiv"
published: "2026"
read_date: "2026-10-27"
read_at: "2026-10-27T15:25:47+08:00"
status: "已读"
tags: ["Training & Optimization", "Knowledge Distillation"]
search_terms: ["softmax", "log-softmax", "cross-entropy", "Softmax Collapse", "Softmax Cancellation", "数值稳定性", "浮点舍入", "grokking", "Slingshot", "log1p", "expm1"]
one_liner: "直接计算微小梯度：高置信度时避免用接近 1 的概率减 1，蒸馏接近老师时从 logits 差计算概率差，修复梯度消失与虚假更新。"
---

## 研究问题

标准 softmax 会先减去最大 logit，避免指数溢出，但这不能防止小量在加法和减法中丢失。本文研究两个梯度错误：高置信度分类中，正确类别的梯度提前变成 0；软目标蒸馏中，学生接近老师时梯度失真，完全相同时也可能出现非零梯度。

依据为 [论文 v1，2026-10-22](https://arxiv.org/html/2610.31642v1) 与 [作者项目页及 JAX 实现](https://jonbarron.info/softmaxfix/)。本次阅读重点是两个问题的计算过程；对话中做了标量 FP32 舍入和 NumPy 运算路径演示，没有运行论文的训练实验，也没有验证具体 Attention kernel。

## 核心方法

![左图为高置信度时标准实现的目标梯度突然归零，右图为学生接近老师时标准实现留下非零梯度噪声；修复实现保留微小梯度](media/softmax-cross-entropy-fix/collapse-cancellation.webp "论文 Figure 1：Softmax Collapse 与 Softmax Cancellation。来源：Barron 等，arXiv:2610.31642v1，CC BY 4.0；将原始图例与两个子图重新排版，曲线、坐标和标签保持原样。")

### 1. Softmax Collapse：直接保留非目标概率质量

设 logits 为 $z$，正确类别为 $t$，模型概率为 $q=\operatorname{softmax}(z)$。硬标签交叉熵的梯度是 $g_t=q_t-1$、$g_j=q_j$（$j\ne t$）。定义正确类别相对于其余类别总质量的 margin：

$$
m=z_t-\log\sum_{j\ne t}e^{z_j},\qquad r=e^{-m}.
$$

常见计算与修复计算数学等价，但数值表现不同：

$$
g_t=\underbrace{\frac{1}{1+r}-1}_{\text{standard}}
=\underbrace{-\frac{r}{1+r}}_{\text{fixed}}.
$$

当 $1+r$ 舍入成 1 时，前者变成 0；后者的分子仍保留小量 $r$，直到更极端的下溢范围才失去它。作者把正确类别排除在 log-sum-exp 之外，再用稳定的 softplus 计算：

$$
a=\operatorname{LSE}_{j\ne t}(z_j-z_t),\qquad
L=\operatorname{softplus}(a).
$$

这里 $\operatorname{LSE}(x)=\log\sum_j e^{x_j}$，$\operatorname{softplus}(a)=\log(1+e^a)$。实现必须稳定地处理这个表达式，不能先用普通加法形成 $1+e^a$。自动求导沿着 $z_j-z_t$ 得到：

$$
g_j=e^{z_j-z_t-L}\quad(j\ne t),\qquad
g_t=-\sum_{j\ne t}g_j.
$$

独立的 log-softmax/softmax 则排除最大 logit，按相同思路处理；硬标签交叉熵可直接使用已知目标 $t$，无需额外 argmax。没有改变精确数学中的目标函数。

### 2. Softmax Cancellation：从 logits 差直接计算概率差

老师概率为 $p=\operatorname{softmax}(z_T)$，学生概率为 $q=\operatorname{softmax}(z_S)$，老师视作固定目标。梯度应为 $q-p$。直接相减两个近似相等的概率容易丢失有效数字；不同前向、反向路径还可能在 $z_S=z_T$ 时留下残差。

作者令 $\delta=z_S-z_T$，先在老师分布下去除共同偏移，再计算对数概率比：

$$
d_k=\delta_k-\sum_i p_i\delta_i,\qquad
s_k=d_k-\operatorname{log1p}\left(\sum_i p_i\operatorname{expm1}(d_i)\right).
$$

在精确算术中 $s_k=\log(q_k/p_k)$，所以概率差可以直接写成：

$$
u_k=p_k\operatorname{expm1}(s_k),\qquad
g_k=u_k-q_k\sum_i u_i.
$$

`expm1(x)` 和 `log1p(x)` 分别稳定计算 $e^x-1$、$\log(1+x)$；最后一项修正残余的梯度和误差。作者实际代码在 $|s_k|\ge1$ 时回退到 $q_k-p_k$，并用自定义反向将 $g$ 作为学生 logits 的梯度，前向仍返回交叉熵。仅替换前向的一个 `log` 不能完成这项修复。

## 关键发现

- **论文的数值诊断：** 随高置信度 margin 增大，标准目标梯度出现突然归零；随学生趋近老师，标准蒸馏梯度出现非零误差底部。Figure 1 直接比较了这些行为与修复后的结果。
- **论文的训练证据：** 在受控 grokking 实验中，修复缓解 Slingshot 损失尖峰和相关泛化失败；在学生由老师权重初始化的蒸馏实验中，修复避免舍入误差推动学生漂移。它们不是大规模语言或视频模型的收益证明。
- **本次计算演示：** 对 $[9,-9,-9]$，逐步 FP32 舍入可得到标准分母恰好为 1；稳定路径保留约 $3.046\times10^{-8}$ 的损失与相应梯度。对 $z_S=z_T=[2,1,0]$，两种 NumPy FP32 运算路径仍可能产生约 $10^{-8}$ 的概率差。后者说明舍入路径问题，不代表所有框架都输出同一数值。

## 我的提问

### Q1：减去最大值已经是数值稳定实现，为什么还会有 bug？

它解决的是指数溢出。这里还有两类精度损失：一个可表示的小数加到 1 上后被舍掉，以及两个非常接近的近似数相减后，误差与真实差异一样大甚至更大。

例如 FP32 可以保存约 $2.061\times10^{-9}$，却会把 $1+2.061\times10^{-9}$ 舍入成 1。提高精度能推迟问题出现，但不能消除这种计算方式的局限。

### Q2：给一个 Softmax Collapse 的完整计算例子，作者实现怎么做？

取 $z=[9,-9,-9]$，$t=0$。减去目标 logit 后是 $[0,-18,-18]$，其中 $e^{-18}\approx1.523\times10^{-8}$。常见分母为：

$$
D=1+2e^{-18}\xrightarrow{\mathrm{FP32}}1.
$$

于是目标概率变成 1、损失变成 0、目标梯度变成 0，而其他类别仍有正梯度：

$$
g_{\mathrm{standard}}\approx[0,\ 1.523\times10^{-8},\ 1.523\times10^{-8}].
$$

作者排除目标槽位，计算：

$$
a=\operatorname{LSE}(-\infty,-18,-18)
=-18+\log2\approx-17.30685,
$$

$$
L=\operatorname{softplus}(a)\approx3.046\times10^{-8}.
$$

对应的 JAX 计算步骤如下，变量名略作简化，`jax`、`jnp` 分别指 JAX 与 `jax.numpy`。完整函数见 [作者第 2 节](https://jonbarron.info/softmaxfix/#sec:bug1)。

1. 取出正确类别分数：`z_t = jnp.take_along_axis(z, t[..., None], axis=-1)`。
2. 标记正确类别位置：`is_t = jax.nn.one_hot(t, z.shape[-1], dtype=bool)`。
3. 排除正确类别：`others = jnp.where(is_t, -jnp.inf, z - z_t)`。
4. 只对其余类别求 LSE：`a = jax.nn.logsumexp(others, axis=-1)`。
5. 返回稳定损失：`jax.nn.softplus(a)`。

自动求导得到约 $[-3.046\times10^{-8},1.523\times10^{-8},1.523\times10^{-8}]$。关键在于目标梯度来自其他类别梯度之和的相反数，不再依靠接近 1 的概率减 1。softmax 对全部 logits 的共同平移不变，因此梯度和在数学上本应为 0；标准结果破坏了这个性质。

### Q3：给一个 Softmax Cancellation 的例子，为什么相同输入也会产生梯度？

取 $z_T=z_S=[2,1,0]$。梯度理论上是 $q-p=[0,0,0]$。但老师前向可能计算 `exp(shifted) / sum(exp(shifted))`，学生损失的反向则可能通过 `exp(shifted - log(sum(exp(shifted))))` 恢复概率。数学等价不保证浮点结果逐位相同。

本次 NumPy 2.0.2 FP32 运算路径演示得到：

| 类别 | 老师的 softmax 概率 | 学生通过 exp(log-softmax) 恢复的概率 | 概率差 |
| --- | --- | --- | --- |
| 0 | 0.6652409434 | 0.6652409434 | 0 |
| 1 | 0.2447284758 | 0.2447284907 | $1.4901\times10^{-8}$ |
| 2 | 0.0900305733 | 0.0900305808 | $7.4506\times10^{-9}$ |

这个例子中老师概率的 FP32 求和恰为 1，因此对应的 `exp(log_q) * sum(p) - p` 反向表达式也得到表中的差。它是运算路径演示，没有调用某个框架的融合交叉熵反向 kernel。

若两边使用完全相同的确定性 softmax 再直接相减，相同输入会得到 0。论文所说的相等时仍有梯度，需要结合实际前向与反向的运算方式理解，不能解释为相同函数必然给相同输入算出不同结果。

作者的方法先算 $\delta=z_S-z_T=[0,0,0]$，随后 $d=0$、$\operatorname{expm1}(d)=0$、$s=0$、$u=0$，最终 $g=0$。不再先制造两组略有不同的概率近似值，然后从它们的差推断学生是否偏离老师。

### Q4：学生只是接近老师时，expm1 为什么有帮助？实际反向怎么接入？

若 $x=10^{-8}$，FP32 的普通 `exp(x)` 可能先舍入成 1，再减 1 就得到 0；`expm1(x)` 直接保留约 $10^{-8}$ 的结果。`log1p` 同样避免先形成一个舍入成 1 的中间值。这里需要保留的正是接近老师时的微小概率差。

作者先用第 2 节修复后的 log-softmax/softmax 得到 $\log q$、$q$、$p$。按核心方法中的公式计算 $d,s$，再用分支 `jnp.where(jnp.abs(s) < 1, p * jnp.expm1(s), q - p)` 得到 $u$，修正为 $g$。完整函数见 [作者第 3 节](https://jonbarron.info/softmaxfix/#sec:bug2)。

自定义反向的关键是停止梯度运算 $\operatorname{sg}$：前向保持输入值，反向导数为 0。作者返回的表达式等价于：

$$
L_{\mathrm{forward}}=-\sum_k p_k\operatorname{sg}(\log q_k),
$$

$$
L_{\mathrm{out}}=L_{\mathrm{forward}}+
\sum_k\operatorname{sg}(g_k)
\left(z_{S,k}-\operatorname{sg}(z_{S,k})\right).
$$

最后一项前向数值为 0，但对学生 logits 的导数为 $g$，因此保留损失值并指定稳定梯度。这里按蒸馏设定将老师视作固定目标，只讨论对学生 logits 的导数。交叉熵在学生等于老师时等于老师分布的熵，通常不为 0；应当为 0 的是学生梯度。

### Q5：这是不是说 Attention 都错了，换上去就能涨点？

不能这样推断。Attention 使用 softmax，值得检查它的数值行为；实际是否触发、误差大小和后果还取决于上游梯度、精度、融合 kernel 及反向实现。本文的主要训练证据是受控 grokking 和蒸馏，没有证明所有 Attention、语言模型或视频模型都会获得明显收益。

## 局限与疑问

- 作者明确说明，实验关注的极端高置信度和极小学生—老师差异，在常规大规模训练中并不常见。不能把问题的存在等同于实际训练一定受损。
- 本次只核对公式、作者代码与具体算例，没有运行 JAX 完整实现、训练复现或 Attention kernel 对照；数值与训练结论应区分。
- 把梯度修正为和接近 0，不代表任意浮点归约都保证逐位精确的零和。最终还要检查目标实现的误差、下溢行为和吞吐。
- 对自有任务的下一步应是记录实际 margin 与学生—老师差异范围，再比较损失及梯度；性能开销和模型收益均待验证。

## 我的判断

本次阅读判断（助手整理）：最有价值的是保留原始数学目标、重新安排浮点运算的思路，以及两个容易检查的条件——高置信度下目标梯度是否提前消失，学生等于老师时梯度是否为 0。适合作为训练数值诊断的参考；是否值得替换现有训练实现，需要真实任务上的检查。用户尚未给出独立的性能或复现评价。

## 下次只看这些

1. **Collapse：** $1/(1+r)-1$ 容易变成 0；直接计算 $-r/(1+r)$，并通过排除目标后的 LSE 与稳定 softplus 得到正确反向。
2. **Cancellation：** 不从两组接近的概率相减寻找微小差异；从 $z_S-z_T$ 出发，用 `expm1`、`log1p` 计算，再接入自定义反向。
3. **回看 Q2、Q3：** $[9,-9,-9]$ 演示丢掉目标梯度，$z_S=z_T=[2,1,0]$ 演示不同运算路径的虚假梯度；这不等于大模型收益已经得到验证。
