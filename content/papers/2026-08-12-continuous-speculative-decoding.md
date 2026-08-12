---
title: "Continuous Speculative Decoding for Autoregressive Image Generation"
paper_url: "https://arxiv.org/abs/2411.11925"
authors: "Zili Wang et al."
venue: "ECCV 2026"
published: "2026"
read_date: "2026-08-12"
read_at: "2026-08-12T14:47:08+08:00"
status: "已精读"
tags: ["Speculative Decoding", "Autoregressive Generation", "Diffusion Models", "Inference Acceleration"]
one_liner: "CSpD 让小型 continuous visual AR model 起草多个 diffusion-sampled latent tokens，再用共享噪声对齐后的近似路径 likelihood ratio 由大模型并行验证，以有偏但可计算的连续 acceptance rule 换取最高约 2.7× 加速。"
---

## 研究问题

MAR、xAR、Harmon 等 continuous visual autoregressive models 不使用离散 VQ codebook，而是按位置自回归地预测连续视觉 latent；每个 token 又由一个条件 diffusion head 经过多步去噪采样。它们避免了离散量化带来的重建损失，但仍要顺序生成大量 token，推理开销很高。

LLM speculative decoding 可以让小 draft model 连续提出多个 token，再由大 target model 一次并行验证；对离散 categorical distribution，候选概率和拒绝后的残差分布都可直接枚举。但 continuous visual AR 面临两个障碍：连续 token 的 marginal density 需要积分掉整条 diffusion trajectory，难以计算；拒绝后需要采样的 $[p-q]_+$ 分布也没有显式归一化常数。论文试图在不训练新模型、不修改架构的前提下，把 speculative decoding 推广到这种“自回归外循环 + diffusion token sampler 内循环”。

## 核心方法

- **Draft 多步、target 并行验证**：记 target distribution 为 $p$、draft distribution 为 $q$。Draft model 自回归地产生 $γ$ 个连续候选 token；target 在这些候选构成的 teacher-forced prefix 上，一次计算所有候选位置的条件分布。分布可以并行算，但 acceptance 必须从左到右进行；遇到第一个 rejection 后，后续 draft 全部丢弃。
- **理想 acceptance rule**：对 draft token $x_0\sim q$，理想规则仍与离散 speculative sampling 相同：

$$
a(x_0)=\min\left(1,\frac{p(x_0)}{q(x_0)}\right).
$$

  难点在于 $p(x_0)$ 与 $q(x_0)$ 是 diffusion trajectory 的边缘密度，直接计算需要对所有中间状态积分。
- **用两条对齐路径近似 marginal ratio**：令 $Y_p=x^p_{0:T}$ 和 $Y_q=x^q_{0:T}$ 分别为 target、draft 的去噪路径，并令二者共享最终 token $x_0^p=x_0^q=x_0$。论文不用严格但几乎为零的 shared-path ratio $p(Y)/q(Y)$，而采用 surrogate $p(Y_p)/q(Y_q)$。
- **Denoising trajectory alignment**：两模型在每个 reparameterized Gaussian transition 中使用同一份随机噪声：

$$
x^{p/q}_{t-1}
=\mu^{p/q}_t+\sqrt{\Sigma^{p/q}_t}\,\epsilon_t,
\qquad \epsilon_t^p=\epsilon_t^q.
$$

  共享噪声降低两条路径的期望距离；同时中间 Gaussian transition 的 quadratic terms 相消，使路径比值简化为：

$$
\widetilde r(x_0)
=\frac{p(Y_p)}{q(Y_q)}
=\left(
\prod_{t=2}^{T}
\frac{\sqrt{|\Sigma_t^q|}}{\sqrt{|\Sigma_t^p|}}
\right)
\frac{p_\theta(x_0\mid x_1^p)}{q_\theta(x_0\mid x_1^q)}.
$$

  运行时从 $u\sim U(0,1)$ 采样，若 $u\leq\min(1,\widetilde r)$ 就接受。实现时应在 log space 比较 $\log u$ 与 $\min(0,\log\widetilde r)$，避免高维密度下溢。
- **Target token pre-filling**：不同大小模型拥有各自的 prefix embedding，最初几个 AR steps 的分布差异尤其大。论文先让 target 生成少量 token（例如 5%），建立共同的视觉 prefix，再启动 speculative decoding。
- **拒绝后的 residual resampling**：第一个 draft rejection 后，理论上应从

$$
p'(Y)=\frac{\max(0,p(Y)-q(Y))}{Z},
\qquad
Z=\int \max(0,p(Y)-q(Y))\,dY
$$

  采样。论文以 $p$ 作为 proposal，将 rejection-sampling 接受率写成

$$
\alpha_s(Y)=\frac{\max(0,p(Y)-q(Y))}{p(Y)},
$$

  从而消掉未知的 $Z$。再利用 trajectory alignment 和已缓存的 target trajectory，把 proposal sampling 限制为最后一步 Gaussian，避免重新运行完整 diffusion head。

## 关键发现

- 直接在同一条 draft trajectory 上计算 $p(Y)/q(Y)$ 时，论文测得平均 ratio 只有 $5.33\times10^{-23}$，acceptance 为 0%；改成两条独立路径但不对齐时，ratio 为 0.067、acceptance 为 14%；共享噪声对齐后，ratio 提升到 1.86、acceptance 为 32%。这组 ablation 最直接地证明 trajectory alignment 是方法能工作的必要条件。
- 在 ImageNet $256\times256$ 上，MAR-H target + MAR-B draft、$\gamma=32$ 的 acceptance 为 0.19；batch size 从 1 增至 256 时，wall-clock speedup 从 **1.44×** 增至 **2.33×**。
- xAR-H + xAR-B、$\gamma=32$ 的 acceptance 为 0.22，batch size 1/256 分别达到 **1.77×/2.72×**。Harmon-H + Harmon-B 在 $512\times512$、$\gamma=32$ 时 acceptance 只有 0.15，但 batch size 1/32 仍达到 **1.63×/2.54×**。
- MAR/xAR 的 FID、IS 以及 Harmon 的 FID、CLIPScore、Geneval 与 target-only 基本接近。证据支持的是“该近似 criterion 在这些模型上没有造成明显指标下降”，而不是严格证明输出等同于 target marginal distribution。
- 更长 draft window 会降低 acceptance，但在大 batch 上仍可能更快；论文的最大加速均出现在较长 $\gamma$ 和较大 batch，说明收益高度依赖硬件利用率，而不只是 acceptance rate。

## 我的提问

### Q1：Target 到底看什么数值来决定是否接受？

Target 不是输出一个图像质量分数，而是输出对应候选位置 diffusion distribution 的 Gaussian 参数。对于同一个 draft sample $x$，系统分别计算它在 target 和 draft 最后一步 Gaussian 下的 density，并乘上中间 trajectory variance ratio。若记

$$
\log r
=\log\Sigma_{mathrm{traj}}
+\log\mathcal N(x;\mu_p,\Sigma_p)
-\log\mathcal N(x;\mu_q,\Sigma_q),
$$

则接受条件是 $\log u\leq\min(0,\log r)$。例如 $p/q=0.8$ 时以 80% 概率接受，而不是设定固定 L1 threshold；$p/q\geq1$ 时必然接受。

官方代码保存 `draft_mean/var/log_var`、`target_mean/var/log_var` 和 `draft_sample`，再计算 softened Gaussian density ratio。公开实现默认 `inner_temperature=42`，会显著压平 Mahalanobis distance、提高 acceptance；因此复现时必须区分论文公式中的 surrogate ratio 与代码中进一步 relaxed 的 criterion。

### Q2：能否把同样思路直接用于普通 video diffusion？

需要区分模型的生成轴：

- 对 continuous autoregressive video model，可以把一个 frame/block latent 当作连续 AR token，形式上最接近 CSpD；但视频 block 维度远高于图像 patch，细小 per-dimension 误差会累积成极低 likelihood ratio。
- 对 full-clip stochastic video diffusion，可以沿 denoising timestep 起草多个完整视频 states，再由 target 验证 transition density。若 draft/target transition 是同 covariance Gaussian，reflection maximal coupling 比 CSpD 的普通 residual rejection sampling更合适，拒绝后可一次反射得到严格 target sample。
- 对 Wan 等常用的 deterministic flow-matching/ODE sampler，$\sigma_t=0$，transition 没有非退化 Gaussian density，不能直接使用 $p/q$ rule。必须加入 stochastic sampler，或者接受 ASDSV 一类有偏的 error-threshold verification。

此外，video DiT 通常更 compute-bound。若 target 验证 $K$ 个完整 video latents 的成本接近 $K$ 次普通 forward，即使数学上可验证，也未必有 wall-clock 收益。

### Q3：ASDSV 所谓只验证“首尾”，是不是 target 只输入视频首尾两帧？

不是。这里有两条时间轴：video time 是 latent 内部的帧维度，diffusion time 是外部去噪步。ASDSV 的“首尾”是 speculative denoising window 的首尾两个 timestep；每个输入仍是完整的时空 latent，shape 都是 $[B,C,F,H,W]$。Target 只省略中间 $K-2$ 个 denoising forwards，没有省略任何视频帧。

更准确地说，ASDSV 在 draft trajectory 的首尾位置检查 target 与 draft 的局部 denoising/vector-field output 是否接近，并用首尾 L1 error 小于阈值来推断中间也安全。它没有从共同起点独立运行 $K$ 次 target 得到真正的 target-only endpoint，因此这是经验近似，不能保证中间 trajectory 未曾偏离。

### Q4：如果 ASDSV 的首尾差异较大，从哪里重新开始？

它回到最近一个由 target 真正生成的可信 latent，而不是回到初始噪声。设窗口从 $T_N$ 开始，draft rollout 为 $D_{N+1:N+K}$；target 已经从 $T_N$ 正常生成 $T_{N+1}$。验证失败时，整段 draft window 被丢弃，保留 $T_{N+1}$，target 再生成 $T_{N+2}$，随后从 $T_{N+2}$ 开始新一轮 speculation。

这与 CSpD 不同：CSpD 对候选 token 从左到右验证，若第 $j$ 个 token 拒绝，可以保留前 $j-1$ 个并在第 $j$ 位 residual-resample。ASDSV 只看窗口端点，不知道真正的内部 rejection point，只能整窗 rollback。一个自然改进是失败后增加中点检查或二分定位，以额外 target forward 换取保留较长的可信 draft prefix。

## 局限与疑问

- 核心 ratio $p(Y_p)/q(Y_q)$ 是对理想 $p(x_0)/q(x_0)$ 的有偏 surrogate。论文只能说明 alignment 缩短路径距离并约束一阶近似 bias，不能恢复经典 speculative sampling 的严格 target-distribution guarantee。
- 官方实现还用默认 temperature 42 softens likelihood。论文对“保持输出分布”的措辞应理解为经验指标基本不变，而不是数学上的 exact sampling。
- 高维 density ratio 容易数值下溢并出现 acceptance collapse；从 image patch 扩展到完整视频 latent block 时，这一问题会显著放大。
- 后续 diffusion speculative sampling 工作指出，直接以 target 为 proposal 的 residual rejection sampler 平均仍需一次 target sampling，且 draft 越接近 target 时 trial count 方差反而越大。CSpD 依靠缓存并只重采最后一步 Gaussian 来降低开销，但这一优化依赖其特殊的“AR token 内嵌 diffusion head”结构，不能直接外推到普通 full-clip diffusion。
- 当前 target/draft 规模差距仍不大。论文报告 MAR-B/MAR-H 在 batch 128 下的推理成本比 $c=0.38$，远高于 LLM speculative decoding 中常见的约 0.05；batch 1 的加速也明显低于 headline numbers。
- 质量实验覆盖的仍是少量同家族 continuous visual AR checkpoints。不同训练数据、不同 variance parameterization、强 CFG 或确定性 sampler 下的 acceptance 与 bias 仍需单独验证。

## 我的判断

CSpD 最值得记住的不是“连续值也可以套一个 $p/q$”，而是它明确暴露了 continuous speculative decoding 的三个真正难点：marginal likelihood 不可得、独立 diffusion trajectories 几乎不重合、拒绝后的 residual distribution 难采样。共享 reparameterization noise 同时改善路径耦合和简化 density ratio，是方法中最扎实、也最可迁移的机制。

但这不是严格 lossless 的 continuous speculative decoding。它先以两条不同路径的 joint-density ratio 替代 marginal ratio，代码又进一步做 temperature relaxation；因此更适合作为高质量 approximate acceleration baseline。若复现，应同时报告 raw surrogate ratio、relaxed ratio、每步 acceptance、FID/IS 与真实 wall time，不能只报告最终 speedup。

对 video diffusion，直接复制 CSpD 的优先级不高。Block-autoregressive video 可以借鉴 target prefill、shared-noise alignment 和 first-rejection rollback；full-clip video diffusion 则应优先比较 reflection coupling、ASDSV endpoint verification 和大小模型 timestep stitching。更有研究空间的是 motion-aware verifier、按 uncertainty 动态选择 window，以及 verification 失败后的层次化定位与局部 correction。

## 下次只看这些

1. CSpD 的 acceptance 看的是共享噪声对齐后的近似路径 likelihood ratio，不是 L1/CLIP；这个 surrogate 有偏，官方代码还用了 temperature relaxation。
2. 共享 $\epsilon_t$ 把 acceptance 从近乎 0% 提到约 32%，并让中间 Gaussian terms 大量消掉，是整篇论文的关键机制。
3. 向 video 迁移时先区分 AR block、stochastic diffusion 和 deterministic flow matching；ASDSV 的“首尾”是完整视频 latent 的两个去噪 timestep，不是两帧视频。
