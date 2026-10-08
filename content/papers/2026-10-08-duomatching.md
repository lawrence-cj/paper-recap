---
title: "DuoMatching: Joint-Marginal Distribution Matching for Few-Step Video Generation"
paper_url: "https://arxiv.org/abs/2610.03543"
authors: "Jiahao Zhan et al."
venue: "arXiv"
published: "2026"
read_date: "2026-10-08"
read_at: "2026-10-08T16:51:03+08:00"
status: "已读"
tags: ["Video Generation", "Diffusion Models", "Few-Step Sampling", "Knowledge Distillation"]
one_liner: "保留整段视频 DMD，再用图像教师给抽样帧加 marginal DMD；LatentBridge 把时间压缩的视频 latent 转成指定单帧的图像 latent，让画质监督尽量不伤运动。"
---

## 研究问题

视频 DMD（Distribution Matching Distillation，分布匹配蒸馏）把多步视频教师蒸馏为少步学生。对整段视频做 joint DMD 可以缓解自回归 rollout 的质量漂移，但学生仍可能缺少纹理、构图不准确，或遗漏提示词指定的物体和属性。论文 Figure 1 的例子包括毛发细节不足，以及没有生成提示词中的羽毛。

作者认为需要额外的单帧监督：视频教师本身只是对真实视频分布的近似；整段视频的更新同时受帧间关系牵制，因此补好某一帧的细节不一定得到充分的更新信号。图像模型可以提供另一份更直接的单帧参考分布。

需要保留的区别：joint distribution matching 在理想条件下也约束 marginal，并非完全不管单帧。本文增加的是独立的单帧分布目标和图像教师提供的互补能力。

阅读依据：[DuoMatching arXiv v1](https://arxiv.org/html/2610.03543v1)，重点为 §3、Table 1–6 与 Appendix D、G。论文发表于 2026-10-02；[官方代码](https://github.com/JohnZhan2023/DuoMatching)的 README 也确认使用 Causal Forcing++ 的 causal CD 初始化、Wan 视频教师和 Qwen-Image 图像教师。

## 核心方法

### 1. Joint 与 marginal 分别表示什么？

Joint 是整段视频所有帧的联合分布。给定提示词 $c$，它描述哪些帧会一起出现，既包含单帧外观，也包含帧之间的关系：

$$
p(x_1,x_2,\ldots,x_T\mid c).
$$

$x_i$ 是第 $i$ 帧，$T$ 为帧数。Marginal 是单帧的边缘分布：把其他帧积分掉，只保留这一帧可能是什么样子：

$$
p(x_i\mid c)
=\int p(x_1,\ldots,x_T\mid c)\,\mathrm d x_{\setminus i}.
$$

$x_{\setminus i}$ 表示除第 $i$ 帧以外的所有帧。这里仍保留提示词条件，但不再给定其他帧。论文实际采用按抽帧策略取一帧后的分布，记为 $MQ_\theta$，不局限于固定的帧位置。

两帧视频的例子能说明区别：

| 模型 | 一半视频 | 另一半视频 |
| --- | --- | --- |
| A | 红杯 → 红杯 | 蓝杯 → 蓝杯 |
| B | 红杯 → 蓝杯 | 蓝杯 → 红杯 |

两者每个帧位置都是 50% 红杯、50% 蓝杯，marginal 相同；A 保持颜色，B 改变颜色，joint 不同。因此单帧分布相同不保证时间关系相同。

### 2. 整段视频 DMD 与单帧图像 DMD 同时训练

记 $Q_\theta$ 为学生的视频分布，$P_v$ 为视频教师分布，$P_i$ 为图像教师分布；$M$ 表示按给定策略从视频中取出一帧，$MQ_\theta$ 因而是抽样帧的分布。理想目标写成：

$$
\mathcal J(\theta)
=D_{\mathrm{KL}}(Q_\theta\Vert P_v)
+\omega D_{\mathrm{KL}}(MQ_\theta\Vert P_i),
\qquad \omega>0.
$$

实际训练用对应的 DMD surrogate：

$$
\mathcal L_{\mathrm{DuoMatching}}
=\mathcal L_{\mathrm{joint\text{-}DMD}}
+\omega\mathcal L_{\mathrm{marginal\text{-}DMD}}.
$$

- joint DMD 用视频教师监督完整时间序列，约束帧间依赖。
- marginal DMD 用图像教师分别监督少量抽样帧，补充细节、构图和提示词遵循。
- 每个分支都有冻结的教师和可训练的 fake score estimator。后者跟踪学生当前生成分布；图像分支并非只用图像教师单独打分，也不要求它先生成一张配对目标图。

对图像 latent $u$ 按图像教师的噪声日程得到 $u_\tau$ 后，单帧 DMD 的梯度估计为：

$$
\nabla_\theta\mathcal L_{\mathrm{marginal}}
=\mathbb E\left[
w_{\mathrm{img}}(\tau)
\left(s_{\mathrm{fake}}^{\mathrm{img}}-s_{\mathrm{real}}^{\mathrm{img}}\right)^\top
\frac{\partial u_\tau}{\partial\theta}
\right].
$$

$s_{\mathrm{real}}^{\mathrm{img}}$ 来自图像教师，$s_{\mathrm{fake}}^{\mathrm{img}}$ 来自跟踪学生单帧分布的辅助模型；$w_{\mathrm{img}}(\tau)$ 为时间权重。梯度通过 LatentBridge 传回视频学生。

### 3. Marginal Matching Signal：单帧修正为何不同？

这里的 score 指 $\nabla_{u_\tau}\log p_\tau(u_\tau\mid c)$，描述 latent 沿各方向变化时概率密度如何变化。上式中的 image fake score 与 image teacher score 的差，为学生单帧分布提供修正信号；信号经过 Bridge 传回生成器。

Joint score 则同时考虑其他视频 slice。按论文 §3.1 的分解，对教师或学生的加噪视频密度 $p_\tau$ 都有：

$$
\nabla_{z_\tau^l}\log p_\tau(\mathbf z_\tau)
=\nabla_{z_\tau^l}\log p_\tau
\left(z_\tau^l\mid\mathbf z_\tau^{\setminus l}\right).
$$

$z_\tau^l$ 是第 $l$ 个加噪视频 slice，$\mathbf z_\tau^{\setminus l}$ 表示其余 slice，提示词条件在上式中省略。给某一帧补上纹理或遗漏的羽毛，可能同时造成它与周围帧的差异；作者据此解释为什么需要更直接的单帧修正，并保留整段目标来约束跨帧关系。

[Figure 2](https://arxiv.org/html/2610.03543v1#S1.F2) 展示两种 score difference 的空间响应。在该例子中，joint 响应较分散，marginal 响应更连贯，尤其集中在人物过于光滑的黑色衣服区域。每张热图独立归一化，因此只能比较空间结构，不能根据颜色亮度判断两个分支的信号强弱，也不能把这个案例当作所有样本的保证。

### 4. LatentBridge：让图像监督作用在单帧表示上

一个时间压缩的视频 latent slice 可能对应多个 RGB 帧；图像教师的 latent 则对应单帧。即使 VAE 编码器兼容，直接把视频 slice 当图像 latent，也可能把运动信息一起压制。不同 VAE 的 latent 空间还可能不兼容。

LatentBridge 用当前 slice $z^l$、前一个 slice $z^{l-1}$ 和局部帧位置 $i$，预测该帧在图像 VAE 中的 latent：

$$
u^{l,(i)}=B_\phi(z^{l-1},z^l,i).
$$

前一个 slice 提供视频 VAE 因果编码所需的上下文；$i$ 指定共享 slice 中的哪一帧。用视频与其对应帧的两种 VAE 编码预训练 Bridge：

$$
\mathcal L_{\mathrm{Bridge}}
=\mathbb E\left[
\left\Vert B_\phi(z^{l-1},z^l,i)-E_{\mathrm{img}}(x^{l,i})\right\Vert_1
\right].
$$

$x^{l,i}$ 是对应 RGB 帧，$E_{\mathrm{img}}$ 是图像 VAE 编码器。Bridge 在视频学生训练时冻结，但不能切断它到学生的梯度。它有约 10.75M 参数，使用 8 个 FiLM residual blocks；论文在 29,400 个视频上单独预训练。

直接把视频 latent 解码为 RGB，再经图像 VAE 编码并回传梯度，是另一个可行思路；但在论文训练配置下显存溢出。Bridge 同时解决单帧表示不匹配和这条梯度路径的开销问题。

### 5. LVS：按变化分段，再分配抽帧名额

Latent Variation Sampling（LVS）计算相邻干净 latent slice 的均方差：

$$
d_l=\frac{1}{CHW}\left\Vert z^{l+1}-z^l\right\Vert_F^2.
$$

$C,H,W$ 为单个 slice 的维度。取最大的 $K-1$ 个差值位置作为边界，把序列分为 $K$ 段；每段均匀随机抽一个 slice，再随机选其中的一个帧位置。

它用大变化位置来分段，并非只监督变化最大的帧。作者希望避免有限的监督集中于相似画面。默认 $K=4$、$\omega=0.4$；Algorithm 1 让 marginal loss 的权重在训练开始时逐渐增加。图像教师、Bridge 和辅助 score 模型仅参与训练，推理仍使用视频学生。

![DuoMatching 自绘方法图：视频学生通过整段视频与经 LatentBridge 转换的抽样单帧接受两个 DMD 分支监督](media/duomatching/method-overview.webp "原创方法示意：依据 DuoMatching §3 与 Algorithm 1 整理绘制，非论文原图；来源：本笔记整理，版权归 Junsong Chen 所有，未另行授权。")

原始方法图：[论文 Figure 3](https://arxiv.org/html/2610.03543v1#S3.F3)及[项目页方法图](https://johnzhan2023.github.io/DuoMatching/assets/figures/method-1400.webp)。论文采用 arXiv non-exclusive distribution 许可，项目页未找到明确的图片转载许可；代码仓库的 Apache-2.0 声明限定为代码。因此原图只保留链接，本站采用上述原创示意。

## 关键发现

因果生成实验从 Causal Forcing++ 的 causal CD 检查点初始化，在 VidProM 上做 1,000 步 DMD 训练；双向实验从 Wan2.1-T2V-1.3B 初始化，在 Mixkit 上训练 1,200 步。默认图像教师是 Qwen-Image，输出为 81 帧、480×832，使用 8 张 80 GB GPU。自动评测采用 VBench，另准备了 400 条复杂测试提示词。

Table 1 应在相同生成方式和 NFE 内比较；下面第一、二行基线是 Causal Forcing++，第三行是 CausVid。数值均为“基线 → DuoMatching”：

| 设置 | VBench 总分 | 语义分 | Imaging | 动态分 |
| --- | --- | --- | --- | --- |
| 逐帧生成，2 步 | 80.67 → 83.53 | 68.42 → 72.84 | 69.82 → 73.48 | 94.44 → 93.61 |
| 逐块生成，4 步 | 82.79 → 83.51 | 70.84 → 71.61 | 70.13 → 72.41 | 80.56 → 76.67 |
| 双向整段生成，4 步 | 80.63 → 81.45 | 69.23 → 70.89 | 73.00 → 73.77 | 36.67 → 39.17 |

- Table 2 的人工评测邀请 23 人，每人判断 40 对同提示词视频。相对 Causal Forcing++，总体偏好率为 80.87%，视觉质量为 81.30%，语义为 79.57%，时间与运动为 49.57%。总体更受偏好不等于运动更好。
- Table 3：不用额外图像教师时总分 80.67；把 Wan2.1-14B 用作单帧教师后为 81.43；SDXL、FLUX.2-4B、Qwen-Image 分别为 81.47、82.64、83.53。这支持显式 marginal matching 与较强图像教师均有贡献，不能把全部收益归给其中一个。
- Table 4：直接在压缩的视频 slice 上加 marginal DMD，动态分为 76.94；使用 Bridge 后为 93.61，无单帧监督基线为 94.44。Direct 与 Bridge 的峰值 tensor 显存分别为 59.40、59.41 GB；Decode–Encode 在所测配置下 OOM。
- Table 5：LVS 的 $K=4$ 总分 83.53，高于均匀随机的 82.61 和等长分段的 82.68；增至 $K=8$ 后，LVS 的动态分与总分都下降。
- Table 6：权重从 $\omega=0.4$ 提至 0.8，动态分从 93.61 降到 87.78，总分从 83.53 降到 81.57。单帧监督并非越强越好。

### 开源与视频结果（2026-10-08 核对）

- [官方 GitHub](https://github.com/JohnZhan2023/DuoMatching)：已公开训练、推理、LatentBridge 预训练代码和配置，代码采用 Apache-2.0。
- [模型权重](https://huggingface.co/JohnZhan/DuoMatching/tree/main)：文件列表包含 `model.pt` 和 `latent_bridge/`。README 将发布模型描述为从 Wan2.1-T2V-14B 蒸馏的两步模型；14B 指视频教师，学生沿用 Wan2.1-1.3B 因果模型。这里确认了两步权重，未据此认定论文中所有设置的检查点都已发布。
- [项目页视频对比](https://johnzhan2023.github.io/DuoMatching/#comparison)：有与 Causal Forcing++、One-Forcing、Reward Forcing、CausVid 的四组可播放对照，每组 10 对。已核对一个样例的 [DuoMatching 视频](https://johnzhan2023.github.io/DuoMatching/videos/causal-forcing/causal-forcing-051-ours.mp4)与 [CF++ 视频](https://johnzhan2023.github.io/DuoMatching/videos/causal-forcing/causal-forcing-051-baseline.mp4)，两条资源均可访问。

这些是作者发布的结果与代码；本次未运行训练或推理复现。视频转载许可未明确，因此只链接官方展示。

## 我的提问

### Q1：所以就是在 video DMD 上增加 image DMD，并为了正确加这个损失引入 LatentBridge？

是。这是本次讨论最后确认的理解：保留 joint video DMD，再加 marginal image DMD。Bridge 将含时间压缩的信息转为指定单帧的图像 latent，让图像教师提供的监督有合适的作用对象；LVS 则安排有限的抽帧预算。

此前已经有图像先验用于视频生成的工作。本文的具体创新是联合两种 DMD 目标、分析它们的关系，并用 Bridge 与 LVS 支持带时间压缩 latent 的现代视频学生。

### Q2：它与 Self Forcing 是什么关系，能替换 Self Forcing 的一个阶段吗？

Self Forcing 重点改变训练时的生成过程：接着自己生成的历史帧做 self-rollout，以减轻训练条件与推理条件的差异。DMD 是它可以使用的整段分布匹配损失之一；原文也评估了 SiD 和 GAN。

DuoMatching 重点改变监督目标。原则上可以保留 Self Forcing 的 rollout，在原有 video DMD 旁边增加 image DMD；改的是后训练阶段中的损失，整段 DMD 项仍保留。该组合还需要单独预训练 Bridge，并增加图像分支的 fake score estimator。

这是从方法接口作出的可行组合推断。DuoMatching 实际使用 Causal Forcing++ 的 causal CD 检查点并进一步训练，Self Forcing 是比较基线；论文未给出“原版 Self Forcing + DuoMatching”的单独受控实验。

参考：[Self Forcing §3–4](https://arxiv.org/html/2506.08009v2)、[DuoMatching §4](https://arxiv.org/html/2610.03543v1)。

### Q3：Self Forcing 是单阶段训练吗？

Self Forcing 的核心算法是一个 self-rollout 后训练阶段，但原论文从预训练基座出发的完整做法包含 ODE 初始化，再进行 self-rollout 分布匹配。不要把“方法的核心阶段”与“完整实验流程”混在一起。

| 方法 | 少步学生初始化 | 后续训练 |
| --- | --- | --- |
| 原版 Self Forcing | 双向教师的 ODE 轨迹 → 因果学生的终点回归 | self-rollout + 整段分布匹配，主要结果用 DMD |
| Causal Forcing | 先训练多步 AR 教师，再用该教师做 causal ODE 蒸馏 | self-rollout + asymmetric DMD |
| Causal Forcing++ | 同样先训练 AR 教师，以 causal CD 替换 ODE 蒸馏 | self-rollout + asymmetric DMD |
| DuoMatching 的因果实验 | 使用 Causal Forcing++ 的 causal CD 检查点 | joint video DMD + marginal image DMD |

原版 Self Forcing 的教师仍是双向模型，因此不能把它的初始化不加区分地叫作“AR 教师的 causal ODE”。这是本次讨论中需要纠正的用词。参考：[Causal Forcing §3.3](https://arxiv.org/html/2602.02214)、[Causal Forcing++ §3](https://arxiv.org/html/2605.15141v4)。

### Q4：ODE 初始化还是 flow matching loss，训完仍不是 few-step model 吗？

需要区分普通 FM 训练与 ODE 蒸馏。采用 $x_t=(1-t)x_0+t\epsilon$ 的日程时，普通 flow matching 的监督为：

$$
\mathcal L_{\mathrm{FM}}
=\mathbb E\left[\left\Vert v_\theta(x_t,t)- (\epsilon-x_0)\right\Vert_2^2\right].
$$

它学习速度场，通常通过多次求解 ODE 生成样本。ODE 初始化则从教师的完整轨迹取得状态 $x_t^{\mathrm{ODE}}$ 和同一轨迹终点 $x_0^{\mathrm{ODE}}$，训练：

$$
\mathcal L_{\mathrm{ODE}}
=\mathbb E\left[\left\Vert
G_\theta(x_t^{\mathrm{ODE}},t)-x_0^{\mathrm{ODE}}
\right\Vert_2^2\right].
$$

条件中的文本与历史帧在上式中省略。它学习当前状态到终点的映射，已经面向少步生成训练；后续 DMD 继续改善质量，不是必须做过 DMD 才称为少步模型。“训练后仍是多步模型”对应 Causal Forcing 中最前面的 AR 教师训练。

即使骨干仍输出 velocity，也可以经下面的参数化得到终点预测：

$$
G_\theta(x_t,t)=x_t-\sigma_t v_\theta(x_t,t).
$$

保留 velocity 输出形式或同样使用 MSE，不意味着监督仍是普通 FM。在 CausVid 官方实现中，wrapper 将 flow prediction 转为 $x_0$ prediction，再对教师轨迹的最后一个 latent 计算 MSE。

实现依据：[ODERegression.generator_loss](https://github.com/tianweiy/CausVid/blob/master/causvid/ode_regression.py)、[WanDiffusionWrapper 的输出转换](https://github.com/tianweiy/CausVid/blob/master/causvid/models/wan/wan_wrapper.py)。Self Forcing 的官方 README 明确说明它的 ODE 初始化沿用 CausVid 的流程；这里未对 DuoMatching 的完整训练实现做逐行审计。

### Q5：有什么值得借鉴？

Agent 的建议：先按失败类型定位监督缺口，让适合的教师负责对应对象。这里视频教师负责跨帧关系，图像教师提供单帧细节与语义；接入前还必须检查 latent 表示是否真的对应被评价的对象。

若做小规模验证，固定初始化、视频教师、数据和推理预算，依次比较 joint DMD、直接单帧监督、Bridge、LVS，并扫描损失权重和抽帧数。同时看画质、提示词遵循、运动与跨帧一致性，避免只用总分判断收益。额外教师只在训练中使用，也是保留学生推理成本的一种做法。

### Q6：最终 loss 是两项的结合，但 Causal Forcing 不是已经在做 joint DMD 吗？

是。Causal Forcing 最后阶段的 asymmetric DMD 本来就匹配整段视频分布，属于这里说的 joint DMD。DuoMatching §3.2 明确说明 joint 分支沿用 Causal Forcing，新增的是图像教师提供的 marginal DMD；视频学生的最终目标就是上面的两项加权和。Bridge 的 $\ell_1$ loss 用于单独预训练 Bridge。

Joint matching 本身也约束 marginal：若学生完全匹配视频教师的联合分布，单帧分布自然也会匹配该视频教师的单帧分布。额外图像分支提供另一份单帧参考分布，并让单帧修正有独立目标；这也解释了为什么收益既可能来自显式 marginal matching，也可能来自较强图像教师。

“整段视频训练”描述数据与训练过程，不能单凭这点判断采用了 DMD。这里具体指 Causal Forcing 第三阶段的分布匹配；前面的教师训练、ODE/CD 初始化使用各自的损失。

### Q7：与 Causal Forcing++ 的对比是否从同一个 base model 出发？为什么画质差距明显？

主要因果逐帧实验的模型与初始化来源一致，已核实的是以下几项：

| 项目 | Causal Forcing++ | DuoMatching |
| --- | --- | --- |
| 视频学生来源 | Wan2.1-1.3B 衍生的因果模型 | 沿用该因果学生 |
| 少步初始化 | causal CD | 加载 CF++ 公开的 `framewise/causal_cd.pt` |
| 最后阶段的视频 real score 教师 | Wan2.1-14B | Wan2.1-14B |
| 独立图像分支 | 无 | Qwen-Image、image fake score 与 Bridge |

依据：[CF++ §4.1](https://arxiv.org/html/2605.15141v4#S4.SS1)、[DuoMatching README](https://github.com/JohnZhan2023/DuoMatching#training)及[公开训练配置](https://github.com/JohnZhan2023/DuoMatching/blob/main/configs/duomatching_train.yaml)。Duo 从第二阶段的 CD 初始化开始进行最终训练，不是把完成第三阶段的 CF++ 成品检查点再续训。

同一起点不意味着监督资源相同。Duo 多了 Qwen-Image 的能力，因此不能把全部画质提升都归因于损失形式改变。Table 3 的同视频教师单帧分支为 81.43，Qwen-Image 分支为 83.53，支持显式 marginal 目标与图像教师能力均有贡献。展示图是具体案例，平均收益仍应结合 Table 1–2 的自动与人工评测判断。

对照信息仍有待补充：[CF++ 原论文 Table 1](https://arxiv.org/html/2605.15141v4#S4.T1) 的逐帧两步总分为 84.14，而 [DuoMatching Table 1](https://arxiv.org/html/2610.03543v1#S4.T1) 中该基线为 80.67。本次在实验说明中未找到对差异的完整解释；具体基线检查点、采样参数和评测口径待补充。不能跨表直接比较，也不能把模型来源一致写成已验证所有实验变量一致。

### Q8：这个方法是否也不需要视频训练数据？

最终 joint + marginal DMD 阶段不需要真实视频样本。Algorithm 1 只抽提示词和噪声，用当前学生生成视频；视频分支监督这些生成的完整视频 latent，图像分支监督其中经 Bridge 映射的抽样帧。两个 fake score estimator 也在学生新生成的样本上训练，不要求准备与提示词配对的真实视频或图像。

公开的因果训练配置设置 `load_raw_video: false`，`data_path` 指向 `vidprom_filtered_extended.txt` 提示词文件。这里需要区分 DMD 学生训练与此前准备模型的阶段：

| 阶段 | 数据需求 |
| --- | --- |
| 最终 DuoMatching DMD 学生训练 | 提示词、随机噪声、现场生成的视频及抽样帧，无需真实视频样本 |
| LatentBridge 预训练 | 真实视频及其对应帧的两种 VAE 编码；论文使用 29,400 个 OpenVid 视频 |
| CF++ 的 AR 教师与 causal CD 初始化 | 原流程使用视频数据；DuoMatching 实验复用其公开初始化检查点 |

依据：[DuoMatching Algorithm 1](https://arxiv.org/html/2610.03543v1#A7)、[§4 的 Bridge 数据说明](https://arxiv.org/html/2610.03543v1#S4)、[公开因果训练配置](https://github.com/JohnZhan2023/DuoMatching/blob/main/configs/duomatching_train.yaml)及 [CF++ §4.1](https://arxiv.org/html/2605.15141v4#S4.SS1)。因果实现已核对到提示词配置；双向实验在文中写使用 Mixkit，本次未核对该分支的数据加载实现。

如果复用视频教师、图像教师、causal CD 和预训练 Bridge，做最终因果 DMD 训练可以只准备提示词。若要从头训练或重新适配 Bridge，按论文流程仍需视频数据；因此完整方法不能概括为完全无需视频数据。

## 局限与疑问

- 运动存在取舍：逐块四步设置的动态分下降，人工评测相对 Causal Forcing++ 的时间与运动偏好接近持平；画质收益不能写成所有能力都提高。
- 理论 Appendix D 的改善结论针对理想分布空间、足够小的权重，并依赖图像参考的单帧近似误差不高于视频教师 marginal 等假设。它没有证明任意图像教师、任意权重和有限学生都必然受益。
- Bridge 是近似映射。Appendix F 在映射误差有界、损失满足条件时分析改善能否传到真正解码出的帧；Bridge 的 loss 下降不自动等于实际视频质量改善。
- 实验集中在 81 帧、480×832 的视频生成。长时 rollout、音画同步、口型及参考身份保持不能由这些结果直接推出。
- 论文称不增加推理计算；额外图像教师、fake score 与 Bridge 会增加训练工作，但本文未给出完整训练开销或端到端速度对比，不能把 Bridge 的单独预训练时间当作整个方法的成本。
- 与原版 Self Forcing 的直接组合值得验证，但本次没有组合实验，也未复现性能。
- CF++ 基线在两篇论文中的报告分数不同，具体对照条件仍需核查；目前确认的是学生、初始化和视频教师来源。
- 2026-10-08 核对时，公开 `main` 的默认配置写 `max_train_steps: 1800`，论文因果实验写 1,000 步。当前代码默认值与论文描述不完全一致，发布权重的实际训练设置待补充。

## 我的判断

本次确认的理解是“video DMD 加 image DMD，Bridge 让单帧监督正确作用到视频 latent”；同时澄清了 Self Forcing 的 rollout 训练方式、ODE/CD 少步初始化和 DMD 后训练之间的关系。后续讨论进一步明确：joint 指整段联合分布，marginal 指单帧分布；Causal Forcing 最后阶段已有 joint DMD，Duo 增加单帧目标。用户还提出了基线同源性与画质差距的疑问，模型来源已核实，严格对照条件仍待补充。

Agent 的阅读判断：最值得参考的是 Bridge 的表示选择和控制监督强度的消融。Direct 的运动明显下降、Bridge 基本恢复运动，说明损失施加的位置是实质问题；强教师与显式 marginal 监督的贡献也有分别验证。是否复现及最终投入优先级，用户尚未作出判断。

## 下次只看这些

1. Joint 看所有帧共同出现的分布，marginal 看抽样单帧分布。CF 最后阶段已有 joint DMD；Duo 的总损失保留该项，再加 weighted marginal image DMD。
2. Bridge 输入前后两个视频 slice 和局部帧位置，预测图像 latent；预训练后冻结但保留反传。LVS 按变化分段，再每段随机抽一个位置。
3. 初始化与后训练要分清：普通 FM 训练得到多步教师，ODE/CD 蒸馏已训练少步学生，self-rollout DMD 继续改善生成分布；DuoMatching 改的是最后的监督目标。
