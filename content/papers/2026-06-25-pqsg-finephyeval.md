---
title: "FinePhyEval: Evaluating Physical Consistency in Video Generation with PQSG"
paper_url: "https://arxiv.org/abs/2606.25306"
authors: "待补充"
venue: "arXiv"
published: "2026"
read_date: "2026-06-25"
status: "已读"
rating: 3
tags: ["Video Evaluation", "Physical Consistency", "Reward Model", "Benchmark"]
one_liner: "PQSG 把 prompt 拆成对象、动作和物理关系的提问图，再由 VLM 对视频逐项回答，用细粒度证据代替单一物理总分。"
---

## 研究问题

视频的物理一致性包含接触、重力、材料变化、守恒和事件顺序，单一 VLM 打分既不稳定也难定位错误。FinePhyEval 希望建立带人工校准的 benchmark，并用结构化问题图让自动评估更细粒度。

## 核心方法

- 从 prompt 解析对象、动作、状态和物理关系，构建 Prompt Question Scene Graph（PQSG）。
- 将每条要求转为可由视频证据回答的 yes/no 或分级问题。
- VLM 对视频逐项回答并汇总为物理一致性分数；缓存问题和答案便于审计。
- FinePhyEval 数据同时保存 prompt、生成视频、人工 Likert 评分、PQSG 与自动 QA。

可把总分理解为问题级证据的加权汇总：

$$
R_{\mathrm{PQSG}}(v,p)
=\frac{\sum_{q\in Q(p)}w_q\,r_q(v)}{\sum_{q\in Q(p)}w_q},
$$

其中 $Q(p)$ 是由 prompt 生成的问题集合，$r_q$ 是 judge 对问题 $q$ 的判断。

## 关键发现

- 它本身是评测 pipeline/critic，而不是直接提升视频模型的 post-training 方法。
- 历史记录中可用的人类标注视频约 **195** 条，适合校准或评测，不足以直接充当大规模偏好训练集。
- 结构化 QA 可暴露失败发生在对象、动作还是物理关系上，比只保存总分更有再利用价值。

## 我的提问

### Q1：如何把评测方法用于 post-training？

可以将 PQSG 用作 verifier：做 rejection sampling、候选排序、DPO pair 构造或失败类型挖掘。应先在人类标注子集上校准每类问题可靠性，再按类别决定权重，而不是直接最大化原始总分。

### Q2：能否直接把 FinePhyEval 当训练数据？

不建议。规模太小且 judge 生成内容占比高，更适合作为 held-out evaluation/calibration set。训练数据应另行扩展，并保持 prompt、问题、证据帧、答案和置信度的完整链路。

## 局限与疑问

- 问题生成和视频回答都依赖 VLM，存在共享偏差、遗漏和语言先验。
- 二元 QA 会压缩连续程度，也可能忽略画面质量和语义吸引力。
- 优化同一 judge 容易 reward hacking，最终仍需独立模型与人工盲测。

## 我的判断

最适合用作“可解释的物理 failure taxonomy”。若用于训练，应把它放在数据过滤与 pair mining 层，并保留人工校准集作为防止奖励漂移的锚点。

## 下次只看这些

1. PQSG 是结构化评测/critic，不是训练算法。
2. 约 195 条人工数据只适合校准和评测。
3. 用于 post-training 时保留逐题证据，并防止 judge reward hacking。
