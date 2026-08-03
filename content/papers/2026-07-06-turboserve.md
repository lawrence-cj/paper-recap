---
title: "TurboServe: Serving Streaming Video Generation Efficiently and Economically"
paper_url: "https://arxiv.org/abs/2606.19271"
authors: "Youhe Jiang et al."
venue: "arXiv"
published: "2026"
read_date: "2026-07-06"
status: "已读"
rating: 3
tags: ["Video Generation", "Serving Systems", "Scheduling", "Autoscaling"]
one_liner: "TurboServe 把流式视频生成视为有持续状态的长会话，通过 chunk 合批、CPU 挂起、边界迁移和闭环扩缩容同时压低尾延迟与 GPU 成本。"
---

## 研究问题

流式视频生成不是一次性短请求：每个 session 持续产生 chunks、保存跨 chunk 状态，且到达率与生成长度变化大。传统图像/LLM serving 的无状态调度容易造成 GPU 碎片、负载倾斜、状态搬运阻塞和过度供给。

## 核心方法

- **Chunk coalescing**：把处于兼容阶段的不同 session chunk 合并执行，提高 GPU 利用率。
- **Suspend/resume**：暂时不运行的 session state 可下放 CPU，释放 GPU 容量。
- **Migration-aware placement**：用 min-max 目标减少最忙 GPU 的负载，并考虑迁移成本。
- **Chunk-boundary migration**：只在安全边界移动状态，避免破坏正在执行的 diffusion chunk。
- **Closed-loop autoscaling**：使用 hysteresis、比例控制和 capacity headroom，按测得的队列与处理能力调整 GPU 数。

调度目标可概括为

$$
\min_{a}\;\max_g L_g(a)
+\lambda C_{\mathrm{migration}}(a),
$$

其中 $a$ 是 session 到 GPU 的分配，$L_g$ 是设备负载，第二项阻止为了短期均衡而频繁搬运大状态。

## 关键发现

- 固定成本时，worst-chunk latency 平均下降 **37.5%**、最高下降 **51.6%**。
- 固定 latency 目标时，GPU 成本平均下降 **37.2%**、最高下降 **49%**。
- 去掉 migration，成本增加约 **15%**；去掉 autoscaling，成本增加约 **42.9%**。
- 64 GPU 下 scheduler overhead 低于 **15 ms**；在线成本比 oracle 高约 **6.1%**。

## 我的提问

### Q1：为什么必须把 session state 纳入调度？

连续 chunks 复用历史 latent/KV/模型状态；只按当前 batch 长度迁移请求，会把计算均衡问题变成 PCIe/网络搬运问题。调度器必须比较未来负载收益与状态迁移成本，并在 chunk boundary 操作。

### Q2：它对模型侧优化有什么启发？

模型输出 chunk 的大小、状态大小和可中断边界直接决定系统调度空间。设计 streaming video model 时应同时暴露稳定 checkpoint、可序列化 session state 和可预测的每 chunk cost，而不是训练完成后才交给 serving 层补救。

## 局限与疑问

- 结果基于特定 LongLive 风格模型与真实 traces，不等于对所有 Vidu/Wan 类服务有效。
- warm pool 与 cold-start 成本可能被低估；baseline 强度也有限。
- 缺少完整 TTFF、公平性、jitter、生成质量和异构请求指标。
- autoscaling 参数依赖 workload distribution，分布突变时稳定性仍需验证。

## 我的判断

这篇更像 production serving blueprint，而不是单个 kernel 技巧。最应复用的是 stateful session abstraction 和 chunk-boundary contract；实际部署需用自己的 trace 重做容量模型与 autoscaling 稳定性测试。

## 下次只看这些

1. 流式视频请求是有状态 session，不是独立短任务。
2. 合批、CPU suspend、边界迁移和闭环扩缩容必须联合设计。
3. 报告平均成本之外，还要实测 TTFF、尾延迟、jitter 与迁移开销。
