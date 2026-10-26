---
title: "HyperFlow Infinity"
paper_url: "https://github.com/Video-Rebirth/hyperflow-infinity"
authors: "Video Rebirth"
venue: "GitHub / 开源项目"
published: "2026"
read_date: "2026-10-26"
read_at: "2026-10-26T14:52:00+08:00"
status: "已读"
tags: ["Video Generation", "Streaming Video", "Agentic Systems", "Diffusion Models"]
search_terms: ["HyperFlow-Infinity", "MiniMax-H3", "Qwen3-VL", "video harness", "Ref2VA", "scene graph", "latent handoff", "22 frames", "接缝抖动", "self-play"]
one_liner: "Qwen 观察实际生成视频、更新文字记忆并导演下一段，H3 用固定人物图和末尾 7 个 latent 续拍；裁掉 22 帧用于去重，不能保证接缝平滑。"
---

## 研究问题

短视频模型怎样从人物和世界设定出发，持续生成有前后关系的长视频，同时维持人物外观、局部动作和剧情状态？HyperFlow Infinity 提供围绕 H3 + HyperFlow 运行的控制程序，即 video harness：逐段观察、记忆、写提示词、生成、保存和拼接。

本条目整理的是开源项目及代码阅读，不是对一篇已发表论文的复现。依据为 [官方仓库](https://github.com/Video-Rebirth/hyperflow-infinity)、[项目页](https://www.videorebirth.com/lp/hyperflow-infinity)，以及 2026-10-24 首次提交的 [代码快照 6c3291f](https://github.com/Video-Rebirth/hyperflow-infinity/tree/6c3291f319de9a81f63538af9b48da7cbc0b6a3f)。本次未运行模型推理，也未独立评测一小时视频的质量。

## 核心方法

![HyperFlow Infinity 的观察、记忆、导演、生成流程，以及裁掉新片段前 22 帧后的实际接缝](media/hyperflow-infinity/method-and-overlap.png "图 1：根据 Video Rebirth 官方 README 的 How It Works 流程图改绘，新增中文说明和重叠裁剪示意；原始 Mermaid 源码采用 Apache-2.0，许可证与署名随图保存。时间轴是实现示意，不是实测画面。")

- **同一套 Qwen 权重承担多个角色。** 默认本地服务加载 H3 自带的 Qwen3-VL text encoder，使用其语言生成能力。Memory 看上一段视频、提取变化；Director 看视频和记忆、写下一段提示词；Compact 压缩旧事件。这些角色通过不同提示词切换，权重保持冻结。作者所说的 self-play 指持续生成和观察的推理过程，代码没有在线训练或奖励优化。[Qwen 服务](https://github.com/Video-Rebirth/hyperflow-infinity/blob/6c3291f319de9a81f63538af9b48da7cbc0b6a3f/src/hyperflow_infinity/qwen_service.py#L155-L193)
- **记忆记录实际发生的变化。** Scene graph 是 JSON：节点表示人物、地点、物品，状态表示它们当前怎样，事件记录谁对什么做了什么。每段最多新增两个事件，节点状态最多 12 个英文词。默认事件超过 22 条时，把最早 10 条压成章节；导演读取章节、人物状态、地点和物品名称，以及最近 8 条事件。它是由视觉观察生成的文字记忆。[memory.py](https://github.com/Video-Rebirth/hyperflow-infinity/blob/6c3291f319de9a81f63538af9b48da7cbc0b6a3f/src/hyperflow_infinity/memory.py)
- **导演区分意图和事实。** 上一段 prompt 只是拍摄要求。下一段必须根据实际视频与观察记录推进，不能把未拍出的动作当成已完成。例如要求 Alice 拿起钥匙，但视频只拍到她看向钥匙，就不能在下一段假设钥匙已经在她手中。这个例子用于解释规则，并非作者报告的测试案例。[证据优先规则](https://github.com/Video-Rebirth/hyperflow-infinity/blob/6c3291f319de9a81f63538af9b48da7cbc0b6a3f/src/hyperflow_infinity/prompts.py#L75-L84)
- **三种信息分别维持三种连续性。** 固定人物参考图约束主角外观；上一段末尾 7 个视频 latent 帮助动作和画面续接；scene graph 与提示词中的人物描述维持剧情和状态。续段中，人物图同时经过 VAE 和文本侧视觉编码器；末尾 latent 直接读取，不经过 RGB 解码再编码。
- **末尾 latent 作为参考条件，并修改位置编码。** 代码把末尾 7 个 latent 加入 `condition_latents`，使它们的位置与新片段前 7 个 target latent 对应，引导模型重画前段结尾。这里没有把 target prefix 硬固定，也没有跨片段保留 DiT 的 KV cache。新片段生成后按固定帧数裁掉重叠部分。[handoff.py](https://github.com/Video-Rebirth/hyperflow-infinity/blob/6c3291f319de9a81f63538af9b48da7cbc0b6a3f/src/hyperflow_infinity/handoff.py#L25-L124)
- **用规则抑制剧情停滞。** 导演输出场景、是否换场、本段推进的事件和 H3 提示词。程序检查近期事件的英文词集合重合度；默认同场景达到 8 段后要求换场；连续三段观察到字面相同的事件与状态时要求改变局面。不合格的导演输出最多尝试 3 次。这是提示词和规则校验，没有生成多个视频再择优的质量搜索。[导演校验](https://github.com/Video-Rebirth/hyperflow-infinity/blob/6c3291f319de9a81f63538af9b48da7cbc0b6a3f/src/hyperflow_infinity/qwen.py#L111-L171)

整个流程使用 Ref2VA。首段需要开场参考图或视频；没有参考时可先用 Qwen-Image 画图。每段保存请求、响应、视频、latent、记忆和哈希，支持暂停续跑及从历史片段分支。对已经发送却没有完整结果记录的请求，代码会停下来要求检查，避免不确定状态下直接重复生成。[执行与保存](https://github.com/Video-Rebirth/hyperflow-infinity/blob/6c3291f319de9a81f63538af9b48da7cbc0b6a3f/src/hyperflow_infinity/engine.py#L279-L352)

## 关键发现

| 项目 | 代码或作者报告的数值 | 如何理解 |
| --- | --- | --- |
| 视频帧率 | 24 fps | 生成和拼接共用的时间基准 |
| 首段 | 345 帧，14.375 秒 | 开场参考通过原生 Ref2VA 输入 |
| 每个续段 | 生成 141 帧，裁掉前 22 帧，保留 119 帧 | 每轮净增加约 4.96 秒 |
| 官方演示 | 1344×768、一小时 Alice 视频 | 是作者展示，不等于任意长视频质量保证 |
| 官方速度：HF 后端 | 每新增约 5 秒，中位耗时 78.1 秒 | 8×H200、768p、HyperFlow 8 步、FA3 |
| 官方速度：SGLang 后端 | 同条件中位耗时 49.2 秒 | 包含导演、记忆、视频生成和裁剪，未达到实时 |

帧数来自 [engine.py](https://github.com/Video-Rebirth/hyperflow-infinity/blob/6c3291f319de9a81f63538af9b48da7cbc0b6a3f/src/hyperflow_infinity/engine.py#L26-L50)；演示和性能来自 [官方 README](https://github.com/Video-Rebirth/hyperflow-infinity/blob/6c3291f319de9a81f63538af9b48da7cbc0b6a3f/README.md)。这些耗时是作者测量，本次没有复现。

## 我的提问

### Q1：这个 video harness 到底做了什么？

它把短视频模型接入一个持续执行的程序：观察上一段 → 更新记忆 → 决定下一段 → 生成与裁剪 → 保存结果 → 重复。模型权重冻结，但每轮的视频、记忆与提示词会改变。值得关注的实现是观察如何约束后续剧情，以及 latent 条件如何连接相邻片段。

“不引入额外模型”适用于默认核心架构复用 H3 的 Qwen3-VL。没有开场参考时仍会调用 Qwen-Image，可选的 OpenRouter 路径也可替换 VL 模型，因此不能把这句话理解成所有配置都只有一个模型。

### Q2：“裁掉开头重画的 22 帧”是什么意思？

新片段 B 的开头被设计成重画上一段 A 的结尾。例如 A 的结尾是“伸手抓住门把手”，B 的开头重画这部分，随后继续“推门、走进去”。重新生成的画面可能与 A 略有差异，这个动作例子只解释设计意图。

固定操作是：**A 完整保留；B 的前 22 帧裁掉，只拼接 B 的第 23～141 帧。** 上一段的结尾没有被删掉。

$$
N_{\mathrm{keep}}=141-22=119,\qquad
\Delta T=\frac{119}{24}\approx 4.96\ \mathrm{s}.
$$

$N_{\mathrm{keep}}$ 是每个续段保留的帧数，$\Delta T$ 是新增时长。裁掉的 22 帧约为 0.92 秒，用于去除重叠内容。代码按固定位置裁剪，并没有先检测哪些帧画错。[裁剪实现](https://github.com/Video-Rebirth/hyperflow-infinity/blob/6c3291f319de9a81f63538af9b48da7cbc0b6a3f/src/hyperflow_infinity/media.py#L22-L32)

### Q3：既然开头可能画得不同，剪掉以后也可能抖动，对吧？

**对，实际接缝是 A 的最后一帧 → B 的第 23 帧，那里仍可能发生跳变。** 条件 latent 会影响新片段，但没有强制这两帧中的人物位置、姿态、镜头和光照连续。即使位置接上，动作速度也可能突然改变。

需要保留的纠正是：**裁剪的目的在于去重；裁剪本身不是接缝修复。** 最初的动作示意表达的是期望效果，不能当成逐帧精确衔接的保证。当前代码没有看到专门针对视觉接缝的质量检测或自动修复。风险由实现方式可以推知，但跳变的频率与严重程度仍需实际生成验证。

## 局限与疑问

- **长时质量仍需独立评测。** 作者展示一小时视频，公开代码能确认条件输入与固定裁剪流程；这两者都不足以保证长时剧情有趣、身份稳定或每个接缝平滑。与 Echo-Infinity 的示例对照也没有提供共同的量化剧情评测协议。
- **当前只固定一个人物视觉参考。** 续段强制使用一张身份图，配角主要依赖外观文字描述和状态记忆。单主角演示不能直接推广到多人身份都被可靠保持。
- **音频没有沿用上一段的 audio latent。** 虽然同时保存视频和音频 latent，下一段只读取视频部分。每段重新生成声音，然后按视频时间裁剪、拼接原始 PCM；这可以避免逐段 AAC padding 累积，却不能证明声线或音乐连续。默认 Qwen 观察服务也未见音频理解输入。[生成与保存](https://github.com/Video-Rebirth/hyperflow-infinity/blob/6c3291f319de9a81f63538af9b48da7cbc0b6a3f/src/hyperflow_infinity/generator.py#L135-L198)、[音频拼接](https://github.com/Video-Rebirth/hyperflow-infinity/blob/6c3291f319de9a81f63538af9b48da7cbc0b6a3f/src/hyperflow_infinity/media.py#L56-L97)
- **记忆没有严格的恒定大小保证。** 旧事件会压缩，但章节和节点仍继续累积，导演读取全部章节。不能仅凭最近事件窗口，就声称任意时长的上下文开销恒定。
- **观察与去重规则会漏错。** Qwen 可能误读视频；事件词重合与字面状态检查也不能完整判断语义重复、剧情质量或动作完成情况。
- **待验证：** 观察每个拼接点前后的人物位置、动作速度、镜头、光照和音频变化；比较实际接缝与片段内部正常运动，避免把所有帧间差异都算成抖动。

## 我的判断

本次讨论最重要的判断是：**条件续接能帮助连贯，但“裁掉重叠部分”不等于“保证接缝平滑”。** 这是阅读作者 seamless continuation 表述时必须保留的限定。

基于代码，值得借鉴的是把实际生成结果放回下一轮决策，并明确区分拍摄意图与已经发生的事实。当前证据支持它是一套能持续续拍的推理程序；剧情质量、多人一致性、音频连续性和接缝表现，仍需分别验证。不能把 self-play 直接解释为模型在生成过程中学会了更好的导演能力。

## 下次只看这些

1. 三种连续性分开记：固定人物图管身份，末尾 7 个 latent 管局部续接，文字记忆管剧情状态。
2. 141 帧裁掉前 22 帧，只保留后 119 帧；实际接缝是 A 最后一帧 → B 第 23 帧，仍可能跳变。
3. 先看 [handoff.py](https://github.com/Video-Rebirth/hyperflow-infinity/blob/6c3291f319de9a81f63538af9b48da7cbc0b6a3f/src/hyperflow_infinity/handoff.py) 的条件和位置安排，再看 [prompts.py](https://github.com/Video-Rebirth/hyperflow-infinity/blob/6c3291f319de9a81f63538af9b48da7cbc0b6a3f/src/hyperflow_infinity/prompts.py) 如何要求根据实际视频继续剧情；效果结论回到生成结果验证。
