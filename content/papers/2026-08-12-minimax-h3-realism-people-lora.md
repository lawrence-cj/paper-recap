---
title: "Project: MiniMax H3 Realism People LoRA"
paper_url: "https://huggingface.co/fal/MiniMax-H3-Realism-People-LoRA"
authors: "Lovis Odin / fal"
venue: "Hugging Face Project"
published: "2026"
read_date: "2026-08-12"
read_at: "2026-08-12T14:46:25+08:00"
status: "已读"
tags: ["Video Generation", "LoRA", "Data Curation", "Fine-Tuning"]
one_liner: "这是一个用 176 条高分辨率真人短视频把 MiniMax H3 推向自然皮肤、表情和纪录片式运动的 LoRA；当前权重为 rank 32、1500 steps，按有效 batch 1 推算约 8.5 epochs，但原始数据与完整训练配置未公开。"
---

## 研究问题

MiniMax H3 已能生成真人视频，但近景人物仍可能出现塑料皮肤、眼神与微表情不连贯、手部动作不自然，以及过于“AI 化”的灯光和镜头运动。这个项目没有提出新模型，而是探索：能否用一百多条严格筛选、规范化和详细标注的真人短视频，通过 LoRA 把这些窄域缺陷补上，同时保留 H3 的原生同步音频与 T2V/I2V/R2V 能力。

项目主页与核查入口：[当前模型卡](https://huggingface.co/fal/MiniMax-H3-Realism-People-LoRA)、[文件列表](https://huggingface.co/fal/MiniMax-H3-Realism-People-LoRA/tree/main)、[提交历史](https://huggingface.co/fal/MiniMax-H3-Realism-People-LoRA/commits/main)、[19 组同 seed 对比视频](https://huggingface.co/fal/MiniMax-H3-Realism-People-LoRA/blob/main/before-after-comparison.mp4)、[发布讨论](https://www.reddit.com/r/StableDiffusion/comments/1vkubdm/i_trained_an_opensource_realism_lora_for_minimax/)。

## 核心方法

- **数据**：176 条人工筛选的 live-action clips，主体覆盖人像、脸部、工人、运动员与日常人物；建立在第一版 Realism 数据中最强样本之上。早期模型卡还明确写过包含 curated stock footage，但当前模型卡删去了具体来源，原始视频、URL 清单和 captions 均未公开。
- **预处理**：检测慢动作并恢复到自然速度；全部规范为严格的 **24.000 FPS**；每条视频配 structured scene caption。具体时长、裁剪策略、分辨率 bucket 和 caption schema 未公开。
- **训练搜索**：以 MiniMax H3 为 base，使用 fal H3 trainer；共比较 16 个由 steps、rank、learning rate 与训练分辨率组成的配置，并用相同 prompt、相同 seed、LoRA on/off 的人评对比选择最终权重。
- **最终配置**：当前模型卡给出 **rank 32、1500 steps、high-resolution bucket**。作者的判断是训练分辨率比更高 rank 或更长训练更重要，因为皮肤毛孔、细发和胶片颗粒属于高空间频率信息。
- **权重内嵌元数据**：`base_model=minimax-h3-fl2va`、`training_strategy=text_to_video`、`first_frame_conditioning_p=0.0`、`global_step=1500`、`lora_rank=32`。这意味着公开权重本身按 T2V 训练；模型卡声称同一个 adapter 可用于 T2V、I2V 和 R2V，不等于它接受过首帧条件训练。
- **使用方式**：prompt 以 trigger `r34l1sm` 开头；作者建议 LoRA scale **1.0**，较轻效果用 **0.6–0.8**。仓库实际文件名是 [`h3-realism-people-t2v-i2v-r2v.safetensors`](https://huggingface.co/fal/MiniMax-H3-Realism-People-LoRA/blob/main/h3-realism-people-t2v-i2v-r2v.safetensors)；README 示例和 Direct link 仍残留旧文件名，不能据此判断仓库缺少权重。

## 关键发现

### 版本与训练数字

| 版本 | 数据量 | Rank | Steps | 训练分辨率 | Learning rate | 有效 batch 1 下的等效 epochs |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| 初始公开版 | 176 clips | 16 | 5000 | medium | $10^{-4}$ | $5000/176\approx28.4$ |
| 当前替换版 | 176 clips | 32 | 1500 | high | 待补充 | $1500/176\approx8.52$ |

当前版的 **8.52 epochs 只是推算，不是作者披露值**。若 `global_step` 表示 optimizer steps，则对 $N=176$ 条物理样本：

$$
E_{\mathrm{equiv}}
=\frac{S\times B_{\mathrm{micro}}\times G_{\mathrm{acc}}\times W}{N},
$$

其中 $S$ 是 optimizer steps，$B_{\mathrm{micro}}$ 是单设备 micro-batch，$G_{\mathrm{acc}}$ 是梯度累积步数，$W$ 是 GPU 数。作者未公开后三项，因此只能在有效全局 batch 为 1 时得到约 **8.5 次样本遍历**；有效 batch 为 2/4 时则分别约为 **17.0/34.1 epochs**。

- 当前项目提供 19 组 same-prompt、same-seed 的 base/LoRA 对比，trigger 在两边都存在，以减少 trigger 本身造成的混淆；发布讨论称 16 个配置、两个数据版本共做了 100 次 same-seed A/B duels。
- 当前权重约 **242 MB**，遵循 [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)。
- 社区反馈并不一致：近景皮肤和牙齿细节被认为有改善，但也有人观察到肤色偏红/偏白、手指与快速运动变差、人物更容易看向镜头。它更像一个有明确视觉偏好的 realism adapter，而不是无条件提升 H3 的“质量补丁”。

## 我的提问

### Q1：能否拿到作者使用的 176 条原始数据？

不能。仓库只公开 README、LoRA 权重和对比视频；训练 clips、来源 URL、caption、筛选脚本与授权记录均未发布。早期模型卡的 stock-footage 表述只能说明大类来源，不能据此安全重建同一数据集。

### Q2：没有数据时，怎样构造一个可训练、可检查的近似版本？

下面是**复刻建议，不是作者披露的原始分布**。先做总计 176 条的可审计集合：50 条面部/说话近景、45 条中景日常劳动、35 条全身与运动、26 条两人或多人互动、20 条低照度/侧脸/遮挡/手部等困难样本。控制直视镜头、浅景深和统一 LUT 的比例，避免 LoRA 把“真实感”错误绑定成盯镜头、虚化背景或橙青色调。

素材优先考虑官方 **CC BY 4.0** 的 [SA-V](https://ai.meta.com/datasets/segment-anything-video/) 视频，再用 [PLM-Video-Human](https://huggingface.co/datasets/facebook/PLM-Video-Human) 的人物区域和动作 caption 做筛选与重标注。非商用研究可补充 [OpenHumanVid-Talking](https://huggingface.co/datasets/Haosonnn/OpenHumanVid-Talking) 或 [CelebV-Text](https://github.com/celebv-text/CelebV-Text)，但两者都有非商用限制。不要直接批量抓 Pexels：其当前条款明确禁止未经授权将服务/API 内容用于 ML 数据集或训练，参见 [Pexels 官方说明](https://help.pexels.com/hc/en-us/articles/900005880463-What-are-the-Terms-and-Conditions)。

每个 clip 应只含一个连续镜头，统一 24 FPS、自然速度、无水印和硬切；caption 至少描述景别、人物、动作/表情、环境、光线和镜头运动。若使用 [Inline Studio 的 H3 trainer](https://inlinestudio.art/lora-training/minimax-h3)，可从 2–4 秒视频开始，并优先使用满足 H3 `17n+5` 网格的 39/56/73/90 帧；该公开 trainer 的起始配置是 rank/alpha 16、batch 1、LR $10^{-4}$、500–1500 steps，但这不是 fal 原训练器的完整复现。

### Q3：最小可行训练与验收应该怎样做？

先用 30–50 条 clips 在 500 steps 做 smoke test，排除空 caption、LoRA 未挂载和数据偏色；随后用完整数据比较 rank 16/32、LR $5\times10^{-5}$/$10^{-4}$、不同 resolution bucket，并每 250 steps 存 checkpoint。验收固定 20–30 个 prompt 与 seed，对 base/LoRA 做盲测；单独检查皮肤、眼睛、手部、快速动作、多人互动、色偏和“总盯镜头”等失败模式。不能只看训练 loss 或少量近景宣传样例。

## 局限与疑问

- 数据不可得且来源授权未披露，无法逐样本复现或确认能否商用；基础模型许可证不能替代训练素材许可证。
- 当前版本未公开 batch size、GPU 数、gradient accumulation、optimizer、alpha、最终 learning rate、clip length 和 high-resolution bucket 的确切尺寸。
- 项目在发布当天从 rank 16/5000-step medium-resolution 权重替换为 rank 32/1500-step high-resolution 权重，Reddit 主帖仍保留旧的“winner”描述；复现时应以当前模型卡、提交历史和权重元数据为准。
- 19 组对比属于作者提供的定性证据，没有独立 benchmark、用户盲测统计或通用视频质量指标；same-seed 能控制变量，但不能排除 prompt 集偏向。
- “自然皮肤、film lighting、documentary motion”被一起训练在一个 trigger 中，外观、色彩和运动纠缠，难以单独调节；快速动作和手部物理可能因此退化。
- 当前权重按 T2V strategy、无 first-frame conditioning 训练，却宣称跨 T2V/I2V/R2V 使用；跨任务泛化需要分别验证，不能只根据文件名接受结论。

## 我的判断

这个项目最值得保存的不是某个神奇超参数，而是一个小数据 LoRA 的可行 recipe：**少量但强筛选的真人连续镜头、自然速度与统一 FPS、结构化 caption、高分辨率训练，以及严格的同 seed A/B 选型**。作者最终从 5000-step medium run 换成 1500-step high-resolution run，也说明对皮肤和细发这类高频真实感目标，输入信息量可能比单纯增加训练轮数更关键。

它适合作为数据与评测设计的起点，不适合作为完全可复现的基准。若自己训练，应先建立有来源和授权记录的 176 条数据清单，再运行小型 factorial sweep；不要在 learning rate、batch 和分辨率仍未知时追求一比一复制，也不要把约 8.5 epochs 当成已经证实的最优轮数。

## 下次只看这些

1. 当前权重是 **176 clips、rank 32、1500 steps、high-resolution bucket**；有效 batch 1 下约 **8.52 epochs**，但 batch/LR 等关键项未公开。
2. 原始数据没有发布；可从 SA-V + PLM-Video-Human 构造有授权、可审计的近似集合，Pexels 当前条款不允许未经授权用于 ML 训练。
3. 真正值得复用的是 **24 FPS 自然速度 + structured captions + high-res + same-seed A/B**，并重点监控肤色、手部、快速运动和盯镜头偏差。
