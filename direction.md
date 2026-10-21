# 方向地图 Direction Map

> 使用者：3 年经验，Qualcomm QAIRT，把 Qwen3.5 这类模型部署到 HTP 上。CMU MS（并行计算 / 嵌入式），本科 CE。
> 这张地图的唯一目的：让每一次学习都落到「面试能讲、简历能写、有数字能证明」。
> 维护方式：方向 → 子类 → 每个子类必须有【必答题】【我的弹药】【缺口】【最小交付物】四栏，缺一栏就是没学完。

---

## 0. 先定人设（这一节比下面所有技术内容都重要）

**一句话版本**
> 我在高通做端侧 LLM 的部署与量化：把一个 Transformer 图从 PyTorch/ONNX 一路拆到 HTP 上的 op，在精度和时延之间做权衡，能定位到底是哪个 op、哪种量化策略让 W4A16 掉点，或者让时延卡在 memory-bound 上。

**三个可卖的点（按市场上稀缺度排序）**
1. **端侧 / XPU 上的 LLM 落地**：QAIRT(QNN/SNPE) → HTP、VTCM、HVX、op 约束、mixed precision。市场上会写 PyTorch 的多，真把模型压到手机 NPU 上跑出数字的少。
2. **量化全链路**：校准 → 敏感层分析 → 混合精度 → 高级 PTQ（AWQ/GPTQ/SmoothQuant/旋转类）→ QAT 上手的完整闭环。
3. **系统底子**：并行计算 + 嵌入式 + 读得懂 kernel 和访存。这让你能从「部署工程师」往「算子 / 推理框架」平移。

**不要这样卖自己**
- 不要卖成「通用 AI Infra」：vLLM 那套你没有生产经验，硬碰会被 CMU/大厂 infra 组的人按在地上。
- 不要卖成「算法工程师」：没有顶会、没有训练大模型的经验。
- 正确的组合是：**主线 = 端侧推理 / 量化；副线 = 算子优化；补充 = 推理系统常识；差异化 = 应用侧一个能跑的项目。**

---

## 1. 方向一：模型量化（主战场，投入 ~50%）

### 为什么值得
- 是所有「端侧 / 推理」岗位的硬通货：手机厂（高通/联发科/紫光/OPPO/vivo）、车厂与 Tier1（地平线/黑芝麻/Mobileye）、芯片公司（寒武纪/壁仞/燧原）、大厂推理组（字节 Seed、阿里 PAI、腾讯混元）都在招。
- 你的经历天然对口：QAIRT 上做过 PTQ / QAT，知道真实硬件的位宽、layout、per-channel 约束 —— 这是看论文的人没有的。
- 面试可考、可推导、可写代码，最容易做出「深度」。

### 子类（学到什么程度）

| # | 子类 | 到什么程度 | 面试权重 |
|---|---|---|---|
| 1.1 | 数值格式与硬件映射 | int8/int4/fp8/mxfp4/nvfp4、对称 vs 非对称、per-tensor/per-channel/per-group(32/64/128)、scale/zero-point、HTP 的位宽与对齐约束 | ★★★★★ |
| 1.2 | PTQ 基础链路 | 校准集选择、MinMax/Percentile/MSE/KL、bias correction、逐层敏感度、混合精度搜索 | ★★★★★ |
| 1.3 | 高级 PTQ | GPTQ(Hessian/OBS)、AWQ、SmoothQuant、AdaRound/BRECQ、Hadamard/QuaRot/SpinQuant、PrefixQuant、KV cache 量化 | ★★★★☆ |
| 1.4 | QAT | STE 的梯度偏差、LSQ/DoReFa、fake-quant 放置、蒸馏、QAT 什么时候真的比 PTQ 强 | ★★★☆☆ |
| 1.5 | 评估与调优方法论 | ppl vs 任务指标、outlier 分析、掉点归因（哪一层/哪种 pattern）、长尾与安全对齐的影响 | ★★★★☆ |
| 1.6 | 部署侧特化 | W4A16 vs W8A8、weight-only 的解压开销（compute vs bandwidth）、KV 量化的额外读写、group size 与 kernel 的对齐 | ★★★★★ |

### 必须能答的题
1. 为什么 LLM 量化比 CNN 难？（activation outlier、通道动态范围差异巨大、attention/KV 的长尾）
2. 推导 SmoothQuant：`Y = (X·diag(s)⁻¹)(diag(s)·W)`，s 怎么选，为什么能把难度从 activation 搬到 weight。
3. AWQ 的核心假设（按激活幅度挑 1% salient channel，等价于 per-channel scaling）与它为什么不用回归。
4. GPTQ 的误差补偿：OBS 框架、Hessian `H = 2XXᵀ`、Cholesky 分解、为什么按列更新还要 block。
5. AdaRound：为什么 round-to-nearest 不是最优（用二阶展开给出 ΔW 的影响），可学习变量是什么。
6. 旋转类方法（Hadamard / QuaRot / SpinQuant）：为什么正交变换不改变 L2 范数、为什么能压 outlier、在真实硬件上代价是什么（要融合进相邻 linear）。
7. QAT 里 STE 的梯度是错的，为什么还能 work；LSQ 怎么把 step size 变成可学习参数。
8. 给定一个模型和硬件，你怎么决定 W4A16 / W8A8 / 混合精度？给决策树。
9. per-group 量化 group size 变小，精度↑但 kernel 效率↓，你怎么选？
10. 你怎么证明「你的量化方案好」？（数字：ppl、任务指标、端侧 TTFT/TPOT、内存）

### 我的弹药 vs 缺口
- 弹药：QAIRT 上的真实 PTQ/QAT 流程；HTP 的量化约束与混合精度；掉点排查经验。
- 缺口：① 上述算法的**数学推导**（现在多半是「知道名字」）；② 自己从零写一版 AWQ/GPTQ；③ 公开的数字对比表。
- 最小交付物（2 周）：同一个 Qwen3.5 小模型（0.6B/1.7B 级别），跑 RTN / GPTQ / AWQ 三种量化 + 一个旋转类方法，产出「ppl + 端侧时延 + 内存」三列表，并写一页《我的量化决策树》。

---

## 2. 方向二：NPU / GPU 算子与 kernel 优化（第二战场，投入 ~25%）

### 为什么值得
- 这是「部署工程师」往「推理/算子工程师」平移的唯一通道，薪资天花板明显更高。
- 你已经有 HTP 的硬件直觉（VTCM、HVX、DDR 带宽），缺的是「自己写 kernel 并算清楚为什么快」。
- llama.cpp / ggml 是最低成本的学习载体：图 API 简单，多后端（CPU/CUDA/Metal/Vulkan/**Hexagon**），可以对照着看同一张图在不同后端上怎么落地。

### 子类

| # | 子类 | 到什么程度 | 面试权重 |
|---|---|---|---|
| 2.1 | 计算图与 lowering | ggml graph（tensor/op/graph/allocator）、ONNX→后端、op fusion、layout(NCHW/NHWC) 的代价 | ★★★★☆ |
| 2.2 | GEMM 与 tiling | roofline、arithmetic intensity、shared memory/VTCM 分块、double buffering、寄存器分块 | ★★★★★ |
| 2.3 | Memory-bound op | DW conv、elementwise、softmax、layernorm、RMSNorm：为什么慢、怎么算带宽上限 | ★★★★★ |
| 2.4 | 后端差异 | CPU(SIMD/多线程)、CUDA、HVX/HTP、Metal/Vulkan：同一 op 的不同写法 | ★★★★☆ |
| 2.5 | 量化 kernel | int8 dot(dp4a/dot4)、W4A16 的 dequant-in-register、unpack 的指令开销 | ★★★★★ |
| 2.6 | 图级优化 | 常量折叠、prefill vs decode 的瓶颈切换、arena 内存复用、多后端调度与拷贝 | ★★★★☆ |
| 2.7 | Profiling | nsys/ncu、llama-bench、ggml graph dump、HTP 侧的 profile/op 时间线 | ★★★★☆ |

### 必须能答的题
1. 给你一个 op，怎么判断它 memory-bound 还是 compute-bound？现场算 arithmetic intensity 并和硬件的 ops/byte 比。
2. 为什么 decode 阶段是 memory-bound 而 prefill 是 compute-bound？用 batch=1、weight 只读一次算给面试官看。
3. DW conv 为什么在 GPU 上效率低？（每 channel 只有 1 个 filter、无法用 tensor core、访存不连续）
4. im2col + GEMM 的代价是什么？为什么 depthwise 用 im2col 特别亏（激活膨胀 K² 倍）？
5. int8 GEMM 怎么用 dp4a 实现？累加溢出怎么办？为什么需要 zero-point 修正项？
6. W4A16 的 kernel 里，unpack 和 dequant 占了多少时间？怎么用 `lop3`/移位把它摊薄？
7. conv + bias + relu 怎么融合？融合在图上做还是在 kernel 里做，各有什么限制？
8. HTP 上 VTCM 的作用是什么？为什么要 tiling + double buffer？和 GPU 的 shared memory 有什么异同？
9. 一个 RMSNorm 在 HTP 上要跑几个 pass？能不能一个 pass 做完？
10. 你怎么把 300M MACs 的 MobileNetV2 从 40ms 优化到 10ms？说出你的排查顺序。

### 我的弹药 vs 缺口
- 弹药：HTP 架构与 VTCM/HVX 直觉、QAIRT 的 op 约束、会读 profile。
- 缺口：① 没手写过 CUDA/HVX/NEON kernel；② 没做过定量的 roofline 分析；③ ggml 源码只读过皮毛。
- 最小交付物（2 周）：① 用 ggml 手搭 MobileNetV2 前向并跑通，dump 出 op 图；② 写 3 个 CUDA kernel（depthwise、softmax、RMSNorm），和 PyTorch 对比并给出 roofline 解释；③ 一页《conv 在 ggml 里到底被 lower 成了什么》。

---

## 3. 方向三：通用 AI Infra / 模型架构优化（补齐常识，投入 ~15%）

### 为什么值得
- 不是你的主战场，但**面试官默认你应该懂**。答不上「PagedAttention 解决什么问题」会显得不像做推理的。
- 另外，模型架构的演进（linear attention、稀疏注意力、KV 压缩）直接决定你明天要部署什么，属于「必须跟」的部分。

### 子类

| # | 子类 | 到什么程度 |
|---|---|---|
| 3.1 | 推理引擎基础 | PagedAttention、continuous batching、prefix caching、chunked prefill、speculative decoding |
| 3.2 | 调度与内存 | KV cache 管理/swap/抢占、TP/PP/EP、MoE 的 all-to-all |
| 3.3 | 高效 attention 架构 | MLA、Gated DeltaNet / linear attention、DSA 类稀疏注意力、混合层比例（如 3:1） |
| 3.4 | KV 压缩与投机 | KV 量化、NativeKV/SPD 这类「原生稀疏/压缩」思路、EAGLE 系列 draft 结构 |
| 3.5 | 端云协同 | 端侧小模型 + 云端大模型、端侧 draft + 云端 verify、路由 |

### 必须能答的题
1. PagedAttention 解决什么问题？block size 的权衡是什么？
2. continuous batching 怎么处理 prefill 和 decode 混跑？chunked prefill 的动机？
3. KV cache 的内存怎么算？给一个 7B/32 层/32 头/128 head_dim、4k 上下文的数字。
4. MLA 相比 MHA/GQA 省的是什么？为什么它需要「吸收」进 weight？
5. linear attention（GDN/Mamba2）的复杂度优势在哪、代价是什么？（固定大小的 recurrent state）
6. 为什么现在流行「3:1 混合」而不是全 linear？
7. EAGLE 的 draft 模型长什么样？为什么在 feature 层做 draft 比在 token 层好？
8. 投机解码在端侧有意义吗？（batch=1、memory-bound 的场景下，收益来自哪里）
9. 一个 397B-A17B 的 MoE，在 8 卡上你会怎么切？
10. 端侧部署时，你怎么决定「模型放端上还是放云上」？

### 我的弹药 vs 缺口
- 弹药：端侧部署视角、真实的内存/时延约束感。
- 缺口：**没跑过 vLLM**、没读过 PagedAttention/Orca 原文、没做过调度。
- 最小交付物（1 周）：本地跑通 vLLM，做一份「不同 batch/长度下的 TTFT/TPOT/吞吐」报告；写一页《prefill 与 decode 的 roofline 差异》。

---

## 4. 方向四：AI 应用（差异化，投入 ~10%）

### 为什么值得
- 不指望它当主技能，但它决定「你能不能把技术讲成一个产品故事」，也是转机器人 / 具身智能的入口。
- 你有 RAG、传统 CV、聚类项目的底子，捡起来成本低。

### 子类

| # | 子类 | 到什么程度 |
|---|---|---|
| 4.1 | RAG / Agentic RAG | chunking、embedding、hybrid retrieval、rerank、query rewrite、tool use、评测（召回率/忠实度） |
| 4.2 | 传统 CV 与端侧 CV | 检测/分割/跟踪、TFLite/QNN 部署、后处理开销 |
| 4.3 | 机器人与 VLA | VLA 模型结构、action head、推理时延预算、sim2real |
| 4.4 | VLM / world model | 视觉 token 数控制、patch merge、视频时序建模 |

### 必须能答的题
1. RAG 的三个失败模式分别在哪一层？你怎么定位和修？
2. Agentic RAG 和普通 RAG 的本质区别？（检索变成可决策的动作）
3. 端侧 VLM 的 token 数怎么压？（spatial merge、动态分辨率、early exit）
4. VLA 的推理链路时延怎么拆？哪些能放到 NPU 上？
5. 你为什么对机器人感兴趣？—— 这题必须有一个「做过的事」来支撑，不能只有热情。

### 最小交付物
- 一个带评测指标的小 RAG（哪怕只有 50 条测试集，要有 recall@k 和忠实度）。
- 一个端侧 VLM 或机器人 demo（能录视频）。

---

## 5. 岗位画像与匹配度（决定投哪里）

| 岗位类型 | 典型公司 | 考什么 | 我的匹配度 | 策略 |
|---|---|---|---|---|
| 端侧推理 / 部署 | 高通、联发科、OPPO/vivo/小米、地平线、黑芝麻 | 量化、op 约束、内存带宽、profile | ★★★★★ | 主投，讲 QAIRT 的真实 case |
| 量化 / 压缩研究工程 | 字节 Seed、阿里、上海 AI Lab、微软 | 算法推导 + 代码实现 + 实验设计 | ★★★☆☆ | 补推导和复现 |
| 算子 / kernel | NVIDIA、寒武纪、壁仞、Meta、字节 | CUDA/汇编、roofline、手写 kernel | ★★☆☆☆ | 6 个月后再冲，先补 CUDA |
| 推理框架 | vLLM/SGLang 团队、腾讯、蚂蚁 | 调度、KV 管理、分布式 | ★★☆☆☆ | 只求「能聊」，不求「能面」 |
| 应用 / 具身 | 机器人公司、创业公司 | 项目完整度、产品感 | ★★★☆☆ | 用 demo 讲故事 |

---

## 6. 8 周节奏（可改，但别同时开四条线）

| 周 | 主线（量化） | 副线（算子） | 只做一件事 |
|---|---|---|---|
| 1 | 数值格式 + PTQ 链路 | MobileNetV2 的 ggml 图 | 搭出 MobileNetV2 前向 |
| 2 | AWQ / GPTQ 推导 + 复现 | im2col / GEMM / roofline | 量化三件套对比表 |
| 3 | SmoothQuant / 旋转类 | DW conv 与 memory-bound | 手写 depthwise kernel |
| 4 | 敏感层 + 混合精度 | 量化 kernel（dp4a / W4A16） | 决策树一页纸 |
| 5 | QAT + KV 量化 | Qwen3.5 的图拆解 | Qwen3.5 部署笔记 |
| 6 | 评估方法论 | RMSNorm/softmax kernel | 三个 kernel 的 roofline |
| 7 | infra 常识突击 | vLLM 跑通 | TTFT/TPOT 报告 |
| 8 | 简历 + 项目复述 + 模拟面试 | — | 三分钟讲清每个项目 |

---

## 7. 每个知识点的产出模板（一页纸，写不出就别说过学会了）

```
1. 一句话结论（面试官只记得住这一句）
2. 数学 / 伪代码（能白板推）
3. 至少 3 个数字（形状、带宽、时延、精度）
4. 三个坑（真实部署里会踩的）
5. 60 秒口述版（对着镜子说一遍）
6. 和我工作怎么连（一句「我在高通的时候…」）
7. 自测题 3 道（不看答案能答出来吗）
```

## 8. 简历与复述

- 每个项目：**背景 → 我的动作 → 数字结果 → 学到了什么**。没有数字的项目不要写。
- 准备三个「深挖问题」的答案：① 你调过最难的一个精度问题是什么；② 你怎么定位一个 op 的性能瓶颈；③ 如果让你重做一遍，你会改什么。
- 反问清单：团队现在部署的模型多大、卡在算力还是带宽、量化是自研还是用现成工具链、有没有 kernel 自研。

## 9. 资源（只列真的会看的）

- 量化：GPTQ / AWQ / SmoothQuant / QuaRot / SpinQuant / AdaRound 原文；`llm-compressor`、`AutoAWQ` 源码。
- 算子：ggml（`ggml/src/ggml.c` 的 op 定义、`ggml-cpu` 的 kernel、`ggml-hexagon`）；CUDA C Programming Guide 的 roofline 章节。
- 架构：PagedAttention、Orca、EAGLE、Gated DeltaNet(2412.06464)、Qwen3.5 技术报告。
- 系统：CMU 15-418 / 15-740 的 notes（你上过，捡起来最快）。
