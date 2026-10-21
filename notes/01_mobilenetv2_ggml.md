# 01 · MobileNetV2 的 Graph Op 与 llama.cpp / ggml 的落地

> 学习目标：拿一个「小到能讲清、经典到面试官都懂」的 CNN，反向拆出「图 → op → kernel」这条链，并算清楚每个 op 是 compute-bound 还是 memory-bound。这条链就是方向二的全部基本功。

---

## 0. 一句话结论

**MobileNetV2 在 batch=1 时几乎全网络都是 memory-bound（算术强度只有个位数到几十 MAC/Byte，远低于现代 NPU/GPU 的平衡点），所以它的优化 90% 是「减少搬运 + 融合 + 换布局」，而不是「提高算力利用率」；而 ggml 里 conv 被统一 lower 成 `im2col + mul_mat`（GEMM），这对 pointwise 很友好、对 depthwise 极其浪费 —— 这就是为什么真正的后端（CUDA/HTP/Metal）都要给 depthwise 写 direct kernel。**

---

## 1. 为什么选 MobileNetV2 当练手样本

它同时具备了你需要练的所有图级模式，而且只有 ~50 个 conv：

| 模式 | MobileNetV2 里的体现 | 对应的通用问题 |
|---|---|---|
| 普通 3×3 conv | stem conv (s2) | im2col / direct / Winograd 的取舍 |
| depthwise 3×3 | 17 个 bottleneck 的核心 | 极低算术强度、layout 敏感 |
| pointwise 1×1 | 每个 bottleneck 的 expand 与 project | 等价于 GEMM，最适合量化 |
| 残差相加 | stride=1 的 bottleneck | 融合 add + activation |
| 激活 | ReLU6 | 算子融合、int8 饱和 |
| 全局池化 | GAP → 1×1×1280 | reduction op 的并行方式 |
| 全连接 | FC 1000 | 其实就是 GEMV，decode 时同款瓶颈 |

而且它是**端侧 CV 的 Hello World**：你在高通做 NPU 部署，MobileNet 系列一定在某个 demo 里出现过，面试时讲它比讲 ResNet 更有「我真的部署过」的味道。

---

## 2. 先把网络拆成 op 序列

MobileNetV2 1.0 / 224 的结构（`t`=expand ratio，`c`=输出通道，`n`=重复次数，`s`=第一个 block 的 stride）：

| Stage | t | c | n | s | 输出尺寸 |
|---|---|---|---|---|---|
| conv2d 3×3 | – | 32 | 1 | 2 | 112×112×32 |
| bneck | 1 | 16 | 1 | 1 | 112×112×16 |
| bneck | 6 | 24 | 2 | 2 | 56×56×24 |
| bneck | 6 | 32 | 3 | 2 | 28×28×32 |
| bneck | 6 | 64 | 4 | 2 | 14×14×64 |
| bneck | 6 | 96 | 3 | 1 | 14×14×96 |
| bneck | 6 | 160 | 3 | 2 | 7×7×160 |
| bneck | 6 | 320 | 1 | 1 | 7×7×320 |
| conv2d 1×1 | – | 1280 | 1 | 1 | 7×7×1280 |
| avgpool 7×7 | – | – | 1 | – | 1×1×1280 |
| fc | – | 1000 | 1 | – | 1000 |

**一个 stride=1 的 bottleneck（inverted residual）展开成 op 就是 6 个节点：**

```
x ──► conv1x1(expand, t·c) ──► ReLU6 ──► conv3x3 dw ──► ReLU6 ──► conv1x1(project, c) ──► add(x) 
```

**一个 stride=2 的 bottleneck**：没有残差，DW 的 stride=2 做下采样，**不要**在 1×1 卷积上做 stride（这是 MobileNetV2 和 V1 的关键区别，面试常问）。

**参数量与计算量（我按标准实现复算，224×224，输出 1000 类）**

| 项 | 数值 |
|---|---|
| 参数量 | ≈ 3.47 M |
| MACs | ≈ 3.0–3.5 亿（口径不同会差 10%~15%，论文写 300M） |
| pointwise 1×1 占 MACs | ≈ 3/4 |
| depthwise 占 MACs | ≈ 1/10 |
| depthwise 占**访存量** | 远高于 1/10（这就是矛盾所在） |

一句话记住：**MobileNetV2 用 depthwise 把 FLOPs 砍掉了 8~9 倍，但没有砍掉访存；它把问题从「算不动」变成了「搬不动」。**

---

## 3. 映射到 ggml：先搞清它的张量约定

ggml 是 llama.cpp 的计算图库。它和你熟悉的 PyTorch 有三处根本差异，**这三处决定了后面所有的性能结论**：

1. **`ne[0]` 是最内层（连续）维度**。PyTorch 的 `[N,C,H,W]` 在 ggml 里是 `ne = {W, H, C, N}`。所以「通道」是第 2 维，不是第 0 维。这本质上是 **NHWC** 的内存排布（channel 最内），只是索引写法反过来了。
2. **没有 stride/autograd**，只有 `ggml_tensor{type, ne[4], nb[4], data, op, src[]}`，图和内存是分开的（`ggml_context` 管 tensor 元数据，backend buffer 管 data，`ggml_gallocr` 做内存复用）。
3. **op 是「一个 op 一个 tensor」**，融合要么在构图时手工合并，要么在后端 kernel 里做；没有 PyTorch 那种自动的 fusion pass。

**MobileNetV2 在 ggml 里的 op 对照表**

| PyTorch | ggml API | ggml op | 备注 |
|---|---|---|---|
| `nn.Conv2d(k=3,s=2)` | `ggml_conv_2d(ctx, w, x, s0,s1,p0,p1,d0,d1)` | `IM2COL` + `MUL_MAT` | w: `[KW,KH,IC,OC]`, x: `[W,H,IC,N]` |
| `nn.Conv2d(groups=C)` (DW) | `ggml_conv_2d_dw(...)` | 历史上也是 `IM2COL`+`MUL_MAT` | ggml.h 的注释原话是 `// depthwise (via im2col and mul_mat)` |
| `nn.Conv2d(k=1)` | 直接 `ggml_mul_mat` + `reshape` | `MUL_MAT` | 1×1 conv 就是 GEMM，别走 conv 路径 |
| `F.relu6` | `ggml_clamp(x, 0, 6)` | `CLAMP` | 有些后端会把它融进 conv 的 epilogue |
| `x + identity` | `ggml_add` | `ADD` | 需要 shape/broadcast 合法 |
| `F.adaptive_avg_pool2d` | `ggml_pool_2d(AVG, k=H×W)` 或 `ggml_mean` | `POOL_2D` / `MEAN` | reduction，注意别退化成多次 pass |
| `nn.Linear` | `ggml_mul_mat` | `MUL_MAT` | 权重 `[in, out]` |

> 版本提示：ggml 的 op 集合和签名在演进（新增过 `conv_3d`、`conv_2d_dw_direct`，也调整过 im2col 的参数）。上面这行「depthwise 走 im2col+matmul」是 ggml.h 里长期存在的注释；**具体以你本地 `ggml/include/ggml.h` 的注释和 `ggml/src/ggml.c` 里 `ggml_conv_2d()` 的实现为准**，5 分钟就能确认。

---

## 4. 核心：conv 在 ggml 里到底被 lower 成了什么

### 4.1 公式

`ggml_conv_2d(a=weight, b=input)` 的语义是：

```
im2col = ggml_im2col(ctx, a, b, s0, s1, p0, p1, d0, d1, is_2D=true, dst_type)
      # 形状 [IC·KH·KW, OW, OH, N] —— 把每个输出位置需要的输入 patch 拉平成一列
res   = ggml_mul_mat(ctx,
            reshape(im2col, IC·KH·KW, OW·OH·N),   # [K, M]
            reshape(a,      IC·KH·KW, OC))        # [K, N_out]
      # -> [OC, OW·OH·N]，再 reshape 回 [OW, OH, OC, N]
```

也就是说：**卷积 = 一次 im2col 重排 + 一次大 GEMM**。这也是几乎所有框架（Caffe 时代至今）的默认做法。

### 4.2 代价：im2col 的中间张量有多大

以 112×112 的特征图为例（fp32）：

| 层 | 输入大小 | im2col 张量 | 膨胀 |
|---|---|---|---|
| 普通 3×3 conv，IC=3 | 112·112·3·4 ≈ 0.6 MB | 112·112·(3·9)·4 ≈ 1.35 MB | 2.25× |
| **depthwise 3×3，C=96** | 112·112·96·4 ≈ 4.8 MB | 112·112·(96·9)·4 ≈ **43 MB** | **9×** |

结论：**对普通 conv，im2col 只是「多一次写 + 少一点局部性」；对 depthwise，它是灾难性的 —— 为了做 9 次乘加，先把激活复制 9 份。**

### 4.3 所以真实后端会怎么做 depthwise

| 方案 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| im2col + GEMM | 上面那条路 | 复用现成 GEMM，零开发量 | 9× 访存 + 巨大中间 buffer |
| direct（NCHW） | 每个输出 channel 独立算 3×3 | 无 im2col | 内层只有 3 个元素，难以向量化 |
| **direct（NHWC / channel-last）** | 一次读 8/16 个 channel，用 SIMD/HVX 向量化，滑窗复用寄存器 | 访存 1×，向量友好 | 需要专门的 kernel |
| 转置 + 分组卷积技巧 | 把 DW 变成 grouped conv 调库 | 开发量小 | 仍受分组数限制 |

**这就是为什么端侧 NPU/GPU 上「layout 决定生死」**：depthwise 只有在 channel-last 下才能被向量化，而 channel-last 又要求整个图不出现 transpose（transpose 会引入一次全量读写）。

---

## 5. 定量分析：每个 op 的 arithmetic intensity

定义：`AI = MACs / Bytes`（Bytes 按「读输入 + 读权重 + 写输出」算，fp32）。硬件平衡点 = 峰值算力 / 峰值带宽。取一个典型端侧 NPU：**int8 40 TOPS、LPDDR5 带宽 60 GB/s → 平衡点 ≈ 667 ops/Byte ≈ 333 MAC/Byte**（fp32 存储时还要再除以 4，实际更苛刻）。

| 层 | MACs | 访存（fp32） | AI (MAC/B) | 结论 |
|---|---|---|---|---|
| stem 3×3 s2 (3→32, 224²) | 43.4 M | 0.6 + 0.4 MB ≈ 1.0 MB | ≈ 43 | memory-bound |
| pw 1×1 96→24 @56² | 7.2 M | 0.30 + 0.075 MB | ≈ 19 | memory-bound |
| **dw 3×3 96 @112²** | 10.8 M | 4.8 + 4.8 MB | **≈ 1.1** | **严重 memory-bound** |
| fc 1280→1000 | 1.28 M | 5.1 MB(权重) | ≈ 0.25 | GEMV，纯带宽 |

**把这张表拍在面试桌上，你就赢了一半**：它解释了为什么「TOPS 翻倍」对 MobileNet 几乎没用，为什么端侧优化第一优先级是「把特征图留在 VTCM/片上」，第二是「融合掉 elementwise」，第三才是「换更快的 GEMM」。

---

## 6. 图级优化：llama.cpp/ggml 里真实存在的那些手段

按「收益/成本」排序，逐条对上 ggml 的机制：

1. **后端调度与拷贝最小化**（`ggml_backend_sched`）：把整张图按后端切段，段之间只在必要处拷贝。跨后端一次拷贝可能就是几 MB。
2. **内存 arena 复用**（`ggml_gallocr`）：按生命周期复用 buffer。CNN 里 im2col 张量是最大的那块，复用能省几百 MB。
3. **权重常驻 + 量化**：CNN 权重小（3.5M），量化收益主要在带宽和 cache；但对 LLM 是决定性的。
4. **op 融合**：ggml 里没有自动 pass，靠两条路 —— ① 构图时手工写成 `GEMM+bias+act` 的专用 op；② 后端 kernel 的 epilogue（CUDA 的 `mul_mat` 变体、HTP 的 op 融合）。
5. **多线程切分**（`ggml-cpu` 的 `n_threads`）：按输出行/列切分；对 memory-bound op，线程数超过带宽饱和点后收益归零。
6. **布局不变式**：一个好的图应该「从输入到输出只出现一种 layout」。ggml 的 `ne[0]=W` 让 conv 天然是 channel-last；反而是你把 PyTorch 权重转过来时要小心 permute。

**llama.cpp 里的真实对应物**：它的 vision tower（LLaVA 的 CLIP、Qwen-VL 的 patch embed）走的就是 `conv2d`/`im2col` 这条路；`ggml/src/ggml-hexagon/` 是高通 HTP 后端（有自己的 HTP op node 与 repack），正好可以对照看「同一张图在 CPU 与 HTP 上被切成什么样」。

---

## 7. 量化在这个图里的位置

- **1×1 conv = GEMM = 量化收益最大**：直接调 int8 GEMM（`dp4a`/HTP 的 dot），权重 per-channel int8、激活 per-tensor。
- **depthwise 量化最敏感**：每个 channel 只有一个 3×3 filter，per-channel scale 数量等于通道数（可接受），但**激活的动态范围在 channel 间差异极大**，per-tensor 激活量化容易掉点 → 实践中常用 per-channel 或 mixed precision 保住它。
- **ReLU6 与 int8 饱和**：6/scale 之后要能被 int8 表示，否则 clamp 值本身就被量化误差吃掉。
- **残差 add 的两个输入量化 scale 不同** → 必须 requantize（一次额外的 int32→int8 变换），这是端侧 CNN 的固定开销，HTP 上通常靠「同 scale 约束」在编译期消掉。

---

## 8. 把视角切到 HTP（你的主场，面试差异化就在这里）

| 通用问题 | 在高通 HTP 上的具体形态 |
|---|---|
| 片上内存不够 | VTCM 容量有限（百 KB ~ 数 MB 级），必须 tiling；一个 112×112×96 的 fp16 特征图就是 2.4 MB，放不下 → 按 channel 或按行切 |
| depthwise 怎么快 | HVX 向量指令 + channel 方向向量化；关键是 layout 必须是 NHWC，且 channel 数是 32/64 的倍数 |
| 带宽 | 每一层都要 DDR↔VTCM 搬一次；memory-bound 层占了大头，profile 里先看 DMA 时间 |
| 量化约束 | per-channel、对称量化为主；某些 op 只吃特定位宽；混合精度要按层配 |
| ReLU6 / clamp | 尽量融进 conv 的 epilogue，否则多一次全量读写 |
| 图切分 | 不支持的 op 会 fallback 到 CPU，一次 fallback = 一次同步 + 拷贝，可能比 op 本身还贵 |

**面试时这样讲**（60 秒版）：
> 「我在高通把 LLM 部署到 HTP 上，但 HTP 的优化逻辑和 CNN 是同一套：VTCM 太小所以要 tiling，带宽比算力先饱和，所以第一步永远是看 DMA 而不是看 MAC 利用率。我拿 MobileNetV2 做过对照实验：它的 depthwise 层算术强度只有 1 MAC/Byte 量级，在 40 TOPS 的 NPU 上跑不满 5% 的算力，瓶颈全在搬运 —— 这也解释了为什么端侧第一优化是融合和布局，而不是换更大的 MAC array。」

---

## 9. 面试问答（背下来）

**Q1：im2col 的代价是什么？为什么 depthwise 不适合用 im2col？**
im2col 把输入按 kernel 窗口复制成 `[K, M]` 矩阵，访存膨胀 K² 倍（3×3 就是 9×），并且要一块 `IC·KH·KW × OW·OH` 的中间 buffer。普通 conv 的 IC·K² 大、GEMM 复用高，这笔开销可以摊薄；depthwise 每个 channel 只有 9 个乘加，膨胀 9 倍后几乎全是搬运，所以必须 direct。

**Q2：为什么 MobileNetV2 在 NPU 上算力利用率很低？**
因为它把 FLOPs 换成了访存。每一层的 arithmetic intensity 都远低于硬件的 ops/byte 平衡点，算力单元大部分时间在等数据。要提升只能从「减少字节数」入手：融合、量化、把中间结果留在片上、channel-last 避免 transpose。

**Q3：stride 放在 depthwise 上还是 1×1 上，为什么？**
放在 depthwise 上。1×1 conv 做 stride 会直接丢掉空间信息且无法做「先降采样再压缩」的组合；MobileNetV2 的设计是 DW 负责下采样、PW 负责升/降 channel，这样 DW 的输入通道数最小，正好省访存。

**Q4：inverted residual 为什么用 linear bottleneck（最后一层不加激活）？**
低维流形上做 ReLU 会破坏信息（论文的实证结论：去掉最后的激活，精度更好）。从量化角度看还有额外好处：输出没有 clamp，动态范围更连续，per-tensor 量化误差更小。

**Q5：一个 int8 的 3×3 conv 怎么实现？**
权重 per-channel int8、激活 per-tensor int8；内层用 `dp4a`/`sdot` 做 4 路 int8 点积累加到 int32；结束后 `int32 * (sa*sw) → requantize → int8`，bias 提前折进 int32。注意 zero-point 的修正项 `-zp_a·Σw` 要预计算。

**Q6：conv 融合顺序上有什么坑？**
`conv+bn` 在推理时可以完全折进权重（`w' = w·γ/σ`，`b' = β - μγ/σ`），零成本；`conv+relu` 只能在 epilogue 融；跨层的 `conv+add+relu` 需要三个输入都在片上，tiling 策略会变复杂。融合不是越多越好，受片上内存限制。

**Q7：给定一张图，你怎么判断该先优化哪个 op？**
先 profile 拿每个 op 的耗时和 DMA 字节；算 AI 分类；对 memory-bound 的看「字节数是否可减」（融合/量化/layout），对 compute-bound 的看「MAC 利用率」（tiling/向量化/是否 fallback）。永远先做占用时间最多的那个，别做「看起来最酷」的。

**Q8：llama.cpp 的 conv 实现和 cuDNN 差在哪？**
ggml 是「正确性优先、后端自治」：图层面统一 lower 成 im2col+GEMM，性能靠各后端自己的 `mul_mat` / `im2col` kernel。cuDNN 有隐式 GEMM、Winograd、FFT、自动 tuning。差在「手工调优的深度」，不在算法框架。

---

## 10. 动手清单（照着做，一周内完成）

- [ ] clone llama.cpp，`grep -n "ggml_conv_2d" ggml/src/ggml.c ggml/include/ggml.h`，把 lowering 的 20 行代码读明白，抄进笔记。
- [ ] 用 ggml 的 C API 手搭 MobileNetV2 的**前 3 个 block**（stem + 两个 bneck），输入随机数，和 PyTorch 对一遍 max abs error（应 < 1e-5）。
- [ ] 用 `ggml_graph_print()` dump 图，数一数「一个 bottleneck 生成了几个 op」，和你的预期对比。
- [ ] 关掉 `GGML_OPENMP` 单线程跑，再开 8 线程跑，记录加速比，解释为什么 memory-bound 层加速比 < 线性。
- [ ] 写一个 30 行的 Python 脚本，把每层的 AI 算出来画成表（就是第 5 节那张），贴进 `整理.docx`。
- [ ] 加分：把 `ggml_conv_2d_dw` 换成一个手写的 direct depthwise（NCHW 也行），对比时间，解释差异。

---

## 11. 参考资料

- MobileNetV2 论文（Sandler et al., CVPR 2018）—— 重点看 §4「Linear Bottlenecks」和 §5「Inverted Residuals」的动机，不是看精度表。
- ggml 源码：`ggml/include/ggml.h`（op 与 API 注释）、`ggml/src/ggml.c`（`ggml_conv_2d` / `ggml_im2col` / `ggml_conv_2d_dw`）、`ggml/src/ggml-cpu/`（im2col 与 GEMM kernel）。
- llama.cpp 的 `ggml/src/ggml-hexagon/`（HTP 后端）+ 它的 `htp-opnode.h` / `htp-repack.cpp`，看「一个 op 在 HTP 上如何被 repack 成硬件友好的布局」。
- CMU 15-418 的 roofline 讲义（你上过，两页纸捡回来最快）。
