# 04 · 量化基础与高通 QAIRT / AIMET 的工程流水线

> 学习目标：能从一个 FP32 模型出发，说清「每一步在改什么、为什么这么改、代价是什么」，并且能在没有 GPU 的笔记本上用 numpy 复现文中的关键结论。
>
> 三个来源：① Nagel et al., *A White Paper on Neural Network Quantization* (arXiv:2106.08295)，本笔记的骨架；② 高通 AIMET 文档（2.35.x）里的 PTQ 技术与 LLM recipes，即 QAIRT 生态实际在跑的算法；③ 本机实测（`tools/quant_basics.py`，纯 numpy，可重跑）。

---

## 0. 一句话结论

**量化只有一个核心矛盾：在固定的整数网格上，把有限的"表示预算"分配给"最影响输出"的那部分数值。所有技术都在这三件事上做文章 —— 选阈值（校准）、改分布（CLE / 旋转 / scaling）、改舍入（AdaRound / GPTQ）。** 而工程上能不能落地，取决于第二个问题：**这套数学在目标硬件上要几个 cycle、多少字节。**

---

## 1. 量化在做什么：从硬件说起

### 1.1 为什么要量化

一次 MAC 的代价：fp32 乘法器面积/能耗 ≈ int8 的 4~8 倍；搬一个数：4 字节 → 1 字节。端侧是**功耗和带宽受限**，不是算力受限（见 note 03），所以量化几乎是唯一"三赢"（内存、带宽、算力）的手段。

### 1.2 三个参数

```
x̂ = s · (clamp(⌊x/s⌉ + z, 0, 2^b − 1) − z)

s : scale（步长，浮点）
z : zero-point（整数，保证 0 被精确表示）
b : bit-width
```

`z` 存在的意义：让 `0` 落在网格上。否则 ReLU 的零点、padding 的零都会带上量化误差 —— 对 CNN 是灾难。

### 1.3 定点推理的完整链路（面试要能画出来）

```
        int8 W          int8 x
          │               │
          └──────┬────────┘
                 ▼
        int32 累加器（bias 也折进 int32）
                 ▼
   requantize: int32 → ×(s_w·s_x/s_out) → +z_out → clamp → int8
                 ▼
             下一层
```

三个必须记住的点：

1. **累加器必须比输入宽**（典型 int32），否则溢出。`K=4096` 的 int8 点积，最坏情况 `127×127×4096 ≈ 6.6e7`，int16 直接爆，int32 安全。
2. **bias 不量化到 int8**，它用 int32 存、scale 是 `s_w·s_x`。这是白皮书第 2.1 节的明确约定。
3. **requantize 是每层都要付的固定成本**：一次 int32→int8 的乘加+饱和，在 NPU 上通常融进前一层的 epilogue。

---

## 2. 误差只有两种：裁剪 vs 舍入

设步长 `s`：

- **舍入误差**：`|δ| ≤ s/2`，均匀分布，方差 `s²/12`。
- **裁剪误差**：超过 `[q_min, q_max]` 的值被 hard clip，误差可能是几十个 `s`。

**这两个误差是跷跷板**：`s` 由 `(q_max − q_min)/(2^b − 1)` 决定，把 range 放宽 → `s` 变大 → 舍入误差线性变差；range 收紧 → 尾部被裁 → 裁剪误差暴涨。

### 本机实测（Student-t 重尾分布，std=1，max=288.7）

| 保留分位数 | clip 值 | int8 MSE | int4 MSE |
|---|---|---|---|
| 90% | 1.34 | 3.02e-01 | 2.91e-01 |
| 95% | 1.81 | 2.38e-01 | 2.32e-01 |
| 99% | 3.33 | 1.45e-01 | 1.57e-01 |
| 99.5% | 4.24 | 1.20e-01 | **1.45e-01** |
| 99.9% | 7.44 | 7.83e-02 | 1.67e-01 |
| 99.99% | 16.3 | 4.84e-02 | 3.62e-01 |
| 99.999% | 32.0 | **3.98e-02** | 6.04e-01 |
| 100%（MinMax） | 288.7 | 3.06e-01 | 9.29e-01 |

**两个可以直接讲的结论：**

1. **MinMax 在重尾分布上是最差的选择**（int8 下 MSE 比最优差 7.7 倍）。这就是为什么"校准"是一个独立的技术环节，而不是取个 max 就完事。
2. **位宽越低，越要激进地裁**：int8 的最优 clip 只有 max 的 **12.5%**（≈99.999 分位），int4 只有 **1.7%**（≈99.5 分位）。因为 4 bit 只有 16 个格子，把一个 outlier 纳进来的代价（步长变大）远大于裁掉它的代价。

> 反直觉但重要：**int4 的难点不是"精度不够"，而是"该丢谁"**。谁该丢，就是校准算法要回答的问题。

---

## 3. 四种基本选择

### 3.1 对称 vs 非对称：为什么工业界是「权重对称 + 激活非对称」

白皮书把非对称量化的乘积展开成四项：

```
Ŵ·x̂ = s_w·s_x·(W_int·x_int)                     ← 主体，int8×int8
     − s_w·z_w·s_x·x_int                        ← 依赖输入 x，每 batch 都要算
     − s_w·s_x·z_x·W_int                        ← 可预计算（折进 bias）
     + s_w·z_w·s_x·z_x                          ← 可预计算（折进 bias）
```

- 第 2 项**依赖输入数据**，等价于"多加一个通道"，有真实延迟和功耗代价。
- 所以：**权重用对称（z_w = 0，直接消掉两项）**；激活用非对称（ReLU 输出天然非负，非对称能省一半网格）。
- 这是"数学上更优"输给"硬件上更便宜"的经典案例，面试可以聊。

### 3.2 粒度：per-tensor / per-channel / group

| 粒度 | 谁有 | 硬件代价 | 什么时候用 |
|---|---|---|---|
| per-tensor | 权重 + 激活 | 无（一个 scale 出 sum） | 激活几乎必须用它 |
| per-channel | 只有权重（沿输出通道） | 极低（每个输出通道一个 scale，累加器不跨通道） | CNN/线性层的默认 |
| group (32/64/128) | 只有权重 | 要解包 + 更多 scale，**且累加器要按组 rescale** | 4bit LLM 的主流 |

### 本机实测（4096×4096 权重，通道间量级差 2400 倍）

| 方式 | int8 相对误差 | int4 相对误差 |
|---|---|---|
| per-tensor | 0.136 | 0.792 |
| per-channel | 0.045 | 0.476 |
| group=128 | 0.016 | 0.225 |
| group=64 | 0.013 | 0.183 |
| group=32 | 0.010 | 0.145 |

**读法**：8bit 时 per-channel 就够了（0.136→0.045，3× 改善）；**4bit 时不分组根本不能用**（0.79 的相对误差意味着信息基本丢光），group=128 能把误差砍到 1/3.5，group=32 再砍一刀。

**这就是为什么端侧 4bit 一定是 group-wise，而 group size 是「精度 ↔ 解包开销」的旋钮**：group 越小，精度越好，但 kernel 要处理更多 scale、更多次 rescale。工业界 32/64/128 三档，高通 LPBQ 用的是 **64**。

### 3.3 W-only vs W8A8

| 方案 | 权重 | 激活 | 适用 | 关键约束 |
|---|---|---|---|---|
| W8A8 | int8 | int8 | 服务端高吞吐、有 int8 tensor core | 激活 outlier 要处理（SmoothQuant） |
| W4A16 | int4 | fp16 | **端侧 decode** | 只省带宽，算力还是 fp16 |
| W4A8 | int4 | int8 | 新的甜点 | kernel 要混合精度支持 |

结合 note 03 的结论：**端侧 decode 是带宽墙，所以 W4A16 收益最大；prefill 是算力墙，W4A16 帮不上，需要 W8A8 或 fp16。** 这就是为什么端侧方案常写成 "W4A16 + prefill 用 fp16"。

### 3.4 格式谱系

`int8 / int4(nibble) / fp8(E4M3,E5M2) / mxfp4 / nvfp4`。
- **fp8**：动态范围大，不需要校准（scale 按 block 共享指数），适合训练和 H100 类硬件；
- **mxfp4 / nvfp4**：block 级共享指数（MX 是 32 个元素共享一个 E8M0 指数；NVFP4 是 16 个元素共享一个 FP8 指数），本质是"硬件原生支持的 group-wise"，比 int4+scale 更省元数据；
- 端侧目前主流仍是 **int4 + group scale**，因为 NPU 的 MAC 是定点。

---

## 4. 校准：怎么定阈值

PTQ 的第一步永远是给每个 quantizer 定 `(s, z)`。白皮书列了四类方法（3.1 节）：

| 方法 | 做法 | 特点 |
|---|---|---|
| **MinMax** | `q = max|x|` | 零成本，对 outlier 极敏感 |
| **Percentile** | `q = percentile(|x|, p)` | 一个超参，稳定，工业界最常用 |
| **MSE** | 搜 `q` 使 `‖x − x̂‖²` 最小 | 通常最好，代价是要多跑几遍量化 |
| **KL / Cross-Entropy** | 搜 `q` 使 `KL(P‖Q)` 最小 | TensorRT 的经典做法，对分类任务友好 |
| **BN re-estimation** | 重新统计 BN 的均值方差（或用无标签数据） | 模型有 BN 时几乎必做，零成本大收益 |

### 本机实测（同一个重尾张量）

| 方法 | clip 值 | int8 MSE | int4 MSE |
|---|---|---|---|
| MinMax | 288.7 | 3.06e-01 | 9.29e-01 |
| Percentile 99.9% | 7.44 | 7.83e-02 | 1.67e-01 |
| **MSE 搜索** | 36.05 / 4.77 | **3.96e-02** | **1.43e-01** |
| KL 散度 | 144.3 | 1.14e-01 | 8.87e-01 |

**结论**：在这类重尾激活上 **MSE 搜索 > Percentile > KL > MinMax**。注意 KL 在这里表现不好 —— 它优化的是分布形状而不是数值误差，且两种位宽下都选了约 50% 的分位（144 vs max 288）—— 对"少数 outlier 决定下游"的 LLM 不一定合适。**这也是为什么 AIMET 的默认收尾是 SeqMSE 而不是 KL。**

> ⚠️ 一个实现坑（我自己踩了）：搜 MSE 时**不能只用分位数当候选**。尾部 0.001% 的分位数之间 clip 从 32 跳到 288，网格会直接漏掉最优点（我第一版就漏了，把最优 clip 算成了 9.46）。**要用均匀的 clip 网格搜。**

> 工程上的现实：**权重通常直接用 MinMax**（分布好、且反正后面有 AdaRound/SeqMSE 补救），**激活才值得花力气校准**。

---

## 5. 高级 PTQ 四件套（白皮书的 standard pipeline）

白皮书给出的标准流程是：

```
CLE（或 BN re-estimation） → 校准 → Bias correction → AdaRound → 验收
```

### 5.1 CLE：Cross-Layer Equalization（重点）

**核心事实：ReLU 是正齐次的（positively homogeneous）**

```
ReLU(α·x) = α·ReLU(x),   ∀α > 0
```

所以对相邻两层（中间是 ReLU）：

```
W1 → ReLU → W2
```

可以任意地「把缩放从 W1 挪到 W2」，函数不变：

```
W1' = W1 / s,   b1' = b1 / s,   W2' = W2 · s      ⇒  输出完全相同
```

**问题**：如果 W1 的第 0 个通道范围是 100，第 1 个是 0.1，per-tensor 量化用一个 scale 覆盖全部 → 小通道全变成 0。

**CLE 的解法**：选 `s_i` 让两层在通道 `i` 上的"范围"**在几何中点相遇**：

```
r1_i = max|W1[i, :]|          （中间层第 i 个通道，来自 W1 的行）
r2_i = max|W2[:, i]|          （同一个通道，来自 W2 的列）
s_i  = sqrt(r1_i / r2_i)

W1'[i, :] = W1[i, :] / s_i
b1'[i]    = b1[i] / s_i
W2'[:, i] = W2[:, i] · s_i
```

验证：`r1_i/s_i = sqrt(r1_i·r2_i)`，`r2_i·s_i = sqrt(r1_i·r2_i)` —— 两边相等。

**本机实测（两层 ReLU 网络，人为制造 2000 倍通道失衡）**

| | CLE 之前 | CLE 之后 |
|---|---|---|
| int8 输出相对误差 | 0.808 | **0.022** |
| int4 输出相对误差 | 1.000 | **0.396** |
| FP 等价性（应恒等于原网络） | — | 5.6e-16（数值零） |

**int8 下误差降低 37 倍**，而且函数完全没变 —— 这是"免费"的精度。这就是为什么 CLE 几乎总是 PTQ 的第一步。

**三个必须知道的限制（面试高频）：**

1. **要求中间激活是正齐次的**：ReLU、ReLU6、LeakyReLU 可以；**GELU、SiLU/Swish、softmax 不行**。
2. **前面有 BN 要先 fold**（AIMET 的 CLE = BNF → cross-layer scaling → high-bias fold 三步）。fold 之后 BN 消失，正好露出 ReLU，CLE 才能作用。
3. **Transformer 用不了**（MLP 是 GELU/SiLU，attention 里没有正齐次激活）。**所以 LLM 量化里没有 CLE 的位置** —— 它被"旋转"（SpinQuant/QuaRot）取代了，那是同一个思想的连续版本：用一个正交矩阵把 outlier 摊平，而不是用对角矩阵。

> 这句话在面试里很值钱：**CLE 是「对角缩放版的旋转」，旋转是「连续版的 CLE」**。两者都在利用"函数对某个变换不变"这一性质。

### 5.2 Bias correction

量化权重后，输出的期望会漂移：

```
Δy = Ŵ·x + b − (W·x + b) = ΔW·x
E[Δy] = ΔW · E[x]        （ΔW = Ŵ − W）
```

在 bias 里减掉它：

```
b' = b − ΔW · μ_x        （μ_x 用校准集统计，按输入通道算，conv 要对 kernel 位置求和）
```

**本机实测（4bit，测试集与校准集有轻微分布偏移）**：平均输出偏移 1.640 → 1.356，**只降了 17%**。

为什么不是 100%？因为 bias correction **只修均值，不修方差**：`ΔW·x` 里 `x` 的随机波动造成的误差它管不了。**这是一个很好的"边界感"素材**：任何 PTQ 技术都只修它建模的那一项。

### 5.3 AdaRound

**问题**：round-to-nearest 最小化的是 `|w − ŵ|`，但我们真正关心的是 `|Wx − Ŵx|`。两者不等价 —— 有的权重往"上"取整更好，有的往"下"取整更好。

**做法**：把舍入变成可学习变量：

```
ŵ = s · (⌊w/s⌋ + δ),   δ ∈ {0, 1}
δ̂ = clip(σ(α)·(ζ − γ) + γ, 0, 1)     （ζ=1.1, γ=−0.1 的 rectified sigmoid，连续松弛）
目标:  min_α  ‖Wx − Ŵx‖²_F  +  λ·Σ(1 − |2δ̂ − 1|^β)
```

- 第一项：让量化后的**层输出**逼近 FP 输出；
- 第二项：正则，把 `δ̂` 推向 0/1（否则松弛后的解没法落到整数上）；
- 只需无标签的校准数据，**逐层做**（不端到端），典型 1k 次迭代。

**为什么有效（二阶直觉）**：`Δy = Σ_i Δw_i x_i`。同样大小的 `Δw`，乘在 `|x|` 大的通道上危害更大。AdaRound 会优先让"输入幅度小的那一路"承担误差 —— 这就是**误差分配**，和 GPTQ 的 Hessian 是同一个思想的不同实现。

### 5.4 SeqMSE（AIMET 的收尾）

逐层搜索 `(clip, scale)`，使**量化后该层输出**与 FP 输出的 MSE 最小。它和 AdaRound 正交（一个改阈值、一个改舍入）。

它是高通 **Recipe 2（LPBQ + SeqMSE）的收尾**；Recipe 1 里干同一件事的是 **AdaScale**（不搜 MSE，而是把 scale 变成可学习参数、用 block-wise 蒸馏对齐 FP 输出）——**一个"搜"、一个"学"，这是两套 recipe 的分水岭。**

---

## 6. QAT：什么时候值得做

| 技术 | 核心 | 学什么 |
|---|---|---|
| **STE** | 前向量化、反向直通：`∂L/∂w := ∂L/∂ŵ` | 梯度是错的，但方向对；误差被 `s` 限制住 |
| **LSQ** | `ŵ = s·round(w/s)`，把 `s` 也变成可学习参数 | `∂ŵ/∂s = round(w/s) − w/s`（分段），比固定 scale 好 0.5~1 bit |
| **PACT** | 把 clip 阈值 `α` 变成可学习参数 | 自动学 percentile，替代手工校准 |

**判断标准**：PTQ 到 8bit 通常够；**4bit 且精度要求高，才上 QAT**。QAT 的成本是"要训练管线 + 有标签数据 + 调超参"，而 LLM 的 QAT 成本极高（见下）。

**LLM 为什么很少做 QAT**：几十 B 参数、RLHF 后的模型、几千卡时的成本，而 PTQ（GPTQ/AWQ/SpinQuant）已经能把 4bit 做到可接受。**所以 LLM 量化 = PTQ 的天下，QAT 只在极少数（如 Gemini/Gemma 的量化版本、端侧固定模型的极致压缩）出现。**

---

## 7. 工业界 LLM 量化的三条主线

### 7.1 粒度：per-channel → group

见 §3.2。4bit 必须 group，group size 是精度/速度旋钮。

### 7.2 旋转：把 outlier 摊平

**问题**：LLM 的激活有"massive outliers"—— 少数几个通道的值比其他大 100 倍，per-tensor int8 的 scale 被这几个通道绑架，其余通道全丢精度。

**解法**：在权重和激活之间插入一个正交矩阵 `R`：

```
y = W x = (W R)(Rᵀ x)
```

`R` 正交 ⇒ `‖Rᵀx‖ = ‖x‖`，函数不变；但 `Rᵀx` 的**逐通道分布被摊平**了（outlier 被"抹"到所有通道上），于是 per-tensor/per-channel 量化都变好。

- **QuaRot / SpinQuant**：`R` 用 Hadamard 矩阵或学习得到的旋转（Cayley 优化）。
- **成本**：旋转要和相邻的 linear 融合（`W ← WR` 离线做），**但激活侧的 `Rᵀx` 是真实计算**，在端侧要占一次 elementwise（或融进前一层）。这就是"数学免费、工程不免费"。
- **和 CLE 的关系**：CLE 是 `diag(s)`，旋转是正交阵 `R`，都是"利用函数不变性重分布数值范围"。CLE 只能用于齐次激活，旋转不受限 —— 所以 Transformer 用旋转。

### 7.3 混合精度：官方默认就是 w4a16kv8

不是所有层都值得 int4。**高通 QAIRT-dev 对 LLM 的默认方案是 `w4a16kv8` + 若干 MP（mixed precision）例外**：

| 部分 | 精度 | 理由 |
|---|---|---|
| Decoder linear 权重 | **W4**，per-channel，对称 | 占参数 95%+，收益最大 |
| 输出激活 | **A16**，per-tensor，非对称 | LLM 激活 outlier 多，INT8 容易崩 |
| KV cache 激活 | **KV8**，per-tensor，**对称** | 省内存；且 Genie 上机执行要求 8bit |
| LM-head 权重 | **8bit**（MP） | 降到 4bit 掉点大，官方明确说"需要进一步研究恢复" |
| Embedding 权重 | 16bit 默认，可降 8bit（MP） | 词表大、纯带宽；精度保住就降 |
| Layer/RMSNorm 权重 | 16bit 默认，可降 8bit（MP） | 输出直接进残差，错不起 |

**"W4A16" 这个叫法本身就是混合精度**：权重 4bit、激活 16bit、KV 8bit，三个位宽同时存在。

> 为什么 KV 是**对称**、而普通激活是**非对称**？普通激活（ReLU 后非负）用非对称能省一半网格；但 KV 要参与 `QKᵀ` 和 `PV` 的 **INT8×INT8 GEMM**，对称可以把 §3.1 里那两个 zero-point 修正项完全消掉。**精度上非对称更好，速度上对称更便宜 —— KV 选了速度。** 这一条把"数学"和"硬件"接上了，面试可以直接讲。

### 7.4 其余方法一句话定位（对照你的 docx）

| 方法 | 一句话 | 关键公式 / 开关 | 适合 |
|---|---|---|---|
| **SmoothQuant** | 把量化难度从激活搬到权重 | `Y = (X·diag(s)⁻¹)(diag(s)·W)`，`s_j = max|X_j|^α / max|W_j|^(1−α)`，α≈0.5 | W8A8 |
| **AWQ** | 只保护 1% 显著通道，per-channel scaling | 按激活幅度挑通道，`s = (mean|X|)^α / (mean|W|)^(1−α)`，α 网格搜 | W4A16（weight-only） |
| **OmniQuant** | 学出来的等价变换 + 可学习裁剪 | LWC（学 clip）+ LET（per-channel scale & shift），block-wise | W4A16 |
| **QuaRot / SpinQuant** | 正交旋转压 outlier | `W → WR`，`x → Rᵀx` | W4A8 / W4A4 |
| **Hadamard** | 最简单的一种旋转 | `H` 矩阵，O(n log n) 快速变换 | 同上，免训练 |

**这五个的分工可以这样记**（一句话讲清区别，面试常问）：

- **SmoothQuant**：动激活（scale 进权重），为了 W8A8；
- **AWQ**：动权重（挑通道），为了 W4A16；
- **OmniQuant**：两个都动，但用梯度学；
- **旋转类**：动坐标系（正交变换），为了低比特激活；
- **GPTQ/AdaRound**：动舍入（误差补偿），和上面四个正交，可以叠加。

---

## 8. 高通 QAIRT-dev 的实战配方

> **三个来源，分清哪个是哪个：**
> - **8.2–8.3**：AIMET 官方的两套 LLM recipe 与实测数字（我在文档站抓到的）。
> - **8.4–8.9**：QAIRT-dev 文档里的默认位宽策略、exceptions、RMSNorm/LPBQ/AdaScale 的具体做法（来自你贴在 `Qualcomm/PTQ技术.docx` 里的那份流水线，我逐条核对并补了"为什么"）。
> - **8.10**：一页对照表，标出你的 docx 里还有哪些小节没展开。
>
> 注意：QAIRT 文档站是前端渲染的，我抓不到正文，所以 8.1 的流程图是我按 ARM/CPU/GPU/NPU 通用流水线画的；**以你手上文档的措辞为准，算法层面对得上**。

### 8.1 五个阶段

```
①  训练框架（PyTorch / HF）
      │  export
②  ONNX（静态图，便于插 quantizer）
      │  AIMET / QAIRT 量化：校准 → (CLE/AdaRound/SpinQuant/AdaScale/SeqMSE) → encodings
③  量化后的 ONNX + encodings（scale/offset/bitwidth 的 JSON）
      │  QNN converter（吸收 encodings，做图优化、layout、融合）
④  DLC / QNN context binary（针对 HTP 编译）
      │  on-device
⑤  HTP 上执行（int4 权重解包 + int16/int8 激活 MAC）
```

**每一步都可能掉精度，但只有第 ③ 步之后才能看到真实数字** —— 因为模拟量化（fake quant）和真实定点 kernel 之间有 gap：requantize 的舍入方式、累加器饱和、per-channel 的实际支持情况，都会在 ③ 之后才暴露。**这就是"仿真过了、上机崩了"的根源。**

### 8.2 两套官方 recipe（INT4 权重 / INT16 激活）

**Recipe 1：PCQ + SpinQuant + AdaScale**

| 组件 | 干什么 |
|---|---|
| **PCQ** | per-output-channel 权重量化（4bit） |
| **SpinQuant** | 插入学习到的旋转，压激活 outlier |
| **AdaScale** | 在量化输出上乘一个训练得到的 scale，修**幅度**偏差（方向基本没坏），用 block-wise 蒸馏对齐 FP 输出 |
| 定位 | **性能优先**：目标设备上的 token rate / TTFT 更好，但校准慢（要跑蒸馏） |

**Recipe 2：LPBQ + SeqMSE**

| 组件 | 干什么 |
|---|---|
| **LPBQ**（Low-Power Blockwise Quantization） | block-wise 权重量化（AIMET recipe 用 **block_size=64**，QAIRT-dev 的 Qwen3-Embedding 例子用 **32**），4bit 存储 / 8bit 累加，详见 §8.7 |
| **SeqMSE** | 逐层校准，最小化量化/FP 输出的 MSE |
| 定位 | **精度优先**：粒度更细（64 vs per-channel），但要付 INT4→INT8 解包的性能代价 |

### 8.3 官方实测数字（AIMET 2.35，INT4 W / INT16 A）

| 模型 | FP32 PPL / MMLU | Recipe 1（PCQ+SpinQuant+AdaScale） | Recipe 2（LPBQ+SeqMSE） |
|---|---|---|---|
| LLaMA 3.2 1B | 12.14 / 46.06 | 13.68 / 41.82（113 min） | 13.84 / 43.53（21 min） |
| LLaMA 3.2 3B | 10.13 / 60.74 | 11.14 / 57.28（290 min） | 10.55 / 59.29（73 min） |
| Qwen2.5 1.5B | 12.41 / 54.65 | 13.35 / 50.27（134 min） | 14.33 / 49.97（38 min） |
| Qwen3 4B | 12.41 / 70.06 | 13.79 / 65.07（274 min） | 12.77 / 65.36（95 min） |
| Qwen3 8B | 8.99 / 74.96 | 9.49 / 69.89（403 min） | — |
| Phi-3.5 mini | 5.77 / 68.89 | 6.50 / 62.51（112 min） | 6.41 / 63.90（93 min） |
| Gemma 3 4B | 13.41 / 58.41 | — | **19.61 / 52.79**（105 min） |

**怎么读这张表（面试可以直接用）：**

1. **即使官方最优 recipe，INT4 也有 1~1.5 PPL 的损失**（LLaMA 3.2 1B：12.14→13.68）。"4bit 无损"是营销话术，真实是"4bit 可用"。
2. **Model-dependent 极其严重**：Gemma 3 4B 直接掉到 19.61（+46%），Qwen3 4B 只掉 0.36。**同一个 recipe 换个模型可能就崩** —— 这就是为什么量化工程师的价值在于"会调"，不是"会跑脚本"。
3. **Recipe 1 慢 5 倍但性能更好，Recipe 2 快 5 倍但精度略低**（LLaMA 1B：113 vs 21 分钟）。**这就是"精度-时间-性能"三角**，工程上通常是"先用 Recipe 2 快速验证，精度不够再上 Recipe 1"。
4. **MMLU 掉得比 PPL 多**（46.06→41.82 是 4 个点）。PPL 对量化的宽容度更高，**只看 PPL 会高估量化质量** —— 这是面试里很实用的一个 caveat。

**几个官方 FAQ 里的细节，价值很高：**

- **校准只用了 20 个 batch**（WikiText），SeqMSE 也是 20 个 batch。校准不是越多越好，**多样性 > 数量**。
- **KV cache 的精度要看模型**：Qwen2.5 用 INT8 KV 会明显掉点，因为它动态范围更宽、INT8 的 256 个离散值不够，官方直接给了 **INT16**。这是一个"不要无脑 INT8"的绝佳例子。
- **recipe 产物不能直接上机**：官方明确说 "not directly compatible with QAIRT，需要额外的 adaptation steps"。**模拟量化通过 ≠ 设备通过**。

### 8.4 QAIRT-dev 的逐张量策略（一手资料）

这张表是 QAIRT-dev 对 **LLM 语言模型**的完整默认配置，比"W4A16"三个字信息量大得多：

| 类型 | 层 | 位宽 | 对称性 | 粒度 | 官方给的理由 |
|---|---|---|---|---|---|
| W | Linear | 4 / 8 | 对称 | per-channel | 参数主体，收益最大 |
| W | Layer / RMSNorm | 16 / 8 | **非对称** | **per-tensor** | 输出进残差，要升级 |
| W | Embedding | 16 / 8 | 非对称 | per-tensor | 非零均值；先省内存 |
| W | LM-head | 8 | 对称 | per-channel | 让 linear matmul / conv 更快 |
| A | 输出激活 | 16 | 非对称 | per-tensor | 精度 |
| A | KV cache | 8 | **对称** | per-tensor | 省内存 |
| A | MatMul input 1（Q） | 16 | 非对称 | per-tensor | **要过 softmax** |
| A | MatMul input 2（K/V） | 8 | 对称 | per-tensor | `INT8×INT8` 硬件通路 |
| A | `self_attn_Concat` | 8 | 对称 | — | 不被复用；让 o_proj 走 `INT8×INT8` |

**逐条读出来的四个"为什么"：**

1. **softmax 前必须 16bit**：`QKᵀ` 的输出要过 softmax，而 softmax 对 outlier 极敏感（`exp` 会把误差指数放大，再做归一化），Q 留在 16bit 是硬约束。**softmax 之后可以降到 8**：attention 权重落在 [0,1]，动态范围天然好，`P·V` 正好走 INT8×INT8。
2. **`self_attn_Concat` 也压到 8**：它只喂给 `o_proj`、不被其它地方复用，压到 8bit 就能让 output projection 变成 INT8×INT8 GEMM。**一个"没有下游复用"的张量，是免费的低比特机会。**
3. **LM-head 是 8 不是 4**：官方原话是"降到 4bit 掉点大，需要进一步研究恢复"。**输出层是量化最不划算的地方**：它只算一次（decode），省下的带宽有限，但误差直接决定 argmax。
4. **Norm 权重是"非对称"**：因为 γ 围绕 1（Qwen3 是围绕 1 的 zero-centered `1+w`），非零均值 → 非对称更省网格。**这和 Linear 权重的"对称"是相反的选择，原因是分布不同。**

> **ViT（LVM 里的视觉塔）是另一套：`w8a16`。** 官方明确说 w4 在 ViT 上掉点大，所以视觉塔默认 8bit，Norm 也可以跟到 8bit。**同一份 recipe 里，LLM 和 ViT 用不同位宽** —— 这也是"量化是 model-dependent"的直接证据。
>
> 还有个**工程现实**：ViT 没有 KV cache，本来不需要"第二输入 8bit"（16×8）这套；但当前仍然用 16×8，因为 **16×16 需要 `use_dynamic_16_bit_weights` 标志，而 `qairt-quantizer` 里这个 flag 有 bug**。另外注意 16×8 会连累 `k/v-proj` 的输出也变成 8bit。**"配置被工具链的 bug 决定"是工程常态，面试讲出来比背论文真实。**

### 8.5 两个必须知道的图级开关

**① `propagate_output_encodings`（Concat 编码统一）**

图上任何一个 `Concat`（新旧 K/V 拼接、rotary 的 cos/sin 两半拼接），强制它的所有输入 quantizer 与输出共享同一组 `scale/offset`。

不这么做会怎样：每个输入各自独立校准 → 拼接时两条支路的量化网格不重合 → 拼出来的张量必须**再量化一次**到统一 scale（凭空多一层误差），而且 **KV cache 每跨一个 step 都要拼一次，误差会随 step 累积漂移**。

**这是一个"图级约束比算法更重要"的例子**：数学上每个张量都有自己的最优 scale，但工程上"共享 scale"换来的是零重新量化和跨 step 的一致性。

**② `clip_weights_to_7f7f`**

权重裁剪到 `[-127, 127]`，避开 `-128`。有些 kernel 对 `-128` 有不对称处理（比如对称量化里 `-128` 没有对应的正数），裁掉一个码点换取 kernel 的规整性。**这类"少一个码点"的细节，是仿真和真机对不上的常见原因之一。**

配套的还有：**全局激活裁剪默认关闭**（LLM 激活 outlier 太多，全局 clip 一定伤精度）。

### 8.6 RMSNorm 为什么必须 per-tensor（最容易做错的一条）

先看事实：**Linear 的权重用 per-channel，Norm 的权重用 per-tensor。** 为什么同样是权重，粒度不同？

- `x` 是 residual stream（fp16），`γ` 是 per-hidden-dim 的缩放向量。
- RMSNorm 的语义是「**全局**归一化 + **均匀**缩放」：归一化因子（RMS）在所有通道上共享，`γ` 只调整各通道的相对重要性，**不改变归一化结构**。
- 如果对 `γ` 做 per-channel 量化：每个通道有自己的 scale → 每个通道的量化噪声**不同** → 归一化之后，这个"不均匀噪声"被隐式地再归一化了一次 → **残差流的通道间失衡**。
- 后果是**逐层累积**的：residual imbalance → layer-to-layer drift → 深度越深越糟。

所以：**Norm 的 weight 用 per-tensor（+ 非对称，因为 γ≈1）**。

**这是本笔记最值得记的一条工程直觉：per-channel 不是"更细就一定更好"。** 粒度要匹配层的语义 ——

- Linear 的输出通道之间**本来就独立**（每个通道一个独立的权重行），所以 per-channel 是纯收益；
- Norm 的输出通道之间**被同一个 RMS 绑在一起**，per-channel 反而破坏了这个约束。

> 面试里如果被问"per-channel 一定比 per-tensor 好吗"，这就是标准答案：**看这个张量的通道之间有没有"共享的归一化/耦合"，有耦合的（Norm、softmax 输入）必须 per-tensor。**

### 8.7 LPBQ：为什么它叫"低功耗"块量化

这一条最容易漏 —— 只说 `block_size` 是不够的，真正的设计在 scale 的存法上：

- **存储**：权重 INT4；**解压/累加**：INT8。所以它是"4bit 存储、8bit 计算"的混合方案。
- **分块**：沿 channel 维切 block（AIMET recipe 用 64，QAIRT-dev 的 Qwen3-Embedding 例子用 **32**），每块一个 scale（按块内最大幅值定）。
- **关键：两级 scale。** 如果每块都存一个 fp16 scale，元数据是 `16bit / 64 weights = 0.25 bit/weight`，4bit 直接变 4.25bit，白亏 6%；block 越小亏得越多（32 就是 0.5 bit/weight）。所以 LPBQ 用：
  ```
  有效 scale = channel 级 fp16 scale × block 级低精度 multiplier
  ```
  multiplier 用很低的精度存（几个 bit），元数据降到 `0.06（b=64）/ 0.13（b=32）` bit/weight 量级。**这就是 "Low-Power" 里 "low" 的真正含义 —— 省的是元数据带宽，不是算力。**

**官方给的适用条件（三条，全中才用）：**

1. 是量化过的 Conv2d（Linear 在 QAIRT 里常被 lower 成 Conv2d）；
2. 已经是 INT4；
3. **不是 hadamard / rotation / special mixing 层** —— 因为旋转之后权重分布被改变了，block 内的 max 不再有代表性，分块反而更差。

**通常不对 `lm_head` 做 LPBQ 或 SeqMSE**（输出层敏感 + 只算一次）。

> block size 没有最优值，只有三角权衡：**越小越准（块内范围更窄）、元数据越多、解包越碎**。32 和 64 都在用，看模型和硬件。

### 8.8 AdaScale：保方向，不保长度

官方对 AdaScale 的动机描述非常精炼，值得原样记住：

> **量化通常保留方向（cosine 相似度高），但扭曲长度（L2 norm）。**

所以 AdaScale 干的事是：在**量化输出**上再乘一个训练得到的逐通道 scale，专门修 magnitude 偏差 —— 它修的不是量化网格，而是**量化之后剩下的系统性幅度误差**。

为什么这招有效：量化误差在**方向**上近似随机（正负抵消），但在**幅度**上是**系统性**的（int4 网格太粗，round 的期望不为零，整个向量的长度会系统性偏大或偏小）。

**和 SeqMSE 的分工，一句话：**

| | SeqMSE | AdaScale |
|---|---|---|
| 怎么找 scale | **搜**（离散候选，逐层最小化 MSE） | **学**（梯度下降 + block-wise 蒸馏） |
| 代价 | 便宜（20 个 batch） | 贵（128 batch × 2048 iter） |
| 修什么 | 网格（clip）选得对不对 | 量化后的**幅度**偏差 |

### 8.9 一个真实的端到端 recipe：Qwen3-Embedding

这段是把上面所有概念串起来的最好样本（QAIRT-dev 里的实际配置）：

| 项 | 设置 | 为什么 |
|---|---|---|
| 基础 | `QuantizationSimModel` + `post_training_tf` + `htp_quantsim_config_v81.json`，W8A16 | 从保守的 8bit 起步 |
| K/V matmul 第二输入 8bit | **关闭** | 这个模型上不划算（省的不够赔的） |
| Concat 编码统一 | **开** | 见 §8.5 |
| 混合精度 | `exceptions_mlp4_16x16.json` | MLP 权重 4bit、激活 16bit |
| Embedding | 8bit | 省内存且精度够 |
| RMSNorm weight | 强制 **per-tensor, 16bit 非对称** | 见 §8.6 |
| LPBQ | 开，`block_size=32`、`decompressed_bw=8`、对称 | 把 decoder 权重从 8bit 压到 4bit |
| SeqMSE | 40 batch、20 候选、MSE loss | 收尾 |
| 校准 | 100 batch，领域数据（T2Retrieval） | 校准集要贴近真实分布 |
| 权重 clip | `clip_weights_to_7f7f` 恒开 | 见 §8.5 |
| 全局激活 clip | 关 | LLM outlier 多 |
| AdaScale | **跳过**（`SKIP_ADASCALE=True`） | 这条 recipe 不用它（省时间）；实际项目里看精度余量 |

**最有价值的是官方的总结（我原样抄下来了）：**

> "这实际上已经是一个 W4A16 的混合精度 recipe 了 —— 那个 '8-bit' 的 `default_param_bw` 只是个起点，LPBQ 随后把大部分 decoder 权重覆盖成 4bit，embedding 留在 8bit，其余 16bit。**它并不是一次干净的、均匀的 W4A16 实验。**"

**面试可以直接用**：recipe 的"名字"和"实际位宽"经常不一致，**要看最终 encodings，不要看脚本名字**。这是很多"我们做了 W4A16"的说法的真实含义。

### 8.10 查漏补缺：你的 docx → 本文对照

| 你 docx 里的小节 | 本文位置 | 状态 |
|---|---|---|
| Quantization 基础（W/A、per-tensor/channel） | §1–3 | ✅ |
| CLE | §5.1 | ✅ + 本机实测 37× |
| Bias Correction | §5.2 | ✅ + 本机实测 17% |
| LLM scheme（w4a16kv8 + MP） | §7.3、§8.4 | ✅ **本轮补齐** |
| Vision scheme（w8a16） | §8.4 | ✅ **本轮补齐** |
| Exceptions / `propagate_output_encodings` | §8.5 | ✅ **本轮补齐** |
| RMSNorm quantization | §8.6 | ✅ **本轮补齐** |
| Gate_up MLP heavy tails | §7.3、§8.4 | ✅ 有解释 |
| Qwen3 Embedding recipe | §8.9 | ✅ **本轮补齐** |
| SeqMSE | §5.4 | ✅ |
| LPBQ | §8.7 | ✅ **本轮补齐**（两级 scale） |
| AdaScale | §8.8 | ✅ **本轮补齐**（方向/长度） |
| OmniQuant | §7.4 | 🔸 只有一句话定位，待展开 |
| SpinQuant / Hadamard | §7.2 | ✅ |
| SmoothQuant / AWQ | §7.4 | 🔸 只有公式与定位，待展开 |
| PrefixQuant | — | ⬜ **待查证后写** |
| TurboQuant | — | ⬜ **待查证后写** |

> 关于最后两个：`PrefixQuant` 我理解是"outlier 集中在开头几个 token，把这部分 prefix 的 KV 单独处理/缓存，剩下的就能静态量化"，但细节没把握；`TurboQuant` 我需要查原文再写。**不确定的东西我不写进笔记** —— 你告诉我具体是哪篇（或贴一段原文），我半小时内补完。

### 8.11 选型决策树

```
能用 fp16 吗？── 能 → 停（端侧大部分场景其实 fp16 就够，先测再量化）
   │不能
   ▼
精度优先？（医疗/金融/长上下文 RAG）
   ├─ 是 → LPBQ(32/64) + SeqMSE，KV 用 INT16，embedding INT16
   └─ 否 → PCQ + SpinQuant + AdaScale，KV 试 INT8
              │
              ▼
        上机测 PPL/任务指标 → 掉点超标？
              ├─ 否 → 定稿
              └─ 是 → ① KV 升到 INT16 ② 敏感层单独升 8bit ③ 换 Recipe 1
```

### 8.12 从 recipe 到设备：还差三步

1. **混合精度 profile**：把 embedding / lm_head / KV / 敏感层的位宽写进 encodings（AIMET 的 mixed-precision 工具可以直接产出）。
2. **QAIRT 侧的 op 支持检查**：HTP 对某些 op（比如动态 shape 的 attention、Gated DeltaNet 的 scan）只支持特定精度，**不支持的会 fallback 到 CPU，性能直接崩**。
3. **上机验收**：同一份 encodings 在仿真和真机上可能差 1~2%，要用真机的 PPL/任务指标说话。

---

## 9. 精度调优方法论（掉点归因五步）

这是我建议你固化成肌肉记忆的流程：

1. **换指标**：先确认不是评测本身的问题（PPL 掉但 MMLU 没掉？换任务集）。
2. **二分定位**：只量化权重 vs 只量化激活 vs 都量化 —— 立刻知道问题在哪一侧。
3. **逐层对比**：把第 `i` 层输出和 FP 对比，找第一个误差突增的层（AIMET 的 layer output generation 就是干这个）。
4. **归类**：误差突增的层是哪一类？
   - 有 outlier 的激活 → 旋转 / 提高该层精度 / per-channel
   - 权重分布差 → AdaRound / SeqMSE
   - 结构问题（LN、softmax、残差） → 这些层干脆留 fp16
5. **改一个变量再测**：**一次只改一个**，否则你不知道是谁起的作用。

**我自己的经验顺序**（按性价比）：
`KV 精度 > 敏感层混合精度 > 校准方法 > 旋转 > AdaRound/SeqMSE > QAT`。

---

## 10. 面试问答（17 题）

**Q1：量化的两种误差是什么？怎么权衡？**
舍入误差（≤ s/2，方差 s²/12）和裁剪误差（超出范围被 clip）。range 放宽 → 舍入变差、裁剪变好。**位宽越低越要激进裁剪**（实测：int8 最优 clip ≈ max 的 12.5%，int4 只有 1.7%）。

**Q2：为什么权重对称、激活非对称？**
对称权重让 `z_w=0`，把非对称展开里两项数据相关的项直接消掉；激活非对称（ReLU 后非负）能省一半网格。**本质是数学最优让位于硬件成本。**

**Q3：per-tensor / per-channel / group 的取舍？**
per-tensor 硬件零成本但误差大；per-channel 对权重几乎免费，是 CNN/linear 默认；**4bit 必须 group**（实测 per-tensor 相对误差 0.79，group=64 是 0.18）。group 越小越准、解包越贵，工业界 32/64/128。

**Q4：什么是 CLE？公式怎么写？为什么有效？**
利用 ReLU 的正齐次性 `ReLU(αx)=αReLU(x)`，在相邻两层间做 per-channel 缩放：`s_i = sqrt(r1_i/r2_i)`，`W1[i,:]/=s_i`、`b1[i]/=s_i`、`W2[:,i]*=s_i`。函数不变，但把通道间的量级差异抹平，per-tensor 量化不再被 outlier 通道绑架。**实测 int8 输出误差 0.81 → 0.02。**

**Q5：CLE 有什么限制？为什么 LLM 不用？**
要求中间激活正齐次（ReLU/ReLU6），前面有 BN 要先 fold；**GELU/SiLU 不齐次，所以 Transformer 用不了**。LLM 里被"旋转"（QuaRot/SpinQuant）取代 —— 旋转是连续版、CLE 是对角版的同一个思想。

**Q6：bias correction 修什么？为什么修不干净？**
修 `E[Δy] = ΔW·E[x]` 这个均值漂移：`b' = b − ΔW·μ_x`。**它只修均值，不修方差**（`ΔW·x` 的随机项），所以实测只降了 17%。

**Q7：AdaRound 为什么比 round-to-nearest 好？**
RN 最小化 `|w−ŵ|`，但目标是 `|Wx−Ŵx|`。AdaRound 用连续松弛把舍入变成可学习变量，逐层最小化输出 MSE，让"输入幅度小的通道"承担误差。**本质是误差分配问题，和 GPTQ 的 Hessian 同源。**

**Q8：W4A16 为什么在 decode 有效、prefill 无效？**
量化只减字节不减 FLOPs。decode 是带宽墙（AI≈1），减 4× 字节 ≈ 4× 加速；prefill 是算力墙（AI≈460），减权重字节没用（见 note 03）。

**Q9：LLM 量化最难的是什么？**
**activation outlier**。少数通道比其他大 100 倍，per-tensor int8 的 scale 被绑架。解法：旋转（QuaRot/SpinQuant）、per-channel/per-group、或把激活留在 16bit（W4A16）。**注意 outlier 是"少数通道"，不是"少数 token"** —— 这是它规律可建模的原因。

**Q10：为什么 embedding 用 INT16、lm_head 用 INT8？**
embedding 是查表、纯带宽，量化省的是内存；但每个 token 只查一行，误差会直接进模型，所以保 16bit。lm_head 只在最后算一次、且对 argmax 敏感，用 8bit 平衡。

**Q11：KV cache 量化要注意什么？**
看动态范围。Qwen2.5 用 INT8 KV 会明显掉点（256 个离散级不够），官方直接给 INT16。**"KV 一定要 INT8"是错的** —— 这是个模型相关的决策。

**Q12：PTQ 和 QAT 怎么选？**
8bit PTQ 基本够；4bit 且精度要求高才上 QAT。**LLM 几乎不做 QAT**（成本太高），靠 PTQ + 旋转 + 混合精度。

**Q13：KV cache 为什么用对称量化，而普通激活用非对称？**
普通激活（ReLU 后非负）用非对称能省一半网格；但 KV 要参与 `QKᵀ`/`PV` 的 INT8×INT8 GEMM，**对称可以把 zero-point 修正项完全消掉**（§3.1 展开里的那两项）。精度上非对称更优、速度上对称更便宜，KV 选了速度。

**Q14：为什么 RMSNorm 的权重必须 per-tensor，Linear 却可以 per-channel？**
Norm 的语义是「全局归一化 + 均匀缩放」，通道之间被同一个 RMS 绑在一起。per-channel 会让每个通道的噪声不同，归一化后被隐式再归一化 → 残差流通道失衡 → 逐层累积漂移。**Linear 的输出通道本来就独立，per-channel 是纯收益；Norm 不是。** 判据是"通道之间有没有耦合"。

**Q15：图上哪些张量最值得压到 8bit？**
**没有下游复用的那些**。比如 `self_attn_concat`（只喂 o_proj）：压到 8bit 就能让 output projection 走 INT8×INT8 GEMM，收益直接。反过来，要过 softmax 的 `Q` 必须留 16bit。

**Q16：LPBQ 的 "low power" 低在哪？**
不在算力，在**元数据**。每 block 存一个 fp16 scale，元数据是 `16/64 = 0.25 bit/weight`，4bit 变 4.25bit；所以 LPBQ 用「channel 级 fp16 scale × block 级低精度 multiplier」，元数据降到 ~0.1 bit/weight。

**Q17：量化后输出方向没变、长度变了，说明什么？怎么修？**
说明误差在方向上近似随机抵消、在幅度上是**系统性**的（int4 网格粗，round 的期望不为零）。修法就是 AdaScale：在量化输出上乘一个训练得到的逐通道 scale。**这也是为什么"cosine 相似度高"不能证明量化没伤模型。**

---

## 11. 动手实验

先跑：`python tools/quant_basics.py`（纯 numpy，30 秒内跑完），对照第 2/3/5 节的表。

然后四个扩展，每个都能变成面试里的一句话：

- [ ] **把 group size 扫全**：在 §3.2 的代码上扫 `g ∈ {16,32,64,128,256}`，画出"相对误差 vs 每个权重的字节数"曲线，找到拐点。验收：能说出"为什么高通 LPBQ 选 64，不是 32"。
- [ ] **给 CLE 加个反例**：把激活换成 GELU，验证 CLE 之后函数**不再等价**（误差 >> 0）。验收：能解释"为什么 Transformer 用不了 CLE"。
- [ ] **手写 AWQ 的 scaling**：按激活幅度挑 1% 的显著通道，做 per-channel scaling `s = (mean|X|)^α / (mean|W|)^(1−α)`，看输出误差降多少。验收：能说出 AWQ 和 SmoothQuant 的区别（一个按激活挑、一个按通道平滑）。
- [ ] **GPTQ 的 block size**：把 `tools/quant_basics.py` 的 `gptq()` 改成真正的 block（128 列一块），比较 2bit 下 RTN / GPTQ-block 的差距。验收：能解释"为什么 GPTQ 要用 Cholesky 而不是直接求逆"。

---

## 12. 参考

- **A White Paper on Neural Network Quantization**（Nagel et al., 2021, arXiv:2106.08295）—— 本笔记的 §1~§6 骨架。重点读 3.1（校准）、3.2（CLE）、3.3（bias）、3.4（AdaRound）、§4（QAT）。
- **AIMET 文档**（2.35.x）：[PTQ techniques](https://qualcomm.github.io/aimet-pages/releases/2.35.1/ptq_techniques/index.html)、[Cross-layer equalization](https://qualcomm.github.io/aimet-pages/releases/2.35.1/ptq_techniques/cle.html)、[LLM quantization recipes](https://qualcomm.github.io/aimet-pages/releases/2.35.1/tutorials/quantization_recipe.html)、[Low-Power Blockwise Quantization](https://qualcomm.github.io/aimet-pages/releases/2.35.1/techniques/lpbq.html)。
- **QAIRT 文档**：[Pipeline quantization recipes](https://docs.qualcomm.com/doc/80-87189-2/topic/pipeline_quantization_recipes.html)（前端渲染，需在浏览器里看）。
- **你自己的摘录**：`Qualcomm/PTQ技术.docx` —— §8.4~§8.9 的一手细节（w4a16kv8 表、exceptions、RMSNorm、LPBQ、AdaScale、Qwen3-Embedding recipe）全部来自这里，我做了逐条核对和补"为什么"。
- 算法原文：GPTQ (arXiv:2210.17323)、AWQ (arXiv:2306.00978)、SmoothQuant (arXiv:2211.10438)、QuaRot (arXiv:2404.00456)、SpinQuant (arXiv:2405.16406)、OmniQuant (arXiv:2308.13137)。
- 本机脚本：`tools/quant_basics.py`（本文所有"实测"数字的来源）。
