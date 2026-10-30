# 05 · 端侧推理加速：投机解码（SPD）与 KV cache 优化

> 学习目标：说清「量化之外，端侧还能怎么快」，尤其是投机解码与 KV cache 这两条线；能算加速比、能解释为什么无损、能说出在手机 NPU 上落地时哪一步会崩。
>
> 来源：投机解码原理与仿真为本文自算（`tools/spec_decode_sim.py`）；EAGLE 家族来自 arXiv:2401.15077 系列 + vLLM Speculators 文档；QAIRT 的 LADE/SSD/Eaglet 配置来自你贴的 QAIRT-dev 文档；DSpark 来自 DeepSeek 技术报告与公开报道；移动 NPU 的 KV 复用来自 EuroSys'27 论文。

---

## 0. 一句话结论

**端侧 decode 的瓶颈只有一个：每生成一个 token，都要把权重和 KV 从 DDR 搬到片上。量化解决「一次搬多少」，投机解码解决「搬几次」，KV 优化解决「KV 那一份搬多少」—— 三者正交、可以叠加，但收益会互相稀释，必须联合调参。**

---

## 1. 全景：端侧加速除了量化还有什么

| 层面 | 技术 | 省掉什么 | 代价 |
|---|---|---|---|
| 权重/激活 | **量化**（note 04） | 每字节数 | 掉点、校准 |
| 解码 | **投机解码 / PLD / Medusa** | 串行步数 | draft 内存、复杂度 |
| KV | **量化 / 压缩 / 驱逐 / 复用 / 分层存储** | KV 读写字节 | 精度、实现复杂度 |
| 图 | 融合、静态 shape、layout | 搬运次数 | 编译期约束 |
| 模型 | 蒸馏、剪枝、稀疏、MoE | 参数或 FLOPs | 要重训 |
| 输入 | prompt 压缩、前缀缓存 | prefill 计算 | 有损 / 命中率 |
| 调度 | 早退、并行解码 | 层数 | 精度 |

**四条主线的关系，记住这一句**：

```
prefill（TTFT）← 前缀复用 / KV 复用 / 静态图
decode （TPOT）← 量化 / 投机解码 / KV 压缩
```

两者优化的东西不一样，**不要用 decode 的手段去救 TTFT，反之亦然**（这是 note 03 的 roofline 结论）。

---

## 2. 投机解码（SPD）：原理与账

### 2.1 为什么 decode 是串行的

decode 每步只能出一个 token（自回归），而每一步都要读一遍全部权重 → **batch=1 时是纯带宽墙**（note 03：$AI \approx 1$ FLOP/Byte）。算力单元 95% 的时间在等数据。

投机解码的洞察：**既然带宽是瓶颈、算力是闲的，那就一次喂给算力多个 token。** 用小模型（draft）猜 γ 个，再用大模型（target）一次前向并行验证 γ+1 个 —— 验证的成本几乎等于生成 1 个 token（权重只读一遍），但可能一次吐 3~4 个。

### 2.2 怎么保证无损：拒绝采样

设 target 分布 $p$，draft 分布 $q$。对 draft 采出的 token $x$：

```
接受概率 = min(1, p(x) / q(x))
若拒绝 → 从 norm(max(0, p − q)) 重新采一个
```

**本机仿真验证（V=50 的词表，40 万样本）**：

| 采样方式 | 与真实 p 的 TV 距离 | 结论 |
|---|---|---|
| 只接受，拒绝后**重采同一个** draft token | 0.362 | **有偏，错的** |
| 严格拒绝采样（从 `max(p−q,0)` 重采） | **0.0032** | 无偏（自然噪声 ~0.0037） |

**这是投机解码最容易被问倒的地方**：很多人知道"按比值接受"，但不知道**拒绝之后必须重新采样**，否则分布是偏的。0.36 的 TV 距离意味着输出分布被彻底改变。

### 2.3 三个数字：接受率、每步产出、加速比

单 token 接受率有解析式，不用仿真：

```
α = Σ_x min(p(x), q(x))          # 就是 1 − TV(p,q)
```

一次 draft γ 个 token，期望接受长度（几何分布求和）：

```
E[accepted] = (1 − α^(γ+1)) / (1 − α)
```

加上 draft 的成本 $c\cdot\gamma$（$c$ = draft 成本 / target 成本）：

```
speedup(α, γ, c) = [(1 − α^(γ+1)) / (1 − α)] / (1 + c·γ)
```

**本机算出来的最优 γ（扫 1~16）**：

| α | c=0.05（draft 很小） | c=0.1 | c=0.2（draft 偏大） |
|---|---|---|---|
| 0.6 | γ=4, **1.92×** | γ=3, 1.67× | γ=2, 1.40× |
| 0.8 | γ=8, **3.09×** | γ=6, **2.47×** | γ=4, 1.87× |
| 0.9 | γ=13, 4.67× | γ=10, 3.43× | γ=7, 2.37× |

**α=0.8、c=0.1 时的曲线（记住这条拐点）**：

| γ | 每步产出 token | 加速比 |
|---|---|---|
| 2 | 2.44 | 2.03× |
| 4 | 3.36 | 2.40× |
| **6** | 3.95 | **2.47×（峰值）** |
| 8 | 4.33 | 2.40× |
| 12 | 4.73 | 2.15× |

**三个必须记住的推论：**

1. **γ 不是越大越好**：超过拐点后，draft 成本线性涨、收益对数涨，加速比反而下降。
2. **收益上限由 c 决定，不是由 α 决定**：c 从 0.05 涨到 0.2，加速比从 3.1× 掉到 1.9×。所以"draft 要足够小"比"draft 要足够准"更硬。
3. **α 有上限**：α=0.9 已经很好，模型再好也顶不到 1（否则就是它本身了）。

### 2.4 为什么端侧比云端更划算（这条最重要）

云端大 batch 时，decode 是**算力墙**：验证 6 个 token 真的要多花 6 倍算力，$c$ 也接近参数比 → 收益被吞吐摊薄，所以云端更看重"吞吐"而不是"单请求延迟"。

端侧 batch=1 时，decode 是**带宽墙**：验证 γ+1 个 token 只多读一点激活，**权重还是只读一遍** → verify 的边际成本接近 0，而 draft 模型小（典型 1/10 参数）→ $c \approx 0.05 \sim 0.1$。**这是投机解码在端侧最舒服的区间。**

> 面试话术：**"投机解码本质是拿闲置的算力换串行的步数。端侧 decode 的算力本来就闲置着，所以这笔交易在端侧最划算。"**

---

## 3. EAGLE 家族：为什么要在 feature 层做 draft

### 3.1 EAGLE-1 的核心：不在 token 层猜，在 feature 层猜

普通投机解码要训一个独立的小语言模型当 draft。EAGLE 说：**不用，直接在 target 的特征上接一个极小的 head。**

```
输入：token embedding(t) ⊕ hidden_feature(h_t)      ← 上一层的隐状态也喂进去
输出：下一位置的 hidden feature  ĥ_{t+1}
      → 过 target 的（冻结的）LM head → 得到 token
```

**为什么 feature 层更好（三个理由，面试常问）：**

1. **信息更多**：token 只是词表的离散选择，hidden feature 是连续的、携带分布信息；draft 在 feature 上回归比在 token 上分类更"平滑"。
2. **参数极少**：一个 head 通常几十 M 甚至更少，而独立小模型要几 B —— **$c$ 直接小一个量级**（回到 §2.3 的结论：c 决定上限）。
3. **天然对齐**：draft 用的是 target 自己的特征空间和 LM head，不需要额外蒸馏对齐。

**代价："feature uncertainty"**。推理时 draft 的输入是"已经采样出来的 token"，而训练时是"真实的下一 token" —— 两者的特征分布不一致。EAGLE-1 的处理是在训练时给输入加噪声/做增强，让 draft 见过"被采样错"的情况。

### 3.2 EAGLE-2：树从"固定"变"动态"

EAGLE-1 用固定的 draft tree（每层展开几个分支写死）。问题是：**不同 prompt 的确定性差别巨大**（代码里的 `return` 几乎必然，闲聊里下一个词很随机）。

EAGLE-2 用 draft 自己的置信度动态决定树的形状：**置信度高的节点多展开，低的剪掉**。收益是同样预算下覆盖更多高概率路径。

### 3.3 EAGLE-3：多层特征 + training-time test

两个改动：

1. **多层特征融合**：不只取最后一层 hidden，而是把低/中/高层的特征拼起来 —— 低层保留"词法"信息，高层是"语义"，对 draft 都有用。
2. **Training-time test（训练时模拟推理）**：训练阶段就让 draft 吃"自己前几步的输出"，模拟推理时的误差累积；同时**放开"必须精确预测 feature"的约束**，改成直接优化"预测出的 token 对不对"。

> 这两条其实在解决同一个问题：**draft 是自回归的，误差会沿 draft 长度累积，越往后接受率越低。**（记住这个"接受率衰减"，下一节 DSpark 就是专门打它的。）

**vLLM Speculators 文档里对 EAGLE-3 的实现描述（服务端视角，最具体的一份）**：

| 环节 | 做法 |
|---|---|
| 输入 | target 在**选定若干层**的 hidden states → 拼接 → 与 token embedding 一起过 FC 投影 |
| draft 本体 | **Llama 风格的 decoder 层（默认只 1 层）** + LM head |
| 训练目标 | **最小化与 target logits 的 KL 散度**（不是 MSE，也不是逐 token 分类） |
| 推理 | draft 自回归出 K 个 token（每次把自己的预测喂回去）→ target **一次前向验证 K 个** → 接受**最长正确前缀** → 从最后一个被接受的 token 继续 |
| 词表 | **支持 cross-tokenizer**（draft 词表可以和 target 不同）← 这条对应 §4.3 的 `draft_token_map` |

**"最长正确前缀"这个词要记住**：验证阶段只要前 2 个对、第 3 个错，就只接受 2 个 —— 这也是为什么"接受长度"是投机解码的核心指标。

### 3.4 从服务端 EAGLE 到端侧 Eaglet：五个必须改的地方

论文到 NPU 之间隔着五道改造，这五条就是 §4 的铺垫：

| 服务端（vLLM） | 端侧（QAIRT Eaglet） | 为什么必须改 |
|---|---|---|
| 运行时参数 `num_speculative_tokens` | 编译期常量 `draft_len=6` | NPU 的图 shape 编译期定死，改 K 要重新 build |
| 动态 draft tree（EAGLE-2） | 固定 `n_branches=6` 的树 | 静态图不能有数据依赖的分支 |
| 词表随 target | `draft_token_map.json` 显式映射 | draft 常用裁剪词表省带宽，id 空间不一致 |
| draft 用 fp16/bf16 | draft 单独量化（W8/W4） | 要和 target 的 encoding 语义对齐，否则 logits 对不上 |
| KV 由 framework 管 | **KV 布局在两张图之间共享** | 不一致就要每步拷贝，把收益吃光（§4.4） |

> **一句话**：投机解码在论文里是"算法"，在端侧是"图工程"。**能跑通的人很多，能把它和 KV、量化、静态图对齐的人很少** —— 这就是你的差异化。

---

## 4. 高通 QAIRT-dev 的三种投机方案

QAIRT-dev 把它做成一个开关：`build()` 之前给 `builder.speculative_config` 赋一个 config，三选一。

| 方案 | 全称 | 要 draft 模型吗 | 关键参数 | 一句话机制 |
|---|---|---|---|---|
| **LADE** | Look-Ahead Decoding | ❌ 不要 | `window=8, ngram=5, gcap=8` | Jacobi 迭代 + n-gram 池，无模型「猜」候选 |
| **SSD** | Self-Speculative Decoding | ❌ 不要（只有 `ssd_tensor.pt` 权重） | `forecast_token_count=4, forecast_prefix=16, branches=[4,4]` | 模型自带 forecast 头预测未来 token |
| **Eaglet** | EAGLE-Based Decoding | ✅ 要（`draft_model.onnx`） | `draft_len=6, n_branches=6` | 标准 EAGLE：draft head 出候选，target 验证 |

**先记住这个分类轴**：**"draft 从哪来"决定了一切** —— 不要模型的最省内存、要模型的精度最高。

### 4.1 LADE = Lookahead Decoding（不需要 draft 模型）

论文：*Break the Sequential Dependency of LLM Inference Using Lookahead Decoding*（arXiv:2402.02057，Fu et al.）。核心是两个机制：

1. **Jacobi 迭代**：一次前向同时更新未来 W 个位置 —— 这些位置互为"猜测"，迭代几轮后会收敛出不少正确的 n-gram。这就把"串行生成"变成了"并行猜"。
2. **Lookahead 分支 + n-gram 池**：把迭代过程中出现过的高频 n-gram 收进池子，再从池子里挑候选去做**精确验证**（验证仍然走标准的拒绝采样，所以无损）。

三个参数对应论文的 W / N / G：

| 参数 | 含义 | 调大 | 调小 |
|---|---|---|---|
| `window=8` | 向前看多少步（窗口 W） | 并行度高，但无效计算多 | 省算力，一次能采到的 n-gram 少 |
| `ngram=5` | 收集的 n-gram 长度 N | 单次命中收益大，命中率降 | 命中率高，但收益小 |
| `gcap=8` | 候选池容量（guess pool cap） | 候选多，验证成本涨 | 验证便宜，但漏好候选 |

**为什么它特别适合端侧**：不需要第二个模型（省内存、$c$ 里没有模型成本）、不需要训练、不需要数据集。代价是**用 FLOPs 换步数** —— 而端侧 decode 的算力本来就闲着（note 03）。

**什么时候不划算**：算力已经吃紧时（NPU 利用率已经很高、或者在做 prefill），LADE 的额外 FLOPs 会直接变成时间。论文的加速比是 MT-bench 最高 1.8×、代码补全多卡 4×。

### 4.2 SSD = Self-Speculative Decoding（不需要独立模型文件）

```python
SsdBuilderConfig(
    forecast_token_count=4,   # 一次预测 4 个未来 token
    forecast_prefix=16,       # 用最近 16 个位置的 hidden 当 forecast 的输入
    branches=[4, 4],          # 两层候选树：第一层 4 个，每个再分 4 个
    ssd_tensor_file="./ssd_tensor.pt",   # 训好的 forecast 头权重
)
```

**和 Eaglet 的本质区别是 draft 从哪来**：

- **Eaglet**：外挂一个独立 draft 模型（独立 ONNX、独立量化、独立校准）→ 灵活但多一份内存和 $c$；
- **SSD**：模型**内建** forecast 头（权重就是一个 pt）→ $c$ 更小、没有第二个模型的调度开销，但**换 target 就要重训**。

`branches=[4,4]` 是一棵**候选树**：根 → 4 个孩子 → 每个孩子再 4 个，共 20 个候选位置。树的意义和 EAGLE 一样 —— **draft 越往后越不准，用分支覆盖多种可能**。代价是验证时要一次算 20 个位置（但 note 03 说过：带宽墙下验证多个位置的边际成本很低）。

### 4.3 Eaglet = EAGLE-Based Decoding

```python
EagletBuilderConfig(
    draft_len=6,                 # γ：draft 出 6 个 token
    n_branches=6,                # 树宽：每个位置展开 6 个候选
    draft_model_path="./draft_model.onnx",
    draft_token_map="./draft_token_map.json",   # ← 最容易被忽略的参数
)
```

**`draft_token_map.json` 是干嘛的？** 它是 **draft 词表 → target 词表** 的映射表。

为什么会有不一致：
1. draft 常用**裁剪词表**（只留高频 token）省 embedding 和 LM head 的带宽；
2. draft 和 target 的 tokenizer 版本不同；
3. 多模态模型里 draft 只覆盖文本 token。

vLLM 的 EAGLE-3 文档明确写了支持 **cross-tokenizer vocabularies**，端侧把这个能力落成了一个 JSON。

**这是一个"看起来是算法、其实是工程"的点**：映射表错了不会崩，只会让 α 掉一半 —— 和 note 04 里"RMSNorm 少了 +1"是同一类坑：**静默错误，最难查。**

**和 vLLM 版 EAGLE-3 的对照（服务端 vs 端侧）**：

| 维度 | vLLM EAGLE-3 | QAIRT-dev Eaglet |
|---|---|---|
| draft 结构 | Llama 风格 1 层 + FC 投影 + LM head | ONNX 图（结构由导出决定） |
| 训练目标 | 与 target logits 的 KL | 同（离线训好再导出） |
| 验证单位 | 自回归 K 个 → 取最长正确前缀 | `n_branches=6` 的树 |
| 词表 | 支持 cross-tokenizer | `draft_token_map.json` |
| **形状** | **运行时参数** | **编译期常量**（改 γ 要重新 build） |

最后一行是端侧与服务端最大的差别，也是面试最容易被问的点：**在 NPU 上"调参"意味着重新编译。**

### 4.4 双图构建：prefill 和 decode 是两张图

这段是 QAIRT-dev 里最硬的工程细节，直接决定投机解码能不能跑出性能。

```python
# 父 builder 管一张图（通常是 prefill）
child = builder.attach_model_for_arn(arn=1, model_path="ar1_model.onnx", name="decode")

# 子图继承父的编译/转换/校准/量化配置，权重与父共享
child.set_compilation_options(options={"graphs.vtcm_size_in_mb": 8})        # 只改 graph 级
child.set_transformation_options(options={"mha2sha.m2s_additional_start_points": [...]})

builder.sync_attached_builder_options()   # build 前强制校验 context/device/compiler/backend 一致
```

**为什么必须两张图**：prefill 的 M 大、decode 的 M=1，两者的最佳 tiling、VTCM 用量、甚至 kernel 选择都不同（note 03：一个 compute-bound、一个 memory-bound）。**用一张图跑两个阶段，必然有一个是浪费的。**

**为什么只能改 graph 级选项**：父子**共享权重**。context / device / compiler / backend 一变，权重的布局和编码就对不上 → 所以 `sync_attached_builder_options()` 在 build 前强制校验。**这条约束是"共享权重"的直接推论，不是设计者的偏好。**

**和投机解码的关系**：draft 图（`draft_len=6`）和 verify 图（1 个 token）shape 不同，但**KV cache 必须共享**。如果两张图的 KV 布局不一致，每步都要拷贝一次 —— 投机解码的收益直接被吃光。**这就是为什么"能跑通"和"跑得快"是两件事。**

### 4.5 三条方案怎么选

```
要极致精度、且愿意训一个 draft？ ── 是 → Eaglet（唯一可训练调优，代价是多一份模型）
   │否
   ▼
不想维护第二个 ONNX、target 又固定？ ── 是 → SSD（内建 forecast 头，换模型要重训）
   │否
   ▼
不想训练、不想加模型、decode 算力还有余量？ ── 是 → LADE（零训练零模型，用 FLOPs 换步数）
```

**三条方案的共同点**（面试可以一句话总结）：**都是"猜 + 验"，都走拒绝采样保证无损，都要在静态图里预留固定的候选槽位。**

### 4.6 另一条产品线：QEfficient（Cloud AI）的 SpD / PLD

容易混淆的是：高通有**两条**推理产品线，投机解码的实现方式不同。

| 产品线 | 目标硬件 | draft 路径 |
|---|---|---|
| **QAIRT-dev GenAI API**（端侧，你在用的） | HTP | LADE / SSD / Eaglet |
| **QEfficient**（Cloud AI 100） | 云端加速卡 | TLM（一次返回多个 logits）、multiprojection heads、**PLD** |

**PLD（Prompt Lookup Decoding）值得单独记**：从 prompt 里做 n-gram 匹配直接抄候选，**零参数、零训练、$c=0$**。在**复制型任务**（摘要、长文翻译、带引用的 RAG、代码补全）上 α 很高，开放式创作几乎没用。**端侧最便宜的加速手段，很多场景被严重低估。**

---

## 5. DeepSeek 的 DSpark：把"验证预算"也做成自适应的

> 来源：DeepSeek 2026-06 随 V4 发布的 DSpark（技术报告《DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation》）+ 开源栈 DeepSpec。以下是公开报道里的要点。

### 5.1 它想解决的两个痛点

| 痛点 | 现象 | DSpark 的解法 |
|---|---|---|
| **接受率随位置衰减** | 并行 draft 越往后越不准（后面的 token 没有依赖前面的） | **半自回归生成**：保留并行 draft 的高吞吐，但加一个轻量串行模块建模 block 内 token 之间的依赖 |
| **验证预算被浪费** | 高负载时，把注定被拒的尾部 token 也送进验证，白烧算力 | **置信度调度验证**：一个 confidence head 预测每个 token 的存活概率，配合硬件感知的前缀调度器，**按实时吞吐给每个请求定验证长度** |

这两个痛点，其实就是 §3.3 里 EAGLE-3 也在打的同一场仗 —— 只不过 EAGLE 从"训练"侧下手，DSpark 从"调度"侧下手。**这是当前投机解码最前沿的两个方向：让 draft 更准，和让验证更省。**

### 5.2 工程细节（对做部署的人最有价值）

- **异步调度**：兼容零开销调度（ZOS）和 CUDA graph replay，用**前两步**的历史预测来决定当前的截断长度，从而把调度延迟藏起来，避免流水线停顿。
- **无损**：保证目标模型的输出分布完全还原（和 §2.2 的拒绝采样一致）。

### 5.3 数字

| 对比 | 指标 | 结果 |
|---|---|---|
| vs EAGLE-3 | 平均接受长度（Qwen3 4B/8B/14B） | **+26.7% ~ +30.9%** |
| vs DFlash（并行草稿） | 平均接受长度 | +16.3% ~ +18.4% |
| vs 线上 MTP-1 基线 | 同等总吞吐下的生成速度 | **Flash +60%~85%，Pro +57%~78%** |

### 5.4 DeepSpec：把"训 draft"变成标准流程

开源的三段式流水线（数据准备 → 训练 → 评估），内置三种 draft：**DSpark / DFlash / EAGLE3**，目标模型支持 Qwen3 和 Gemma。

一个很真实的细节：**Qwen3-4B 的默认配置，target cache 就有 ~38 TB**，默认按单节点 8 卡跑。**这说明"训一个 draft head"根本不是小工程** —— 端侧团队想自研 draft，第一道坎是数据基建，不是算法。

> 面试可以说：**"投机解码的算法已经收敛了（拒绝采样 + feature-level draft），现在的差异全在工程：draft 训练的数据管线、验证预算的调度、以及和静态图的适配。"**

---

## 6. KV cache：端侧真正的第二瓶颈

### 6.1 先算账

note 02/03 已经算过：Qwen3.5-9B 级配置，4k 上下文 KV 是 **134 MB（fp16）**，32k 是 **1.07 GB**；`9B@4bit` 的权重是 4.5 GB。

| 上下文 | KV (fp16) | 权重 (W4) | KV 占总内存的比例 |
|---|---|---|---|
| 4k | 134 MB | 4.5 GB | 3% |
| 32k | 1.07 GB | 4.5 GB | 19% |
| 128k | 4.3 GB | 4.5 GB | **49%** |

**短上下文时 KV 可以忽略，长上下文时 KV 和权重一样大。** 而端侧的手机内存是 8~16 GB，**KV 直接决定"能不能跑长上下文"**。

### 6.2 三条 KV 优化路线

| 路线 | 做法 | 代表 | 有损? |
|---|---|---|---|
| **量化** | KV 存 int8 / int4 | KIVI、note 04 的 KV8 | 有损（要看模型动态范围） |
| **压缩/驱逐** | 只留"重要"的 token：attention sink + 滑窗、按累积 attention 分数淘汰 | StreamingLLM、H2O、SnapKV | 有损 |
| **复用** | 相同上下文不重算，直接复用 KV | prefix caching、CacheBlend、Mooncake/LMCache | **prefix 无损，non-prefix 有损** |

**优先级建议**：`复用 > 量化 > 压缩`。因为复用省的是**整个 prefill**，量化只省 KV 的字节。

### 6.3 prefix vs non-prefix 复用

- **prefix reuse（无损）**：新请求和缓存里的 prompt **开头**相同 → 前面的 KV 直接拿来用。因为 prefix 的 KV 不依赖后面的 token，所以数学上完全等价。vLLM 的 prefix caching、ChunkAttention 都是这个。
- **non-prefix reuse（有损）**：复用的是**中间某一段**（比如同一份文档被不同问题引用）。问题在于：这段 KV 是在"旧的上下文"里算出来的，新上下文里它的注意力依赖变了。

  **CacheBlend 的做法**：先在前几层比较"缓存 KV"和"重算 KV"的偏差 $|\Delta KV|$，选出偏差最大的 top-K 个 token **只重算这些**，其余复用。FLOPs 降到原来的 $r$（重算比例），精度损失很小。

> 为什么前几层就能判断？因为**KV 偏差在层间高度相关** —— 前几层偏得厉害的 token，后面也偏。

### 6.4 在移动 NPU 上做 KV 复用，难在哪（EuroSys'27 那篇的核心）

这是最近最贴你工作的一篇：*Dynamic Flow, Static Graph: KV Cache Reuse for Efficient LLM Serving on Mobile NPUs*（arXiv:2609.34727，EuroSys'27）。

**两个根本矛盾：**

1. **动态 vs 静态**：non-prefix 复用需要"运行时选哪些 token 重算"，而移动 NPU 只能跑**离线编译好的静态图**（Qualcomm HTP、MediaTek APU、Apple ANE 都一样）。GPU 是 SIMT + 动态 shape，天生能做；NPU 不行。
2. **KV 太大 vs 带宽太小**：见下表。

**端侧内存层级（论文 Table 1）**：

| 层级 | 容量 | 带宽 |
|---|---|---|
| Cache / TCM | 2–8 MB | — |
| LPDDR（CPU-NPU 共享） | 8–16 GB | **~60 GB/s** |
| Flash | 128 GB – 1 TB | **~500 MB/s** |

对比云端：HBM 1–3 TB/s、NVLink、RDMA。**端侧的 DRAM 带宽只有云端的 1/20，flash 更是 1/1000。**

**他们的四个设计（都可迁移到你的工作里）：**

| 设计 | 解决什么 |
|---|---|
| **chunk-and-pad + 有效性掩码 + 强制选最后一个 token** | 把"动态选 token"塞进固定 shape 的静态图 |
| **多图调度（动态规划）** | 运行时在若干张预编译图里挑一组、合并成一次调用，最小化 padding |
| **三级 KV 存储（NPU buffer / CPU DRAM / flash）+ 前缀树-哈希-语义混合索引** | 容量不够就下沉，但要能快速找回 |
| **cost-aware prefetch/eviction + 二维流水线** | 关键洞察：**reload 和 recompute 的代价不同**，要按"谁更便宜"来决定淘汰谁；同时把 KV 加载/重存和 NPU 计算重叠 |

**数字**：TTFT 降 **40%~60%**；prefill 加速 **1.5~5.4×**（vs 无复用）、**1.3~2.5×**（vs 只做 prefix caching）；同时避免了"直接全量 non-prefix 复用"带来的 **~30% 质量下降**。

**还有一个细节值得记**：KV 偏差 $\Delta KV$ 的动态范围是 **0 到 2000**，跨 prompt 分布差异极大 —— 所以**静态 int8/int16 量化会溢出/下溢**，必须混合精度。**"一个数值范围跨三个数量级"就是量化的死区**（呼应 note 04 的 clip 讨论）。

### 6.5 端侧 KV 的三个"反直觉"

1. **KV 复用不是免费的**：从 flash 读 1 GB KV 要 2 秒（500 MB/s），比重算可能还慢。**必须按"重算成本 vs 搬运成本"决策** —— 这正是 cost-aware eviction 的核心。
2. **prefix 复用的命中率取决于应用写法**：system prompt 放最前面 → 命中率极高；把时间戳/用户名放在开头 → 命中率归零。**这是产品设计问题，不是算法问题。**
3. **量化 KV 和复用 KV 会打架**：量化的 KV 复用时要先反量化，多一次搬 运；而复用的目的是少搬运。**两个优化叠在一起可能互相抵消。**

### 6.6 如果 `NativeKV` 指这三种之一

NativeKV是指KV以HMX Native的Data Format储存，可以省去一步转化，稳定提高在高通HMX上Prefill/long-context perf; 但总bytes不变，Decode阶段提升有限。

Qwen3一个模型的architecture，在Notebook2 change hardcoding中，一份weights，分成prefill decode两个graph share，一个可以AR128，一个可以AR1。而Split模型则纯粹因为VTCM的内存限制，与这个无关。NativeKV在Decode阶段，由于KV update不是32/128等Block，高通尚不支持，Genie fall back原本的KV。
 
---

## 7. 组合拳：一个端侧 decode 的完整加速栈

按"投入产出比"排序，也是我建议的落地顺序：

```
① 量化（W4A16）            → 权重字节 ÷4        ← note 04
② 融合 / 静态图            → 少搬运、少 launch   ← note 03
③ KV 量化（KV8）           → KV 字节 ÷2         ← note 04
④ 前缀复用                 → 砍掉重复 prefill    ← §6.3
⑤ 投机解码（LADE/SSD/Eaglet/PLD）→ 串行步数 ÷2~3 ← §2-§5
⑥ non-prefix KV 复用       → 长上下文场景才值得   ← §6.4
```

**三个叠加时的坑（面试加分项）：**

1. **量化之后要重调 γ**：target 变快了 4×，如果 draft 没同步量化，$c$ 相对变大 → 最优 γ 左移、加速比下降。**"量化 + 投机"不是简单相乘。**
2. **投机解码和 KV 复用作用于不同阶段**：投机省 TPOT，复用省 TTFT，**别指望一个指标同时变好**。
3. **静态图是天花板**：动态树（EAGLE-2）、动态重算（CacheBlend）、动态 batch 都要在"编译期把可能形状枚举出来"，否则一律 fallback 到 CPU —— **一次 fallback 就能把收益吃光。**

---

## 8. 面试问答（16 题）

**Q1：投机解码为什么能做到无损？**
因为用了拒绝采样：draft 出 $x$ 后以 $\min(1, p(x)/q(x))$ 接受；**拒绝时必须从 $\text{norm}(\max(0, p-q))$ 重采**，而不是重采 draft。这样才能保证最终分布严格等于 $p$。我仿真验证过：只接受不重采，TV 距离 0.36；正确重采后 0.003（和噪声同量级）。

**Q2：接受率怎么算？**
$\alpha = \sum_x \min(p(x), q(x)) = 1 - \mathrm{TV}(p,q)$。**不用仿真，这是个解析式。**

**Q3：加速比怎么估？最优 γ 是多少？**
$\text{speedup} = \frac{1-\alpha^{\gamma+1}}{(1-\alpha)(1+c\gamma)}$，$c$ 是 draft/target 的成本比。γ 太大时 draft 成本线性涨、收益对数涨，所以有峰值。我算过：α=0.8、c=0.1 时最优 γ=6、加速 2.47×。

**Q4：为什么端侧比云端更划算？**
端侧 batch=1 是带宽墙，验证 γ+1 个 token 权重只读一遍，**verify 边际成本≈0**；draft 又小（c≈0.05~0.1）。云端大 batch 是算力墙，验证真要多花算力，收益被吞吐摊薄。

**Q5：EAGLE 为什么在 feature 层做 draft，而不是 token 层？**
① feature 是连续的、信息更多，回归比分类平滑；② head 只有几十 M，c 小一个量级（而 c 决定收益上限）；③ 直接用 target 的 LM head，天然对齐，不用蒸馏。

**Q6：EAGLE-2 和 EAGLE-3 各改了什么？**
EAGLE-2：draft tree 从固定变动态，按 draft 的置信度决定展开哪些分支。EAGLE-3：多层 feature 融合（低层词法 + 高层语义）+ training-time test（训练时就模拟推理时的误差累积）。

**Q7：DSpark 解决什么？**
两个：① **接受率随 draft 位置衰减** → 半自回归生成，块内加轻量串行模块；② **验证预算浪费** → confidence head 预测每个 token 的存活率，配合硬件感知调度器按实时吞吐动态决定验证长度。另外它的调度是异步的，能藏延迟、不打断 CUDA graph。

**Q8：PLD 什么时候有用？**
从 prompt 里做 n-gram 匹配直接抄候选，$c=0$、零训练。在**复制型任务**（摘要、长文翻译、带引用的 RAG、代码补全）上 α 很高；开放式创作几乎没用。端侧最便宜的加速手段。

**Q9：为什么 KV cache 会成为端侧第二瓶颈？**
给数字：4k 上下文 134 MB、32k 1.07 GB、128k 4.3 GB，而权重 W4 才 4.5 GB —— 长上下文时 KV 和权重同量级。手机的 LPDDR 只有 ~60 GB/s，比云端 HBM 慢 20 倍。

**Q10：prefix 复用和 non-prefix 复用，为什么后者有损？**
prefix 的 KV 只依赖前面的 token，数学上等价；non-prefix 段是在旧上下文里算的，新上下文下注意力依赖变了。CacheBlend 的补救是：前几层比对缓存 KV 和重算 KV 的偏差，只重算偏差最大的 top-K token（利用"KV 偏差层间强相关"）。

**Q11：在移动 NPU 上做 KV 复用，最难的是什么？**
**动态 vs 静态**。non-prefix 复用要运行时选 token，而 NPU 只能跑编译期定 shape 的图（HTP/APU/ANE 都一样）。解法是把动态选择编码成"固定 shape + 有效性掩码"，再用多图调度（DP）减少 padding。其次是内存层级：KV 要到 flash，500 MB/s，**reload 可能比重算还慢**，所以要 cost-aware 决策。

**Q12：量化 + 投机一起用要注意什么？**
① 量化后 target 变快，draft 若没同步量化，$c$ 相对变大 → 最优 γ 左移、收益下降；② 两者都吃带宽，叠加不是相乘；③ draft 必须和 target 用同一套 encoding 语义，否则 logits 对不上、α 崩。

**Q13：QAIRT 里三种投机方案怎么选？**
**看 draft 从哪来**：`Eaglet` 要外挂 draft 模型（精度最好、可训练，但多一份内存和 $c$）；`SSD` 用模型内建的 forecast 头（省一个模型文件，但换 target 要重训）；`LADE` 完全不要模型（Jacobi 迭代 + n-gram 池，零训练，靠多花 FLOPs 换步数）。

**Q14：LADE 不需要 draft 模型，那它靠什么"猜"？**
靠 **Jacobi 迭代**：一次前向同时更新未来 W 个位置，这些位置互为猜测，迭代几轮会收敛出不少正确的 n-gram；再用一个 **lookahead n-gram 池**（容量 gcap）收集候选，最后走标准拒绝采样精确验证。参数 `window / ngram / gcap` 就是论文的 W / N / G。

**Q15：`draft_token_map.json` 是干什么的？为什么需要它？**
draft 和 target 的**词表可能不一致**（draft 常用裁剪词表省带宽、或 tokenizer 版本不同、或只覆盖文本 token）。它做 draft token id → target token id 的映射。vLLM 的 EAGLE-3 文档把这个能力叫 "cross-tokenizer vocabularies"。**坑在于：映射错了不会崩，只会让接受率掉一半** —— 属于静默错误。

**Q16：为什么 prefill 和 decode 要编译成两张图？为什么子图只能改 graph 级选项？**
prefill 的 M 大、decode 的 M=1，最佳 tiling / VTCM 用量 / kernel 选择都不同（一个是 compute-bound、一个是 memory-bound），一张图跑两个阶段必然有一个浪费。而父子图**共享权重**，context / device / compiler / backend 一变，权重的布局和编码就对不上，所以只有 `graphs.*`（如 VTCM 大小）能独立覆盖，`sync_attached_builder_options()` 会在 build 前强制校验。**投机解码尤其依赖这一点：draft 图和 verify 图 shape 不同，但 KV 必须共享同一套布局，否则每步都要拷贝。**

---

## 9. 动手实验

- [ ] **跑仿真**：`python tools/spec_decode_sim.py`，把 §2 的三张表换成你自己的数字。验收：能解释为什么"拒绝后必须重采"。
- [ ] **画一张 γ-α 热力图**：把 speedup 画成 (α, γ) 的二维图，标出每个 c 下的脊线。验收：能说出"α 低于多少时投机解码不值得做"。
- [ ] **真机测一次 PLD**：llama.cpp 支持 draft 模型（`-md`），拿一个带长文档引用的 prompt 跑一遍，记录接受长度。验收：给出"复制型任务 vs 创作型任务"的 α 对比。
- [ ] **算一次 KV 复用的账**：假设 system prompt 2k token、命中率 90%、手机 DDR 60 GB/s、prefill 200 tok/s。问：复用省多少 ms？从 flash 读 2k token 的 KV（fp16, 32 层, 4 kv head, 256 dim）要多少 ms？**两者谁大？**（这题能直接当面试题用）
- [ ] **读 Qwen3.5 的 KV 结构**：用 note 02 的配置算"每 token 每层 2048 元素 × 8 层"是从哪来的，再算 32k 上下文下 KV 占手机内存的百分比。

---

## 10. 参考

**投机解码**
- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*, ICML 2023（拒绝采样的原始证明）。
- Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling*, 2023。
- **EAGLE 家族**：EAGLE-1 [arXiv:2401.15077](https://arxiv.org/abs/2401.15077)（*Speculative Sampling Requires Rethinking Feature Uncertainty*）；EAGLE-2 arXiv:2406.16858；EAGLE-3 arXiv:2503.01840；代码 [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE)。
- **EAGLE-3 的工程实现（vLLM 视角，写得最具体）**：[Speculators — Eagle3](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/eagle3/)（多层 hidden 拼接 + FC + 1 层 Llama 风格 decoder + KL 训练 + 最长正确前缀）。
- **LADE**：*Break the Sequential Dependency of LLM Inference Using Lookahead Decoding*，[arXiv:2402.02057](https://arxiv.org/abs/2402.02057)；代码 [hao-ai-lab/LookaheadDecoding](https://github.com/hao-ai-lab/LookaheadDecoding)。
- **DSpark / DeepSpec**：[技术报告](https://github.com/deepseek-ai/DeepSpec/blob/main/DSpark_paper.pdf)、[仓库](https://github.com/deepseek-ai/DeepSpec)、[机器之心报道](https://eu.36kr.com/zh/p/3871135542416645)。
- **QAIRT-dev**：[Speculative Decoding](https://docs.qualcomm.com/doc/80-99100-3/topic/Speculative-Decoding.html) + [Speculative Decoding Tutorial](https://docs.qualcomm.com/doc/80-87189-2/topic/speculative_decoding_tutorial.html)（前端渲染，浏览器里看；LADE/SSD/Eaglet 的 config 来自你贴的文档）。
- **QEfficient（另一条产品线，Cloud AI）**：[仓库](https://github.com/quic/efficient-transformers)（README 的 news 里有 TLM / multiprojection heads 的时间线）。

**KV cache / 端侧部署**
- *Dynamic Flow, Static Graph: KV Cache Reuse for Efficient LLM Serving on Mobile NPUs*, [arXiv:2609.34727](https://arxiv.org/abs/2609.34727)（EuroSys'27）。
- CacheBlend (arXiv:2405.16444)、StreamingLLM (arXiv:2309.17453)、H2O (arXiv:2306.14048)、KIVI (arXiv:2402.02750)。
-高通 GenieX 双 runtime（llama.cpp / QAIRT）与 Q4_0-on-NPU 的约束：[GenieX 博客](https://qualcomm.github.io/qidk/blogs/geniex_android.html)。
- 本机脚本：`tools/spec_decode_sim.py`。

**待办**
- `NativeKV`：对应 §6.6 的 A/B/C 哪一种？给我一条链接或一页资料，我把那节扩成主体。
- `Eaglet`：已确认是 QAIRT-dev 的 EAGLE-Based Decoding（§4.3）。如果你那边还有内部的调参经验（`draft_len`/`n_branches` 怎么选、`draft_token_map` 怎么生成），补进来会更值钱。
