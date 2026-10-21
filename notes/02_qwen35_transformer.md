# 02 · Qwen3.5 在 transformers 里的实现（含端侧部署视角）

> 学习目标：能不看代码，把 Qwen3.5 的一层画出来，说清「哪些层是 full attention、哪些是 Gated DeltaNet、每个矩阵的形状、状态有多少字节」，并且知道这些结构在 HTP 上分别会变成什么 op、哪里最容易被量化搞坏。

---

## 0. 一句话结论

**Qwen3.5 是「3:1 混合注意力」的 decoder：每 4 层里有 3 层是 Gated DeltaNet（线性注意力，维护一个固定的 recurrent state），1 层是带输出门控的 GQA full attention；它没有 Qwen3 的 qkv bias，但多了一条 `q_proj` 的 gate 分支和「zero-centered RMSNorm（`*(1+w)`）」——这两个细节是移植时最容易静默出错的地方。**

---

## 1. 家族与配置（先看 config.json，别背我的数）

`transformers` 里对应的实现是 `src/transformers/models/qwen3_5/{configuration,modeling}_qwen3_5.py`（由 `modular_qwen3_5.py` 生成）。

**默认 `Qwen3_5TextConfig`（docstring 里称 "Qwen3.5-9B style"）**

| 字段 | 值 | 说明 |
|---|---|---|
| `vocab_size` | 248320 | |
| `hidden_size` | 4096 | |
| `intermediate_size` | 12288 | SwiGLU 的中间维 |
| `num_hidden_layers` | 32 | |
| `num_attention_heads` | 16 | |
| `num_key_value_heads` | 4 | GQA，group=4 |
| `head_dim` | 256 | 注意不是 hidden/heads=256，恰好相等 |
| `partial_rotary_factor` | **0.25** | 只对 64 维做 RoPE，其余 192 维直通 |
| `rms_norm_eps` | 1e-6 | |
| `attention_bias` | False | Qwen2 有 qkv bias，Qwen3/3.5 没有 |
| `tie_word_embeddings` | False | embed 与 lm_head 分开（约各 1.0B 参数） |
| `linear_conv_kernel_dim` | 4 | 线性注意力前的短卷积 |
| `linear_num_key_heads` / `linear_key_head_dim` | 16 / 128 | key_dim = 2048 |
| `linear_num_value_heads` / `linear_value_head_dim` | 32 / 128 | value_dim = 4096 |
| `full_attention_interval` | 4 | 决定 `layer_types` |

**层类型是算出来的，不是写死的**：

```python
self.layer_types = ["linear_attention" if (i+1) % 4 else "full_attention"
                    for i in range(num_hidden_layers)]
```

32 层 → full attention 落在 **index 3, 7, 11, ..., 31，共 8 层**；其余 24 层是 linear。

> ⚠️ 官方首发的是 **Qwen3.5-397B-A17B（MoE，17B 激活）**，和上面这个 dense 配置不是一套代码路径（dense 的 `Qwen3_5TextConfig` 里 `base_model_ep_plan = None  # no Moe`）。**你部署的到底是哪个尺寸、dense 还是 MoE，一定要看 checkpoint 的 `config.json`**；下面的结构分析对两者通用（GDN + full attention 的骨架是共享的），但 MLP 部分 MoE 版是 router + experts。

---

## 2. DecoderLayer：一个 block 只有 5 个节点

```python
residual = x
x = input_layernorm(x)                     # RMSNorm, zero-centered
x = token_mixer(x)                         # 二选一：linear_attn 或 self_attn
x = residual + x
residual = x
x = post_attention_layernorm(x)            # RMSNorm
x = mlp(x)                                 # SwiGLU
x = residual + x
```

标准的 pre-norm 残差结构，**没有 bias，没有 dropout（推理），没有 LayerScale**。所以一层图上是：

```
RMSNorm → (GDN | GQA+RoPE+gate) → Add → RMSNorm → SwiGLU(gate,up,down) → Add
```

**MLP**（SwiGLU，与 Llama 相同）：

```
down( silu(gate(x)) * up(x) )
gate/up: 4096 → 12288     down: 12288 → 4096      (共 3 × 50.3M = 151M 参数/层)
```

---

## 3. Full attention 层：GQA + 输出门控 + 部分 RoPE

```python
self.q_proj = Linear(4096, 16*256*2)   # 8192 = query + gate
self.k_proj = Linear(4096, 4*256)      # 1024
self.v_proj = Linear(4096, 4*256)      # 1024
self.o_proj = Linear(16*256, 4096)
self.q_norm = RMSNorm(256)   # per-head
self.k_norm = RMSNorm(256)   # per-head
```

forward 里最关键的三行：

```python
query_states, gate = torch.chunk(
    self.q_proj(h).view(*shape, -1, self.head_dim*2), 2, dim=-1)
...
attn_output = attn_output * torch.sigmoid(gate)      # 输出门控
attn_output = self.o_proj(attn_output)
```

三个必须记住的点：

1. **`q_proj` 一次算出 q 和 gate，且是按 head 交错存放的**（`view(..., -1, 512)` 再 chunk）。如果你按「前 4096 是 q、后 4096 是 gate」去切片，会全错 —— 必须按 `[head, 512]` 再拆。
2. **`q_norm` / `k_norm` 是 per-head 的 RMSNorm（作用在 256 维上）**，位置在 RoPE 之前。这是 Qwen3 以来的传统（Qwen2 没有）。
3. **RoPE 只转前 64 维**（`partial_rotary_factor=0.25`），后 192 维原样拼回去：

```python
rotary_dim = cos.shape[-1]                       # 64
q_rot, q_pass = q[..., :64], q[..., 64:]
q_embed = cat([q_rot*cos + rotate_half(q_rot)*sin, q_pass], -1)
```

> 部署提醒：部分 RoPE 意味着你**不能直接复用 Llama 的 RoPE kernel**，要把「rotate 前 64 维 + 其余拷贝」写成一个融合 op，否则会多两次全量读写。

---

## 4. Linear attention 层：Gated DeltaNet（重点，也是难点）

### 4.1 模块结构

| 组件 | 形状（默认配置） | 作用 |
|---|---|---|
| `in_proj_qkv` | 4096 → 2·2048 + 4096 = **8192** | 一次算出 q,k,v |
| `conv1d` | depthwise, in=out=8192, **k=4**, groups=8192 | 短卷积，带因果 padding |
| `in_proj_z` | 4096 → 4096 | 输出门控（gate） |
| `in_proj_b` | 4096 → 32 | β（写入门控） |
| `in_proj_a` | 4096 → 32 | 衰减 g 的输入 |
| `A_log` / `dt_bias` | 32 / 32 | 可学习衰减参数 |
| `norm` | Gated RMSNorm(128) | 用 z 门控的归一化 |
| `out_proj` | 4096 → 4096 | |

**forward 的关键顺序**（顺序错了精度就崩）：

```python
mixed_qkv = in_proj_qkv(x).transpose(1,2)          # [B, 8192, L]
mixed_qkv = causal_conv1d(mixed_qkv, w, act="silu")# depthwise, k=4, 因果
q, k, v = split(mixed_qkv, [2048, 2048, 4096])
beta = sigmoid(in_proj_b(x))
g    = -exp(A_log) * softplus(in_proj_a(x) + dt_bias)   # ≤ 0，log 空间的衰减
q, k = repeat_interleave(q, 2), repeat_interleave(k, 2) # 16 → 32 heads (GQA 式扩展)
core = gated_delta_rule(q, k, v, g, beta, state)
core = gated_rmsnorm(core, z)      # 用 z 做门控
out  = out_proj(core)
```

### 4.2 递推公式（白板必须能写）

state `S ∈ R^{d_k × d_v}`（每个 v-head 一个，`128 × 128`）：

```
S_t = exp(g_t) · S_{t-1} + k_t · ( β_t ⊙ (v_t − S_{t-1}ᵀ k_t) )ᵀ
o_t = S_tᵀ q_t
```

三项含义：**遗忘**（`exp(g_t)` 衰减）、**纠错**（`v − Sᵀk` 是「当前状态对 v 的预测残差」，delta rule 只写残差）、**写入**（`β_t` 控制写多狠）。`q,k` 先做 L2 归一化，`q` 再除以 `√d_k`。

### 4.3 训练用 chunked，解码用 recurrent

```python
if seq_len == 1 and has_previous_state:
    torch_recurrent_gated_delta_rule(...)     # 逐 token，O(d_k·d_v)
else:
    torch_chunk_gated_delta_rule(..., chunk_size=64)  # 块内用矩阵乘
```

`chunk` 版的核心技巧是 **UT transform（WY 表示）**：块内 64 步的递推等价于解一个下三角方程组（`solve_triangular`），把「64 次串行更新」变成「几次矩阵乘 + 一次三角求解」，从而能吃满 GEMM。块间仍然是串行的 scan。

> 这就是你部署时最该关心的东西：**prefill 走 chunked（matmul 密集、可并行），decode 走 recurrent（逐 token、纯带宽）**。NPU 对「串行 scan + 三角求解」都不友好，通常是整块 offload 到 CPU 或写专用的固定形状 kernel。

---

## 5. RMSNorm 是「zero-centered」的（最容易踩的坑）

```python
self.weight = nn.Parameter(torch.zeros(dim))     # 注意：初始化为 0
output = x * rsqrt(mean(x²) + eps) * (1.0 + self.weight)
```

也就是 **`y = norm(x) · (1 + w)`**，而不是 Llama 的 `y = norm(x) · w`。

后果：
- 如果你把权重按 Llama 的方式直接搬进推理引擎，**权重差 1，输出全错但不会崩**（数值只是偏了），极难 debug。
- 转换脚本里必须显式做 `w' = 1 + w`，或者引擎里加一个 `+1` 的常量。
- 这个 norm 出现在 4 个地方：`input_layernorm`、`post_attention_layernorm`、`final norm`、`q_norm`/`k_norm`、以及 GDN 里的 `RMSNormGated`。

---

## 6. M-RoPE：位置编码是 3 维的

- `position_ids` 形状 `(4, B, L)`：**第 0 行是纯文本位置**（只用于 mask / 增量生成的偏移），**第 1~3 行是 T/H/W** 三个轴。
- 文本 token：三行相同；图像/视频 token：按 patch grid 分别给 T、H、W 递增的坐标。
- `mrope_section = [11, 11, 10]`：在 32 个频率槽里按 `(T,H,W)` 循环交错取（`for dim, offset in ((1,1),(2,2))`）。
- 频率最后 `cat([hw, hw])` 拼成 64 维 → 正好等于 `head_dim × 0.25`。

**纯文本推理可以退化成普通 1D RoPE**（三行相同），但**多模态输入必须实现完整的 mrope**，否则图像位置全乱。

---

## 7. 视觉塔（如果要部署 VL）

| 项 | 值 |
|---|---|
| patch embed | `Conv3d(3, 1152, kernel=[2,16,16], stride=[2,16,16])` |
| 深度 | 27 个 block，hidden 1152，16 头（head_dim 72） |
| qkv | **fused 且带 bias**（和语言模型相反） |
| 位置编码 | 2D axial RoPE（head_dim 的 1/2 给 H，1/2 给 W） |
| 下采样 | `spatial_merge_size = 2` → 4 个 patch 拼 1 个 token |
| 投影 | PatchMerger: LayerNorm → Linear(4608) → GELU → Linear(3584) |

部署含义：一个 448×448 的图 = 28×28 patch / 4 = **196 个视觉 token**；1024×1024 = 64×64/4 = **1024 个 token**。视觉 token 数是端侧 VLM 的第一成本项。

---

## 8. 状态与带宽：用数字说话（默认 32 层配置，fp16）

### 8.1 Full attention 的 KV cache

每 token 每层：`2 × 4 heads × 256 = 2048` 元素；8 层合计 **16384 元素/token**。

| 上下文 | KV 总大小（fp16） | int8 |
|---|---|---|
| 1k | 33.6 MB | 16.8 MB |
| 4k | 134 MB | 67 MB |
| 32k | 1.07 GB | 537 MB |

注意：**只有 1/4 的层有 KV**，这本身就是相对 Qwen3 的 4× 节省。

### 8.2 Gated DeltaNet 的固定状态

每层：`conv_state` = `8192 × 3` 元素（k=4 缓存前 3 个），`recurrent_state` = `32 × 128 × 128 = 524,288` 元素 ≈ **1 MB (fp16) / 2 MB (fp32)**。

24 层合计：**≈ 25 MB (fp16) / 50 MB (fp32)**，**与序列长度无关**。

### 8.3 于是有一个反直觉的结论

| 上下文长度 | decode 每步要读的状态（fp16） |
|---|---|
| 512 | KV 16.8 MB vs GDN 状态 25 MB → **线性层更亏** |
| ~768 | 两者打平 |
| 4096 | KV 134 MB vs GDN 25 MB → **线性层赢 5×** |

**所以「linear attention 省内存」是有前提的：序列足够长。** 端侧短上下文（比如 512 token 的语音助手）用 GDN 反而吃带宽。这个数字在面试里非常加分，因为它说明你不是在背结论。

---

## 9. 部署视角：op 清单与量化敏感点

| 结构 | 对应 op | HTP 上的坑 |
|---|---|---|
| Linear / MLP | GEMM（weight-only W4/W8 友好） | 常规，注意 per-channel 对齐 |
| RMSNorm(`1+w`) | reduce-mean + rsqrt + scale | 必须 fp32 算 variance；`+1` 别漏 |
| partial RoPE | 旋转前 64 维 | 要和后续 GEMM 融合，否则多一次读写 |
| GQA attention | QKᵀ → softmax → PV | 长上下文时 softmax 是第二瓶颈 |
| sigmoid gate | elementwise | 融合进 o_proj 的 epilogue |
| **depthwise conv1d (k=4)** | 1D 深度卷积 | 只有 4 个 tap，极度 memory-bound；很多 NPU 的 DW conv 只支持 2D，需要 reshape 成 H=1 |
| **gated delta rule** | 串行 scan / 三角求解 | **最难的 op**：NPU 无原生支持；prefill 可拆成 matmul，decode 只能写出固定形状的专用 kernel 或放 CPU |
| gated RMSNorm | reduce + silu(z) | 融合掉 |

**量化敏感度排序（我的经验 + 结构推理）：**

1. `g`（衰减）与 `β`：必须 fp32 算 `exp/softplus/sigmoid`，量化到 int8 会直接把长期记忆搞坏。
2. `recurrent_state`：跨 token 累加，误差会累积 → 保持 fp16 或 fp32，别量化。
3. `q_norm`/`k_norm` 与 partial RoPE：低比特下旋转角的误差会被 attention 放大。
4. 视觉塔的 Conv3d + LayerNorm：LayerNorm 的 bias 对量化不友好。
5. MLP / o_proj：最安全，优先压到 4bit。

---

## 10. 面试问答

**Q1：Qwen3.5 和 Qwen3 最大的区别？**
三层：① 3:1 混合注意力（Gated DeltaNet + GQA），长上下文更省；② 去掉 qkv bias，q_proj 增加输出 gate，RMSNorm 改成 zero-centered；③ 原生多模态（视觉塔 + M-RoPE），不再是外挂 adapter。

**Q2：为什么是 3:1 而不是全 linear？**
纯线性注意力在「精确回忆」（needle-in-haystack、复制、长距离精确对齐）上弱于 full attention；保留 1/4 的 full attention 层，相当于给模型留了一条「精确检索通道」，代价是 KV 只增加 1/4。工程上 3:1 是精度-成本的甜点。

**Q3：Gated DeltaNet 和 Mamba2 的关系？**
结构同源（都来自 SSM/线性注意力），GDN 在 Mamba2 的「门控衰减」上加了 delta rule：写入的不是 `v`，而是 `v − Sᵀk` 的残差。这让状态更新有「纠错」能力，等价于在线做一次梯度步（所以叫 delta rule）。

**Q4：复杂度对比？**
full attention：训练 O(L²d)，decode 每 token O(Ld) 且要读全部 KV。GDN：训练 O(L·d_k·d_v)（chunked 后是 matmul 密集），decode 每 token O(d_k·d_v) 且读固定大小状态。在 L=4096 时单层 decode 的 MAC 差约 30 倍（1.05M vs 33.5M）。

**Q5：为什么 chunked 版能加速？**
把块内 64 步的串行递推用 UT transform 变成「解一个 unit lower triangular 方程」，从而变成矩阵乘 + 三角求解，能吃满 tensor core / MAC array；块间只剩 L/64 次串行 scan。

**Q6：`g = -exp(A_log)·softplus(a + dt_bias)` 里每个部分的作用？**
`A_log` 保证衰减系数为正（`exp` 后），`softplus` 保证输入非负、且处处可导（`dt` 的离散化），负号让 `g ≤ 0` 从而 `exp(g) ∈ (0,1]` 是一个合法的遗忘因子。`dt_bias` 给初始时间步一个偏置，防止训练初期衰减过快。

**Q7：KV cache 多大？**
`2 × n_kv × head_dim × n_full_layers × L × bytes = 2×4×256×8×L×2`。4k 上下文 = 134 MB fp16。只有 1/4 层有 KV 是关键。

**Q8：为什么 q_proj 输出 2 倍？多出来的 gate 有什么用？**
对 attention 输出做 `sigmoid` 门控（类似 Qwen3-Next 的 gated attention）。作用是抑制 attention sink 和 massive activation，提升训练稳定性；代价是推理多一个 elementwise mul 和一个 8192 宽的 GEMM 输出。

**Q9：移植时最容易出的三个 bug？**
① RMSNorm 少了 `+1`；② q/gate 按 head 交错而不是前后切半；③ RoPE 只该转 64 维却转了全部 256 维。三个都不会崩，只会「效果变差」，所以必须用逐层对齐（和 HF 的 hidden_states 逐层比 max abs error）。

**Q10：如果让你把它压到手机上，你的顺序是什么？**
① 先跑通 fp16，逐层对齐；② 量 MLP/o_proj 到 4bit（W4A16），保 GDN 的状态与 g/β 为 fp32；③ 把 conv1d/RMSNorm/gate 全部融合；④ KV 只做 int8 或干脆不做；⑤ 用 profile 找剩下的 memory-bound op；⑥ 最后才考虑改 chunk size / 换 kernel。

---

## 11. 自测（不看上文能答出来吗）

1. 32 层里第几层是 full attention？为什么是这些层？
2. 画出 GDN 一层的 8 个矩阵，标出每个的输入输出维度。
3. 写出 delta rule 的递推式，并解释「为什么写残差而不是写 v」。
4. `layer_types` 是怎么算出来的？换成 28 层会怎样？
5. 上下文 512 时，KV 和 GDN 状态哪个更占带宽？给出数字。

---

## 12. 参考资料

- HF 实现：`transformers/models/qwen3_5/modeling_qwen3_5.py`、`configuration_qwen3_5.py`（注意文件头写着由 `modular_qwen3_5.py` 生成，改代码要改 modular）。
- Gated DeltaNet 原文：*Gated Delta Networks: Improving Mamba2 with Delta Rule*，arXiv:2412.06464。
- Qwen3-Next 的技术说明（3:1 混合的出处）。
- 社区解读：[Qwen3.5: Nobody Agrees on Attention Anymore](https://huggingface.co/blog/mlabonne/qwen35)（只当索引看，架构以代码为准）。
- llama.cpp 侧：`qwen3next` 那套 `ssm_conv` + `gated_delta_net` 的实现，是「同一个数学在 C 里怎么写」的最好对照。
