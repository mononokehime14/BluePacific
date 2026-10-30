# 06 · 算子优化入门：从 ggml 的 OpenCL / Hexagon kernel 读起

> 目标读者：没写过 GPU/NPU kernel，但会 C、懂 tensor 是什么、能读开源代码。
> 读完你应该能做到三件事：① 看到一个 op，说清它被拆成了什么 kernel、为什么这么拆；② 读一个真实的 kernel 不迷路；③ 遇到"我的 kernel 为什么慢/为什么没被选中"有固定套路去查。
>
> 代码依据：`llama.cpp` master 的 `ggml/src/ggml-opencl/` 与 `ggml/src/ggml-hexagon/`（我逐个文件抓下来读的，关键符号都给了原文）。

---

## 0. 一句话结论

**kernel 之所以快，不是因为它"写得巧妙"，而是因为「数据的形状」在它执行之前就已经被安排好了**：权重离线 repack 成硬件喜欢的 tile 布局，激活在片上现场量化成 kernel 想要的格式，DMA 提前把下一个 tile 搬进来，线程只做纯计算。

所以算子优化的本质是一句话：**让数据以 kernel 想要的形状，出现在它想让它出现的地方。** 你那次把 QK 改成 contiguous，做的就是这件事 —— 你不是"优化了一个 kernel"，你是**让一段数据满足了某个快 kernel 的入场条件**。

---

## 1. 心智模型：一个 kernel 到底在干什么

### 1.1 SIMT：一条指令，很多个"我"

GPU/NPU 不是"一个很快的核"，而是"很多个不太快的核，共用一条指令流"。

```
一次 kernel 启动
 ├── work-group 0        ← 一组 work-item，能共享 __local 内存、能同步
 │    ├── work-item 0    ← 一个"线程"，有自己的寄存器
 │    ├── work-item 1
 │    └── ... (通常 64~256 个)
 ├── work-group 1
 └── ...
```

三个词要分清（OpenCL 叫法 / CUDA 叫法）：

| OpenCL | CUDA | 是什么 |
|---|---|---|
| work-item | thread | 最小的执行体，有私有寄存器 |
| sub-group / wavefront | warp | **硬件真正同时执行的一批**（Adreno 通常 64，NVIDIA 32） |
| work-group | block | 一组 work-item，可共享 `__local` 内存 + 可 barrier |

**为什么这个结构决定了性能**：同一个 wavefront 里的 32/64 个 work-item 是**锁步（lock-step）**执行的 —— 它们执行同一条指令，只是数据不同。所以：

- **如果它们访问的内存是连续的** → 硬件一次就能取回来（合并访问 / coalesced）；
- **如果它们访问的内存是散开的** → 硬件要发很多次请求，每次只用到一小部分（非合并访问 / uncoalesced），带宽利用率可能掉到 1/8。

这一条就是"为什么 layout 决定性能"的最底层原因。

### 1.2 三种内存，带宽差两个数量级

| 内存 | OpenCL 写法 | 典型容量 | 相对延迟 | 谁管 |
|---|---|---|---|---|
| 私有 | 普通变量 | 每线程几十~几百个寄存器 | 1 | 编译器 |
| 片上共享 | `__local` | 16~64 KB / work-group | ~20 | 你手动管 |
| 全局 | `__global` | GB 级 | ~500 | 你手动管（+ cache） |

**所有 kernel 优化的核心矛盾**：全局内存太慢，所以要想办法**重复利用已经搬到片上/寄存器里的数据**（这就是 tiling 的全部意义）。

### 1.3 合并访问长什么样

假设 32 个 work-item 要读一个 32×32 的矩阵的一列（列优先存储，行是连续的）：

```
好的情况（连续读）：  item0 读 [0], item1 读 [1], item2 读 [2] ...
  → 一次 128B 的事务就能服务 32 个 item

坏的情况（跨步读）：  item0 读 [0], item1 读 [32], item2 读 [64] ...
  → 每个 item 落在不同的 128B 行上 → 32 次事务，每次只用 4B
  → 有效带宽 = 1/32
```

**这就是为什么"转置"在 kernel 里是一个专门的优化话题**：要么用 `__local` 中转，要么干脆在数据布局上就别转（pre-transpose）。

### 1.3.1 本机实测：跨步到底贵多少

`tools/layout_cost.py`（纯 numpy，30 秒跑完），8192×8192 f32：

| 实验 | 时间 | 有效带宽 | 结论 |
|---|---|---|---|
| 连续读 `x[:, :4096]` 拷出 134 MB | 13.4 ms | 10.0 GB/s | 基线 |
| 跨步读 `x[:, ::2]` 拷出同样 134 MB | 21.9 ms | 6.1 GB/s | **慢 1.63×** |

搬的字节数完全一样，只是"每个 lane 隔一个读" —— 一半的 cache line 被浪费掉。

**但有一个反例必须记（我自己先踩了）**：

| 实验 | 时间 | 带宽 |
|---|---|---|
| `sum(axis=1)`（连续方向） | 22.8 ms | 11.8 GB/s |
| `sum(axis=0)`（跨步方向） | 13.4 ms | 20.0 GB/s |

**结果反了** —— 因为 numpy 对两个 axis 用了不同实现：`axis=0` 每行调一次向量化内积，`axis=1` 走逐元素累加。**实现质量盖过了访存模式。**

> 这个反例值得贴在桌上：**布局的收益只在同一套实现内部比较才有意义**。跨实现比（numpy vs BLAS、C 代码 vs SASS）会被"向量化程度"污染。这也是为什么 kernel 优化最终要看反汇编，而不是看 C 代码猜。

### 1.4 一个 op 的三种成本

任何 kernel 的时间都能拆成：

```
T = max( 计算时间, 访存时间 ) + 同步/启动开销
计算时间 = FLOPs / 峰值算力
访存时间 = Bytes / 峰值带宽
```

这就是 note 03 的 roofline，只不过从"op 级"下沉到了"kernel 级"。**动手前先算这两个数**，能省掉 80% 的瞎调。

---

## 2. ggml 的契约：tensor 是怎么描述的

### 2.1 `ne[]` 和 `nb[]`：后者才是真相

```c
struct ggml_tensor {
    enum ggml_type type;
    int64_t ne[4];   // 每一维有多少元素
    size_t  nb[4];   // 每一维跨多少字节（stride，单位是 byte）
    void *  data;
    ...
};
```

- `ne[0]` 是最内层（连续）维度。
- **`nb[]` 才是内存布局的真相**：`nb[0]` 是相邻元素的间距。

一个"连续"的 f32 tensor 有 `nb[0] == 4`、`nb[1] == 4*ne[0]`、`nb[2] == nb[1]*ne[1]`。
一个"不连续"的 tensor（比如 permute/reshape 之后的视图）会不满足这个关系。

**你那次优化的全部秘密就在这一行判断里**：kernel 想用向量化加载（一次读 4 个 f32 / 8 个 f16），它必须相信 `nb[0] == sizeof(type)`。如果 `nb[0]` 是别的值，它只能走标量慢路径，或者干脆不支持。

> 记忆点：**`ne` 说"有多少"，`nb` 说"怎么摆"。快 kernel 只吃特定的 `nb`。**

### 2.2 后端要实现的三个接口

一个 ggml backend（OpenCL、CUDA、Hexagon 都一样）本质上只干三件事：

```c
supports_op(op)          // 这个 op 我能不能接？（不能就回落到 CPU）
compute_forward(tensor)  // 怎么算这个 op？（内部选 kernel）
buffer_type              // 内存怎么分配？（对齐、host-visible、共享内存…）
```

**`supports_op` 是分层的**：它不只判断"我有没有这个 op"，还判断"这个 op 的**这个形状/这个 dtype/这个布局**我能不能接"。这就是为什么同一个 op 在不同 shape 下可能落到不同后端 —— 也是为什么你改个 layout 就能改变路径。

### 2.3 谁决定 op 跑在哪

`ggml_backend_sched` 把整张图按"每个 op 的支持情况"切成段，段之间插入拷贝。**一次跨后端的拷贝可能比 op 本身还贵**（note 03 讲过）。所以"我的 kernel 没被选中"往往不是 kernel 的问题，而是图被切碎了。

---

## 3. 解剖 OpenCL 后端（llama.cpp）

### 3.1 代码地图

| 文件 | 里面有什么 |
|---|---|
| `ggml/src/ggml-opencl/ggml-opencl.cpp` | 后端主体：设备探测、kernel 编译与缓存、每个 op 的 `compute_forward`、profiling |
| `ggml/src/ggml-opencl/kernels/*.cl` | OpenCL C 源码（`add.cl`、`cvt.cl`、`cpy.cl`、`mul_mm*.cl` …） |
| `ggml/src/ggml-opencl/cl-program-cache.*` | 编译产物缓存（避免每次启动重新编译） |

编译时，`.cl` 会被嵌进二进制（代码里是 `#include "add.cl.h"`），运行时不读文件。所以**你改 kernel 必须重新编译 llama.cpp**，改不了"运行时替换"。

### 3.2 一个 kernel 的诞生：从字符串到 `cl_kernel`

后端初始化的代码长这样（原文结构，做了删减）：

```c
{
    const std::string kernel_src {
        #include "add.cl.h"          // ← 编译期就把 .cl 的文本嵌进来了
    };
    backend_ctx->program_add =
        build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

    CL_CHECK((backend_ctx->kernel_add         = clCreateKernel(backend_ctx->program_add, "kernel_add", &err), err));
    CL_CHECK((backend_ctx->kernel_add_row     = clCreateKernel(backend_ctx->program_add, "kernel_add_row", &err), err));
    CL_CHECK((backend_ctx->kernel_add_f16     = clCreateKernel(backend_ctx->program_add, "kernel_add_f16", &err), err));
    CL_CHECK((backend_ctx->kernel_add_row_f16 = clCreateKernel(backend_ctx->program_add, "kernel_add_row_f16", &err), err));
}
```

**四个要点**：

1. **一个 `.cl` 文件可以导出很多 entry point**（`kernel_add` / `kernel_add_row` / `kernel_add_f16` / …）。这些名字就是后端在 dispatch 时挑的"骡子"。
2. `program` 是编译单元，`kernel` 是可调用的具体函数。`program` 只建一次，`kernel` 每个 entry 一个。
3. 命名约定非常直白：`kernel_<op>_<变体>`，变体里通常编码了 dtype 和布局（`_f16`、`_row`、`_nc` = non-contiguous?）。
4. **所有 kernel 的句柄都挂在 backend_ctx 上**，dispatch 时就是"从结构体里挑一个指针"。

### 3.3 kernel 注册表：一个 op 为什么有一堆 kernel

看真实的结构体字段（`ggml-opencl.cpp` 里 `struct ggml_backend_opencl_context` 的一部分）：

```c
cl_program program_mul_mm_f32_f32_l4_lm;
cl_program program_mul_mm_f16_f32_l4_lm;
cl_program program_mul_mm_f16_f32_l4_lm_n8;     // n8 = N 方向分块的一种变体
cl_program program_mul_mm_q8_0_f32_l4_lm;
...
cl_kernel kernel_mul_mm_f16_f32_kqv;            // attention 专用
cl_kernel kernel_mul_mm_f16_f32_kq;             // attention 专用
...
cl_kernel kernel_mul_mm_q2_k_f32_l4_lm;
cl_kernel kernel_mul_mm_q3_k_f32_l4_lm;
cl_kernel kernel_mul_mm_f32_f32_l4_lm;
cl_kernel kernel_mul_mm_f16_f32_l4_lm;
```

**这就是"算子优化"真正的战场**：不是写一个通用 kernel，而是**为常见的 (dtype × shape × 硬件) 组合各写一个特化版本**，然后写一套 if-else 把它们分派好。

命名可以解码成：

```
kernel_ mul_mm _ f16 _ f32 _ l4 _ lm
         └ op  └ 权重类型 └ 激活类型 └ 分块配置 └ 家族名
```

`l4_lm` 这一族还带两个**运行时环境变量**调参（我从代码里 grep 到的）：

- `GGML_OPENCL_LM_BK`：K 方向分块大小（BK = block K）
- `GGML_OPENCL_LM_HALF`：是否走半精度累加
- 另外还有 `_n8` 变体（N 方向分块）

> **给你的第一个硬知识**：你之前记成 `matmul_fp16_fp32_i4_lm`，实际名字是 **`mul_mm_f16_f32_l4_lm`**（`l4`，不是 `i4`）。面试时把名字说准，可信度差一档。

### 3.4 编译选项与硬件分档

同一个 kernel 在不同 GPU 上要用不同的编译选项，后端里是这么干的：

```c
std::string compile_opts = std::string("-cl-std=") + opencl_c_std +
                           " -cl-mad-enable -cl-unsafe-math-optimizations"
                           " -cl-finite-math-only -cl-fast-relaxed-math";
if (backend_ctx->adreno_use_large_buffer) {
    compile_opts += " -qcom-enable-large-buffer ";
}
```

而且它会**解析设备名和驱动版本**来分档：

```c
enum ADRENO_GPU_GEN { ADRENO_UNKNOWN, A6X, A7X, A8X, X1E, X2E };
// 通过 device_name 里是否含 "730"/"740"/"750" 之类判断代际
static ADRENO_GPU_GEN get_adreno_gpu_gen(const char *device_name);
// 还会从 driver_version 里抠出编译器版本（E031 / E17 / DX）
static ggml_cl_compiler_version get_adreno_cl_compiler_version(const char *driver_version);
```

**为什么值得知道**：kernel 的"最优参数"依赖硬件代际和编译器版本。你在 A7X 上调好的 tile，到 A8X 上可能不是最优 —— 所以成熟的后端都是"分档 + 环境变量兜底"。

### 3.5 内建 profiling：你自己就能看到 kernel 名和耗时

后端里有一个 `ProfilingInfo` 结构，记录的东西非常全：

```c
struct ProfilingInfo {
    std::string op_name;
    std::string kernel_name;      // ← 实际用了哪个 kernel
    cl_kernel kernel;
    cl_event  evt;
    cl_ulong  cmd_queued_duration_ns;
    cl_ulong  cmd_submit_duration_ns;
    cl_ulong  cmd_duration_ns;    // ← 纯 kernel 执行时间
    cl_ulong  cmd_complete_duration_ns;
    size_t global_size[3];        // ← NDRange 大小
    size_t local_size[3];         // ← work-group 大小
    size_t output_size[4];
};
```

**这就是排查"我的 kernel 有没有被选中"最直接的手段**：看 `kernel_name`，一眼就知道走了哪条路。

> 开关名字我没在抓到的那段代码里确认（文件太大，我只拿到了前 100KB）。你在本地 `grep -n "profiling_info" ggml-opencl.cpp` 就能找到触发方式。

---

## 4. 解剖 Hexagon 后端（ggml-hexagon）

### 4.1 它和 OpenCL 的根本不同

| 维度 | OpenCL（Adreno GPU） | Hexagon（HTP） |
|---|---|---|
| 谁搬数据 | 硬件 cache 自动 | **你显式发 DMA 到 VTCM** |
| 片上内存 | `__local` 16~64KB | **VTCM，容量更大但要自己布局** |
| 并行模型 | work-group + barrier | **work_queue + N 个线程，手动切分行/块** |
| 向量指令 | 隐式（编译器自动向量化） | **HVX 显式 128B 向量** |
| 权重格式 | 可离线 repack（SoA） | **tiled 格式 + `htp-repack`** |

一句话：**GPU 是"缓存友好就快"，NPU 是"搬运管得好才快"**。这就是为什么 Hexagon 的 kernel 里全是 `dma_queue_push` / `dma_queue_pop`。

### 4.2 读一段真代码：`hvx_mm_2d_tiled`

这是 Hexagon 上二维 matmul 的主循环，我删掉次要分支后逐段注释：

```c
static void hvx_mm_2d_tiled(unsigned int nth, unsigned int ith, void * data) {
    // ith = 第几个线程，nth = 总共几个线程
    const uint32_t src0_start_row = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);

    const uint32_t tile_size         = htp_tensor_get_tile_size(src0->type);
    const uint32_t aligned_tile_size = htp_tensor_get_tile_padded_size(src0->type);
    uint32_t n_k_tiles_w      = ne00 / htp_tensor_get_tile_k(src0->type);
    uint32_t tile_row_stride  = n_k_tiles_w * tile_size;

    // ① 预取：先把 n_prefetch 个 tile 推进 DMA 队列（流水线的"预冲"）
    uint32_t ct_start = src0_start_row / 32;      // 按 32 行一个 tile 切
    uint32_t ct_end   = (src0_end_row + 31) / 32;
    uint32_t push_ct  = ct_start;
    for (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++) {
        dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned,
                       src0_row + push_ct * tile_row_stride), aligned_tile_size, ...);
    }

    // ② 主循环：一次处理一个权重 tile（32 行 × K）
    for (uint32_t ct = ct_start; ct < ct_end; ct++) {
        const uint8_t * w_tile = (void *) dma_queue_pop(dma_q).dst;   // 等 DMA 到货

        int valid_rows = (int)ne0 - (int)(ct * 32);                   // 尾块处理
        valid_rows = MIN(32, MAX(0, valid_rows));

        // ③ 对每一行激活，做一次 32 路点积（vec_dot_32x1 = 同时算 32 个输出列）
        for (uint32_t ir1 = 0; ir1 < act_nrows; ++ir1) {
            const uint8_t * act_col = (const uint8_t *)(act_data + ir1 * act_stride);
            float * dst_ptr = &dst_row[ct * 32];
            mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_col, valid_rows, src2_ptr);
        }

        // ④ 回填：算完的 tile buffer 立刻投给下一次预取（零额外分配）
        if (push_ct < ct_end) {
            dma_queue_push(dma_q, dma_make_data(w_tile, src0_row + push_ct * tile_row_stride), ...);
            push_ct++;
        }
    }
}
```

**这段代码里藏着五个 kernel 优化的通用套路**：

| 套路 | 代码里的体现 | 为什么有效 |
|---|---|---|
| **分块（tiling）** | 按 32 行一个 tile | 让工作集能放进 VTCM |
| **双缓冲 / 预取** | 先 push `n_prefetch` 个，循环里 pop 一个、push 一个 | 把 DMA 和计算重叠，隐藏搬运延迟 |
| **buffer 复用** | 弹出的 tile buffer 直接用于下一次 push | 零额外内存、零分配 |
| **向量化** | `vec_dot_32x1` 一次算 32 列 | 一条 HVX 指令干 32 个 MAC |
| **边界处理不放大成本** | `valid_rows` 只影响尾块 | 主循环无分支 |

**注意 `vec_dot_32x1` 是个函数指针**。它在入口处按权重类型赋值：

```c
switch (type) {
    case HTP_TYPE_Q4_0:  mmctx->type = "q4_0_tiled-f32";  mmctx->vec_dot_32x1 = tiled_vec_dot_q4_0_32x1;  ...
    case HTP_TYPE_Q8_0:  mmctx->type = "q8_0_tiled-f32";  ...
    case HTP_TYPE_F16:   mmctx->type = "f16_tiled-f16";    ...
}
```

**热循环里没有类型分支** —— 类型判断在循环外做一次。这是所有高性能 kernel 的共同点：**把分支提到循环外，或者用模板/函数指针做特化**。

### 4.3 为什么是 32 行，不是 8 或 64

因为 HVX 的向量长度是 128 字节 = 32 个 f32。选 32 行意味着**一个 tile 的一列正好是一条向量**，`vec_dot_32x1` 里每个 lane 处理一个输出。

这就是"tile 尺寸不是拍脑袋定的，是被向量宽度决定的"。**换硬件（64B 向量的芯片）这个数就要变** —— 所以代码里到处是 `VLEN` 宏。

### 4.4 激活在 VTCM 里现场量化

注意 `need_quant` 那条分支：权重量化过了（Q4_0/Q8_0），但**激活是 f32**，而 HVX 的 int8 点积需要激活也是 int8 —— 于是它在片上做一次量化：

```c
if (need_quant) {
    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, ...);  // 先 DMA 到 VTCM
    work_queue_run(ctx->work_queue, quant_task_func, mmctx, n_quant_tasks); // 再在 VTCM 里量化
} else {
    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act, ...);      // 直接 DMA
}
```

**为什么要现场量化**：算子是 `Q4_0 权重 × f32 激活`，量化粒度必须和权重对齐（Q4_0 的 block 是 32 个元素），而激活的量化参数依赖运行时数据（per-block scale）。**这一步是纯带宽换精度**，也是 note 04 里"W4A16 的 kernel 要 unpack+quant"的具体形态。

### 4.5 trace 事件：这个后端天生可 profile

代码里到处是这种配对调用：

```c
htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);        // 初始化
htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);   // HVX 计算
htp_trace_event_start(tr, HTP_TRACE_EVT_A_PREP, r);      // 激活准备
htp_trace_event_start(tr, HTP_TRACE_EVT_W_DEQUANT, i);   // 权重反量化
htp_trace_event_start(tr, HTP_TRACE_EVT_O_PROC, ...);    // 输出处理
```

**这五类事件正好对应 kernel 的五个阶段**。profiling 的时候你会直接看到"HVX_COMP 占 30%、A_PREP 占 50%" —— 那就说明瓶颈在激活准备，不在计算。

**这是我认为最值得抄的做法**：自己写 kernel 时，把时间按阶段打点，比只看总时间有用十倍。

### 4.6 模型很大时，NPU 的内存和 VTCM 到底怎么操作

> 这一节是面试原题的展开：「模型一大，NPU 的内存和 VTCM 怎么处理？」

**先纠正前提：VTCM 里从来不放模型。**

VTCM 是**暂存区（scratchpad）**，只放「当前这个 op 正在算的那一个 tile」。所以：

```
模型变大  →  VTCM 用量基本不变
          →  变的是三件事：
             ① 每 token 要从 DDR 搬多少权重（线性增长）
             ② KV cache 多大
             ③ 整个模型能不能塞进 DDR
```

量级上：Hexagon 的 VTCM 是 **MB 级**，一个 9B 模型 4bit 也有 **4.5 GB** —— 差 1000 倍，永远不可能常驻。

**① 编译期：VTCM 是一块「要预约」的内存，预算算不出来就编译失败**

它没有 cache、没有缺页，编译器必须**静态**算出这块 buffer 需要多少字节。代码里就是两句：

```c
// 把 act / src0 / bias / dst 的偏移全部算出来
htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, m_layout_rows,
                             octx->n_threads, dst_row_size, src0_row_size, act_row_size, ...);
const size_t vtcm_size = L.total_bytes;

// 装不下就直接失败，没有「换页」这个选项
if (octx->ctx->vtcm_size < vtcm_size) {
    FARF(ERROR, "matmul-%s : current VTCM reservation %zu is too small, needed %zu\n", ...);
    return HTP_STATUS_VTCM_TOO_SMALL;
}
```

**这句 `HTP_STATUS_VTCM_TOO_SMALL` 就是"模型一大"在编译期的样子。**
（对应的旋钮你在 QAIRT 里也见过：`graphs.vtcm_size_in_mb`，它是**图级编译选项**，prefill 和 decode 两张图可以不一样 —— 见 note 05 §4.4。）

**② 运行期：权重是「流」过 VTCM 的，不是「驻留」**

```c
// 按 32 行一个 tile 切；K 方向再切成 n_k_tiles_w 段
uint32_t ct_start = src0_start_row / 32;
for (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++)
    dma_queue_push(...);                       // 预取 n_prefetch 个 tile

for (uint32_t ct = ct_start; ct < ct_end; ct++) {
    const uint8_t * w_tile = (void *) dma_queue_pop(dma_q).dst;   // 等这块到货
    ... 计算 ...
    if (push_ct < ct_end) {                                        // 这块 buffer 立刻复用
        dma_queue_push(dma_q, dma_make_data(w_tile, ...));
        push_ct++;
    }
}
```

所以 VTCM 里同时只存在 **`n_prefetch` 个 tile**，跟模型有 1B 还是 30B 参数无关。模型变大只意味着**循环跑得更久、DMA 次数更多**。

**③ 三个分块旋钮，就是"能不能塞下"的全部自由度**

| 旋钮 | 代码里 | 作用 | 调小的代价 |
|---|---|---|---|
| tile 行数 | 固定 32（= HVX 向量宽度） | 一个 tile 的计算高度 | 改不了，是硬件定的 |
| **K 方向分块** | `n_k_tiles_w = ne00 / tile_k` | 一次搬多少 K | 更多次 DMA 启动开销 |
| **激活行分块** | `m_chunk` | 一次放多少行激活 | 更多次循环、尾部浪费 |
| **每线程切片** | `vtcm_*_size_per_thread = L.*_bytes / n_threads` | 多线程瓜分 VTCM | 线程太多 → 每片太小 |

**这三者是一道方程**：`VTCM 总量 = f(tile 行数, K 块, m 行数, 线程数)`。模型大、激活大时，编译器/你会把 `m_chunk` 调小、把 K 块调小，**用更多的 DMA 换更低的 VTCM 峰值**。

**④ 连最小 tile 都放不下怎么办（按代价排序）**

```
降精度（W4A16 → 更小的 tile 需求）
  → 换更小的分块配置（BK / BM）
    → 把 op 拆成多次调用（一次算一半 N）
      → 换 kernel 变体（有的变体 VTCM 足迹小、但慢）
        → fallback 到 CPU（最差：一次 fallback = 一次同步 + 拷贝，收益全没）
```

**⑤ 如果是"模型比 DDR 还大"（另一层问题）**

这时 VTCM 已经无关了，要解决的是**权重根本放不进内存**：

| 手段 | 做法 | 代价 |
|---|---|---|
| 权重流式（mmap） | 按层加载，一次只驻留几层 | 每层边界一次 I/O；flash 只有 ~500 MB/s |
| MoE 稀疏 | 只加载被路由到的专家 | 路由抖动 → I/O 抖动 |
| 分层 offload | 热层留 DDR、冷层留 flash | 需要预取调度（note 05 §6.4 的 cost-aware prefetch） |
| 量化 | W4/W2 | 精度 |

**60 秒口述版（背这段）**：

> "VTCM 不是缓存，是 scratchpad，它只放当前 op 的一个 tile，从来不放模型 —— 所以模型变大不会撑爆 VTCM，只会让权重流式的次数变多。真正被撑爆的是三样东西：每 token 的权重带宽、KV cache、以及整个模型能不能装进 DDR。编译期如果某个 op 的 tile 装不下，会直接报 VTCM_TOO_SMALL，解法是把 m 或 K 方向的分块调小，用更多次 DMA 换更低的 VTCM 峰值。"

**可能的追问**：

- **为什么 NPU 不用 cache 自动管理？** 因为 scratchpad + DMA 可以做到**确定性**：什么时候搬、搬多少、和计算怎么重叠，全在编译期定好，没有 cache miss 的抖动。代价是软件复杂度全压在编译器和你身上。
- **VTCM 和 GPU 的 shared memory 有什么区别？** 量级类似、角色类似，但 GPU 的 shared memory 是**每个 work-group 私有**、生命周期是"一次 kernel"；HTP 的 VTCM 是**整个 DSP 共享**、生命周期是"一次 graph"，而且要显式 DMA 填充。
- **为什么 VTCM 大小是编译选项？** 因为它决定 tiling 方案，而 tiling 决定 kernel 的 shape 和 DMA 序列 —— 全是编译期常量。运行时改 VTCM = 重新编译（note 05 §4.4）。

---

## 5. 案例复盘：你那次 "contiguous 触发快 kernel"

### 5.1 机理：kernel 选择是一条判定链

后端的 `mul_mat` 分发（`ggml_cl_mul_mat`）本质上是一串 `if`。下面这段是**示意**（真实条件以你本地代码为准，我在前 100KB 里没抓到那个函数）：

```c
// 示意，不是原文
if (src0->type == F16 && src1->type == F32 && ggml_is_contiguous(src1) && N >= 阈值) {
    kernel = kernel_mul_mm_f16_f32_l4_lm;      // 快路径（l4_lm 家族）
} else if (...) {
    kernel = kernel_mul_mm_f16_f32;            // 通用路径
} else {
    // 逐行 GEMV，或者干脆不支持 → 回落 CPU
}
```

**你做的事**：QK 那个 matmul 的 src1（K）在 reshape/permute 之后 `nb[0]` 不等于 `sizeof(f16)`，第一个条件不成立 → 落到慢路径。你插入了一次 `ggml_cont()`（或等价操作）把它变成连续布局 → 条件成立 → 命中 `l4_lm`。

### 5.2 为什么 QK 会不连续

CLIP 里 Q/K 会经历 `reshape → permute → reshape`（为了拆 head）。每次 permute 都只改 `nb[]`，**不搬数据** —— 这是 ggml 的零拷贝设计。代价是：**任何依赖"连续"假设的下游 kernel 都用不了**。

```
原始 [L, D]  →  reshape/permute 成 [H, L, Dh]  →  nb 变了，data 没动  →  不连续
```

### 5.3 代价与收益的账（这才是要记住的）

`ggml_cont()` 是**一次真实的拷贝**：O(L·D) 字节。收益是让 N 次 matmul 走快路径。

```
拷贝成本   = L·D·2 bytes / 带宽
收益       = (N 个 token 的 matmul 时间) × (1 − 1/加速比)
判断标准   = 收益 > 成本
```

对 CLIP 这种 L 大、matmul 次数多的场景，收益远大于成本 —— 所以"多一次拷贝反而更快"是完全合理的。**这就是 kernel 优化里最常见的反直觉：为了少搬，先多搬一次。**

### 5.4 30 分钟定位这类问题的固定套路

```bash
# ① 确认你的 op 走了哪个 kernel（有 profiling 就看 kernel_name）
grep -n "kernel_mul_mm_f16_f32_l4_lm" ggml/src/ggml-opencl/ggml-opencl.cpp

# ② 读它所在的那串 if，把每个条件翻译成 tensor 属性
#    （type / nb[0] / nb[1] / ne[1] / 是否 contiguous / 是否需要 pad）

# ③ 回到构图处，检查你的 tensor 是不是满足这些属性
#    ggml 里可以用 ggml_is_contiguous(t) 直接断言

# ④ 加一次 ggml_cont，测 before/after
```

**四条命令，半小时。** 这套流程对 CUDA / Vulkan / Hexagon 后端一样适用 —— 只是 grep 的文件不同。

---

## 6. 算子优化的通用方法论（六步）

```
1. 定位：这个 op 占总时间的多少？（profile，别看总时间）
2. 归类：算 roofline —— memory-bound 还是 compute-bound？（note 03）
3. 找路：它现在走的是哪个 kernel？（kernel_name）
4. 对比：有没有更快的变体？条件差在哪？（读 dispatch 的 if 链）
5. 动手：改布局 / 换变体 / 加特化 —— 一次只改一个变量
6. 验收：before/after 的 GB/s 或 GFLOPS，以及"离峰值还有多远"
```

**六步里最容易跳过、也最致命的是第 4 步**：很多人一上来就改 kernel 代码，但 90% 的情况是"你压根没走到那个 kernel"。

> **一句话方法论**：先确认"路对不对"，再谈"路快不快"。

---

## 7. 面试问答（13 题）

**Q1：一个 work-group 和一次 kernel 启动是什么关系？**
work-group 是硬件调度和共享内存的单位；一次启动会产生很多 work-group。同一个 wavefront（32/64 个 work-item）锁步执行，所以访问模式（合并与否）直接决定有效带宽。

**Q2：为什么说 `nb[]` 比 `ne[]` 重要？**
`ne` 只说形状，`nb` 说内存布局。kernel 的快路径通常要求特定 `nb`（比如 `nb[0]==sizeof(type)` 才能向量化加载）。permute/reshape 只改 `nb` 不搬数据，所以"形状对、布局不对"是性能问题的头号来源。

**Q3：ggml 后端要实现哪几个接口？**
`supports_op`（能不能接）、`compute_forward`（怎么算）、`buffer_type`（内存怎么分配）。`supports_op` 不只判断 op 类型，还判断 dtype/形状/布局 —— 这就是为什么换个 layout 就换路径。

**Q4：为什么一个 op 要写几十个 kernel 变体？**
因为最优 tile/向量化方式依赖 (dtype × shape × 硬件)。通用 kernel 只能取所有情况的交集，性能差一个档。所以成熟后端都是"特化 + 分派"。

**Q5：`mul_mm_f16_f32_l4_lm` 这名字怎么读？**
`mul_mm` = 矩阵乘；`f16_f32` = 权重 f16 × 激活 f32；`l4_lm` = 分块配置 + kernel 家族；`_n8` 是 N 方向分块的变体。同族还有 `_kqv`/`_kq` 这种 attention 专用版。

**Q6：OpenCL 后端怎么保证 kernel 在不同 Adreno 上都能跑好？**
分档：解析 `device_name` 判断代际（A6X/A7X/A8X/X1E/X2E）、解析 driver 版本拿编译器版本、按档位选编译选项（如 `-qcom-enable-large-buffer`），再加环境变量（`GGML_OPENCL_LM_BK` 等）兜底调参。

**Q7：Hexagon 和 GPU 后端最本质的区别？**
GPU 靠 cache 自动搬运，NPU 靠**显式 DMA 到 VTCM**。所以 NPU kernel 的结构是"DMA 预取 + 双缓冲 + 片上计算"，性能首先取决于搬运调度，而不是算力利用率。

**Q8：为什么要按 32 行一个 tile？**
因为 HVX 向量是 128 字节 = 32 个 f32。32 行让"一列 = 一条向量"，`vec_dot_32x1` 每个 lane 算一个输出。tile 尺寸是被向量宽度决定的，不是拍脑袋。

**Q9：`vec_dot_32x1` 为什么用函数指针？**
把类型分支提到热循环外。循环内只有纯粹的向量运算，没有 `switch`。这是所有高性能 kernel 的共性：**分支在循环外，特化在编译期或入口处**。

**Q10：为什么激活要"现场量化"？**
权重是 Q4_0（block=32），激活是 f32，HVX 的 int8 点积要求两边都是 int8，且量化粒度要对齐。激活的 scale 依赖运行时数据（per-block），所以只能搬到 VTCM 后现场算。这是 W4A16 kernel 的标准形态。

**Q11：你那次 contiguous 优化，本质是什么？**
不是改 kernel，是**让数据满足快 kernel 的入场条件**。代价是一次 O(L·D) 拷贝，收益是后续所有 matmul 走特化路径。判断标准是"拷贝成本 < 省下的时间"。


**Q12：怎么快速判断一个改动值不值得做？**
先算 roofline（省了多少字节 / 多了多少 FLOPs），再 profile 确认这个 op 占多少时间。**如果这个 op 只占总时间 2%，你优化它 2 倍也只省 1%。**

**Q13：模型一大，NPU 的内存和 VTCM 怎么办？**
先纠正前提：**VTCM 是 scratchpad 不是 cache，只放当前 op 的一个 tile，从不用来放模型**。模型变大只让权重流式次数变多，不会撑爆 VTCM。真正撑不住的是三样：每 token 的权重带宽、KV cache、以及模型能否装进 DDR。编译期 VTCM 预算算不出来会直接报 `VTCM_TOO_SMALL`，解法是调小 m / K 方向的分块（用更多 DMA 换更低峰值）；若模型 > DDR，则走权重流式（mmap）/ MoE 稀疏 / 分层 offload / 量化。

---

## 8. 动手实验

**没设备也能做的（先做这个）**

- [ ] **跑布局实验**：`python tools/layout_cost.py`，把 §1.3.1 的两个表换成你自己的数字。验收：能解释为什么 A2 里"连续方向反而更慢"。

- [ ] **读 dispatch**：`grep -n "kernel_mul_mm_f16_f32_l4_lm" ggml/src/ggml-opencl/ggml-opencl.cpp`，把那个 if 条件抄下来，逐条翻译成 tensor 属性。
- [ ] **读一个 kernel 的结构**：打开 `hvx_mm_2d_tiled`，标出"预取 / 主循环 / 回填 / 边界"四段。验收：能不看代码复述这四段。
- [ ] **画一张自己的 kernel 时间分解图**：照 `HTP_TRACE_EVT_*` 的五个阶段，给你的目标 op 设计打点方案。

**有 OpenCL 设备再做（Intel/AMD 核显也行）**

- [ ] `cmake -DGGML_OPENCL=ON` 编译后用 `llama-bench` 跑同一模型，对比 `-ngl 0` 与 `-ngl 99`。
- [ ] 找到 profiling 开关（`grep -n "profiling_info" ggml-opencl.cpp`），打开它，导出一个 op 的 `kernel_name + cmd_duration_ns + global/local size` 表。
- [ ] 验证"kernel 选择随 shape 变"：同一个 matmul，把 `N`（token 数）从 1 扫到 512，看 `kernel_name` 什么时候切换。**这一条能直接印证 note 03 的 roofline 拐点。**

**在你的 mtmd 优化上复盘**

- [ ] 把当时的 patch 找出来，回答三个问题：① 加 `ggml_cont` 之前走的是哪个 kernel？② 拷贝了多少 MB？③ matmul 省了多少 ms？算一下收益/成本比。
- [ ] 如果比值不高，试试更省的替代：能不能在**上游**就生成连续布局（避免这次拷贝）？这是"优化搬运"的最高境界 —— 不搬。

---

## 9. 参考

- OpenCL 规范第 3、6 章（执行模型、内存模型）—— 只读这两章就够，别通读。
- `ggml/src/ggml-opencl/ggml-opencl.cpp`（[GitHub](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-opencl/ggml-opencl.cpp)）：kernel 注册表、编译选项、`ProfilingInfo`。
- `ggml/src/ggml-hexagon/htp/matmul-ops.c`（[GitHub](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/htp/matmul-ops.c)）：`hvx_mm_2d_tiled`、`vec_dot_32x1`、VTCM 布局、现场量化。
- `ggml/src/ggml-hexagon/htp-repack.cpp`：权重怎么被重排成 tiled 格式（**这是"布局决定性能"最完整的案例**）。
- 本套笔记：note 01（im2col/GEMM 的图级视角）、note 03（roofline，判断值不值得优化）、note 04（W4A16 在 kernel 里的样子）。\n- 本机脚本：`tools/layout_cost.py`（§1.3.1 的数字来源）。
