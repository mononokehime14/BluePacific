"""布局决定带宽：在本机上量一下"连续访问 vs 跨步访问"的差距。

跑法：  python tools/layout_cost.py
产物：  .tmp/layout_cost_result.md

三个实验，都对应 note 06 §1.3：
  A. 同一个矩阵，按行求和 vs 按列求和（内存布局不变，访问模式变）
  B. 同一个 matmul，B 连续 vs B 转置（数学等价，访存模式不同）
  C. 模拟 wavefront 的合并访问 vs 跨步访问，数"内存事务"次数
"""

import time

import numpy as np

OUT = []


def log(s=""):
    print(s)
    OUT.append(str(s))


def bench(fn, rep=5):
    fn()
    best = float("inf")
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


# ================================================================ A. 跨步读
log("## A. 同样搬 134 MB：连续读 vs 跨步读\n")
n = 8192
half = n // 2
x = np.random.rand(n, n).astype(np.float32)   # 256 MB
dst = np.empty((n, half), np.float32)

read_bytes = x[:, :half].nbytes
t_c = bench(lambda: np.copyto(dst, x[:, :half]))   # 连续的行块
t_s = bench(lambda: np.copyto(dst, x[:, ::2]))     # 隔一个取一个（stride = 8B）
log("| 方式 | 时间 | 有效带宽（按读字节算） | 说明 |")
log("|---|---|---|---|")
log(f"| x[:, :half]（连续） | {t_c*1e3:.1f} ms | {read_bytes/t_c/1e9:.1f} GB/s | 每个 lane 读相邻元素 |")
log(f"| x[:, ::2]（跨步） | {t_s*1e3:.1f} ms | {read_bytes/t_s/1e9:.1f} GB/s | 每个 lane 隔一个，只用到一半 cache line |")
log("")
log(f"→ 搬的字节数一样，跨步读 **慢了 {t_s/t_c:.2f}×**。")
log("")

# ================================================================ A2. 一个反例
log("## A2. 反例：numpy 的 sum(axis=) 不吃这套教训\n")
t_row = bench(lambda: x.sum(axis=1))
t_col = bench(lambda: x.sum(axis=0))
log(f"- sum(axis=1)（连续方向）：{t_row*1e3:.1f} ms，{x.nbytes/t_row/1e9:.1f} GB/s")
log(f"- sum(axis=0)（跨步方向）：{t_col*1e3:.1f} ms，{x.nbytes/t_col/1e9:.1f} GB/s")
log("")
log("这里反而是 axis=0 更快 —— 因为 numpy 对不同 axis 用了不同实现：")
log("axis=0 每行调一次向量化内积，axis=1 走的是逐元素累加。**实现质量盖过了访存模式。**")
log("教训：**布局的收益只在同一套实现内部比较才有意义**；跨实现比会被向量化程度污染。")
log("(这正是为什么 kernel 优化必须看 SASS/HVX 反汇编，不能只看 C 代码。)")
log("")

# ================================================================ B. matmul 的布局
log("## B. 同一个 matmul：B 连续 vs B 转置\n")
m = k = 2048
A = np.random.rand(m, k).astype(np.float32)
B = np.random.rand(k, k).astype(np.float32)
Bt = np.ascontiguousarray(B.T)   # 数学上等价于用 B.T，但显式拷贝成连续布局

t1 = bench(lambda: A @ B)
t2 = bench(lambda: A @ Bt)
t3 = bench(lambda: A @ B.T)      # 不拷贝，直接给转置视图
flops = 2 * m * k * k
log("| 写法 | 时间 | GFLOPS | 说明 |")
log("|---|---|---|---|")
log(f"| A @ B（B 连续） | {t1*1e3:.1f} ms | {flops/t1/1e9:.1f} | 基线 |")
log(f"| A @ Bt（显式拷贝成连续） | {t2*1e3:.1f} ms | {flops/t2/1e9:.1f} | 多一次拷贝，换连续布局 |")
log(f"| A @ B.T（转置视图，不拷贝） | {t3*1e3:.1f} ms | {flops/t3/1e9:.1f} | BLAS 内部自己处理 |")
log("")
log("注意：numpy/BLAS 通常会自动处理转置，所以这里差距不大。")
log("但在手写 kernel 里，跨步的 B 会直接让向量化失效 —— 这就是 note 06 §5 的场景。")
log("")

# ================================================================ C. 合并访问 vs 跨步
log("## C. 模拟 wavefront：32 个 lane 一次读 32 个数\n")
LANE = 32
LINE = 128  # 一个 cache line 128 B


def transactions(addrs):
    """数一次 wavefront 访问会打多少个 128B 事务。"""
    return len({a // LINE for a in addrs})


log("| 模式 | 每个 lane 的地址 | 需要的事务数 | 有效带宽 |")
log("|---|---|---|---|")
contig = [i * 4 for i in range(LANE)]           # 连续 4B
strided = [i * 4 * 1024 for i in range(LANE)]   # 跨 4KB
log(f"| 合并（连续） | base + lane×4B | {transactions(contig)} | {LANE*4/(transactions(contig)*LINE)*100:.1f}% |")
log(f"| 跨步（span 4KB） | base + lane×4KB | {transactions(strided)} | {LANE*4/(transactions(strided)*LINE)*100:.1f}% |")
log("")
log("→ 同样是读 128 字节，跨步时硬件要发 32 个事务，**有效带宽掉到 1/32**。")
log("  这就是为什么 kernel 里要先 `__local` 中转再转置，而不是直接跨步读。")

with open(".tmp/layout_cost_result.md", "w", encoding="utf-8") as f:
    f.write("# 布局代价实测\n\n" + "\n".join(OUT) + "\n")
print("\n-> .tmp/layout_cost_result.md")
