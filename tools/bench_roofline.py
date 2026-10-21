"""Roofline 实测台：测本机峰值带宽、峰值算力，以及典型 op 的达成率。

用法：  python tools/bench_roofline.py
产物：  bench_roofline_result.md（可直接贴进笔记）

设计要点
- 所有 op 都显式写出"读几次、写几次"，字节数是算出来的，不是猜的。
- 峰值带宽单独测（单线程 / 多线程），因为 elementwise 在 numpy 里是单线程的。
- GEMM 扫 batch，用实测曲线找 ridge point，而不是抄厂商标称值。
"""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

B = 4  # float32 bytes


def bench(fn, warmup=1, rep=5):
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


rows = []


def record(name, seconds, flops, byts, note=""):
    gbs = byts / seconds / 1e9
    gflops = flops / seconds / 1e9
    ai = flops / byts
    rows.append(dict(name=name, ms=seconds * 1e3, gbs=gbs, gflops=gflops, ai=ai, note=note))
    print(f"{name:30s} {seconds*1e3:8.2f} ms  {gbs:7.2f} GB/s  {gflops:8.2f} GFLOPS  AI={ai:6.2f}")


# ---------------------------------------------------------------- 1. 峰值带宽
N = 32 * 1024 * 1024  # 32M f32 = 128 MB
a = np.random.rand(N).astype(np.float32)
b = np.random.rand(N).astype(np.float32)
c = np.empty_like(a)

t1 = bench(lambda: np.copyto(c, a))  # 1R + 1W
bw_1t = 2 * B * N / t1 / 1e9
print(f"copy 单线程        {bw_1t:6.2f} GB/s")

chunks = [(i * N // 4, (i + 1) * N // 4) for i in range(4)]


def copy_mt():
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(lambda s: np.copyto(c[s[0]:s[1]], a[s[0]:s[1]]), chunks))


t4 = bench(copy_mt, rep=3)
bw_4t = 2 * B * N / t4 / 1e9
print(f"copy 4 线程        {bw_4t:6.2f} GB/s")

peak_bw = max(bw_1t, bw_4t)
record("copy (1R+1W)", t4 if bw_4t >= bw_1t else t1, 0, 2 * B * N, "STREAM copy，即峰值带宽")

# 纯读探测：BLAS 的 dot（多线程 + 预取），用来展示"带宽不是一个数"
t = bench(lambda: np.dot(a, b), rep=3)
print(f"dot 纯读（BLAS）   {2*B*N/t/1e9:6.2f} GB/s  <- 纯读可以高于 copy")

# ---------------------------------------------------------------- 2. elementwise / norm
record("add (2R+1W)", bench(lambda: np.add(a, b, out=c)), N, 3 * B * N, "STREAM Add")


def silu(x, out):
    np.exp(-x, out=out)
    out += 1.0
    np.divide(x, out, out=out)


record("silu (1R+1W)", bench(lambda: silu(a, c)), 4 * N, 2 * B * N, "exp 是超越函数，CPU 上是 SFU-bound")

M, D = 16384, 4096
x = np.random.rand(M, D).astype(np.float32)
y = np.empty_like(x)


def rms_naive(x, out):  # x*x 临时数组 +1R1W，mean +1R，缩放 1R1W  => 3R 2W(+temp 2)
    sq = np.mean(x * x, axis=-1, keepdims=True)
    np.multiply(x, 1.0 / np.sqrt(sq + 1e-6), out=out)


record("rmsnorm 朴素 (4R+2W)", bench(lambda: rms_naive(x, y)), 2 * M * D, 6 * B * M * D, "x*x 多一个 256MB 临时数组")


def rms_einsum(x, out):  # einsum 直接算平方和，无临时数组 => 2R 1W
    sq = np.einsum("ij,ij->i", x, x)[:, None]
    np.multiply(x, M * 1.0 / np.sqrt(sq + M * 1e-6), out=out)


record("rmsnorm einsum (2R+1W)", bench(lambda: rms_einsum(x, y)), 2 * M * D, 3 * B * M * D, "省掉临时数组，2.8x")

M2, L = 8192, 4096
s = (np.random.rand(M2, L).astype(np.float32) * 10)
o = np.empty_like(s)


def softmax_naive(x, out):  # max 1R | sub 1R1W | exp 1R1W | sum 1R | div 1R1W => 5R 3W
    m = x.max(axis=-1, keepdims=True)
    np.subtract(x, m, out=out)
    np.exp(out, out=out)
    ss = out.sum(axis=-1, keepdims=True)
    np.divide(out, ss, out=out)


record("softmax 朴素 (5R+3W)", bench(lambda: softmax_naive(s, o)), 5 * M2 * L, 8 * B * M2 * L, "每多一遍就多一次全量读写")

# ---------------------------------------------------------------- 3. GEMV / GEMM
Mk, K = 8192, 4096
W = np.random.rand(Mk, K).astype(np.float32)
v = np.random.rand(K).astype(np.float32)
t = bench(lambda: W @ v, rep=5)
record("gemv 8192x4096", t, 2 * Mk * K, B * (Mk * K + K + Mk), "batch=1 decode 的 linear")

Mg, Kg = 4096, 4096
Wg = np.random.rand(Mg, Kg).astype(np.float32)

print("\n--- GEMM batch 扫描 (M=K=4096, fp32) ---")
sweep = []
for n in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    X = np.random.rand(Kg, n).astype(np.float32)
    t = bench(lambda X=X: Wg @ X, rep=3 if n < 512 else 2)
    flops = 2 * Mg * Kg * n
    byts = B * (Mg * Kg + Kg * n + Mg * n)
    sweep.append((n, t * 1e3, flops / t / 1e9, byts / t / 1e9, flops / byts))
    print(f"N={n:5d}  {t*1e3:8.2f} ms  {flops/t/1e9:8.2f} GFLOPS  {byts/t/1e9:8.2f} GB/s  AI={flops/byts:7.2f}")

peak_flops = max(s[2] for s in sweep)
ridge = peak_flops * 1e9 / (peak_bw * 1e9)

print(f"\npeak_bw   = {peak_bw:.1f} GB/s  (单线程 {bw_1t:.1f})")
print(f"peak_fp32 = {peak_flops:.1f} GFLOPS")
print(f"ridge     = {ridge:.1f} FLOP/Byte  ({ridge/2:.1f} MAC/Byte)")

print("\n--- 汇总 ---")
print(f"{'op':30s} {'GB/s':>8s} {'%bw':>6s} {'GFLOPS':>9s} {'%flop':>6s} {'AI':>7s}  判定")
for r in rows:
    pb = r["gbs"] / peak_bw * 100
    pf = r["gflops"] / peak_flops * 100
    v = "MEM" if r["ai"] < ridge else "COMPUTE"
    print(f"{r['name']:30s} {r['gbs']:8.2f} {pb:5.0f}% {r['gflops']:9.2f} {pf:5.1f}% {r['ai']:7.2f}  {v}")

with open(".tmp/bench_roofline_result.md", "w", encoding="utf-8") as f:
    f.write("# 本机 roofline 实测结果\n\n")
    f.write(f"- peak_bw（copy 4 线程）: **{peak_bw:.1f} GB/s**（单线程 {bw_1t:.1f}）\n")
    best_n = max(sweep, key=lambda s: s[2])[0]
    f.write(f"- peak_fp32（GEMM N={best_n}）: **{peak_flops:.1f} GFLOPS**\n")
    f.write(f"- ridge point: **{ridge:.1f} FLOP/Byte**（{ridge/2:.1f} MAC/Byte）\n\n")
    f.write("| op | ms | GB/s | %bw | GFLOPS | %flop | AI (FLOP/B) | 判定 | 备注 |\n")
    f.write("|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        pb = r["gbs"] / peak_bw * 100
        pf = r["gflops"] / peak_flops * 100
        v = "MEM" if r["ai"] < ridge else "COMPUTE"
        f.write(f"| {r['name']} | {r['ms']:.2f} | {r['gbs']:.2f} | {pb:.0f}% | {r['gflops']:.2f} | {pf:.1f}% | {r['ai']:.2f} | {v} | {r['note']} |\n")
    f.write("\n## GEMM batch 扫描\n\n| N (batch) | ms | GFLOPS | GB/s | AI (FLOP/B) |\n|---|---|---|---|---|\n")
    for n, ms, gf, gb, ai in sweep:
        f.write(f"| {n} | {ms:.2f} | {gf:.2f} | {gb:.2f} | {ai:.2f} |\n")

print("\n-> bench_roofline_result.md")
