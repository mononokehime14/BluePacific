"""量化基础实验台（纯 numpy，无需 GPU / torch）。

跑法：  python tools/quant_basics.py
产物：  .tmp/quant_basics_result.md

六个实验，每个都对应笔记里的一个结论：
  A. 裁剪 vs 舍入：int8/int4 下最优 clip 分位数不同
  B. 四种校准方法（MinMax / Percentile / MSE / KL）的对比
  C. 粒度：per-tensor / per-channel / group(128/64/32)
  D. CLE：两层 ReLU 网络，per-tensor int8 的输出误差
  E. Bias correction：量化权重的期望偏移
  F. GPTQ vs RTN：4bit 下 WX 的相对误差
"""

import numpy as np

rng = np.random.default_rng(20261024)
OUT = []


def log(s=""):
    print(s)
    OUT.append(str(s))


# ---------------------------------------------------------------- 基础算子
def q_sym(x, bits, clip=None):
    """对称量化（zero-point=0），返回反量化后的浮点。clip 给定时用它做阈值。"""
    qmax = 2 ** (bits - 1) - 1
    c = np.max(np.abs(x)) if clip is None else clip
    s = max(c / qmax, 1e-12)
    return s * np.clip(np.round(x / s), -qmax - 1, qmax), s


def q_sym_channel(x, bits, axis=0):
    """per-channel（沿 axis）对称量化。x 的第 axis 维是通道。"""
    qmax = 2 ** (bits - 1) - 1
    c = np.max(np.abs(x), axis=axis, keepdims=True)
    s = np.maximum(c / qmax, 1e-12)
    return s * np.clip(np.round(x / s), -qmax - 1, qmax), s


def q_group(x, bits, g, axis=0):
    """group-wise：沿 axis 每 g 个通道共享一个 scale。"""
    qmax = 2 ** (bits - 1) - 1
    xm = np.moveaxis(x, axis, 0)
    n = xm.shape[0]
    pad = (-n) % g
    if pad:
        xm = np.concatenate([xm, np.zeros((pad,) + xm.shape[1:], xm.dtype)], 0)
    sh = xm.shape
    xg = xm.reshape(-1, g, *sh[1:])
    c = np.max(np.abs(xg), axis=1, keepdims=True)
    s = np.maximum(c / qmax, 1e-12)
    q = np.clip(np.round(xg / s), -qmax - 1, qmax) * s
    q = q.reshape(sh)[:n]
    return np.moveaxis(q, 0, axis)


def rel_err(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(a)


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def heavy_tail(n, df=3.0, scale=1.0):
    """Student-t：模拟 LLM 激活的 outlier 结构。"""
    x = rng.standard_t(df, size=n)
    return (scale * x / x.std()).astype(np.float64)


# ================================================================ A. clip vs round
log("## A. 裁剪 vs 舍入：最优分位数随位宽变化\n")
x = heavy_tail(2_000_000)
log(f"激活：Student-t(df=3)，max| x| = {np.max(np.abs(x)):.1f}，std = {x.std():.3f}")
log("")
log("| 保留分位数 | int8 MSE | int4 MSE |")
log("|---|---|---|")
best = {}
for p in [90, 95, 99, 99.5, 99.9, 99.99, 99.999, 100]:
    clip = np.percentile(np.abs(x), p)
    e8 = mse(x, q_sym(x, 8, clip)[0])
    e4 = mse(x, q_sym(x, 4, clip)[0])
    log(f"| {p}% | {e8:.6e} | {e4:.6e} |")
    for b, e in ((8, e8), (4, e4)):
        if b not in best or e < best[b][1]:
            best[b] = (p, e)
log("")
log(f"→ int8 最优：{best[8][0]}%（MSE {best[8][1]:.3e}）；  int4 最优：{best[4][0]}%（MSE {best[4][1]:.3e}）")
log("")

# ================================================================ B. 校准方法
log("## B. 四种校准方法（同一个张量，int8 与 int4）\n")


def calib_minmax(x):
    return np.max(np.abs(x))


def calib_pct(x, p=99.9):
    return np.percentile(np.abs(x), p)


def calib_mse(x, bits):
    """在候选 clip 上搜 MSE 最小（标准做法，不是解析解）。"""
    a = np.abs(x)
    # 注意：不能只用分位数做候选——尾部 0.001% 的分位数之间跨度极大，
    # 会漏掉真正的最优 clip。用「均匀 clip 网格」更稳。
    cands = np.linspace(np.percentile(a, 50), a.max(), 600)
    best_c, best_e = None, np.inf
    for c in cands:
        e = mse(x, q_sym(x, bits, c)[0])
        if e < best_e:
            best_c, best_e = c, e
    return best_c


def calib_kl(x, bits, bins=2048):
    """KL: map float histogram onto 2^bits quant bins, pick the clip minimising KL(P||Q)."""
    a = np.abs(x)
    c_max = float(a.max())
    hist, _ = np.histogram(a, bins=bins, range=(0, c_max))
    hist = hist.astype(np.float64)
    hist /= max(hist.sum(), 1.0)
    qmax = 2 ** (bits - 1) - 1
    best_c, best_kl = c_max, np.inf
    for frac in np.linspace(0.5, 1.0, 51):
        nb = max(int(round(frac * bins)), qmax + 1)
        P = hist[:nb].copy()
        P[-1] += hist[nb:].sum()
        P /= P.sum()
        idx = np.minimum(np.arange(nb) * (qmax + 1) // nb, qmax)
        Q = np.zeros(qmax + 1)
        np.add.at(Q, idx, P)
        Q /= Q.sum()
        # 把每个量化桶的质量均摊回它覆盖的直方图桶，才能和 P 同支撑比较
        cnt = np.bincount(idx, minlength=qmax + 1)
        Qb = Q[idx] / np.maximum(cnt[idx], 1)
        m = (P > 0) & (Qb > 0)
        kl = float(np.sum(P[m] * np.log(P[m] / Qb[m])))
        if kl < best_kl:
            best_c, best_kl = c_max * nb / bins, kl
    return best_c


log("| 方法 | clip 值 | int8 MSE | int4 MSE |")
log("|---|---|---|---|")
for name, fn in [
    ("MinMax", lambda x, b: calib_minmax(x)),
    ("Percentile 99.9%", lambda x, b: calib_pct(x, 99.9)),
    ("MSE 搜索", lambda x, b: calib_mse(x, b)),
    ("KL 散度", lambda x, b: calib_kl(x, b)),
]:
    c8, c4 = fn(x, 8), fn(x, 4)
    log(f"| {name} | int8:{c8:.2f} / int4:{c4:.2f} | {mse(x, q_sym(x, 8, c8)[0]):.6e} | {mse(x, q_sym(x, 4, c4)[0]):.6e} |")
log("")

# ================================================================ C. 粒度
log("## C. 粒度：per-tensor / per-channel / group\n")
M, K = 4096, 4096
# 模拟 LLM 权重：每个输出通道的量级不同（真实模型里很常见，差 10~100 倍）
chan_scale = np.exp(rng.normal(0, 1.0, size=(M, 1)))
W = (rng.standard_normal((M, K)) * 0.02 * chan_scale).astype(np.float64)
log(f"权重 {M}x{K}，通道间量级差 max/min = {(chan_scale.max()/chan_scale.min()):.1f}x")
log("")
log("| 方式 | int8 相对误差 | int4 相对误差 |")
log("|---|---|---|")
for bits in (8, 4):
    pass
r_t8 = rel_err(W, q_sym(W, 8)[0])
r_t4 = rel_err(W, q_sym(W, 4)[0])
r_c8 = rel_err(W, q_sym_channel(W, 8)[0])
r_c4 = rel_err(W, q_sym_channel(W, 4)[0])
log(f"| per-tensor | {r_t8:.4f} | {r_t4:.4f} |")
log(f"| per-channel | {r_c8:.4f} | {r_c4:.4f} |")
for g in (128, 64, 32):
    log(f"| group={g} | {rel_err(W, q_group(W, 8, g)):.4f} | {rel_err(W, q_group(W, 4, g)):.4f} |")
log("")

# ================================================================ D. CLE
log("## D. CLE（Cross-Layer Equalization）在两层 ReLU 网络上\n")
D_in, D_mid, D_out = 128, 256, 128
# 故意制造通道间的不均衡：中间层输出通道的量级差异巨大
imb = np.exp(rng.normal(0, 2.0, size=(D_mid, 1)))
W1 = (rng.standard_normal((D_mid, D_in)) * 0.1 * imb).astype(np.float64)
b1 = np.zeros(D_mid)
W2 = (rng.standard_normal((D_out, D_mid)) * 0.05 / np.sqrt(imb.T)).astype(np.float64)
b2 = np.zeros(D_out)
X = rng.standard_normal((D_in, 512))


def net(W1, b1, W2, b2, X):
    h = np.maximum(W1 @ X + b1[:, None], 0)
    return W2 @ h + b2[:, None]


Y = net(W1, b1, W2, b2, X)


def quant_layer(W, bits):
    return q_sym(W, bits)[0]


for bits in (8, 4):
    y_q = net(quant_layer(W1, bits), b1, quant_layer(W2, bits), b2, X)
    log(f"- int{bits}，CLE 之前：输出相对误差 = {rel_err(Y, y_q):.4f}")

# CLE：per-channel 缩放，s_i = sqrt(r1_i / r2_i)
r1 = np.max(np.abs(W1), axis=1)          # 中间层第 i 个通道：W1 的第 i 行
r2 = np.max(np.abs(W2), axis=0)          # 同一个通道：W2 的第 i 列
s = np.sqrt(np.maximum(r1, 1e-12) / np.maximum(r2, 1e-12))
W1e = W1 / s[:, None]
b1e = b1 / s
W2e = W2 * s[None, :]
assert rel_err(Y, net(W1e, b1e, W2e, b2, X)) < 1e-10, "CLE 必须数学等价"
log(f"- CLE 后 FP 等价性检查：相对误差 = {rel_err(Y, net(W1e, b1e, W2e, b2, X)):.2e}（应为 0）")
for bits in (8, 4):
    y_q = net(quant_layer(W1e, bits), b1e, quant_layer(W2e, bits), b2, X)
    log(f"- int{bits}，CLE 之后：输出相对误差 = {rel_err(Y, y_q):.4f}")
log("")

# ================================================================ E. bias correction
log("## E. Bias correction\n")
Wq = q_sym(W2, 4)[0]
Xcal = rng.standard_normal((D_in, 512))
Xtest = rng.standard_normal((D_in, 512)) + 0.3   # 与校准集轻微偏移，模拟真实数据
h_cal = np.maximum(W1 @ Xcal + b1[:, None], 0)
h_test = np.maximum(W1 @ Xtest + b1[:, None], 0)
mu_h = h_cal.mean(axis=1)                        # 校准集上中间层激活的均值
dW = Wq - W2
b2c = b2 - dW @ mu_h                             # b' = b - ΔW·E[x]
err_before = np.abs(Wq @ h_test - W2 @ h_test).mean()
err_after = np.abs(Wq @ h_test + b2c[:, None] - (W2 @ h_test + b2[:, None])).mean()
log(f"- 4bit 权重，**测试集上**平均输出偏移：校正前 {err_before:.5f}")
log(f"- 加上 b' = b - ΔW·E[x] 之后：{err_after:.5f}（降低 {100*(1-err_after/err_before):.0f}%）")
log("")

# ================================================================ F. GPTQ vs RTN
log("## F. GPTQ vs RTN（4bit weight-only）\n")
Mf, Kf, Ntok = 256, 256, 512
Wf = (rng.standard_normal((Mf, Kf)) * 0.05).astype(np.float64)
# 激活带 outlier channel（LLM 的典型形态）
Xf = rng.standard_normal((Kf, Ntok))
out_ch = rng.choice(Kf, size=8, replace=False)
Xf[out_ch] *= 20.0
Yf = Wf @ Xf


def rtn(W, bits=4):
    return q_sym_channel(W, bits, axis=1)[0]


def gptq(W, X, bits=4, damp=0.01, block=64):
    Wq = W.copy()
    H = X @ X.T
    d = np.diag(np.diag(H))
    H = H + damp * d + 1e-8 * np.eye(H.shape[0])
    Hinv = np.linalg.inv(np.linalg.cholesky(H))
    Hinv = Hinv.T @ Hinv  # H^-1
    # 每输出通道一个 scale（GPTQ 的 per-channel symmetric）
    qmax = 2 ** (bits - 1) - 1
    s = np.maximum(np.max(np.abs(W), axis=1) / qmax, 1e-12)
    for j in range(W.shape[1]):
        w = Wq[:, j]
        q = s * np.clip(np.round(w / s), -qmax - 1, qmax)
        err = (w - q) / Hinv[j, j]
        Wq[:, j] = q
        if j + 1 < W.shape[1]:
            Wq[:, j + 1:] -= np.outer(err, Hinv[j, j + 1:])
    return Wq


Wr, Wg = rtn(Wf), gptq(Wf, Xf)
log(f"- 权重相对误差：RTN {rel_err(Wf, Wr):.4f} vs GPTQ {rel_err(Wf, Wg):.4f}（GPTQ 更差是正常的，它不优化权重）")
log(f"- **输出误差** ||WX-ŴX||/||WX||：RTN {rel_err(Yf, Wr @ Xf):.4f} vs GPTQ {rel_err(Yf, Wg @ Xf):.4f}")
log(f"- 输出 MSE：RTN {mse(Yf, Wr @ Xf):.3e} vs GPTQ {mse(Yf, Wg @ Xf):.3e}")
log("")

with open(".tmp/quant_basics_result.md", "w", encoding="utf-8") as f:
    f.write("\n".join(OUT) + "\n")
print("\n-> .tmp/quant_basics_result.md")
