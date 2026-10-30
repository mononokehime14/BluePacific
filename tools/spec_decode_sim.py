"""投机解码（speculative decoding）仿真：验证无损性 + 算加速比。

跑法：  python tools/spec_decode_sim.py
产物：  .tmp/spec_decode_result.md

三个实验：
  A. 无损性：用 draft q + 拒绝采样，从 p 里采样 —— 经验分布必须等于 p
  B. 接受率：给定 p 和不同"草稿质量"，测实际接受率
  C. 加速比：E[tokens]/step ÷ (1 + c·γ)，扫 γ 找最优
"""

import numpy as np

rng = np.random.default_rng(7)
OUT = []


def log(s=""):
    print(s)
    OUT.append(str(s))


# ================================================================ A. 无损性
log("## A. 无损性验证（拒绝采样）\n")

V = 50
p = rng.dirichlet(np.ones(V) * 0.7)          # target 分布
q = rng.dirichlet(np.ones(V) * 0.7)          # draft 分布（故意和 p 不同）


def sample_from_p_with_draft(p, q, n):
    """标准 speculative sampling：draft 出一个 token，按 min(1,p/q) 接受。"""
    out = []
    while len(out) < n:
        x = rng.choice(V, p=q)               # draft 采一个
        u = rng.random()
        if u < min(1.0, p[x] / q[x]):        # 接受
            out.append(x)
    return np.array(out)


def sample_rejection_exact(p, q, n):
    """严格的拒绝采样：拒绝后从 norm(max(0, p-q)) 重采，保证分布精确等于 p。"""
    out = []
    while len(out) < n:
        x = rng.choice(V, p=q)
        r = rng.random()
        if r < min(1.0, p[x] / q[x]):
            out.append(x)
        else:
            d = np.maximum(p - q, 0)
            if d.sum() <= 1e-12:
                out.append(x)
            else:
                out.append(rng.choice(V, p=d / d.sum()))
    return np.array(out)


N = 400_000
s1 = sample_from_p_with_draft(p, q, N)
s2 = sample_rejection_exact(p, q, N)


def tv(a, b):
    return 0.5 * float(np.abs(a - b).sum())


emp1 = np.bincount(s1, minlength=V) / N
emp2 = np.bincount(s2, minlength=V) / N
log(f"- 只接受、拒绝后重采同样的 draft token：TV 距离 = {tv(emp1, p):.4f}  ← **有偏**")
log(f"- 严格拒绝采样（拒绝后从 max(p-q,0) 重采）：TV 距离 = {tv(emp2, p):.4f}  ← **无偏**")
log(f"- 参考：两个独立 40 万样本之间的自然噪声约 {tv(p, np.bincount(rng.choice(V, p=p, size=N), minlength=V)/N):.4f}")
log("")

# ================================================================ B. 接受率
log("## B. 草稿质量 → 接受率\n")
log("用 q = softmax(log p / T) 构造不同质量的 draft（T 越小越像 target）\n")
log("| 温度 T | 接受率 α | 与 p 的 TV 距离 |")
log("|---|---|---|")
for T in [0.5, 0.7, 0.85, 1.0, 1.3, 2.0]:
    lp = np.log(p + 1e-12) / T
    qq = np.exp(lp - lp.max())
    qq /= qq.sum()
    alpha = float(np.minimum(p, qq).sum())     # 单步理论接受率
    log(f"| {T} | {alpha:.3f} | {tv(p, qq):.3f} |")
log("")
log("→ 注意 α = Σ min(p, q)，这就是单 token 接受率的解析式，**不用仿真**。")
log("")

# ================================================================ C. 加速比
log("## C. 加速比随 γ（每次 draft 的 token 数）变化\n")


def speedup(alpha, gamma, c):
    accepted = (1 - alpha ** (gamma + 1)) / (1 - alpha)   # 每步平均产出 token
    return accepted / (1 + c * gamma)


log("公式：speedup = [(1-α^(γ+1))/(1-α)] / (1 + c·γ)，c = draft 成本 / target 成本\n")
for alpha in (0.6, 0.8, 0.9):
    for c in (0.05, 0.1, 0.2):
        best = max(((g, speedup(alpha, g, c)) for g in range(1, 17)), key=lambda t: t[1])
        log(f"- α={alpha}, c={c}: 最优 γ={best[0]}, 加速 {best[1]:.2f}×")
log("")

log("α=0.8、c=0.1 时逐 γ 的加速比：")
log("")
log("| γ | 每步产出 token | 加速比 |")
log("|---|---|---|")
for g in range(1, 13):
    log(f"| {g} | {(1-0.8**(g+1))/0.2:.2f} | {speedup(0.8, g, 0.1):.2f}× |")
log("")

# ================================================================ D. 端侧视角
log("## D. 端侧 vs 云端：为什么投机解码在端侧更划算\n")
log("decode 是带宽墙（note 03）：batch=1 时验证 γ+1 个 token 只多读一次激活，")
log("权重只读一次 >> 激活，所以 **verify 的边际成本接近 0**。\n")
log("| 场景 | c = draft/target 成本 | 说明 |")
log("|---|---|---|")
log("| 云端大 batch | draft 也要占算力，c 接近参数比 | 收益被吞吐摊薄 |")
log("| 端侧 batch=1 | c ≈ draft 参数 / target 参数（0.05~0.1） | 收益最大 |")
log("")
log("反例：如果 draft 模型太大（c > 0.3），最优 γ 会掉到 1~2，加速比接近 1 —— ")
log("**投机解码的收益上限由 c 决定，不是由 α 决定。**")

with open(".tmp/spec_decode_result.md", "w", encoding="utf-8") as f:
    f.write("# 投机解码仿真结果\n\n" + "\n".join(OUT) + "\n")
print("\n-> .tmp/spec_decode_result.md")
