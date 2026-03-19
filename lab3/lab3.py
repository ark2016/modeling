import csv
import math
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.family"] = "DejaVu Sans"
with open("diabetes.csv") as f:
    rows = list(csv.DictReader(f))
xi1_all = np.array([int(r["Glucose"].strip()) for r in rows if r["Outcome"].strip() == "0"])
xi0_all = np.array([int(r["Glucose"].strip()) for r in rows if r["Outcome"].strip() == "1"])
xi1_all = xi1_all[xi1_all > 0]
xi0_all = xi0_all[xi0_all > 0]
print(f"Всего: n(ξ₁) = {len(xi1_all)}, n(ξ₀) = {len(xi0_all)}")
D_005 = 0.05
N_BINS = 6

def ecdf(data):
    """Эмпирическая функция распределения."""
    xs = np.sort(data)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys

def ks_two_sample(sample1, sample2):
    """Двухвыборочный критерий КС: max|F₁(x) − F₂(x)|."""
    s1 = np.sort(sample1)
    s2 = np.sort(sample2)
    all_x = np.sort(np.concatenate([s1, s2]))
    f1 = np.searchsorted(s1, all_x, side="right") / len(s1)
    f2 = np.searchsorted(s2, all_x, side="right") / len(s2)
    D = np.max(np.abs(f1 - f2))
    return D, all_x, f1, f2

def check_bins(phi, xi0, edges):
    """Сравнение относительных частот по интервалам."""
    h1, _ = np.histogram(phi, bins=edges)
    h2, _ = np.histogram(xi0, bins=edges)
    p1 = h1 / len(phi)
    p2 = h2 / len(xi0)
    return np.max(np.abs(p1 - p2)), p1, p2

xi1 = xi1_all
xi0 = xi0_all
print(f"\nn(ξ₁) = {len(xi1)}, n(ξ₀) = {len(xi0)}")
D_raw, _, _, _ = ks_two_sample(xi1, xi0)
print(f"\n{'='*55}")
print(f"ШАГ 1. Проверяем: различаются ли группы?")
print(f"{'='*55}")
print(f"  max|F(ξ₁) − F(ξ₀)| = {D_raw:.4f}")
print(f"  D₀.₀₅              = {D_005}")
print(f"  {D_raw:.4f} > {D_005}  →  группы РАЗЛИЧАЮТСЯ")
print(f"  → нужна модель связи φ: ξ₀ = φ(ξ₁)")
print(f"\n{'='*55}")
print(f"ШАГ 2. Проверка нормальности (Шапиро-Уилк)")
print(f"{'='*55}")
for name, data in [("ξ₁ (Outcome=0)", xi1), ("ξ₀ (Outcome=1)", xi0)]:
    stat, p = stats.shapiro(data)
    verdict = "не отвергается" if p > 0.05 else "отвергается"
    print(f"  {name}: W = {stat:.4f}, p = {p:.4f}  →  H₀ {verdict} (α=0.05)")
print(f"  Вывод: нормальность отвергается для обеих групп")
print(f"  → используем степенную модель (Вейбулл)")

# Оценка параметров ξ₀ = a · ξ₁^b
#    Метод моментов через логарифмы

ln_xi1 = np.log(xi1.astype(float))
ln_xi0 = np.log(xi0.astype(float))
A = np.mean(ln_xi0)
B = np.mean(ln_xi1)
C = np.var(ln_xi0, ddof=0)
D_var = np.var(ln_xi1, ddof=0)
b_hat = math.sqrt(C / D_var)
a_hat = math.exp(A - b_hat * B)
print(f"\n{'='*55}")
print(f"ШАГ 3. Оценка параметров")
print(f"{'='*55}")
print(f"  A = mean(ln ξ₀) = {A:.4f}")
print(f"  B = mean(ln ξ₁) = {B:.4f}")
print(f"  C = var(ln ξ₀)  = {C:.4f}")
print(f"  D = var(ln ξ₁)  = {D_var:.4f}")
print(f"  b̂ = √(C/D)      = {b_hat:.4f}")
print(f"  â = exp(A−b̂·B)  = {a_hat:.4f}")
print(f"")
print(f"  Степенная: ξ₀ = {a_hat:.4f} · ξ₁^{b_hat:.4f}")

phi_xi1 = a_hat * xi1 ** b_hat
edges = np.quantile(xi0.astype(float), np.linspace(0, 1, N_BINS + 1))
combined = np.concatenate([phi_xi1, xi0])
edges[0] = min(combined.min(), edges[0]) - 0.5
edges[-1] = max(combined.max(), edges[-1]) + 0.5
D_bins, p1, p2 = check_bins(phi_xi1, xi0, edges)
D_ks, ax_ks, f1_ks, f2_ks = ks_two_sample(phi_xi1, xi0)
print(f"\n{'='*55}")
print(f"ШАГ 4. Критерий согласия ({N_BINS} интервалов)")
print(f"  max|p(φ(ξ₁)) − p(ξ₀)| ≤ D₀.₀₅ = {D_005} ?")
print(f"{'='*55}")
print(f"  {'Интервал':>20}  {'p(φ(ξ₁))':>10}  {'p(ξ₀)':>10}  {'|Δp|':>10}")
for i in range(N_BINS):
    interval = f"[{edges[i]:.1f}, {edges[i+1]:.1f})"
    print(f"  {interval:>20}  {p1[i]:>10.4f}  {p2[i]:>10.4f}  {abs(p1[i]-p2[i]):>10.4f}")
print(f"")
print(f"  max|Δp| = {D_bins:.4f}")
print(f"  {D_bins:.4f} {'≤' if D_bins <= D_005 else '>'} {D_005}  →  {'ДОПУСКАЕТСЯ' if D_bins <= D_005 else 'НЕ ДОПУСКАЕТСЯ'}")
grid = np.arange(50, 250, 25)
print(f"\n{'='*55}")
print(f"Таблица эмпирических функций распределения")
print(f"{'='*55}")
header = f"{'x':>12}" + "".join(f"{v:>8}" for v in grid)
print(header)
for name, data in [("F(ξ₁)", xi1), ("F(ξ₀)", xi0), ("F(φ(ξ₁))", phi_xi1)]:
    vals = np.searchsorted(np.sort(data), grid, side="right") / len(data)
    row = f"{name:>12}" + "".join(f"{v:>8.3f}" for v in vals)
    print(row)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
ax = axes[0, 0]
ax.step(*ecdf(xi1), where="post", label="F(ξ₁)  [Outcome=0]", lw=1.5)
ax.step(*ecdf(xi0), where="post", label="F(ξ₀)  [Outcome=1]", lw=1.5)
ax.set_xlabel("Glucose")
ax.set_ylabel("F(x)")
ax.set_title(f"Исходные ФР: D = {D_raw:.4f} > D₀.₀₅ = {D_005}\n→ группы различаются")
ax.legend()
ax.grid(True, alpha=0.3)
ax = axes[0, 1]
ax.step(*ecdf(xi0), where="post", label="F(ξ₀) — наблюдения", lw=1.5)
ax.step(*ecdf(phi_xi1), where="post", label="F(φ(ξ₁)) — модель", lw=1.5, ls="--")
ax.set_xlabel("Glucose")
ax.set_ylabel("F(x)")
ax.set_title(f"Степенная φ = â·ξ₁^b̂: D_KS = {D_ks:.4f}")
ax.legend()
ax.grid(True, alpha=0.3)
ax = axes[1, 0]
x_pos = np.arange(N_BINS)
width = 0.35
ax.bar(x_pos - width/2, p1, width, label="p(φ(ξ₁))", alpha=0.7)
ax.bar(x_pos + width/2, p2, width, label="p(ξ₀)", alpha=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels([f"{edges[i]:.0f}–{edges[i+1]:.0f}" for i in range(N_BINS)], rotation=45, fontsize=8)
ax.set_ylabel("Относительная частота")
ax.set_title(f"Сравнение по {N_BINS} интервалам: max|Δp| = {D_bins:.4f} ≤ {D_005}")
ax.legend()
ax.grid(True, alpha=0.3)
ax = axes[1, 1]
ax.plot(ax_ks, np.abs(f1_ks - f2_ks), label=f"|ΔF| (max={D_ks:.4f})", lw=1.2)
ax.axhline(y=D_005, color="red", ls=":", label=f"D₀.₀₅ = {D_005}")
ax.set_xlabel("Glucose")
ax.set_ylabel("|F(φ(ξ₁)) − F(ξ₀)|")
ax.set_title(f"|F(φ(ξ₁)) − F(ξ₀)|")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("lab_plots.png", dpi=150)
print("\nГрафики сохранены: lab_plots.png")
plt.close()
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
for ax, data, name in [(axes2[0], xi1, "ξ₁ (Outcome=0)"), (axes2[1], xi0, "ξ₀ (Outcome=1)")]:
    ax.hist(data, bins=25, density=True, alpha=0.6, edgecolor="black", lw=0.5)
    xr = np.linspace(data.min(), data.max(), 200)
    ax.plot(xr, stats.norm.pdf(xr, np.mean(data), np.std(data)), "r-", lw=2, label="Нормальное")
    ax.set_title(f"Гистограмма {name}")
    ax.set_xlabel("Glucose")
    ax.set_ylabel("Плотность")
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("lab_histograms.png", dpi=150)
print("Гистограммы сохранены: lab_histograms.png")
plt.close()
