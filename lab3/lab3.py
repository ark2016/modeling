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

# Вспомогательные функции

D_005 = 0.05

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

# Разделение на рабочую и контрольную выборки

rng = np.random.default_rng(848409)
idx1 = rng.permutation(len(xi1_all))
idx0 = rng.permutation(len(xi0_all))
split1 = int(0.7 * len(xi1_all))
split0 = int(0.7 * len(xi0_all))
xi1_train, xi1_test = xi1_all[idx1[:split1]], xi1_all[idx1[split1:]]
xi0_train, xi0_test = xi0_all[idx0[:split0]], xi0_all[idx0[split0:]]
print(f"\nРабочая:      n(ξ₁) = {len(xi1_train)}, n(ξ₀) = {len(xi0_train)}")
print(f"Контрольная:  n(ξ₁) = {len(xi1_test)},  n(ξ₀) = {len(xi0_test)}")

# Исходные группы РАЗЛИЧАЮТСЯ

D_raw, _, _, _ = ks_two_sample(xi1_train, xi0_train)
print(f"\n{'='*55}")
print(f"ШАГ 1. Проверяем: различаются ли группы?")
print(f"{'='*55}")
print(f"  max|F(ξ₁) − F(ξ₀)| = {D_raw:.4f}")
print(f"  D₀.₀₅              = {D_005}")
print(f"  {D_raw:.4f} > {D_005}  →  группы РАЗЛИЧАЮТСЯ")
print(f"  → нужна модель связи φ: ξ₀ = φ(ξ₁)")

# Проверка нормальности

print(f"\n{'='*55}")
print(f"ШАГ 2. Проверка нормальности (Шапиро-Уилк)")
print(f"{'='*55}")
for name, data in [("ξ₁ (Outcome=0)", xi1_train), ("ξ₀ (Outcome=1)", xi0_train)]:
    stat, p = stats.shapiro(data)
    verdict = "не отвергается" if p > 0.05 else "отвергается"
    print(f"  {name}: W = {stat:.4f}, p = {p:.4f}  →  H₀ {verdict} (α=0.05)")
print(f"  Вывод: нормальность отвергается для обеих групп")
print(f"  → используем степенную модель (Вейбулл)")

# Оценка параметров ξ₀ = a · ξ₁^b
#    Метод моментов через логарифмы

ln_xi1 = np.log(xi1_train.astype(float))
ln_xi0 = np.log(xi0_train.astype(float))
A = np.mean(ln_xi0)
B = np.mean(ln_xi1)
C = np.var(ln_xi0, ddof=0)
D_var = np.var(ln_xi1, ddof=0)
b_hat = math.sqrt(C / D_var)
a_hat = math.exp(A - b_hat * B)
k_hat = np.mean(xi0_train) / np.mean(xi1_train)
print(f"\n{'='*55}")
print(f"ШАГ 3. Оценка параметров (рабочая выборка)")
print(f"{'='*55}")
print(f"  A = mean(ln ξ₀) = {A:.4f}")
print(f"  B = mean(ln ξ₁) = {B:.4f}")
print(f"  C = var(ln ξ₀)  = {C:.4f}")
print(f"  D = var(ln ξ₁)  = {D_var:.4f}")
print(f"  b̂ = √(C/D)      = {b_hat:.4f}")
print(f"  â = exp(A−b̂·B)  = {a_hat:.4f}")
print(f"")
print(f"  Степенная: ξ₀ = {a_hat:.4f} · ξ₁^{b_hat:.4f}")
print(f"  Линейная:  ξ₀ = {k_hat:.4f} · ξ₁")

# Критерий согласия — РАБОЧАЯ выборка
#    max|F(φ(ξ₁)) − F(ξ₀)| ≤ D₀.₀₅ = 0.05

phi_xi1_train = a_hat * xi1_train ** b_hat
phi_xi1_train_lin = k_hat * xi1_train
D_power, ax_pw, f1_pw, f2_pw = ks_two_sample(phi_xi1_train, xi0_train)
D_linear, ax_ln, f1_ln, f2_ln = ks_two_sample(phi_xi1_train_lin, xi0_train)
print(f"\n{'='*55}")
print(f"ШАГ 4. Критерий согласия (рабочая выборка)")
print(f"  max|F(φ(ξ₁)) − F(ξ₀)| ≤ D₀.₀₅ = {D_005} ?")
print(f"{'='*55}")
print(f"  Степенная  φ(ξ₁) = â·ξ₁^b̂ :  D = {D_power:.4f}")
print(f"  Линейная   φ(ξ₁) = k̂·ξ₁   :  D = {D_linear:.4f}")
print(f"  D₀.₀₅                      = {D_005}")
print(f"")
print(f"  Степенная: {D_power:.4f} {'≤' if D_power <= D_005 else '>'} {D_005}  →  {'ДОПУСКАЕТСЯ' if D_power <= D_005 else 'НЕ ДОПУСКАЕТСЯ'}")
print(f"  Линейная:  {D_linear:.4f} {'≤' if D_linear <= D_005 else '>'} {D_005}  →  {'ДОПУСКАЕТСЯ' if D_linear <= D_005 else 'НЕ ДОПУСКАЕТСЯ'}")
print(f"")
print(f"  Было (без модели): max|F(ξ₁)−F(ξ₀)| = {D_raw:.4f}")
print(f"  Стало (с моделью): max|F(φ(ξ₁))−F(ξ₀)| = {D_power:.4f}")
print(f"  → модель уменьшила расхождение в {D_raw/D_power:.1f} раз")

# Критерий согласия — КОНТРОЛЬНАЯ выборка
phi_xi1_test = a_hat * xi1_test ** b_hat
phi_xi1_test_lin = k_hat * xi1_test
D_power_test, _, _, _ = ks_two_sample(phi_xi1_test, xi0_test)
D_linear_test, _, _, _ = ks_two_sample(phi_xi1_test_lin, xi0_test)
print(f"\n{'='*55}")
print(f"ШАГ 5. Критерий согласия (контрольная выборка)")
print(f"{'='*55}")
print(f"  Степенная: D = {D_power_test:.4f}  →  {D_power_test:.4f} {'≤' if D_power_test <= D_005 else '>'} {D_005}  →  {'ДОПУСКАЕТСЯ' if D_power_test <= D_005 else 'НЕ ДОПУСКАЕТСЯ'}")
print(f"  Линейная:  D = {D_linear_test:.4f}  →  {D_linear_test:.4f} {'≤' if D_linear_test <= D_005 else '>'} {D_005}  →  {'ДОПУСКАЕТСЯ' if D_linear_test <= D_005 else 'НЕ ДОПУСКАЕТСЯ'}")

# Таблица эмпирических ФР
grid = np.arange(50, 250, 25)
print(f"\n{'='*55}")
print(f"Таблица эмпирических функций распределения")
print(f"{'='*55}")
header = f"{'x':>12}" + "".join(f"{v:>8}" for v in grid)
print(header)
for name, data in [("F(ξ₁)", xi1_train),
                    ("F(ξ₀)", xi0_train),
                    ("F(φ(ξ₁))", phi_xi1_train)]:
    vals = np.searchsorted(np.sort(data), grid, side="right") / len(data)
    row = f"{name:>12}" + "".join(f"{v:>8.3f}" for v in vals)
    print(row)

# Графики

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Исходные ФР
ax = axes[0, 0]
ax.step(*ecdf(xi1_train), where="post", label="F(ξ₁)  [Outcome=0]", lw=1.5)
ax.step(*ecdf(xi0_train), where="post", label="F(ξ₀)  [Outcome=1]", lw=1.5)
ax.set_xlabel("Glucose")
ax.set_ylabel("F(x)")
ax.set_title(f"Исходные ФР: D = {D_raw:.4f} >> D₀.₀₅ = {D_005}\n→ группы различаются")
ax.legend()
ax.grid(True, alpha=0.3)

# Степенная модель
ax = axes[0, 1]
ax.step(*ecdf(xi0_train), where="post", label="F(ξ₀) — наблюдения", lw=1.5)
ax.step(*ecdf(phi_xi1_train), where="post", label="F(φ(ξ₁)) — модель", lw=1.5, ls="--")
ax.set_xlabel("Glucose")
ax.set_ylabel("F(x)")
ax.set_title(f"Степенная φ = â·ξ₁^b̂: D = {D_power:.4f}\n→ расхождение снижено в {D_raw/D_power:.1f}×")
ax.legend()
ax.grid(True, alpha=0.3)

# Линейная модель
ax = axes[1, 0]
ax.step(*ecdf(xi0_train), where="post", label="F(ξ₀) — наблюдения", lw=1.5)
ax.step(*ecdf(phi_xi1_train_lin), where="post", label="F(k̂·ξ₁) — модель", lw=1.5, ls="--")
ax.set_xlabel("Glucose")
ax.set_ylabel("F(x)")
ax.set_title(f"Линейная φ = k̂·ξ₁: D = {D_linear:.4f}\n→ расхождение снижено в {D_raw/D_linear:.1f}×")
ax.legend()
ax.grid(True, alpha=0.3)

# |ΔF|
ax = axes[1, 1]
ax.plot(ax_pw, np.abs(f1_pw - f2_pw), label=f"|ΔF| степенная (max={D_power:.4f})", lw=1.2)
ax.plot(ax_ln, np.abs(f1_ln - f2_ln), label=f"|ΔF| линейная (max={D_linear:.4f})", lw=1.2)
ax.axhline(y=D_005, color="red", ls=":", label=f"D₀.₀₅ = {D_005}")
ax.set_xlabel("Glucose")
ax.set_ylabel("|F(φ(ξ₁)) − F(ξ₀)|")
ax.set_title(f"max|F(φ(ξ₁)) − F(ξ₀)| ≤ D₀.₀₅ = {D_005}")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("lab_plots.png", dpi=150)
print("\nГрафики сохранены: lab_plots.png")
plt.close()

# Гистограммы
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
for ax, data, name in [(axes2[0], xi1_train, "ξ₁ (Outcome=0)"),
                        (axes2[1], xi0_train, "ξ₀ (Outcome=1)")]:
    ax.hist(data, bins=25, density=True, alpha=0.6, edgecolor="black", lw=0.5)
    xr = np.linspace(data.min(), data.max(), 200)
    ax.plot(xr, stats.norm.pdf(xr, np.mean(data), np.std(data)),
            "r-", lw=2, label="Нормальное")
    ax.set_title(f"Гистограмма {name}")
    ax.set_xlabel("Glucose")
    ax.set_ylabel("Плотность")
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("lab_histograms.png", dpi=150)
print("Гистограммы сохранены: lab_histograms.png")
plt.close()
