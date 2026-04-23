import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import erf, sqrt


def mean_(x):
    s = 0.0
    for v in x:
        s += v
    return s / len(x)


def std_(x, ddof=1):
    m = mean_(x)
    s = 0.0
    for v in x:
        d = v - m
        s += d * d
    return (s / (len(x) - ddof)) ** 0.5


def median_(x):
    xs = sorted(x)
    n = len(xs)
    return xs[n // 2] if n % 2 == 1 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def percentile_(x, p):
    xs = sorted(x)
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] * (1 - (k - lo)) + xs[hi] * (k - lo)


def pearson_(x, y):
    mx, my = mean_(x), mean_(y)
    num = sx = sy = 0.0
    for i in range(len(x)):
        dx = x[i] - mx
        dy = y[i] - my
        num += dx * dy
        sx  += dx * dx
        sy  += dy * dy
    return num / (sx * sy) ** 0.5


def normal_cdf_(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def transpose_(A):
    n, m = len(A), len(A[0])
    return [[A[i][j] for i in range(n)] for j in range(m)]


def matmul_(A, B):
    nA, mA = len(A), len(A[0])
    mB = len(B[0])
    C = [[0.0] * mB for _ in range(nA)]
    for i in range(nA):
        for j in range(mB):
            s = 0.0
            for k in range(mA):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C


def matvec_(A, b):
    return [sum(A[i][k] * b[k] for k in range(len(b))) for i in range(len(A))]


def inverse_(A):
    n = len(A)
    M = [[A[i][j] for j in range(n)] + [1.0 if i == j else 0.0 for j in range(n)]
         for i in range(n)]
    for col in range(n):
        # выбор главного элемента по столбцу
        piv = col
        for i in range(col + 1, n):
            if abs(M[i][col]) > abs(M[piv][col]):
                piv = i
        if abs(M[piv][col]) < 1e-12:
            raise ValueError("Матрица вырождена")
        if piv != col:
            M[col], M[piv] = M[piv], M[col]
        # нормируем ведущую строку
        p = M[col][col]
        for j in range(2 * n):
            M[col][j] /= p
        # зануляем столбец у остальных строк
        for i in range(n):
            if i != col:
                f = M[i][col]
                if f != 0.0:
                    for j in range(2 * n):
                        M[i][j] -= f * M[col][j]
    return [[M[i][j + n] for j in range(n)] for i in range(n)]


def ols_fit_(X, y):
    Xt = transpose_(X)
    XtX_inv = inverse_(matmul_(Xt, X))
    Xty = matvec_(Xt, y)
    return matvec_(XtX_inv, Xty), XtX_inv


def predict_(X, beta):
    return matvec_(X, beta)


def mae_(y, y_hat):
    s = 0.0
    for i in range(len(y)):
        d = y[i] - y_hat[i]
        s += d if d >= 0 else -d
    return s / len(y)


def rmse_(y, y_hat):
    s = 0.0
    for i in range(len(y)):
        d = y[i] - y_hat[i]
        s += d * d
    return (s / len(y)) ** 0.5


def mape_(y, y_hat):
    s = 0.0
    n_used = 0
    for i in range(len(y)):
        if y[i] == 0:
            continue
        d = y[i] - y_hat[i]
        s += (d if d >= 0 else -d) / abs(y[i])
        n_used += 1
    return s / n_used * 100.0 if n_used > 0 else float('nan')


def r_squared_(y, y_hat):
    ybar = mean_(y)
    ss_res = sum((y[i] - y_hat[i]) ** 2 for i in range(len(y)))
    ss_tot = sum((y[i] - ybar) ** 2     for i in range(len(y)))
    return 1.0 - ss_res / ss_tot


rng = np.random.default_rng(42)
xs = rng.normal(0, 1, 200); zs = rng.normal(0, 1, 200)
eps = rng.normal(0, 0.5, 200)
y_syn = [2 + 3 * xs[i] + 5 * zs[i] + eps[i] for i in range(200)]
X_syn = [[1.0, xs[i], zs[i]] for i in range(200)]
b_syn, _ = ols_fit_(X_syn, y_syn)
print(f"Самопроверка МНК: β̂ = "
      f"[{b_syn[0]:.3f}, {b_syn[1]:.3f}, {b_syn[2]:.3f}]  (ожидалось [2, 3, 5])")


df_full = pd.read_csv('diabetes.csv')
print(f"\nИсходная выборка: {len(df_full)} наблюдений (данные используются как есть, "
      f"без медицинской интерпретации значений)")

y_name  = 'Glucose'
x_names = ['Pregnancies', 'BloodPressure', 'SkinThickness',
           'BMI', 'DiabetesPedigreeFunction', 'Age']


train = df_full.loc[df_full.index % 2 == 0].reset_index(drop=True)
test  = df_full.loc[df_full.index % 2 == 1].reset_index(drop=True)
print(f"\nРазделение 50/50 по исходным индексам: "
      f"train = {len(train)}, test = {len(test)}")


print("\nОписательная статистика по train:")
for c in x_names + [y_name]:
    x = train[c].tolist()
    print(f"  {c:26s}  mean={mean_(x):7.2f}  std={std_(x):6.2f}  "
          f"min={min(x):7.2f}  max={max(x):7.2f}")

print("\nВыбросы (3σ / IQR / MAD) на train:")
for c in x_names + [y_name]:
    x = train[c].tolist()
    mu, sd = mean_(x), std_(x)
    q1, q3 = percentile_(x, 25), percentile_(x, 75)
    med = median_(x)
    mad = median_([abs(v - med) for v in x])
    n3  = sum(1 for v in x if v < mu - 3 * sd or v > mu + 3 * sd)
    niq = sum(1 for v in x if v < q1 - 1.5 * (q3 - q1)
                          or v > q3 + 1.5 * (q3 - q1))
    nmd = sum(1 for v in x if abs(v - med) > 3 * 1.4826 * mad) if mad > 0 else 0
    print(f"  {c:26s}  3σ={n3:3d}  IQR={niq:3d}  MAD={nmd:3d}")


print("\nПирсоновские корреляции (Y, X_j) по train:")
r_crit = 1.96 / (len(train) - 2 + 1.96 ** 2) ** 0.5
print(f"  H0: ρ = 0;  r_крит(0.05, n={len(train)}) ≈ {r_crit:.3f}")
y_train_list = train[y_name].tolist()
for c in x_names:
    r = pearson_(train[c].tolist(), y_train_list)
    verdict = "подтверждено" if abs(r) > r_crit else "НЕ подтверждено"
    print(f"  r(Y, {c:26s}) = {r:+.3f}  → H0 отвергается? {verdict}")


mu_tr = {c: mean_(train[c].tolist()) for c in x_names}
sd_tr = {c: std_(train[c].tolist())  for c in x_names}

def standardize(df_part):
    return [[(df_part[c].iloc[i] - mu_tr[c]) / sd_tr[c] for c in x_names]
            for i in range(len(df_part))]

Z_tr = standardize(train)
Z_te = standardize(test)
y_tr = train[y_name].tolist()
y_te = test[y_name].tolist()

X_tr = [[1.0] + row for row in Z_tr]
X_te = [[1.0] + row for row in Z_te]


beta, XtX_inv = ols_fit_(X_tr, y_tr)
print("\nОценки коэффициентов:")
names = ['β0 (intercept)'] + [f'β{i+1} ({x_names[i]})' for i in range(len(x_names))]
for name, b in zip(names, beta):
    print(f"  {name:42s} = {b:+.4f}")


n  = len(y_tr)
p  = len(beta)
k  = p - 1
y_hat_tr = predict_(X_tr, beta)
e_tr = [y_tr[i] - y_hat_tr[i] for i in range(n)]
ss_res = sum(ei * ei for ei in e_tr)
ss_tot = sum((y_tr[i] - mean_(y_tr)) ** 2 for i in range(n))
sigma2 = ss_res / (n - p)
SE   = [sqrt(sigma2 * XtX_inv[j][j]) for j in range(p)]
t_st = [beta[j] / SE[j] for j in range(p)]
pval = [2.0 * (1.0 - normal_cdf_(abs(ts))) for ts in t_st]
R2_train = 1.0 - ss_res / ss_tot
F_stat   = (R2_train / k) / ((1.0 - R2_train) / (n - p))

# при df = n - p = 377 t ≈ N(0,1): t_крит(0.025) ≈ 1.96;
# F_крит(0.05, 6, 377) ≈ 2.12 (таблица Фишера)
t_crit = 1.96
F_crit = 2.12

print("\n=== Проверка гипотез ===")
print(f"H0 (модель в целом): β1 = ... = βk = 0")
print(f"  F = {F_stat:.3f},  F_крит(0.05; {k}, {n-p}) ≈ {F_crit}")
print(f"  F > F_крит: {'ДА → H0 отвергается → модель значима' if F_stat > F_crit else 'НЕТ'}")

print(f"\nH0 (для каждого βj): βj = 0;  t_крит(0.025, df={n-p}) ≈ {t_crit}")
print(f"  {'Коэффициент':36s} {'β̂':>10s} {'SE':>7s} {'t':>7s} {'p':>7s}  вердикт")
for j in range(p):
    verdict = "ЗНАЧИМ" if abs(t_st[j]) > t_crit else "не значим"
    print(f"  {names[j]:36s} {beta[j]:+10.4f} {SE[j]:7.3f} "
          f"{t_st[j]:+7.3f} {pval[j]:7.4f}  {verdict}")


y_hat_te = predict_(X_te, beta)

print("\n*** MAE (обязательная метрика) ***")
print(f"    MAE_train = {mae_(y_tr, y_hat_tr):.3f} мг/дл")
print(f"    MAE_test  = {mae_(y_te, y_hat_te):.3f} мг/дл")

print("\n--- Справочно ---")
print(f"    RMSE_test  = {rmse_(y_te, y_hat_te):.3f} мг/дл")
print(f"    MAPE_test  = {mape_(y_te, y_hat_te):.2f} %")
print(f"    R²_train   = {R2_train:.4f}")
print(f"    R²_test    = {r_squared_(y_te, y_hat_te):.4f}")

e_te = [y_te[i] - y_hat_te[i] for i in range(len(y_te))]
print(f"    mean(e)    = {mean_(e_te):+.3f} мг/дл")
print(f"    std(e)     = {std_(e_te):.3f} мг/дл")


fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for i, c in enumerate(x_names):
    ax = axes.flat[i]
    ax.scatter(df_full[c], df_full[y_name], alpha=0.4, s=12)
    ax.set_xlabel(c); ax.set_ylabel('Glucose')
plt.tight_layout(); plt.savefig('scatter.png', dpi=110); plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
axes[0].scatter(y_hat_te, y_te, alpha=0.5, s=15)
lo = min(min(y_te), min(y_hat_te)); hi = max(max(y_te), max(y_hat_te))
axes[0].plot([lo, hi], [lo, hi], 'r--', lw=1)
axes[0].set_xlabel('Предсказанное ŷ'); axes[0].set_ylabel('Реальное y')
axes[0].set_title('Predicted vs Actual')

axes[1].scatter(y_hat_te, e_te, alpha=0.5, s=15)
axes[1].axhline(0, color='r', lw=1)
axes[1].set_xlabel('Предсказанное ŷ'); axes[1].set_ylabel('Остаток e')
axes[1].set_title('Residuals vs Predicted')

axes[2].hist(e_te, bins=28, edgecolor='black', alpha=0.8)
axes[2].axvline(0, color='r', lw=1.5)
axes[2].axvline(mean_(e_te), color='orange', lw=1.5, linestyle='--',
                label=f'mean = {mean_(e_te):.2f}')
axes[2].set_xlabel('Остаток e'); axes[2].set_ylabel('Частота')
axes[2].set_title('Распределение остатков')
axes[2].legend()

plt.tight_layout(); plt.savefig('diagnostics.png', dpi=110); plt.close()