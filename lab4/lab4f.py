import numpy as np
import csv

with open("../lab3/diabetes.csv", "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    data = np.array([[float(x) for x in row] for row in reader])

X = data[:, :-1]
N, d = X.shape
short = [f"X{k+1}" for k in range(d)]

def linreg(x, y):
    n = len(x)
    mx = x.mean()
    my = y.mean()
    a = np.sum((x - mx) * (y - my)) / np.sum((x - mx) ** 2)
    b = my - a * mx
    return a, b

def ks_normal(residuals):
    r = (residuals - residuals.mean()) / residuals.std(ddof=1)
    r_sorted = np.sort(r)
    n = len(r)
    from math import erf, sqrt
    def cdf(x):
        return 0.5 * (1 + erf(x / sqrt(2)))
    emp = np.arange(1, n + 1) / n
    theo = np.array([cdf(v) for v in r_sorted])
    D_plus = np.max(emp - theo)
    D_minus = np.max(theo - (np.arange(n) / n))
    D = max(D_plus, D_minus)
    sqrt_n = np.sqrt(n)
    lam = (sqrt_n + 0.12 + 0.11 / sqrt_n) * D
    p = 0.0
    for k in range(1, 101):
        p += 2 * ((-1) ** (k - 1)) * np.exp(-2 * (k ** 2) * (lam ** 2))
    p = max(0.0, min(1.0, p))
    return D, p

pairs = [(0, 7), (3, 4), (3, 5), (2, 7), (1, 4), (2, 5), (1, 5), (1, 7)]

print("Pair | alpha | beta | D | p-value | verdict")
print("-" * 70)
results = []
for i, j in pairs:
    x = X[:, i]
    y = X[:, j]
    a, b = linreg(x, y)
    res = y - (a * x + b)
    D, p = ks_normal(res)
    verdict = "подтверждено" if p > 0.05 else "не подтверждено"
    print(f"{short[i]}–{short[j]} | α={a:+.4f} | β={b:+.4f} | D={D:.4f} | p={p:.4f} | {verdict}")
    results.append((short[i], short[j], a, b, D, p, verdict))

with open("ks_results.csv", "w") as f:
    f.write("pair,alpha,beta,D,p_value,verdict\n")
    for r in results:
        f.write(f"{r[0]}-{r[1]},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f},{r[5]:.4f},{r[6]}\n")