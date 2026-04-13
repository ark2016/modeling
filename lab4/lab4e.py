import numpy as np
import csv

with open("../lab3/diabetes.csv", "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    data = np.array([[float(x) for x in row] for row in reader])

X = data[:, :-1]
N, d = X.shape
short = [f"X{k+1}" for k in range(d)]

def rankdata(x):
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j < len(x) - 1 and x[order[j + 1]] == x[order[j]]:
            j += 1
        if j > i:
            avg_rank = np.mean(ranks[order[i:j + 1]])
            ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    return ranks

def spearman(x, y):
    rx = rankdata(x)
    ry = rankdata(y)
    d = rx - ry
    d2 = np.sum(d**2)
    return 1 - 6 * d2 / (N * (N**2 - 1))

S = np.zeros((d, d))
for i in range(d):
    for j in range(d):
        S[i, j] = spearman(X[:, i], X[:, j])

print("Spearman correlation (rho_s = 1 - 6*sum(d_i^2) / (n*(n^2-1)))")
print(f"N={N}, d={d}\n")
print("\t" + "\t".join(short))
for i in range(d):
    row = "\t".join([f"{S[i,j]:.3f}" for j in range(d)])
    print(f"{short[i]}\t{row}")

with open("spearman.csv", "w") as f:
    f.write("," + ",".join(short) + "\n")
    for i in range(d):
        f.write(short[i] + "," + ",".join([f"{S[i,j]:.3f}" for j in range(d)]) + "\n")