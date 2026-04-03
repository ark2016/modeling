import numpy as np
import csv
import matplotlib.pyplot as plt

with open("../lab3/diabetes.csv", "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    data = np.array([[float(x) for x in row] for row in reader])

X = data[:, :-1]
N, d = X.shape
means = X.mean(axis=0)
Bin = (X >= means).astype(int)
short = [f"X{k+1}" for k in range(d)]

R_bern = np.zeros((d, d))
Q_ass = np.zeros((d, d))
Q_col = np.zeros((d, d))
Q_con = np.zeros((d, d))

for i in range(d):
    for j in range(d):
        a = np.sum((Bin[:, i] == 1) & (Bin[:, j] == 1))
        b = np.sum((Bin[:, i] == 1) & (Bin[:, j] == 0))
        c = np.sum((Bin[:, i] == 0) & (Bin[:, j] == 1))
        dd = np.sum((Bin[:, i] == 0) & (Bin[:, j] == 0))
        P_A = (a + b) / N
        P_B = (a + c) / N
        P_AB = a / N
        P_A_given_B = P_AB / P_B if P_B > 0 else 0
        R_bern[i, j] = P_B * (P_A_given_B - P_A)
        ad = a * dd
        bc = b * c
        Q_ass[i, j] = (ad - bc) / (ad + bc) if (ad + bc) != 0 else 0
        sad = np.sqrt(a * dd)
        sbc = np.sqrt(b * c)
        Q_col[i, j] = (sad - sbc) / (sad + sbc) if (sad + sbc) != 0 else 0
        denom = np.sqrt((a + b) * (c + dd) * (a + c) * (b + dd))
        Q_con[i, j] = (ad - bc) / denom if denom != 0 else 0

def print_matrix(name, M):
    print(f"\n{name}")
    print("\t" + "\t".join(short))
    for i in range(d):
        row = "\t".join([f"{M[i,j]:.3f}" for j in range(d)])
        print(f"{short[i]}\t{row}")

def save_csv(filename, M):
    with open(filename, "w") as f:
        f.write("," + ",".join(short) + "\n")
        for i in range(d):
            f.write(short[i] + "," + ",".join([f"{M[i,j]:.3f}" for j in range(d)]) + "\n")

def plot_heatmap(title, M, filename, vmin, vmax):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(M, cmap="RdBu", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(d))
    ax.set_yticks(range(d))
    ax.set_xticklabels(short, fontsize=9)
    ax.set_yticklabels(short, fontsize=9)
    for i in range(d):
        for j in range(d):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)

print("Бинаризация по среднему: >= mean -> 1, < mean -> 0")
print(f"N = {N}, d = {d}")

print_matrix("1) Коэфф. Бернштейна: R_B = P(B)(P(A|B) - P(A))", R_bern)
print_matrix("2) Коэфф. контингенции (Пирсона): (ad - bc) / sqrt((a+b)(c+d)(a+c)(b+d))", Q_con)
print_matrix("3) Коэфф. коллигации (Юла): (sqrt(ad) - sqrt(bc)) / (sqrt(ad) + sqrt(bc))", Q_col)
print_matrix("4) Коэфф. ассоциации: (ad - bc) / (ad + bc)", Q_ass)

save_csv("bernstein.csv", R_bern)
save_csv("contingency.csv", Q_con)
save_csv("colligation_yule.csv", Q_col)
save_csv("association.csv", Q_ass)

plot_heatmap("R_B = P(B)(P(A|B) - P(A))", R_bern, "heatmap_bernstein.png", -0.25, 0.25)
plot_heatmap("Contingency (Pearson)", Q_con, "heatmap_contingency.png", -1, 1)
plot_heatmap("Colligation (Yule)", Q_col, "heatmap_colligation.png", -1, 1)
plot_heatmap("Association", Q_ass, "heatmap_association.png", -1, 1)

print("\nПример таблицы четырёх полей (X1 vs X2):")
a = np.sum((Bin[:, 0] == 1) & (Bin[:, 1] == 1))
b = np.sum((Bin[:, 0] == 1) & (Bin[:, 1] == 0))
c = np.sum((Bin[:, 0] == 0) & (Bin[:, 1] == 1))
dd = np.sum((Bin[:, 0] == 0) & (Bin[:, 1] == 0))
print(f"\t\tX2=1\tX2=0")
print(f"X1=1\t\t{a}\t{b}")
print(f"X1=0\t\t{c}\t{dd}")
print(f"Бернштейн  = {R_bern[0,1]:.3f}")
print(f"Контингенц.= {Q_con[0,1]:.3f}")
print(f"Коллигация = {Q_col[0,1]:.3f}")
print(f"Ассоциация = {Q_ass[0,1]:.3f}")

print("\nТоп-10 связей (по |ассоциации|):")
pairs = []
for i in range(d):
    for j in range(i+1, d):
        pairs.append((abs(Q_ass[i,j]), R_bern[i,j], Q_con[i,j], Q_col[i,j], Q_ass[i,j], short[i], short[j]))
pairs.sort(reverse=True)
print(f"  {'Пара':>10}  {'Берн.':>7}  {'Конт.':>7}  {'Колл.':>7}  {'Асс.':>7}")
for _, rb, qk, qc, qa, ni, nj in pairs[:10]:
    print(f"  {ni}--{nj:>3}  {rb:+.3f}  {qk:+.3f}  {qc:+.3f}  {qa:+.3f}")