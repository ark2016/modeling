import numpy as np
import csv
import matplotlib.pyplot as plt

with open("../lab3/diabetes.csv", "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    data = np.array([[float(x) for x in row] for row in reader])

X = data[:, :-1]
y_true = data[:, -1].astype(int)
N, d = X.shape
short = [f"X{k+1}" for k in range(d)]

print("I. BINARIZATION")
means = X.mean(axis=0)
Bin = (X >= means).astype(int)
print(f"N={N}, d={d}")
for j in range(d):
    n1 = np.sum(Bin[:, j] == 1)
    print(f"  {short[j]}: mean={means[j]:.2f}, n1={n1}, n0={N-n1}")

print("\nII. COEFFICIENTS")

R_bern = np.zeros((d, d))
Q_ass = np.zeros((d, d))
Q_col = np.zeros((d, d))
Q_con = np.zeros((d, d))

for i in range(d):
    for j in range(d):
        a = np.sum((Bin[:, i] == 1) & (Bin[:, j] == 1))
        b = np.sum((Bin[:, i] == 1) & (Bin[:, j] == 0))
        c = np.sum((Bin[:, i] == 0) & (Bin[:, j] == 1))
        dd_ = np.sum((Bin[:, i] == 0) & (Bin[:, j] == 0))
        P_A = (a + b) / N
        P_B = (a + c) / N
        P_AB = a / N
        P_A_given_B = P_AB / P_B if P_B > 0 else 0
        R_bern[i, j] = P_B * (P_A_given_B - P_A)
        ad = a * dd_
        bc = b * c
        Q_ass[i, j] = (ad - bc) / (ad + bc) if (ad + bc) != 0 else 0
        sad = np.sqrt(a * dd_)
        sbc = np.sqrt(b * c)
        Q_col[i, j] = (sad - sbc) / (sad + sbc) if (sad + sbc) != 0 else 0
        denom = np.sqrt((a + b) * (c + dd_) * (a + c) * (b + dd_))
        Q_con[i, j] = (ad - bc) / denom if denom != 0 else 0

def print_matrix(name, M):
    print(f"\n{name}")
    print("\t" + "\t".join(short))
    for i in range(d):
        row = "\t".join([f"{M[i,j]:.3f}" for j in range(d)])
        print(f"{short[i]}\t{row}")

print_matrix("R_B = P(B)(P(A|B) - P(A))", R_bern)
print_matrix("K_P = (ad-bc)/sqrt((a+b)(c+d)(a+c)(b+d))", Q_con)
print_matrix("Q_Y = (sqrt(ad)-sqrt(bc))/(sqrt(ad)+sqrt(bc))", Q_col)
print_matrix("Q_A = (ad-bc)/(ad+bc)", Q_ass)

print("\n|Q_A| >= |Q_Y| >= |K_P|:")
ok = True
for i in range(d):
    for j in range(i + 1, d):
        if abs(Q_ass[i, j]) < abs(Q_col[i, j]) - 1e-10 or abs(Q_col[i, j]) < abs(Q_con[i, j]) - 1e-10:
            print(f"  FAIL: {short[i]}--{short[j]}")
            ok = False
if ok:
    print("  OK")

print("\nIII. DIMENSION REDUCTION")
mean_abs_qa = []
for i in range(d):
    vals = [abs(Q_ass[i, j]) for j in range(d) if i != j]
    m = np.mean(vals)
    mean_abs_qa.append(m)
    print(f"  {short[i]}: mean|Q_A|={m:.3f}")

threshold = 0.15
drop = [i for i in range(d) if mean_abs_qa[i] < threshold]
keep = [i for i in range(d) if mean_abs_qa[i] >= threshold]
print(f"drop: {[short[i] for i in drop]}")
print(f"keep: {[short[i] for i in keep]} ({len(keep)}/{d})")

print("\nredundant (|Q_A|>0.7):")
for i in range(d):
    for j in range(i + 1, d):
        if abs(Q_ass[i, j]) > 0.7:
            print(f"  {short[i]}--{short[j]}: Q_A={Q_ass[i,j]:+.3f}")

X_reduced = X[:, keep]
d_reduced = len(keep)
kept_names = [short[i] for i in keep]

print("\nIV. K-MEANS")

class KMeans:
    def __init__(self, n_clusters=2, max_iter=300, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = np.inf
        self.n_iter_ = 0

    def _distance(self, X, centroids):
        XX = np.sum(X**2, axis=1, keepdims=True)
        CC = np.sum(centroids**2, axis=1, keepdims=True)
        dist_sq = XX - 2 * X @ centroids.T + CC.T
        return np.sqrt(np.maximum(dist_sq, 0.0))

    def fit_predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)
        N_, d_ = X.shape
        idx = rng.choice(N_, size=self.n_clusters, replace=False)
        centroids = X[idx].copy()
        for it in range(1, self.max_iter + 1):
            dists = self._distance(X, centroids)
            labels = np.argmin(dists, axis=1)
            new_centroids = np.empty_like(centroids)
            for j in range(self.n_clusters):
                members = X[labels == j]
                if len(members) > 0:
                    new_centroids[j] = members.mean(axis=0)
                else:
                    new_centroids[j] = X[rng.integers(N_)]
            shift = np.linalg.norm(new_centroids - centroids, axis=1).max()
            centroids = new_centroids
            if shift < self.tol:
                self.n_iter_ = it
                break
        else:
            self.n_iter_ = self.max_iter
        dists = self._distance(X, centroids)
        labels = np.argmin(dists, axis=1)
        self.centroids_ = centroids
        self.labels_ = labels
        self.inertia_ = sum(np.sum((X[labels == j] - centroids[j])**2) for j in range(self.n_clusters))
        return labels

    @staticmethod
    def elbow(X, k_range=range(1, 11), **kwargs):
        inertias = []
        for k in k_range:
            km = KMeans(n_clusters=k, **kwargs)
            km.fit_predict(X)
            inertias.append(km.inertia_)
        return inertias

def run_kmeans(X_in, label, kept_names):
    k_range = range(1, 11)
    inertias = KMeans.elbow(X_in, k_range=k_range, random_state=42)
    print(f"\n--- {label} ---")
    print("elbow:")
    for k, iner in zip(k_range, inertias):
        print(f"  k={k}: {iner:.1f}")

    km = KMeans(n_clusters=2, random_state=42)
    pred = km.fit_predict(X_in)

    acc1 = np.mean(pred == y_true)
    acc2 = np.mean((1 - pred) == y_true)
    acc = max(acc1, acc2)
    if acc2 > acc1:
        pred = 1 - pred

    print(f"\nK=2, iter={km.n_iter_}, inertia={km.inertia_:.2f}")
    print(f"clusters: {[int(np.sum(pred == j)) for j in range(2)]}")
    print(f"true: 0={int(np.sum(y_true==0))}, 1={int(np.sum(y_true==1))}")
    print(f"accuracy: {acc:.4f}")

    print(f"\nV. CLUSTER PROFILES ({label})")
    print(f"{'':>5}  " + "  ".join([f"{n:>8}" for n in kept_names]))
    for cl in range(2):
        vals = X_reduced[pred == cl].mean(axis=0)
        print(f"C{cl}:   " + "  ".join([f"{v:8.2f}" for v in vals]))
    print(f"All:  " + "  ".join([f"{v:8.2f}" for v in X_reduced.mean(axis=0)]))

    print(f"\n{'':>5}  " + "  ".join([f"{n:>8}" for n in kept_names]))
    for cl in range(2):
        vals = (X_reduced[pred == cl].mean(axis=0) - X_reduced.mean(axis=0)) / (X_reduced.std(axis=0) + 1e-10)
        print(f"C{cl}:   " + "  ".join([f"{v:+8.2f}" for v in vals]))

    return pred, km, inertias

print("\n=== A. Без нормализации ===")
pred_raw, km_raw, inertias_raw = run_kmeans(X_reduced, "raw", kept_names)

mean_r = X_reduced.mean(axis=0)
std_r = X_reduced.std(axis=0)
X_norm = (X_reduced - mean_r) / (std_r + 1e-10)

print("\n=== B. С z-нормализацией ===")
pred_norm, km_norm, inertias_norm = run_kmeans(X_norm, "z-norm", kept_names)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for row, (pred, inertias, title) in enumerate([
    (pred_raw, inertias_raw, "raw"),
    (pred_norm, inertias_norm, "z-norm"),
]):
    axes[row, 0].plot(list(range(1, 11)), inertias, "o-", color="steelblue", linewidth=2)
    axes[row, 0].set_xlabel("k")
    axes[row, 0].set_ylabel("Inertia")
    axes[row, 0].set_title(f"Elbow ({title})")
    axes[row, 0].grid(alpha=0.3)

    X_use = X_reduced if title == "raw" else X_norm
    X_c = X_use - X_use.mean(axis=0)
    cov = X_c.T @ X_c / len(X_c)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx_sort = np.argsort(eigvals)[::-1]
    pc2 = eigvecs[:, idx_sort[:2]]
    X_pca = X_c @ pc2
    colors_map = ["#4C72B0", "#DD8452"]
    for j in range(2):
        mask = pred == j
        axes[row, 1].scatter(X_pca[mask, 0], X_pca[mask, 1], s=12, alpha=0.5, color=colors_map[j], label=f"C{j}")
    axes[row, 1].set_title(f"K-Means ({title})")
    axes[row, 1].set_xlabel("PC1")
    axes[row, 1].set_ylabel("PC2")
    axes[row, 1].legend(fontsize=8)
    axes[row, 1].grid(alpha=0.3)
    for j in range(2):
        mask = y_true == j
        axes[row, 2].scatter(X_pca[mask, 0], X_pca[mask, 1], s=12, alpha=0.5, color=colors_map[j], label=f"Out {j}")
    axes[row, 2].set_title(f"True ({title})")
    axes[row, 2].set_xlabel("PC1")
    axes[row, 2].set_ylabel("PC2")
    axes[row, 2].legend(fontsize=8)
    axes[row, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("final_kmeans.png", dpi=150)

def save_csv(filename, M):
    with open(filename, "w") as f:
        f.write("," + ",".join(short) + "\n")
        for i in range(d):
            f.write(short[i] + "," + ",".join([f"{M[i,j]:.3f}" for j in range(d)]) + "\n")

save_csv("bernstein.csv", R_bern)
save_csv("contingency.csv", Q_con)
save_csv("colligation_yule.csv", Q_col)
save_csv("association.csv", Q_ass)

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

plot_heatmap("R_B = P(B)(P(A|B) - P(A))", R_bern, "heatmap_bernstein.png", -0.25, 0.25)
plot_heatmap("Contingency (Pearson)", Q_con, "heatmap_contingency.png", -1, 1)
plot_heatmap("Colligation (Yule)", Q_col, "heatmap_colligation.png", -1, 1)
plot_heatmap("Association", Q_ass, "heatmap_association.png", -1, 1)