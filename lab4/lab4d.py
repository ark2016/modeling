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

print("\nSPLIT 50/50 (alternating)")
idx_work = np.arange(0, N, 2)
idx_ctrl = np.arange(1, N, 2)
X_work = X[idx_work]
X_ctrl = X[idx_ctrl]
Bin_work = Bin[idx_work]
Bin_ctrl = Bin[idx_ctrl]
print(f"work: {len(idx_work)}, ctrl: {len(idx_ctrl)}")
print(f"\nmeans comparison:")
print(f"  {'':>5} " + " ".join([f"{s:>8}" for s in short]))
print(f"  all:  " + " ".join([f"{X[:, j].mean():8.2f}" for j in range(d)]))
print(f"  work: " + " ".join([f"{X_work[:, j].mean():8.2f}" for j in range(d)]))
print(f"  ctrl: " + " ".join([f"{X_ctrl[:, j].mean():8.2f}" for j in range(d)]))

print("\nII. COEFFICIENTS (work sample)")
N_w = len(idx_work)
R_bern = np.zeros((d, d))
Q_ass = np.zeros((d, d))
Q_col = np.zeros((d, d))
Q_con = np.zeros((d, d))

for i in range(d):
    for j in range(d):
        a = np.sum((Bin_work[:, i] == 1) & (Bin_work[:, j] == 1))
        b = np.sum((Bin_work[:, i] == 1) & (Bin_work[:, j] == 0))
        c = np.sum((Bin_work[:, i] == 0) & (Bin_work[:, j] == 1))
        dd_ = np.sum((Bin_work[:, i] == 0) & (Bin_work[:, j] == 0))
        P_A = (a + b) / N_w
        P_B = (a + c) / N_w
        P_AB = a / N_w
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

print("\nIII. DIMENSION REDUCTION (work sample)")
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

kept_names = [short[i] for i in keep]
X_work_r = X_work[:, keep]
X_ctrl_r = X_ctrl[:, keep]

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

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.argmin(self._distance(X, self.centroids_), axis=1)

    @staticmethod
    def elbow(X, k_range=range(1, 11), **kwargs):
        inertias = []
        for k in k_range:
            km = KMeans(n_clusters=k, **kwargs)
            km.fit_predict(X)
            inertias.append(km.inertia_)
        return inertias

print("\nIV. K-MEANS (work sample, z-norm)")
mean_w = X_work_r.mean(axis=0)
std_w = X_work_r.std(axis=0)
X_work_norm = (X_work_r - mean_w) / (std_w + 1e-10)

k_range = range(1, 11)
inertias = KMeans.elbow(X_work_norm, k_range=k_range, random_state=42)
print("\nelbow:")
for k, iner in zip(k_range, inertias):
    print(f"  k={k}: {iner:.1f}")

km = KMeans(n_clusters=2, random_state=42)
pred_work = km.fit_predict(X_work_norm)
print(f"\nK=2, iter={km.n_iter_}, inertia={km.inertia_:.2f}")
print(f"work clusters: {[int(np.sum(pred_work == j)) for j in range(2)]}")

print("\nV. CONTROL SAMPLE VALIDATION")
X_ctrl_norm = (X_ctrl_r - mean_w) / (std_w + 1e-10)
pred_ctrl = km.predict(X_ctrl_norm)
print(f"ctrl clusters: {[int(np.sum(pred_ctrl == j)) for j in range(2)]}")

print(f"\nwork profiles:")
print(f"{'':>5}  " + "  ".join([f"{n:>8}" for n in kept_names]))
for cl in range(2):
    vals = X_work_r[pred_work == cl].mean(axis=0)
    print(f"C{cl}:   " + "  ".join([f"{v:8.2f}" for v in vals]))

print(f"\nctrl profiles:")
print(f"{'':>5}  " + "  ".join([f"{n:>8}" for n in kept_names]))
for cl in range(2):
    vals = X_ctrl_r[pred_ctrl == cl].mean(axis=0)
    print(f"C{cl}:   " + "  ".join([f"{v:8.2f}" for v in vals]))

print(f"\nwork deviations (sigma):")
print(f"{'':>5}  " + "  ".join([f"{n:>8}" for n in kept_names]))
for cl in range(2):
    vals = (X_work_r[pred_work == cl].mean(axis=0) - X_work_r.mean(axis=0)) / (X_work_r.std(axis=0) + 1e-10)
    print(f"C{cl}:   " + "  ".join([f"{v:+8.2f}" for v in vals]))

print(f"\nctrl deviations (sigma):")
print(f"{'':>5}  " + "  ".join([f"{n:>8}" for n in kept_names]))
for cl in range(2):
    vals = (X_ctrl_r[pred_ctrl == cl].mean(axis=0) - X_ctrl_r.mean(axis=0)) / (X_ctrl_r.std(axis=0) + 1e-10)
    print(f"C{cl}:   " + "  ".join([f"{v:+8.2f}" for v in vals]))

print("\nVI. LEAVE-ONE-FEATURE-OUT")
d_r = len(keep)
X_all_r = X[:, keep]
mean_all = X_all_r.mean(axis=0)
std_all = X_all_r.std(axis=0)
X_all_norm = (X_all_r - mean_all) / (std_all + 1e-10)

km_full = KMeans(n_clusters=2, random_state=42)
pred_full = km_full.fit_predict(X_all_norm)

print(f"full ({d_r} features): clusters {[int(np.sum(pred_full == j)) for j in range(2)]}")
print(f"\n  {'removed':>10}  {'changed':>8}  {'%':>6}  note")
for f_idx in range(d_r):
    cols = [c for c in range(d_r) if c != f_idx]
    X_reduced = X_all_norm[:, cols]
    km_r = KMeans(n_clusters=2, random_state=42)
    pred_r = km_r.fit_predict(X_reduced)
    acc1 = np.sum(pred_r == pred_full)
    acc2 = np.sum((1 - pred_r) == pred_full)
    changed = N - max(acc1, acc2)
    pct = changed / N * 100
    note = "removable" if changed <= 10 else ""
    print(f"  {kept_names[f_idx]:>10}  {changed:>8}  {pct:>5.1f}%  {note}")

def build_logical_model(Bin_data, labels, feature_names):
    n_feat = Bin_data.shape[1]
    truth = {}
    for row, label in zip(Bin_data, labels):
        key = tuple(row)
        if key not in truth:
            truth[key] = [0, 0]
        truth[key][label] += 1
    model = {}
    for key, counts in truth.items():
        model[key] = 0 if counts[0] >= counts[1] else 1
    minterms = [key for key, val in model.items() if val == 1]
    def minterm_str(mt, names):
        parts = []
        for bit, name in zip(mt, names):
            if bit == 1:
                parts.append(name)
            else:
                parts.append(f"~{name}")
        return "".join(parts)
    sdnf_terms = [minterm_str(mt, feature_names) for mt in minterms]
    simplified = list(minterms)
    changed_flag = True
    while changed_flag:
        changed_flag = False
        new_terms = []
        used = set()
        for i in range(len(simplified)):
            for j in range(i + 1, len(simplified)):
                diff_pos = []
                for k in range(len(simplified[i])):
                    if simplified[i][k] != simplified[j][k]:
                        diff_pos.append(k)
                if len(diff_pos) == 1:
                    merged = list(simplified[i])
                    merged[diff_pos[0]] = -1
                    merged = tuple(merged)
                    if merged not in new_terms:
                        new_terms.append(merged)
                    used.add(i)
                    used.add(j)
                    changed_flag = True
            if i not in used:
                if simplified[i] not in new_terms:
                    new_terms.append(simplified[i])
        for i in range(len(simplified)):
            if i not in used:
                if simplified[i] not in new_terms:
                    new_terms.append(simplified[i])
        simplified = new_terms
    def term_str(mt, names):
        parts = []
        for bit, name in zip(mt, names):
            if bit == -1:
                continue
            elif bit == 1:
                parts.append(name)
            else:
                parts.append(f"~{name}")
        return "".join(parts) if parts else "1"
    dnf_terms = [term_str(mt, feature_names) for mt in simplified]
    return model, sdnf_terms, dnf_terms

def apply_logical_model(model, Bin_data, n_feat):
    preds = []
    for row in Bin_data:
        key = tuple(row)
        if key in model:
            preds.append(model[key])
        else:
            min_dist = float('inf')
            best = 0
            for mk, mv in model.items():
                dist = sum(1 for a, b in zip(key, mk) if a != b)
                if dist < min_dist:
                    min_dist = dist
                    best = mv
            preds.append(best)
    return np.array(preds)

print("\nVII. LOGICAL MODELS")

feat_indices_for_logic = list(range(d))
Bin_work_all = Bin[idx_work]
Bin_ctrl_all = Bin[idx_ctrl]

km_logic = KMeans(n_clusters=2, random_state=42)
mean_w_all = X_work[:, feat_indices_for_logic].mean(axis=0)
std_w_all = X_work[:, feat_indices_for_logic].std(axis=0)
X_wn = (X_work[:, feat_indices_for_logic] - mean_w_all) / (std_w_all + 1e-10)
pred_logic_work = km_logic.fit_predict(X_wn)

X_cn = (X_ctrl[:, feat_indices_for_logic] - mean_w_all) / (std_w_all + 1e-10)
pred_logic_ctrl = km_logic.predict(X_cn)

print(f"\n--- full ({d} features) ---")
model_full, sdnf_full, dnf_full = build_logical_model(Bin_work_all, pred_logic_work, short)
print(f"truth table entries: {len(model_full)}")
print(f"SDNF terms (y=1): {len(sdnf_full)}")
print(f"DNF terms: {len(dnf_full)}")
if len(dnf_full) <= 20:
    print(f"DNF: {' v '.join(dnf_full)}")
pred_ctrl_logic = apply_logical_model(model_full, Bin_ctrl_all, d)
acc1 = np.mean(pred_ctrl_logic == pred_logic_ctrl)
acc2 = np.mean((1 - pred_ctrl_logic) == pred_logic_ctrl)
acc_logic = max(acc1, acc2)
print(f"ctrl match (logic vs kmeans): {acc_logic:.4f} ({int(acc_logic * len(idx_ctrl))}/{len(idx_ctrl)})")

print(f"\n--- leave-one-feature-out ---")
print(f"  {'removed':>10}  {'SDNF':>6}  {'DNF':>6}  {'ctrl_match':>10}")
print(f"  {'none':>10}  {len(sdnf_full):>6}  {len(dnf_full):>6}  {acc_logic:>10.4f}")
for f_idx in range(d):
    cols = [c for c in range(d) if c != f_idx]
    feat_names_r = [short[c] for c in cols]
    Bin_w_r = Bin_work_all[:, cols]
    Bin_c_r = Bin_ctrl_all[:, cols]
    X_wr = X_work[:, cols]
    X_cr = X_ctrl[:, cols]
    m_w = X_wr.mean(axis=0)
    s_w = X_wr.std(axis=0)
    X_wr_n = (X_wr - m_w) / (s_w + 1e-10)
    X_cr_n = (X_cr - m_w) / (s_w + 1e-10)
    km_r = KMeans(n_clusters=2, random_state=42)
    pred_w_r = km_r.fit_predict(X_wr_n)
    pred_c_r = km_r.predict(X_cr_n)
    model_r, sdnf_r, dnf_r = build_logical_model(Bin_w_r, pred_w_r, feat_names_r)
    pred_c_logic_r = apply_logical_model(model_r, Bin_c_r, len(cols))
    a1 = np.mean(pred_c_logic_r == pred_c_r)
    a2 = np.mean((1 - pred_c_logic_r) == pred_c_r)
    acc_r = max(a1, a2)
    print(f"  {short[f_idx]:>10}  {len(sdnf_r):>6}  {len(dnf_r):>6}  {acc_r:>10.4f}")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

X_work_c = X_work_norm - X_work_norm.mean(axis=0)
cov_w = X_work_c.T @ X_work_c / len(X_work_c)
eigvals_w, eigvecs_w = np.linalg.eigh(cov_w)
idx_sort_w = np.argsort(eigvals_w)[::-1]
pc2_w = eigvecs_w[:, idx_sort_w[:2]]

colors_map = ["#4C72B0", "#DD8452"]

axes[0, 0].plot(list(k_range), inertias, "o-", color="steelblue", linewidth=2)
axes[0, 0].set_xlabel("k")
axes[0, 0].set_ylabel("Inertia")
axes[0, 0].set_title("Elbow (work)")
axes[0, 0].grid(alpha=0.3)

X_pca_w = X_work_c @ pc2_w
for j in range(2):
    mask = pred_work == j
    axes[0, 1].scatter(X_pca_w[mask, 0], X_pca_w[mask, 1], s=12, alpha=0.5, color=colors_map[j], label=f"C{j}")
axes[0, 1].set_title("K-Means work (PCA)")
axes[0, 1].set_xlabel("PC1")
axes[0, 1].set_ylabel("PC2")
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(alpha=0.3)

X_ctrl_c = X_ctrl_norm - X_ctrl_norm.mean(axis=0)
X_pca_c = X_ctrl_c @ pc2_w
for j in range(2):
    mask = pred_ctrl == j
    axes[0, 2].scatter(X_pca_c[mask, 0], X_pca_c[mask, 1], s=12, alpha=0.5, color=colors_map[j], label=f"C{j}")
axes[0, 2].set_title("K-Means ctrl (PCA)")
axes[0, 2].set_xlabel("PC1")
axes[0, 2].set_ylabel("PC2")
axes[0, 2].legend(fontsize=8)
axes[0, 2].grid(alpha=0.3)

X_all_c = X_all_norm - X_all_norm.mean(axis=0)
cov_all = X_all_c.T @ X_all_c / len(X_all_c)
eigvals_a, eigvecs_a = np.linalg.eigh(cov_all)
idx_sort_a = np.argsort(eigvals_a)[::-1]
pc2_a = eigvecs_a[:, idx_sort_a[:2]]
X_pca_all = X_all_c @ pc2_a

for j in range(2):
    mask = pred_full == j
    axes[1, 0].scatter(X_pca_all[mask, 0], X_pca_all[mask, 1], s=12, alpha=0.5, color=colors_map[j], label=f"C{j}")
axes[1, 0].set_title("K-Means full (PCA)")
axes[1, 0].set_xlabel("PC1")
axes[1, 0].set_ylabel("PC2")
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(alpha=0.3)

changed_counts = []
for f_idx in range(d_r):
    cols = [c for c in range(d_r) if c != f_idx]
    X_reduced = X_all_norm[:, cols]
    km_r = KMeans(n_clusters=2, random_state=42)
    pred_r = km_r.fit_predict(X_reduced)
    acc1 = np.sum(pred_r == pred_full)
    acc2 = np.sum((1 - pred_r) == pred_full)
    changed_counts.append(N - max(acc1, acc2))

axes[1, 1].barh(kept_names, changed_counts, color="steelblue")
axes[1, 1].set_xlabel("objects changed")
axes[1, 1].set_title("Leave-one-feature-out")
axes[1, 1].grid(alpha=0.3, axis='x')

axes[1, 2].axis('off')

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