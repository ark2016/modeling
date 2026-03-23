import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, init="kmeans++", metric="euclidean", random_state=None, n_init=10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.metric = metric
        self.random_state = random_state
        self.n_init = n_init
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = np.inf
        self.n_iter_ = 0

    def _distance(self, X, centroids):
        if self.metric == "euclidean":
            XX = np.sum(X**2, axis=1, keepdims=True)
            CC = np.sum(centroids**2, axis=1, keepdims=True)
            dist_sq = XX - 2 * X @ centroids.T + CC.T
            return np.sqrt(np.maximum(dist_sq, 0.0))
        elif self.metric == "manhattan":
            return np.sum(np.abs(X[:, np.newaxis, :] - centroids[np.newaxis, :, :]), axis=2)
        elif self.metric == "cosine":
            X_n = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
            C_n = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)
            return 1.0 - X_n @ C_n.T

    def _init_centroids(self, X, rng):
        N, d = X.shape
        if self.init == "random":
            return X[rng.choice(N, size=self.n_clusters, replace=False)].copy()
        centroids = np.empty((self.n_clusters, d), dtype=X.dtype)
        centroids[0] = X[rng.integers(N)]
        for i in range(1, self.n_clusters):
            dist = self._distance(X, centroids[:i])
            min_dist = dist.min(axis=1)
            probs = min_dist**2
            probs /= probs.sum()
            centroids[i] = X[rng.choice(N, p=probs)]
        return centroids

    def _fit_single(self, X, rng):
        centroids = self._init_centroids(X, rng)
        for it in range(1, self.max_iter + 1):
            dists = self._distance(X, centroids)
            labels = np.argmin(dists, axis=1)
            new_centroids = np.empty_like(centroids)
            for j in range(self.n_clusters):
                members = X[labels == j]
                new_centroids[j] = members.mean(axis=0) if len(members) > 0 else X[rng.integers(X.shape[0])]
            shift = np.linalg.norm(new_centroids - centroids, axis=1).max()
            centroids = new_centroids
            if shift < self.tol:
                break
        dists = self._distance(X, centroids)
        labels = np.argmin(dists, axis=1)
        inertia = sum(np.sum((X[labels == j] - centroids[j])**2) for j in range(self.n_clusters))
        return centroids, labels, inertia, it

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        rng = np.random.default_rng(self.random_state)
        best = (None, None, np.inf, 0)
        for _ in range(self.n_init):
            res = self._fit_single(X, rng)
            if res[2] < best[2]:
                best = res
        self.centroids_, self.labels_, self.inertia_, self.n_iter_ = best
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.argmin(self._distance(X, self.centroids_), axis=1)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    @staticmethod
    def elbow(X, k_range=range(1, 11), **kwargs):
        inertias = []
        for k in k_range:
            km = KMeans(n_clusters=k, **kwargs)
            km.fit(X)
            inertias.append(km.inertia_)
        return inertias


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(42)
    centers = np.array([[2, 2], [-2, -2], [2, -2], [-2, 2]])
    X = np.vstack([rng.normal(loc=c, scale=0.6, size=(150, 2)) for c in centers])
    rng.shuffle(X)
    km = KMeans(n_clusters=4, init="kmeans++", random_state=0)
    labels = km.fit_predict(X)
    print(f"Итераций: {km.n_iter_}, inertia: {km.inertia_:.2f}")
    print(f"Центроиды:\n{km.centroids_}")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for j in range(km.n_clusters):
        mask = labels == j
        axes[0].scatter(X[mask, 0], X[mask, 1], s=15, alpha=0.6, label=f"Кластер {j}")
    axes[0].scatter(km.centroids_[:, 0], km.centroids_[:, 1], s=200, c="black", marker="X", edgecolors="white", linewidths=1.5, zorder=10)
    axes[0].set_title("K-Means (2D)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    k_range = range(1, 9)
    inertias = KMeans.elbow(X, k_range=k_range, random_state=0)
    axes[1].plot(list(k_range), inertias, "o-", color="steelblue", linewidth=2)
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Inertia")
    axes[1].set_title("Elbow Method")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("kmeans_demo.png", dpi=150)
    print("Сохранено: kmeans_demo.png")

    # --- Diabetes dataset ---
    import csv
    with open(r"..\lab3\diabetes.csv", "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        data = np.array([[float(x) for x in row] for row in reader])
    X_diab = data[:, :-1]  # все признаки кроме Outcome
    y_true = data[:, -1].astype(int)
    feature_names = header[:-1]
    mean = X_diab.mean(axis=0)
    std = X_diab.std(axis=0)
    X_norm = (X_diab - mean) / (std + 1e-10)
    k_range_d = range(1, 11)
    inertias_d = KMeans.elbow(X_norm, k_range=k_range_d, random_state=0)
    km_diab = KMeans(n_clusters=2, init="kmeans++", random_state=0)
    pred = km_diab.fit_predict(X_norm)
    acc1 = np.mean(pred == y_true)
    acc2 = np.mean((1 - pred) == y_true)
    acc = max(acc1, acc2)
    if acc2 > acc1:
        pred = 1 - pred
    print(f"\n=== Diabetes dataset ===")
    print(f"Размер: {X_diab.shape[0]} x {X_diab.shape[1]} признаков")
    print(f"Итераций: {km_diab.n_iter_}, inertia: {km_diab.inertia_:.2f}")
    print(f"Совпадение с Outcome: {acc:.2%}")
    print(f"Размеры кластеров: {[np.sum(pred == j) for j in range(2)]}")
    print(f"Реальное распределение: 0={np.sum(y_true==0)}, 1={np.sum(y_true==1)}")
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    axes2[0].plot(list(k_range_d), inertias_d, "o-", color="steelblue", linewidth=2)
    axes2[0].set_xlabel("k")
    axes2[0].set_ylabel("Inertia")
    axes2[0].set_title("Elbow (Diabetes)")
    axes2[0].grid(alpha=0.3)
    X_c = X_norm - X_norm.mean(axis=0)
    cov = X_c.T @ X_c / len(X_c)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    pc2 = eigvecs[:, idx[:2]]
    X_pca = X_c @ pc2
    colors_map = ["#4C72B0", "#DD8452"]
    for j in range(2):
        mask = pred == j
        axes2[1].scatter(X_pca[mask, 0], X_pca[mask, 1], s=12, alpha=0.5, color=colors_map[j], label=f"Кластер {j}")
    axes2[1].scatter([], [], s=0)
    axes2[1].set_title("K-Means кластеры (PCA)")
    axes2[1].set_xlabel("PC1")
    axes2[1].set_ylabel("PC2")
    axes2[1].legend(fontsize=8)
    axes2[1].grid(alpha=0.3)
    for j in range(2):
        mask = y_true == j
        axes2[2].scatter(X_pca[mask, 0], X_pca[mask, 1], s=12, alpha=0.5, color=colors_map[j], label=f"Outcome {j}")
    axes2[2].set_title("Реальные метки (PCA)")
    axes2[2].set_xlabel("PC1")
    axes2[2].set_ylabel("PC2")
    axes2[2].legend(fontsize=8)
    axes2[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("kmeans_diabetes.png", dpi=150)
    print("Сохранено: kmeans_diabetes.png")