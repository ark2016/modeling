import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path

IMAGES_DIR = Path(__file__).parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)


def chebyshev_nodes(k):
    """t_m = cos(π(2m−1)/(2k)), m=1..k  — нули T_k(x) (ф. 7)."""
    m = np.arange(1, k + 1)
    return np.cos(np.pi * (2 * m - 1) / (2 * k))


def chebyshev_matrix(nodes, l):
    """T[i,j] = T_i(nodes[j]) = cos(i·arccos(nodes[j])) (ф. 5)."""
    theta = np.arccos(np.clip(nodes, -1.0, 1.0))
    i = np.arange(l)[:, None]
    return np.cos(i * theta[None, :])


def fejer_weights(l):
    """w_i = (l−i)/l,  i = 0..l−1  (ф. 10)."""
    return (l - np.arange(l)) / l


def _build_interp_matrix(x_from, x_to, kind="cubic"):
    """Матрица линейной/кубической интерполяции L: L @ f(x_from) = f(x_to)."""
    n = len(x_from)
    if n < 4 and kind == "cubic":
        kind = "linear"
    L = np.zeros((len(x_to), n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        L[:, j] = interp1d(x_from, e, kind=kind, fill_value="extrapolate")(x_to)
    return L


def _is_uniform(grid, tol=1e-8):
    """Проверка: равномерная ли сетка."""
    diffs = np.diff(grid)
    return np.max(np.abs(diffs - diffs[0])) < tol * (grid[-1] - grid[0])


def compute_coefficients(A, l, k=None, fejer=False, interp_kind="cubic",
                         x_grid=None, y_grid=None):
    """Вычисление коэффициентов D (ф. 13–14).
    Для равномерных сеток: квадратура Чебышёва (ф. 13–14).
    Для нерегулярных сеток: МНК min ||Z - Tx^T D Ty||_F^2.
    A      : (m × n) матрица высот
    l      : число коэффициентов разложения
    k      : узлов квадратуры, по умолчанию 8·max(m,n)
    fejer  : суммирование Фейера (ф. 10)
    x_grid : координаты узлов по x в [-1,1] (по умолчанию linspace)
    y_grid : координаты узлов по y в [-1,1] (по умолчанию linspace)
    """
    m, n = A.shape
    if x_grid is None:
        x_grid = np.linspace(-1, 1, m)
    if y_grid is None:
        y_grid = np.linspace(-1, 1, n)

    # нерегулярная сетка → прямой МНК через Кронекерово произведение
    if not _is_uniform(x_grid) or not _is_uniform(y_grid):
        Tx = chebyshev_matrix(x_grid, l)  # (l, m)
        Ty = chebyshev_matrix(y_grid, l)  # (l, n)
        K = np.kron(Ty.T, Tx.T)           # (m*n, l*l)
        d, *_ = np.linalg.lstsq(K, A.ravel(), rcond=None)
        D = d.reshape(l, l)
        if fejer:
            fw = fejer_weights(l)
            D = D * fw[:, None] * fw[None, :]
        return D

    # равномерная сетка → квадратура Чебышёва (ф. 13–14)
    if k is None:
        k = 8 * max(m, n)
    t = chebyshev_nodes(k)
    Lyt = _build_interp_matrix(y_grid, t, kind=interp_kind)
    Lxt = _build_interp_matrix(x_grid, t, kind=interp_kind)
    Tt = chebyshev_matrix(t, l)
    fw = fejer_weights(l) if fejer else np.ones(l)
    # Этап 1: разложение по y → C (l × m)
    C = (2.0 / k) * Tt @ (Lyt @ A.T)
    C[0, :] /= 2.0  # коррекция нормы T_0: ⟨T_0,T_0⟩ = k, а не k/2
    C *= fw[:, None]
    # Этап 2: разложение по x → D (l × l)
    D = (2.0 / k) * Tt @ (Lxt @ C.T)
    D[0, :] /= 2.0
    D *= fw[:, None]
    return D


def reconstruct(D, x, y):
    """Z = Txᵀ · D · Ty  (ф. 15)."""
    l = D.shape[0]
    return chebyshev_matrix(x, l).T @ D @ chebyshev_matrix(y, l)

def _deriv_coeffs_1d(c, interval_length):
    """p_{l-1}=0, p_{l-2}=2(l-1)c_{l-1}, p_j=p_{j+2}+2(j+1)c_{j+1} (ф. 11).
    Масштабирование p_j *= 2/L (ф. 12)."""
    l = len(c)
    p = np.zeros(l)
    if l < 2:
        return p
    p[l - 2] = 2.0 * (l - 1) * c[l - 1]
    for j in range(l - 3, -1, -1):
        p[j] = p[j + 2] + 2.0 * (j + 1) * c[j + 1]
    p[0] /= 2.0
    p *= 2.0 / interval_length
    return p


def deriv_operator(l, interval_length):
    """Матричная форма оператора E дифференцирования (ф. 16)."""
    E = np.zeros((l, l))
    for col in range(l):
        e = np.zeros(l)
        e[col] = 1.0
        E[:, col] = _deriv_coeffs_1d(e, interval_length)
    return E


def compute_derivatives(D, Lx, Ly):
    """P=ED, Q=DEᵀ, R=E²D, T=D(Eᵀ)², S=EDEᵀ  (ф. 16)."""
    l = D.shape[0]
    Ex, Ey = deriv_operator(l, Lx), deriv_operator(l, Ly)
    return {
        "P": Ex @ D,
        "Q": D @ Ey.T,
        "R": Ex @ Ex @ D,
        "T": D @ Ey.T @ Ey.T,
        "S": Ex @ D @ Ey.T,
    }


def horizontal_curvature(p, q, r, t, s):
    """kh = −(q²r − 2pqs + p²t) / (p²+q²)^{3/2}  (ф. 4)."""
    num = -(q**2 * r - 2*p*q*s + p**2 * t)
    den = (p**2 + q**2) * np.sqrt(p**2 + q**2)
    return np.where(den < 1e-15, 0.0, num / den)


def metrics(orig, recon):
    d = orig - recon
    return {"RMSE": float(np.sqrt(np.mean(d**2))),
            "MAX":  float(np.max(np.abs(d)))}


def generate_terrain(m=100, n=100, seed=42, noise_std=15.0):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1, 1, m)
    y = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, y, indexing="ij")
    z  = 2000 * np.exp(-((X-0.2)**2 + (Y+0.1)**2) / 0.08)
    z += 1500 * np.exp(-((X+0.5)**2 + (Y-0.4)**2) / 0.05)
    z += 300 * np.sin(2*np.pi*X) * np.cos(3*np.pi*Y)
    z += 500 + 200*X - 100*Y
    if noise_std > 0:
        z += rng.normal(0, noise_std, z.shape)
    return z


def load_real_dem(path):
    p = Path(path)
    if not p.exists():
        return None
    if p.suffix == ".npy":
        return np.load(p)
    if p.suffix == ".csv":
        return np.loadtxt(p, delimiter=",")
    return None


def plot_surface(Z, title="", filename=None):
    m, n = Z.shape
    X, Y = np.meshgrid(np.linspace(0, 1, m), np.linspace(0, 1, n), indexing="ij")
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap=cm.terrain, linewidth=0, antialiased=True, rcount=200, ccount=200)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (м)")
    ax.set_title(title)
    if filename:
        fig.savefig(IMAGES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(orig, recon, l_val, met, label="", filename=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    im0 = axes[0].imshow(orig.T, origin="lower", cmap="terrain", aspect="auto")
    axes[0].set_title("Исходный рельеф")
    plt.colorbar(im0, ax=axes[0], shrink=0.7)
    im1 = axes[1].imshow(recon.T, origin="lower", cmap="terrain", aspect="auto")
    axes[1].set_title(f"Реконструкция (l={l_val})")
    plt.colorbar(im1, ax=axes[1], shrink=0.7)
    diff = orig - recon
    vmax = max(abs(diff.min()), abs(diff.max()), 1e-10)
    im2 = axes[2].imshow(diff.T, origin="lower", cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    axes[2].set_title(f"Ошибка (RMSE={met['RMSE']:.4f}, MAX={met['MAX']:.4f})")
    plt.colorbar(im2, ax=axes[2], shrink=0.7)
    fig.suptitle(f"{label}l = {l_val}", fontsize=14)
    fig.tight_layout()
    if filename:
        fig.savefig(IMAGES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_curvature(kh, l_val, filename=None):
    # Логарифмическое преобразование (ф. 17, n=8)
    kh_log = np.sign(kh) * np.log(1 + 1e8 * np.abs(kh))
    fig, ax = plt.subplots(figsize=(8, 6))
    vmax = np.percentile(np.abs(kh_log), 99)
    im = ax.imshow(kh_log.T, origin="lower", cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_title(f"Горизонтальная кривизна kh (l={l_val})")
    plt.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    if filename:
        fig.savefig(IMAGES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fejer_comparison(A, l_values, filename=None):
    m, n = A.shape
    xg, yg = np.linspace(-1, 1, m), np.linspace(-1, 1, n)
    fig, ax = plt.subplots(figsize=(10, 6))
    for fejer, label, style, color in [
        (True,  "С суммированием Фейера",  "o-",  "tab:blue"),
        (False, "Без суммирования Фейера", "s--", "tab:red"),
    ]:
        rmses = []
        for lv in l_values:
            D = compute_coefficients(A, lv, fejer=fejer)
            Z = reconstruct(D, xg, yg)
            rmses.append(metrics(A, Z)["RMSE"])
        ax.semilogy(l_values, rmses, style, color=color, label=label, markersize=3)
    ax.set_xlabel("Число коэффициентов l")
    ax.set_ylabel("RMSE")
    ax.set_title("Влияние суммирования Фейера на точность")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if filename:
        fig.savefig(IMAGES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)

def run_sanity_checks():
    print("\n--- Предельные случаи ---")
    g = np.linspace(-1, 1, 30)
    xx, yy = np.meshgrid(g, g, indexing="ij")
    cases = [
        ("const",   np.full((30, 30), 42.0),             5),
        ("линейн",  3*xx + 7*yy + 10,                   10),
        ("квадр",   xx**2 + yy**2,                      10),
        ("sin·cos", np.sin(np.pi*xx)*np.cos(np.pi*yy),  30),
    ]
    header = "  {:20s}" + "  {:>10s}" * len(cases)
    print(header.format("Режим", *[c[0] for c in cases]))
    print("  " + "-" * (20 + 12 * len(cases)))
    for label, fejer in [("С Фейером", True), ("Без Фейера", False)]:
        parts = []
        for _, A_test, l_test in cases:
            D = compute_coefficients(A_test, l_test, fejer=fejer)
            Z = reconstruct(D, g, g)
            parts.append(f"{metrics(A_test, Z)['RMSE']:.2e}")
        print(f"  {label:20s}  {'  '.join(parts)}")


def main():
    print("=" * 65)
    print("  Лаб. 2 — Статическая модель рельефа (полиномы Чебышёва)")
    print("=" * 65)
    run_sanity_checks()
    dem_dir = Path(__file__).parent
    A = load_real_dem(str(dem_dir / "dem.npy"))
    if A is None:
        A = load_real_dem(str(dem_dir / "dem.csv"))
    if A is None:
        print("\nРеальные данные не найдены → синтетический рельеф 100×100")
        A = generate_terrain(100, 100, noise_std=15.0)
    else:
        print(f"\nДанные загружены: {A.shape}")
    m, n = A.shape
    l_max = min(m, n)
    print(f"Сетка: {m}×{n},  z ∈ [{A.min():.1f}, {A.max():.1f}],  l = min(M,N) = {l_max}")
    plot_surface(A, "Исходный рельеф", "original_3d.png")
    xg = np.linspace(-1, 1, m)
    yg = np.linspace(-1, 1, n)
    l_demo = sorted(set([max(l_max // 8, 4), l_max // 4, l_max // 2, l_max]))
    for fejer, tag, desc in [
        (False, "nofejer", "Без суммирования Фейера"),
        (True,  "fejer",   "С суммированием Фейера"),
    ]:
        print(f"\n{'='*65}")
        print(f"  {desc}")
        print(f"{'='*65}")
        print(f"  {'l':>5}  {'RMSE':>12}  {'MAX':>12}")
        print(f"  {'-'*35}")
        for lv in l_demo:
            D = compute_coefficients(A, lv, fejer=fejer)
            Z = reconstruct(D, xg, yg)
            met = metrics(A, Z)
            print(f"  {lv:5d}  {met['RMSE']:12.4f}  {met['MAX']:12.4f}")
            plot_comparison(A, Z, lv, met, f"{desc}, ", f"{tag}_l{lv}.png")
            if fejer:
                plot_surface(Z, f"{desc}, l={lv}", f"surface_{tag}_l{lv}.png")
    l_range = list(range(2, l_max + 1, max(1, l_max // 40)))
    if l_max not in l_range:
        l_range.append(l_max)
    print("\nГрафик: Фейер vs без Фейера...")
    plot_fejer_comparison(A, l_range, "fejer_comparison.png")
    print("Горизонтальная кривизна...")
    for lv in [l_max, l_max // 2]:
        D = compute_coefficients(A, lv)
        dv = compute_derivatives(D, 2.0, 2.0)
        pv = reconstruct(dv["P"], xg, yg)
        qv = reconstruct(dv["Q"], xg, yg)
        rv = reconstruct(dv["R"], xg, yg)
        tv = reconstruct(dv["T"], xg, yg)
        sv = reconstruct(dv["S"], xg, yg)
        kh = horizontal_curvature(pv, qv, rv, tv, sv)
        plot_curvature(kh, lv, f"curvature_l{lv}.png")
    print(f"\nГрафики → {IMAGES_DIR.resolve()}")


if __name__ == "__main__":
    main()