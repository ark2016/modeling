import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from lab2 import compute_coefficients, reconstruct

TESTING_DIR = Path(__file__).parent / "testing"
TESTING_DIR.mkdir(exist_ok=True)

# Параметры тестов
# Тест 1: аналитическая функция
N_GRID = 20 # размер сетки n x n
X_RANGE = (-3, 3) # диапазон x
Y_RANGE = (-3, 3) # диапазон y
Z_RANGE = (-1, 1) # ограничение на z
# Тест 2: псевдослучайная сетка
NX, NY = 10, 10 # размер сетки (можно 20x10 и т.д.)
Z_MIN, Z_MAX = -1, 1 # ограничения на z
SEED = 42 # seed генератора
# Тест 3: сгущение
DENSIFY_SIZES = [20, 30] # до каких размеров m x m сгущать

def normalize(v):
    lo, hi = v.min(), v.max()
    return 2 * (v - lo) / (hi - lo) - 1

# Тест 1: z = sin(sqrt(x^2+y^2)), погрешность между узлами интерполяции

def test_analytical(n=20, x_range=(-3, 3), y_range=(-3, 3), z_range=(-1, 1)):
    print("=" * 65)
    print(f"  Тест 1: z = sin(sqrt(x^2+y^2)),  сетка {n}x{n}")
    print(f"  x in {list(x_range)},  y in {list(y_range)},  z in {list(z_range)}")
    print("=" * 65)
    x = np.linspace(*x_range, n)
    y = np.linspace(*y_range, n)
    X, Y = np.meshgrid(x, y, indexing="ij")
    A = np.sin(np.sqrt(X**2 + Y**2))
    # промежуточные точки
    x_mid = (x[:-1] + x[1:]) / 2
    y_mid = (y[:-1] + y[1:]) / 2
    Xm, Ym = np.meshgrid(x_mid, y_mid, indexing="ij")
    Z_true = np.sin(np.sqrt(Xm**2 + Ym**2))
    # нормализация
    to_norm = lambda v: 2 * (v - x_range[0]) / (x_range[1] - x_range[0]) - 1
    x_mid_norm = to_norm(x_mid)
    y_mid_norm = to_norm(y_mid)
    # поиск минимума MAX погрешности между узлами
    best_l, best_max = 2, np.inf
    results = []
    for l in range(2, n + 1):
        D = compute_coefficients(A, l)
        Z_model = reconstruct(D, x_mid_norm, y_mid_norm)
        err = np.abs(Z_model - Z_true)
        min_err = float(np.min(err))
        max_err = float(np.max(err))
        results.append((l, min_err, max_err))
        if max_err < best_max:
            best_max = max_err
            best_l = l
    best_min = [r[1] for r in results if r[0] == best_l][0]
    print(f"  Лучшее l = {best_l}:")
    print(f"    MIN погрешности между узлами = {best_min:.6f}")
    print(f"    MAX погрешности между узлами = {best_max:.6f}")
    # график: погрешность vs l
    ls = [r[0] for r in results]
    min_errs = [r[1] for r in results]
    max_errs = [r[2] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ls, max_errs, "o-", markersize=4, label="MAX погрешности")
    ax.plot(ls, min_errs, "s-", markersize=4, label="MIN погрешности")
    ax.axvline(best_l, color="g", ls="--", alpha=0.7, label=f"лучшее l={best_l}")
    ax.set_xlabel("l")
    ax.set_ylabel("Погрешность между узлами")
    ax.set_title(f"z = sin(sqrt(x^2+y^2)),  сетка {n}x{n}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(TESTING_DIR / "test1_analytical.png", dpi=150)
    plt.close(fig)
    # 3D: оригинал
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, A, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"Оригинал: z = sin(sqrt(x²+y²)),  сетка {n}x{n}")
    fig.tight_layout()
    fig.savefig(TESTING_DIR / "test1_original.png", dpi=150)
    plt.close(fig)
    # 3D: восстановленная модель
    D_best = compute_coefficients(A, best_l)
    xn = to_norm(x)
    yn = to_norm(y)
    Z_model = reconstruct(D_best, xn, yn)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z_model, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"Восстановленная модель (l={best_l}),  MAX={best_max:.4f}")
    fig.tight_layout()
    fig.savefig(TESTING_DIR / "test1_reconstructed.png", dpi=150)
    plt.close(fig)
    return A, best_l

# Тест 2: псевдослучайная сетка
def test_random(nx=10, ny=10, x_range=(-3, 3), y_range=(-3, 3), z_min=-1, z_max=1, seed=42):
    print(f"\n{'='*65}")
    print(f"  Тест 2: псевдослучайная сетка {nx}x{ny},  z in [{z_min}, {z_max}]")
    print("=" * 65)
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(*x_range, nx))
    y = np.sort(rng.uniform(*y_range, ny))
    Z = rng.uniform(z_min, z_max, (nx, ny))
    print(f"  x = {np.array2string(x, precision=3, separator=', ')}")
    print(f"  y = {np.array2string(y, precision=3, separator=', ')}")
    print(f"\n  Исходная матрица Z ({nx}x{ny}):")
    _print_matrix(Z, x, y)
    x_norm = normalize(x)
    y_norm = normalize(y)
    l_max = min(nx, ny)
    x_mid = (x_norm[:-1] + x_norm[1:]) / 2
    y_mid = (y_norm[:-1] + y_norm[1:]) / 2
    best_l = l_max
    results = []
    for l in range(2, l_max + 1):
        D = compute_coefficients(Z, l, x_grid=x_norm, y_grid=y_norm)
        Z_recon = reconstruct(D, x_norm, y_norm)
        err = np.abs(Z_recon - Z)
        Z_mid = reconstruct(D, x_mid, y_mid)
        overshoot = max(0.0, float(Z_mid.max()) - z_max, z_min - float(Z_mid.min()))
        results.append((l, float(np.min(err)), float(np.max(err)), overshoot))
    for l, mn, mx, ov in reversed(results):
        if ov < 0.5 * (z_max - z_min):
            best_l = l
            break
    D = compute_coefficients(Z, best_l, x_grid=x_norm, y_grid=y_norm)
    Z_recon = reconstruct(D, x_norm, y_norm)
    err = np.abs(Z_recon - Z)
    print(f"\n  Выбрано l = {best_l}:")
    print(f"    MIN погрешности в узлах = {float(np.min(err)):.2e}")
    print(f"    MAX погрешности в узлах = {float(np.max(err)):.2e}")
    ls = [r[0] for r in results]
    min_errs = [r[1] for r in results]
    max_errs = [r[2] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ls, max_errs, "o-", markersize=5, label="MAX погрешности")
    ax.plot(ls, min_errs, "s-", markersize=5, label="MIN погрешности")
    ax.axvline(best_l, color="g", ls="--", alpha=0.7, label=f"выбрано l={best_l}")
    ax.set_xlabel("l")
    ax.set_ylabel("Погрешность в узлах")
    ax.set_title(f"Псевдослучайная сетка {nx}x{ny}: погрешность")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(TESTING_DIR / "test2_error.png", dpi=150)
    plt.close(fig)
    # 3D: исходные точки
    Xo, Yo = np.meshgrid(x, y, indexing="ij")
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Xo, Yo, Z, cmap="terrain", linewidth=0.5, antialiased=True)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"Исходная псевдослучайная сетка {nx}x{ny}")
    fig.tight_layout()
    fig.savefig(TESTING_DIR / "test2_original.png", dpi=150)
    plt.close(fig)
    # 3D: восстановленная модель на более густой сетке
    m_vis = 40
    xg = np.linspace(-1, 1, m_vis)
    yg = np.linspace(-1, 1, m_vis)
    Z_vis = reconstruct(D, xg, yg)
    x_vis = np.linspace(x.min(), x.max(), m_vis)
    y_vis = np.linspace(y.min(), y.max(), m_vis)
    Xv, Yv = np.meshgrid(x_vis, y_vis, indexing="ij")
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Xv, Yv, Z_vis, cmap="terrain", linewidth=0, antialiased=True)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"Восстановленная модель (l={best_l}),  MAX = {float(np.max(err)):.2e}")
    fig.tight_layout()
    fig.savefig(TESTING_DIR / "test2_reconstructed.png", dpi=150)
    plt.close(fig)
    return D, x, y, Z

# Тест 3: сгущение сетки
def test_densify(D, x_orig, y_orig, Z_orig, m=20):
    nx, ny = len(x_orig), len(y_orig)
    print(f"\n{'='*65}")
    print(f"  Тест 3: сгущение {nx}x{ny} -> {m}x{m}")
    print("=" * 65)
    xg = np.linspace(-1, 1, m)
    yg = np.linspace(-1, 1, m)
    Z_dense = reconstruct(D, xg, yg)
    x_phys = np.linspace(x_orig.min(), x_orig.max(), m)
    y_phys = np.linspace(y_orig.min(), y_orig.max(), m)
    print(f"\n  Сгущённая матрица высот ({m}x{m}):")
    _print_matrix(Z_dense, x_phys, y_phys)
    # 3D: исходная сетка
    Xo, Yo = np.meshgrid(x_orig, y_orig, indexing="ij")
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Xo, Yo, Z_orig, cmap="terrain", linewidth=0.5, antialiased=True)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"Исходная сетка {nx}x{ny}")
    fig.tight_layout()
    fig.savefig(TESTING_DIR / f"test3_original_{nx}x{ny}.png", dpi=150)
    plt.close(fig)
    # 3D: сгущённая сетка
    X, Y = np.meshgrid(x_phys, y_phys, indexing="ij")
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z_dense, cmap="terrain", linewidth=0, antialiased=True)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"Сгущение {nx}x{ny} -> {m}x{m}")
    fig.tight_layout()
    fig.savefig(TESTING_DIR / f"test3_densify_{m}.png", dpi=150)
    plt.close(fig)
    return Z_dense

def _print_matrix(Z, x_coords, y_coords, max_cols=10, max_rows=12):
    nr, nc = Z.shape
    show_r = min(nr, max_rows)
    show_c = min(nc, max_cols)
    trunc_r = nr > max_rows
    trunc_c = nc > max_cols
    header = "       "
    for j in range(show_c):
        header += f"{y_coords[j]:8.3f}"
    if trunc_c:
        header += "     ..."
    print(header)
    for i in range(show_r):
        row = f"  {x_coords[i]:6.3f}"
        for j in range(show_c):
            row += f"{Z[i, j]:8.3f}"
        if trunc_c:
            row += "     ..."
        print(row)
    if trunc_r:
        print(f"  {'...':>6}")

def main():
    test_analytical(n=N_GRID, x_range=X_RANGE, y_range=Y_RANGE, z_range=Z_RANGE)
    D, x, y, Z = test_random(nx=NX, ny=NY, z_min=Z_MIN, z_max=Z_MAX, seed=SEED)
    for m in DENSIFY_SIZES:
        test_densify(D, x, y, Z, m=m)

if __name__ == "__main__":
    main()