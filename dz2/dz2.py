import random
import numpy as np
import matplotlib.pyplot as plt
from math import comb

def p_win_analytical(N: int, K: int, q: float) -> float:
    """
    Вероятность выигрыша по формуле полной вероятности.
    P(A) = (1/N)(1 - q) + ((N-1)/N) · q/(N - K - 1)
    """
    p_h1 = 1.0 / N
    p_h2 = (N - 1.0) / N
    p_a_h1 = 1.0 - q
    p_a_h2 = q / (N - K - 1)
    return p_h1 * p_a_h1 + p_h2 * p_a_h2


def simulate_once(N: int, K: int, q: float) -> bool:
    """Один раунд обобщённой игры Монти Холла."""
    prize = random.randint(0, N - 1)
    choice = random.randint(0, N - 1)
    available_to_open = [d for d in range(N) if d != choice and d != prize]
    opened = set(random.sample(available_to_open, K))
    if random.random() < q:
        remaining = [d for d in range(N) if d != choice and d not in opened]
        final = random.choice(remaining)
    else:
        final = choice
    return final == prize


def simulate_series(N: int, K: int, q: float, n_trials: int = 100_000) -> float:
    """Оценка вероятности выигрыша методом Монте-Карло."""
    wins = sum(1 for _ in range(n_trials) if simulate_once(N, K, q))
    return wins / n_trials


def check_classic_monty_hall():
    """Проверка адекватности на классической задаче (N=3, K=1)."""
    print("=" * 70)
    print("ПРОВЕРКА АДЕКВАТНОСТИ: парадокс Монти Холла (N=3, K=1)")
    print("=" * 70)
    N, K, n_trials = 3, 1, 200_000
    for q, label in [(0.0, "не менять (q=0)"), (1.0, "менять (q=1)"), (0.5, "случайно (q=0.5)")]:
        p_a = p_win_analytical(N, K, q)
        p_s = simulate_series(N, K, q, n_trials)
        print(f"  «{label}»: аналит. {p_a:.6f}, имит. {p_s:.6f}, "
              f"|Δ|={abs(p_a - p_s):.6f}")
    print(f"\n  Текст Лукьяненко: 67/100 ≈ M[K]=100·2/3=66.67; 6/9 = M[K]=9·2/3=6")
    print()


def experiment_vary_K_and_N():
    """Сравнение модели и эксперимента: K=1..N-2, N=3..10, q=1."""
    print("=" * 70)
    print("ЭКСПЕРИМЕНТ: P(выигрыш | q=1) для N=3..10, K=1..N-2")
    print("=" * 70)
    q, n_trials = 1.0, 100_000
    results = {}
    print(f"{'N':>3} {'K':>3} {'P(аналит)':>12} {'P(имитац)':>12} {'|Δ|':>10}")
    print("-" * 45)
    for N in range(3, 11):
        for K in range(1, N - 1):
            p_a = p_win_analytical(N, K, q)
            p_s = simulate_series(N, K, q, n_trials)
            results[(N, K)] = (p_a, p_s)
            print(f"{N:>3} {K:>3} {p_a:>12.6f} {p_s:>12.6f} {abs(p_a-p_s):>10.6f}")
    print()
    return results


def experiment_vary_q():
    """Влияние вероятности смены выбора q на P(выигрыш)."""
    print("=" * 70)
    print("ЭКСПЕРИМЕНТ: влияние вероятности смены выбора q")
    print("=" * 70)
    q_values = np.linspace(0, 1, 21)
    n_trials = 50_000
    configs = [(3, 1), (5, 2), (5, 3), (10, 8)]
    for N, K in configs:
        print(f"\n  N={N}, K={K}:")
        print(f"  {'q':>6} {'P(аналит)':>12} {'P(имитац)':>12}")
        print(f"  " + "-" * 34)
        for q in q_values:
            p_a = p_win_analytical(N, K, q)
            p_s = simulate_series(N, K, q, n_trials)
            print(f"  {q:>6.2f} {p_a:>12.6f} {p_s:>12.6f}")
    print()


def plot_results():
    """Построение графиков."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for N in [3, 5, 7, 10]:
        Ks = list(range(1, N - 1))
        ps = [p_win_analytical(N, K, q=1.0) for K in Ks]
        ax.plot(Ks, ps, 'o-', label=f'N={N}', markersize=5)
    ax.set_xlabel('K (число открытых дверей)')
    ax.set_ylabel('P(выигрыш)')
    ax.set_title('Стратегия «всегда менять» (q=1)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax = axes[1]
    q_arr = np.linspace(0, 1, 100)
    for N, K in [(3, 1), (5, 2), (10, 5), (10, 8)]:
        ps = [p_win_analytical(N, K, q) for q in q_arr]
        ax.plot(q_arr, ps, label=f'N={N}, K={K}')
    ax.set_xlabel('q (вероятность смены выбора)')
    ax.set_ylabel('P(выигрыш)')
    ax.set_title('Влияние вероятности смены выбора')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('monty_hall_results.png', dpi=150, bbox_inches='tight')
    print("  График сохранён: monty_hall_results.png")
    plt.close()
    fig, ax = plt.subplots(figsize=(8, 6))
    N_range = range(3, 11)
    max_K = max(N - 2 for N in N_range)
    data = np.full((len(list(N_range)), max_K), np.nan)
    for i, N in enumerate(N_range):
        for K in range(1, N - 1):
            data[i, K - 1] = p_win_analytical(N, K, q=1.0)
    im = ax.imshow(data, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1, origin='lower')
    ax.set_yticks(range(len(list(N_range))))
    ax.set_yticklabels([str(n) for n in N_range])
    ax.set_xticks(range(max_K))
    ax.set_xticklabels([str(k+1) for k in range(max_K)])
    ax.set_ylabel('N (число дверей)')
    ax.set_xlabel('K (число открытых дверей)')
    ax.set_title('P(выигрыш) при q=1')
    plt.colorbar(im, ax=ax, label='P(выигрыш)')
    for i, N in enumerate(N_range):
        for K in range(1, N - 1):
            val = data[i, K - 1]
            if not np.isnan(val):
                ax.text(K - 1, i, f'{val:.3f}', ha='center', va='center', fontsize=8, color='black' if val > 0.5 else 'white')
    plt.tight_layout()
    plt.savefig('monty_hall_heatmap.png', dpi=150, bbox_inches='tight')
    print("  Тепловая карта сохранена: monty_hall_heatmap.png")
    plt.close()


def print_formula():
    """Вывод аналитической формулы."""
    print("=" * 70)
    print("АНАЛИТИЧЕСКАЯ МОДЕЛЬ")
    print("=" * 70 + "\n")
    print("  H1: угадал (P=1/N),  H2: не угадал (P=(N-1)/N)")
    print("  P(A|H1) = 1 - q")
    print("  P(A|H2) = q / (N - K - 1)")
    print("  P(A) = (1/N)(1-q) + ((N-1)/N) · q/(N-K-1)\n")
    print("  q=0: P(A) = 1/N")
    print("  q=1: P(A) = (N-1) / (N·(N-K-1))\n")
    p = p_win_analytical(3, 1, 1.0)
    print(f"  Проверка (N=3, K=1, q=1): P = {p:.6f} = 2/3 ✓\n")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    print_formula()
    check_classic_monty_hall()
    experiment_vary_K_and_N()
    experiment_vary_q()
    plot_results()
    print("Все эксперименты завершены.")