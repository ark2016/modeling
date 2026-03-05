import numpy as np

g = 9.81
alpha = np.radians(32) # угол выстрела, град
v0 = 115.0 # начальная скорость, м/с
x0 = 0.0
y0 = 0.0
C = 0.15
r = 0.14 # радиус ядра, м
rho_material = 7300 # плотность материала (олово), кг/м³
s = np.pi * r**2 # площадь поперечного сечения, м²
m = rho_material * (4 / 3) * np.pi * r**3 # масса = плотность * объём шара
rho = 1.29 # плотность воздуха, кг/м³
beta = C * s * rho / 2

# Модель Галилея
# y = - \frac{g}{2v_{0}^{2}\cos^{2}\alpha}x^{2}+(\tan \alpha)x,\;0<\alpha< \frac{\pi}{2}
def galileo_trajectory(x_arr):
    cos_a = np.cos(alpha)
    tan_a = np.tan(alpha)
    return -(g / (2 * v0**2 * cos_a**2)) * x_arr**2 + tan_a * x_arr


def galileo_range():
    cos_a = np.cos(alpha)
    tan_a = np.tan(alpha)
    return v0**2 * np.sin(2 * alpha) / g


# Модель Ньютона
# \begin{cases}
# m \frac{du}{dt} = -\beta u \sqrt{ u^{2}+w^{2 }}\;,& \frac{dx}{dt}=u \\
# m \frac{dw}{dt}=-g-\beta w\sqrt{  u^{2}+w^{2 }} \;, & \frac{dy}{dt}=w
# \end{cases}
def newton_rhs(t, state):
    x, y, u, w = state
    speed = np.sqrt(u**2 + w**2)
    bm = beta / m
    dxdt = u
    dydt = w
    dudt = -bm * u * speed
    dwdt = -g - bm * w * speed
    return np.array([dxdt, dydt, dudt, dwdt])


def rk4_step(f, t, state, h):
    """
    Один шаг метода Рунге-Кутты 4-го порядка.
    """
    k1 = f(t, state)
    k2 = f(t + h/2, state + h/2 * k1)
    k3 = f(t + h/2, state + h/2 * k2)
    k4 = f(t + h,   state + h * k3)
    return state + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)


def newton_trajectory(h=0.0001):
    u0 = v0 * np.cos(alpha)
    w0 = v0 * np.sin(alpha)
    state = np.array([x0, y0, u0, w0])
    t = 0.0
    trajectory = [state.copy()]

    while True:
        state_new = rk4_step(newton_rhs, t, state, h)
        t += h
        if state_new[1] < 0:
            # Линейная интерполяция для точки падения (y=0)
            y_prev = state[1]
            y_curr = state_new[1]
            frac = y_prev / (y_prev - y_curr)
            state_land = state + frac * (state_new - state)
            state_land[1] = 0.0
            trajectory.append(state_land)
            break
        state = state_new
        trajectory.append(state.copy())
    trajectory = np.array(trajectory)

    return trajectory[:, 0], trajectory[:, 1] # x, y

if __name__ == "__main__":
    x_max_gal = galileo_range()
    x_gal = np.linspace(0, x_max_gal, 50)
    y_gal = galileo_trajectory(x_gal)
    print("=== Модель Галилея (без сопротивления) ===")
    print(f"Дальность: {x_max_gal:.6f} м")
    print(f"Макс. высота: {np.max(y_gal):.6f} м")
    print(f"{'x, м':>10}  {'y, м':>10}")
    for xi, yi in zip(x_gal[::5], y_gal[::5]):
        print(f"{xi:10.6f}  {yi:10.6f}")
    x_newt, y_newt = newton_trajectory()
    print("=== Модель Ньютона (с сопротивлением воздуха) ===")
    print(f"beta = {beta:.6e}")
    print(f"Дальность: {x_newt[-1]:.6f} м")
    print(f"Макс. высота: {np.max(y_newt):.6f} м")
    step = max(1, len(x_newt) // 10)
    print(f"{'x, м':>10}  {'y, м':>10}")
    for xi, yi in zip(x_newt[::step], y_newt[::step]):
        print(f"{xi:10.6f}  {yi:10.6f}")
    print("=== Сравнение дальности ===")
    print(f"Галилей: {x_max_gal:.6f} м")
    print(f"Ньютон:  {x_newt[-1]:.6f} м")
    print(f"Разница: {abs(x_max_gal - x_newt[-1]):.6f} м")