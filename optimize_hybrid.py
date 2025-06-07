import time
import numpy as np
import matplotlib.pyplot as plt

import optimize
from optimize import compute_metrics
from step_response import simulate_axis_step

# Декоратор для логгирования этапов
def log_progress(func):
    def wrapper(*args, **kwargs):
        print(f"\n>>> Запуск {func.__name__} ...")
        t0 = time.time()
        result = func(*args, **kwargs)
        t1 = time.time()
        print(f"<<< Завершено {func.__name__} (J = {result.fun:.6f}, time = {t1-t0:.2f}s)")
        return result
    return wrapper

# Оборачиваем методы optimize.py
optimize.nelder_mead_manual = log_progress(optimize.nelder_mead_manual)
optimize.steepest_descent = log_progress(optimize.steepest_descent)
optimize.bfgs_manual      = log_progress(optimize.bfgs_manual)

# Параметры модели
I = {'roll': 0.1, 'pitch': 0.1, 'yaw': 0.2}
singificantCoef = {'roll': 0.45, 'pitch': 0.45, 'yaw': 0.1}
dt, t_fin, t_step = 0.001, 7.0, 1.0
amp = np.deg2rad(10)
lambda_os = 0.5

# Начальные PID-параметры (Pp, Ip, Dp, Pv, Iv, Dv) для каждой оси
x0 = np.hstack([
    [3. , 0.5, 2. , 1. , 2. , 2.],  # roll
    [3. , 0.5, 2. , 1. , 2. , 2.],  # pitch
    [3. , 0.5, 2. , 1. , 2. , 2.]   # yaw
])

# Границы для параметров
bounds_axis = np.array([[1,50],[0,5],[0,10],[1,50],[0,5],[0,10]])
bounds = np.tile(bounds_axis, (3,1))

# Суммарный критерий J = ∑(ISE + λ·OS) по трём осям
def objective_total(x):
    J = 0.0
    for idx, axis in enumerate(['roll','pitch','yaw']):
        xi = x[6*idx:6*(idx+1)]
        cfg = dict(zip(('Pp','Ip','Dp','Pv','Iv','Dv'), xi.tolist()), OGR=amp*2.0)
        t, y, _ = simulate_axis_step(I[axis], cfg, dt, t_fin, t_step, amp)
        ISE, OS, _ = compute_metrics(t, y, amp)
        J += singificantCoef[axis]*(ISE + lambda_os * OS)
    return J

if __name__ == "__main__":
    # Stage 1: global exploration
    res1 = optimize.nelder_mead_manual(
        objective_total, x0, bounds, args=(),
        alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5,
        max_iter=100, tol=1e-2, restarts=10, jitter=0.5
    )
    x1 = res1.x

    # Stage 2: local tuning
    res2 = optimize.steepest_descent(
        objective_total, x1, bounds, args=(),
        alpha0=1.0, beta=0.9, c1=1e-4, rho=0.5,
        max_iter=50, tol_grad=1e-4
    )
    x2 = res2.x

    # Stage 3: superlinear refinement
    res3 = optimize.bfgs_manual(
        objective_total, x2, bounds, args=(),
        max_iter=20, tol_grad=1e-5,
        c1=1e-4, rho=0.5, eps_grad=1e-6, gamma=1e-3
    )
    x3 = res3.x

    print("\n=== Optimal PID parameters ===")
    for i, name in enumerate(['roll','pitch','yaw']):
        vals = x3[6*i:6*(i+1)]
        print(f"{name:6s}: {np.round(vals,3)}")

    # Построение переходных функций до и после
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for idx, axis in enumerate(['roll','pitch','yaw']):
        # До оптимизации
        xi0 = x0[6*idx:6*(idx+1)]
        cfg0 = dict(zip(('Pp','Ip','Dp','Pv','Iv','Dv'), xi0.tolist()), OGR=amp*2.0)
        t0, y0, _ = simulate_axis_step(I[axis], cfg0, dt, t_fin, t_step, amp)
        # После оптимизации
        xi3 = x3[6*idx:6*(idx+1)]
        cfg3 = dict(zip(('Pp','Ip','Dp','Pv','Iv','Dv'), xi3.tolist()), OGR=amp*2.0)
        t3, y3, _ = simulate_axis_step(I[axis], cfg3, dt, t_fin, t_step, amp)

        ax = axs[idx]
        ax.plot(t0, y0, linestyle='--', label='До оптимизации')
        ax.plot(t3, y3, label='После оптимизации')
        ax.set_title(axis.capitalize())
        ax.set_xlabel('Время, с')
        if idx == 0:
            ax.set_ylabel('Угол, рад')
        ax.legend(); ax.grid(True)

    plt.tight_layout()
    plt.show()