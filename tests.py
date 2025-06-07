# optimize_hybrid_tests.py

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

import optimize_hybrid
from optimize_hybrid import objective_total, x0, bounds, I, dt, t_fin, t_step, amp
from optimize import compute_metrics
from step_response import simulate_axis_step

# Папка для результатов
RESULTS_DIR = "hybrid_test_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Гиперпараметры точно как в optimize_hybrid.py
NM_PARAMS = dict(alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5,
                 max_iter=100, tol=1e-2, restarts=10, jitter=0.5)
SD_PARAMS = dict(alpha0=1.0, beta=0.9, c1=1e-4, rho=0.5,
                 max_iter=50, tol_grad=1e-4)
BFGS_PARAMS = dict(max_iter=20, tol_grad=1e-5,
                   c1=1e-4, rho=0.5, eps_grad=1e-6, gamma=1e-3)

def run_full_hybrid(x_initial):
    """Три этапа оптимизации: Nelder-Mead → Steepest Descent → BFGS."""
    mod = optimize_hybrid.optimize  # это импорт optimize
    # Stage 1
    res1 = mod.nelder_mead_manual(
        objective_total, x_initial, bounds, (), **NM_PARAMS
    )
    x1 = res1.x
    # Stage 2
    res2 = mod.steepest_descent(
        objective_total, x1, bounds, (), **SD_PARAMS
    )
    x2 = res2.x
    # Stage 3
    res3 = mod.bfgs_manual(
        objective_total, x2, bounds, (), **BFGS_PARAMS
    )
    return res3.x

def test_step_response():
    """1) Для каждого из каналов: построить «до/после» +5° и проверить ΔISE, OS и Ts."""
    x_opt = run_full_hybrid(x0)
    for idx, axis_name in enumerate(['roll','pitch','yaw']):
        inertia = I[axis_name]
        # исходный PID
        xi0 = x0[6*idx:6*(idx+1)]
        cfg0 = dict(zip(('Pp','Ip','Dp','Pv','Iv','Dv'), xi0), OGR=amp*2)
        t0, y0, _ = simulate_axis_step(inertia, cfg0, dt, t_fin, t_step, amp)
        I0, OS0, Ts0 = compute_metrics(t0, y0, amp)
        # оптимальный PID
        xi_opt = x_opt[6*idx:6*(idx+1)]
        cfg_opt = dict(zip(('Pp','Ip','Dp','Pv','Iv','Dv'), xi_opt), OGR=amp*2)
        t1, y1, _ = simulate_axis_step(inertia, cfg_opt, dt, t_fin, t_step, amp)
        I1, OS1, Ts1 = compute_metrics(t1, y1, amp)
        # график
        plt.figure()
        plt.plot(t0, np.rad2deg(y0), '--', label='до оптимизации')
        plt.plot(t1, np.rad2deg(y1),  '-', label='после оптимизации')
        plt.title(f"{axis_name.capitalize()} step-response (+5°)")
        plt.xlabel("Время, с"); plt.ylabel("Угол, °")
        plt.legend(); plt.grid(True)
        fn = os.path.join(RESULTS_DIR, f"{axis_name}_step.png")
        plt.savefig(fn); plt.close()
        print(f"Сохранён {fn}")
        # проверки
        assert I1 <= I0 * 0.97, f"{axis_name}: ΔISE < 3% ({I0:.4f} → {I1:.4f})"
        assert OS1 <= OS0 * 0.90,  f"{axis_name}: Overshoot не ↓10% ({OS0:.1f}% → {OS1:.1f}%)"
        assert Ts1 <= Ts0 * 0.90,  f"{axis_name}: Ts не ↓10% ({Ts0:.3f}s → {Ts1:.3f}s)"

def test_angle_variation():
    """2) График откликов всех трёх осей при скачке +15°, проверка ΔISE ≥2%."""
    x_opt = run_full_hybrid(x0)
    new_amp = np.deg2rad(15)
    plt.figure(figsize=(8,4))
    for idx, axis_name in enumerate(['roll','pitch','yaw']):
        inertia = I[axis_name]
        xi_opt = x_opt[6*idx:6*(idx+1)]
        cfg_opt = dict(zip(('Pp','Ip','Dp','Pv','Iv','Dv'), xi_opt), OGR=new_amp*2)
        t, y, _ = simulate_axis_step(inertia, cfg_opt, dt, t_fin, t_step, new_amp)
        plt.plot(t, np.rad2deg(y), label=axis_name.capitalize())
        # baseline при новой амплитуде
        xi0 = x0[6*idx:6*(idx+1)]
        cfg0 = dict(zip(('Pp','Ip','Dp','Pv','Iv','Dv'), xi0), OGR=new_amp*2)
        t0, y0, _ = simulate_axis_step(inertia, cfg0, dt, t_fin, t_step, new_amp)
        I0, *_ = compute_metrics(t0, y0, new_amp)
        I1, *_ = compute_metrics(t, y, new_amp)
        assert I1 <= I0 * 0.98, f"{axis_name}: ΔISE <2% при +15° ({I0:.4f} → {I1:.4f})"
    plt.title("Отклики осей при скачке +15°")
    plt.xlabel("Время, с"); plt.ylabel("Угол, °")
    plt.legend(); plt.grid(True)
    fn = os.path.join(RESULTS_DIR, "all_axes_15deg.png")
    plt.savefig(fn); plt.close()
    print(f"Сохранён {fn}")

def test_multiple_initials():
    """5) 10 запусков с разными x0, таблица начальных→конечных параметров."""
    table_path = os.path.join(RESULTS_DIR, "hybrid_multiple_runs.csv")
    with open(table_path, 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.writer(csvf)
        header = [f"x0_{i}" for i in range(18)] + [f"xopt_{i}" for i in range(18)]
        writer.writerow(header)
        for run in range(10):
            # небольшой рандомный старт ±10%
            jitter = 0.1*(np.random.rand(18)-0.5)
            x_start = x0*(1 + jitter)
            x_end = run_full_hybrid(x_start)
            writer.writerow(list(np.round(x_start,4)) + list(np.round(x_end,4)))
    print(f"Таблица результатов сохранена: {table_path}")

if __name__ == "__main__":
    try:
        test_step_response()
        test_angle_variation()
        test_multiple_initials()
        print("\nВсе тесты успешно завершены.")
        sys.exit(0)
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
