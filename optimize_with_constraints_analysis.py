"""
optimize_with_constraints.py

Расширение модуля оптимизации с учётом «жёстких» ограничений и выводом промежуточных результатов для оценки качества.
"""
import numpy as np
import matplotlib.pyplot as plt

from optimize import compute_metrics, nelder_mead_manual, steepest_descent, bfgs_manual
from pid import PIDController
from step_response import simulate_axis_step

# Параметры модели гироскопа и оптимизации
I = {'roll': 0.1, 'pitch': 0.1, 'yaw': 0.2}
significantCoef = {'roll': 0.45, 'pitch': 0.45, 'yaw': 0.1}
amplitude = np.deg2rad(10)
lambda_os = 0.5
mu_effort = 1e-2
x0 = np.hstack([[3.,0.5,2.,1.,2.,2.]]*3)
bounds_axis = np.array([[1,30],[0,5],[0,10],[1,30],[0,5],[0,10]])
bounds = np.tile(bounds_axis,(3,1))
OS_max, Ts_max = 5.0, 0.8
# Параметры дискретизации
dt_opt, t_fin_opt = 0.005, 5.0
dt_fine, t_fin_fine = 0.001, 7.0
t_step = 1.0
amp = amplitude

# -------------------- Штрафные функции --------------------
def penalty_bounds(x):
    h_min = bounds[:,0] - x
    h_max = x - bounds[:,1]
    return np.sum(np.maximum(0.0, h_min)**2 + np.maximum(0.0, h_max)**2)

def simulate_with_torque(I_val, cfg, dt, t_final):
    pid_p = PIDController(kp=cfg['Pp'], ki=cfg['Ip'], kd=cfg['Dp'], dt=dt,
                          u_min=-cfg['OGR'], u_max=cfg['OGR'])
    pid_v = PIDController(kp=cfg['Pv'], ki=cfg['Iv'], kd=cfg['Dv'], dt=dt,
                          u_min=-cfg['OGR'], u_max=cfg['OGR'])
    t = np.arange(0, t_final, dt)
    torque = np.zeros_like(t)
    angle = omega = 0.0
    for i, ti in enumerate(t):
        setpoint = 0.0 if ti < t_step else amplitude
        rate_cmd = pid_p.update(setpoint - angle)
        u = pid_v.update(rate_cmd - omega)
        torque[i] = u
        omega += (u/I_val)*dt
        angle += omega*dt
    return t, torque

def penalty_metrics(x, dt, t_fin):
    pen = 0.0
    for idx, axis in enumerate(['roll','pitch','yaw']):
        xi = x[6*idx:6*(idx+1)]
        cfg = dict(zip(('Pp','Ip','Dp','Pv','Iv','Dv'), xi), OGR=amp*2)
        t, y, _ = simulate_axis_step(I[axis], cfg, dt, t_fin, t_step, amp)
        _, OS, Ts = compute_metrics(t, y, amp)
        pen += max(0.0, OS-OS_max)**2 + max(0.0, Ts-Ts_max)**2
    return pen

def penalty_effort(x, dt, t_fin):
    pen = 0.0
    for idx, axis in enumerate(['roll','pitch','yaw']):
        xi = x[6*idx:6*(idx+1)]
        cfg = dict(zip(('Pp','Ip','Dp','Pv','Iv','Dv'), xi), OGR=amp*2)
        t, torque = simulate_with_torque(I[axis], cfg, dt, t_fin)
        pen += np.trapz(torque**2, t)
    return pen

def J_pen(x, R, dt, t_fin):
    J = 0.0
    for axis, coef in significantCoef.items():
        idx = ['roll','pitch','yaw'].index(axis)
        xi = x[6*idx:6*(idx+1)]
        cfg = dict(zip(('Pp','Ip','Dp','Pv','Iv','Dv'), xi), OGR=amp*2)
        t, y, _ = simulate_axis_step(I[axis], cfg, dt, t_fin, t_step, amp)
        ISE, OS, _ = compute_metrics(t, y, amp)
        J += coef*(ISE + lambda_os*OS)
    return J + R*(penalty_bounds(x) + penalty_metrics(x, dt, t_fin)) + mu_effort*penalty_effort(x, dt, t_fin)

# -------------------- Вывод метрик --------------------
def print_metrics(label, x, dt, t_fin):
    print(f"--- {label} metrics (dt={dt}, t_fin={t_fin}) ---")
    for axis in ['roll','pitch','yaw']:
        idx = ['roll','pitch','yaw'].index(axis)
        xi = x[6*idx:6*(idx+1)]
        cfg = dict(zip(('Pp','Ip','Dp','Pv','Iv','Dv'), xi), OGR=amp*2)
        t, y, _ = simulate_axis_step(I[axis], cfg, dt, t_fin, t_step, amp)
        ISE, OS, Ts = compute_metrics(t, y, amp)
        print(f"{axis.capitalize():6s} -> ISE: {ISE:.3f}, OS: {OS:.2f}%, Ts: {Ts:.3f}s")
    effort = penalty_effort(x, dt, t_fin)
    print(f"Energy penalty: {effort:.3f}\n")

# -------------------- Основной процесс --------------------

def run_unconstrained():
    """Проведение оптимизации без ограничений"""
    x = x0.copy()
    # Грубая оптимизация без штрафов (R=0, mu=0)
    for R_dummy in [0]:
        res1 = nelder_mead_manual(lambda x: J_base(x, 0, dt_opt, t_fin_opt), x, bounds, (),
                                   alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5, jitter=0.5,
                                   max_iter=50, restarts=3, tol=1e-2)
        x = res1.x
        res2 = steepest_descent(lambda x: J_base(x, 0, dt_opt, t_fin_opt), x, bounds, (),
                                 alpha0=1.0, beta=0.9, c1=1e-4, rho=0.5,
                                 max_iter=20, tol_grad=1e-4)
        x = res2.x
        res3 = bfgs_manual(lambda x: J_base(x, 0, dt_opt, t_fin_opt), x, bounds, (),
                            max_iter=10, tol_grad=1e-5, c1=1e-4, rho=0.5,
                            eps_grad=1e-6, gamma=1e-3)
        x = res3.x
    # Финальная доводка
    res_final = bfgs_manual(lambda x: J_base(x, 0, dt_fine, t_fin_fine), x, bounds, (),
                             max_iter=20, tol_grad=1e-6, c1=1e-4, rho=0.5,
                             eps_grad=1e-6, gamma=1e-3)
    return res_final.x

if __name__ == "__main__":
    # Определение базовой функции качества без штрафов
    def J_base(x, R_unused, dt, t_fin):
        J = 0.0
        for axis, coef in significantCoef.items():
            idx = ['roll','pitch','yaw'].index(axis)
            xi = x[6*idx:6*(idx+1)]
            cfg = dict(zip(('Pp','Ip','Dp','Pv','Iv','Dv'), xi), OGR=amp*2)
            t, y, _ = simulate_axis_step(I[axis], cfg, dt, t_fin, t_step, amp)
            ISE, OS, _ = compute_metrics(t, y, amp)
            J += coef*(ISE + lambda_os*OS)
        return J

    # 1) Без ограничений
    x_unc = run_unconstrained()
    print_metrics('Без ограничений', x_unc, dt_fine, t_fin_fine)

    # 2) С ограничениями
    x = x0.copy()
    print_metrics('Initial', x, dt_fine, t_fin_fine)
    for R in [1e3, 1e4, 1e5]:
        print(f"=== Optimization R={R:.0e}, coarse ===")
        res1 = nelder_mead_manual(
            lambda x: J_pen(x, R, dt_opt, t_fin_opt), x, bounds, (),
            alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5, jitter=0.5,
            max_iter=50, restarts=3, tol=1e-2
        )
        x = res1.x
        res2 = steepest_descent(
            lambda x: J_pen(x, R, dt_opt, t_fin_opt), x, bounds, (),
            alpha0=1.0, beta=0.9, c1=1e-4, rho=0.5,
            max_iter=20, tol_grad=1e-4
        )
        x = res2.x
        res3 = bfgs_manual(
            lambda x: J_pen(x, R, dt_opt, t_fin_opt), x, bounds, (),
            max_iter=10, tol_grad=1e-5, c1=1e-4, rho=0.5,
            eps_grad=1e-6, gamma=1e-3
        )
        x = res3.x
        print_metrics(f'After R={R:.0e}', x, dt_opt, t_fin_opt)

    print("=== Final tuning constrained ===")
    res_final = bfgs_manual(
        lambda x: J_pen(x, 1e5, dt_fine, t_fin_fine), x, bounds, (),
        max_iter=20, tol_grad=1e-6, c1=1e-4, rho=0.5,
        eps_grad=1e-6, gamma=1e-3
    )
    x_opt = res_final.x
    print_metrics('Constrained Final', x_opt, dt_fine, t_fin_fine)

    # Сравнительный вывод
    print("=== Сравнение PID параметров ===")
    for label, vec in [('Без ограничений', x_unc), ('С ограничениями', x_opt)]:
        print(f"{label} parameters:")
        for i, axis in enumerate(['roll','pitch','yaw']):
            vals = np.round(vec[6*i:6*(i+1)], 3)
            print(f"  {axis:6s}: {vals}")

    # Рисунки: шаговые характеристики
        # Рисунок 1: до оптимизации и после (только initial vs constrained)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for idx, axis in enumerate(['roll','pitch','yaw']):
        cfg0 = dict(zip(('Pp','Ip','Dp','Pv','Iv','Dv'), x0[6*idx:6*(idx+1)]), OGR=amp*2)
        t0, y0, _ = simulate_axis_step(I[axis], cfg0, dt_fine, t_fin_fine, t_step, amp)
        cfg_con = dict(zip(('Pp','Ip','Dp','Pv','Iv','Dv'), x_opt[6*idx:6*(idx+1)]), OGR=amp*2)
        t_con, y_con, _ = simulate_axis_step(I[axis], cfg_con, dt_fine, t_fin_fine, t_step, amp)

        ax = axs[idx]
        ax.plot(t0, y0, '--', label='До оптимизации')
        ax.plot(t_con, y_con, '-', label='После оптимизации')
        ax.set_title(axis.capitalize())
        ax.set_xlabel('Время, с')
        if idx == 0:
            ax.set_ylabel('Угол, рад')
        ax.legend(); ax.grid(True)
    plt.tight_layout(); plt.show()

    # Рисунок 2: сравнение без ограничений и с ограничениями
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for idx, axis in enumerate(['roll','pitch','yaw']):
        cfg_unc = dict(zip(('Pp','Ip','Dp','Pv','Iv','Dv'), x_unc[6*idx:6*(idx+1)]), OGR=amp*2)
        t_unc, y_unc, _ = simulate_axis_step(I[axis], cfg_unc, dt_fine, t_fin_fine, t_step, amp)
        cfg_con = dict(zip(('Pp','Ip','Dp','Pv','Iv','Dv'), x_opt[6*idx:6*(idx+1)]), OGR=amp*2)
        t_con, y_con, _ = simulate_axis_step(I[axis], cfg_con, dt_fine, t_fin_fine, t_step, amp)

        ax = axs[idx]
        ax.plot(t_unc, y_unc, '-.', label='Без ограничений')
        ax.plot(t_con, y_con, '-', label='С ограничениями')
        ax.set_title(axis.capitalize())
        ax.set_xlabel('Время, с')
        if idx == 0:
            ax.set_ylabel('Угол, рад')
        ax.legend(); ax.grid(True)
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    x = x0.copy()
    print_metrics('Initial', x, dt_fine, t_fin_fine)
    for R in [1e3, 1e4, 1e5]:
        print(f"=== Optimization R={R:.0e}, coarse ===")
        res1 = nelder_mead_manual(
            lambda x: J_pen(x, R, dt_opt, t_fin_opt), x, bounds, (),
            alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5, jitter=0.5,
            max_iter=50, restarts=3, tol=1e-2
        )
        x = res1.x
        res2 = steepest_descent(
            lambda x: J_pen(x, R, dt_opt, t_fin_opt), x, bounds, (),
            alpha0=1.0, beta=0.9, c1=1e-4, rho=0.5,
            max_iter=20, tol_grad=1e-4
        )
        x = res2.x
        res3 = bfgs_manual(
            lambda x: J_pen(x, R, dt_opt, t_fin_opt), x, bounds, (),
            max_iter=10, tol_grad=1e-5, c1=1e-4, rho=0.5,
            eps_grad=1e-6, gamma=1e-3
        )
        x = res3.x
        print_metrics(f'After R={R:.0e}', x, dt_opt, t_fin_opt)

    print("=== Final tuning ===")
    res_final = bfgs_manual(
        lambda x: J_pen(x, 1e5, dt_fine, t_fin_fine), x, bounds, (),
        max_iter=20, tol_grad=1e-6, c1=1e-4, rho=0.5,
        eps_grad=1e-6, gamma=1e-3
    )
    x_opt = res_final.x
    print_metrics('Final', x_opt, dt_fine, t_fin_fine)

    # Результаты
    print("=== Optimal PID parameters ===")
    for i, axis in enumerate(['roll','pitch','yaw']):
        vals = np.round(x_opt[6*i:6*(i+1)], 3)
        print(f"{axis:6s}: {vals}")

    # Визуализация
    fig, axs = plt.subplots(1,3,figsize=(15,4), sharey=True)
    for idx, axis in enumerate(['roll','pitch','yaw']):
        xi0 = x0[6*idx:6*(idx+1)]
        cfg0 = dict(zip(('Pp','Ip','Dp','Pv','Iv','Dv'), xi0), OGR=amp*2)
        t0, y0, _ = simulate_axis_step(I[axis], cfg0, dt_fine, t_fin_fine, t_step, amp)

        xi1 = x_opt[6*idx:6*(idx+1)]
        cfg1 = dict(zip(('Pp','Ip','Dp','Pv','Iv','Dv'), xi1), OGR=amp*2)
        t1, y1, _ = simulate_axis_step(I[axis], cfg1, dt_fine, t_fin_fine, t_step, amp)

        ax = axs[idx]
        ax.plot(t0, y0, '--', label='Before')
        ax.plot(t1, y1, '-', label='After')
        ax.set_title(axis.capitalize())
        ax.set_xlabel('Time, s')
        if idx==0:
            ax.set_ylabel('Angle, rad')
        ax.legend(); ax.grid(True)
    plt.tight_layout(); plt.show()
