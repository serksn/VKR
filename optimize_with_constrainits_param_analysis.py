import numpy as np
import pandas as pd
import logging
from concurrent.futures import ProcessPoolExecutor
from optimize import compute_metrics, nelder_mead_manual, steepest_descent, bfgs_manual
from step_response import simulate_axis_step
from optimize_with_constraints_analysis import (
    x0, bounds, dt_opt, t_fin_opt, dt_fine, t_fin_fine,
    t_step, amp, I, significantCoef, lambda_os,
    penalty_bounds, penalty_metrics, penalty_effort
)

# логгирование
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def J_pen_param(x, R, mu, dt, t_fin):
    J = 0.0
    for axis, coef in significantCoef.items():
        idx = ['roll','pitch','yaw'].index(axis)
        xi = x[6*idx:6*(idx+1)]
        cfg = dict(zip(('Pp','Ip','Dp','Pv','Iv','Dv'), xi), OGR=amp*2)
        t, y, _ = simulate_axis_step(I[axis], cfg, dt, t_fin, t_step, amp)
        ISE, OS, _ = compute_metrics(t, y, amp)
        J += coef * (ISE + lambda_os*OS)
    return J + R*(penalty_bounds(x)+penalty_metrics(x, dt, t_fin)) + mu*penalty_effort(x, dt, t_fin)

def optimize_for(params):
    R, mu = params
    logging.info(f"R={R}, mu={mu} start")
    x = x0.copy()
    # nelder-mead (меньше итераций и рестартов)
    res1 = nelder_mead_manual(
        lambda xx: J_pen_param(xx, R, mu, dt_opt, t_fin_opt),
        x, bounds, (),
        alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5, jitter=0.5,
        max_iter=15, restarts=1, tol=1e-2
    )
    x = res1.x
    # градиентный спуск (меньше итераций)
    res2 = steepest_descent(
        lambda xx: J_pen_param(xx, R, mu, dt_opt, t_fin_opt),
        x, bounds, (),
        alpha0=1.0, beta=0.9, c1=1e-4, rho=0.5,
        max_iter=5, tol_grad=1e-4
    )
    x = res2.x
    # BFGS (меньше итераций)
    res3 = bfgs_manual(
        lambda xx: J_pen_param(xx, R, mu, dt_fine, t_fin_fine),
        x, bounds, (),
        max_iter=10, tol_grad=1e-6, c1=1e-4, rho=0.5,
        eps_grad=1e-6, gamma=1e-3
    )
    x_opt = res3.x
    logging.info(f"R={R}, mu={mu} done")
    # сбор метрик
    data = {'R': R, 'mu': mu}
    for axis in ['roll','pitch','yaw']:
        idx = ['roll','pitch','yaw'].index(axis)
        xi = x_opt[6*idx:6*(idx+1)]
        cfg = dict(zip(('Pp','Ip','Dp','Pv','Iv','Dv'), xi), OGR=amp*2)
        t, y, _ = simulate_axis_step(I[axis], cfg, dt_fine, t_fin_fine, t_step, amp)
        ISE, OS, Ts = compute_metrics(t, y, amp)
        data[f'ISE_{axis}'] = ISE
        data[f'OS_{axis}']  = OS
        data[f'Ts_{axis}']  = Ts
    data['EnergyPenalty'] = penalty_effort(x_opt, dt_fine, t_fin_fine)
    return data

# В конце файла замените вывод на:
if __name__ == "__main__":
    params = [(R, mu) for R in (1e3, 1e4, 1e5) for mu in (1e-3, 1e-2, 1e-1)]
    with ProcessPoolExecutor() as exe:
        results = list(exe.map(optimize_for, params))
    df = pd.DataFrame(results)

    print("\n=== Influence of R and μ ===\n")
    # Заменяем to_markdown на to_string для отсутствующего tabulate
    print(df.to_string(index=False, float_format="%.4f"))
