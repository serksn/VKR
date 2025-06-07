# sensitivity_analysis.py

import numpy as np
import pandas as pd

import optimize
from logging_wrapper import log_calls

# --- Обёртка логирования для ключевых функций optimize ---
for name in ('nelder_mead_manual', 'steepest_descent', 'bfgs_manual'):
    if hasattr(optimize, name):
        original = getattr(optimize, name)
        setattr(optimize, name, log_calls(original))

from step_response import simulate_axis_step
from optimize import compute_metrics, objective_ISE, run_optim

# ---- Параметры моделирования ----
dt     = 0.002       # шаг интегрирования
t_fin  = 7.0         # длительность моделирования
t_step = 1.0         # время «скачка»
amp    = np.deg2rad(10)  # амплитуда в радианах

# ---- Моменты инерции для трёх осей (численные значения!) ----
I_vals = {
    'Roll':  0.1,
    'Pitch': 0.1,
    'Yaw':   0.2
}

# ---- Базовые PID-конфигурации (для «неоптимизируемых» осей) ----
base_cfgs = {
    'Roll':  {'Pp':4.0,'Ip':0.2,'Dp':2.0, 'Pv':8.0,'Iv':1.1,'Dv':1.0,'OGR':np.deg2rad(20)},
    'Pitch': {'Pp':4.0,'Ip':0.8,'Dp':2.0, 'Pv':8.0,'Iv':1.1,'Dv':1.0,'OGR':np.deg2rad(20)},
    'Yaw':   {'Pp':2.0,'Ip':0.02,'Dp':0.1,'Pv':0.5,'Iv':0.01,'Dv':0.05,'OGR':np.deg2rad(60)},
}

# ---- Начальная точка и границы оптимизации (по 6 параметров PID) ----
x0     = np.array([3.0, 0.5, 0.2,   2.0, 0.1, 0.05])
bounds = np.array([[1,50], [0,5], [0,10], [1,50], [0,5], [0,10]])

# ---- Методы и критерий ----
methods   = ['Симплекс Нелдера-Мида', 'Наискорейший спуск', 'BFGS']
crit_name = 'Квадратичный'

# ---- Вычисляем эталонное (базовое) значение глобального ISE ----
metrics_base = {}
for ax in ('Roll','Pitch','Yaw'):
    t_base, y_base, _ = simulate_axis_step(
        I_vals[ax],
        base_cfgs[ax],
        dt, t_fin, t_step, amp
    )
    ISE_base, OS_base, Ts_base = compute_metrics(t_base, y_base, amp)
    metrics_base[ax] = (ISE_base, OS_base, Ts_base)

baseline_global_ISE = sum(m[0] for m in metrics_base.values())
print(f"\n=== Baseline global ISE (no optimization) = {baseline_global_ISE:.6f} ===\n")

# ---- Сбор результатов оптимизаций и сравнение с базой ----
rows = []

for axis in ('Roll','Pitch','Yaw'):
    optimize.axis_name = axis
    inertia = I_vals[axis]
    base = base_cfgs[axis]
    # Задаём стартовую точку хуже базовых (или равную им)
    vals = np.array([base[k] for k in ('Pp','Ip','Dp','Pv','Iv','Dv')])
    if axis in ('Roll','Pitch'):
        x0 = vals * 0.1
    else:  # Yaw
        x0 = vals

    for method in methods:
        print(f"\n>>> Оптимизация оси '{axis}' методом '{method}'")
        res, t_opt = run_optim(
            method,
            crit_name,
            objective_ISE,
            x0.copy(),
            bounds,
            (inertia, dt, t_fin, t_step, amp)
        )
        print(f"<<< Готово: {axis}/{method} за {t_opt:.2f}s, J_opt = {res.fun:.6f}")

        # Собираем PID-конфигурации: для оптимизируемой оси – найденные, остальным – базовые
        cfgs = {}
        for ax, base in base_cfgs.items():
            if ax == axis:
                cfgs[ax] = dict(zip(('Pp','Ip','Dp','Pv','Iv','Dv'), res.x), OGR=base['OGR'])
            else:
                cfgs[ax] = base

        # Симуляция всех трёх каналов и сбор метрик
        metrics = {}
        for ax in ('Roll','Pitch','Yaw'):
            t, y, _ = simulate_axis_step(
                I_vals[ax],
                cfgs[ax],
                dt, t_fin, t_step, amp
            )
            ISE, OS, Ts = compute_metrics(t, y, amp)
            metrics[ax] = (ISE, OS, Ts)

        global_ISE = sum(m[0] for m in metrics.values())

        # Рассчитываем изменение качества ISE относительно базового
        delta_ISE      = baseline_global_ISE - global_ISE
        delta_ISE_pct  = delta_ISE / baseline_global_ISE * 100

        rows.append({
            'Ось':           axis,
            'Метод':         method,
            'Начальная оценка качества':   baseline_global_ISE,
            'Оптимизированная оценка качества':  global_ISE,
            'Δ I':          delta_ISE,
            'Δ I (%)':      delta_ISE_pct,
            'T_опт (s)':      t_opt
        })

# Вывод итоговой таблицы
df = pd.DataFrame(rows)
print("\n=== Изменение глобального ISE при оптимизации одной оси ===")
print(df.to_string(index=False))
