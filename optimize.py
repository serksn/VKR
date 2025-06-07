# optimize.py

import time
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial

from step_response import simulate_axis_step

# ---------------- Метрики качества ----------------
def compute_metrics(t, y, setpoint):
    """
    Возвращает классические метрики:
      ISE  = ∫ (e(t))^2 dt
      OS%  = max((y_max–setpoint)/|setpoint|,0)·100
      Ts   = первое время, когда |e| <= 2%·|setpoint|
    """
    e = setpoint - y
    ISE = np.trapz(e**2, t)
    y_max = np.max(y)
    OS = max(0.0, (y_max - setpoint) / abs(setpoint)) * 100.0 if setpoint!=0 else 0.0
    tol = 0.02 * abs(setpoint)
    idx = np.where(np.abs(e) <= tol)[0]
    Ts = t[idx[0]] if idx.size else t[-1]
    return ISE, OS, Ts

# ---------------- Целевые функции ----------------
def objective_ISE(x, I, dt, t_fin, t_step, amp):
    cfg = dict(zip(('Pp','Ip','Dp','Pv','Iv','Dv'), x.tolist()), OGR=amp*2.0)
    t, y, _ = simulate_axis_step(I, cfg, dt, t_fin, t_step, amp)
    e = amp - y
    return np.trapz(e**2, t)

def objective_composite(x, I, dt, t_fin, t_step, amp, axis_name='None', x0=None):
    cfg = dict(zip(('Pp','Ip','Dp','Pv','Iv','Dv'), x.tolist()), OGR=amp*2.0)
    t, y, _ = simulate_axis_step(I, cfg, dt, t_fin, t_step, amp)
    e = amp - y
    e_dot = np.gradient(e, t)
    e_dot = np.convolve(e_dot, np.ones(5)/5, mode='same')
    if axis_name not in _mu_cache:
        _mu_cache[axis_name] = init_mu(axis_name, I, dt, t_fin, t_step, amp, x0)
    mu = _mu_cache[axis_name]
    I1 = np.trapz(e**2, t)
    I2 = np.trapz(e_dot**2, t)
    return I1 + (mu**2) * I2

# ------------- Градиент по конечным разностям -------------
def approx_grad(f, x, eps, *args):
    n = len(x); g = np.zeros(n)
    f0 = f(x, *args)
    for i in range(n):
        dx = np.zeros(n); dx[i] = eps
        g[i] = (f(x+dx, *args) - f0) / eps
    return g

def init_mu(axis_name, I, dt, t_fin, t_step, amp, x0):
    t,y,_ = simulate_axis_step(I,
        dict(zip(('Pp','Ip','Dp','Pv','Iv','Dv'), x0), OGR=amp*2),
        dt, t_fin, t_step, amp)
    e = amp - y
    OS = max(0,(np.max(y)-amp)/amp)
    # более «перерегулированной» оси даём больший вес
    return (1+OS) * (t_fin/10.0)

_mu_cache = {}

# ---------------- HYPER-PARAMS по осям ----------------
DEFAULT_HYPER = {
    'Квадратичный': {
        'Симплекс Нелдера-Мида':    dict(alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5, max_iter=200, tol=1e-3, restarts=5, jitter=0.3),
        'Наискорейший спуск':       dict(alpha0=0.5, beta=0.9, c1=1e-4, rho=0.5, max_iter=300, tol_grad=1e-3),
        'BFGS':                     dict(max_iter=500, tol_grad=1e-4, c1=1e-4, rho=0.5, eps_grad=1e-6, gamma=1e-3),
    },
    'Составной': {
        'Симплекс Нелдера-Мида':    dict(alpha=0.8, gamma=2.0, rho=0.6, sigma=0.4, max_iter=300, tol=5e-4, restarts=7, jitter=0.4),
        'Наискорейший спуск':       dict(alpha0=0.4, beta=0.85, c1=5e-5, rho=0.5, max_iter=400, tol_grad=5e-4),
        'BFGS':                     dict(max_iter=700, tol_grad=5e-5, c1=1e-4, rho=0.5, eps_grad=1e-6, gamma=5e-4),
    }
}

AXIS_SPECIFIC_HYPER = {
    # Для оси Yaw сделаем более мягкие шаги
    'Yaw': {
        'Квадратичный': {
            'Симплекс Нелдера-Мида': dict(alpha=0.7, gamma=2.0, rho=0.6, sigma=0.3, max_iter=250, tol=1e-3, restarts=7, jitter=0.4),
            'Наискорейший спуск':    dict(alpha0=0.3, beta=0.85, c1=1e-4, rho=0.5, max_iter=350, tol_grad=1e-3),
            'BFGS':                  dict(max_iter=600, tol_grad=1e-4, c1=1e-4, rho=0.5, eps_grad=1e-6, gamma=1e-3),
        },
        'Составной': {
            'Симплекс Нелдера-Мида': dict(alpha=0.6, gamma=2.0, rho=0.7, sigma=0.3, max_iter=350, tol=5e-4, restarts=9, jitter=0.5),
            'Наискорейший спуск':    dict(alpha0=0.25, beta=0.8, c1=5e-5, rho=0.5, max_iter=450, tol_grad=5e-4),
            'BFGS':                  dict(max_iter=800, tol_grad=5e-5, c1=1e-4, rho=0.5, eps_grad=1e-6, gamma=5e-4),
        }
    }
}

def get_hyper(axis_name, crit_name, method):
    # сначала общий
    h = DEFAULT_HYPER[crit_name][method]
    # если для этой оси есть специфичные — обновим
    key = 'Yaw' if 'Yaw' in axis_name else axis_name
    spec = AXIS_SPECIFIC_HYPER.get(key, {}).get(crit_name, {})
    h_spec = spec.get(method)
    if h_spec:
        h = h_spec
    return h

x0_ISE = {}

# Функция, которая для данной оси один раз считает оптимум по ISE
def init_hot_start(axis_name, I, x0, bounds, args):
    # Берём гиперпараметры именно для ISE + симплекс (или другой метод)
    hp = get_hyper(axis_name, 'Квадратичный', 'Симплекс Нелдера-Мида')
    # Запускаем свой Nelder–Mead по чистому ISE
    res_ISE = nelder_mead_manual(
        objective_ISE,    # целевая функция ISE
        x0,               # первоначальная точка [3,0.5,2,...]
        bounds,
        args,
        **hp
    )
    return res_ISE.x

# ------------ Реализация Nelder–Mead вручную ------------
def nelder_mead_manual(f, x0, bounds, args, *,
                       alpha, gamma, rho, sigma, max_iter, tol, restarts, jitter):
    n = len(x0)
    best = None; best_val = np.inf

    for _ in range(restarts):
        # jitter начальной точки
        x0r = x0 * (1 + jitter*(np.random.rand(n)-0.5))
        x0r = np.clip(x0r, bounds[:,0], bounds[:,1])

        # init simplex
        simplex = [x0r.copy()]
        for i in range(n):
            y = x0r.copy()
            rng = bounds[i,1] - bounds[i,0]
            δ = 0.2*rng if rng>0 else 0.1
            y[i] = np.clip(y[i]+δ, bounds[i,0], bounds[i,1])
            simplex.append(y)
        fs = [f(xx, *args) for xx in simplex]
        no_improve=0; f_best=np.min(fs)

        for it in range(max_iter):
            idx = np.argsort(fs)
            simplex = [simplex[i] for i in idx]
            fs      = [fs[i]      for i in idx]
            if np.std(fs)<tol: break

            # реструктуризация при застое
            if fs[0]<f_best-tol:
                f_best=fs[0]; no_improve=0
            else:
                no_improve+=1
                if no_improve>=10:
                    x_b=simplex[0]
                    simplex=[np.clip(x_b + sigma*(x-x_b), bounds[:,0],bounds[:,1]) for x in simplex]
                    fs=[f(x,*args) for x in simplex]
                    no_improve=0
                    continue

            x_bar = np.mean(simplex[:-1],axis=0)
            # отражение
            x_r = np.clip(x_bar + alpha*(x_bar - simplex[-1]), bounds[:,0],bounds[:,1])
            f_r = f(x_r,*args)
            if fs[0] <= f_r < fs[-2]:
                simplex[-1],fs[-1] = x_r,f_r; continue
            # экспансия
            if f_r < fs[0]:
                x_e = np.clip(x_bar + gamma*(x_r - x_bar), bounds[:,0],bounds[:,1])
                f_e = f(x_e,*args)
                if f_e< f_r: simplex[-1],fs[-1] = x_e,f_e
                else:        simplex[-1],fs[-1] = x_r,f_r
                continue
            # сжатие
            if f_r<fs[-1]:
                x_c = np.clip(x_bar + rho*(x_r - x_bar), bounds[:,0],bounds[:,1])
            else:
                x_c = np.clip(x_bar - rho*(x_bar - simplex[-1]), bounds[:,0],bounds[:,1])
            f_c = f(x_c,*args)
            if f_c<fs[-1]:
                simplex[-1],fs[-1] = x_c,f_c; continue
            # редукция
            x0b=simplex[0]
            simplex=[x0b] + [np.clip(x0b + sigma*(x-x0b), bounds[:,0],bounds[:,1]) for x in simplex[1:]]
            fs=[f(x,*args) for x in simplex]

        cand = simplex[np.argmin(fs)]
        val  = np.min(fs)
        if val<best_val:
            best_val, best = val, cand

    return type('R', (), {'x': best, 'fun': best_val})

# ------------ Steepest Descent с моментумом ------------
def steepest_descent(f, x0, bounds, args, *,
                     alpha0, beta, c1, rho, max_iter, tol_grad):
    x = x0.copy(); v = np.zeros_like(x)
    f_val = f(x, *args)

    for _ in range(max_iter):
        g = approx_grad(f, x, 1e-6, *args)
        if np.linalg.norm(g)<tol_grad: break
        v = beta*v + (1-beta)*g
        p = -v
        α = alpha0
        # Armijo
        while True:
            xn = np.clip(x + α*p, bounds[:,0], bounds[:,1])
            f_new = f(xn,*args)
            if f_new <= f_val + c1*α*np.dot(g,p): break
            α *= rho
            if α<1e-6: break
        x, f_val = xn, f_new
    return type('R', (), {'x': x, 'fun': f_val})

# ------------ Собственный BFGS ------------
def bfgs_manual(f, x0, bounds, args, *,
                max_iter, tol_grad, c1, rho, eps_grad, gamma):
    x = x0.copy(); n = len(x)
    H = np.eye(n)
    f_val = f(x,*args)
    g     = approx_grad(f,x,eps_grad,*args)

    for _ in range(max_iter):
        if np.linalg.norm(g)<tol_grad: break
        p = -H.dot(g)
        α = 1.0
        # Armijo
        while True:
            xn = np.clip(x + α*p, bounds[:,0],bounds[:,1])
            f_new = f(xn,*args)
            if f_new <= f_val + c1*α*g.dot(p): break
            α *= rho
            if α<1e-8: break
        s = xn - x
        g_new = approx_grad(f,xn,eps_grad,*args)
        y = g_new - g
        ys = y.dot(s)
        if ys>1e-8:
            ρb = 1.0/ys
            I_n= np.eye(n)
            H = (I_n - ρb*np.outer(s,y))@H@(I_n - ρb*np.outer(y,s)) + ρb*np.outer(s,s) + gamma*I_n
        x,f_val,g = xn, f_new, g_new

    return type('R', (), {'x': x, 'fun': f_val})

# ------------ Обёртка запуска ------------
def run_optim(method, crit_name, obj_fun, x0, bounds, args):
    hp = get_hyper(axis_name, crit_name, method)
    t0 = time.time()

    if method == 'Симплекс Нелдера-Мида':
        res = nelder_mead_manual(obj_fun, x0, bounds, args, **hp)

    elif method == 'Наискорейший спуск':
        res = steepest_descent(obj_fun, x0, bounds, args, **hp)

    elif method == 'BFGS':
        res = bfgs_manual(obj_fun, x0, bounds, args, **hp)

    else:
        raise ValueError(method)

    t_cpu = time.time() - t0
    return res, t_cpu

# ------------ Обёртка запуска (параллельная версия) ------------
def run_optim_parallel(task):
    method, crit_name, obj_fun, x0, bounds, args, axis_name = task
    hp = get_hyper(axis_name, crit_name, method)
    t0 = time.time()
    if method == 'Симплекс Нелдера-Мида':
        res = nelder_mead_manual(obj_fun, x0, bounds, args + (axis_name, x0), **hp)
    elif method == 'Наискорейший спуск':
        res = steepest_descent(obj_fun, x0, bounds, args + (axis_name, x0), **hp)
    elif method == 'BFGS':
        res = bfgs_manual(obj_fun, x0, bounds, args + (axis_name, x0), **hp)
    else:
        raise ValueError(method)
    t_cpu = time.time() - t0
    # вместо объекта возвращаем только данные
    return (
        axis_name,
        crit_name,
        method,
        res.x,      # массив оптимальных коэффициентов
        res.fun,    # значение целевой функции
        t_cpu       # время работы
    )

# ----------------------- MAIN -----------------------
if __name__=='__main__':
    Ixx, Iyy, Izz = 0.1, 0.1, 0.2
    dt, t_fin, t_step = 0.002, 7.0, 1.0
    amp = np.deg2rad(10)

    x0     = np.array([3., 0.5, 2., 1., 2., 2.])
    bounds = np.array([
        [1, 10],  # Pp
        [0, 2],   # Ip
        [0, 5],   # Dp
        [1, 20],  # Pv
        [0, 2],   # Iv
        [0, 5],   # Dv
    ])
    axes    = [('Roll (φ)', Ixx), ('Pitch (θ)', Iyy), ('Yaw (ψ)', Izz)]
    methods = ['Симплекс Нелдера-Мида','Наискорейший спуск','BFGS']
    criteria= [('Составной', objective_composite)]

    # Заголовок таблицы
    header = f"{'Ось':10s} {'Критерий':12s} {'Метод':22s} {'Kритерий':>10s} {'OS%':>8s} {'Ts(с)':>8s} {'T_opt(с)':>10s}  Параметры"
    print(header)

    # 1) Первоначальные результаты без оптимизации
    print("\n=== Результаты ДО оптимизации ===")
    initial_results = {}
    for axis_name, I in axes:
        args = (I, dt, t_fin, t_step, amp)
        cfg = dict(zip(('Pp','Ip','Dp','Pv','Iv','Dv'), x0), OGR=amp*2)
        t_init, y_init, _ = simulate_axis_step(I, cfg, dt, t_fin, t_step, amp, torque_disturbance=0.1)
        ISE_init, OS_init, Ts_init = compute_metrics(t_init, y_init, amp)
        initial_results[axis_name] = (t_init, y_init, ISE_init, OS_init, Ts_init)
        print(f"{axis_name:10s} {'Исходные':12s} {'Параметры':22s} {ISE_init:10.4f} {OS_init:8.2f} {Ts_init:8.3f} {'-':>10s}  {np.round(x0,3)}")

    # 2) Горячие старты по ISE
    print("\n=== Горячие старты по ISE ===")
    for axis_name, I in axes:
        args = (I, dt, t_fin, t_step, amp)
        x0_ISE[axis_name] = init_hot_start(axis_name, I, x0, bounds, args)
        print(f"Горячий старт для {axis_name}: {np.round(x0_ISE[axis_name], 3)}")

    # 3) Основная оптимизация composite (параллельная)
    print("\n=== Основная оптимизация ===")
    tasks = []
    for crit_name, crit_fun in criteria:
        for axis_name, I in axes:
            args = (I, dt, t_fin, t_step, amp)
            x_start = x0_ISE[axis_name]
            for method in methods:
                tasks.append((method, crit_name, crit_fun, x_start.copy(), bounds.copy(), args, axis_name))

    # Параллельное выполнение
    with Pool(min(len(tasks), cpu_count())) as pool:
        optim_results = pool.map(run_optim_parallel, tasks)

    # Сбор и вывод результатов
    result_idx = 0
    for crit_name, crit_fun in criteria:
        for axis_name, I in axes:
            args = (I, dt, t_fin, t_step, amp)
            x_start = x0_ISE[axis_name]
            
            # Результаты горячего старта
            cfg_hot = dict(zip(('Pp','Ip','Dp','Pv','Iv','Dv'), x_start), OGR=amp*2)
            t_hot, y_hot, _ = simulate_axis_step(I, cfg_hot, dt, t_fin, t_step, amp, torque_disturbance=0.1)
            ISE_hot, OS_hot, Ts_hot = compute_metrics(t_hot, y_hot, amp)
            print(f"{axis_name:10s} {crit_name:12s} {'Горячий старт':22s} {ISE_hot:10.4f} {OS_hot:8.2f} {Ts_hot:8.3f} {'-':>10s}  {np.round(x_start,3)}")
            
            # Результаты оптимизации
            for method in methods:
                axis_r, crit_r, method_r, x_opt, fun_opt, t_opt = optim_results[result_idx]
                #res, t_opt = optim_results[result_idx]
                result_idx += 1
                cfg = dict(zip(('Pp','Ip','Dp','Pv','Iv','Dv'), x_opt), OGR=amp*2)
                t_post, y_post, _ = simulate_axis_step(I, cfg, dt, t_fin, t_step, amp, torque_disturbance=0.1)
                ISE1, OS1, Ts1 = compute_metrics(t_post, y_post, amp)
                print(f"{axis_name:10s} {crit_name:12s} {method:22s} {fun_opt:10.4f} {OS1:8.2f} {Ts1:8.3f} {t_opt:10.3f}  {np.round(x_opt,3)}")

    # 4) Визуализация (с добавлением исходных параметров)
    for crit_name, _ in criteria:
        fig, axs = plt.subplots(1,3,figsize=(15,4), sharey=True)
        for idx, (axis_name, I) in enumerate(axes):
            ax = axs[idx]
            
            # Исходные параметры
            t_init, y_init, _, _, _ = initial_results[axis_name]
            ax.plot(t_init, y_init, 'r--', label='Исходные параметры')
            
            # Горячий старт
            x_start = x0_ISE[axis_name]
            cfg_hot = dict(zip(('Pp','Ip','Dp','Pv','Iv','Dv'), x_start), OGR=amp*2)
            
            # Оптимизированные
            for method in methods:
                axis_r, crit_r, method_r, x_opt, fun_opt, t_opt = run_optim_parallel(
                    (method, crit_name, objective_composite,
                    x_start.copy(), bounds.copy(),
                    (I, dt, t_fin, t_step, amp),
                    axis_name)
                )
                cfg = dict(zip(('Pp','Ip','Dp','Pv','Iv','Dv'), x_opt), OGR=amp*2)
                t_post, y_post, _ = simulate_axis_step(I, cfg, dt, t_fin, t_step, amp, torque_disturbance=0.1)
                ax.plot(t_post, y_post, label=method)
            
            ax.axvline(t_step, color='0.5', ls=':')
            ax.axhline(amp, color='0.5', ls=':')
            if idx==0: ax.set_ylabel('Угол, рад')
            ax.set_title(axis_name)
            ax.set_xlabel('Время, с')
            ax.grid(True)
            ax.legend(fontsize='small', loc='best')
        fig.suptitle(f"Сравнение переходных процессов ({crit_name} критерий)", fontsize=14)
        plt.tight_layout(rect=[0,0,1,0.95])
        plt.show()

    
    main_cfgs = {
        'Roll': {
            'Pp': 4.0, 'Ip': 0.2,  'Dp': 2.0,
            'Pv': 8.0, 'Iv': 1.1,  'Dv': 1.0,
            'OGR': np.deg2rad(20)
        },
        'Pitch': {
            'Pp': 4.0, 'Ip': 0.8,  'Dp': 2.0,
            'Pv': 8.0, 'Iv': 1.1,  'Dv': 1.0,
            'OGR': np.deg2rad(20)
        },
        'Yaw': {
            'Pp': 2.0, 'Ip': 0.02, 'Dp': 0.1,
            'Pv': 0.5, 'Iv': 0.01, 'Dv': 0.05,
            'OGR': np.deg2rad(60)
        }
    }
    axes_main = [('Roll', Ixx), ('Pitch', Iyy), ('Yaw', Izz)]

    print("\n=== Метрики качества для исходных PID из main.py ===")
    for axis_name, I in axes_main:
        cfg = main_cfgs[axis_name]
        t_pre, y_pre, _ = simulate_axis_step(I, cfg, dt, t_fin, t_step, amp)
        ISE0, OS0, Ts0 = compute_metrics(t_pre, y_pre, amp)
        print(f"{axis_name:6s} → ISE = {ISE0:.4f}, Overshoot = {OS0:.2f}%, Ts = {Ts0:.3f}s")
