import os
import numpy as np
import matplotlib.pyplot as plt
import optimize_hybrid
from optimize_hybrid import objective_total, x0, bounds, I, dt, t_fin, t_step, amp
from optimize import compute_metrics
from step_response import simulate_axis_step

# Папка для результатов (единожды)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Параметры алгоритма (копируются из optimize_hybrid.py)
NM_PARAMS = dict(alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5,
                 max_iter=100, tol=1e-2, restarts=10, jitter=0.5)
SD_PARAMS = dict(alpha0=1.0, beta=0.9, c1=1e-4, rho=0.5,
                 max_iter=50, tol_grad=1e-4)
BFGS_PARAMS = dict(max_iter=20, tol_grad=1e-5,
                   c1=1e-4, rho=0.5, eps_grad=1e-6, gamma=1e-3)

def run_full_hybrid(x_initial):
    """Три этапа: NM → Steepest Descent → BFGS"""
    mod = optimize_hybrid.optimize
    r1 = mod.nelder_mead_manual(objective_total, x_initial, bounds, (), **NM_PARAMS)
    r2 = mod.steepest_descent (objective_total, r1.x,      bounds, (), **SD_PARAMS)
    r3 = mod.bfgs_manual      (objective_total, r2.x,      bounds, (), **BFGS_PARAMS)
    return r3.x

# Угловые каналы
AXES = [('roll',  I['roll']),
        ('pitch', I['pitch']),
        ('yaw',   I['yaw'])]
