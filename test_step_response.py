from common import RESULTS_DIR, AXES, x0, amp, dt, t_fin, t_step, bounds, run_full_hybrid
from optimize_hybrid import objective_total, bounds as bnd, x0 as x0_h
from optimize_hybrid import I as I_all
from optimize import compute_metrics
from step_response import simulate_axis_step
import numpy as np
import matplotlib.pyplot as plt

def test_step_response():
    x_opt = run_full_hybrid(x0)
    for axis, inertia in AXES:
        # baseline
        xi0 = x0[6*AXES.index((axis, inertia)):6*(AXES.index((axis, inertia))+1)]
        cfg0 = dict(zip(('Pp','Ip','Dp','Pv','Iv','Dv'), xi0), OGR=amp*2)
        t0, y0, _ = simulate_axis_step(inertia, cfg0, dt, t_fin, t_step, amp)
        I0, OS0, Ts0 = compute_metrics(t0, y0, amp)
        # optimized
        x_slice = x_opt[6*AXES.index((axis, inertia)):6*(AXES.index((axis, inertia))+1)]
        cfg1 = dict(zip(('Pp','Ip','Dp','Pv','Iv','Dv'), x_slice), OGR=amp*2)
        t1, y1, _ = simulate_axis_step(inertia, cfg1, dt, t_fin, t_step, amp)
        I1, OS1, Ts1 = compute_metrics(t1, y1, amp)

        # график
        plt.figure()
        plt.plot(t0, np.rad2deg(y0), '--', label='до')
        plt.plot(t1, np.rad2deg(y1),  '-', label='после')
        plt.title(f"{axis.capitalize()} +5°")
        plt.xlabel("t, с"); plt.ylabel("°"); plt.grid(); plt.legend()
        fn = f"{RESULTS_DIR}/{axis}_step.png"
        plt.savefig(fn); plt.close()
        print("Saved", fn)
        
        print(f"ISE: до оптимизации {I0} , после оптимизации {I1}")
        print(f"OS: до оптимизации {OS0} , после оптимизации {OS1}")
        print(f"Ts: до оптимизации {Ts0} , после оптимизации {Ts1}")

        assert I1 <= I0*0.97, f"{axis}: ΔISE <3%"
        assert OS1 <= OS0*0.90, f"{axis}: OS не ↓10%"
        assert Ts1 <= Ts0*0.90, f"{axis}: Ts не ↓10%"

if __name__ == "__main__":
    test_step_response()
    print("test_step_response passed")
