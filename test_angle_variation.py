# optimize_hybrid_tests/test_angle_variation.py

import os
import numpy as np
import matplotlib.pyplot as plt

from common import RESULTS_DIR, AXES, x0, amp, dt, t_fin, t_step, run_full_hybrid
from step_response import simulate_axis_step
from optimize import compute_metrics

def test_angle_variation_subplots():
    """
    Строит три субплота (Roll, Pitch, Yaw), в каждом – отклик
    на скачки +5° и +15° с оптимальными PID-параметрами,
    проверяет ΔISE ≥ 2 % для каждой амплитуды.
    """
    # 1. Находим оптимальный вектор параметров единожды
    x_opt = run_full_hybrid(x0)

    # 2. Два тестовых скачка: +5° и +15°
    amps = [amp, np.deg2rad(15)]

    # 3. Подготовка фигуры с 3 субплотами
    fig, axes_plots = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    for idx, (axis_name, inertia) in enumerate(AXES):
        ax = axes_plots[idx]

        # исходные и оптимальные параметры для текущего канала
        params0    = x0    [6*idx:6*(idx+1)]
        params_opt = x_opt[6*idx:6*(idx+1)]

        for amp_i in amps:
            deg = int(np.rad2deg(amp_i))

            # baseline
            cfg0 = dict(zip(('Pp','Ip','Dp','Pv','Iv','Dv'), params0), OGR=amp_i*2)
            t0, y0, _ = simulate_axis_step(inertia, cfg0, dt, t_fin, t_step, amp_i)
            I0, OS0, Ts0 = compute_metrics(t0, y0, amp_i)

            # optimized
            cfg1 = dict(zip(('Pp','Ip','Dp','Pv','Iv','Dv'), params_opt), OGR=amp_i*2)
            t1, y1, _ = simulate_axis_step(inertia, cfg1, dt, t_fin, t_step, amp_i)
            I1, OS1, Ts1 = compute_metrics(t1, y1, amp_i)

            # график только оптимизированных траекторий для двух амплитуд
            ax.plot(t1, np.rad2deg(y1), label=f"{int(deg)}°")

            # вывод и проверка ΔISE ≥2%
            delta_pct = (I0 - I1) / I0 * 100
            print(f"{axis_name.capitalize()} {deg}°: ISE {I0:.4f} → {I1:.4f} (Δ={delta_pct:.1f}%)")
            assert delta_pct >= 2.0, f"{axis_name} при {int(deg)}°: ΔISE < 2%"

        ax.set_title(axis_name.capitalize())
        ax.set_xlabel("Время, с")
        if idx == 0:
            ax.set_ylabel("Угол, °")
        ax.grid(True)
        ax.legend(title="Амплитуда")

    fig.suptitle("Отклики оптимизированных PID при скачках +5° и +15°")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 4. Сохранение рисунка
    fn = os.path.join(RESULTS_DIR, "subplots_angle_variation.png")
    plt.savefig(fn)
    plt.close(fig)
    print(f"График с субплотами сохранён: {fn}")

if __name__ == "__main__":
    test_angle_variation_subplots()
    print("test_angle_variation_subplots passed")
