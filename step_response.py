# step_response.py

import numpy as np
import matplotlib.pyplot as plt

from pid import PIDController

def simulate_axis_step(
    I: float,
    cfg: dict,
    dt: float,
    t_final: float,
    t_step: float,
    amp: float,
    torque_disturbance: float = 0.0
):
    """
    Симуляция каскадного PID: угол→угловая скорость→момент
      I      — момент инерции
      cfg    — словарь с ключами Pp, Ip, Dp, Pv, Iv, Dv, OGR
      dt     — шаг интегрирования
      t_final— общая длительность моделирования
      t_step — время скачка
      amp    — амплитуда скачка (в радианах)
    """
    # outer PID (угол→скорость)
    pid_p = PIDController(
        kp=cfg['Pp'], ki=cfg['Ip'], kd=cfg['Dp'],
        dt=dt, u_min=-cfg['OGR'], u_max=+cfg['OGR']
    )
    # inner PID (скорость→момент)
    pid_v = PIDController(
        kp=cfg['Pv'], ki=cfg['Iv'], kd=cfg['Dv'],
        dt=dt, u_min=-cfg['OGR'], u_max=+cfg['OGR']
    )

    t = np.arange(0, t_final, dt)
    phi = np.zeros_like(t)
    rate= np.zeros_like(t)

    angle = 0.0
    omega = 0.0

    for i, ti in enumerate(t):
        # задаём «скачок»
        setpoint = 0.0 if ti < t_step else amp

        # внешний регулятор: скорость
        rate_cmd = pid_p.update(setpoint - angle)

        # внутренний регулятор: момент (u)
        torque   = pid_v.update(rate_cmd - omega) + torque_disturbance

        # динамика: I*domega/dt = torque → интегрируем
        domega   = torque / I
        omega   += domega * dt
        angle   += omega * dt

        rate[i]  = omega
        phi[i]   = angle
    

    return t, phi, rate


if __name__ == '__main__':
    # параметры
    dt      = 0.0005
    t_final = 60.0
    t_step  = 1.0
    amp     = np.deg2rad(10)  # 10° скачок

    # моменты инерции
    Ixx, Iyy, Izz = 0.1, 0.1, 0.2
    
    # ROLL (φ)
    cfg_phi = {
        'Pp': 4.0,     # чуть поменьше «P»
        'Ip': 0.2,    # небольшой интеграл, чтобы убрать steady-state error
        'Dp': 2.0,     # чуть больше демпфирования
        'Pv': 8.0,     # inner-loop чуть помягче
        'Iv': 1.1,     # небольшой интеграл
        'Dv': 1.0,     # чуть больше демпфа
        'OGR': np.deg2rad(20)
    }

    # PITCH (θ)
    cfg_theta = {
        'Pp': 4.0,     # чуть поменьше «P»
        'Ip': 0.8,    # небольшой интеграл, чтобы убрать steady-state error
        'Dp': 2.0,     # чуть больше демпфирования
        'Pv': 8.0,     # inner-loop чуть помягче
        'Iv': 1.1,     # небольшой интеграл
        'Dv': 1.0,     # чуть больше демпфа
        'OGR': np.deg2rad(20)
    }

    # YAW (ψ) — оставляем прежним, он уже устраивает
    cfg_psi = {
        'Pp': 2.0,
        'Ip': 0.02,
        'Dp': 0.1,
        'Pv': 0.5,
        'Iv': 0.01,
        'Dv': 0.05,
        'OGR': np.deg2rad(60)
    }

    # симуляция для каждого канала
    tφ, φ, pφ = simulate_axis_step(Ixx, cfg_phi,   dt, t_final, t_step, amp)
    tθ, θ, pθ = simulate_axis_step(Iyy, cfg_theta, dt, t_final, t_step, amp)
    tψ, ψ, pψ = simulate_axis_step(Izz, cfg_psi,   dt, t_final, t_step, amp)


    # отрисовка φ
    plt.figure(figsize=(6,4))
    plt.plot(tφ, φ,  lw=2, label='φ (roll)')
    plt.axvline(t_step, color='k', ls='--', label='step at t=1s')
    plt.axhline(amp,  color='k', ls='--')
    plt.title("Step Response: Roll (φ)")
    plt.xlabel("Time, s")
    plt.ylabel("Angle, rad")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # отрисовка θ
    plt.figure(figsize=(6,4))
    plt.plot(tθ, θ,  lw=2, label='θ (pitch)', color='C1')
    plt.axvline(t_step, color='k', ls='--')
    plt.axhline(amp,  color='k', ls='--')
    plt.title("Step Response: Pitch (θ)")
    plt.xlabel("Time, s")
    plt.ylabel("Angle, rad")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # отрисовка ψ
    plt.figure(figsize=(6,4))
    plt.plot(tψ, ψ,  lw=2, label='ψ (yaw)', color='C2')
    plt.axvline(t_step, color='k', ls='--')
    plt.axhline(amp,  color='k', ls='--')
    plt.title("Step Response: Yaw (ψ)")
    plt.xlabel("Time, s")
    plt.ylabel("Angle, rad")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
