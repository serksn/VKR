# dynamics.py
import numpy as np
from kinematics import rotation_B_to_I

def translational_dynamics_inertial(v_I, angles, F_motors, cfg, wind_I=None):
    """
    v_I      — скорость центра масс в ИКС
    angles   — (phi, theta, psi)
    F_motors — массив тяги от моторов [8]
    cfg      — словарь с полями:
               MASS, g, DRAG_COEFF({'x','y','z'}), DT
    wind_I   — вектор ветра в ИКС (m/s), по умолчанию ноль
    """
    m  = cfg['MASS']
    g  = cfg['g']
    Ct = cfg['DRAG_COEFF']
    dt = cfg['DT']
    
    if wind_I is None:
        wind_I = np.zeros(3)

    # 1) тяга в B
    F_thr_B = np.array([0.0, 0.0, np.sum(F_motors)])
    R_ib    = rotation_B_to_I(*angles).T
    F_I     = R_ib @ F_thr_B

    # 2) аэродраг (квадратичный)
    v_rel = v_I - wind_I
    drag = -np.array([
        Ct['x'] * v_rel[0] * abs(v_rel[0]),
        Ct['y'] * v_rel[1] * abs(v_rel[1]),
        Ct['z'] * v_rel[2] * abs(v_rel[2]),
    ])

    # 3) гравитация
    grav = np.array([0.0, 0.0, -m*g])

    # 4) ускорение + интеграция
    a_I     = (F_I + drag + grav) / m
    v_rel    += a_I * dt         # semi-implicit Euler
    v_next  = v_rel

    # 5) защита от Inf/Nan
    if not np.isfinite(v_next).all():
        v_next = np.zeros(3)

    return v_next

def rotational_dynamics(omega_body, M_motors, F_motors, cfg):
    dt = cfg['DT']
    Ixx,Iyy,Izz = cfg['INERTIA']['Ixx'], cfg['INERTIA']['Iyy'], cfg['INERTIA']['Izz']
    Cq   = cfg['DRAG_MOMENT_COEFF']
    ΓM   = cfg['MIX_M_M']
    ΓF   = cfg['MIX_M_F']

    # моменты
    Mt    = ΓM.dot(M_motors)
    Md_F  = ΓF.dot(F_motors)
    dragM = -np.array([
        Cq['x'] * omega_body[0]*abs(omega_body[0]),
        Cq['y'] * omega_body[1]*abs(omega_body[1]),
        Cq['z'] * omega_body[2]*abs(omega_body[2]),
    ])
    Mtot  = Mt + Md_F + dragM

    α = np.array([Mtot[0]/Ixx, Mtot[1]/Iyy, Mtot[2]/Izz])
    omega_body += α * dt       # semi-implicit
    ω_next = omega_body

    # clamp экстремальных угловых скоростей
    ω_next = np.clip(ω_next, -50.0, +50.0)
    if not np.isfinite(ω_next).all():
        ω_next = np.zeros(3)

    return ω_next
