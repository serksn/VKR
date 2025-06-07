# main.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from flight_controller import FlightController
from motors import Motor
from kinematics import rotation_B_to_I, body_omega_to_euler_rates
from dynamics import translational_dynamics_inertial, rotational_dynamics

# 1) Настройка
dt         = 0.005
T_full     = 60.0
steps_full = int(T_full / dt)
t_full     = np.arange(steps_full) * dt

# 2) Конфиг
cfg = {
    'DT': dt,
    'MASS': 1.2, 'g': 9.81,
    'DRAG_COEFF':       {'x':1.0, 'y':1.0, 'z':0.5},
    'INERTIA':          {'Ixx':0.1,'Iyy':0.1,'Izz':0.2},
    'DRAG_MOMENT_COEFF':{'x':1000.,'y':1000.,'z':1000.},
    'WIND': np.array([1.0, 0.5, 0.0]),  # ветер в ИКС: 1 м/с по X, 0.5 м/с по Y
    'MIX_M_F': np.zeros((3,8)),
    'MIX_M_M': np.zeros((3,8)),
    'REGX':     {'Pp':5,'Ip':0,'Dp':10,'Pv':3,'Iv':0,'Dv':15,'OGRV':2,'U_LIM':2},
    'REGY':     {'Pp':5,'Ip':0,'Dp':10,'Pv':3,'Iv':0,'Dv':15,'OGRV':2,'U_LIM':2},
    'REGZ':     {'Pp':5,'Ip':0,'Dp':10,'Pv':5,'Iv':0,'Dv':20,'OGRV':2,'U_LIM':2},
    'REGPHI':   {'Pp':10,'Ip':0.01,'Dp':1,'Pv':5,'Iv':0.005,'Dv':0.5,'OGR':np.deg2rad(15),'M_LIM':100},
    'REGTHETA': {'Pp':10,'Ip':0.01,'Dp':1,'Pv':5,'Iv':0.005,'Dv':0.5,'OGR':np.deg2rad(15),'M_LIM':100},
    'REGPSI':   {'Pp':1, 'Ip':0.0, 'Dp':0.2,'Pv':0.5,'Iv':0.0,'Dv':0.1,'OGR':np.deg2rad(90),'M_LIM':100},
    #'REGPHI':   {'Pp':11.347,'Ip':0.0001,'Dp':0.0001,'Pv':22.263,'Iv':3.107,'Dv':0.0001,'OGR':np.deg2rad(15),'M_LIM':100},
    #'REGTHETA': {'Pp':23,'Ip':0.0004,'Dp':3,'Pv':20,'Iv':0.7,'Dv':0.00001,'OGR':np.deg2rad(15),'M_LIM':100},
    #'REGPSI':   {'Pp':24, 'Ip':0.001, 'Dp':3,'Pv':21,'Iv':2,'Dv':0.02,'OGR':np.deg2rad(90),'M_LIM':100},
    'TRAJ': {k: np.zeros(steps_full) for k in ('x','y','z','phi','theta','psi')},
    'MOTORS': [{
        'k':1.0,'T':0.02,
        'Cf':0.000202268,'Cm':0.001,
        'w_base':0.0,'w_min':0.0,'w_max':600.0
    }] * 8
}

# 3) w_base для hover
Cf = cfg['MOTORS'][0]['Cf']
hover_w = np.sqrt((cfg['MASS']*cfg['g']) / (8*Cf))
for m in cfg['MOTORS']:
    m['w_base'] = hover_w

# 4) «Стики» по x,y,z,φ,θ,ψ
stick = {
    'x': np.piecewise(t_full,
                      [t_full<2, (t_full>=2)&(t_full<5), t_full>=5],
                      [0, lambda t: (t-2)/3, 1.0]),
    'y': np.piecewise(t_full,
                      [t_full<8, (t_full>=8)&(t_full<11), t_full>=11],
                      [0, lambda t: 0.5*(t-8)/3, 0.5]),
    'z': np.piecewise(t_full,
                      [t_full<0.5, (t_full>=0.5)&(t_full<1.5), t_full>=1.5],
                      [0, lambda t: 1.2*(t-0.5)/1, 1.2]),
    'phi':   np.piecewise(t_full, [t_full<12, t_full>=12], [0, np.deg2rad(10)]),
    'theta': np.piecewise(t_full, [t_full<15, t_full>=15], [0, np.deg2rad(-8)]),
    'psi':   np.piecewise(t_full, [t_full<18, t_full>=18], [0, np.deg2rad(45)]),
}

# 5) Построим Γ и Γ⁺ из геометрии восьми моторов
L = 0.2
angs = np.deg2rad(np.arange(8)*45)
xy   = np.stack([L*np.cos(angs), L*np.sin(angs)], axis=1)
M_F  = np.zeros((3,8)); M_F[2,:] = 1/8
M_M  = np.zeros((3,8))
M_M[0,:] =  xy[:,1]
M_M[1,:] = -xy[:,0]
dirs     = np.array([1,-1,1,-1,1,-1,1,-1])
M_M[2,:] = dirs
Gamma    = np.vstack([M_F, M_M])
Gamma_p  = np.linalg.pinv(Gamma)

cfg['MIX_M_F']    = M_F
cfg['MIX_M_M']    = M_M
cfg['GAMMA_PLUS'] = Gamma_p

# 6) Инициализация
fc     = FlightController(cfg, dt); fc.reset()
motors = [Motor(**cfg['MOTORS'][i], dt=dt) for i in range(8)]
for m in motors: m.reset()

# 7) История
hist = {
    't':       t_full,
    'pos':     np.zeros((steps_full,3)),
    'angles':  np.zeros((steps_full,3)),
    'omega':   np.zeros((steps_full,3)),
    'w_cmds':  np.zeros((steps_full,8)),
    'w_act':   np.zeros((steps_full,8))
}

v_I        = np.zeros(3)
pos        = np.zeros(3)
angles     = np.zeros(3)
omega_body = np.zeros(3)

# 8) Цикл 6-DOF
for i in range(steps_full):
    setp = {k: stick[k][i] for k in stick}
    meas = {'x':pos[0],'y':pos[1],'z':pos[2],
            'phi':angles[0],'theta':angles[1],'psi':angles[2]}

    # body-speed
    R_ib   = rotation_B_to_I(*angles).T
    v_body = R_ib @ v_I
    body   = {'vx':v_body[0],'vy':v_body[1],'vz':v_body[2],
              'p':omega_body[0],'q':omega_body[1],'r':omega_body[2]}

    # контроллер
    w_cmds            = fc.compute_u(setp, meas, body)
    hist['w_cmds'][i] = w_cmds
    u6                = fc.last_u6

    # моторы
    Fm = np.zeros(8); Mm = np.zeros(8)
    for j,m in enumerate(motors):
        out                = m.update(w_cmds[j])
        hist['w_act'][i,j] = out['w']
        Fm[j], Mm[j]       = out['F'], out['M']

    # трансляция
    wind_I = cfg.get('WIND', np.zeros(3))
    v_I             = translational_dynamics_inertial(v_I, angles, Fm, cfg, wind_I)
    pos            += v_I * dt
    hist['pos'][i]  = pos

    # ротация
    omega_body       = rotational_dynamics(omega_body, Mm, Fm, cfg)
    hist['omega'][i] = omega_body
    rates            = body_omega_to_euler_rates(omega_body, *angles)
    angles          += rates * dt
    hist['angles'][i]= angles

# 9) График углов
fig, ax = plt.subplots(figsize=(10,4))
cols = ['#0072BD','#D95319','#77AC30']
lbls = ['roll (φ)','pitch (θ)','yaw (ψ)']
for idx, col, lbl in zip(range(3), cols, lbls):
    ax.plot(t_full, hist['angles'][:,idx], color=col, lw=2.5, label=lbl)
ax.set_xlabel('t, s'); ax.set_ylabel('rad')
ax.set_title('6-DOF Step Responses')
ax.legend(); ax.grid(True); plt.show()

# 10) Траектория 3D + XY
pos3 = hist['pos']
fig = plt.figure(figsize=(8,6))
ax3 = fig.add_subplot(111, projection='3d')
ax3.plot(pos3[:,0], pos3[:,1], pos3[:,2], lw=2)
ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
ax3.set_title('3D Trajectory'); plt.show()

plt.figure(figsize=(6,6))
plt.plot(pos3[:,0], pos3[:,1], '-o', markevery=steps_full//20)
plt.xlabel('X'); plt.ylabel('Y'); plt.title('XY Projection')
plt.grid(True); plt.axis('equal'); plt.show()
