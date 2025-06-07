# flight_controller.py

import numpy as np
from pid import AxisRegulator, AngleRegulator

class FlightController:
    """
    Каскадный контроллер для 6 DOF:
      – По X/Y/Z: позиция→скорость→u_trans
      – По φ/θ/ψ: угол→угловая скорость→момент
    Выход: 6-мерный вектор u = [ux,uy,uz,uφ,uθ,uψ], который микшируется в 8 моторов.
    """

    def __init__(self, cfg: dict, dt: float):
        # --- Линейные каскады X, Y, Z ---
        rx, ry, rz = cfg['REGX'], cfg['REGY'], cfg['REGZ']

        # X
        self.reg_x_p = AxisRegulator({
            'P':     rx['Pp'],
            'I':     rx['Ip'],
            'D':     rx['Dp'],
            'OGRV':  rx['OGRV'],
            'U_LIM': rx['U_LIM']
        }, dt)
        self.reg_x_v = AxisRegulator({
            'P':     rx['Pv'],
            'I':     rx['Iv'],
            'D':     rx['Dv'],
            'OGRV':  rx['OGRV'],
            'U_LIM': rx['U_LIM']
        }, dt)

        # Y
        self.reg_y_p = AxisRegulator({
            'P':     ry['Pp'],
            'I':     ry['Ip'],
            'D':     ry['Dp'],
            'OGRV':  ry['OGRV'],
            'U_LIM': ry['U_LIM']
        }, dt)
        self.reg_y_v = AxisRegulator({
            'P':     ry['Pv'],
            'I':     ry['Iv'],
            'D':     ry['Dv'],
            'OGRV':  ry['OGRV'],
            'U_LIM': ry['U_LIM']
        }, dt)

        # Z
        self.reg_z_p = AxisRegulator({
            'P':     rz['Pp'],
            'I':     rz['Ip'],
            'D':     rz['Dp'],
            'OGRV':  rz['OGRV'],
            'U_LIM': rz['U_LIM']
        }, dt)
        self.reg_z_v = AxisRegulator({
            'P':     rz['Pv'],
            'I':     rz['Iv'],
            'D':     rz['Dv'],
            'OGRV':  rz['OGRV'],
            'U_LIM': rz['U_LIM']
        }, dt)

        # --- Угловые каскады φ, θ, ψ ---
        rφ = cfg['REGPHI']
        self.reg_phi_p   = AngleRegulator({
            'P':   rφ['Pp'],
            'I':   rφ['Ip'],
            'D':   rφ['Dp'],
            'OGR': rφ['OGR'],
            'M_LIM': rφ['M_LIM']
        }, dt)
        self.reg_phi_v   = AngleRegulator({
            'P':   rφ['Pv'],
            'I':   rφ['Iv'],
            'D':   rφ['Dv'],
            'OGR': rφ['OGR'],
            'M_LIM': rφ['M_LIM']
        }, dt)

        rθ = cfg['REGTHETA']
        self.reg_theta_p = AngleRegulator({
            'P':   rθ['Pp'],
            'I':   rθ['Ip'],
            'D':   rθ['Dp'],
            'OGR': rθ['OGR'],
            'M_LIM': rθ['M_LIM']
        }, dt)
        self.reg_theta_v = AngleRegulator({
            'P':   rθ['Pv'],
            'I':   rθ['Iv'],
            'D':   rθ['Dv'],
            'OGR': rθ['OGR'],
            'M_LIM': rθ['M_LIM']
        }, dt)

        rψ = cfg['REGPSI']
        self.reg_psi_p   = AngleRegulator({
            'P':   rψ['Pp'],
            'I':   rψ['Ip'],
            'D':   rψ['Dp'],
            'OGR': rψ['OGR'],
            'M_LIM': rψ['M_LIM']
        }, dt)
        self.reg_psi_v   = AngleRegulator({
            'P':   rψ['Pv'],
            'I':   rψ['Iv'],
            'D':   rψ['Dv'],
            'OGR': rψ['OGR'],
            'M_LIM': rψ['M_LIM']
        }, dt)

        # 8×6 микшер (псевдо-обратный Γ из cfg)
        self.GammaPlus = cfg['GAMMA_PLUS']


    def reset(self):
        """Сбросить внутренние состояния всех PID-регуляторов."""
        for reg in (
            self.reg_x_p, self.reg_x_v,
            self.reg_y_p, self.reg_y_v,
            self.reg_z_p, self.reg_z_v,
            self.reg_phi_p, self.reg_phi_v,
            self.reg_theta_p, self.reg_theta_v,
            self.reg_psi_p, self.reg_psi_v
        ):
            reg.reset()


    def compute_u(self, setp: dict, meas: dict, body: dict) -> np.ndarray:
        """
        Вычислить команды в моторы:
          setp: {'x','y','z','phi','theta','psi'}
          meas: текущее положение/углы
          body: {'vx','vy','vz','p','q','r'} — скорости в теле
        Возвращает 8-мерный вектор ω_cmds.
        """

        # -- Линейные каналы --
        vx_set = self.reg_x_p.compute(setp['x'], meas['x'])
        ux     = self.reg_x_v.compute(vx_set,     body['vx'])

        vy_set = self.reg_y_p.compute(setp['y'], meas['y'])
        uy     = self.reg_y_v.compute(vy_set,     body['vy'])

        vz_set = self.reg_z_p.compute(setp['z'], meas['z'])
        uz     = self.reg_z_v.compute(vz_set,     body['vz'])

        # -- Угловые каналы --
        p_set  = self.reg_phi_p.compute(  setp['phi'],   meas['phi'])
        uφ     = self.reg_phi_v.compute(  p_set,         body['p'])

        q_set  = self.reg_theta_p.compute(setp['theta'], meas['theta'])
        uθ     = self.reg_theta_v.compute(q_set,         body['q'])

        r_set  = self.reg_psi_p.compute(  setp['psi'],   meas['psi'])
        uψ     = self.reg_psi_v.compute(  r_set,         body['r'])

        # Составляем общий вектор управления
        u6 = np.array([ux, uy, uz, uφ, uθ, uψ])
        self.last_u6 = u6.copy()

        # Микшируем в 8 моторов
        return self.GammaPlus.dot(u6)
