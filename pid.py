# pid.py
import numpy as np

class PIDController:
    """
    Общий PID с анти-винд-апом и защитой от NaN/overflow.
    """
    def __init__(self, kp, ki, kd, dt, u_min=None, u_max=None):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.dt = dt
        self.u_min, self.u_max = u_min, u_max
        self._integral = 0.0
        self._prev_err = 0.0

    def reset(self):
        self._integral = 0.0
        self._prev_err = 0.0

    def update(self, error):
        # P-term
        P = self.kp * error

        # I-term с анти-виндапом (back-calculation)
        self._integral += error * self.dt
        I = self.ki * self._integral
        if self.u_max is not None:
            u_unsat = P + I + self.kd*(error-self._prev_err)/self.dt
            if u_unsat > self.u_max or u_unsat < self.u_min:
                # отменяем последний шаг интеграции
                self._integral -= error * self.dt
                I = self.ki * self._integral

        # D-term с простым фильтром
        rawD = (error - self._prev_err) / self.dt
        # α = dt/(Tf+dt), Tf≈0.01…0.02
        alpha = self.dt / (0.02 + self.dt)
        self._d_filt = alpha*rawD + (1-alpha)*getattr(self, '_d_filt', 0.0)
        D = self.kd * self._d_filt

        # суммарный выход + saturate
        u = P + I + D
        if self.u_max is not None and u > self.u_max: u = self.u_max
        if self.u_min is not None and u < self.u_min: u = self.u_min

        # защита от NaN
        if not np.isfinite(u):
            self.reset()
            u = 0.0

        self._prev_err = error
        return u


class AxisRegulator:
    def __init__(self, cfg, dt):
        self.pid = PIDController(
            kp=cfg['P'], ki=cfg['I'], kd=cfg['D'],
            dt=dt, u_min=-cfg['OGRV'], u_max=cfg['U_LIM']
        )
    def reset(self):
        self.pid.reset()
    def compute(self, setpt, meas):
        return self.pid.update(setpt - meas)


class AngleRegulator:
    def __init__(self, cfg, dt):
        self.pid = PIDController(
            kp=cfg['P'], ki=cfg['I'], kd=cfg['D'],
            dt=dt, u_min=-cfg['OGR'], u_max=cfg['M_LIM']
        )
    def reset(self):
        self.pid.reset()
    def compute(self, setpt, meas):
        return self.pid.update(setpt - meas)
