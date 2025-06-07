# motors.py

class Motor:
    """
    Мотор: 1-го порядка динамика w
    w_cmd → (k/(T s + 1)) → w_actual
    F = Cf·w², M = Cm·w²
    """
    def __init__(self, k, T, Cf, Cm, w_base, w_min, w_max, dt):
        self.k, self.T, self.Cf, self.Cm = k, T, Cf, Cm
        self.w_base, self.w_min, self.w_max = w_base, w_min, w_max
        self.dt = dt
        self.w = 0.0
        self.q = 0.0

    def reset(self):
        self.w = 0.0
        self.q = 0.0

    def update(self, w_cmd):
        # суммируем с hover-компонентом и сатурируем
        tgt = w_cmd + self.w_base
        tgt = max(self.w_min, min(self.w_max, tgt))
        # 1-й порядок
        dw = (self.k*tgt - self.w)/self.T
        self.w += dw*self.dt
        self.q += self.w*self.dt
        # тяга и момент
        F = self.Cf * self.w**2
        M = self.Cm * self.w**2
        return {'w':self.w, 'q':self.q, 'F':F, 'M':M}
