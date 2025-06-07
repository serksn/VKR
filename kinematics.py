# kinematics.py
import numpy as np

def rotation_B_to_I(phi, theta, psi):
    cφ, sφ = np.cos(phi), np.sin(phi)
    cθ, sθ = np.cos(theta), np.sin(theta)
    cψ, sψ = np.cos(psi),   np.sin(psi)
    Rz = np.array([[ cψ, -sψ, 0],
                   [ sψ,  cψ, 0],
                   [  0,   0,  1]])
    Ry = np.array([[ cθ, 0, sθ],
                   [  0, 1,  0],
                   [-sθ, 0, cθ]])
    Rx = np.array([[1,   0,   0],
                   [0,  cφ, -sφ],
                   [0,  sφ,  cφ]])
    return Rz @ Ry @ Rx

def body_omega_to_euler_rates(omega_b, phi, theta, psi):
    p, q, r = omega_b
    # ограничим theta, чтобы не было tan(±90°) → Inf
    θ = np.clip(theta, -1.56, +1.56)
    tanθ = np.tan(θ)
    secθ = 1.0 / (np.cos(θ) + 1e-6)
    phi_dot   = p + q*np.sin(phi)*tanθ + r*np.cos(phi)*tanθ
    theta_dot =     q*np.cos(phi)         - r*np.sin(phi)
    psi_dot   =     q*np.sin(phi)*secθ   + r*np.cos(phi)*secθ
    return np.array([phi_dot, theta_dot, psi_dot])
