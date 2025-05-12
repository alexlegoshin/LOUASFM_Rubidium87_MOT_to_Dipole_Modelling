import numpy as np
import matplotlib.pyplot as plt

# --- Константы ---
c = 3e8  # м/с
hbar = 1.055e-34  # Дж·с
k_B = 1.38e-23  # Дж/К
lambda_laser = 1012e-9  # м
omega = 2 * np.pi * c / lambda_laser
omega_0 = 2 * np.pi * 296e12
delta = omega - omega_0
Gamma = 2 * np.pi * 6e6  # рад/с
P = 2  # Вт
w_0_input = 243e-6  # м

# --- Параметры фокусировки ---
s = 1231e-3  # расстояние от источника до линзы, м
f = 50e-3  # фокусное расстояние, м
z = s - f

# ABCD-матрица
z_R = np.pi * w_0_input ** 2 / lambda_laser
M = np.array([[1, z], [0, 1]]) @ np.array([[1, 0], [-1 / f, 1]])
q_in = 1j * z_R
q_out = (M[0, 0] * q_in + M[0, 1]) / (M[1, 0] * q_in + M[1, 1])
w_0 = np.sqrt(-lambda_laser / (np.pi * np.imag(1 / q_out)))
z_R = np.pi * w_0 ** 2 / lambda_laser


# --- Потенциал ---
def I_gauss(r: float, z: float) -> float:
    w_z = w_0 * np.sqrt(1 + (z / z_R) ** 2)
    return (2 * P / (np.pi * w_z ** 2)) * np.exp(-2 * r ** 2 / w_z ** 2)


def U_gauss(x):
    r = 0  # нас интересует ось, т.е. r = 0
    z = x  # явно обозначаем, что x — это продольная координата
    I = I_gauss(r, z)
    prefactor = (3 * np.pi * c ** 2 * Gamma) / (2 * omega_0 ** 3)
    U = prefactor * (1 / delta + 1 / (omega + delta)) * I
    return U


# --- Визуализация (debug.) ---
if __name__ == "__main__":
    z_vals = np.linspace(-3e-3, 3e-3, 500)
    U_vals = [U_gauss(z) for z in z_vals]

    plt.style.use('dark_background')
    plt.figure(figsize=(8, 5))
    plt.plot(z_vals * 1e3, U_vals, color='y')
    plt.xlabel('z (мм)')
    plt.ylabel('U (Дж)')
    plt.title('Потенциал Гауссовой дипольной ловушки')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.axvline(0, color='r', linestyle='--')
    plt.tight_layout()
    plt.show()
