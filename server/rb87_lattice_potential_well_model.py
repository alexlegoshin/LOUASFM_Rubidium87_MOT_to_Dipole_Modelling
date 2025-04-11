import numpy as np
import matplotlib.pyplot as plt

# --- Константы ---
c = 3e8  # м/с
hbar = 1.0545718e-34  # Дж·с
k_B = 1.38e-23  # Дж/К
lambda_laser = 1012e-9  # м
omega_0 = 2 * np.pi * 296e12
delta = 2 * np.pi * 88e12  # рад/с
Gamma = 2 * np.pi * 3e6  # рад/с
P = 2  # Вт
w_0 = 243e-6  # м
k_laser = 2 * np.pi / lambda_laser


# --- Потенциал ---
def I_lattice(x: float) -> float:
    envelope = np.exp(-2 * x ** 2 / w_0 ** 2)
    standing_wave = np.cos(k_laser * x) ** 2
    return (2 * P / (np.pi * w_0 ** 2)) * envelope * standing_wave


def U_lattice(x: float) -> float:
    I = I_lattice(x)
    prefactor = (3 * np.pi * c ** 2 * Gamma) / (2 * omega_0 ** 3)
    U = prefactor * (1 / delta - 1 / (2 * omega_0 + delta)) * I
    return U / (k_B * 1e-6)  # мкК


# --- Визуализация (debug.) ---
if __name__ == "__main__":
    x_vals = np.linspace(-5e-3, 5e-3, 2000)
    U_vals = [U_lattice(x) for x in x_vals]

    plt.style.use("dark_background")
    plt.figure(figsize=(9, 4))
    plt.plot(x_vals * 1e3, U_vals, color='cyan', label='Оптическая решётка')
    plt.xlabel("x (мм)")
    plt.ylabel("U (мкК)")
    plt.title("Потенциал дипольной оптической решётки")
    plt.axhline(0, color='gray', lw=0.5, ls="--")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
