import numpy as np
import matplotlib.pyplot as plt

# === ФИЗИЧЕСКИЕ КОНСТАНТЫ ===
kB = 1.38e-23  # Больцмана, Дж/К
hbar = 1.055e-34
m_Rb87 = 1.443e-25  # масса атома рубидия-87, кг
lambda_laser = 780e-9  # длина волны, м
k = 2 * np.pi / lambda_laser
gamma = 2 * np.pi * 6.07e6  # ширина перехода, рад/с
I_sat = 16.7  # насыщение, мВт/см^2

# === ПАРАМЕТРЫ ЛАЗЕРОВ И ПОЛЯ ===
s0 = 1.0
detuning = -0.5 * gamma
B_gradient = 0.01  # Тл/м
mu_eff = 9.274e-24  # эффективный магнитный момент (мю_Бора)

# === ПАРАМЕТРЫ СИМУЛЯЦИИ ===
n_atoms = 200
x0 = -0.01  # начальное положение, м
v_range = np.linspace(0.0, 1.5, n_atoms)  # начальные скорости

T_total = 1e-4  # общее время, с
dt = 1e-9  # шаг, с
n_steps = int(T_total / dt)

# === ИНИЦИАЛИЗАЦИЯ ===
positions = np.zeros((n_steps, n_atoms))
velocities = np.zeros((n_steps, n_atoms))
temperatures = np.empty(n_steps)
times = np.linspace(0, T_total, n_steps)

positions[0, :] = x0
velocities[0, :] = v_range
temperatures[0] = m_Rb87 * np.mean(velocities[0]**2) / (2 * kB)

# === ФУНКЦИИ ===
def B_field(x):
    return B_gradient * x

def zeeman_shift(x):
    return mu_eff * B_field(x) / hbar

def scattering_force(x, v):
    delta_plus = detuning - k * v + zeeman_shift(x)
    delta_minus = detuning + k * v - zeeman_shift(x)

    f_plus = hbar * k * gamma / 2 * (s0 / (1 + s0 + (2 * delta_plus / gamma)**2))
    f_minus = -hbar * k * gamma / 2 * (s0 / (1 + s0 + (2 * delta_minus / gamma)**2))

    F = f_plus + f_minus
    delta_eff = 0.5 * (np.abs(delta_plus) + np.abs(delta_minus))
    F[delta_eff > 5 * gamma] = 0
    return F

# === ОСНОВНОЙ ЦИКЛ ===
for t in range(1, n_steps):
    x = positions[t - 1]
    v = velocities[t - 1]
    a = scattering_force(x, v) / m_Rb87

    v_new = v + a * dt
    x_new = x + v * dt

    # Ограничение на размер ловушки ±0.025 м
    for i in range(n_atoms):
        if x_new[i] > 0.025:
            x_new[i] = 0.025
            v_new[i] = -v_new[i] / 2
        elif x_new[i] < -0.025:
            x_new[i] = -0.025
            v_new[i] = -v_new[i] / 2

    velocities[t] = v_new
    positions[t] = x_new
    v_cm = np.mean(v_new)
    temperatures[t] = m_Rb87 * np.mean((v_new - v_cm)**2) / (2 * kB)

    # Отладочный вывод каждые 1000 шагов
    if t % 1000 == 0:
        print(f"t = {times[t]:.6f} s | mean v = {np.mean(v):.3f} m/s | T = {temperatures[t]*1e6:.2f} µK")

# === ГРАФИК ===
plt.figure(figsize=(8, 5))
plt.plot(times[1:] * 1e3, temperatures[1:] * 1e6)
plt.xlabel("Время (мс)")
plt.ylabel("Температура (мкК)")
plt.title("Охлаждение в 1D MOT (Rb87)")
plt.grid()
plt.tight_layout()
plt.show()

np.savetxt('../results_postprocessing/mot_simulation_results/times.csv', times, delimiter=',')
np.savetxt('../results_postprocessing/mot_simulation_results/positions.csv', positions, delimiter=',')
np.savetxt('../results_postprocessing/mot_simulation_results/velocities.csv', velocities, delimiter=',')
np.savetxt('../results_postprocessing/mot_simulation_results/temperatures.csv', temperatures, delimiter=',')
