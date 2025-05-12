import numpy as np
import matplotlib.pyplot as plt
from MOTorNOT.integration import Solver
from MOTorNOT.coils import QuadrupoleCoils
from MOTorNOT.beams import UniformBeam
from MOTorNOT.mot import MOT
from MOTorNOT.integration import generate_initial_conditions

# Физические константы
hbar = 1.055e-34  # постоянная Планка (Дж*с)
kB = 1.38e-23  # постоянная Больцмана

# Физические параметры
atomic_mass = 1.443e-25  # кг (масса рубидия-87 кг)
atomic_mass_aem = 87 # масса в а.е.м.
wavelength = 780e-9  # длина волны лазера (м)
k = 2 * np.pi / wavelength
gamma = 2 * np.pi * 6.07e6  # естественная ширина перехода (рад/с)
B_gradient = 10  # Тл/м (градиент магнитного поля)
mu_eff = 9.274e-24  # эффективный магнитный момент (мю_Бора)
detuning = -gamma  # детюнинг (рад/с)
s0 = 0.5  # интенсивность в единицах насыщения
I_sat = 16.7 # Порог насыщения мВт/см^2
lande_factor = 1 # фактор Ланде

# Параметры симуляции
n_atoms = 200  # число атомов
x_start = -0.01  # стартовая позиция в м
v_range = np.linspace(0.1, 5, n_atoms)  # начальные скорости, м/с
t_max = 5e-5  # период симуляции, с
dt = 5e-9  # шаг симуляции, с

def B_field(x):
    return B_gradient * x

def zeeman_shift(x):
    return mu_eff * B_field(x) / hbar

def scattering_force(x, v):
    delta_plus = detuning - k * v + zeeman_shift(x)
    delta_minus = detuning + k * v - zeeman_shift(x)

    f_plus = hbar * k * gamma / 2 * (s0 / (1 + s0 + (2 * delta_plus / gamma)**2))
    f_minus = -hbar * k * gamma / 2 * (s0 / (1 + s0 + (2 * delta_minus / gamma)**2))

    return f_plus + f_minus

def acceleration(xv, t):
    x, v = xv
    a = scattering_force(x, v) / atomic_mass
    return np.array([v, a])

def initialize_mot():
    # Лучи и катушки
    beam1 = UniformBeam(direction=[-1, 0, 0], power=10e-3, radius=5e-3, detuning=-4 * gamma, handedness=1)
    beam2 = UniformBeam(direction=[1, 0, 0], power=10e-3, radius=5e-3, detuning=-4 * gamma, handedness=1)
    coils = QuadrupoleCoils(radius=0.1, offset=0.1, turns=50, current=50, axis=0)  # поле вдоль x

    # Добавляем ширину естественного резонанса (Гц)
    mot = MOT([beam1, beam2], coils.field)
    mot.atom['gamma'] = gamma  # Ширина естественного резонанса (Гц)
    mot.atom['Isat'] = I_sat  # мВт/см^2
    mot.atom['mass'] = atomic_mass_aem    # атомная масса в а.е.м.
    mot.atom['wavelength'] = wavelength  # Длина волны лазера для рубидия-87 (м)
    mot.atom['gF'] = lande_factor  # Фактор Ланде для рубидия-87

    return mot

def simulate_mot():
    X0, V0 = generate_initial_conditions(x_start, v_range, phi=90, theta=90)
    mot = initialize_mot()

    solver = Solver(mot.acceleration, X0, V0)
    sol = solver.run(t_max, dt)

    times = np.arange(0, t_max, dt)
    positions = sol.X[:, :, 0]  # (time, atom)
    velocities = sol.V[:, :, 0]  # (time, atom)

    temperatures = []
    for t in range(velocities.shape[0]):
        v_squared = velocities[t] ** 2  # вектор: v_i² всех атомов в момент времени t
        avg_v_squared = np.mean(v_squared)  # среднее по атомам
        T = (atomic_mass / (2 * kB)) * avg_v_squared
        temperatures.append(T)
        print(T)
    temperatures = np.array(temperatures)

    avg_level_populations = np.zeros_like(times)  # заглушка
    avg_velocity_distributions = np.mean(np.abs(velocities), axis=1)

    return times, positions, velocities, temperatures, avg_level_populations, avg_velocity_distributions

def plot_results(times, positions, velocities, temperatures, avg_level_populations, avg_velocity_distributions):
    plt.figure(figsize=(8, 6))
    plt.plot(times, temperatures)
    plt.title('Температура ловушки (K)')
    plt.xlabel('Время (с)')
    plt.ylabel('Температура (K)')
    plt.tight_layout()
    plt.show()


# Тестовый код для проверки поведения симуляции
if __name__ == "__main__":
    results = simulate_mot()
    plot_results(*results)
