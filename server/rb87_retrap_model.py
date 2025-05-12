import numpy as np
import matplotlib.pyplot as plt
from rb87_gauss_potential_well_model import U_gauss
from rb87_lattice_potential_well_model import U_lattice

# Параметры атома и ловушки
mass_Rb = 87 * 1.66e-27  # кг
trap_depth = 1e-20  # Дж
trap_radius = 10e-6  # м

# Параметры симуляции
dt = 1e-5  # шаг по времени
time_max = 1e-3  # общее время симуляции (сек)
steps = int(time_max / dt)


# Упрощённая модель потенциала
def U_simple(x):
    return np.where(np.abs(x) < trap_radius, trap_depth, 0)


# Вероятность захвата (упрощённая)
def trapping_probability(x, v, potential_fn):
    kinetic_energy = 0.5 * mass_Rb * v ** 2
    potential_energy = potential_fn(x)
    total_energy = kinetic_energy + potential_energy
    if abs(x) > trap_radius:
        return False  # атом ушел
    return True  # атом остаётся в ловушке


# Моделируем движение атомов
def simulate_retrap(positions, velocities, trap_radius, trap_depth):
    n_atoms = len(positions)
    velocities_over_time = []
    trapped_flags_over_time = []
    trapped_flags = np.ones(n_atoms, dtype=bool)

    time_points = np.linspace(0, time_max, steps)
    for t in time_points:
        new_velocities = []
        for i in range(n_atoms):
            if trapped_flags[i]:
                F = (U_lattice(positions[i] + 1e-5) - U_lattice(positions[i] - 1e-5)) / (2e-5)  # сила
                if i == 0:
                    print(f"t = {t:.6e} s | x = {positions[i]:.3e} m | F = {F:.3e} N | v = {velocities[i]:.3e} m/s")
                velocities[i] += (F / mass_Rb) * dt
                positions[i] += velocities[i] * dt
            new_velocities.append(velocities[i])
        velocities_over_time.append(new_velocities)
        trapped_flags_over_time.append(trapped_flags.copy())

    # Рассчитываем температуру
    temperatures = []
    for step_velocities in velocities_over_time:
        T_classical = np.mean(0.5 * mass_Rb * np.array(step_velocities) ** 2) / (0.5 * 1.38e-23)
        temperatures.append(T_classical)

    return positions, velocities, temperatures


# Визуализация результатов
def plot_results(time_points, temperatures):
    plt.plot(time_points, temperatures)
    plt.xlabel('Время (с)')
    plt.ylabel('Температура (K)')
    plt.title('Температура атомов в ловушке')
    plt.show()


# Запуск симуляции
times = np.loadtxt('../results_postprocessing/mot_simulation_results/times.csv', delimiter=',')
positions = np.loadtxt('../results_postprocessing/mot_simulation_results/positions.csv', delimiter=',')
velocities = np.loadtxt('../results_postprocessing/mot_simulation_results/velocities.csv', delimiter=',')
temperatures = np.loadtxt('../results_postprocessing/mot_simulation_results/temperatures.csv', delimiter=',')
print('Начальная позиция нескольких атомов (м) ', positions[-1][100:109])
print('Начальная скорость нескольких атомов (м/с) ', velocities[-1][100:109])

time_points = np.linspace(0, time_max, steps)
positions, velocities, temperatures = simulate_retrap(positions[-1], velocities[-1], trap_radius, trap_depth)
temperatures = temperatures[:len(time_points)]

# Построение графиков
plot_results(time_points, temperatures)

print('Массив температур (по таймсепу) ', temperatures)
