import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import k, hbar, pi
from rb87_gauss_potential_well_model import U_gauss
from rb87_lattice_potential_well_model import U_lattice

import os
import sys

# Параметры атома
mass_Rb = 87 * 1.66e-27  # кг

# Параметры ловушки
trap_depth = 1e-27  # Дж, глубина ловушки (~70 мкК)
trap_radius = 10e-6  # м, характерный радиус ловушки

# Параметры симуляции
dt = 1e-4  # шаг по времени
time_max = 1  # общее время симуляции (сек)
steps = int(time_max / dt)


# --- Решение уравнения Шрёдингера ---
def solve_schrodinger(potential_fn, mass, x_grid):
    dx = x_grid[1] - x_grid[0]
    N = len(x_grid)

    T = -0.5 * (hbar ** 2 / mass) * (
            np.diag(np.ones(N - 1), 1)
            - 2 * np.diag(np.ones(N), 0)
            + np.diag(np.ones(N - 1), -1)
    ) / dx ** 2

    V = np.diag([potential_fn(x) for x in x_grid])
    H = T + V

    energies, wavefuncs = np.linalg.eigh(H)
    return energies, wavefuncs


# --- Вероятность захвата ---
def trapping_probability(x0, v0, potential_fn, energies, wavefuncs, x_grid, trap_radius):
    """
    Функция для вычисления вероятности того, что атом останется в ловушке или выйдет из неё
    с учётом потенциальных барьеров и возможности туннелирования для разных типов потенциала.
    """

    # Кинетическая энергия атома
    kinetic_energy = 0.5 * mass_Rb * v0 ** 2
    potential_energy = potential_fn(x0)
    total_energy = kinetic_energy + potential_energy

    # Проверка на выход атома за пределы ловушки
    if abs(x0) > trap_radius:  # если атом вышел за пределы ловушки по координате x
        return True  # В данном случае считаем, что атом всё равно может вернуться

    # Для решётки или другого потенциала нужно учитывать барьерную ширину
    # Ожидаем, что в случае с решёткой потенциал имеет периодический вид.
    if np.any(x_grid):  # если есть сетка, мы будем искать барьерную ширину
        min_potential = np.min(potential_fn(x_grid))  # минимальное значение потенциала
        max_potential = np.max(potential_fn(x_grid))  # максимальное значение потенциала

        # Барьерная ширина может быть рассчитана как разница между максимумами потенциала
        barrier_width = np.abs(x_grid[np.argmax(potential_fn(x_grid))] - x_grid[np.argmin(potential_fn(x_grid))])

        if barrier_width > 0:  # если барьер существует
            tunneling_prob = np.exp(-barrier_width / (hbar * 1e-9))  # простая модель туннелирования
            if np.random.rand() < tunneling_prob:  # вероятность туннелирования атома
                return True  # если сработала вероятность туннелирования, атом выходит из ловушки
    return False  # иначе атом остаётся в ловушке


# --- Температура классическая и квантовая ---
def compute_temperatures_over_time(velocities_over_time, energies, wavefuncs, x_grid, trapped_flags_over_time):
    T_classical_list = []
    T_quantum_list = []

    for step_velocities, step_trapped in zip(velocities_over_time, trapped_flags_over_time):
        trapped_velocities = step_velocities[step_trapped]
        if len(trapped_velocities) == 0:
            T_classical = 0.0
            T_quantum = 0.0
        else:
            T_classical = np.mean(0.5 * mass_Rb * trapped_velocities ** 2) / (0.5 * k)
            step_energies = energies[energies < 0]
            if len(step_energies) == 0:
                T_quantum = 0.0
            else:
                pop = np.exp(-step_energies / (k * T_classical))
                pop /= np.sum(pop)
                avg_energy = np.sum(step_energies * pop)
                T_quantum = avg_energy / k if avg_energy > 0 else 0.0

        T_classical_list.append(T_classical)
        T_quantum_list.append(T_quantum)

    return T_classical_list, T_quantum_list


# --- Основная симуляция ---
def simulate_retrap(positions, velocities, potential_type, trap_radius, trap_depth):
    assert len(positions) == len(velocities)
    n_atoms = len(positions)

    if potential_type == 'gaussian':
        potential_fn = U_gauss
    elif potential_type == 'lattice':
        potential_fn = U_lattice
    else:
        raise NotImplementedError("Поддерживаются только 'gaussian' и 'lattice'")

    x_grid = np.linspace(-trap_radius * 3, trap_radius * 3, 1000)
    energies, wavefuncs = solve_schrodinger(potential_fn, mass_Rb, x_grid)

    velocities_over_time = []
    trapped_flags_over_time = []
    trapped_flags = np.ones(n_atoms, dtype=bool)
    time_points = np.arange(0, time_max, dt)

    for i in range(n_atoms):
        if i % (n_atoms // 100 or 1) == 0:
            percent = 100 * i / n_atoms
            sys.stdout.write(f"\rПрогресс (перезахват в дипольную): {percent:.1f}%")
            sys.stdout.flush()

        x = positions[i]
        v = velocities[i]

        for t in time_points:
            # Вычисляем потенциальную силу
            F = - (potential_fn(x + 1e-5) - potential_fn(x - 1e-5)) / (2e-5)

            # Обновляем скорость и позицию
            v += (F / mass_Rb) * dt
            x += v * dt

            if i == 0:
                velocities_over_time.append(np.zeros(n_atoms))
                trapped_flags_over_time.append(np.ones(n_atoms, dtype=bool))
            velocities_over_time[-1][i] = v
            trapped_flags_over_time[-1][i] = trapped_flags[i]

            # Проверяем, остался ли атом в ловушке
            if abs(x) > trap_radius or potential_fn(x) > trap_depth:
                prob = trapping_probability(x, v, potential_fn, energies, wavefuncs, x_grid, trap_radius)
                if np.random.rand() > prob:  # если не туннелирует
                    trapped_flags[i] = False
                    # Попробуем туннелировать обратно, если атом покинул ловушку
                    tunneling_prob = np.exp(-np.abs(potential_fn(x) - trap_depth) / (hbar * 1e-9))  # упрощенная модель
                    if np.random.rand() < tunneling_prob:  # вероятность туннелирования
                        trapped_flags[i] = True
                        x = np.random.uniform(-trap_radius, trap_radius)  # возвращаем атом обратно в ловушку
                        v = np.random.normal(0, np.sqrt(k * 300e-6 / mass_Rb))  # охладим атом
                        break

    print(f"\rПрогресс (перезахват в дипольную): 100%, моделирование завершено")
    print(
        f"Перезахвачено в дипольную ловушку: {np.sum(trapped_flags)} из {n_atoms} атомов ({100 * np.sum(trapped_flags) / n_atoms:.1f}%)")
    with open('../results_postprocessing/dipole_simulation_results/retrapped.txt', 'w') as f:
        f.write(f"{100 * np.sum(trapped_flags) / n_atoms:.3f}")
    T_classical, T_quantum = compute_temperatures_over_time(velocities_over_time, energies, wavefuncs, x_grid,
                                                            trapped_flags_over_time)
    return trapped_flags, positions, velocities, T_classical, T_quantum


# --- Визуализация температур ---
def plot_results(time_points, T_classical_list, T_quantum_list):
    results_dir = "../results_postprocessing/dipole_simulation_results"
    os.makedirs(results_dir, exist_ok=True)
    plt.style.use('dark_background')

    # Убедимся, что длины совпадают
    min_len = min(len(time_points), len(T_classical_list), len(T_quantum_list))
    time_points = time_points[:min_len]
    T_classical_list = T_classical_list[:min_len]
    T_quantum_list = T_quantum_list[:min_len]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_points, T_classical_list, label="Классическая температура", color='cyan', linewidth=1.5)
    ax.plot(time_points, T_quantum_list, label="Квантовая температура", color='magenta', linewidth=1.5)
    ax.set_xlabel("Время (с)")
    ax.set_ylabel("Температура (K)")
    ax.set_title("Температура атомов в дипольной ловушке во времени")
    ax.legend()
    ax.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/temperature_evolution.png", dpi=300)

    np.savetxt(f"{results_dir}/T_classical_list.txt", T_classical_list)
    np.savetxt(f"{results_dir}/T_quantum_list.txt", T_quantum_list)
    np.savetxt(f"{results_dir}/time_points.txt", time_points)

    plt.close()


# --- Test (debug.) ---
if __name__ == "__main__":
    time_points = []
    T_classical_list = []
    T_quantum_list = []

    for t in np.linspace(0, time_max, 100):
        positions = np.random.normal(0, trap_radius * 2, 1000)
        velocities = np.random.normal(0, np.sqrt(k * 300e-6 / mass_Rb), 1000)

        result, _, _, T_cl, T_q = simulate_retrap(
            positions, velocities,
            potential_type='gaussian',
            trap_radius=trap_radius,
            trap_depth=trap_depth
        )

        time_points.append(t)
        T_classical_list.append(T_cl)
        T_quantum_list.append(T_q)

    print("Done")
    plot_results(time_points, T_classical_list, T_quantum_list)
