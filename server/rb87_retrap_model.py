import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import k, hbar, pi
from rb87_gauss_potential_well_model import U_gauss
from rb87_lattice_potential_well_model import U_lattice

import os

# Параметры атома
mass_Rb = 87 * 1.66e-27  # кг

# Параметры ловушки
trap_depth = 1e-27  # Дж, глубина ловушки (~70 мкК)
trap_radius = 10e-6  # м, характерный радиус ловушки

# Параметры симуляции
dt = 1e-6  # шаг по времени
time_max = 0.5  # общее время симуляции (сек)
steps = int(time_max / dt)

# --- Решение уравнения Шрёдингера ---
def solve_schrodinger(potential_fn, mass, x_grid):
    dx = x_grid[1] - x_grid[0]
    N = len(x_grid)

    T = -0.5 * (hbar**2 / mass) * (
        np.diag(np.ones(N - 1), 1)
        - 2 * np.diag(np.ones(N), 0)
        + np.diag(np.ones(N - 1), -1)
    ) / dx**2

    V = np.diag([potential_fn(x) for x in x_grid])
    H = T + V

    energies, wavefuncs = np.linalg.eigh(H)
    return energies, wavefuncs

# --- Вероятность захвата ---
def trapping_probability(x0, v0, potential_fn, energies, wavefuncs, x_grid):
    kinetic_energy = 0.5 * mass_Rb * v0**2
    potential_energy = potential_fn(x0)
    total_energy = kinetic_energy + potential_energy

    bound_states = energies[energies < total_energy]

    if len(bound_states) == 0:
        return 1.0
    elif len(bound_states) > 10:
        return 0.0
    else:
        return 1.0 - len(bound_states) / 10

# --- Температура классическая и квантовая ---
def compute_temperatures(velocities, energies, wavefuncs, x_grid, trapped_flags):
    trapped_velocities = velocities[trapped_flags]

    if len(trapped_velocities) == 0:
        return 0.0, 0.0

    T_classical = np.mean(0.5 * mass_Rb * trapped_velocities**2) / (0.5 * k)

    # Квантовая температура: считаем среднюю энергию в ловушке по Больцману
    energies = energies[energies < 0]
    if len(energies) == 0:
        return T_classical, 0.0

    pop = np.exp(-energies / (k * T_classical))
    pop /= np.sum(pop)

    avg_energy = np.sum(energies * pop)
    T_quantum = avg_energy / k if avg_energy > 0 else 0.0

    return T_classical, T_quantum

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

    trapped_flags = np.ones(n_atoms, dtype=bool)
    for i in range(n_atoms):
        prob = trapping_probability(positions[i], velocities[i], potential_fn, energies, wavefuncs, x_grid)
        if np.random.rand() > prob:
            trapped_flags[i] = False

    T_classical, T_quantum = compute_temperatures(velocities, energies, wavefuncs, x_grid, trapped_flags)
    return trapped_flags, positions, velocities, T_classical, T_quantum

# --- Визуализация температур ---
def plot_results(time_points, T_classical_list, T_quantum_list):
    results_dir = "../results_postprocessing/dipole_simulation_results"
    os.makedirs(results_dir, exist_ok=True)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_points, T_classical_list, label="Классическая температура", color='cyan')
    ax.plot(time_points, T_quantum_list, label="Квантовая температура", color='magenta')
    ax.set_xlabel("Время (с)")
    ax.set_ylabel("Температура (K)")
    ax.set_title("Температура атомов в дипольной ловушке")
    ax.legend()
    ax.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/temperature_evolution.png")
    plt.close()

# --- Пример использования ---
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
