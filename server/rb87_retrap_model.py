import numpy as np
from scipy.constants import k, hbar, pi
from rb87_gauss_potential_well_model import U_gauss
from rb87_lattice_potential_well_model import U_lattice

# Параметры атома
mass_Rb = 87 * 1.66e-27  # кг

# Параметры ловушки
trap_depth = 1e-27  # Дж, глубина ловушки (~70 мкК)
trap_radius = 10e-6  # м, характерный радиус ловушки
trap_shape = 'gaussian'  # или 'lattice'

# Параметры туннелирования
barrier_width = 5e-6  # м, толщина потенциального барьера
tunneling_prefactor = 1e3  # подгоночный параметр для вероятности туннелирования

# Параметры симуляции
dt = 1e-6  # шаг по времени
time_max = 0.5  # общее время симуляции (сек)
steps = int(time_max / dt)

def is_trapped(x, v, potential_fn, U0):
    """
    Проверка, захвачен ли атом (по полной энергии) и не вылетел ли по туннелированию.
    """
    kinetic_energy = 0.5 * mass_Rb * v ** 2
    potential_energy = potential_fn(x)
    total_energy = kinetic_energy + potential_energy

    if total_energy >= 0:
        return False  # не захвачен, вылетел

    # Вероятность туннелирования
    if potential_energy < 0:
        barrier_energy = abs(U0 - total_energy)
        tunneling_prob = np.exp(-tunneling_prefactor * barrier_width * np.sqrt(2 * mass_Rb * barrier_energy) / hbar)
        if np.random.rand() < tunneling_prob:
            return False  # туннелировал

    return True


def simulate_retrap(positions, velocities, potential_type):
    """
    Основная функция симуляции перезахвата атомов.
    """
    assert len(positions) == len(velocities)
    n_atoms = len(positions)

    # Выбор потенциальной ямы
    if potential_type == 'gaussian':
        potential_fn = U_gauss
        U0 = min(U_gauss(x) for x in np.linspace(-trap_radius*3, trap_radius*3, 1000))
    elif potential_type == 'lattice':
        potential_fn = U_lattice
        U0 = min(U_lattice(x) for x in np.linspace(-trap_radius*3, trap_radius*3, 1000))
    else:
        raise NotImplementedError("Поддерживается только 'gaussian' и 'lattice'")

    trapped_flags = np.ones(n_atoms, dtype=bool)

    for i in range(n_atoms):
        if not is_trapped(positions[i], velocities[i], potential_fn, U0):
            trapped_flags[i] = False

    return trapped_flags


# Пример использования
if __name__ == "__main__":
    # Эти данные должны приходить из simulate_mot()
    positions = np.random.normal(0, trap_radius * 2, 1000)
    velocities = np.random.normal(0, np.sqrt(k * 300e-6 / mass_Rb), 1000)

    result = simulate_retrap(positions, velocities)
    print(f"Перезахвачено {np.sum(result)} из {len(result)} атомов.")
