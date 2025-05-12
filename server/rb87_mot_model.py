import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import k, c, hbar, pi

mu_B = 9.274e-24  # магнетон Бора, Дж/Тл
g_F = 0.5  # фактор Ланде (приблизительно для F=2)
magnetic_gradient = 50.0  # Тл/м, типичный градиент
import time
import os

# Физические параметры
mass_Rb = 87 * 1.66e-27  # кг, масса атома Rb-87
wavelength = 780.241e-9  # м, длина волны охлаждающего лазера
k_L = 2 * pi / wavelength  # волновой вектор
Gamma = 2 * pi * 6.07e6  # Реальное значение для рубидия-87
delta = -Gamma / 2  # Детюнинг охлаждающего лазера в рад/с (отрицательный для охлаждения) TODO: сделать зависимой от мощности лазера и т. п.
P_laser = 5.0  # Вт, тестовая мощность одного пучка для реалистичного s
beam_diameter = 9e-3  # м, диаметр пучка
I_s = 25  # Вт/м^2, насыщенность перехода
T0 = 360.00  # К, начальная температура атомов
v_threshold = 0.5  # Порог скорости для захваченных атомов (м/с)
beam_radius = beam_diameter / 2  # Радиус пучка

# Флуктуации в параметрах
wavelength_fluctuation = 1e-16  # абсолют флуктуации длины волны
power_fluctuation = 0.001  # абсолют флуктуации мощности
magnetic_fluctuation = 0.001  # абсолют флуктуации магнитного поля

# Параметры репампера (интенсивность, частота)
repumper_intensity = 1e1  # интенсивность репампера в Вт
repumper_wavelength = 780.244e-9  # длина волны репампера
repumper_effect = 0.975  # коэффициент влияния репампера на переходы

# Параметры симуляции
atoms_quantity = 300  # Число атомов — в реальности 500 000
nsim = 1  # Количество симуляций
timesim = 2e-3  # Период симуляции
dtsim = 2e-8  # Шаг симуляции (в секундах)
itera = timesim / dtsim

# Параметры переходов между уровнями (примерные значения)
transition_probabilities = {
    (0, 1): 5e-21,  # Переход с уровня 0 (F=1) на уровень 1 (F=2)
    # (1, 2): 0.6,  # Удалено: возбуждение F=2 -> F'=3 теперь только через лазер
    (2, 3): 5e-21,  # Переход с уровня 2 (F=3) на уровень 3 (темновой)
    (1, 0): 0,  # Переход с уровня 1 (F=2) на уровень 0 (F=1)
    # (2, 1): 0.1,  # Удалено: спад F'=3 -> F=2 теперь только через спонтанный спад
    (3, 2): 5e-21,  # Переход с уровня 3 (темновой) на уровень 2 (F=3)
}


def transition_between_levels(levels, temperature, dt):
    """
    Функция для обработки переходов между уровнями в процессе работы ловушки.
    Это простая модель с вероятностными переходами между уровнями атомов.

    levels - массив уровней атомов
    temperature - температура ловушки (средняя кинетическая энергия)
    dt - временной шаг
    """
    temperature = max(min(temperature, 1e4), 1e-3)  # защита от переполнения
    for i in range(len(levels)):
        # Для каждого атома с вероятностью по температуре и переходам между уровнями
        for (start, end), prob in transition_probabilities.items():
            # Вычисляем вероятность перехода на основе температуры
            exponent = -abs(start - end) * temperature / k
            if exponent < -700:
                transition_prob = 0.0
            else:
                transition_prob = prob * np.exp(exponent) * dt
            if np.random.rand() < transition_prob:
                if levels[i] == start:
                    levels[i] = end  # Переход на новый уровень
    return levels


def repumper_effect_on_levels(levels, velocities, I_repumper, temperature, delta_wavelength=wavelength_fluctuation):
    """
    Воздействие репампера: переводит атомы с F=1 на F=2, а также может учитывать возможные утечки на темновый уровень F=4.

    levels - массив уровней атомов (0 - F=1, 1 - F=2, 2 - F=3, 3 - F=4)
    velocities - массив скоростей атомов
    I_repumper - интенсивность репампирующего лазера
    delta_wavelength - флуктуации длины волны (по умолчанию 0)

    Возвращает массив новых уровней атомов после воздействия репампера.
    """
    # Волновой вектор репампирующего лазера с учётом флуктуаций
    k_L_eff_repumper = 2 * np.pi / (repumper_wavelength * (1 + np.random.normal(0, delta_wavelength)))

    # Вероятность перехода с F=1 на F=2 (основной эффект репампера)
    temp_safe = max(temperature, 1e-1)  # Устанавливаем минимальное значение температуры
    v_th = np.sqrt(2 * k * temp_safe / mass_Rb)
    transition_prob = repumper_effect * I_repumper / I_s * np.exp(-velocities ** 2 / v_th**2)

    # Создаём копию массива уровней
    level_transition_prob = np.copy(levels)

    # Только атомы на уровне F=1 могут перейти на F=2
    mask_F1 = levels == 0  # индексы атомов на уровне F=1
    transitions = np.random.rand(len(levels)) < transition_prob  # случайные события перехода
    level_transition_prob[mask_F1 & transitions] = 1  # Переход F=1 -> F=2

    # Дополнительные спонтанные утечки с F=2 и F=3 на F=4 (полностью темновый уровень)
    leak_prob = 1e-21  # Маленькая вероятность утечки на темновый уровень (параметр можно подстраивать)
    mask_F2_F3 = (levels == 1) | (levels == 2)  # Атомы на F=2 и F=3
    leaks = np.random.rand(len(levels)) < leak_prob  # случайные утечки
    level_transition_prob[mask_F2_F3 & leaks] = 3  # Переход F=2,3 -> F=4

    return level_transition_prob


def scattering_force(v, delta, I, r, delta_wavelength=wavelength_fluctuation, delta_magnetic=magnetic_fluctuation):
    """Сила рассеяния для доплеровского охлаждения с флуктуациями в длине волны, мощности и магнитном поле."""
    # Без флуктуаций длины волны и магнитного поля
    k_L_eff = 2 * pi / wavelength
    I_r = I * np.exp(-(r ** 2) / (beam_radius ** 2))
    s = I_r / I_s  # Нормированная интенсивность с учётом формы пучка
    delta_effective = delta

    return hbar * k_L_eff * I_r * Gamma / (2 * (1 + s + 4 * (delta_effective - k_L_eff * v) ** 2 / Gamma ** 2))


# Детюнинг с учётом магнитного поля (Зеемановский сдвиг)
def delta_with_magnetic(x):
    return delta - mu_B * g_F * magnetic_gradient * x / hbar


def simulate_mot(n_atoms=atoms_quantity, time_max=timesim, dt=dtsim, n_simulations=nsim):
    """Моделирование движения атомов в МОЛ с расчётом температуры, временем жизни и репампером."""
    num_steps = int(time_max / dt)
    times = np.linspace(0, time_max - dt, num_steps)
    positions_all = np.zeros((n_simulations, n_atoms, num_steps))
    velocities_all = np.zeros((n_simulations, n_atoms, num_steps))
    temperatures_all = []
    velocity_distributions_all = []
    level_populations_all = []

    for sim_index in range(n_simulations):  # многократное моделирование для усреднения
        velocities = np.random.normal(0, np.sqrt(k * T0 / mass_Rb), n_atoms)
        positions = np.random.uniform(-beam_radius, beam_radius, n_atoms)  # Инициализация позиций в пределах пучка
        # Для тестов: все атомы в F=2
        levels = np.random.choice([0, 1, 2, 3], size=n_atoms, p=[0.0, 1.0, 0.0, 0.0])  # 100% F=2
        T_avg_trapped = T0
        velocities_to_avg = []
        positions_to_avg = []
        temperatures = []
        velocity_distributions = []
        levels_distributions = []

        start_time = time.time()

        for i, t in enumerate(times):

            # Моделирование динамики с учётом влияния флуктуаций
            P_laser_eff = P_laser * (1 + np.random.normal(0, power_fluctuation))  # флуктуации мощности
            forces = np.zeros(n_atoms)
            # Новый цикл расчёта силы и переходов между F=2 и F'=3
            for j in range(n_atoms):
                if levels[j] == 1:  # F=2 — только из него возможна стимуляция
                    intensity = P_laser_eff / (pi * (beam_diameter / 2) ** 2)
                    I_r = intensity * np.exp(-positions[j] ** 2 / beam_radius ** 2)
                    s = I_r / I_s
                    prob_exc = s / (1 + s + 4 * ((delta - k_L * velocities[j]) / Gamma) ** 2) * Gamma * dt
                    if np.random.rand() < prob_exc:
                        levels[j] = 2  # возбуждение F=2 -> F'=3
                elif levels[j] == 2:  # F'=3 — возбуждённый уровень
                    if np.random.rand() < Gamma * dt:
                        levels[j] = 1  # спонтанный спад обратно на F=2

            # Сила охлаждения действует только на уровне F=2
            for j in range(n_atoms):
                if levels[j] == 1:
                    intensity = P_laser_eff / (pi * (beam_diameter / 2) ** 2)
                    F_plus = scattering_force(
                        +velocities[j],
                        delta_with_magnetic(positions[j]),
                        intensity,
                        positions[j],
                        delta_wavelength=wavelength_fluctuation,
                        delta_magnetic=magnetic_fluctuation
                    )
                    F_minus = scattering_force(
                        -velocities[j],
                        delta_with_magnetic(positions[j]),
                        intensity,
                        positions[j],
                        delta_wavelength=wavelength_fluctuation,
                        delta_magnetic=magnetic_fluctuation
                    )
                    forces[j] = F_minus - F_plus  # Сила направлена против скорости
                    if i % 1000 == 0 and j == 0:
                        print(f"F_minus = {F_minus:.2e}, F_plus = {F_plus:.2e}, Force = {forces[j]:.2e}")
                        if abs(forces[j]) < 1e-23:
                            print("⚠️ Force is too weak — cooling might be ineffective.")
                else:
                    forces[j] = 0  # на других уровнях сила не действует

            velocities += forces / mass_Rb * dt
            positions += velocities * dt
            boundary = 2.5 * beam_diameter
            out_of_bounds = np.abs(positions) > boundary
            velocities[out_of_bounds] *= -1  # отражаем скорость
            positions[out_of_bounds] = np.sign(positions[out_of_bounds]) * boundary  # фиксируем позицию на границе
            positions_all[sim_index, :, i] = positions
            velocities_all[sim_index, :, i] = velocities

            # Распределение скоростей для каждого уровня
            # Вычисление средних значений скорости для уровней
            if np.any(levels == 1):
                avg_level_1_velocity = np.nanmean(velocities[levels == 1])
            else:
                avg_level_1_velocity = np.nan

            if np.any(levels == 2):
                avg_level_2_velocity = np.nanmean(velocities[levels == 2])
            else:
                avg_level_2_velocity = np.nan

            if np.any(levels == 3):
                avg_level_3_velocity = np.nanmean(velocities[levels == 3])
            else:
                avg_level_3_velocity = np.nan

            if np.any(levels == 4):
                avg_level_4_velocity = np.nanmean(velocities[levels == 4])
            else:
                avg_level_4_velocity = np.nan

            # Добавляем результаты в список
            velocity_distributions.append([
                avg_level_1_velocity,
                avg_level_2_velocity,
                avg_level_3_velocity,
                avg_level_4_velocity
            ])

            # Отбор только захваченных атомов, которые находятся внутри пучка (в пределах радиуса)
            trapped_atoms = (np.abs(positions) < beam_radius) & (np.abs(velocities) < v_threshold) & ((levels == 1) | (levels == 0))
            trapped_positions = positions[trapped_atoms]
            trapped_velocities = velocities[trapped_atoms]

            # Средняя температура только для захваченных атомов
            if len(trapped_velocities) > 0:
                T_avg_trapped = np.mean(mass_Rb * trapped_velocities ** 2) / k
            else:
                T_avg_trapped = -1  # Если нет захваченных атомов, температура равна -1

            # Средняя температура по всем атомам
            T_avg_total = np.mean(mass_Rb * velocities ** 2) / k
            if i % 1000 == 0:
                print(f"[{t*1e6:.1f} μs] trapped: {T_avg_trapped:.5f} K, total: {T_avg_total:.5f} K")

            temperatures.append(T_avg_trapped)
            temperature = T_avg_trapped

            # Переходы между уровнями в процессе работы МОЛ
            levels_ = transition_between_levels(levels, temperature, dt)
            levels = levels_

            # Переходы между уровнями из-за репампера
            levels_ = repumper_effect_on_levels(levels, velocities, repumper_intensity, temperature)
            levels = levels_

            # Заселённость уровней (распределение)
            level_counts = np.bincount(levels, minlength=4)
            level_distribution = level_counts / np.sum(level_counts)
            levels_distributions.append(level_distribution)

            # Оценка прогресса с учётом времени
            if len(times) > 1000 and i % (len(times) // 1000) == 0:
                elapsed_time = time.time() - start_time  # прошедшее время
                time_per_iteration = elapsed_time / (i + 1)  # время на одну итерацию
                time_remaining = time_per_iteration * (len(times) - i) + \
                                 time_per_iteration * len(times) * (n_simulations - 1 - sim_index)

                days = time_remaining // 86400 # 86400 секунд в сутках
                time_struct = time.gmtime(time_remaining % 86400)  # Преобразуем остаток времени
                formatted_time = time.strftime('%H:%M:%S', time_struct)

                print(f"\rПрогресс (МОЛ): симуляция {sim_index + 1}/{n_simulations}, "
                      f"актуальная симуляция завершена на {100 * i / len(times):.3f}%, "
                      f"примерное время до полного завершения моделирования: {int(days)} дней, {formatted_time}",
                      end='')

        temperatures_all.append(temperatures)
        level_populations_all.append(levels_distributions)
        velocity_distributions_all.append(velocity_distributions)

    # Усреднение температур и заселённости уровней за несколько симуляций
    # Усреднение по симуляциям
    avg_positions = np.mean(positions_all, axis=0)
    avg_velocities = np.mean(velocities_all, axis=0)
    # Транспонирование массивов для корректной формы [num_steps, n_atoms]
    avg_positions = avg_positions.T
    avg_velocities = avg_velocities.T
    avg_temperatures = np.mean(temperatures_all, axis=0)
    avg_level_populations = np.mean(level_populations_all, axis=0)
    avg_velocity_distributions = np.mean(velocity_distributions_all, axis=0)

    print(f"\rПрогресс (МОЛ): 100%, моделирование завершено")
    print(f"Температура облака: {avg_temperatures[-1]:.10f}К")

    return times, avg_positions, avg_velocities, avg_temperatures, avg_level_populations, avg_velocity_distributions


def plot_results(times, positions, velocities, temperatures, avg_level_populations, avg_velocity_distributions):
    """Графики распределений, температуры и времени жизни ловушки."""

    save_folder = "../results_postprocessing/mot_simulation_results"
    os.makedirs(save_folder, exist_ok=True)

    # Позиции атомов на разных временных моментах
    fig, ax = plt.subplots(1, 2, figsize=(22, 6))
    ax[0].hist(positions[0] * 1e3, bins=50, density=True, alpha=0.7, color='r', label="0%")
    ax[0].hist(positions[len(positions) // 5] * 1e3, bins=50, density=True, alpha=0.7, color='orange', label="20%")
    ax[0].hist(positions[len(positions) // 2] * 1e3, bins=50, density=True, alpha=0.7, color='y', label="50%")
    ax[0].hist(positions[len(positions) * 4 // 5] * 1e3, bins=50, density=True, alpha=0.7, color='g', label="80%")
    ax[0].hist(positions[len(positions) - 1] * 1e3, bins=50, density=True, alpha=0.7, color='b', label="100%")
    ax[0].set_xlabel("Позиция (мм)")
    ax[0].set_ylabel("Плотность вероятности")
    ax[0].set_title(f"Положение атомов в моменты 0%, 20%, 50%, 80%, 100% периода моделирования")
    ax[0].legend()

    ax[1].hist(velocities[0], bins=50, density=True, alpha=0.7, color='r', label="0%")
    ax[1].hist(velocities[len(velocities) // 5], bins=50, density=True, alpha=0.7, color='orange', label="20%")
    ax[1].hist(velocities[len(velocities) // 2], bins=50, density=True, alpha=0.7, color='y', label="50%")
    ax[1].hist(velocities[len(velocities) * 4 // 5], bins=50, density=True, alpha=0.7, color='g', label="80%")
    ax[1].hist(velocities[len(velocities) - 1], bins=50, density=True, alpha=0.7, color='b', label="100%")
    ax[1].set_xlabel("Скорость (м/с)")
    ax[1].set_ylabel("Плотность")
    ax[1].set_title(f"Динамика атомов в моменты 0%, 20%, 50%, 80%, 100% периода моделирования")
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(f'{save_folder}/positions_velocities.png')

    # Температура в зависимости от времени
    fig, ax = plt.subplots(1, 1, figsize=(22, 6))
    ax.plot(times * 1e6, temperatures, color='g')
    ax.set_xlabel("Время (μs)")
    ax.set_ylabel("Средняя температура (K)")
    ax.set_title("Зависимость средней температуры захваченных в МОЛ атомов от времени")
    plt.tight_layout()
    plt.savefig(f'{save_folder}/temperature_vs_time.png')

    # Заселённости уровней
    fig, ax = plt.subplots(1, 2, figsize=(22, 6))
    level_1_population = avg_level_populations[:, 0]
    level_2_population = avg_level_populations[:, 1]
    level_3_population = avg_level_populations[:, 2]
    level_4_population = avg_level_populations[:, 3]
    ax[0].plot(times, level_1_population, 'g-', label="Уровень F=1")
    ax[0].plot(times, level_2_population, 'b-', label="Уровень F=2")
    ax[0].plot(times, level_3_population, 'r-', label="Уровень F'=3")
    ax[0].set_xlabel('Время')
    ax[0].set_ylabel('Заселённость уровней')
    ax[0].set_title('Заселённости уровней в зависимости от времени')
    ax[0].legend()

    # Динамика атомов по уровням
    level_1_velocities, level_2_velocities, level_3_velocities, level_4_velocities = avg_velocity_distributions.T
    ax[1].plot(times, abs(level_1_velocities), 'g-', label="Уровень F=1")
    ax[1].plot(times, abs(level_2_velocities), 'b-', label="Уровень F=2")
    ax[1].plot(times, abs(level_3_velocities), 'r-', label="Уровень F'=3")
    ax[1].set_xlabel('Время')
    ax[1].set_ylabel('Средняя скорость (м/с)')
    ax[1].set_title('Динамика атомов по уровням в зависимости от времени')
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(f'{save_folder}/level_populations_and_velocities.png')

    # Распределение скоростей для каждого уровня
    fig_velocities, ax_velocities = plt.subplots(1, 3, figsize=(22, 6))

    for i, level_velocities in enumerate([level_1_velocities, level_2_velocities, level_3_velocities]):
        for j, time_point in enumerate([0, len(times) // 5, len(times) // 2, 4 * len(times) // 5, len(times) - 1]):
            # Фильтрация NaN значений
            clean_velocities = np.array([v for v in level_velocities if not np.isnan(v)])

            # Строим гистограмму только с чистыми данными
            if len(clean_velocities) > 0:
                ax_velocities[i].hist(clean_velocities, bins=50, density=True, alpha=0.7, label=f"{j * 20}%",
                                      color=['r', 'orange', 'y', 'g', 'b'][j])

        ax_velocities[i].set_xlabel("Скорость (м/с)")
        ax_velocities[i].set_ylabel("Плотность")
        ax_velocities[i].set_title(f"Распределение скоростей для Уровня {i + 1}")

    plt.tight_layout()
    plt.savefig(f'{save_folder}/velocity_distributions.png')

    plt.close('all')  # Закрываем все графики, чтобы освободить память
