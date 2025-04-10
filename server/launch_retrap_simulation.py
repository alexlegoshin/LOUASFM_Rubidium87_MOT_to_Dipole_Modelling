import pandas as pd

from rb87_mot_model import simulate_mot, plot_results as mot_graphs
from rb87_retrap_model import simulate_retrap, plot_results as retrap_graphs

# Запуск моделирования МОЛ

times, positions, velocities, temperatures, avg_level_populations, avg_velocity_distributions = simulate_mot()
mot_graphs(times, positions, velocities, temperatures, avg_level_populations, avg_velocity_distributions)

# Сохранение данных: Преобразуем данные в DataFrame
data = {
    'Time (s)': times,
    'Position (m)': positions,
    'Velocity (m/s)': velocities,
    'Temperature (K)': temperatures,
    'Level F=1': [pop[0] for pop in avg_level_populations],
    'Level F=2': [pop[1] for pop in avg_level_populations],
    'Level F=3': [pop[2] for pop in avg_level_populations],
    'Level F=4': [pop[3] for pop in avg_level_populations],
    'Avg Velocity F=1 (m/s)': [vel[0] for vel in avg_velocity_distributions],
    'Avg Velocity F=2 (m/s)': [vel[1] for vel in avg_velocity_distributions],
    'Avg Velocity F=3 (m/s)': [vel[2] for vel in avg_velocity_distributions],
    'Avg Velocity F=4 (m/s)': [vel[3] for vel in avg_velocity_distributions]
}

df = pd.DataFrame(data)

# Сохраняем в CSV файл
df.to_csv('../results_postprocessing/mot_simulation_results/mot_simulation_data.csv', index=False)
print("Данные сохранены в '../results_postprocessing/mot_simulation_results/mot_simulation_data.csv'")

import numpy as np

# Используем финальные позиции и скорости из симуляции МОЛ
final_positions = positions[-1]
final_velocities = velocities[-1]

# Запуск симуляции перезахвата для гауссовой ловушки
_, _, _, T_classical, T_quantum = simulate_retrap(
    final_positions,
    final_velocities,
    potential_type='gaussian',
    trap_radius=10e-6,
    trap_depth=1e-27
)

# Построение графиков для перезахвата
retrap_graphs(
    [times[-1]], [T_classical], [T_quantum]
)
