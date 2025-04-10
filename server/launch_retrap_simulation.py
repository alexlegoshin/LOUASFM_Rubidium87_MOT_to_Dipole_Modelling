import numpy as np

from rb87_mot_model import simulate_mot, plot_results as mot_graphs
from rb87_retrap_model import simulate_retrap, plot_results as retrap_graphs

# Запуск моделирования МОЛ

times, positions, velocities, temperatures, avg_level_populations, avg_velocity_distributions = simulate_mot()
mot_graphs(times, positions, velocities, temperatures, avg_level_populations, avg_velocity_distributions)

# Сохранение данных в файлы
np.savetxt('../results_postprocessing/mot_simulation_results/times.csv', times, delimiter=',')
np.savetxt('../results_postprocessing/mot_simulation_results/positions.csv', positions, delimiter=',')
np.savetxt('../results_postprocessing/mot_simulation_results/velocities.csv', velocities, delimiter=',')
np.savetxt('../results_postprocessing/mot_simulation_results/temperatures.csv', temperatures, delimiter=',')
np.savetxt('../results_postprocessing/mot_simulation_results/avg_level_populations.csv', avg_level_populations, delimiter=',')
np.savetxt('../results_postprocessing/mot_simulation_results/avg_velocity_distributions.csv', avg_velocity_distributions, delimiter=',')

print(f'MOT graphs done')

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
