import pandas as pd

from rb87_mot_model import simulate_mot, plot_results as mot_graphs
from rb87_retrap_model import simulate_retrap

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
df.to_csv('mot_simulation_data.csv', index=False)

print("Данные сохранены в 'mot_simulation_data.csv'")

