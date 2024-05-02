import numpy as np
import random
import scipy.stats as stats


def initialize_population(size, dim, X_min, X_max):
    """Инициализация популяции частиц случайными значениями в заданном диапазоне."""
    return np.random.uniform(X_min, X_max, (size, dim))


def fitness_function(x):
    """Пример функции пригодности для оценки качества решений."""
    return np.sum(x ** 2)


def compute_velocity(particle, best_personal, best_global):
    """Вычисление новой скорости частицы."""
    inertia = 0.7  # Инерционная константа
    c1, c2 = 2.05, 2.05  # Константы обучения
    return (inertia * particle['velocity'] +
            c1 * (best_personal - particle['position']) +
            c2 * (best_global - particle['position']))


def update_position(particle):
    """Обновление позиции частицы."""
    particle['position'] += particle['velocity']


def generate_theta(D):
    # uniform_random_values = np.random.random(D)
    # theta = stats.norm.ppf(uniform_random_values)
    rndn_values = np.random.randn(D)
    theta = 1 / rndn_values
    return theta


def main():
    dim = 10  # Размерность задачи оптимизации
    X_min, X_max = -10, 10  # Диапазон значений переменных
    population_size = 5  # Размер популяции
    maxit = 10000  # Максимальное количество итераций
    it = 0  # Текущая итерация
    T = 100  # Время эволюции
    t = 1  # Начальное значение t
    A = np.random.random()  # Константа для выбора изменения параметра частицы

    # Генерация начальной популяции X^{i,0} -- популяция i на итерации it
    population_init = initialize_population(population_size, dim, X_min, X_max)
    population = []
    for i in range(population_size):
        population.append({
            'position': population_init[i],
            'velocity': np.zeros(dim),
            'best_position': population_init[i],
            'best_fitness': fitness_function(population_init[i]),
        })

    global_best = min(population, key=lambda p: p['best_fitness'])
    global_best_position = global_best['best_position']
    global_best_fitness = global_best['best_fitness']

    while it <= maxit:

        # Выбор двух частиц для сравнения
        r, k = random.sample(range(population_size), 2)

        for i in [r, k]:
            # Вычисление новой скорости для частицы i
            velocity = compute_velocity(population[i],
                                        population[i]['best_position'],
                                        global_best_position)

            # Вычисляем вектор новой скорости
            mask = np.random.choice([0, 1], size=len(velocity), p=[0.5, 0.5])
            new_velocity = velocity * mask
            # Устанавливаем новую скорость для новой частицы

            population[i]['velocity'] = new_velocity

            # Обновление позиции
            update_position(population[i])

            # Проверка на выход за пределы границ
            population[i]['position'] = np.clip(population[i]['position'], X_min, X_max)

            # Оценка новой позиции
            new_fitness = fitness_function(population[i]['position'])

            # Обновление частицы, если она лучше предыдущей
            if new_fitness < population[i]['best_fitness']:
                population[i]['best_position'] = population[i]['position'].copy()
                population[i]['best_fitness'] = new_fitness

            # Обновление глобальной информации
            if new_fitness < global_best_fitness:
                global_best_position = population[i]['position'].copy()
                global_best_fitness = new_fitness

            # HBF process
            # Расчёт шагов и коэффициентов корректировки
            theta = generate_theta(dim)
            beta = 0.5 + np.random.random()
            adjustment_coefficient1 = np.exp(t / 10)
            adjustment_coefficient2 = (1 - t / T) * (X_max - X_min)
            step_size = (t / T) * 1e-5 * (1 / np.abs(np.abs(theta))) ** (1 / beta) * np.sign(theta)

            X_hbf = population[i]['best_position'] + step_size * adjustment_coefficient1
            Y_hbf = global_best_position + step_size * adjustment_coefficient2
            Z_pso = population[i]

            h = np.exp(2 - 2 * t / T)
            H = np.abs(2 * random.random() * h - h)

            # Позиция для рассчитанной частицы
            positions = []

            # Для каждого решения частицы
            for j in range(dim):
                rand1 = np.random.random()
                rand2 = np.random.random()
                rand3 = np.random.random()
                rand4 = np.random.random()
                # Различные случайные изменения
                if rand1 <= rand2 + A * (t / T) + rand3:
                    if H > rand4:
                        positions.append(X_hbf[j])
                    else:
                        positions.append(Y_hbf[j])
                else:
                    positions.append(Z_pso['position'][j])

            # Проверяем на принадлежность области поиска
            checked_positions = np.clip(positions, X_min, X_max)

            # Оценка новой позиции
            new_fitness = fitness_function(checked_positions)

            # Обновление частицы, если она лучше предыдущей
            if new_fitness < population[i]['best_fitness']:
                population[i]['best_position'] = positions.copy()
                population[i]['best_fitness'] = new_fitness

            # Обновление глобальной информации
            if new_fitness < global_best_fitness:
                global_best_position = population[i]['position'].copy()
                global_best_fitness = new_fitness

            # Обновление итерации
            t += 1
            it += 1
            # Замена и корректировка по необходимости
            if t == T:
                t = 1
                # m = random.choice(population)
                # m['position'] = np.copy(global_best_position)
                # mask = np.random.choice([0, 1], size=dim, p=[0.5, 0.5])
                # ne
                # m['best_position'] =
                # m['best_fitness'] = global_best_fitness

        print(f"Итерация {it}: Лучшее значение = {global_best_fitness}")


if __name__ == "__main__":
    main()
