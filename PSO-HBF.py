import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

def calculate_average(arrays):
    df = pd.DataFrame(arrays)
    
    df = df.transpose()
    
    mean_values = df.mean(axis=1, skipna=True)
    
    return mean_values.tolist()


def initialize_population(size, dim, X_min, X_max):
	return np.random.uniform(X_min, X_max, (size, dim))


def fitness_function(x, mode):
	if mode == 'sphere':
		return np.sum(x ** 2)
	elif mode == 'rastrigin':
		return 10 * len(x) + np.sum(x**2 - 10*np.cos(2 * np.pi * x))
	elif mode == 'schwefel':
		return np.max(np.abs(x))
	


def compute_velocity(particle, best_personal, best_global):
	inertia = 0.7298  # Инерционная константа
	c1, c2 = 1.4961, 1.4961  # Константы обучения
	r1 = np.random.random()
	r2 = np.random.random()
	return (inertia * particle['velocity'] +
			c1 * r1 * (best_personal - particle['position']) +
			c2 * r2 * (best_global - particle['position']))


def update_position(particle):
	particle['position'] += particle['velocity']


def generate_theta(D):
	theta = np.random.randn(D)
	return theta


def main(mode = 'sphere', dimen = 2, pop_size = 4):
	global_value = []
	time_pos = []
	X_min, X_max = -100, 100  # Диапазон значений переменных
	if mode == 'rastrigin':
		X_min, X_max = -5, 5  # Диапазон значений переменных

	V_min, V_max = 0.1*X_min, 0.1*X_max  # Диапазон значений скорости

	dim = dimen  # Размерность задачи оптимизации

	population_size = pop_size  # Размер популяции
	maxit = 1e6  # Максимальное количество итераций
	n = round(population_size / 3) # Количество изменяемых частиц за итерацию
	it = 0  # Текущая итерация
	T = 100  # Момент времени запуска сценария смены позиции случайной частицы
	t = 1  # Начальное значение t
	A = np.random.random()  # Константа для выбора изменения параметра частицы

	# Генерация начальной популяции X^{i,0} -- популяция i на итерации it
	population_init = initialize_population(population_size, dim, X_min, X_max)
	population = []

	# Остановка, если разница не меняется на протяжении 20 итераций
	count_stop = 0

	for i in range(population_size):
		population.append({
			'position': population_init[i],
			'velocity': np.zeros(dim),
			'best_position': population_init[i],
			'best_fitness': fitness_function(population_init[i], mode),
		})

	global_best = min(population, key=lambda p: p['best_fitness'])
	global_best_position = global_best['best_position']
	global_best_fitness = global_best['best_fitness']

	while it <= maxit - n and count_stop < 1000:
		particles = random.sample(range(population_size), n)

		# Для критерия остановки
		old_global = global_best_fitness

		for p in particles:
			old_particle = population[p]
			new_particle = {
				'position': old_particle['position'].copy(),
				'velocity': old_particle['velocity'].copy(),
				'best_position': old_particle['best_position'].copy(),
				'best_fitness': old_particle['best_fitness'].copy(),
			}
			# Вычисление новой скорости для частицы p
			velocity = compute_velocity(old_particle,
										old_particle['best_position'],
										global_best_position)

			# Проверка выхода значений скорости за границы
			over_upper_velocity = velocity > V_max
			for g in range(len(over_upper_velocity)):
				if (over_upper_velocity[g]):
					velocity[g] = np.random.uniform(V_min, V_max) 
			over_lower_velocity = velocity < V_min
			for g in range(len(over_lower_velocity)):
				if (over_lower_velocity[g]):
					velocity[g] = np.random.uniform(V_min, V_max) 

			# Вычисляем вектор новой скорости
			mask = np.random.choice([0, 1], size=dim, p=[0.5, 0.5])
			new_velocity = velocity * mask

			# Устанавливаем новую скорость для новой частицы
			new_particle['velocity'] = new_velocity

			# Обновление позиции
			update_position(new_particle)

			# Проверка на выход за пределы границ
			new_particle['position'] = np.clip(new_particle['position'], X_min, X_max)

			# Оценка новой позиции
			new_fitness = fitness_function(new_particle['position'], mode)

			# обновление позиции и скорости старой частицы
			population[p]['velocity'] = new_particle['velocity'].copy()
			population[p]['position'] = new_particle['position'].copy()

			# Обновление частицы, если она лучше предыдущей
			if new_fitness < old_particle['best_fitness']:
				population[p]['best_position'] = population[p]['position'].copy()
				population[p]['best_fitness'] = new_fitness

			# Обновление глобальной информации
			if new_fitness < global_best_fitness:
				global_best_position = population[p]['position'].copy()
				global_best_fitness = new_fitness

			# HBF process
			# Расчёт шагов и коэффициентов корректировки
			theta = generate_theta(dim)
			beta = 0.5 + np.random.random()
			adjustment_coefficient1 = np.exp(t / 10)
			adjustment_coefficient2 = (1 - t / T) * (X_max - X_min)
			step_size = (t / T) * 1e-5 * (1 / np.abs(theta)) ** (1 / beta) * np.sign(theta)

			X_hbf = population[p]['best_position'] + step_size * adjustment_coefficient1
			Y_hbf = global_best_position + step_size * adjustment_coefficient2
			Z_pso = population[p]['position']

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
				if rand1 <= rand2 + A + rand3 * (t / T):
					if H > rand4:
						positions.append(X_hbf[j])
					else:
						positions.append(Y_hbf[j])
				else:
					positions.append(Z_pso[j])

			# Проверяем на принадлежность области поиска
			checked_positions = np.clip(positions, X_min, X_max)

			# Оценка новой позиции
			new_fitness = fitness_function(checked_positions, mode)

			population[p]['position'] = checked_positions.copy()

			# Обновление частицы, если она лучше предыдущей
			if new_fitness < population[p]['best_fitness']:
				population[p]['best_position'] = checked_positions.copy()
				population[p]['best_fitness'] = new_fitness

			# Обновление глобальной информации
			if new_fitness < global_best_fitness:
				global_best_position = population[p]['position'].copy()
				global_best_fitness = new_fitness

			# Обновление итерации
			t += 1
			it += 1
			
			# Замена и корректировка по необходимости
			if t == T:
				t = 1
				m = random.choice(population)
				new_position = []
				random_position = initialize_population(1, dim, X_min, X_max)[0]
				mask = np.random.choice([0, 1], size=dim, p=[0.5, 0.5])
				for i in range(len(mask)):
					if mask[i] == 1:
						new_position.append(global_best_position[i])
					else:
						new_position.append(random_position[i])
				new_position = np.array(new_position).copy()
				new_fitness = fitness_function(new_position, mode)
				if (new_fitness < fitness_function(m['position'], mode)):
					m['position'] = new_position.copy()
					m['best_position'] = np.array(new_position).copy()
					m['best_fitness'] = new_fitness

			# print(f"Итерация {it}: Лучшее значение = {global_best_fitness}")
			time_pos.append(it)
			global_value.append(global_best_fitness)

			if np.abs(old_global - global_best_fitness) < 1e-6:
				count_stop += 1
			else:
				count_stop = 0

	return global_value, time_pos

if __name__ == "__main__":
	general_value = []
	num_iter = 10

	# mode = 'sphere'
	mode = 'rastrigin'
	# mode = 'schwefel'

	# n = 2
	# n = 4
	# n = 8
	n = 16
	# n = 32

	N = 50

	num_of_iterations = 0
	for i in range(num_iter + 1):
		values, iterations = main(mode, n, N)
		general_value.append(values)
		if len(iterations) > num_of_iterations:
			num_of_iterations = len(iterations)
		left = 10 * i // 100
		right = 10 - left
		print('\r[', '#' * left, ' ' * right, ']', f' {i:.0f}%', sep='', end='', flush=True)

	mean_values = calculate_average(general_value)

	plt.figure(figsize=(10, 5))
	plt.plot([i for i in range(num_of_iterations)], mean_values, marker='o', linestyle='-', color='b')
	plt.title(r'Зависимость значения f от количества итераций')
	plt.xlabel('Число итераций')
	plt.ylabel(r'Вычисленное значение f')
	plt.text(num_of_iterations - 1, mean_values[-1], f'{mean_values[-1]:.7f}', fontsize=10, ha='left', va='bottom')
	plt.grid(True)
	plt.show()

	# main('rastrigin')
	# main('schwefel')
