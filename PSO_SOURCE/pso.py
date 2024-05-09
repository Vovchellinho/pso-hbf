import numpy as np
import random

def sphere_function(x):
	"""A simple sphere function."""
	return np.sum((x - 0.345) ** 2)

def initialize_population(m, D, min_var, max_var):
	"""Initialize the population."""
	pop = np.random.rand(m, D)
	pop = pop * (max_var - min_var) + min_var
	return pop

def main():
	# Parameters
	n_swarm = 6
	n_var = 1000

	min_var = -100 * np.ones(n_var)
	max_var = 100 * np.ones(n_var)

	w = 0.7298
	c1 = 1.4961
	c2 = 1.4961

	# Initialize populations
	pop = initialize_population(n_swarm, n_var, min_var, max_var)
	pop_cost = np.array([np.append(p, sphere_function(p)) for p in pop])

	pop_cost = pop_cost[np.argsort(pop_cost[:, -1])]  # Sort by cost
	g_best = pop_cost[0, :]
	p_best = pop_cost.copy()

	v = np.zeros((n_swarm, n_var))

	upper_v = 0.1 * max_var
	lower_v = 0.1 * min_var

	EFs = 0
	T_h = 100
	it1 = 0

	while EFs <= 3e6:
		# Main loop
		for i in range(n_swarm):
			it = i - it1 * T_h

			if it % T_h == 0:
				loc = np.random.permutation(n_var)
				loc1 = loc[:int(n_var * random.random() - 1) + 1]
				Xs = np.random.permutation(n_swarm)
				
				pop[Xs[0], loc1] = np.random.rand(len(loc1)) * (max_var[loc1] - min_var[loc1]) + min_var[loc1]
				Ccost = sphere_function(pop[Xs[0]])
				
				p_best[Xs[0]] = np.append(pop[Xs[0]], Ccost)
				it1 += 1
				EFs += 1
			
			two_pop = np.random.permutation(n_swarm)[:2]

			for j in two_pop:
				v[j] = (
					c1 * random.random() * (p_best[j, :n_var] - pop_cost[j, :n_var])
					+ c2 * random.random() * (g_best[:n_var] - pop_cost[j, :n_var])
					+ w * v[j]
				)

				# Clipping velocity to upper_v and lower_v
				v[j] = np.clip(v[j], lower_v, upper_v)

				# New position calculation
				pop[j, :n_var] = pop_cost[j, :n_var] + v[j]

				# Boundary checks
				pop[j, :n_var] = np.clip(pop[j, :n_var], min_var, max_var)

				# Calculate new cost
				CCcost = sphere_function(pop[j, :n_var])
				pop_cost1 = np.array(pop_cost.copy())
				if CCcost <= pop_cost[j, -1]:
					pop_cost1[j] = np.append(pop[j, :n_var], CCcost)
				else:
					pop_cost1[j] = pop_cost[j]

				EFs += 1

				# HBF strategy implementation
				vx = np.random.randn(n_var)
				beta = 0.5 + random.random()
				stepsize = 0.00001 * (it / T_h) * ((1. / np.abs(vx)) ** (1 / beta)) * np.sign(vx)
				Xs = p_best[j, :n_var] + (stepsize) * np.exp(it / 10)
				Xa = p_best[j, :n_var] + 1 * (stepsize) * (1 - it / T_h) * (max_var - min_var)

				x = g_best[:n_var] if j < n_swarm else pop[j, :n_var]

				H = np.abs(2 * random.random() * np.exp(2 - 2 * it / T_h) - np.exp(2 - 2 * it / T_h))
				rr = np.random.rand(n_var)
				cc = np.random.rand(n_var)

				xx = np.where(rr <= cc * it / T_h)[0]

				k1 = np.zeros(n_var)
				k2 = np.zeros(n_var)

				if H > 0.3 * random.random():
					k1[xx] = 1
				else:
					k2[xx] = 1

				k3 = np.zeros(n_var)
				xx = np.where(rr > cc * it / T_h)[0]
				k3[xx] = 1

				z = Xs * k1 + Xa * k2 + k3 * x

				z = np.clip(z, min_var, max_var)

				zcost = sphere_function(z)
				EFs += 1

				if zcost < p_best[j, -1]:
					p_best[j] = np.append(z, zcost)

				if pop_cost1[j, -1] < p_best[j, -1]:
					p_best[j] = pop_cost1[j]

		pop_cost = pop_cost1.copy()
		pop_cost = pop_cost[np.argsort(pop_cost[:, -1])]
		pop_cost_sort1 = p_best[np.argsort(p_best[:, -1])]

		g_best1 = pop_cost_sort1[0]

		if g_best1[-1] < g_best[-1]:
			g_best = g_best1

		if i % 100 == 0:
			print(f"EFs>> {EFs}: Best Cost = {g_best[-1]}")

if __name__ == "__main__":
	main()
