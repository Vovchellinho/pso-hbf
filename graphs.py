import matplotlib.pyplot as plt

# sphere
# L = [1, 1, 1, 1, 1]
# t = [3968, 4746, 6544, 13792, 32566]
# Q = [1.299e-12, 1.3015e-09, 1.138e-07, 2.157e-06, 2.246e-05]
# n = [2, 4, 8, 16, 32]

# rastrigin
# L = [0.9, 0.72, 0.45, 0.45, 0.39]
# t = [4835, 6992, 14987, 47844, 141658]
# Q = [0.099, 0.3084, 1.0646, 1.184, 1.2936]
# n = [2, 4, 8, 16, 32]

# schwefel
L = [1, 1, 1, 1, 1]
t = [5949, 13749, 24447, 49459, 111710]
Q = [3.097e-07, 4.903e-06, 7.466e-05, 0.00049, 0.00308]
n = [2, 4, 8, 16, 32]

plt.figure(figsize=(10, 5))
plt.plot(n, L, marker='o', linestyle='-', color='b', markersize='5')
plt.xlabel(r'n')
plt.ylabel(r'L', rotation=0)
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(n, t, marker='o', linestyle='-', color='b', markersize='5')
plt.xlabel(r'n')
plt.ylabel(u't\u0304', rotation=0)
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.yscale("log")
plt.plot(n, Q, marker='o', linestyle='-', color='b', markersize='5')
plt.xlabel(r'n')
plt.ylabel(r'Q', rotation=0)
plt.grid(True)
plt.show()