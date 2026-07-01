from sim import *

random = np.random.default_rng(42)
N = 64
l = sim.L(N)
K = 1.0
T = 2.0
dt = 0.01

x = initial_positions(N, l, random)
v = initial_velocities(N, T, random)
a = net_force(x, l, K)

print("Temperature before:", sim.temp(v))
for step in range(200):
    x, v, a = dynamize(x, v, a, l, K, dt)

print("T after: ", sim.temp(v))
plt.scatter(x[:, 0], x[:, 1])
plt.gca().set_aspect("equal")
plt.show()