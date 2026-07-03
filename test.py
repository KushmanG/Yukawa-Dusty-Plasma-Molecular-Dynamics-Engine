from sim import *

'''
The run.py file will run only if all run_flag == True
'''
# Initializations
random = np.random.default_rng(42)
N = 256
l = sim.L(N)
K = 1.0
T = 2.0
dt = 0.01

x = initial_positions(N, l, random)
v = initial_velocities(N, T, random)
a = net_force(x, l, K)

# Check if net force on a particle is 0 due to Newton's 3rd Law
# Machine precision ~ 1e-13
flag1 = True
total_force = np.abs(a.sum(axis = 0))
print("Net Force (should be ~0): ", total_force)
if (np.all(total_force < 1e-10)):
    print("\n\n3rd Law: PASS\n\n")
else:
    print("\n\n3rd Law: FAIL\n\n")
    flag1 = False


# Check for Energy Conservation
Gamma = 400.0
T = 1.0 / Gamma
x = initial_positions(N, l, random)
v = initial_velocities(N, T, random)
a = net_force(x, l, K)

energies = []
# T = 20s
for step in range(2000):
    x, v, a = dynamize(x, v, a, l, K, dt)          
    KE = 0.5 * np.sum(v*v)
    PE = potential_energy(x, l, K)
    energies.append(KE + PE)

energies = np.array(energies) # Converting the initialized python array into an np ndarray
energy_drift = (energies.max() - energies.min()) / abs(energies.mean())
print("Relative Energy Drift:", energy_drift)

flag2 = True
if (energy_drift < 1e-2):
    print("\n\nEnergy Conservation Check: PASS\n\n")
elif (energy_drift < 1e-1):
    print("\n\nEnergy Conservation Check: !!MARGINAL!!\n\n")
else:
    print("\n\nEnergy Conservation Check: FAIL\n\n")
    flag2 = False


# Check Thermostat: TIME-AVERAGED temperature must converge to target
Gamma = 400.0
T_target = 1.0 / Gamma
friction = 1.0

x = initial_positions(N, l, random)
v = initial_velocities(N, 2.0, random)      # start HOT
a = net_force(x, l, K)

T_samples = []
for step in range(6000):
    x, v, a = thermostat(x, v, a, T_target, friction, dt, l, K)
    if step > 1000:                          # sample only after equilibration
        T_samples.append(sim.temp(v))

T_mean = np.mean(T_samples)
print("T mean (equilibrated):", T_mean, " target:", T_target)

flag3 = True
if (np.abs(T_mean - T_target) < 0.05*T_target):    # within 5% of target
    print("\n\nThermostat Convergence: PASS\n\n")
else:
    print("\n\nThermostat Convergence: FAIL\n\n")
    flag3 = False

# g(r) normalization: random gas must give flat g(r) ~ 1
flag5 = True
frames = 50
gas = [random.uniform(0, l, size=(N, 2)) for _ in range(frames)]   # The Loop Length (50) is the number of frames
r, g = radial_distribution_function(gas, N, l, n_bins=80)
tail = g[5:]                                    # skip the small-r bins (unreliable)
g_mean = tail.mean()
print("g(r) mean (ideal gas):", g_mean)
if np.abs(g_mean - 1.0) < 0.05:
    print("g(r) Normalization: PASS")
else:
    print("g(r) Normalization: FAIL")
    flag5 = False

