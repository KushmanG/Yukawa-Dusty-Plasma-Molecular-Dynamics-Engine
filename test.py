from sim import *

'''
The run.py file will run only if all run_flag == True
'''
# Initializations
random = np.random.default_rng(42)
N = 256
l = L(N)
K = 1.0
Gamma = 400
T1 = 2.0        # Hot
T2 = 1/Gamma    # Cold
dt = 0.01

x = initial_positions(N, l, random)
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


# Check for Energy Conservation (NVE); fresh state so the test is independent
v = initial_velocities(N, T2, random)

energies = []
# Experiment Time = 20s [VARIABLE] --> Change by changing range limit 
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
# Need to re-initialize x and a too because they were dynamized in the last test, we need to start with initial conditions again
friction = 1.0
x = initial_positions(N, l, random)
v = initial_velocities(N, T1, random)   # start HOT at T1
a = net_force(x, l, K)

T_samples = []
for step in range(6000):
    x, v, a = thermostat(x, v, a, T2, friction, dt, l, K, random)
    '''Here we are using v(T1) to start HOT and T2 as the thermostat target temperature which is cold to make it tend towards equilibrium'''
    if step > 1000:
        T_samples.append(temp(v))
    '''After 1000 samples, the Temperature would have been equilibriated so we only consider samples after that'''

T_mean = np.mean(T_samples)
print("Equilibriated Mean Temperature: ", T_mean, "\nTarget Temperature: ", T2)

flag3 = True
if (np.abs(T_mean - T2) < 0.05*T2):    # within 5% of target
    print("\n\nThermostat Convergence: PASS\n\n")
else:
    print("\n\nThermostat Convergence: FAIL\n\n")
    flag3 = False

# g(r) normalization: random gas must give flat g(r) ~ 1
flag4 = True
frames = 50
bins = 80
gas = [random.uniform(0, l, size=(N, 2)) for _ in range(frames)]   # The Loop Length (50) is the number of frames
r, g = radial_distribution_function(gas, N, l, bins)
tail = g[5:]                                    # skip the small-r bins (unreliable)
g_mean = tail.mean()
print("g(r) mean (ideal gas):", g_mean)
if np.abs(g_mean - 1.0) < 0.05:
    print("g(r) Normalization: PASS")
else:
    print("g(r) Normalization: FAIL")
    flag4 = False


# g(r) structure: a crystal must be far more ordered than a liquid
flag5 = True
# Frame generator function for g(r) fitting.
''' Each frame contains the positions of all particles after every dt time interval '''
def sample_frames(Gamma, steps = 6000, every = 20):
    T = 1.0 / Gamma
    x = initial_positions(N, l, random)
    v = initial_velocities(N, T, random)
    a = net_force(x, l, K)
    frames = []
    for step in range(steps):
        x, v, a = thermostat(x, v, a, T, friction, dt, l, K, random)
        if step > steps//2 and step % every == 0:
            frames.append(x.copy())
    return frames

r_crystal, g_crystal = radial_distribution_function(sample_frames(400.0), N, l, bins)
r_liquid, g_liquid  = radial_distribution_function(sample_frames(2.0),   N, l, bins)
peak_crystal, peak_liquid = g_crystal.max(), g_liquid.max()
print("g(r) Peak Value for Crystal: ", peak_crystal, " g(r) Peak value for Liquid: ", peak_liquid)

if peak_crystal > 2.5 and peak_crystal > 1.5 * peak_liquid:
    print("g(r) Structure: PASS")
else:
    print("g(r) Structure: FAIL")
    flag5 = False

# Aggregate verdict (Program runs iff all flags are true)
all_flags = [flag1, flag2, flag3, flag4, flag5]

if all(all_flags):
    print("ALL CHECKS PASSED")
else:
    failed = [name for name, ok in zip(
        ["3rd Law", "Energy", "Thermostat", "g(r) norm", "g(r) structure", "Melting line"],
        all_flags) if not ok]
    print("FAILED CHECKS:", ", ".join(failed))