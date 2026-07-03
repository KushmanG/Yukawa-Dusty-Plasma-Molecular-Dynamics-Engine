'''
This single file is meant to be READ as much as RUN

It teaches the implementation of classical molecular dynamics (MD) for 2D dusty plasma. If you have never written an MD code before, you
can learn it here       :)

##### PHYSICS ######
1. We simulate N point particles confined to a square box of side L with PERIODIC
   boundary conditions i.e:
    The box tiles space, so a particle leaving the right edge re-enters on the left

2. The particles repel each other through a screened Coulomb (Yukawa) interaction

3. We work in REDUCED UNITS so that no SI constants ever appear:
        Particle mass            m   = 1
        Wigner-Seitz radius      a   = 1   (the natural length scale, see below)
        Energy scale             eps = 1
        Boltzmann constant       k_B = 1

4. In these units the pair potential and the magnitude of the (repulsive) force
   between two particles separated by distance r are

    U(r) = (exp(-rK))/r         
    F(r) = -dU/dr = (rK+1)exp(-rK)/r^2
    [K is Kappa i.e the Screening Parameter]

    --> The force vector on particle i due to particle j is ALONG the line joining them

5. The 2 parameters that define a state point are:

    1. Coupling Parameter GAMMA (G) = Interaction Energy/Thermal Energy

           Big G  -> Cold and Ordered  -> Triangular Crystal Behaviour
           Small G  -> Hot and Disordered -> FLuid Behaviour

           *** We set the temperature directly from it:  T = 1 / Gamma.

    2. Screening Parameter KAPPA (K) = Avg-Inter-Dust Distance/Debye Length

        Big K --> Strong Screening, Dust particles barely "feel" each other --> Gaseous Behaviour
        Small K --> Weak Screening, Strong influence of particles on e/o --> Order Lattice Behaviour/"Plasma Crystal" 

6. Density/Box Size:
     In 2D the Wigner-Seitz radius a"" is defined as "There exists only one particle in a region of radius a"
     pi * a**2 = 1 / n, where n = N / L**2 is the number density. 
     
     With a = 1 the area per particle is exactly pi, so the
     TOTAL area is N * pi and the box side is fixed:

        L = sqrt(N * pi)

    *** The density is pinned by our choice of units.

    

    ##### WHAT THIS PROGRAM DOES ######
    1. Simulates a strongly coupled case and a weakly coupled case with a Langevin Thermostat holding each at its target T = 1/Gamma.

    2. Runs a Thermostat-OFF (NVE) check on each, reporting the relative drift of the total energy to test forces and timestep validity 

    3. Saves ONE multi-panel figure to the "./figures/" dir 
    One snapshot .png file contains:
        1. Crystal snapshot 
        2. Liquid snapshot
        3. The pair-correlation function g(r) of both 
        4. Temperature vs Time curve showing equilibration to the target.

    4. Prints a short validation report in the CLI (Terminal)

    ##### HOW TO RUN TS ##### 
    Check README.md    :D
'''

'''imports.py has all the necessary libraries'''
from imports import * 

''' N is an integer L is a float'''
def L(N):
    return np.sqrt(N * np.pi)

''' 
Now we are using a function for L here instead of a global variable cz L IS infact a FUNCTION of the
Number of particles N, and for the sake of the modularity of the code I don't wanna take it as an
input by using "N = input("Enter Number of particles: ")
'''

def initial_positions (N, L, random: np.random.Generator, jitter = 0.3):
    #Parameter Intialization for the Triangular Latti
    spacing = np.sqrt(3)/2
    row_count = int(np.ceil(np.sqrt(N/spacing)))
    clm_count = int(np.ceil(N/row_count))

    dx = L/clm_count  # clm spacing
    dy = L/row_count  #row spacing

    # Initializing a General grid
    clms = np.arange(clm_count)  # makes an array [0,1,2,3,4....clm_count]
    rows = np.arange(row_count)  # makes an array [0,1,2,3,4....row_count]

    # Turning it into a Triangular Lattice
    c,r = np.meshgrid(clms,rows)
    ''' Legacy Code
    if (r%2 == 0):
        x = c*dx
    else:
        x = c*dx + 0.5*dx
    Same thing for np has a different syntax''' 
    x = np.where(r%2 == 0, c*dx, (c+0.5)*dx)
    y = r*dy
    
    
    # "sites" is an (N,2) matrix with each row containing coordinates to one of the particles
    sites = np.column_stack([x.ravel(), y.ravel()])

    # Number of rows = len(sites) = No. of particles since there is one row per particle
    generated_particle_count = len(sites)
    gpc = generated_particle_count # I aint writing that long name every time
    # Bad formatting lol mb, but I like it this way (sorry for autism)

    # If extra lattice sites have been generated
    if gpc > N:
        keep = random.choice(gpc, size = N, replace = False) #Ranomly keep some of them
        sites = sites[keep]
        sites += jitter * dx * random.standard_normal(sites.shape)
    return sites%L

def temp(v):
    '''
    Total Kinetic Energy = 0.5 * sum(v*v) 
    KE of each particle is 0.5 * k_B * T * Degrees of Freedom
    For monoatomic particle in 2D space there are only 2 Degrees of Freedom --> x and y axes
    So KE/particle = 2 * 0.5 * k_B * T = k_B*T 
    and k_B as we decided is 1 so KE/particle = T
    and there are N particles in the system so Total KE = NT = 0.5 * (v*v)

    Therefore, T(v) i.e Temperature as a function of velocity = sum(v*v)/2N

    We need this because when deriving velocities randomly into an array from the gaussian and substracting 
    the drift from it MAY not give the Temperature we needed as per the thermostat, so we normalize the 
    verlocity using this function
    '''
    N = len(v) #No. of particles = length of the velocity array cz each particle has one velocity :D
    return np.sum(v*v)/(2*N)

def initial_velocities(N, T, random: np.random.Generator):
    std_deviation = np.sqrt(T)
    v = std_deviation * random.standard_normal((N,2)) #Since all parameters are 1, the distribution is essentially just a gaussian
    v -= v.mean(axis = 0) #Since v_av = 0, we had to normalize each velocity, by removing the drift
    v *= np.sqrt(T/temp(v))
    return v

'''
Commit #1 and tests Completed:
Till now I have written python to initialise particle positions and velocities
To test it:
1. Checked if all particles are in bound:
    random = np.random.default_rng(seed = 42)
    N = 64 
    box = L(N)
    pos = initial_positions(N,box,random)
    print(pos.shape, pos.min(), pos.max(), box)
    This returned the values:
    (64, 2) 0.0843012857910879 14.137241752271624 14.179630807244127
    So that checks (Checked multiple times, pasted one result)

2. Plotted a scatterplot for N = 256 using these functions:
    plt.scatter(pos[:, 0], pos[:, 1])
    plt.gca().set_aspect("equal"); plt.show()

    And we got a staggered bunch of particles that seemed to roughly have
    a hexagonal arrangement but with the jitter

Here on: Force/Potential/Langevin Step/Pair Correlation etc...
'''

def net_force(positions, L, kappa):
    '''
    The interaction force between two particles is:
        F vector = [{(Kr+1)exp(-Kr)}/r^3]*(r vector)

    So we need arrays of:
        1. Magnitude of distance between all pair of particles (NxN) 
        2. Displacement vector between two particles

    Now the r vector [2] can easily be calculated by using vector addition law, just substract the position vectors (stored in the "positions" array)
    of the two particles

    For Magnitude, |r vector| = sqrt(dot product of r vector with itself) 
    So the array of all distances is dist = np.sqrt(np.sum(disp*disp), axis = -1))

    So the final force of interaction is just disp * (np.exp(-K*r) * (K*r + 1)/(dist**3))

    Two important factors to consider:
    1. in the array "dist", 0 should be replaced with a placeholder (prefferably 1) for i = j to prevent 1/0 indeterminate form
    2. Minimum Image Convention Normalization (Added since the simulation was fauly on prior tests)

    So this Minimum Image convention is required because we want to study a portion, a snapshot of the entire plasma, not the entire plasma
    so the boundaries are only pseudo boundaries not real walls, particles don't bounce off them, if one particle crosses that boundary
    another particle must enter from the other side
    For this we Just substract  Box Length * np.round(original displacement vector/Box Length) from the original displacement vector
    ''' 
    K = kappa #Cz I like it this way
    disp = positions[:, None, :] - positions[None, :, :]
    disp -= L * np.round(disp/L)  #minimum image convention

    dist = np.sqrt(np.sum(disp*disp, axis = -1))
    np.fill_diagonal(dist, 1.0) #To prevent indet form

    spf = np.exp(-K*dist) * (K*dist + 1)/(dist*dist*dist)

    return np.sum(disp * spf[:, :, None], axis = 1)

def potential_energy(positions, l, kappa):
    '''
    Total Yukawa potential energy: U = sum over pairs i<j of exp(-K r)/r.
    Same disp / minimum-image / distance machinery as net_force, but each
    pair contributes a scalar energy, not a force vector.
    '''
    K = kappa
    disp = positions[:, None, :] - positions[None, :, :]
    disp -= l * np.round(disp / l)
    dist = np.sqrt(np.sum(disp*disp, axis=-1))
    np.fill_diagonal(dist, 1.0)              # dodge 1/0 on the diagonal

    U = np.exp(-K*dist) / dist               # U(r) for every pair, (N, N)
    np.fill_diagonal(U, 0.0)                 # self-pairs contribute nothing
    return 0.5 * np.sum(U)  #Multiplied by 0.5 cz each U[i,j] is counted twice, once as U[i,j] then as U[j,i]


'''
So now we have initialized the velocity and position and defined the interaction between them
But all this calculates things in a certain instant, single snapshot
We need to make it move, i.e periodically update everything as per equations of motion
So after a dt time interval:
    v_new = v + 0.5*a*dt
    x_new = (x + v*dt) % L (Only after the velocity is updates)
    a_new = net_force(x_new, L, kappa) {Since m = 1, a = F}

    v_new = v + 0.5*a*dt --> Update again, Loop n times --> ndt = Total time of simulation
'''

def dynamize(x, v, a, l, kappa, dt):
    v = v + 0.5*a*dt
    x = (x + v*dt) % l
    a = net_force(x, l, kappa)
    v += 0.5*a*dt

    return x, v, a

'''
Commit #2 and tests completed:
The simulation was static till v1, now it is dynamic

Future versions will mainly have:
1. Output formatting
2. In the tests (using run.py) turns out temperature is changing, so we need a thermostat [DONE]
3. g(r) histograms  [DONE]
4. Energy conservation and Potential Energy checks [DONE]
5. Most important check:
    Melting-line check:
    Sweep Γ at fixed K, find where it crystallizes, confirm it matches the published Hartmann/Donkó 2D Yukawa phase diagram.
6. main() function
7. Dataset generation
8. Start working on the ML model after validating the simulation
'''

'''
Building the Thermostat:
We will be using the Langevin Thermostat.
Reason:
The Langevin thermostat is the standard way to model dusty plasma dynamics because real dust grains sit in a neutral 
gas background — collisions with gas atoms produce both a drag force and random thermal kicks. This is different 
from thermostats like Nosé-Hoover, which are mathematical constructs; the Langevin term here has direct physical 
meaning (Epstein gas friction)


'''

def thermostat(x, v, a, T, friction, dt, l, K):
    c1 = np.exp(-friction * (dt/2)) 
# velocity transform happens in two steps, first in the first half time interval the in the second half time interval
    c2 = np.sqrt(T * (1 - c1*c1))
    eta1 = np.random.standard_normal(v.shape)

    v = c1*v + c2*eta1

    x, v, a = dynamize(x, v, a, l, K, dt)

    eta2 = np.random.standard_normal(v.shape)
    v = c1*v + c2*eta2

    return x, v, a


#Standard RDF implementation, copy pasted
def radial_distribution_function(snapshots, N, l, n_bins=150):
    '''
    Radial pair correlation g(r), averaged over a list of position snapshots.
    Histogram every minimum-image pair distance, then divide by the ideal-gas
    expectation so g -> 1 when there is no structure. Trust only r < l/2.
    '''
    r_max = l / 2.0
    edges = np.linspace(0.0, r_max, n_bins + 1)     # bin boundaries
    counts = np.zeros(n_bins)
    off_diag = ~np.eye(N, dtype=bool)               # mask that drops self-pairs (i==j)

    for x in snapshots:
        disp = x[:, None, :] - x[None, :, :]
        disp -= l * np.round(disp / l)
        dist = np.sqrt(np.sum(disp * disp, axis=-1))
        r = dist[off_diag]                          # all pair distances, self-pairs removed
        counts += np.histogram(r[r < r_max], bins=edges)[0]

    n = N / (l * l)                                 # number density
    annulus = np.pi * (edges[1:]**2 - edges[:-1]**2)   # area of each ring
    g = counts / (len(snapshots) * N * n * annulus)    # divide out the geometry
    centers = 0.5 * (edges[1:] + edges[:-1])        # bin midpoints (for plotting)
    return centers, g

'''
This function has some usage constraints on some variables so that the PASS/FAIL verdict it gives is actually reasonable:

1. frames ≥ 50            
    Larger number of frames smoothens the g(r) output
    This variable can be edited by changing the loop length in line 87 of test.py

2. sample_every ≈ 20 steps   (not consecutive)
    Independent frames only; back-to-back frames are near-duplicates.
    Test function yet to be made
3. sample only step > steps/2   
    Don't average in the un-equilibrated transient.
    Test function yet to be made

4. r < L/2            
    Not a variable the function caps at l/2. Just don't read past it.

5. N ≥ 256                 
    test.py line 8  ->  N = 256

6. n_bins ≈ 80             
    test.py line 88  
7. ignore bins[0:5]        
    (Small-r region is unreliable)
    test.py line 89  ->  tail = g[5:]   (change the slice index to change the ignored bins)

    NOTE:
    All of these parameters are YET TO BE decided by optimization using hit and trial methods or BACKED BY ANY LITERATURE
'''

'''
Commit#3:
Test functions have been made, Thermostat and g(r) histogram generator has been made, requirements.txt 
has been updated and a concrete validation ladder architecture is under process

Following tasks:
1. Make a testing_parameters.txt file to support why any parameter was taken that was, either backed by some
   literature or by showing my hit and trial optimization and findings
2. Finish melting line check and structure test
3. Finalize test output format
4. Work on final output format and CLI
'''