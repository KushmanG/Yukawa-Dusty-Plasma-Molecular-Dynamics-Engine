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
Commit #2 and tests Completed:
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