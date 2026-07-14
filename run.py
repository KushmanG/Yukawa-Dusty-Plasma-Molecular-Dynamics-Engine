from test import run_validation
from sim import *


def parse_args():
    p = argparse.ArgumentParser(description= "Generating a 2D Dusty Plasma Dataset")
    p.add_argument("--skip-validation", action = "store_true")
    p.add_argument("--pilot", action = "store_true", help = "tiny 3x3x3 grid for a fast smoke test") #action = "store_true" makes it a boolean CLI arg

    return p.parse_args()

args = parse_args()

'''
Validation can be skipped for quick inspections on the same variables by:

        python3 run.py --skip-validation
'''
# If the function returns false, exit the file, it does not run.
if not args.skip_validation:
    if not run_validation():
        sys.exit("Validation Failed")

'''
DATASET GRID, 2 possibilities:
Dataset size = Number of gamma values x Number of kappa values x Seeds per gamma-kappa pair

1. Pilot dataset, small dataset of 3x3x3 = 27 samples for a quick overview of what the final sample
   would look like
2. Usable Dataset-1, 10x7x20 = 1400 data points

We use logarithmic spacing between adjacent gamma values and linear spacing between adjacent kappa values
'''
if args.pilot:
    gammas = np.logspace(np.log10(2), np.log10(1200), 3)    # 3 gamma values
    kappas = np.linspace(1, 3, 3)                           # 3 kappa values
    n_seeds = 3                                             # 3 seeds per gamma-kappa pair
else:
    gammas = np.logspace(np.log10(2), np.log10(1200), 10)   # 10 gamma values
    kappas = np.linspace(1, 3, 7)                           # 7 kappa values
    n_seeds = 20                                            # 20 seeds per gamma-kappa pair

n_samples = len(gammas) * len(kappas) * n_seeds
print(f"Grid: {len(gammas)} gammas x {len(kappas)} kappas x {n_seeds} seeds = {n_samples} samples")
print(f"  gammas = {np.round(gammas, 1)}")
print(f"  kappas = {np.round(kappas, 2)}")

# Deciding output directory and making them
out_dir = "pilot_data" if args.pilot else "data"
os.makedirs(out_dir, exist_ok = True)

'''
So each sample shall have a unique sample id for identification ofcourse but also each sample 
must have a unique starting point so that we can get unique spatial arrangements corresponding to 
each gamma-kappa pair, so we need different seeds for np.random.default_rng(), so we take the sample id
which is a universal incremental counter as the seed of every rng value we choose
'''

id = 0 # Initialized to 0
manifest_rows = [] # Array, will later be converted to a csv (manifest.csv holding all data of a sample)

N = 256                            
dt = 0.01
friction = 1.0
steps = 6000

# 1 gamma, 1 kappa and 1 seed describes one sample, so this triple loop iterates over all samples
'''
In each snapshot we need to basically get the final equilibriated snapshot's positions and plot image
and along with that we need to append all the metadata in the sample's directory and in the manifest.csv

So the steps go like:
1. Initialize temperature, box length and rng 
2. Equilibriate the sample 
2.5. Make a per sample directory
3. Save the positions in positions.npy
4. Save metadata.npy
5. Plot the final sample and save it as snapshot.png
*** 6. Close the plot to prevent data overload (Reminder)
7. Append data to manifest.csv
8. id++ 
'''

for gamma in gammas:
    for kappa in kappas:
        for s in range(n_seeds):
            # Step 1: Initializations
            T = 1.0/gamma
            random = np.random.default_rng(seed = id)
            l = L(N)

            # Step 2: Equilibriations
            x = initial_positions(N, l, random)
            v = initial_velocities(N, T, random)
            a = net_force(x, l, kappa)
            for step in range(steps):
                x, v, a = thermostat(x, v, a, T, friction, dt, l, kappa, random)

            # Step 2.5: Sample dir
            dir = os.path.join(out_dir, f"sample_{id:04d}")
            os.makedirs(dir, exist_ok = True)

            # Step 3: Saving positions
            np.save(os.path.join(dir, "positions.npy"), x)

            # Step 4: Saving metadata.
            # Current metadata schema: [Gamma, Kappa, N], editable
            np.save(os.path.join(dir, "metadata.npy"), np.array([gamma, kappa, N]))

            # Step 5: Save snapshot.png
            figure, axis = plt.subplots(figsize=(4, 4))
            axis.scatter(x[:, 0], x[:, 1], s=10)
            axis.set_aspect("equal") #Square Box
            axis.axis("off") # We are removing axes so that there is no extra stuff except the particles in a snapshots

            figure.savefig(os.path.join(dir, "snapshot.png"), dpi=80)

            # Step 6: Close the plot
            plt.close(figure)          

            # Step 7: Append it all to the global record i.e manifest.csv
            seed = id % n_seeds + 1
            manifest_rows.append((id, gamma, kappa, seed))
            print(f"[{id+1}/{n_samples}]\ngamma = {gamma}\nkappa = {kappa}") # Prints how many samples have been printed and their corresponding gamma kappa values
            ''' Like:
                    [2/27]
                        gamma = 2.0
                        kappa = 1.0
            '''

            # Step 8: Go to the next id
            id = id+1


'''
manifest.csv stores one row per sample, useful for indexing for train/val/test splits
'''
manifest_path = os.path.join(out_dir, "manifest.csv")
with open(manifest_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["sample_id", "gamma", "kappa", "seed"])   # header
    writer.writerows(manifest_rows)                            # all rows at once

print(f"Done: {id} samples written to {out_dir}/")
print(f"Manifest: {manifest_path}")
