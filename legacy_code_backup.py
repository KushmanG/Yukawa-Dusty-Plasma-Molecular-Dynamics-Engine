from test import *

#Melting line check test.py

'''
# ---------- Melting-line check: sweep Gamma at fixed kappa, locate the transition ----------
# Literature anchor -- Hartmann, Kalman, Donko, Kutasi, PRE 72, 026409 (2005):
#     Gamma_m(kappa) = 131 / (1 - 0.388 k^2 + 0.138 k^3 - 0.0138 k^4)
#     at kappa = 1  ->  Gamma_m ~ 178
# We sweep Gamma across that value and check OUR sim melts near the same place.
'''
flag6 = True
psi6_cutoff    = 2.5     # first minimum of g(r): captures exactly the 6 nearest-neighbour shell
psi6_threshold = 0.6     # midway between the liquid floor (~0.4) and the crystal plateau (~0.9)

gamma_sweep = [50, 100, 150, 200, 250, 350]
psi6_values = []
for G in gamma_sweep:
    frames = sample_frames(float(G))
    psi = np.mean([melting_line_check(f, l, psi6_cutoff) for f in frames])  # average over frames
    psi6_values.append(psi)
    print("Gamma = %4d   <|psi6|> = %.3f" % (G, psi))

psi6_values = np.array(psi6_values)
gamma_sweep = np.array(gamma_sweep)

# OUR melting point = where psi6 first crosses the threshold (linear interpolation between the
# bracketing sweep points). Sweep runs cold->hot? No -- Gamma increases = colder = MORE ordered,
# so psi6 rises with Gamma; we look for the upward crossing.
gamma_m = np.nan
for i in range(1, len(gamma_sweep)):
    if psi6_values[i-1] < psi6_threshold <= psi6_values[i]:
        g0, g1 = gamma_sweep[i-1], gamma_sweep[i]
        p0, p1 = psi6_values[i-1], psi6_values[i]
        gamma_m = g0 + (psi6_threshold - p0) * (g1 - g0) / (p1 - p0)
        break

gamma_lit = 131.0 / (1 - 0.388*K**2 + 0.138*K**3 - 0.0138*K**4)
print("\nDetected Gamma_m ~ %.0f    Literature Gamma_m(k=%.1f) ~ %.0f" % (gamma_m, K, gamma_lit))

if np.isfinite(gamma_m) and np.abs(gamma_m - gamma_lit) < 0.4 * gamma_lit:   # within 40%
    print("\n\nMelting-Line Check: PASS\n\n")
else:
    print("\n\nMelting-Line Check: FAIL\n\n")
    flag6 = False


# Melting line check simulator function

def melting_line_check(positions, l, r_cutoff):
    disp = positions[:, None, :] - positions[None, :, :]
    disp -= l * np.round(disp / l)
    dist = np.sqrt(np.sum(disp*disp, axis=-1))
    
    angles = np.arctan2(disp[:,:,1], disp[:,:,0])
    c = np.cos(6*angles)
    s = np.sin(6*angles)

    neighbour = (dist < r_cutoff) & (dist > 1e-9)
    neighbour_count = np.maximum(np.sum(neighbour, axis = 1),1) 
    ''' If some particle has 0 neighbours in its  vicinity (r_cutoff) then we don't want division with zero so np.maximum(... , 1) is a way to prevent it'''

    C = np.sum(c * neighbour, axis = 1)/neighbour_count   # per-particle Re(psi6_i)
    S = np.sum(s * neighbour, axis = 1)/neighbour_count   # per-particle Im(psi6_i)

    '''
    PER-PARTICLE order parameter  <|psi6_i|>  :  take the magnitude FIRST (per particle),
    THEN average. This is the standard melting diagnostic.

    The alternative -- np.sqrt(C.mean()**2 + S.mean()**2) -- is the GLOBAL |<psi6>|, which
    averages the complex phasors before taking the magnitude. That version cancels to ~0
    whenever the crystal splits into domains pointing different ways (which happens here
    because L/b is non-integer, so the lattice can't perfectly tile the periodic box).
    Per-particle avoids that false-melting artifact: liquid floor ~0.4, crystal ~0.9.
    '''
    psi6_i = np.sqrt(C*C + S*S)   # each particle's own order, in [0, 1]
    return psi6_i.mean()
    
    '''
    Legacy Code:
    c_sum = np.sum(c * neighbour)
    s_sum = np.sum(s * neighbour)

    return np.sqrt(c_sum * c_sum + s_sum * s_sum)

    Claude says ts needs normalization:
    THE GRADES ANALOGY:
    Two students take quizzes:
    - Student A answers 8 questions, gets all 8 right â raw score 8.
    - Student B answers 4 questions, gets all 4 right â raw score 4.

    Both are perfect â equally good. But their raw scores (8 vs 4) look different only because they answered different numbers of questions.

    If you want "the average performance," you do not add raw scores (8 + 4 = 12, then... 12 of what?). You first convert each to a fraction:
    - A: 8/8 = 1.0
    - B: 4/4 = 1.0

    Then average: (1.0 + 1.0)/2 = 1.0. Correct â both perfect, average is perfect.

    If you'd skipped the divide and just summed raw scores, Student A (more questions) would dominate the total, even though they're no better than B.

    Now map it to Ïâ

    - "Questions answered" = number of neighbors = counts[i].
    - "Raw score" = sum of that particle's neighbor cosines = np.sum(c*neighbour, axis=1)[i].
    - "Fraction correct (0â1)" = raw score Ã· counts = c_i â this is the per-particle normalization.
    - "Class average" = .mean() over all particles.


    A particle with 8 neighbors can pile up a bigger raw cosine sum than one with 4 neighbors — just because it has more neighbors, 
    not because it's more ordered. Dividing by counts cancels that out: each particle reports its own order on a fair 0-to-1 scale, 
    then you average those.
    '''


# Melting line check on standby - needs more research