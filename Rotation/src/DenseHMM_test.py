import os
import numpy as np
import csv
import torch
import pyBigWig
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Normal

data_dir = '/projects/b1042/AmaralLab/liebe/Accessibility_coding/GlobalNetwork/data/A549/ATAC-seq/bigwig/'
output_dir = '/projects/b1042/AmaralLab/liebe/Accessibility_coding/AccessibilityStates/data/pomegranate'
os.makedirs(output_dir, exist_ok=True)

bw_file = 'ENCFF622SLI.bigWig'
n_states = 2

resolution = 100
chrom = 'chr14'
cl = 'A549'

COLOR_NAME_MAP = {
    0: 'Purple', 1: 'Blue', 2: 'Green', 3: 'Yellow',
    4: 'Orange', 5: 'Red', 6: 'Crimson'
}
RGB_COLOR_MAP = {
    0: '106,13,173', 1: '0,0,255', 2: '0,255,0', 3: '255,255,0',
    4: '255,165,0', 5: '255,0,0', 6: '153,0,0'
}


# === basic functions ===
def binned_mean(bw_path, chrom, resolution=100):
    """
    Get binned observation sequence from BigWig file. Average read intensity per bin
    """

    print(f"Reading data from: {os.path.basename(bw_path)}", flush=True)

    bw = pyBigWig.open(bw_path)
    if chrom not in bw.chroms():
        print(f"\tChromosome '{chrom}' not found in {bw_path}. Skipping file.", flush=True)
        bw.close()
        return None

    chrom_len  = bw.chroms()[chrom]
    n_bins = int(np.ceil(chrom_len / resolution))

    vals = bw.stats(chrom, 0, chrom_len, nBins=n_bins) # gets mean by default
    arr = np.array([0.0 if (v is None or np.isnan(v)) else float(v) for v in vals], dtype=np.float32)
    bw.close()
    return arr


def hmm_states_to_bed(states, chrom, resolution, filename, color_map):
    """
    Writes predicted HMM state regions to a BED file with RGB colors for IGV.
    """

    with open(filename, 'w') as f:
        # enable color in igv
        track_name = os.path.basename(filename).replace('.bed', '')
        f.write(f'track name="{track_name}" description="HMM States" itemRgb="On"\n')

        for i, state in enumerate(states):
            start = i * resolution
            end = start + resolution
            color = color_map.get(state, '255,255,255') # default to white if state not in map
            # format: chrom, start, end, name, score, strand, thickStart, thickEnd, itemRgb
            f.write(f"{chrom}\t{start}\t{end}\tstate_{state}\t0\t.\t{start}\t{end}\t{color}\n")

    print(f"\tWrote colored state annotations to {filename}", flush=True)
    

# === paramaeter estimation ===
def estimate_hmm_params(x, y, n_states, eps=1e-8):
    """
    Estimate HMM parameters from a decoded state path y and observations x.
    Everything is on the transformed scale of x.
    
    Output:
      means, variances, oriors, transition matrix, logL 
    """
    
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=int).ravel()
    T = x.size

    means = np.zeros(n_states, dtype=float)
    variances = np.zeros(n_states, dtype=float)

    # per-state means/vars
    for k in range(n_states):
        mask = (y == k)
        if mask.any():
            vals = x[mask]
            m = float(vals.mean())
            v = float(vals.var(ddof=1)) if vals.size > 1 else 0.0
            if v <= 0 or not np.isfinite(v):
                v = 1e-6
        else:
            # state not used; assign something benign
            m = 0.0
            v = 1.0
        means[k] = m
        variances[k] = v

    # initial dist
    pi = np.zeros(n_states, dtype=float)
    pi[y[0]] = 1.0

    # transition counts
    counts = np.zeros((n_states, n_states), dtype=float)
    for t in range(T - 1):
        i = y[t]
        j = y[t + 1]
        counts[i, j] += 1.0

    # convert counts -> probabilities with epsilon smoothing
    A = counts + eps
    row_sums = A.sum(axis=1, keepdims=True)
    A = A / row_sums

    # logL ≈ sum_t log N(x_t | mu_{y_t}, var_{y_t}) + sum_t log A_{y_t, y_{t+1}}
    means_per_t = means[y]
    vars_per_t = variances[y]

    # gaussian log pdf
    log_norm_const = -0.5 * np.log(2.0 * np.pi * vars_per_t)
    log_quad = -0.5 * (x - means_per_t) ** 2 / vars_per_t
    log_emissions = log_norm_const + log_quad
    logL_emissions = float(log_emissions.sum())

    trans_probs = A[y[:-1], y[1:]]
    log_trans = np.log(trans_probs)
    logL_trans = float(log_trans.sum())

    logL = logL_emissions + logL_trans

    return means, variances, pi, A, logL


def save_model_parameters(means, variances, pi, A, logL, out_path):
    """
    Output a text file containing the following paramteres:
      - final log-likelihood on X
      - AIC/BIC (approximate k)
      - per-state mean/variance
      - transition matrix
      - inferred initial probs
    """
    
    n_states = len(means)
    T_len = int(sum(1 for _ in pi))  # not used directly; will replace below
    T_len = int(A.sum()) + 1  # ~ #steps ≈ T-1; good enough for AIC/BIC

    # #params: (N-1) priors + 2N emissions (mean+var) + N(N-1) transitions
    N = n_states
    k = (N - 1) + 2 * N + N * (N - 1)
    aic = 2 * k - 2 * logL
    bic = k * np.log(T_len) - 2 * logL

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f, delimiter='\t')

        # metrics
        w.writerow(["Metric", "Value"])
        w.writerow(["Log Likelihood (path-based)", f"{logL:.4f}"])
        w.writerow(["AIC", f"{aic:.4f}"])
        w.writerow(["BIC", f"{bic:.4f}"])
        w.writerow([])

        # emissions
        w.writerow(["Section", "State Index", "Mean (std. log1p)", "Variance"])
        for i in range(N):
            w.writerow(["Emissions", i, f"{means[i]:.4f}", f"{variances[i]:.6f}"])
        w.writerow([])

        # priors
        w.writerow(["Section", "Priors"])
        w.writerow(["pi"] + [f"{p:.4f}" for p in pi])
        w.writerow([])

        # transitions
        w.writerow(["Section", "Transition Matrix Rows"])
        header = ["From\\To"] + [f"State {i}" for i in range(N)]
        w.writerow(header)
        for i in range(N):
            w.writerow([f"State {i}"] + [f"{p:.4f}" for p in A[i]])

    print(f"\tWrote empirical parameter file to {out_path}", flush=True)

            

bw_path = os.path.join(data_dir, bw_file)

# load and stabilize data
observations = binned_mean(bw_path, chrom, resolution)
if observations is None:
    raise SystemExit("No data returned from binned_mean; aborting.")

x = np.log1p(observations.astype(np.float32))
med = float(np.median(x))
mad = float(np.median(np.abs(x - med))) or 1.0
x = (x - med)/mad

# shape (1, T, 1) for densehmm
X = torch.from_numpy(x.reshape(1, -1, 1))

print(f"\n\t === Training with {n_states} states ===", flush=True)

# build and fit hmm - give diff means so they start in diff places
init_means = np.linspace(-1.0, 1.0, n_states, dtype=np.float32)
dists = [Normal(means=[float(m)], covs=[[1.0]]) for m in init_means]


model = DenseHMM(distributions=dists, 
                 verbose=True, 
                 max_iter=20, 
                 random_state=42)
model.fit(X)

# decode state seq
y_raw = model.predict(X)[0].detach().cpu().numpy()

# empirically estimate parameters from the decoded path
means_emp, vars_emp, pi_emp, A_emp, logL_emp = estimate_hmm_params(
    x, y_raw, n_states
)

# sort states by empirical mean
sort_idx = np.argsort(means_emp)
remap = {orig: new for new, orig in enumerate(sort_idx)}
y_sorted = np.vectorize(remap.get)(y_raw)

# also sort params for reporting
means_sorted = means_emp[sort_idx]
vars_sorted = vars_emp[sort_idx]
pi_sorted = pi_emp[sort_idx]
A_sorted = A_emp[sort_idx][:, sort_idx]

# save BED states (using sorted labels)
bed_path = os.path.join(
    output_dir, f"{cl}.{chrom}.{os.path.splitext(bw_file)[0]}.{n_states}states.bed"
)
hmm_states_to_bed(y_sorted, chrom, resolution, bed_path, RGB_COLOR_MAP)
print(f"Wrote {bed_path}", flush=True)

# save empirical parameter file
param_path = os.path.join(
    output_dir, f"{cl}.{chrom}.{n_states}states.params.tsv"
)
save_model_parameters(means_sorted, vars_sorted, pi_sorted, A_sorted, logL_emp, param_path)
print(f"Wrote {param_path}", flush=True)
