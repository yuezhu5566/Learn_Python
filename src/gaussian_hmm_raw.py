import numpy as np
import pyBigWig
import os
import multiprocessing
from pyhhmm.gaussian import GaussianHMM



data_dir = '/projects/b1042/AmaralLab/liebe/Accessibility_coding/AccessibilityStates/data/ATAC_total_bw/'
output_dir = '/projects/b1042/AmaralLab/liebe/Accessibility_coding/AccessibilityStates/data/pyhhmm/raw'
os.makedirs(output_dir, exist_ok=True)

bw_files = [
    'ENCFF669VTY.ATACtotal.100bp.bw',  # rep1
    'ENCFF899WFS.ATACtotal.100bp.bw',  # rep2
    'ENCFF964ZKK.ATACtotal.100bp.bw'   # rep3
]

resolution = 100
chrom = 'chr10'
cl = 'A549'

COLOR_NAME_MAP = {
    0: 'Purple', 1: 'Blue', 2: 'Green', 3: 'Yellow',
    4: 'Orange', 5: 'Red', 6: 'Crimson'
}
RGB_COLOR_MAP = {
    0: '106,13,173', 1: '0,0,255', 2: '0,255,0', 3: '255,255,0',
    4: '255,165,0', 5: '255,0,0', 6: '153,0,0'
}

# === functions ===
def binned_mean(bw_path, chrom, resolution=100):
    """
    Get binned observation sequence from BigWig file. Average read intensity per bin
    """

    print(f"Reading data from: {os.path.basename(bw_path)}...", flush=True)

    bw = pyBigWig.open(bw_path)
    if chrom not in bw.chroms():
        print(f"    Chromosome '{chrom}' not found in {bw_path}. Skipping file.", flush=True)
        bw.close()
        return None

    chrom_len  = bw.chroms()[chrom]
    n_bins = int(np.ceil(chrom_len / resolution))

    vals = bw.stats(chrom, 0, chrom_len, nBins=n_bins) # gets mean by default
    arr = np.array([0.0 if (v is None or np.isnan(v)) else float(v) for v in vals], dtype=np.float64)
    bw.close()
    return arr.reshape(-1, 1)


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


def write_transitions_to_bed(states, chrom, resolution, filename):
    """
    Identifies and writes state transition points to a BED file.
    """

    with open(filename, 'w') as f:
        track_name = os.path.basename(filename).replace('.bed', '')
        f.write(f'track name="{track_name}" description="HMM State Transitions"\n')

        for i in range(1, len(states)):
            if states[i] != states[i-1]:
                start = i * resolution
                end = start + 1 # 1 bp site
                name = f"transition_{states[i-1]}_to_{states[i]}"
                f.write(f"{chrom}\t{start}\t{end}\t{name}\t0\t.\n")


def save_model_parameters(model, sorted_indices, color_name_map, log_likelihood_history, filename):
    """
    Saves all learned HMM parameters and training history to a text file.
    """

    with open(filename, 'w') as f:
        f.write("HMM Model Parameters (States sorted by mean signal)\n")
        f.write("="*50 + "\n\n")

        f.write("Priors (Initial State Probabilities):\n")
        sorted_pi = model.pi[sorted_indices]
        for i, p in enumerate(sorted_pi):
            f.write(f"  P(start in State {i}): {p:.4f}\n")
        f.write("\n" + "="*50 + "\n\n")

        f.write("State Emission Means:\n")
        means = model.means.flatten()
        for new_state, original_state in enumerate(sorted_indices):
            color_name = color_name_map.get(new_state, "Unknown")
            f.write(f"  State {new_state} ({color_name}): Mean signal = {means[original_state]:.4f}\n")
        f.write("\n" + "="*50 + "\n\n")

        f.write("State Emission Variances (Diagonal of Covariance Matrix):\n")
        variances = model.covars.flatten()
        sorted_variances = variances[sorted_indices]
        for i, var in enumerate(sorted_variances):
             f.write(f"  State {i}: Variance = {var:.4f}\n")
        f.write("\n" + "="*50 + "\n\n")

        f.write("Transition Matrix (Probabilities of switching from row state to column state):\n")
        sorted_A = model.A[sorted_indices, :][:, sorted_indices]
        n_states = sorted_A.shape[0]

        header = "From\\To\t" + "\t".join([f"State {i}" for i in range(n_states)])
        f.write(header + '\n')

        for i, row in enumerate(sorted_A):
            row_str = f"State {i}\t" + "\t".join([f"{p:.4f}" for p in row])
            f.write(row_str + '\n')
        f.write("\n" + "="*50 + "\n\n")

        f.write("Training Log-Likelihood History:\n")
        if log_likelihood_history:
            for i, ll in enumerate(log_likelihood_history):
                f.write(f"  Iteration {i}: {ll:.4f}\n")
        else:
            f.write("  No log-likelihood history was captured.\n")

    print(f"    Wrote all model parameters to {filename}", flush=True)


def process_replicate(bw_file):
    """
    Full analysis pipeline for a single replicate file.
    Reads the data then loops through the desired number of states, training
    a model and saving the results for each.
    """

    replicate_name = bw_file.split('.')[0]
    print(f"\n===== Processing {replicate_name} =====", flush=True)

    input_bw_path = os.path.join(data_dir, bw_file)
    observations = binned_mean(input_bw_path, chrom, resolution)

    if observations is None:
        print(f"===== Finished {replicate_name} (SKIPPED) =====", flush=True)
        return
    
    # ensure no infs/NaNs
    if not np.isfinite(observations).all():
        print(f"[ERROR] Non-finite values in observations for {replicate_name}", flush=True)
        print("    min, max:", np.nanmin(observations), np.nanmax(observations), flush=True)
        return

    observations = np.log1p(observations) # stabilize via log-transform coverage
    training_data = [observations]

    # inner loop for testing 2-7 states
    for n_states in range(7, 8):
        print(f"\n\t--- Training HMM with {n_states} states for {replicate_name} ---", flush=True)

        model = GaussianHMM(
            n_states=n_states,
            n_emissions=1,
            covariance_type='diagonal'
        )

        model, log_likelihood_history = model.train(training_data, n_iter=20)

        # ensure log_likelihood_history is always a list-like object
        log_likelihood_history = np.atleast_1d(log_likelihood_history)

        logL, state_seq_raw = model.decode(training_data)

        means = model.means.flatten()
        sorted_indices = np.argsort(means)
        state_map = {original_state: new_state for new_state, original_state in enumerate(sorted_indices)}

        state_seq_sorted = [state_map[s] for s in state_seq_raw[0]]

        print(f"  Final Log-Likelihood: {log_likelihood_history[-1]:.2f}", flush=True)


        # === save outputs ===
        # 1. save model parameters
        params_filename = f"{cl}.{chrom}.{replicate_name}.{n_states}states_parameters.txt"
        params_filepath = os.path.join(output_dir, params_filename)
        save_model_parameters(model, sorted_indices, COLOR_NAME_MAP, log_likelihood_history, params_filepath)

        # 2. save colored state regions
        regions_filename = f"{cl}.{chrom}.{replicate_name}.{n_states}states_regions.bed"
        regions_filepath = os.path.join(output_dir, regions_filename)
        hmm_states_to_bed(state_seq_sorted, chrom, resolution, regions_filepath, RGB_COLOR_MAP)

        # 3. save state transition points
        transitions_filename = f"{cl}.{chrom}.{replicate_name}.{n_states}states_transitions.bed"
        transitions_filepath = os.path.join(output_dir, transitions_filename)
        write_transitions_to_bed(state_seq_sorted, chrom, resolution, transitions_filepath)


    print(f"===== FINISHED {replicate_name} =====", flush=True)

# === execution ===
num_processes = min(len(bw_files), os.cpu_count())

if __name__ == '__main__':
    print(f"Processing {len(bw_files)} replicates using {num_processes} processes.")

    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(process_replicate, bw_files)

    print("\n===== All parallel analyses complete. =====")
