import pysam
import numpy as np
from sklearn.mixture import GaussianMixture
import json
import sys
import os



data_dir = '/projects/b1042/AmaralLab/liebe/Accessibility_coding/GlobalNetwork/data/A549/ATAC-seq/bam/pe'
output_dir = '/projects/b1042/AmaralLab/liebe/Accessibility_coding/AccessibilityStates/data/fragment_tracks'
os.makedirs(output_dir, exist_ok=True)

out_json = os.path.join(output_dir, 'A549_fragment_GMM_4comp.json')

bam_files = [
    'ENCFF607DTB.bam', #rep1
    'ENCFF701BDT.bam', #rep2
    'ENCFF616DYV.bam' #rep3
]

max_fragments = 1_000_000
min_len = 20
max_len = 1000
mapq_min = 30
n_components = 4


# === functions ===
def collect_fragment_lengths(bam_files):
    """
    """
    
    lengths = []
    
    for bam_path in bam_files:
        print(f"Collecting fragment lengths from {bam_path}", flush=True)
        
        bam = pysam.AlignmentFile(bam_path, "rb")
        cnt = 0
        
        for read in bam.fetch(until_eof=True):
            if not read.is_paired or read.is_unmapped or read.mate_is_unmapped:
                continue
            if read.is_secondary or read.is_supplementary:
                continue
            if read.mapping_quality < mapq_min:
                continue
                
            tlen = abs(read.template_length)
            if tlen < min_len or tlen > max_len:
                continue
            lengths.append(tlen)
            
            cnt += 1
            if cnt >= max_fragments:
                break
                
        bam.close()
        
        print(f"\tCollected {cnt} fragments from {bam_path}", flush=True)
        
    return np.array(lengths, dtype=float)


def fit_gmm(lengths, n_components=4):
    """
    """
    
    print(f"Fitting GMM with {n_components} components on {len(lengths)} fragments", flush=True)
    
    X = lengths.reshape(-1, 1)
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=0,
        max_iter=100
    )
    gmm.fit(X)
    
    print('Means:', gmm.means_.flatten())
    print('Variances:', np.array([c[0,0] for c in gmm.covariances_]))
    print('Weights:', gmm.weights_)
    
    return gmm


def save_gmm(gmm, out_json):
    """"""
    
    params = {
        'n_components': int(gmm.n_components),
        'weights': gmm.weights_.tolist(),
        'means': gmm.means_.flatten().tolist(),
        'covars': [float(c[0,0]) for c in gmm.covariances_]
    }
    
    with open(out_json, 'w') as f:
        json.dump(params, f, indent=2)
        
    print(f"Saved GMM parameters to {out_json}")

    
# === execution ===
if __name__ == "__main__":
    #get all bams
    bam_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".bam")
    ])

    if len(bam_files) == 0:
        print('No BAM files found.')
        exit(1)

    print('BAMs found:')
    for b in bam_files:
        print("   ", b)

    # collect fragment lengths
    lengths = collect_fragment_lengths(bam_files)

    if len(lengths) == 0:
        print('No fragments found for GMM fitting.')
        exit(1)

    # fit GMM
    gmm = fit_gmm(lengths, n_components=n_components)

    # save output
    save_gmm(gmm, out_json)

    print("Done.")

