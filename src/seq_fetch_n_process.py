import subprocess

import biom
import qiime2


def fetch_sequences(n_threads, path_to_data):
    command = (
        f"../src/fetch_sequences.sh {path_to_data}/runids "
        f"{path_to_data}  {n_threads}"
    )
    subprocess.run(command, shell=True)


def cluster_sequences(n_threads, path_to_data):
    command = f"../src/cluster_sequences.sh {path_to_data} {n_threads}"
    subprocess.run(command, shell=True)


def filter_sequences(path_to_data, min_prevalence):
    command = f"../src/filter_sequences.sh {path_to_data} {min_prevalence}"
    subprocess.run(command, shell=True)


def rarefy_sequences_w_fixed_seed(path_to_otu, seed):
    # Load the QIIME2 artifact (FeatureTable[Frequency])
    table_artifact = qiime2.Artifact.load(path_to_otu)

    # Convert the artifact to a biom.Table
    table = table_artifact.view(biom.Table)

    # Use the subsample method to subsample 2000 sequences per sample with a
    # fixed seed
    table_subsampled = table.subsample(
        2000,
        axis="sample",
        by_id=False,
        with_replacement=False,
        seed=seed,
    )
    return table_subsampled
