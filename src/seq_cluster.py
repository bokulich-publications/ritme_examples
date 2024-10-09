import subprocess


def cluster_sequences(n_threads, path_to_data):
    command = f"../src/cluster_sequences.sh {path_to_data} {n_threads}"
    subprocess.run(command, shell=True)
