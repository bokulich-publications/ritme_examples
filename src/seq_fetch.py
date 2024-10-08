import subprocess


def fetch_sequences(n_threads, path_to_data):
    command = (
        f"../src/fetch_sequences.sh {path_to_data}/runids "
        f"{path_to_data}  {n_threads}"
    )
    subprocess.run(command, shell=True)
