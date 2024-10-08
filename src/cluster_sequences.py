"""
Module to cluster sequences of each study
"""

import argparse
import os
import subprocess

import pandas as pd
import qiime2 as q2
from qiime2.plugins import vsearch


def parse_arguments():
    """parse CLI arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--path2md", type=str, required=True)
    parser.add_argument("--path2seq", type=str, required=True)
    parser.add_argument("--threads", default=6, type=int)
    return parser.parse_args()


def save_artifact(artifact, path, txt4print):
    """Save artifact in path while printing text info"""
    artifact.save(path)
    print(f"Saved {txt4print}: {path}")


def load_reference_seqs(path_to_data, filename):
    """Loads or creates SILVA reference sequences and related files"""
    path2ref = os.path.join(path_to_data, filename)
    if not os.path.isfile(path2ref):
        command = f"../src/get_silva_reads_n_classifier.sh {path_to_data} 6"
        subprocess.run(command, shell=True)

    return q2.Artifact.load(path2ref)


def cluster_sequences_one_study(study_cohort, path2seq, threads):
    """Cluster sequences of one study"""
    print(f"Clustering: {study_cohort}...")
    path_otu_table = os.path.join(path2seq, f"otu_table_{study_cohort}")

    if os.path.isfile(path_otu_table):
        print(f"Clustered otu table already exists in: {path_otu_table}.")
    else:
        # read ASV sequence table
        asv = q2.Artifact.load(os.path.join(path2seq, f"asv_{study_cohort}.qza"))
        # read representative sequences table
        repseq = q2.Artifact.load(os.path.join(path2seq, f"repseq_{study_cohort}.qza"))
        # read reference sequences
        ref = load_reference_seqs(
            path2seq, "silva-138.1-ssu-nr99-seqs-515f-806r-uniq.qza"
        )

        # closed-reference clustering
        (
            otu_table,
            otu_seq,
            unmatched_seq,
        ) = vsearch.actions.cluster_features_closed_reference(
            sequences=repseq,
            table=asv,
            reference_sequences=ref,
            perc_identity=0.97,
            threads=threads,
        )
        # save outputs
        save_artifact(otu_table, path_otu_table, "otu table")

        save_artifact(
            otu_seq,
            os.path.join(path2seq, f"otu_seq_{study_cohort}.qza"),
            "otu sequences",
        )
        save_artifact(
            unmatched_seq,
            os.path.join(path2seq, f"otu_umn_{study_cohort}.qza"),
            "unmatched sequences",
        )


def cluster_sequences(path2md, path2seq, threads):
    """Cluster denoised sequences saved in path2seq of all study
    subcohorts included in path2md"""
    # read metadata of all studies
    df_md = pd.read_csv(path2md, sep="\t", index_col=0, dtype="str")

    for study_cohort in df_md["study_cohort_name"].unique():
        cluster_sequences_one_study(study_cohort, path2seq, threads)


if __name__ == "__main__":
    args = parse_arguments()
    print(f"Clustering sequences of studies present in metadata {args.path2md}")

    cluster_sequences(
        path2md=args.path2md,
        path2seq=args.path2seq,
        threads=args.threads,
    )
