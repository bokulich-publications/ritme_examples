"""
Module to denoise sequences of each study
"""
# todo: combine this with e_trim_sequences.py

import argparse
import json
import os

import pandas as pd
import qiime2 as q2
from qiime2.plugins import dada2


def parse_arguments():
    """parse CLI arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--path2md", type=str, required=True)
    parser.add_argument("--path2allseq", type=str, required=True)
    parser.add_argument("--path2trunc_len", type=str, required=True)
    parser.add_argument("--threads", default=6, type=int)
    return parser.parse_args()


def read_files_all_studies(path2md, path2trunc_len):
    """Read files with information on all studies:
    metadata and truncation length"""
    # read metadata of all studies
    df_md = pd.read_csv(path2md, sep="\t", index_col=0, dtype="str")
    # read manually identified truncation lengths for all studies
    with open(path2trunc_len, encoding="utf-8") as json_file:
        all_trunc = json.load(json_file)
    return df_md, all_trunc


def save_artifact(artifact, path, txt4print):
    """Save artifact in path while printing text info"""
    artifact.save(path)
    print(f"Saved {txt4print}: {path}")


def get_study_reads(study, df_md, all_trunc, path2seq):
    """Get study sequences and related information needed
    for denoising"""
    # get required metadata
    df_study = df_md[df_md["study_cohort_name"] == study].copy()
    type_of_reads = df_study["library_layout"].unique().tolist()[0]
    trunc = all_trunc[study]

    # get sequences
    seq = q2.Artifact.load(os.path.join(path2seq, "PRJEB5482", "paired_reads.qza"))
    return seq, type_of_reads, trunc


def denoise_sequences(path2md, path2trunc_len, path2seq, threads):
    """Denoise trimmed sequences saved in path2seq of all study
    subcohorts included in path2md"""
    df_md, all_trunc = read_files_all_studies(path2md, path2trunc_len)

    for study_cohort in df_md["study_cohort_name"].unique():
        print(f"Denoising: {study_cohort}...")
        path2save_feat = os.path.join(path2seq, f"asv_{study_cohort}.qza")

        if os.path.isfile(path2save_feat):
            print(f"Denoised sequences already exist: {path2save_feat}.")
        else:
            seq, type_of_reads, trunc = get_study_reads(
                study_cohort, df_md, all_trunc, path2seq
            )
            if "f" not in trunc.keys():
                print("No truncation info available for this study cohort.")
                continue

            # denoise: no need to trim as I removed primers before
            if type_of_reads == "PAIRED":
                (
                    features,
                    repseq,
                    stats,
                ) = dada2.actions.denoise_paired(
                    seq,
                    trunc_len_f=trunc["f"],
                    trunc_len_r=trunc["r"],
                    n_threads=threads,
                )
            elif type_of_reads == "SINGLE":
                (
                    features,
                    repseq,
                    stats,
                ) = dada2.actions.denoise_single(
                    seq, trunc_len=trunc["f"], n_threads=threads
                )
            # save denoised sequences
            save_artifact(features, path2save_feat, "denoised sequences")
            # save representative sequences
            save_artifact(
                repseq,
                os.path.join(path2seq, f"repseq_{study_cohort}.qza"),
                "representative sequences",
            )
            # save dada2 stats
            save_artifact(
                stats,
                os.path.join(path2seq, f"dada2stats_{study_cohort}.qza"),
                "dada2 stats",
            )


if __name__ == "__main__":
    args = parse_arguments()
    print(f"Denoising sequences of studies present in metadata {args.path2md}")

    denoise_sequences(
        path2md=args.path2md,
        path2trunc_len=args.path2trunc_len,
        path2seq=args.path2allseq,
        threads=args.threads,
    )
