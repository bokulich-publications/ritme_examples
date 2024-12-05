"""Functions to fetch metadata"""

import os
from typing import Tuple

import pandas as pd
import qiime2 as q2
import requests
from qiime2.plugins import fondue


def fetch_metadata(
    ids, email, n_jobs, path2md, path2failed
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch metadata of `ids` and store the artifacts in `path2` location.

    Args:
        ids (q2.sdk.result.Artifact): Ids to fetch metadata for.
        email (str): Email.
        n_jobs (int): Number of jobs for q2fondue get-metadata.
        path2md (str): Path to store metadata in.
        path2failed (str): Path to store failed ids in.

    Returns:
        pd.DataFrame: Dataframe of fetched metadata
        pd.DataFrame: Dataframe of failed run IDs.
    """
    if not os.path.isfile(path2md):
        # fetch metadata
        (
            meta_md,
            failed_runs,
        ) = fondue.actions.get_metadata(
            accession_ids=ids, email=email, n_jobs=n_jobs, log_level="INFO"
        )
        # save for future reuse
        meta_md.save(path2md)
        failed_runs.save(path2failed)
    else:
        meta_md = q2.Artifact.load(path2md)
        failed_runs = q2.Artifact.load(path2failed)
        print(f'Metadata was read from file "{path2md}"')

    return meta_md.view(pd.DataFrame), failed_runs.view(pd.DataFrame)


def _fetch_sra_metadata(path2data, ids, email, n_jobs):
    path2md = os.path.join(path2data, "metadata.qza")
    path2failed = os.path.join(path2data, "metadata_failed_runs.qza")

    if not os.path.isfile(path2md):
        print("Fetching metadata...")
        meta, failed = fetch_metadata(ids, email, n_jobs, path2md, path2failed)
        assert failed.shape[0] == 0
        print(f"Metadata was fetched and saved to file {path2md}")
    else:
        meta = q2.Artifact.load(path2md)
        meta = meta.view(pd.DataFrame)
        print(f"Metadata was read from file {path2md}")

    # small postprocess
    meta = meta.reset_index()
    meta.rename(columns={"ID": "Run ID"}, inplace=True)
    return meta


def _make_dirs(path2dir):
    if not os.path.exists(path2dir):
        os.makedirs(path2dir)


def _fetch_n_store_excel_file(url, filedest):
    response = requests.get(url)
    with open(filedest, "wb") as f:
        f.write(response.content)


def _fetching_programmatically_not_allowed(url, filedest):
    raise ValueError(
        f"The metadata can't be fetched programmatically."
        f"Visit the following URL and download the file "
        f'manually into "{filedest}": {url}'
    )


def _fetch_all_supp_material(path2data, url):
    dest_suppmat = os.path.join(path2data, "supp_material")
    _make_dirs(dest_suppmat)

    filedest_all = os.path.join(dest_suppmat, "md.xlsx")

    _fetch_n_store_excel_file(url, filedest_all)

    try:
        _ = pd.read_excel(filedest_all)
    except Exception:
        _fetching_programmatically_not_allowed(url, filedest_all)

    return filedest_all


def save_file(df_md, path2data, tag):
    path2md = os.path.join(path2data, "metadata.qza")
    path2md_proc = path2md.replace(".qza", f"_proc_v{tag}.tsv")

    print(f"Saved processed metadata to: {path2md_proc}")
    df_md.to_csv(path2md_proc, sep="\t")

    # save unique runIDs of each projectIDs from this metadata file for sequence
    # fetching in d_
    unique_ids = df_md["bioproject_id"].unique()
    path2runids = os.path.join(path2data, "runids")
    if not os.path.exists(path2runids):
        os.makedirs(path2runids)

    for id in unique_ids:
        run_ids = df_md[df_md["bioproject_id"] == id].index.tolist()
        df_run_ids = pd.DataFrame({"id": run_ids}).squeeze()

        path2ids = os.path.join(path2runids, f"{id}.tsv")
        print(f"Saved unique project IDs to: {path2ids}")
        df_run_ids.to_csv(path2ids, sep="\t", index=False)

    return path2md_proc
