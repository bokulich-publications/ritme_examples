"""Prepare the Earth Microbiome Project data for use case 4 (EMPO-3 classification).

EMP release 1 ships a fixed train/test membership (mapping columns ``subset_2k``
and ``qc_filtered``), so ritme's ``split_train_test`` is bypassed: the functions
below write ``train_val.pkl`` / ``test.pkl`` directly and reproduce its
preprocessing (``F``-prefixed feature ids, features that are zero in every
sample removed over the full table, rows converted to relative abundances).

Validation raises ``ValueError`` rather than asserting, so the gates survive
``python -O``.
"""

from __future__ import annotations

import os
import resource
from pathlib import Path

import biom
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

FEATURE_PREFIX = "F"
METADATA_COLUMNS = ["empo_3", "empo_2", "study_id"]
CONTROL_CLASSES = {"Sterile water blank", "Mock community"}
TAX_PREFIXES = ["k__", "p__", "c__", "o__", "f__", "g__", "s__"]
DEPTH_QUANTILES = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
_AGG_RANKS = ["phylum", "class", "order", "family", "genus", "species"]


def process_peak_rss_gib() -> float:
    """Process-wide peak resident memory in GiB, as a high-water mark since start.

    Monotonic, so it is a property of the process and not of any one call. It
    also under-reports these tables: ``scipy``'s ``toarray`` allocates with
    ``np.zeros`` and writes only the non-zeros, so untouched zero pages of a
    50 GiB array never become resident. Do not size a job from this number.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def build_metadata(
    mapping_path: str | os.PathLike,
    out_path: str | os.PathLike,
    published_counts: dict[str, int],
    drop_classes_absent_from_test: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Write the metadata TSV for the published EMP split and return it with a report.

    Columns: ``#SampleID`` (index), the EMPO labels, ``study_id`` and ``split``.
    ``train`` is the published 2,000-sample subset, ``test`` every other
    QC-filtered sample; ``published_counts`` states the expected membership of
    each and is enforced before any sample is dropped.

    Control classes are absent from ``qc_filtered`` in EMP release 1, so their
    presence is treated as an error rather than silently filtered.

    With ``drop_classes_absent_from_test`` (the default), classes that occur in
    train but never in test are removed entirely: they cannot be evaluated, and
    keeping them makes a macro one-vs-rest AUROC over the full class list
    undefined. The returned report records exactly what was removed.
    """
    mapping = pd.read_csv(
        mapping_path, sep="\t", dtype=str, index_col="#SampleID", low_memory=False
    )
    is_train = mapping["subset_2k"] == "True"
    is_test = (mapping["qc_filtered"] == "True") & ~is_train
    md = mapping.loc[is_train | is_test, METADATA_COLUMNS].copy()
    md["split"] = np.where(is_train[md.index], "train", "test")

    controls = md.index[md["empo_3"].isin(CONTROL_CLASSES)]
    if len(controls):
        raise ValueError(
            f"{len(controls)} control samples inside the published split "
            f"({sorted(set(md.loc[controls, 'empo_3']))}); EMP release 1 excludes "
            f"them from qc_filtered, so the membership columns changed."
        )
    if md["empo_3"].isna().any():
        raise ValueError(
            f"{int(md['empo_3'].isna().sum())} samples without EMPO-3 label"
        )

    counts = md["split"].value_counts().to_dict()
    if counts != published_counts:
        raise ValueError(
            f"published split membership changed: {counts} != {published_counts}"
        )

    report = {
        "published_counts": counts,
        "n_classes_published": int(md["empo_3"].nunique()),
    }
    absent = sorted(
        set(md.loc[md["split"] == "train", "empo_3"])
        - set(md.loc[md["split"] == "test", "empo_3"])
    )
    report["classes_absent_from_test"] = absent
    if drop_classes_absent_from_test and absent:
        drop = md["empo_3"].isin(absent)
        report["n_samples_dropped"] = int(drop.sum())
        md = md[~drop]
    else:
        report["n_samples_dropped"] = 0
    report["counts"] = md["split"].value_counts().to_dict()
    report["n_classes"] = int(md["empo_3"].nunique())

    md.to_csv(out_path, sep="\t")
    return md, report


def _read_depth_summary(depth: pd.Series) -> dict:
    summary = {
        f"q{int(q * 100):02d}": float(depth.quantile(q)) for q in DEPTH_QUANTILES
    }
    summary["mean"] = float(depth.mean())
    return summary


def build_split_tables(
    biom_path: str | os.PathLike,
    md: pd.DataFrame,
    out_dir: str | os.PathLike,
    feature_ids_path: str | os.PathLike,
    expected_n: dict[str, int],
    declared_n_features: int,
) -> dict:
    """Materialise ``train_val.pkl`` and ``test.pkl`` for one BIOM table.

    Samples are assigned by the ``split`` column of ``md``; ``expected_n`` states
    how many rows each split must have and is required, because a sample-id
    format change would otherwise write empty frames that pass every later
    check. Features are kept when they are non-zero in at least one sample of
    the table restricted to ``md``, prefixed with ``F`` and left in BIOM order;
    each row is divided by its read depth. The metadata columns of ``md``
    precede the feature columns, as in ritme's ``split_train_test`` output.
    ``declared_n_features`` is the table's own observation count, so that the
    retained and removed counts are anchored to something external.

    ``out_dir`` must already exist: in a worktree it is a symlink into the main
    clone, and creating it here would write ~50 GiB to the wrong place and leave
    the next ritme run to regenerate a random split. Returns table statistics.
    """
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        raise FileNotFoundError(
            f"{out_dir} does not exist - create it (or its Task 0 symlink into the "
            f"main clone) before writing splits"
        )

    table = biom.load_table(str(biom_path))
    if len(table.ids(axis="observation")) != declared_n_features:
        raise ValueError(
            f"{biom_path} declares {len(table.ids(axis='observation'))} observations, "
            f"expected {declared_n_features}"
        )
    counts = table.matrix_data.tocsr()  # observations x samples
    obs_ids = np.asarray(table.ids(axis="observation"))
    sample_ids = pd.Index(table.ids(axis="sample"))
    del table

    missing = md.index.difference(sample_ids)
    if len(missing):
        raise ValueError(
            f"{len(missing)} split samples absent from {biom_path} "
            f"(e.g. {list(missing[:3])}); sample-id formats disagree"
        )
    # restrict to the split before deciding which features are informative, as
    # ritme's split_train_test does (it joins metadata and table first)
    keep_samples = sample_ids.get_indexer(md.index)
    counts = counts[:, keep_samples]
    sample_ids = pd.Index(md.index)

    depth = pd.Series(np.asarray(counts.sum(axis=0)).ravel(), index=sample_ids)
    if not (depth > 0).all():
        raise ValueError(f"{int((depth <= 0).sum())} samples without any reads")

    keep = np.asarray(counts.sum(axis=1)).ravel() > 0
    feature_ids = obs_ids[keep]
    Path(feature_ids_path).write_text("\n".join(feature_ids) + "\n")

    # samples x features relative abundances, still sparse
    rel = sp.diags(1.0 / depth.to_numpy()) @ counts[keep].T.tocsr()
    del counts

    stats = {
        "biom": Path(biom_path).name,
        "n_features_declared": int(declared_n_features),
        "n_features": int(keep.sum()),
        "n_features_all_zero_removed": int((~keep).sum()),
        "n_samples_in_split": int(len(sample_ids)),
    }
    all_classes = sorted(md["empo_3"].unique())
    columns = [f"{FEATURE_PREFIX}{f}" for f in feature_ids]
    for split, file_name in (("train", "train_val.pkl"), ("test", "test.pkl")):
        ids = md.index[md["split"] == split]
        if len(ids) != expected_n[split]:
            raise ValueError(
                f"{split}: {len(ids)} samples, expected {expected_n[split]}"
            )

        pos = sample_ids.get_indexer(ids)
        if (pos < 0).any():
            raise ValueError(
                f"{split}: {int((pos < 0).sum())} samples not found in the table"
            )
        dense = rel[pos].toarray()
        if not np.allclose(dense.sum(axis=1), 1.0, atol=1e-3):
            raise ValueError(f"{split}: rows must sum to 1")

        frame = pd.DataFrame(
            dense, index=pd.Index(ids, name=md.index.name), columns=columns, copy=False
        )
        for col in reversed(md.columns):
            frame.insert(0, col, md.loc[ids, col].to_numpy())
        frame.to_pickle(out_dir / file_name)

        stats[f"n_{split}"] = int(len(ids))
        stats[f"read_depth_{split}"] = _read_depth_summary(depth.loc[ids])
        stats[f"empo_3_counts_{split}"] = (
            md.loc[ids, "empo_3"]
            .value_counts()
            .reindex(all_classes, fill_value=0)
            .to_dict()
        )
        stats[f"{file_name}_gib"] = round(
            (out_dir / file_name).stat().st_size / 1024**3, 2
        )
        del dense, frame
    stats["out_dir_resolved"] = str(out_dir.resolve())
    return stats


def _format_lineage(ranks) -> str:
    """Join ranks with ``; ``.

    An unresolved rank becomes its bare prefix (``o__``) and trailing unresolved
    ranks are dropped. A rank value carrying no prefix receives its own (EMP's
    bare ``Unclassified`` kingdom becomes ``k__Unclassified``); one carrying a
    different rank's prefix is an error, because the ranks would be shifted.

    ritme reads a bare prefix, and a lineage that stops short of the requested
    rank, as ``<prefix>unknown`` and sums every such feature into one column at
    that rank - it refines the label with the next rank up only, so it does not
    walk to the nearest resolved rank.
    """
    levels = []
    for prefix, value in zip(TAX_PREFIXES, ranks):
        value = value.strip()
        if value in ("", prefix):
            levels.append(prefix)
        elif value.startswith(prefix):
            levels.append(value)
        elif value[:3] in TAX_PREFIXES:
            raise ValueError(
                f"rank value {value!r} does not belong at rank {prefix!r}: {ranks}"
            )
        else:
            levels.append(prefix + value)
    while levels and levels[-1] in TAX_PREFIXES:
        levels.pop()
    return "; ".join(levels)


def extract_taxonomy(
    biom_path: str | os.PathLike,
    feature_ids_path: str | os.PathLike,
    out_path: str | os.PathLike,
) -> dict:
    """Write the BIOM's observation taxonomy as a ``Feature ID`` / ``Taxon`` TSV.

    Requires an HDF5 BIOM whose ``observation/metadata/taxonomy`` is a 2-D array
    of the seven Greengenes ranks. Covers every id in ``feature_ids_path`` with
    unprefixed ids (ritme adds the ``F`` itself). Returns the feature count, the
    number of lineages resolved to each depth, and the count of features that
    would collapse into a single ``unknown`` column per aggregation rank.
    """
    with h5py.File(biom_path, "r") as h5:
        obs_ids = h5["observation/ids"].asstr()[:]
        ranks = h5["observation/metadata/taxonomy"]
        if ranks.ndim != 2 or ranks.shape[1] != len(TAX_PREFIXES):
            raise ValueError(
                f"expected a (n, {len(TAX_PREFIXES)}) taxonomy array, "
                f"got shape {ranks.shape}"
            )
        ranks = ranks.asstr()[:]
    taxonomy = pd.DataFrame(
        {"Taxon": [_format_lineage(row) for row in ranks]},
        index=pd.Index(obs_ids, name="Feature ID"),
    )

    feature_ids = Path(feature_ids_path).read_text().split()
    missing = set(feature_ids) - set(taxonomy.index)
    if missing:
        raise ValueError(f"{len(missing)} features have no taxonomy entry")
    taxonomy = taxonomy.loc[feature_ids]
    taxonomy.to_csv(out_path, sep="\t")

    n_levels = taxonomy["Taxon"].str.count(";") + 1
    counts = n_levels.value_counts().sort_index()
    return {
        "n_features": int(len(taxonomy)),
        "frac_kingdom_only": round(float((n_levels == 1).mean()), 4),
        "n_levels_counts": {int(k): int(v) for k, v in counts.items()},
        # a lineage shorter than the rank leaves ritme aggregating it into one column
        "n_pooled_into_unknown_at": {
            rank: int((n_levels < i + 1).sum()) for i, rank in enumerate(_AGG_RANKS, 1)
        },
    }


def compare_rarefied_tables(
    subset_path: str | os.PathLike, full_path: str | os.PathLike
) -> dict:
    """Compare EMP's published ``subset_2k`` rarefied table with the full rarefied one.

    Restricted to the samples and features the two tables share, so
    ``identical_counts`` describes that shared block and the two
    ``features_only_in_*`` counts state what it excludes.
    """
    subset = biom.load_table(str(subset_path))
    full = biom.load_table(str(full_path))
    subset_obs = set(subset.ids(axis="observation"))
    full_obs = set(full.ids(axis="observation"))
    shared_obs = [o for o in full.ids(axis="observation") if o in subset_obs]

    full = full.filter(subset.ids(axis="sample"), axis="sample", inplace=False)
    full = full.filter(shared_obs, axis="observation", inplace=False)
    subset = subset.filter(full.ids(axis="sample"), axis="sample", inplace=False)
    subset = subset.filter(shared_obs, axis="observation", inplace=False)
    subset = subset.sort_order(full.ids(axis="sample"), axis="sample")
    subset = subset.sort_order(full.ids(axis="observation"), axis="observation")
    if list(subset.ids(axis="sample")) != list(full.ids(axis="sample")):
        raise ValueError("sample order differs after sorting")
    if list(subset.ids(axis="observation")) != list(full.ids(axis="observation")):
        raise ValueError("observation order differs after sorting")

    n_diff = int(np.count_nonzero((subset.matrix_data - full.matrix_data).data))
    return {
        "shared_samples": int(len(full.ids(axis="sample"))),
        "shared_features": int(len(shared_obs)),
        "features_only_in_subset_table": int(len(subset_obs) - len(shared_obs)),
        "features_only_in_full_table": int(len(full_obs) - len(shared_obs)),
        "identical_counts": n_diff == 0,
        "n_differing_entries": n_diff,
    }
