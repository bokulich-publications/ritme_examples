"""Convert QIIME2 artifacts (.qza) to plain files usable by ritme >= 1.3.0.

ritme dropped the QIIME2 dependency in v1.3.0; feature tables, taxonomies and
phylogenies are now expected as plain ``.tsv`` and ``.nwk`` files. This module
extracts the relevant payload from a ``.qza`` archive (a zip wrapping the
underlying biom / TSV / Newick file) and writes it next to the input.

Usage (CLI):
    python -m src.convert_qiime2_artifacts feature-table input.qza
    python -m src.convert_qiime2_artifacts taxonomy taxonomy.qza
    python -m src.convert_qiime2_artifacts tree tree.qza

Or import the ``convert_*`` functions directly.
"""

import argparse
import os
import shutil
import sys
import tempfile
import zipfile
from typing import Optional


def _extract_payload(qza_path: str, inner_filename: str, work_dir: str) -> str:
    """Extract a single payload file from inside a .qza archive.

    Returns the absolute path of the extracted file.
    """
    with zipfile.ZipFile(qza_path, "r") as z:
        match = [n for n in z.namelist() if n.endswith(f"/data/{inner_filename}")]
        if not match:
            raise FileNotFoundError(
                f"{inner_filename} not found inside {qza_path}; "
                f"contents: {z.namelist()}"
            )
        z.extract(match[0], work_dir)
    return os.path.join(work_dir, match[0])


def convert_feature_table(qza_path: str, out_path: Optional[str] = None) -> str:
    """Convert a QIIME2 FeatureTable .qza to a samples-as-rows .tsv."""
    import biom

    out_path = out_path or qza_path.replace(".qza", ".tsv")
    with tempfile.TemporaryDirectory() as work_dir:
        biom_path = _extract_payload(qza_path, "feature-table.biom", work_dir)
        table = biom.load_table(biom_path)
        df = table.to_dataframe(dense=True).T
        df.index.name = "id"
        df.to_csv(out_path, sep="\t")
    return out_path


def convert_taxonomy(qza_path: str, out_path: Optional[str] = None) -> str:
    """Convert a QIIME2 Taxonomy .qza to a .tsv (Feature ID, Taxon, Confidence)."""
    out_path = out_path or qza_path.replace(".qza", ".tsv")
    with tempfile.TemporaryDirectory() as work_dir:
        tsv_path = _extract_payload(qza_path, "taxonomy.tsv", work_dir)
        shutil.copy(tsv_path, out_path)
    return out_path


def convert_tree(qza_path: str, out_path: Optional[str] = None) -> str:
    """Convert a QIIME2 Phylogeny .qza to a Newick .nwk."""
    out_path = out_path or qza_path.replace(".qza", ".nwk")
    with tempfile.TemporaryDirectory() as work_dir:
        tree_path = _extract_payload(qza_path, "tree.nwk", work_dir)
        shutil.copy(tree_path, out_path)
    return out_path


_KIND_DISPATCH = {
    "feature-table": convert_feature_table,
    "taxonomy": convert_taxonomy,
    "tree": convert_tree,
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kind", choices=sorted(_KIND_DISPATCH))
    parser.add_argument("qza", help="Path to input .qza")
    parser.add_argument(
        "-o",
        "--output",
        help="Output path. Defaults to input with .tsv/.nwk extension.",
    )
    args = parser.parse_args(argv)
    out = _KIND_DISPATCH[args.kind](args.qza, args.output)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
