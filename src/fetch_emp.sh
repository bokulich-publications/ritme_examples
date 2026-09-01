#!/usr/bin/env bash
# Fetch the Earth Microbiome Project release-1 files used by use case 4.
#
# Needs internet access (on Euler: a login node after `module load eth_proxy`).
# Files already present with the expected size are skipped, so the script can
# be called again from a notebook that runs offline.
#
# Usage: fetch_emp.sh [OUTPUT_DIR]   (default: data/u4_emp)
set -euo pipefail

BASE=http://ftp.microbio.me/emp/release1
OUT=${1:-data/u4_emp}
mkdir -p "$OUT"

# "<path relative to BASE> <expected size in bytes>"
FILES=(
  "otu_tables/deblur/emp_deblur_90bp.subset_2k.biom 115517545"
  "otu_tables/deblur/emp_deblur_90bp.subset_2k.rare_5000.biom 56378167"
  "otu_tables/deblur/emp_deblur_90bp.qc_filtered.biom 266725902"
  "otu_tables/deblur/emp_deblur_90bp.qc_filtered.rare_5000.biom 176022277"
  "mapping_files/emp_qiime_mapping_release1.tsv 21876184"
)

for entry in "${FILES[@]}"; do
  read -r path size <<<"$entry"
  dst="$OUT/$(basename "$path")"
  if [[ -f "$dst" && "$(stat -c %s "$dst")" -eq "$size" ]]; then
    echo "[skip] $dst already present"
    continue
  fi
  echo "[fetch] $BASE/$path"
  curl -fsSL --retry 3 -o "$dst.part" "$BASE/$path"
  actual=$(stat -c %s "$dst.part")
  if [[ "$actual" -ne "$size" ]]; then
    echo "size mismatch for $dst: got $actual bytes, expected $size" >&2
    exit 1
  fi
  mv "$dst.part" "$dst"
done
