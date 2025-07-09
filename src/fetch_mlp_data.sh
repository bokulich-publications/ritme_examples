#!/bin/bash

FILE_URL="https://zenodo.org/records/14280080/files/data.zip?download=1"
OUTPUT_FILE="data.zip"
TARGET_DIR="../../data/u3_mlp_nishijima24"
MARKER_FILE="MetaCardis_load.tsv"

# ensure target dir exists
mkdir -p "${TARGET_DIR}"

# skip if we've already successfully fetched+unzipped before
if [ -f "${TARGET_DIR}/${MARKER_FILE}" ]; then
  echo "Data already fetched in ${TARGET_DIR}, skipping."
  exit 0
fi

echo "Downloading file from ${FILE_URL}..."
curl -L -o "${TARGET_DIR}/${OUTPUT_FILE}" "${FILE_URL}"

echo "Unzipping file to ${TARGET_DIR}..."
unzip -o "${TARGET_DIR}/${OUTPUT_FILE}" -d "${TARGET_DIR}"

echo "Unzip successful. Cleaning up…"
rm -f "${TARGET_DIR}/${OUTPUT_FILE}"
rm -f "${TARGET_DIR}/data"/Vandeputte_*
rm -f "${TARGET_DIR}/data/validation_files.txt"
rm -f "${TARGET_DIR}/data/pathway.list"
rm -r "${TARGET_DIR}/data/metalog"

# Move everything from the “data” subfolder into the parent
mv "${TARGET_DIR}/data/"* "${TARGET_DIR}/"

rm -r "${TARGET_DIR}/data"
