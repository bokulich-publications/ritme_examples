#!/bin/bash

DESTINATION_FOLDER="$1"
FILENAME_COUNTS="${DESTINATION_FOLDER}/miTAG.taxonomic.profiles.release.tsv.gz"
URL_COUNTS="$2"

if [ ! -f "${FILENAME_COUNTS%.gz}" ]; then
    curl -L -o "$FILENAME_COUNTS" "$URL_COUNTS"
    gunzip "$FILENAME_COUNTS"
    rm -r "$FILENAME_COUNTS"
else
  echo "OTUs were already downloaded here: ${FILENAME_COUNTS%.gz}"
fi
