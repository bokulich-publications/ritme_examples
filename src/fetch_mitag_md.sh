#!/bin/bash

DESTINATION_FOLDER="$1"
FILENAME_MD="${DESTINATION_FOLDER}/OM.CompanionTables.xlsx"
URL_MD="$2"

if [ ! -f "${FILENAME_MD%.gz}" ]; then
    curl -L -o "$FILENAME_MD" "$URL_MD"
else
  echo "Metadata was already downloaded here: ${FILENAME_MD}"
fi
