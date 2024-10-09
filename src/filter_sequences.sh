#!/bin/bash

if [ -f "$1/otu_table_subr14_filt_rel.qza" ]; then
    echo "$1/otu_table_subr14_filt_rel.qza found - not filtering again"
else
    echo "Filtering sequences ..."
    qiime feature-table filter-features-conditionally \
        --i-table "$1/otu_table_subr14.qza" \
        --p-abundance 0.001 \
        --p-prevalence $2 \
        --o-filtered-table "$1/otu_table_subr14_filt.qza"
fi
