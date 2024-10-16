#!/bin/bash

# read Greengenes 13_8 99% OTUs reference sequences
if [ -f "$1/gg_13_8_99_otus.qza" ]; then
    echo "$1/gg_13_8_99_otus.qza found - not fetching again"
else
    echo "Fetching Greengenes reference ..."
    curl -o "$1/gg_13_8_otus.tar.gz" \
        https://data.qiime2.org/classifiers/greengenes/gg_13_8_otus.tar.gz

    tar -xzf "$1/gg_13_8_otus.tar.gz" -C "$1"

    # import 99% rep_set to Q2
    qiime tools import \
        --type "FeatureData[Sequence]" \
        --input-path "$1/gg_13_8_otus/rep_set/99_otus.fasta" \
        --output-path "$1/gg_13_8_99_otus.qza"

    # clean-up
    rm -r "$1/gg_13_8_otus"
    rm -r "$1/gg_13_8_otus.tar.gz"
fi

# cluster
if [ -f "$1/otu_table_subr14.qza" ]; then
    echo "$1/otu_table_subr14.qza found - not clustering again"
else
    echo "Clustering sequences ..."
    qiime vsearch cluster-features-open-reference \
        --i-table "$1/asv_subramanian14.qza" \
        --i-sequences "$1/repseq_subramanian14.qza" \
        --i-reference-sequences "$1/gg_13_8_99_otus.qza" \
        --p-perc-identity 0.97 \
        --p-threads $2 \
        --o-clustered-table "$1/otu_table_subr14.qza" \
        --o-clustered-sequences "$1/otu_seq_subr14.qza" \
        --o-new-reference-sequences "$1/otu_seq_subr14_new_ref.qza"
fi
