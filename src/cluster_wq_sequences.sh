#!/bin/bash

# read Greengenes 13_8 99% OTUs reference sequences
if [ -f "$1/gg_13_8_99_otus.qza" ]; then
    echo "$1/gg_13_8_99_otus.qza found - not fetching again"
else
    echo "Fetching Greengenes reference ..."
    curl -L -o "$1/gg_13_8_otus.tar.gz" \
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

# quality filter
if [ -f "$1/subr14_seq_quality.qza" ]; then
    echo "$1/subr14_seq_quality.qza found - not performing QC again"
else
    echo "Perform quality control of sequences ..."
    qiime quality-filter q-score \
        --i-demux "$1/trimmed_subramanian14.qza" \
        --p-min-quality 20 \
        --o-filtered-sequences "$1/subr14_seq_quality.qza" \
        --o-filter-stats "$1/subr14_seq_quality_stats.qza"
fi

# dereplicate
if [ -f "$1/subr14_seqs_derep.qza" ]; then
    echo "$1/subr14_seqs_derep.qza found - not dereplicating again"
else
    echo "Dereplicating sequences ..."
    qiime vsearch dereplicate-sequences \
        --i-sequences "$1/subr14_seq_quality.qza" \
        --o-dereplicated-sequences "$1/subr14_seqs_derep.qza" \
        --o-dereplicated-table "$1/subr14_table_derep.qza"
fi

# cluster
if [ -f "$1/otu_table_subr14_wq.qza" ]; then
    echo "$1/otu_table_subr14_wq.qza found - not clustering again"
else
    echo "Clustering sequences ..."
    qiime vsearch cluster-features-open-reference \
        --i-table "$1/subr14_table_derep.qza" \
        --i-sequences "$1/subr14_seqs_derep.qza" \
        --i-reference-sequences "$1/gg_13_8_99_otus.qza" \
        --p-perc-identity 0.97 \
        --p-threads $2 \
        --o-clustered-table "$1/otu_table_subr14_wq.qza" \
        --o-clustered-sequences "$1/otu_seq_subr14_wq.qza" \
        --o-new-reference-sequences "$1/otu_seq_subr14_wq_new_ref.qza"
fi
