#!/bin/bash

# train v4 GG NB classifier
if [ -f "$1/gg-13-8-99-515-806-nb-classifier.qza" ]; then
    echo "$1/gg-13-8-99-515-806-nb-classifier.qza found - not retraining again"
else
    if [ ! -d "$1/gg_13_8_otus" ]; then
        curl -L -o "$1/gg_13_8_otus.tar.gz" \
        https://data.qiime2.org/classifiers/greengenes/gg_13_8_otus.tar.gz

        tar -xzf "$1/gg_13_8_otus.tar.gz" -C "$1"
        rm -r "$1/gg_13_8_otus.tar.gz"
    fi

    # train classifier
    # import needed files
    qiime tools import \
    --type 'FeatureData[Sequence]' \
    --input-path "$1/gg_13_8_otus/rep_set/99_otus.fasta" \
    --output-path "$1/99_otus.qza"

    qiime tools import \
    --type 'FeatureData[Taxonomy]' \
    --input-format HeaderlessTSVTaxonomyFormat \
    --input-path "$1/gg_13_8_otus/taxonomy/99_otu_taxonomy.txt" \
    --output-path "$1/ref-99-taxonomy.qza"

    qiime feature-classifier extract-reads \
    --i-sequences "$1/99_otus.qza" \
    --p-f-primer GTGCCAGCMGCCGCGGTAA \
    --p-r-primer GGACTACHVGGGTWTCTAAT \
    --p-trunc-len 162 \
    --p-min-length 100 \
    --p-max-length 400 \
    --o-reads "$1/99_ref_seqs.qza"

    # train classifier
    qiime feature-classifier fit-classifier-naive-bayes \
    --i-reference-reads "$1/99_ref_seqs.qza" \
    --i-reference-taxonomy "$1/ref-99-taxonomy.qza" \
    --o-classifier "$1/gg-13-8-99-515-806-nb-classifier.qza"
fi

# classify features
if [ -f "$1/taxonomy_subr14.qza" ]; then
    echo "$1/taxonomy_subr14.qza found - not creating taxonomy again"
else
    echo "Assigning taxonomy to features ..."

    qiime feature-classifier classify-sklearn \
    --i-classifier "$1/gg-13-8-99-515-806-nb-classifier.qza" \
    --i-reads "$1/otu_seq_subr14_wq.qza" \
    --o-classification "$1/taxonomy_subr14.qza"
fi
