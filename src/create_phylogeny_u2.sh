#!/bin/bash


if [ -f "$1/fasttree_tree_rooted_suna15.qza" ]; then
    echo "$1/fasttree_tree_rooted_suna15.qza found - not creating phylogeny again"
else
    echo "Fetching SILVA OTU reference sequences ..."
    curl -L -o "$1/sun15_silva_16s_otu_ref_seqs.fna.gz" \
        https://ocean-microbiome.embl.de/data/16S.OTU.SILVA.reference.sequences.fna.gz
    gunzip "$1/sun15_silva_16s_otu_ref_seqs.fna.gz"

    qiime tools import \
        --type "FeatureData[Sequence]" \
        --input-path "$1/sun15_silva_16s_otu_ref_seqs.fna" \
        --output-path "$1/otu_seq_suna15.qza"

    echo "Constructing phylogenetic tree with fasttree ..."
    # sequence alignment
    qiime alignment mafft \
        --i-sequences "$1/otu_seq_suna15.qza" \
        --o-alignment "$1/otu_seq_suna15_alig.qza"

    # alignment masking
    qiime alignment mask \
        --i-alignment "$1/otu_seq_suna15_alig.qza" \
        --o-masked-alignment "$1/otu_seq_suna15_aligm.qza"

    # create de-novo tree
    qiime phylogeny fasttree \
        --i-alignment "$1/otu_seq_suna15_aligm.qza" \
        --o-tree "$1/fasttree_tree_suna15.qza"

    qiime phylogeny midpoint-root \
        --i-tree "$1/fasttree_tree_suna15.qza" \
        --o-rooted-tree "$1/fasttree_tree_rooted_suna15.qza"
fi
