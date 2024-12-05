#!/bin/bash


if [ -f "$1/fasttree_tree_rooted_subr14.qza" ]; then
    echo "$1/fasttree_tree_rooted_subr14.qza found - not creating phylogeny again"
else
    echo "Constructing phylogenetic tree with fasttree ..."

    # sequence alignment
    qiime alignment mafft \
        --i-sequences "$1/otu_seq_subr14_wq.qza" \
        --o-alignment "$1/otu_seq_subr14_wq_alig.qza"

    # alignment masking
    qiime alignment mask \
        --i-alignment "$1/otu_seq_subr14_wq_alig.qza" \
        --o-masked-alignment "$1/otu_seq_subr14_wq_aligm.qza"

    # create de-novo tree
    qiime phylogeny fasttree \
        --i-alignment "$1/otu_seq_subr14_wq_aligm.qza" \
        --o-tree "$1/fasttree_tree_subr14.qza"

    qiime phylogeny midpoint-root \
        --i-tree "$1/fasttree_tree_subr14.qza" \
        --o-rooted-tree "$1/fasttree_tree_rooted_subr14.qza"
fi
