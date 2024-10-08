#!/bin/bash
for file in "$1"/*.tsv; do
  projectid=$(basename ${file})
  projectid=$(echo "$projectid" | sed s/".tsv"//)
  echo "Analysing $projectid ..."

  # create Q2 artifact metadata file for runIDs of thisprojectID
  qiime tools import \
      --type NCBIAccessionIDs \
      --input-path "$1/$projectid.tsv" \
      --output-path "$1/$projectid.qza"

  # get sequences for this projectID
  if [ -d "$2/$projectid" ]; then
      echo "$2/$projectid found - not fetching again"
  else
      echo "Fetching sequences for $projectid"
      i=0
      while [ $i -le 3 ]; do
          echo "Trying: $i"
          qiime fondue get-sequences \
              --i-accession-ids "$1/$projectid.qza" \
              --p-email my@mail.ch \
              --p-retries 3 \
              --p-n-jobs "$3" \
              --output-dir "$2/$projectid" \
              --verbose && break
          let i=i+1
          sleep 2m
      done
  fi
  # remove created tsv and artifact of the 1 projectID files
  # rm -r "$2/$projectid.tsv"
  rm -r "$1/$projectid.qza"

  # check that failed_reads are empty
  refetch=$(python ../src/need2refetch.py "$2/$projectid")
  if [[ "$refetch" == "True" ]];
    then
      echo "Refetching failed_runs for $projectid ..."

      qiime fondue get-sequences \
            --i-accession-ids "$2/$projectid/failed_runs.qza" \
            --p-email my@mail.ch \
            --p-retries 3 \
            --p-n-jobs "$3" \
            --output-dir "$2/$projectid-refetch" \
            --verbose

      # merge refetch with previously fetched
      echo "Merging ..."
      qiime fondue combine-seqs \
            --i-seqs "$2/$projectid/single_reads.qza" "$2/$projectid-refetch/single_reads.qza" \
            --o-combined-seqs "$2/$projectid/single_reads_comb.qza"
      qiime fondue combine-seqs \
            --i-seqs "$2/$projectid/paired_reads.qza" "$2/$projectid-refetch/paired_reads.qza" \
            --o-combined-seqs "$2/$projectid/paired_reads_comb.qza"
      # clean up such that we can continue as if no refetching has happened
      rm -r "$2/$projectid/single_reads.qza"
      rm -r "$2/$projectid-refetch/single_reads.qza"
      rm -r "$2/$projectid/paired_reads.qza"
      rm -r "$2/$projectid-refetch/paired_reads.qza"
      mv "$2/$projectid/single_reads_comb.qza" "$2/$projectid/single_reads.qza"
      mv "$2/$projectid/paired_reads_comb.qza" "$2/$projectid/paired_reads.qza"
  fi
  echo "...finished fetching sequences of $projectid!"
done
