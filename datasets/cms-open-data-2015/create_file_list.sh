#!/bin/bash

PROCESS="single_top"

RECIDS=(
19394
19406
19408
)

mkdir -p $PROCESS

for RECID in ${RECIDS[*]}; do
    echo "creating file list for record $RECID in $PROCESS/$RECID.txt"
    cernopendata-client get-file-locations --protocol xrootd --recid $RECID > tmp.txt
    head -n 1 tmp.txt
    mv tmp.txt $PROCESS/$RECID.txt
    printf "...\n\n"
done
