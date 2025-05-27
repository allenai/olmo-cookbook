#!/bin/bash

# Range: 0000 to 0511
: 'for i in $(seq -f "%04g" 1 511); do
    echo "Syncing checkpoint part-$i..."

    python -m cookbook.remote \
        s3://ai2-llm/checkpoints/mayeec/5xC-30m-superswarm-ee28fc9c-$i \
        weka://oe-data-default/ai2-llm/checkpoints/mayeec/5xC-30m-superswarm-ee28fc9c-$i \
        --workspace ai2/dolma2
done'



python -m cookbook.remote \
    s3://ai2-llm/checkpoints/mayeec/5xC-30m-superswarm-ee28fc9c-0086 \
    weka://oe-data-default/ai2-llm/checkpoints/mayeec/5xC-30m-superswarm-ee28fc9c-0086 \
    --workspace ai2/dolma2

python -m cookbook.remote \
    s3://ai2-llm/checkpoints/mayeec/5xC-30m-superswarm-ee28fc9c-0187 \
    weka://oe-data-default/ai2-llm/checkpoints/mayeec/5xC-30m-superswarm-ee28fc9c-0187 \
    --workspace ai2/dolma2

