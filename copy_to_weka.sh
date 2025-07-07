#!/bin/bash

# Range: 0000 to 0511
: 'for i in $(seq -f "%04g" 1 511); do
    echo "Syncing checkpoint part-$i..."

    python -m cookbook.remote \
        s3://ai2-llm/checkpoints/mayeec/5xC-30m-superswarm-ee28fc9c-$i \
        weka://oe-data-default/ai2-llm/checkpoints/mayeec/5xC-30m-superswarm-ee28fc9c-$i \
        --workspace ai2/dolma2
done'



: 'python -m cookbook.remote \
    s3://ai2-llm/checkpoints/mayeec/5xC-30m-superswarm-ee28fc9c-0349 \
    weka://oe-data-default/ai2-llm/checkpoints/mayeec/5xC-30m-superswarm-ee28fc9c-0349 \
    --allow-dirty \
    --workspace ai2/dolma2'

: 'python -m cookbook.remote \
    s3://ai2-llm/checkpoints/mayeec/5xC-30m-superswarm-ee28fc9c-0187 \
    weka://oe-data-default/ai2-llm/checkpoints/mayeec/5xC-30m-superswarm-ee28fc9c-0187 \
    --allow-dirty \
    --workspace ai2/dolma2 
'


: 'experiments=(
    #'1b-5xC-superswarm-log-linear-cf0dd59f'
    '1b-5xC-superswarm-log-linear-underfit-1b2af97c'
    '1b-5xC-superswarm-natural-4d1a1440'
)'

: 'experiments=(
    'dclm-1b-log-linear-full-eval-constrain-obj-pareto-with-predicted-scores-173bec67'
    'dclm-1b-log-linear-full-eval-constrain-obj-pareto-with-true-scores-4557a88b'
    'dclm-1b-log-linear-full-eval-corr-weighting-constrain-obj-pareto-with-predicted-scores-b9e80c0a'
    'dclm-1b-log-linear-full-eval-corr-weighting-constrain-obj-pareto-with-true-scores-e07ba770'
)'

: 'experiments=(
    'dclm-1b-log-linear-extended-eval-constrain-obj-c24744d0'
)'


: 'experiments=(
    'dclm-1b-log-linear-full-eval-constrain-obj-pareto-with-predicted-scores-d7eaca81'
    'dclm-1b-log-linear-full-eval-constrain-obj-pareto-with-predicted-scores-357a137e'
)


for exp_id in "${experiments[@]}"; do
  echo "Converting: $exp_id"
    python -m cookbook.remote \
        gs://ai2-llm/checkpoints/mayeec/$exp_id/step61000 \
        weka://oe-data-default/ai2-llm/checkpoints/mayeec/$exp_id/step61000 \
        --allow-dirty \
        --workspace ai2/dolma2
done'



: 'for i in $(seq -f "%04g" 2 7); do
    echo "Syncing checkpoint part-$i..."

    python -m cookbook.remote \
        gs://ai2-llm/checkpoints/mayeec/5xC-30m-stackedu-a0e3840c-$i/step22100 \
        weka://oe-data-default/ai2-llm/checkpoints/mayeec/5xC-30m-stackedu-a0e3840c-$i/step22100 \
        --allow-dirty \
        --workspace ai2/dolma2
done'




: 'for i in $(seq -f "%04g" 0 127); do
    echo "Syncing checkpoint part-$i..."


    ignore = 53, 52, 58, 51, 50, 47, 43, 45

    python -m cookbook.remote \
        gs://ai2-llm/checkpoints/mayeec/5xC-30m-stackedu-e250076d-$i \
        weka://oe-data-default/ai2-llm/checkpoints/mayeec/5xC-30m-stackedu-e250076d-$i \
        --allow-dirty \
        --workspace ai2/dolma2
done'



#!/bin/bash

# Indices to ignore (zero-padded)
ignore_set=("0502" "0501" "0479" "0478" "0451" "0406" "0375" "0301", "0296")
ignore_set=("0301")

: 'for i in $(seq -f "%04g" 0 511); do
    # Skip if in ignore list
    if [[ " ${ignore_set[@]} " =~ " ${i} " ]]; then
        echo "Skipping part-$i (in ignore list)"
        continue
    fi

    echo "Syncing checkpoint part-$i..."

    python -m cookbook.remote \
        gs://ai2-llm/checkpoints/mayeec/5xC-30m-superswarm-dclm-conditional-4252db91-$i \
        weka://oe-data-default/ai2-llm/checkpoints/mayeec/5xC-30m-superswarm-dclm-conditional-4252db91-$i \
        --allow-dirty \
        --workspace ai2/dolma2
done'


: 'for i in "${ignore_set[@]}"; do

    echo "Syncing checkpoint part-$i..."

    python -m cookbook.remote \
        gs://ai2-llm/checkpoints/mayeec/5xC-30m-superswarm-dclm-conditional-4252db91-$i \
        weka://oe-data-default/ai2-llm/checkpoints/mayeec/5xC-30m-superswarm-dclm-conditional-4252db91-$i \
        --allow-dirty \
        --workspace ai2/dolma2
done'

: 'ignore_set=("0028" "0023" "0012")
ignore_set=("0008" "0019")
for i in "${ignore_set[@]}"; do

    echo "Syncing checkpoint part-$i..."

    python -m cookbook.remote \
        s3://ai2-llm/checkpoints/ai2-tylerm/olmo2-pdfs-datadelve-5xC-30m-augusta-2048-6d2d4c39-$i \
        weka://oe-training-default/ai2-llm/checkpoints/ai2-tylerm/olmo2-pdfs-datadelve-5xC-30m-augusta-2048-6d2d4c39-$i \
        --allow-dirty \
        --workspace ai2/dolma2
done'


: 'experiments=(
    'regmixer-stack-natural-a4e0986e-0000'
)

for exp_id in "${experiments[@]}"; do
  echo "Converting: $exp_id"
    python -m cookbook.remote \
        gs://ai2-llm/checkpoints/mayeec/$exp_id \
        weka://oe-data-default/ai2-llm/checkpoints/mayeec/$exp_id \
        --allow-dirty \
        --workspace ai2/dolma2
done'

: 'experiments=(
    '1b-5xC-superswarm-pareto-repetition-constraint-4.5-09dc1b5d'
    '1b-5xC-superswarm-repetition-constraint-4-e999ce68'
)

for exp_id in "${experiments[@]}"; do
  echo "Converting: $exp_id"
    python -m cookbook.remote \
        gs://ai2-llm/checkpoints/mayeec/$exp_id/step61000 \
        weka://oe-data-default/ai2-llm/checkpoints/mayeec/$exp_id/step61000 \
        --allow-dirty \
        --workspace ai2/dolma2
done'




# Generate full range and exclude specific numbers
: 'FULL_RANGE=($(seq -f "%04g" 0 127))
EXCLUDED=(0507 0508)
INCLUDED=()

for num in "${FULL_RANGE[@]}"; do
    skip=false
    for exclude in "${EXCLUDED[@]}"; do
        if [[ "$num" == "$exclude" ]]; then
            skip=true
            break
        fi
    done
    if [[ "$skip" == false ]]; then
        INCLUDED+=("$num")
    fi
done

INCLUDED=(0062 0016 0014 0013 0012)

# Function to count running background jobs
count_jobs() {
    jobs -r | wc -l
}

# Process all checkpoints with max 12 concurrent jobs
for i in "${INCLUDED[@]}"; do
    # Wait until we have fewer than 12 jobs running
    while [ $(count_jobs) -ge 12 ]; do
        sleep 5
    done

    echo "Starting evaluation for checkpoint $i..."
    {

        python -m cookbook.remote \
            gs://ai2-llm/checkpoints/mayeec/5xC-30m-stackedu-dense-with-dclm-cd4ac84e-$i \
            weka://oe-data-default/ai2-llm/checkpoints/mayeec/5xC-30m-stackedu-dense-with-dclm-cd4ac84e-$i \
            --allow-dirty \
            --workspace ai2/dolma2


        echo "Completed checkpoint $i"
    } &
done

# Wait for all remaining jobs to complete
wait
echo "All evaluation tasks completed!"
'




experiments=(
    #'1b-5xC-stack-natural-f7d1f44c'
    #'1b-5xC-stack-log-linear-pareto-repetition-5-cb53face'
    #'1b-5xC-superswarm-conditional-dclm-log-linear-78403bc2'
    #'1b-5xC-superswarm-v1-log-linear-on-v2-f15e08c9'
    #'1b-5xC-superswarm-v2-natural-52749eca'
    #'1b-5xC-superswarm-v2-conditional-dclm-natural-61de5e1e'
    '1b-5xC-stack-log-linear-dense-repetition-5-18ee3600'
    '1b-5xC-stack-search-d421574f'
    '1b-5xC-stack-search-0001-01a3fbf2'
)

experiments=(
    #'regmixer-superswarm-v2-conditional-dclm-log-linear-2e6e1ff0-0000'
    #'regmixer-superswarm-v1-log-linear-on-v2-fd7cab69-0000'
    #'regmixer-superswarm-v2-conditional-dclm-41533660-0000'
    'regmixer-superswarm-v2-natural-4f36a33e-0000'
)

experiments=(
    '1b-5xC-superswarm-conditional-dclm-log-linear-constraint-3-4ca6c07f'
    '1b-5xC-superswarm-conditional-dclm-log-linear-constraint-4-c1930276'
    '1b-5xC-superswarm-conditional-dclm-log-linear-constraint-5-7e702ff7'
    '1b-5xC-superswarm-conditional-dclm-collapse-pes2o-log-linear-b65c9504'
    '1b-5xC-superswarm-conditional-dclm-collapse-pes2o-log-linear-constraint-3-47953a8c'
    '1b-5xC-superswarm-conditional-dclm-collapse-pes2o-log-linear-constraint-4-2099326f'
    '1b-5xC-superswarm-conditional-dclm-collapse-pes2o-log-linear-constraint-5-0b61d5af'
)

experiments=(
    'regmixer-superswarm-conditional-dclm-collapse-pes2o-log-linear-fixed-constraint-3-660a6d04-0000'
    'regmixer-superswarm-conditional-dclm-collapse-pes2o-log-linear-fixed-constraint-4-1b497b25-0000'
    'regmixer-superswarm-conditional-dclm-collapse-pes2o-log-linear-fixed-constraint-5-00670414-0000'
    'regmixer-superswarm-conditional-dclm-collapse-pes2o-log-linear-fixed-02a77d23-0000'
)


experiments=(
    '1b-5xC-superswarm-conditional-dclm-collapse-pes2o-log-linear-fixed-e4d26d38'
    '1b-5xC-superswarm-conditional-dclm-collapse-pes2o-log-linear-fixed-constraint-3-7df4e2aa'
    '1b-5xC-superswarm-conditional-dclm-collapse-pes2o-log-linear-fixed-constraint-5-dc4cc209'
    '1b-5xC-superswarm-conditional-dclm-collapse-pes2o-log-linear-constraint-fixed-4-549f2d14'
)


experiments=(
    '1b-5xC-stack-proposed-mix-constrain-5-with-dclm-pstar-75-25-fixed-final-add8d35a'
    '1b-5xC-stack-natural-with-dclm-pstar-75-25-0798d32a'
)

: 'for exp_id in "${experiments[@]}"; do
  echo "Converting: $exp_id"
    python -m cookbook.remote \
        gs://ai2-llm/checkpoints/mayeec/$exp_id/step61000 \
        weka://oe-data-default/ai2-llm/checkpoints/mayeec/$exp_id/step61000 \
        --allow-dirty \
        --workspace ai2/dolma2
done'



: 'FULL_RANGE=($(seq -f "%04g" 0 511))
EXCLUDED=()
INCLUDED=()

for num in "${FULL_RANGE[@]}"; do
    skip=false
    for exclude in "${EXCLUDED[@]}"; do
        if [[ "$num" == "$exclude" ]]; then
            skip=true
            break
        fi
    done
    if [[ "$skip" == false ]]; then
        INCLUDED+=("$num")
    fi
done

INCLUDED=(0424 0276 0277 0270 0269 0268 0267 0265)

# Function to count running background jobs
count_jobs() {
    jobs -r | wc -l
}

# Process all checkpoints with max 12 concurrent jobs
for i in "${INCLUDED[@]}"; do
    # Wait until we have fewer than 12 jobs running
    while [ $(count_jobs) -ge 12 ]; do
        sleep 5
    done

    echo "Starting evaluation for checkpoint $i..."
    {

        python -m cookbook.remote \
            gs://ai2-llm/checkpoints/ai2-tylerm/5xC-30m-superswarm-dclm-stackedu-conditional-0cb55cb5-$i \
            weka://oe-data-default/ai2-llm/checkpoints/ai2-tylerm/5xC-30m-superswarm-dclm-stackedu-conditional-0cb55cb5-$i \
            --allow-dirty \
            --workspace ai2/dolma2

        echo "Completed checkpoint $i"
    } &
done

# Wait for all remaining jobs to complete
wait
echo "All evaluation tasks completed!"
'




: 'experiments=(
    'olmo2-1b_10b-anneal_dclm-natural-b22c56ae'
    'olmo2-1b_10b-anneal_dclm-pstar-8c1dc74d'
    'olmo2-1b_10b-anneal_dclm-p001-5b917927'
    #'olmo2-1b_10b-anneal_dclm-good-search-237bfb4d'
    #'olmo2-1b_10b-anneal_dclm-bad-search-dcbe19fe'
)

for exp_id in "${experiments[@]}"; do
  echo "Converting: $exp_id"
    python -m cookbook.remote \
        gs://ai2-llm/checkpoints/mayeec/$exp_id/step4765 \
        weka://oe-data-default/ai2-llm/checkpoints/mayeec/$exp_id/step4765 \
        --allow-dirty \
        --workspace ai2/dolma2
done'




: 'experiments=(
    #'olmo2-1b_5b-anneal_dclm-natural-c30587d8'
    #'olmo2-1b_5b-anneal_dclm-pstar-4ef42513'
    #'olmo2-1b_5b-anneal_dclm-p001-bb438d96'
    'olmo2-1b_5b-anneal_dclm-good-search-3e2e9aa3'
    'olmo2-1b_5b-anneal_dclm-bad-search-a563ffaa'
)

for exp_id in "${experiments[@]}"; do
  echo "Converting: $exp_id"
    python -m cookbook.remote \
        gs://ai2-llm/checkpoints/mayeec/$exp_id/step2383 \
        weka://oe-data-default/ai2-llm/checkpoints/mayeec/$exp_id/step2383 \
        --allow-dirty \
        --workspace ai2/dolma2
done'


: 'experiments=(
    #'olmo2-1b_1b-anneal_dclm-natural-64e19e13'
    #'olmo2-1b_1b-anneal_dclm-pstar-cbee4012'
    #'olmo2-1b_1b-anneal_dclm-p001-3be63626'
    'olmo2-1b_1b-anneal_dclm-bad-search-92c7f0d2'
    'olmo2-1b_1b-anneal_dclm-good-search-52726470'
)

for exp_id in "${experiments[@]}"; do
  echo "Converting: $exp_id"
    python -m cookbook.remote \
        gs://ai2-llm/checkpoints/mayeec/$exp_id/step476 \
        weka://oe-data-default/ai2-llm/checkpoints/mayeec/$exp_id/step476 \
        --allow-dirty \
        --workspace ai2/dolma2
done'


: 'experiments=(
    'olmo2-1b-2T_10b-anneal_dclm-bad-search-838d05f1'
    'olmo2-1b-2T_10b-anneal_dclm-good-search-8e6ac4b4'
)

for exp_id in "${experiments[@]}"; do
  echo "Converting: $exp_id"
    python -m cookbook.remote \
        gs://ai2-llm/checkpoints/mayeec/$exp_id/step4767 \
        weka://oe-data-default/ai2-llm/checkpoints/mayeec/$exp_id/step4767 \
        --allow-dirty \
        --workspace ai2/dolma2
done


experiments=(
    'olmo2-1b-2T_10b-anneal_dclm-natural-0aa5e0f5'
    'olmo2-1b-2T_10b-anneal_dclm-pstar-2194bab6'
    'olmo2-1b-2T_10b-anneal_dclm-p001-821319b2'
)

for exp_id in "${experiments[@]}"; do
  echo "Converting: $exp_id"
    python -m cookbook.remote \
        gs://ai2-llm/checkpoints/mayeec/$exp_id/step4765 \
        weka://oe-data-default/ai2-llm/checkpoints/mayeec/$exp_id/step4765 \
        --allow-dirty \
        --workspace ai2/dolma2
done
'


: 'experiments=(
    'olmo2-1b-1T_10b-anneal_dclm-bad-search-dd06d90e'
    'olmo2-1b-1T_10b-anneal_dclm-good-search-c1c78f25'
)

for exp_id in "${experiments[@]}"; do
  echo "Converting: $exp_id"
    python -m cookbook.remote \
        gs://ai2-llm/checkpoints/mayeec/$exp_id/step4767 \
        weka://oe-data-default/ai2-llm/checkpoints/mayeec/$exp_id/step4767 \
        --allow-dirty \
        --workspace ai2/dolma2
done


experiments=(
    'olmo2-1b-1T_10b-anneal_dclm-natural-d00b434b'
    'olmo2-1b-1T_10b-anneal_dclm-pstar-dec5c0b0'
    'olmo2-1b-1T_10b-anneal_dclm-p001-396468d9'
)

for exp_id in "${experiments[@]}"; do
  echo "Converting: $exp_id"
    python -m cookbook.remote \
        gs://ai2-llm/checkpoints/mayeec/$exp_id/step4765 \
        weka://oe-data-default/ai2-llm/checkpoints/mayeec/$exp_id/step4765 \
        --allow-dirty \
        --workspace ai2/dolma2
done'



: 'experiments=(
    'olmo2-1b-2T_1b-anneal_dclm-bad-search-f1d90621'
    'olmo2-1b-2T_1b-anneal_dclm-good-search-b59d28c6'
)

for exp_id in "${experiments[@]}"; do
  echo "Converting: $exp_id"
    python -m cookbook.remote \
        gs://ai2-llm/checkpoints/mayeec/$exp_id/step476 \
        weka://oe-data-default/ai2-llm/checkpoints/mayeec/$exp_id/step476 \
        --allow-dirty \
        --workspace ai2/dolma2
done


experiments=(
    'olmo2-1b-2T_1b-anneal_dclm-natural-f183fc9f'
    'olmo2-1b-2T_1b-anneal_dclm-pstar-0aa3f269'
    'olmo2-1b-2T_1b-anneal_dclm-p001-e6fd057b'
)

for exp_id in "${experiments[@]}"; do
  echo "Converting: $exp_id"
    python -m cookbook.remote \
        gs://ai2-llm/checkpoints/mayeec/$exp_id/step473 \
        weka://oe-data-default/ai2-llm/checkpoints/mayeec/$exp_id/step473 \
        --allow-dirty \
        --workspace ai2/dolma2
done'



: 'experiments=(
    'olmo2-1b-1T_1b-anneal_dclm-bad-search-1fa509ec'
    'olmo2-1b-1T_1b-anneal_dclm-good-search-9442fe05'
)

for exp_id in "${experiments[@]}"; do
  echo "Converting: $exp_id"
    python -m cookbook.remote \
        gs://ai2-llm/checkpoints/mayeec/$exp_id/step476 \
        weka://oe-data-default/ai2-llm/checkpoints/mayeec/$exp_id/step476 \
        --allow-dirty \
        --workspace ai2/dolma2
done


experiments=(
    'olmo2-1b-1T_1b-anneal_dclm-natural-3df2a0ad'
    'olmo2-1b-1T_1b-anneal_dclm-pstar-9dd41560'
    'olmo2-1b-1T_1b-anneal_dclm-p001-e0c71f06'
)

for exp_id in "${experiments[@]}"; do
  echo "Converting: $exp_id"
    python -m cookbook.remote \
        gs://ai2-llm/checkpoints/mayeec/$exp_id/step473 \
        weka://oe-data-default/ai2-llm/checkpoints/mayeec/$exp_id/step473 \
        --allow-dirty \
        --workspace ai2/dolma2
done
'





: 'experiments=(
    'olmo2-1b-4T-10b-anneal_dolmino-50-50-ab07de6a'
    'olmo2-1b-4T-10b-anneal_dolmino-40-60-58b35062'
    'olmo2-1b-4T-10b-anneal_dolmino-30-70-e18a66ee'
    'olmo2-1b-4T-10b-anneal_dolmino-60-40-3f5648e8'
    'olmo2-1b-4T-10b-anneal_dolmino-70-30-15672a85'
)

for exp_id in "${experiments[@]}"; do
  echo "Converting: $exp_id"
    python -m cookbook.remote \
        gs://ai2-llm/checkpoints/mayeec/$exp_id/step4769 \
        weka://oe-data-default/ai2-llm/checkpoints/mayeec/$exp_id/step4769 \
        --allow-dirty \
        --workspace ai2/dolma2
done'




: 'experiments=(
    #'olmo2-1b-4T-1b-anneal_dolmino-30-70-8a625c68'
    #'olmo2-1b-4T-1b-anneal_dolmino-50-50-87d554a0'
    #'olmo2-1b-4T-1b-anneal_dolmino-60-40-07eeb856'
    #'olmo2-1b-4T-1b-anneal_dolmino-70-30-d6b647ea'
    'olmo2-1b-4T-1b-anneal_dolmino-40-60-55f875bb'
)

for exp_id in "${experiments[@]}"; do
  echo "Converting: $exp_id"
    python -m cookbook.remote \
        gs://ai2-llm/checkpoints/mayeec/$exp_id/step477 \
        weka://oe-data-default/ai2-llm/checkpoints/mayeec/$exp_id/step477 \
        --allow-dirty \
        --workspace ai2/dolma2
done'



: 'experiments=(
    #'olmo2-1b-1B_10b-anneal_dclm-bad-search-eabbba02'
    #'olmo2-1b-1B_10b-anneal_dclm-good-search-480496d0'
    'olmo2-1b-1B_10b-anneal_dclm-natural-9e35627b'
    'olmo2-1b-1B_10b-anneal_dclm-pstar-2d88e696'
    'olmo2-1b-1B_10b-anneal_dclm-p001-2c8339f1'
)

for exp_id in "${experiments[@]}"; do
  echo "Converting: $exp_id"
    python -m cookbook.remote \
        gs://ai2-llm/checkpoints/mayeec/$exp_id/step4765 \
        weka://oe-data-default/ai2-llm/checkpoints/mayeec/$exp_id/step4765 \
        --allow-dirty \
        --workspace ai2/dolma2
done

experiments=(
    'olmo2-1b-1B_10b-anneal_dclm-bad-search-eabbba02'
    'olmo2-1b-1B_10b-anneal_dclm-good-search-480496d0'
    #'olmo2-1b-1B_10b-anneal_dclm-natural-9e35627b'
    #'olmo2-1b-1B_10b-anneal_dclm-pstar-2d88e696'
    #'olmo2-1b-1B_10b-anneal_dclm-p001-2c8339f1'
)

for exp_id in "${experiments[@]}"; do
  echo "Converting: $exp_id"
    python -m cookbook.remote \
        gs://ai2-llm/checkpoints/mayeec/$exp_id/step4767 \
        weka://oe-data-default/ai2-llm/checkpoints/mayeec/$exp_id/step4767 \
        --allow-dirty \
        --workspace ai2/dolma2
done
'



experiments=(
    'olmo2-1b-1B_1b-anneal_dclm-natural-055385d5'
    'olmo2-1b-1B_1b-anneal_dclm-p001-801d57bb'
    'olmo2-1b-1B_1b-anneal_dclm-pstar-d2582100'
)

for exp_id in "${experiments[@]}"; do
  echo "Converting: $exp_id"
    python -m cookbook.remote \
        gs://ai2-llm/checkpoints/mayeec/$exp_id/step473 \
        weka://oe-data-default/ai2-llm/checkpoints/mayeec/$exp_id/step473 \
        --allow-dirty \
        --workspace ai2/dolma2
done

experiments=(
    'olmo2-1b-1B_1b-anneal_dclm-bad-search-0b34e729'
    'olmo2-1b-1B_1b-anneal_dclm-good-search-099eeaea'
    #'olmo2-1b-1B_10b-anneal_dclm-natural-9e35627b'
    #'olmo2-1b-1B_10b-anneal_dclm-pstar-2d88e696'
    #'olmo2-1b-1B_10b-anneal_dclm-p001-2c8339f1'
)

for exp_id in "${experiments[@]}"; do
  echo "Converting: $exp_id"
    python -m cookbook.remote \
        gs://ai2-llm/checkpoints/mayeec/$exp_id/step476 \
        weka://oe-data-default/ai2-llm/checkpoints/mayeec/$exp_id/step476 \
        --allow-dirty \
        --workspace ai2/dolma2
done
