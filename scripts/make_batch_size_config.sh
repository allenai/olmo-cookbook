#!/bin/bash

CHECKPOINTS="/weka/oe-training-default/ai2-llm/checkpoints"

mkdir src/cookbook/recipes/batch-size

STEP=0
mkdir src/cookbook/recipes/batch-size/step$STEP
for multiplier in 0.125 0.25 0.5 1 2 3 4; do
    python scripts/make_batch_size_config.py \
        "src/cookbook/recipes/love2code/train-1b-5xC-love2code-weka-python-no-prose-hlr.yaml" \
        "src/cookbook/recipes/batch-size/step$STEP/${multiplier}x.yaml" \
        --name "code-step$STEP-${multiplier}x" \
        --load-path $CHECKPOINTS/ai2-tylerm/olmo-cookbook-1b-5xC-love2code-python-no-prose-hlr-c0c0f2d1/step$STEP/ \
        --multiplier $multiplier \
        --start-step $STEP \
        --nodes 1
done

STEP=10000
mkdir src/cookbook/recipes/batch-size/step$STEP
for multiplier in 0.125 0.25 0.5 1 2 3 4; do
    python scripts/make_batch_size_config.py \
        "src/cookbook/recipes/love2code/train-1b-5xC-love2code-weka-python-no-prose-hlr.yaml" \
        "src/cookbook/recipes/batch-size/step$STEP/${multiplier}x.yaml" \
        --name "code-step$STEP-${multiplier}x" \
        --load-path $CHECKPOINTS/ai2-tylerm/olmo-cookbook-1b-5xC-love2code-python-no-prose-hlr-c0c0f2d1/step$STEP/ \
        --multiplier $multiplier \
        --start-step $STEP \
        --nodes 1
done

STEP=20000
mkdir src/cookbook/recipes/batch-size/step$STEP
for multiplier in 0.5 1 2 3 4; do
    python scripts/make_batch_size_config.py \
        "src/cookbook/recipes/love2code/train-1b-5xC-love2code-weka-python-no-prose-hlr.yaml" \
        "src/cookbook/recipes/batch-size/step$STEP/${multiplier}x.yaml" \
        --name "code-step$STEP-${multiplier}x" \
        --load-path $CHECKPOINTS/ai2-tylerm/olmo-cookbook-1b-5xC-love2code-python-no-prose-hlr-c0c0f2d1/step$STEP/ \
        --multiplier $multiplier \
        --start-step $STEP \
        --nodes 1
done

STEP=30000
mkdir src/cookbook/recipes/batch-size/step$STEP
for multiplier in 1 2 4 8; do
    python scripts/make_batch_size_config.py \
        "src/cookbook/recipes/love2code/train-1b-5xC-love2code-weka-python-no-prose-hlr.yaml" \
        "src/cookbook/recipes/batch-size/step$STEP/${multiplier}x.yaml" \
        --name "code-step$STEP-${multiplier}x" \
        --load-path $CHECKPOINTS/ai2-tylerm/olmo-cookbook-1b-5xC-love2code-python-no-prose-hlr-c0c0f2d1/step$STEP/ \
        --multiplier $multiplier \
        --start-step $STEP \
        --nodes 1
done

STEP=40000
mkdir src/cookbook/recipes/batch-size/step$STEP
for multiplier in 1 2 4 8; do
    python scripts/make_batch_size_config.py \
        "src/cookbook/recipes/love2code/train-1b-5xC-love2code-weka-python-no-prose-hlr.yaml" \
        "src/cookbook/recipes/batch-size/step$STEP/${multiplier}x.yaml" \
        --name "code-step$STEP-${multiplier}x" \
        --load-path $CHECKPOINTS/ai2-tylerm/olmo-cookbook-1b-5xC-love2code-python-no-prose-hlr-c0c0f2d1/step$STEP/ \
        --multiplier $multiplier \
        --start-step $STEP \
        --nodes 1
done

STEP=50000
mkdir src/cookbook/recipes/batch-size/step$STEP
for multiplier in 1 2 4 8; do
    python scripts/make_batch_size_config.py \
        "src/cookbook/recipes/love2code/train-1b-5xC-love2code-weka-python-no-prose-hlr.yaml" \
        "src/cookbook/recipes/batch-size/step$STEP/${multiplier}x.yaml" \
        --name "code-step$STEP-${multiplier}x" \
        --load-path $CHECKPOINTS/ai2-tylerm/olmo-cookbook-1b-5xC-love2code-python-no-prose-hlr-c0c0f2d1/step$STEP/ \
        --multiplier $multiplier \
        --start-step $STEP \
        --nodes 1
done