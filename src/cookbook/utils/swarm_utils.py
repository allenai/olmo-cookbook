import concurrent.futures
import logging
import os
from pathlib import Path
import random
from collections import defaultdict
from typing import Tuple, Optional, List, Iterator
from copy import deepcopy
import yaml 
from scipy.spatial.distance import pdist, squareform
import math

import numpy as np
import s3fs
from olmo_core.aliases import PathOrStr
from olmo_core.utils import generate_uuid
from olmo_core.data.types import NumpyDatasetDType
from olmo_core.io import get_file_size
from tqdm import tqdm

from itertools import combinations
from math import comb, ceil


from cookbook.aliases import ExperimentConfig, SourceConfig, SwarmConfig
from cookbook.utils.data import get_token_counts_and_ratios

logger = logging.getLogger(__name__)
logging.getLogger("botocore").setLevel(logging.WARNING)


import hashlib
import json



class ConfigDefaults:
    min_strength: float = 0.1
    max_strength: float = 5.0
    sample_multiplier: int = 10
    maximum_repetition: int = 5
    minimum_weight: float = 2e-3  # 0.002


def generate_weights_grid( 
    sources: list[SourceConfig],
    leaf_distr: dict[str, float],
    num_samples_out: int,
    allow_repetition: bool,
    minimum_weight: float,
    min_source_strength: float, 
    max_source_strength: float,
    min_topic_strength: float,
    max_topic_strength: float,
    max_tokens: int,
    available_tokens: int,
    sample_multiplier: Optional[int],
    fixed_source_weights: Optional[dict[str, float]] = None,
):
    token_scale = available_tokens / max_tokens
    weight_bounds = None 

    prior_dist = np.array([v for _, v in leaf_distr.items()])
    domains = [k for k, _ in leaf_distr.items()]

    if not allow_repetition:
        weight_bounds = [
            (minimum_weight, min(prior_dist[idx] * token_scale, 1.0)) for idx in range(len(prior_dist))
        ]
        grouped_bounds = {domain: weight_bounds[idx] for idx, domain in enumerate(domains)}
        logger.info("Weight bounds:")
        logger.info(grouped_bounds)
    else:
        weight_bounds = [
            (minimum_weight, 1) for idx in range(len(prior_dist))
        ]


    fixed_topic_weights = {}
    some_fixed_topic_weights = False
    topic_priors = {}

    has_topics = any([source.topics for source in sources])

    if has_topics:
        for source in sources:
            if source.topics:
                if source.topics[0].target_ratio is not None:
                    some_fixed_topic_weights = True
                    # this source has topics with a fixed weight, so we use that weight as the prior
                    conditional_weight = np.array([[topic.target_ratio for topic in source.topics]])
                    logger.info(f"Using fixed topic weights for source '{source.name}': {conditional_weight[0]}")
                    fixed_topic_weights[source.name] = conditional_weight
                else:
                    if some_fixed_topic_weights:
                        raise NotImplementedError(
                            f"Source '{source.name}' has topics without fixed weights, but some other source has fixed topic weights. "
                            "This is not supported yet, because it is unclear how to best do grid sampling to create mixes on sources and topics simultaneously."
                        )

                topic_priors[source.name] = np.ones(len(source.topics))



    if fixed_source_weights is not None:
        fixed_source_weights = [fixed_source_weights[source_config.name] for source_config in sorted(sources, key=lambda x: x.name)]

    sample_multiplier = sample_multiplier if sample_multiplier else ConfigDefaults.sample_multiplier
    total_samples = num_samples_out * sample_multiplier

    n_strength_buckets = 15 if min_source_strength != max_source_strength else 1
    n_samples_per_strength = math.ceil(total_samples / n_strength_buckets)


    source_prior = np.ones(len(sources))
    if fixed_source_weights is not None:
        source_samples = np.array([fixed_source_weights] * total_samples)
    else:
        if min_source_strength == max_source_strength:
            source_samples = np.random.dirichlet(source_prior * min_source_strength, total_samples).tolist()
        else:
            source_samples = []
            min_source_strength_log = np.log10(min_source_strength)
            max_source_strength_log = np.log10(max_source_strength)
            for strength in np.logspace(min_source_strength_log, max_source_strength_log, n_strength_buckets):
                samples_per_strength = np.random.dirichlet(source_prior * strength, n_samples_per_strength)
                source_samples.extend(samples_per_strength.tolist())
                print(f"Generated {len(samples_per_strength)} samples for strength {strength:.2f}")


    if has_topics:
        topic_samples = defaultdict(list)
        for source, topic_prior in topic_priors.items():
            if source in fixed_topic_weights:
                # this source has fixed topic weights, so we use those
                conditional_weight = fixed_topic_weights[source]
                topic_samples[source].extend([conditional_weight] * total_samples)
            else:
                if min_topic_strength == max_topic_strength:
                    topic_samples[source].extend(np.random.dirichlet(topic_prior * min_topic_strength, total_samples).tolist())
                else:
                    min_topic_strength_log = np.log10(min_topic_strength)
                    max_topic_strength_log = np.log10(max_topic_strength)
                    for strength in np.logspace(min_topic_strength_log, max_topic_strength_log, n_strength_buckets):
                        samples_per_strength = np.random.dirichlet(topic_prior * strength, n_samples_per_strength)
                        topic_samples[source].extend(samples_per_strength.tolist())

    # convert from source_samples and topic_samples back to a set of leaf-level samples 
    candidates = []

    if has_topics:
        for i, source_sample in enumerate(source_samples):
            leaf_level_sample = {source: samples[i][0]*source_sample[0, j] for j, (source, samples) in enumerate(topic_samples.items())}
            flattened_sample = np.concatenate([arr for arr in list(leaf_level_sample.values())]).reshape(1, -1)
            candidates.append(flattened_sample)
    else:
        candidates.extend(source_samples)

    filtered_candidates = np.array([
        sample
        for sample in candidates
        if all(
            lower <= sample[idx] <= upper
            for idx, (lower, upper) in enumerate(weight_bounds)
        )
    ])

    # Step 3: Hierarchically merge points
    if len(filtered_candidates) < num_samples_out:
        raise ValueError(f"Not enough candidates after filtering: {len(filtered_candidates)} < {num_samples_out}. "
                            "Consider increasing sample_multiplier.") 

    while len(filtered_candidates) > num_samples_out:
        distances = squareform(pdist(filtered_candidates))
        np.fill_diagonal(distances, np.inf)
        closest_pair = np.unravel_index(np.argmin(distances), distances.shape)
        midpoint = (filtered_candidates[closest_pair[0]] + filtered_candidates[closest_pair[1]]) / 2
        filtered_candidates = np.delete(filtered_candidates, closest_pair, axis=0)
        filtered_candidates = np.vstack([filtered_candidates, midpoint])


    selected_samples = [(candidate, np.ones(len(candidates)))for candidate in filtered_candidates]
    if allow_repetition:
        selected_samples = [] 
        for candidate in filtered_candidates:
            reps = []
            for idx, _ in enumerate(domains):
                available_tokens = int(prior_dist[idx] * available_tokens)
                required_tokens = int(candidate[idx] * max_tokens)

                repetition = (
                    np.ceil(required_tokens / available_tokens * 1000) / 1000
                    if available_tokens != 0
                    else 0
                )

                if repetition > ConfigDefaults.maximum_repetition:
                    break

                reps.append(max(1, repetition))

            selected_samples.append((candidate, np.array(reps)))
    return selected_samples

def generate_weights_dirichlet(
    sources: list[SourceConfig], # flat 
    leaf_dist: dict[str, float],
    minimum_source_weight: float,
    minimum_topic_weight: float,
    num_samples_out: int,
    source_temperature: float,
    topic_temperature: float,
    min_source_strength: float, 
    max_source_strength: float,
    min_topic_strength: float,
    max_topic_strength: float,
    max_tokens: int,
    available_tokens: int,
    allow_repetition: bool,
    manual_prior: Optional[dict[str, float]],
    sample_multiplier: Optional[int],
    enable_bound: bool = True,
    nonzero_weight: Optional[list[str]] = None,
    fixed_source_weights: Optional[dict[str, float]] = None
):
    """
    Generate weights for each domain group using a dirichlet distribution.
    """

    token_scale = available_tokens / max_tokens
    logger.info(f"Source token population is {token_scale:.2f}:1 target population.")

    collected_samples: list[Tuple[np.ndarray, np.ndarray]] = []
    weight_bounds = None


    prior_dist = np.array([v for _, v in leaf_dist.items()])
    logger.info(f"Dimension of leaf-level distribution: {len(prior_dist)}")
    domains = [k for k, _ in leaf_dist.items()]
    source_names = [source.name for source in sources]
    idx_to_level = ["source" if name in source_names else "topic" for name in leaf_dist]

    if enable_bound:
        # weight bounds are at the leaf level and computed using the number of available tokens per source/topic.
        weight_bounds = [
            (0.0, min(prior_dist[idx] * token_scale, 1.0)) for idx in range(len(prior_dist))
        ]
        grouped_bounds = {domain: weight_bounds[idx] for idx, domain in enumerate(domains)}
        logger.info("Weight bounds:")
        logger.info(grouped_bounds)


    # split prior distribution into source and topic distributions and tweak it according to the manual prior
    topic_distributions = {}
    source_distribution = []
    for source_config in sorted(sources, key=lambda x: x.name):
        if source_config.topics:
            # this source has topics 
            weights = np.array([leaf_dist[f"{source_config.name}:{topic.name}"] for topic in source_config.topics])
            normalized_weights = weights / weights.sum()
            topic_distributions[source_config.name] = normalized_weights 

            if manual_prior is not None and source_config.name in manual_prior:
                source_distribution.append(manual_prior[source_config.name])
            else:
                source_distribution.append(weights.sum())
        else:
            # this source does not have topics 
            topic_distributions[source_config.name] = np.array([1.0])
            if manual_prior is not None and source_config.name in manual_prior:
                source_distribution.append(manual_prior[source_config.name])
            else:
                source_distribution.append(leaf_dist[source_config.name])

    source_distribution = np.array(source_distribution)
    source_distribution /= source_distribution.sum()

    logger.info(f"Source prior: {source_distribution}")
    logger.info(f"Topic prior: {topic_distributions}")
    if source_temperature < 1.0:
        source_prior = source_distribution**source_temperature
        source_prior = source_prior / np.sum(source_prior)
        logger.info(f"Source prior after temperature scaling: {source_prior}")
    else:
        source_prior = source_distribution

    if topic_temperature < 1.0:
        topic_priors = deepcopy(topic_distributions)
        for source, topic_prior in topic_priors.items():
            topic_prior = topic_prior**topic_temperature
            topic_prior = topic_prior / np.sum(topic_prior)
            topic_priors[source] = topic_prior
        logger.info(f"Topic priors after temperature scaling: {topic_priors}")
    else:
        topic_priors = deepcopy(topic_distributions)

    if not allow_repetition and weight_bounds:
        logger.info("Limiting candidates to within bounds, repetition is disabled...")

    fixed_topic_weights = {}
    for source in sources:
        if source.topics:
            if source.topics[0].weight is not None:
                # this source has topics with a fixed weight, so we use that weight as the prior
                conditional_weight = np.array([[topic.weight for topic in source.topics]])
                logger.info(f"Using fixed topic weights for source '{source.name}': {conditional_weight[0]}")
                fixed_topic_weights[source.name] = conditional_weight

    sample_multiplier = sample_multiplier if sample_multiplier else ConfigDefaults.sample_multiplier

    if fixed_source_weights is not None:
        fixed_source_weights = [fixed_source_weights[source_config.name] for source_config in sorted(sources, key=lambda x: x.name)]

    for _ in tqdm(range(num_samples_out * sample_multiplier)):
        candidates = []

        # first, generate source-level weights
        if min_source_strength == max_source_strength:
            if fixed_source_weights is not None:
                # if we have fixed source weights, we use those
                source_samples = np.array([fixed_source_weights])
            else:
                source_samples = np.random.dirichlet(source_prior * min_source_strength, 1)
        else:
            source_samples = []
            if fixed_source_weights is not None:
                for _ in range(15):
                    source_samples.append(np.array([fixed_source_weights]))
            else:
                min_source_strength_log = np.log10(min_source_strength)
                max_source_strength_log = np.log10(max_source_strength)
                for strength in np.logspace(min_source_strength_log, max_source_strength_log, 15):
                    samples_per_strength = np.random.dirichlet(source_prior * strength, 1)
                    source_samples.append(samples_per_strength)


        # then, generate topic-level weights
        topic_samples = defaultdict(list)
        for source, topic_prior in topic_priors.items():
            if source in fixed_topic_weights:
                # this source has fixed topic weights, so we use those
                conditional_weight = fixed_topic_weights[source]
                if min_topic_strength == max_topic_strength:
                    topic_samples[source].append(conditional_weight)
                else:
                    for _ in range(15):
                        topic_samples[source].append(conditional_weight)
                continue

            if min_topic_strength == max_topic_strength:
                topic_samples[source].append(np.random.dirichlet(topic_prior * min_topic_strength, 1))
            else:
                min_topic_strength_log = np.log10(min_topic_strength)
                max_topic_strength_log = np.log10(max_topic_strength)
                for strength in np.logspace(min_topic_strength_log, max_topic_strength_log, 15):
                    samples_per_strength = np.random.dirichlet(topic_prior * strength, 1)
                    topic_samples[source].append(samples_per_strength)

        # convert from source_samples and topic_samples back to a set of leaf-level samples 
        candidates = []
        for i, source_sample in enumerate(source_samples):
            leaf_level_sample = {source: samples[i][0]*source_sample[0, j] for j, (source, samples) in enumerate(topic_samples.items())}
            flattened_sample = np.concatenate([arr for arr in list(leaf_level_sample.values())]).reshape(1, -1)
            candidates.append(flattened_sample)
            
        filtered_candidates = []

        # If we don't allow repetition, we need to filter out candidates that are outside the bounds
        if weight_bounds and not allow_repetition:
            filtered_candidates = [
                sample
                for sample in candidates
                if all(
                    lower <= sample[0][idx] <= upper
                    for idx, (lower, upper) in enumerate(weight_bounds)
                )
            ]
        else:
            filtered_candidates = candidates

        if nonzero_weight:
            source_names = set(nonzero_weight)
            # Filter candidates
            filtered_candidates = [
                sample for sample in filtered_candidates
                if sample_has_required_sources_and_topics(sample[0], domains, source_names, minimum_source_weight, minimum_topic_weight)
            ]

        if not filtered_candidates:
            # logger.warning("No candidates left after filtering according to weight bounds and nonzero weights!")
            continue

        candidates = random.choice(filtered_candidates)

        if minimum_source_weight == minimum_topic_weight:
            candidates = np.where(candidates < minimum_source_weight, 0, candidates)
            candidates = candidates / np.sum(candidates).reshape(-1, 1)
            candidates = np.round(candidates / minimum_source_weight) * minimum_source_weight
            candidates = candidates / np.sum(candidates)
        else:
            candidates = clip_candidates_by_level(
                candidates,
                idx_to_level,
                domains,
                minimum_source_weight,
                minimum_topic_weight,
                fixed_topic_weights
            )

        if weight_bounds and not allow_repetition:
            # need to check for out-of-bounds candidates again, in case normalization caused bounds to be violated.
            if any(candidates[0][idx] < lower or candidates[0][idx] > upper for idx, (lower, upper), in enumerate(weight_bounds)):
                continue

        selected: Tuple[np.ndarray, np.ndarray] = (
            candidates[0],
            np.ones(candidates.shape[1]),
        )

        reject = False
        if allow_repetition:
            for idx, _ in enumerate(domains):
                available_tokens = int(prior_dist[idx] * available_tokens)
                required_tokens = int(selected[0][idx] * max_tokens)

                repetition = (
                    np.ceil(required_tokens / available_tokens * 1000) / 1000
                    if available_tokens != 0
                    else 0
                )

                if repetition > ConfigDefaults.maximum_repetition:
                    reject = True
                    break

                selected[1][idx] = max(1, repetition)

        if not reject:
            collected_samples.append(selected)

    if len(collected_samples) == 0:
        raise ValueError("No valid samples were generated, please check the configuration!")

    if len(collected_samples) > 10000:
        # when we have a lot of samples, regular sort_and_deduplicate is O(n^2) and takes too long
        deduped = sort_and_deduplicate_with_hash(collected_samples)
    else:
        deduped = sort_and_deduplicate(collected_samples)

    if len(collected_samples) < num_samples_out:
        raise ValueError(
            f"The number of collected samples '{len(collected_samples)}' is less than the required number of samples '{num_samples_out}'!"
        )

    selected_samples = random.sample(deduped, num_samples_out)
    selected_samples = np.stack(selected_samples, axis=0)

    logger.info(f"Number of nonzero domains per swarm run: ")
    print([len(np.where(selected_samples[i][0] != 0)[0]) for i in range(len(selected_samples))])

    all_diffs = []
    for i in range(len(selected_samples)):
        for j in range(i + 1, len(selected_samples)):
            diff = np.linalg.norm(selected_samples[i][0] - selected_samples[j][0])
            if diff < 0.01:
                logger.info(f"Sample {i} and Sample {j} are too close to each other!")
                logger.info(f"Sample {i}: {selected_samples[i][0]}")
                logger.info(f"Sample {j}: {selected_samples[j][0]}")
            all_diffs.append(diff)
            
    return selected_samples



def mk_mixtures(
    config: SwarmConfig, use_cache: bool = True
) -> list[dict[str, Tuple[float, float]]]:
    random.seed(config.seed)
    np.random.seed(config.seed)

    num_samples = config.variants
    sources = config.dataset.sources
    leaf_distr, available_tokens = get_token_counts_and_ratios(
        sources, config.dataset.dtype, use_cache=use_cache
    )

    leaf_items = list(leaf_distr.items())
    domains = [k for k, _ in leaf_items]

    min_source_strength = config.min_source_strength if config.min_source_strength else config.min_strength
    max_source_strength = config.max_source_strength if config.max_source_strength else config.max_strength

    min_topic_strength = config.min_topic_strength if config.min_topic_strength else config.min_strength
    max_topic_strength = config.max_topic_strength if config.max_topic_strength else config.max_strength

    mixtures = generate_weights_grid(
        sources=sources,
        leaf_distr=leaf_distr,
        num_samples_out=num_samples,
        allow_repetition=config.allow_repetition,
        minimum_weight=config.minimum_weight,
        min_source_strength=min_source_strength,
        max_source_strength=max_source_strength,
        min_topic_strength=min_topic_strength,
        max_topic_strength=max_topic_strength,
        max_tokens=config.max_tokens,
        available_tokens=available_tokens,
        fixed_source_weights=config.fixed_source_weights,
        sample_multiplier=config.sample_multiplier,
    )

    weight_maps = []
    for mix in mixtures:
        weight_map = {}
        for idx in range(len(domains)):
            weight_map[domains[idx]] = (mix[0][idx], mix[1][idx])

        weight_maps.append(weight_map)

    for i in range(len(domains)):
        if ':' in domains[i]:
            weights = np.array([mix[0][i] for mix in mixtures])
            logger.info(f"Topic {domains[i]}, min: {weights.min()}, max: {weights.max()}")


    source_to_indices = defaultdict(list)
    for i, domain in enumerate(domains):
        source = domain.split(':', 1)[0] 
        source_to_indices[source].append(i)

    for source, indices in source_to_indices.items():
        source_weights = []
        for mix in mixtures:
            total = sum(mix[0][i] for i in indices)
            source_weights.append(total)
        source_weights = np.array(source_weights)
        logger.info(f"Source {source}, min: {source_weights.min()}, max: {source_weights.max()}")


    return weight_maps

def prettify_mixes(mixes: list[dict[str, Tuple[float, float]]]):
    result = {"mixes": mixes}
    return json.dumps(result, indent=2)


def mk_mixes(
    config: SwarmConfig, group_uuid: str, output: Optional[Path] = None, use_cache: bool = True
) -> list[dict[str, Tuple[float, float]]]:
    mixes = mk_mixtures(config, use_cache=use_cache)
    mix_string = prettify_mixes(mixes)

    if not output:
        output = Path(f"cache/swarms/{config.name}_{group_uuid}.json")

    if output:
        os.makedirs(os.path.dirname(output), exist_ok=True)

        with open(output, "w") as f:
            f.write(mix_string)

        logger.info(f"Mixes saved to {output}:")

    display_mixes = deepcopy(mixes)

    nested_mixes = []
    for mix in display_mixes:
        mix = {k: v for k, v in mix.items() if v[0] > 0}

        # Organize display source → topic → weight
        source_totals = defaultdict(float)
        source_topics = defaultdict(dict)
        for domain, (weight, _) in mix.items():
            if ":" in domain:
                source, topic = domain.split(":", 1)
                source_totals[source] += weight
                source_topics[source][topic] = weight
            else:
                source_totals[domain] += weight

        nested = {}
        for source in source_totals:
            if source in source_topics:
                nested[source] = {"total": source_totals[source], "topics": source_topics[source]}
            else:
                nested[source] = source_totals[source]

        nested_mixes.append(nested)
    logger.info(nested_mixes)
    return mixes