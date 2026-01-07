import concurrent.futures
import logging
import os
from pathlib import Path
import random
from collections import defaultdict
from typing import Tuple, Optional, List, Iterator, Dict
from copy import deepcopy
import yaml 
from scipy.spatial.distance import pdist, squareform
import math
import re
import matplotlib.pyplot as plt 

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


def generate_weights(
    sources: List[SourceConfig],
    leaf_distr: Dict[str, float],
    num_samples_out: int,
    allow_repetition: bool,
    minimum_weight: float,
    min_source_strength: float, 
    max_source_strength: float,
    min_topic_strength: float,
    max_topic_strength: float,
    max_tokens: int,
    available_tokens: int,
    mix_temperature: float,
    sample_multiplier: Optional[int],
    fixed_source_weights: Optional[Dict[str, float]] = None,
    manual_prior: Optional[Dict[str, float]] = None
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
        for source in sources: #sorted(sources, key=lambda x: x.name):
            if source.topics:
                if source.topics[0].target_ratio is not None:
                    some_fixed_topic_weights = True
                    # this source has topics with a fixed weight, so we use that weight as the prior
                    conditional_weight = np.array([topic.target_ratio for topic in source.topics])
                    logger.info(f"Using fixed topic weights for source '{source.name}': {conditional_weight[0]}")
                    fixed_topic_weights[source.name] = conditional_weight
                else:
                    if some_fixed_topic_weights:
                        raise NotImplementedError(
                            f"Source '{source.name}' has topics without fixed weights, but some other source has fixed topic weights. "
                            "This is not supported yet, because it is unclear how to best do grid sampling to create mixes on sources and topics simultaneously."
                        )
                # uniform topic prior 
                topic_priors[source.name] = np.ones(len(source.topics))
            else:
                topic_priors[source.name] = np.array([1.0])  # no topics, so we use a uniform prior



    if fixed_source_weights is not None:
        fixed_source_weights = [fixed_source_weights[source_config.name] for source_config in sources] #sorted(sources, key=lambda x: x.name)]

    sample_multiplier = sample_multiplier if sample_multiplier else ConfigDefaults.sample_multiplier
    total_samples = num_samples_out * sample_multiplier

    n_strength_buckets = 15 if min_source_strength != max_source_strength else 1
    n_samples_per_strength = math.ceil(total_samples / n_strength_buckets)

    if manual_prior is not None:
        source_prior = np.array([manual_prior.get(source_config.name, 0.0) for source_config in sources]) #sorted(sources, key=lambda x: x.name)

        if mix_temperature < 1.0:
            source_prior = source_prior**mix_temperature
            source_prior = source_prior / np.sum(source_prior)
            logger.info(f"Source prior after temperature scaling: {source_prior}")
    else:
        source_prior = np.ones(len(sources))
    if fixed_source_weights is not None:
        source_samples = np.array([fixed_source_weights] * total_samples)
    else:
        if min_source_strength == max_source_strength:
            source_samples = np.random.dirichlet(source_prior * min_source_strength, total_samples)
        else:
            source_samples = []
            min_source_strength_log = np.log10(min_source_strength)
            max_source_strength_log = np.log10(max_source_strength)
            for strength in np.logspace(min_source_strength_log, max_source_strength_log, n_strength_buckets):
                samples_per_strength = np.random.dirichlet(source_prior * strength, n_samples_per_strength)
                source_samples.extend(samples_per_strength)
                print(f"Generated {len(samples_per_strength)} samples for strength {strength:.2f}")

    total_samples = len(source_samples)
    if has_topics:
        topic_samples = defaultdict(list)
        for source, topic_prior in topic_priors.items():
            if source in fixed_topic_weights:
                # this source has fixed topic weights, so we use those
                conditional_weight = fixed_topic_weights[source]
                topic_samples[source].extend(np.array([conditional_weight] * total_samples))
            else:
                if topic_prior == np.array([1]):
                    topic_samples[source].extend([topic_prior] * total_samples)
                elif min_topic_strength == max_topic_strength:
                    topic_samples[source].extend(np.random.dirichlet(topic_prior * min_topic_strength, total_samples))
                else:
                    min_topic_strength_log = np.log10(min_topic_strength)
                    max_topic_strength_log = np.log10(max_topic_strength)
                    for strength in np.logspace(min_topic_strength_log, max_topic_strength_log, n_strength_buckets):
                        samples_per_strength = np.random.dirichlet(topic_prior * strength, n_samples_per_strength)
                        topic_samples[source].extend(samples_per_strength)
    # convert from source_samples and topic_samples back to a set of leaf-level samples 
    candidates = []

    if has_topics:
        for i, source_sample in enumerate(source_samples):
            leaf_level_sample = {source: samples[i]*source_sample[j] for j, (source, samples) in enumerate(topic_samples.items())}
            flattened_sample = np.concatenate([arr for arr in list(leaf_level_sample.values())])
            candidates.append(flattened_sample)
    else:
        candidates.extend(source_samples)
    logger.info(f"Generated {len(candidates)} candidate samples before filtering.")
    filtered_candidates = np.array([
        sample
        for sample in tqdm(candidates)
        if all(
            lower <= sample[idx] <= upper
            for idx, (lower, upper) in enumerate(weight_bounds)
        )
    ])

    # Step 3: Hierarchically merge points
    if len(filtered_candidates) < num_samples_out:
        raise ValueError(f"Not enough candidates after filtering: {len(filtered_candidates)} < {num_samples_out}. "
                            "Consider increasing sample_multiplier.") 

    """while len(filtered_candidates) > num_samples_out:
        distances = squareform(pdist(filtered_candidates))
        np.fill_diagonal(distances, np.inf)
        closest_pair = np.unravel_index(np.argmin(distances), distances.shape)
        midpoint = (filtered_candidates[closest_pair[0]] + filtered_candidates[closest_pair[1]]) / 2
        filtered_candidates = np.delete(filtered_candidates, closest_pair, axis=0)
        filtered_candidates = np.vstack([filtered_candidates, midpoint])
    """

    from sklearn.cluster import AgglomerativeClustering

    clustering = AgglomerativeClustering(n_clusters=num_samples_out, linkage='average')
    labels = clustering.fit_predict(filtered_candidates)

    # Use cluster centroids as representatives
    filtered_candidates = np.array([
        filtered_candidates[labels == i].mean(axis=0)
        for i in range(num_samples_out)
    ])

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

def mk_mixtures(
    config: SwarmConfig, group_uuid: str, use_cache: bool = True
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

    mixtures = generate_weights(
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
        manual_prior=config.manual_prior,
        mix_temperature=config.mix_temperature
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
        
        out_dir = Path("cache") / "swarms" / str(group_uuid)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize source to be filesystem-friendly
        safe_source = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(source))
        out_path = out_dir / f"{safe_source}_source_weights_hist.png"

        # Plot histogram
        plt.figure(figsize=(8, 5))
        plt.hist(source_weights[~np.isnan(source_weights)], bins=10)
        plt.title(f"{source}")
        plt.xlabel("Weight")
        plt.ylabel("Frequency")
        plt.tight_layout()

        # Save & close
        plt.savefig(out_path, dpi=200)
        plt.close()

    return weight_maps

def prettify_mixes(mixes: list[dict[str, Tuple[float, float]]]):
    result = {"mixes": mixes}
    return json.dumps(result, indent=2)


def mk_mixes(
    config: SwarmConfig, group_uuid: str, output: Optional[Path] = None, use_cache: bool = True
) -> list[dict[str, Tuple[float, float]]]:
    mixes = mk_mixtures(config, group_uuid, use_cache=use_cache)
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