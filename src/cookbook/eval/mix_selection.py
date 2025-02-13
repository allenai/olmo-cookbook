from stats import convert_sci, compute_f1, get_sig_clusters, compute_significance
import numpy as np
from itertools import combinations
from tqdm import tqdm

# VERY ROUGH ESTIMATION PROBABLY NOT EXACT!
MODEL_FLOPS = {
    '150M': 1903391232,
    '300M': 3443922944,
    '530M': 5180751744,
    '750M': 6373843968,
    '1B': 10109071360
}

MODEL_PARAMS = {
    "150M": 190354176,
    "300M": 371262464,
    "530M": 597382464,
    "750M": 758220288,
    "1B": 1279395840,
}


def compute_flops(N, mult):
    """ Taken from Model Ladder code in OLMo """
    return MODEL_FLOPS[N] * (MODEL_PARAMS[N] * 20 * mult)


def get_mix(model):
    """ dolma17-75p-DCLM-baseline-25p-1B-5xC => dolma17-75p-DCLM-baseline-25p """
    return '-'.join(model.split('-')[:-2])


def simulate_mix_selection(df, method, sizes, models, mixes, task, mult=None, top_n_clusters=None, last_n=1, alpha=None, quiet=True):
    """ Simulate training and rejecting mixes for TASK in the order of SIZES """
    curr_mixes = mixes
    cumulative_compute = 0

    for curr_size in sizes:
        curr_models = [model for model in models if (curr_size in model) and get_mix(model) in curr_mixes]
        
        if mult is None: raise RuntimeError(f'Must specify chilchilla multiplier to compute FLOPS!')

        cumulative_compute += len(curr_models) * compute_flops(curr_size, mult)
        if len(curr_models) == 1: continue # don't compute significance table if there's only one model

        # Need to be careful with last_n because sometimes it can break...
        # last_n = 10
        # if curr_size == '150M': 
        #     print('manual override for last_n=1!')
        #     last_n = 1
        if isinstance(task, list):
            print('manual override for last_n=1!')
            last_n = 1

        # Step 1: Compute pairwise comparisons between models
        _, p_values, axes = compute_significance(
            df, 
            models=curr_models, 
            metric='acc_per_char', 
            last_n=last_n, 
            alpha=alpha, 
            tasks=[task], 
            do_plot=(not quiet), 
            quiet=quiet
        )

        task_name = task
        if isinstance(task, list): 
            task_name = 'aggregate' # TMP: WONT WORK ON OTHER TASK SETS

        mixes, scores, p_vals = p_values[task_name]

        # Step 2: Keep models in top significance cluster
        if method == 'baseline':
            # Method 1: Only keep the top performing mixes
            curr_mixes = mixes[:len(mixes) // 4]
        elif method == 'perc_sig':
            # Method 2: Keep the top significance clusters
            sig_clusters = get_sig_clusters(p_vals, alpha=alpha)
            curr_mixes = np.array(mixes)[sig_clusters < top_n_clusters].tolist()
        else:
            raise ValueError(method)

        if not quiet: print(f'Current size: {curr_size}\nMixes remaining: {curr_mixes}\n')

    pred_mixes = curr_mixes
    return pred_mixes, cumulative_compute


def run_simulations(df, sorted_sizes, task, models, mixes, top_n_clusters, top_n_clusters_eval, alpha, alpha_eval, last_n, model_pool='prefix', quiet=True):
    """ Run many different simulations on different compute setups and compute the F1 of predicting target mixes """
    results = []

    # Baseline: Train all mixes at 1B scale
    gold_mixes, gold_compute = simulate_mix_selection(
        df, 
        method='perc_sig', 
        sizes=['1B'], 
        models=models, 
        mixes=mixes, 
        task=task, 
        mult=5, 
        top_n_clusters=top_n_clusters_eval, 
        alpha=alpha_eval,
        last_n=last_n,
    )
    results += [('gold', gold_compute, (1, 1, 1))]

    if model_pool == 'prefix':
        # Get prefixes of model sizes (n runs)
        size_sets = []
        for size_idx in range(1, len(sorted_sizes)+1):
            size_sets += [sorted_sizes[:size_idx]]
    elif model_pool == 'combination':
        # Generate all combinations preserving order (2^n runs)
        size_sets = [list(sorted_sizes[:i]) for i in range(1, len(sorted_sizes) + 1)]
        for r in range(2, len(sorted_sizes)):
            size_sets += [list(combo) for combo in combinations(sorted_sizes, r)]
        size_sets.sort(key=lambda x: (len(x), x))
    else: raise ValueError(model_pool)

    for selected_sizes in tqdm(size_sets, desc=f'Running simulations for {task}'):
        if not quiet: print(selected_sizes)

        pred_mixes, pred_compute = simulate_mix_selection(
            df, 
            method='perc_sig', 
            sizes=selected_sizes, 
            models=models, 
            mixes=mixes, 
            task=task, 
            mult=5, 
            top_n_clusters=top_n_clusters, 
            alpha=alpha,
            last_n=last_n,
            quiet=quiet
        )

        # binary_outcome_gold = [mix in gold_mixes for mix in mixes] # 0/1 for all models
        # binary_outcome_pred = [mix in pred_mixes for mix in mixes] # 0/1 for all models
        # p, r, f1 = compute_f1_binary(binary_outcome_gold, binary_outcome_pred)

        p, r, f1 = compute_f1(gold_mixes, pred_mixes)

        if not quiet: print(convert_sci(gold_compute), convert_sci(pred_compute))
        if not quiet: print(f'P={p:.2f} R={r:.2f} F1={f1:.2f}\n')

        results += [(selected_sizes, pred_compute, (p, r, f1))]

    if not quiet: print(gold_mixes, pred_mixes)

    return results