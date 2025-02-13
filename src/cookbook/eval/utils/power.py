# Adapted from: https://github.com/dallascard/NLP-power-analysis
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

np.random.seed(42)

def compute_power(prob_table: np.ndarray, dataset_size: int, alpha: float = 0.05, r: int = 5000) -> tuple[float, float, float, float]:
    """
    Compute statistical power and related metrics using Monte Carlo simulation.

    prob_table (np.array): 2x2 contingency table of probabilities.
    dataset_size (int): Size of the dataset to simulate.
    alpha (float): Significance level for hypothesis testing.
    r (int): Number of simulations to run.

    Returns:
    - power: The statistical power (probability of correctly rejecting the null hypothesis).
    - mean_effect: The average effect size observed across simulations.
    - type_m: The Type M error (exaggeration ratio).
    - type_s: The Type S error (probability of obtaining a significant result in the wrong direction).
    """
    if prob_table[0, 1] == prob_table[1, 0]:
        raise RuntimeError("Power is undefined when the true effect is zero.")

    # Perform Monte Carlo simulations
    pvals, diffs = simulate_samples(prob_table, dataset_size, r)

    # Calculate true difference and its sign
    true_diff = prob_table[0, 1] - prob_table[1, 0]
    true_sign = np.sign(true_diff)

    # Compute power and related metrics
    sig_diffs = [d for p, d in zip(pvals, diffs) if p <= alpha]
    power = sum(p <= alpha and np.sign(d) == true_sign for p, d in zip(pvals, diffs)) / r
    mean_effect = np.mean(diffs)
    type_m = np.mean(np.abs(sig_diffs) / np.abs(true_diff))
    type_s = np.mean(np.sign(sig_diffs) != true_sign)

    return power, mean_effect, type_m, type_s


def simulate_samples(prob_table: np.ndarray, dataset_size: int, r: int) -> tuple[list[float], list[float]]:
    """
    Simulate samples and compute p-values and accuracy differences.

    prob_table (np.array): 2x2 contingency table of probabilities.
    dataset_size (int): Size of the dataset to simulate.
    r (int): Number of simulations to run.

    Returns:
    - pvals: lists of p-values
    - diffs: accuracy differences.
    """
    pvals = []
    diffs = []
    for _ in range(r):
        # Generate sample from multinomial distribution
        sample = np.random.multinomial(n=dataset_size, pvals=prob_table.reshape((4,))).reshape((2,2))
        
        # Compute accuracy difference
        acc_diff = (sample[0,1] - sample[1, 0]) / dataset_size
        
        # Perform McNemar's test
        test_results = mcnemar(sample)
        
        pvals.append(test_results.pvalue)
        diffs.append(acc_diff)
    
    return pvals, diffs


def print_probability_table(prob_table, agreement_rate):
    """ Print the probability table """
    print(f"┌─────────────────────────────────────────────────────────────┐")
    print(f"│             │    M2 Incorrect   │    M2 Correct    │  Sum   │")
    print(f"├─────────────┼───────────────────┼──────────────────┼────────┤")
    print(f"│M1 Incorrect │ {prob_table[0,0]:^17.3f} │ {prob_table[0,1]:^16.3f} │ {prob_table[0,:].sum():^5.3f}  │")
    print(f"│M1 Correct   │ {prob_table[1,0]:^17.3f} │ {prob_table[1,1]:^16.3f} │ {prob_table[1,:].sum():^5.3f}  │")
    print(f"├─────────────┼───────────────────┼──────────────────┼────────┤")
    print(f"│Sum          │ {prob_table[:,0].sum():^17.3f} │ {prob_table[:,1].sum():^16.3f} │ {prob_table.sum():^5.3f}  │")
    print(f"└─────────────────────────────────────────────────────────────┘")
    # print(f"Agreement rate (diagonal entries) = {agreement_rate:.2f}")


def get_prob_table(acc1, acc2, agreement_rate) -> np.ndarray:
    Δacc = acc2 - acc1
    disagreement_rate = 1 - agreement_rate
    if Δacc > 0:
        p_only_1_correct = (disagreement_rate - Δacc) / 2
        p_only_2_correct = (disagreement_rate - Δacc) / 2 + Δacc
    else:
        p_only_1_correct = (disagreement_rate + Δacc) / 2 - Δacc
        p_only_2_correct = (disagreement_rate + Δacc) / 2

    p_both_correct = acc1 - p_only_1_correct
    assert np.abs(p_both_correct - (acc2 - p_only_2_correct)) < 1e-4
    p_both_incorrect = 1. - p_both_correct - p_only_1_correct - p_only_2_correct

    for p in [p_both_correct, p_only_1_correct, p_only_2_correct, p_both_incorrect]: 
        # if p < 0: return float('inf'), float('inf'), float('inf'), float('inf')
        assert p >= 0, [p_both_correct, p_only_1_correct, p_only_2_correct, p_both_incorrect]

    agreement_rate = p_both_correct + p_both_incorrect
    prob_table = np.array([[p_both_incorrect, p_only_2_correct], [p_only_1_correct, p_both_correct]])
    return prob_table


def run_power_test(acc1, acc2, agreement_rate, n_samples, α=0.05, r=1000, quiet=False):
    prob_table = get_prob_table(acc1, acc2, agreement_rate)
    power, mean_effect, type_m, type_s = compute_power(prob_table, n_samples, alpha=α, r=r)

    if not quiet: 
        print("="*50 + f"\Contingency table (agreement rate={agreement_rate:.2f}, # samples={n_samples})\n" + "="*50)
        print_probability_table(prob_table, agreement_rate)
        print(f"\nResults for agreement rate = {agreement_rate:.3f} and α = {α}:")
        print(f"Approx mean effect (Δacc) = {mean_effect:.3f} ({100*mean_effect:.1f}%)")
        print(f"Approx power = {power:.4f} ({100*power:.1f}%)")
        print(f"Approx Type-M error = {type_m:.3f}")
        print(f"Approx Type-S error = {type_s:.3f}")

    return power, mean_effect, type_m, type_s


def run_mcnemar(acc1, acc2, agreement_rate, n_samples):
    prob_table = get_prob_table(acc1, acc2, agreement_rate)

    prob_table = (prob_table * n_samples).astype(np.int64)

    acc_diff = (prob_table[0,1] - prob_table[1, 0]) / n_samples
    result = mcnemar(prob_table)

    return acc_diff, result.pvalue


def compute_pairwise_mcnemar(seg_scores, return_scores=False):
    """ Computes pairwise p-values between a set of scores """
    num_systems, num_segments = seg_scores.shape
    seg_scores = seg_scores.astype(np.float32)
    sys_scores = np.sum(seg_scores, axis=1)

    p_vals = np.empty((num_systems, num_systems)) * np.nan
    
    for i in range(num_systems):
        for j in range(i + 1, num_systems):
            n = num_segments
            acc1, acc2 = np.mean(seg_scores[i, :]), np.mean(seg_scores[j, :])
            agreement_rate = np.sum(seg_scores[i, :] == seg_scores[j, :]) / n
            try:
                _, p_value = run_mcnemar(acc1, acc2, agreement_rate, n)
            except Exception as e:
                print(e)
                p_value = 1
            p_vals[i, j] = p_value
    
    if return_scores:
        return p_vals, sys_scores / seg_scores.shape[1], None
    return p_vals


def calculate_mde(baseline_acc: float, agreement_rate: float, n_samples: int, target_power: float = 0.8, tolerance: float = 1e-4) -> float:
    """ Find the minimum detectable effect (MDE) w/ binary search """
    def calculate_power(Δacc):
        power, _, _, _ = run_power_test(baseline_acc, baseline_acc + Δacc, agreement_rate, n_samples, quiet=True)
        return power
    
    low, high = 0, 1 - baseline_acc
    while high - low > tolerance:
        mid = (low + high) / 2
        power = calculate_power(mid)
        if power < target_power:
            low = mid
        else:
            high = mid
    
    return high


def demo():
    # Set up parameters
    BASELINE_ACC = 0.7  # Accuracy of the baseline model
    DELTA_ACC = 0.02    # Difference in accuracy between the two models
    DATASET_SIZE = 1000 # Number of samples in the dataset
    r = 10000           # Number of Monte Carlo simulations
    α = 0.05            # Significance level for hypothesis testing

    # Calculate accuracies for both models
    acc1 = BASELINE_ACC
    acc2 = BASELINE_ACC + DELTA_ACC

    #### Upper bound: Maximal agreement on instances
    p_both_correct = min(acc1, acc2)    # Probability of both models being correct
    p_diff = abs(acc1 - acc2)           # Absolute difference in accuracies
    p_both_incorrect = 1.0 - max(acc1, acc2)  # Probability of both models being incorrect
    agreement_rate = p_both_correct + p_both_incorrect  # Overall agreement rate

    # Create probability table [[both incorrect, only M1 correct], [only M2 correct, both correct]]
    if acc2 > acc1:
        prob_table = np.array([[p_both_incorrect, 0], [p_diff, p_both_correct]])
    else:
        prob_table = np.array([[p_both_incorrect, p_diff], [0, p_both_correct]])

    print("="*50 + "\nProbability table for maximal agreement\n" + "="*50)
    print_probability_table(prob_table, agreement_rate)

    # Compute power and related metrics
    power, mean_effect, type_m, type_s = compute_power(prob_table, DATASET_SIZE, alpha=α, r=r)

    # Print the results (upper bounds)
    print("\nUpper bounds:")
    print(f"Approx power = {power:.3f}")
    print(f"Approx Type-M error = {type_m:.3f}")
    print(f"Approx Type-S error = {type_s:.3f}\n\n")

    #### Lower bound: Maximal disagreemnet on instances
    if (2 - acc1 - acc2) <= 1:
        p_both_incorrect = 0
        only_M1_correct = 1.0 - acc1
        only_M2_correct = 1.0 - acc2
        p_both_correct = 1.0 - only_M1_correct - only_M2_correct
    else:
        p_both_correct = 0
        only_M1_correct = acc1
        only_M2_correct = acc2
        p_both_incorrect = 1.0 - only_M1_correct - only_M2_correct
    agreement_rate = p_both_correct + p_both_incorrect
    prob_table = np.array([[p_both_incorrect, only_M1_correct], [only_M2_correct, p_both_correct]])
    
    print("="*50 + "\nProbability table for minimal agreement\n" + "="*50)
    print_probability_table(prob_table, agreement_rate)

    power, mean_effect, type_m, type_s = compute_power(prob_table, DATASET_SIZE, alpha=α, r=r)

    print("\nLower bounds:")
    print(f"Approx power = {power:.3f}")
    print(f"Approx Type-M error = {type_m:.3f}")
    print(f"Approx Type-S error = {type_s:.3f}\n\n")

    # Alternatively, we can use an ESTIMATED agreement rate (e.g., from previous model comparisons)
    agreement_rate = 0.975

    acc1 = BASELINE_ACC
    acc2 = BASELINE_ACC + DELTA_ACC

    disagreement_rate = 1 - agreement_rate
    if DELTA_ACC > 0:
        p_only_1_correct = (disagreement_rate - DELTA_ACC) / 2
        p_only_2_correct = (disagreement_rate - DELTA_ACC) / 2 + DELTA_ACC
    else:
        p_only_1_correct = (disagreement_rate + DELTA_ACC) / 2 - DELTA_ACC
        p_only_2_correct = (disagreement_rate + DELTA_ACC) / 2

    p_both_correct = acc1 - p_only_1_correct
    assert np.abs(p_both_correct - (acc2 - p_only_2_correct)) < 1e-4
    p_both_incorrect = 1. - p_both_correct - p_only_1_correct - p_only_2_correct

    for p in [p_both_correct, p_only_1_correct, p_only_2_correct, p_both_incorrect]:
        assert p >= 0

    agreement_rate = p_both_correct + p_both_incorrect
    prob_table = np.array([[p_both_incorrect, p_only_2_correct], [p_only_1_correct, p_both_correct]])
    print("="*50 + f"\nProbability table for agreement rate = {agreement_rate}\n" + "="*50)
    print_probability_table(prob_table, agreement_rate)

    power, mean_effect, type_m, type_s = compute_power(prob_table, DATASET_SIZE, alpha=α, r=r)

    print(f"Results for agreement rate = {agreement_rate}:")
    print(f"Approx power = {power:.3f}")
    print(f"Approx Type-M error = {type_m:.3f}")
    print(f"Approx Type-S error = {type_s:.3f}\n\n")


def demo_mde():
    mde = calculate_mde(
        baseline_acc=0.7, 
        agreement_rate=0.975, 
        n_samples=1000
    )
    print(f"💅✨ Minimum Δacc for power > 0.8: \033[92m{mde:.6f}\033[0m ({mde*100:.2f}%)")
    power, mean_effect, type_m, type_s = run_power_test(
        acc1 = 0.7, 
        acc2 = 0.7 + mde, 
        agreement_rate = 0.975, 
        n_samples = 1000
    )


if __name__ == '__main__': 
    # demo_mde()
    demo()