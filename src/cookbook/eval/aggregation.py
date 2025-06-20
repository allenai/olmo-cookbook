from itertools import product
import logging
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

def results_to_ndarray(results, all_models):
    """ Convert to numpy arrays ensuring consistent model ordering """
    tasks = []
    scores = []
    for task_name, task_scores in results.items():
        task_scores_list = []
        for model in all_models:
            if model not in results[task_name]:
                raise ValueError(f"Model {model} missing from task {task_name}")
            task_scores_list.append(results[task_name][model])
        scores.append(task_scores_list)
        tasks.append(task_name)
    return tasks, np.array(scores)


def remove_incomplete_tasks(tasks, results_np, partial=False):
    # Remove any rows in results_small_np that contain None values
    valid_rows = ~np.any(np.equal(results_np, None), axis=1)
    invalid_task_indices = np.where(~valid_rows)[0]
    if len(invalid_task_indices) > 0:
        if partial:
            logger.warning(f"Removing tasks with partial/empty results (None): {[tasks[i] for i in invalid_task_indices]}")
            tasks = [t for i, t in enumerate(tasks) if i not in invalid_task_indices]
            results_np = results_np[valid_rows]
            results_np = np.array(results_np, dtype=np.float64) # ensure float dtype
            if results_np.ndim == 1:
                # ensure we are still using a 2d nparray
                results_np = results_np.reshape(1, -1)
        else:
            raise RuntimeError('Some tasks have missing data')
        
    return tasks, results_np


def compute_pca(data):
    # Compute eigenvalues
    data_centered = data - np.mean(data, axis=0) # Center the data
    cov_matrix = np.cov(data_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Make sure PC1 correlates positively with the original scores
    # by checking if the sum of PC1 weights is positive
    if np.sum(eigenvectors[:,0]) < 0:
        eigenvectors[:,0] = -eigenvectors[:,0]

    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)

    # Calculate weights that would recover the PCs
    weights = eigenvectors / np.sqrt(np.sum(eigenvectors**2, axis=0))

    return data_centered, weights, explained_variance_ratio


def compute_kendall_tau(scores_small, scores_target):
    """ Fast Kendall Tau for two 1D arrays """
    assert len(scores_small.shape) == 1, f"Expected 1D array for scores_small, got shape {scores_small.shape}"
    assert len(scores_target.shape) == 1, f"Expected 1D array for scores_target, got shape {scores_target.shape}"
    scores_small = np.array(scores_small)
    scores_target = np.array(scores_target)
    small_diffs = scores_small[:, np.newaxis] > scores_small[np.newaxis, :]
    target_diffs = scores_target[:, np.newaxis] > scores_target[np.newaxis, :]
    mask = np.triu(np.ones_like(small_diffs), k=1).astype(bool)
    agreements = (small_diffs == target_diffs)[mask]
    decision_acc = np.mean(agreements)
    kendall_tau = 2 * decision_acc - 1
    return decision_acc, kendall_tau


def neg_kendall_tau(w, results_small, results_large):
    w = np.array(w)
    w /= np.sum(w)
    weighted = np.sum(w[:, None] * results_small, axis=0)
    _, tau = compute_kendall_tau(weighted, results_large)
    return -tau


def generate_simplex_grid(n, steps):
    raw = [p for p in product(range(steps + 1), repeat=n) if sum(p) == steps]
    return np.array(raw) / steps


def hit_and_run_dirichlet(n_samples, alpha, bounds=None, burn_in=100, thinning=10):
    """ Approximate bounded dirichlet sampler. Exact sampling w/ bounds is intractable """
    n = len(alpha)
    if bounds is None:
        a = np.zeros(n)
        b = np.ones(n)
    else:
        bounds = np.array(bounds)
        a, b = bounds[:, 0], bounds[:, 1]
    
    # Start at a valid point inside the simplex and bounds
    while True:
        x = np.random.dirichlet(alpha)
        if np.all((x >= a) & (x <= b)):
            break

    samples = []

    def get_line_limits(x, d, a, b):
        t_min, t_max = -np.inf, np.inf
        for i in range(len(d)):
            if d[i] != 0:
                t0 = (a[i] - x[i]) / d[i]
                t1 = (b[i] - x[i]) / d[i]
                t_lo, t_hi = min(t0, t1), max(t0, t1)
                t_min = max(t_min, t_lo)
                t_max = min(t_max, t_hi)
            elif not (a[i] <= x[i] <= b[i]):
                return None
        return (t_min, t_max) if t_min <= t_max else None

    total_steps = burn_in + n_samples * thinning
    for _ in range(total_steps):
        d = np.random.randn(n)
        d -= np.mean(d)
        d /= np.linalg.norm(d)
        limits = get_line_limits(x, d, a, b)
        if limits is None:
            continue
        t = np.random.uniform(*limits)
        x_new = x + t * d
        x_new /= np.sum(x_new)
        if np.all((x_new >= a) & (x_new <= b)):
            x = x_new
            if _ >= burn_in and (_ - burn_in) % thinning == 0:
                samples.append(x.copy())

    samples = np.array(samples)
    
    assert np.allclose(samples.sum(axis=1), 1, atol=1e-8)
    assert np.all(samples >= a - 1e-10)
    assert np.all(samples <= b + 1e-10)

    return samples


def grid_search_optimizer(objective_fn, w_init, results_small, results_large, n_samples=50_000, steps=100, _type="approximate", bounds=None):
    """ Simple grid search over weights between tasks """
    n = len(w_init)
    if _type == "approximate":
        samples = hit_and_run_dirichlet(n_samples, np.ones(n), bounds)
    elif _type == "grid":
        if bounds is not None:
            raise NotImplementedError()
        samples = generate_simplex_grid(n, steps)
    else:
        raise ValueError(_type)

    best_val = float('inf')
    best_w = None

    for w in tqdm(samples, desc="Running samples"):
        val = objective_fn(w, results_small, results_large)
        if val < best_val:
            best_val = val
            best_w = w

    return np.array(best_w), best_val