import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np

# Global dictionary to store colors for labels
LABEL_COLOR_MAP = {}
COLOR_IDX = {'col': 0}

def plot_heatmap(ax: plt.Axes, values, mix_names, mix_scores=None, sig_clusters=None, _type='p_values', alpha=0.01):
    """ Plot a pairwise heatmap of statistical significance """
    # Reorder values matrix according to sorted mixes
    mask = np.isnan(values)

    # Create a custom colormap that maps values between 0.5-0.95 to viridis
    # and values outside that range to grey
    if _type == 'p_values':
        def custom_colormap(value):
            if np.isnan(value):
                return (0, 0, 0, 0)
            elif value < alpha: # or value > (1-alpha):
                return (1, 1, 1, 0.05)
            else:
                return plt.cm.viridis(value)
    elif _type == 'power':
        def custom_colormap(value):
            if np.isnan(value) or value < 0:
                return (0, 0, 0, 0)
            elif value > 0.8:
                return (1, 1, 1, 0.05)
            else:
                return plt.cm.viridis(value)

    # Apply custom colors
    colors = [[custom_colormap(val) for val in row] for row in values]
    ax.imshow(colors)

    if mix_scores is not None:
        mix_names = [f'{name} (score={score:.3f})' for name, score in zip(mix_names, mix_scores)]
    
    if sig_clusters is not None:
        # Find indices where the significance cluster changes, then add a vertical line
        change_indices = np.where(sig_clusters[:-1] != sig_clusters[1:])[0] + 1
        for idx in change_indices:
            ax.axvline(x=idx-0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax.axhline(y=idx-0.5, xmin=0, xmax=1, color='red', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xticks(range(len(mix_names)))
    ax.set_yticks(range(len(mix_names)))
    ax.set_xticklabels(mix_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(mix_names, fontsize=8)

    # Add colorbar only for the viridis range
    norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, fraction=0.05, pad=0.04)
    label = r'$p$' + f'-values (highlighted if not significant,' + r'$\alpha$=' + f'{alpha})'
    if len(values) < 15:
        label = r'$p$' + f'-values'
    cbar.set_label(label)

    # Add value annotations with smaller font
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if not mask[i,j]:
                ax.text(j, i, f'{values[i,j]:.2f}', ha='center', va='center', fontsize=7)

    return ax


def assign_color(label):
    if label not in LABEL_COLOR_MAP:
        available_colors = list(mcolors.TABLEAU_COLORS.keys())
        assigned_color = available_colors[COLOR_IDX['col'] % len(available_colors)]
        LABEL_COLOR_MAP[label] = assigned_color
        COLOR_IDX['col'] += 1
    return LABEL_COLOR_MAP[label]


def lighten_color(color, amount=0.2):
    r, g, b = mcolors.to_rgb(color)
    new_r = min(r + (1 - r) * amount, 1)
    new_g = min(g + (1 - g) * amount, 1)
    new_b = min(b + (1 - b) * amount, 1)
    return new_r, new_g, new_b


def plot_training(ax: plt.Axes, x, y, xlabel: str, ylabel: str, label=None, title=None, color=None, fit=None, ci=None, sma_window=None):
    if color is None and label is not None:
        label_for_color = label
        # label_for_color = label.replace('_rc', '').replace('_mc', '').replace('_val', '').replace('_test', '') # peteish32 override
        # if '_5shot' in label_for_color: label_for_color = label_for_color.split('_5shot')[0]
        color = assign_color(label_for_color)
        # if 'rc' in label: 
        #     color = lighten_color(color, amount=0.5)

    if xlabel == 'step':
        if sma_window is not None:
            import numpy as np
            sma = np.convolve(y, np.ones(sma_window)/sma_window, mode='valid')
            x_sma = x[sma_window-1:]
            x_plt, y_plt = x_sma, sma
        else:
            x_plt, y_plt = x, y
        
        ax.plot(x_plt, y_plt, label=label, color=color, linewidth=0.5, marker='.', markersize=2)
        # ax.plot(df_slice[xlabel], df_slice[ylabel].rolling(window=5).mean(), label=label, color=color, linewidth=0.5, marker='.', markersize=2)
    else:
        ax.scatter(x, y, label=label, color=color, s=3)

    if ci is not None:
        ax.fill_between(x, y - ci, y + ci, alpha=0.1, color=color)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title: ax.set_title(title)
    if label is not None: ax.legend()

    ax.tick_params(axis='both', which='major', labelsize=8)

    # add fitted log function
    if fit is not None:
        # ax.set_xscale('log')

        from scipy.optimize import curve_fit
        import numpy as np
        import warnings
        warnings.filterwarnings("ignore", message="invalid value encountered in log")

        def log(x, a, b, c): return a * np.log(b * x) + c
        # def log(x, epsilon, k, gamma): return epsilon - k * np.exp(-gamma * x) # samir err

        # x, y = df_slice[xlabel].values, df_slice[ylabel].values
        x_max = np.max(x)
        x_scaled = x / x_max
        popt, _ = curve_fit(log, x_scaled, y, maxfev=10000)
        
        x_fit = np.linspace(min(x), max(x), 100)
        x_fit_scaled = x_fit / x_max
        y_fit = log(x_fit_scaled, *popt)

        ax.plot(x_fit, y_fit, color=color, alpha=0.5, linestyle='dotted')

    return ax


def plot_simulation_results(results: dict, ax: plt.Axes, f1_only: bool=False, f1_label: str="F1 Score"):
    ax.grid(True)
    
    results = sorted(results, key=lambda x: x[1])

    # Extract the gold entry
    gold_entry = next((x for x in results if x[0] == 'gold'), None)
    results = [x for x in results if x[0] != 'gold']

    labels        = [d[0] for d in results]
    x_values      = [d[1] for d in results]
    prec_values   = [d[2][0] for d in results]
    recall_values = [d[2][1] for d in results]
    f1_values     = [d[2][2] for d in results]

    x_values = np.array(x_values, dtype=np.float64)

    # Add gold compute
    if gold_entry is not None:
        _, x_val, (p, r, f1) = gold_entry

        # Only add a label if it doesn't exist yet in the table
        label = None
        handles, labels = ax.get_legend_handles_labels()
        if '1B Compute' not in labels:
            label = '1B Compute'

        ax.axvline(x=x_val, color='black', linestyle='--', label=label, alpha=0.8)
        ax.scatter(x_val, f1, color='gold', marker='*', s=100, edgecolor='black')

    if not f1_only:
        ax.plot(x_values, prec_values, label="Precision", marker='o', alpha=0.8)
        ax.plot(x_values, recall_values, label="Recall", marker='s', alpha=0.8)
    ax.plot(x_values, f1_values, label=f1_label, marker='^', alpha=0.8)

    ax.set_xlabel("Cumulative Compute")
    ax.set_ylabel("Metric")
    ax.set_ylim(-0.1, 1.1)
    ax.legend()

    # # Add labels near each point
    # texts = []
    # for i, (x, y) in enumerate(zip(x_values, prec_values)):
    #     texts += [ax.text(x, y*(1+(0.02*i)), f"{labels[i]}", fontsize=8, ha='right', va='bottom')]

    # adjust_text(
    #     texts,
    #     arrowprops=dict(arrowstyle="->", color='gray', lw=0.5),
    #     force_text=(1, 1),  # Stronger force to push text labels apart
    #     expand_text=(1, 1),  # Increase spacing around text
    #     lim=300  # Increase the iterations to ensure better optimization
    # )

    # Add log scale
    ax.set_xscale('log')

    return ax