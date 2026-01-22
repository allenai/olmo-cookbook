import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# Global dictionary to store colors for labels
LABEL_COLOR_MAP = {}
COLOR_IDX = {"col": 0}


def plot_heatmap(
    ax: plt.Axes,
    values,
    mix_names,
    mix_scores=None,
    sig_clusters=None,
    _type="p_values",
    alpha=0.01,
    plot_clean=False,
):
    """Plot a pairwise heatmap of statistical significance"""
    # Reorder values matrix according to sorted mixes
    mask = np.isnan(values)

    # Create a custom colormap that maps values between 0.5-0.95 to viridis
    # and values outside that range to grey
    if _type == "p_values":

        def custom_colormap(value):
            if np.isnan(value):
                return (0, 0, 0, 0)
            elif value < alpha:  # or value > (1-alpha):
                return (1, 1, 1, 0.05)
            else:
                return plt.cm.get_cmap("viridis")(value)

    elif _type == "power":

        def custom_colormap(value):
            if np.isnan(value) or value < 0:
                return (0, 0, 0, 0)
            elif value > 0.8:
                return (1, 1, 1, 0.05)
            else:
                return plt.cm.get_cmap("viridis")(value)

    else:
        raise ValueError(f"Unknown type {_type}")

    # Apply custom colors
    colors = [[custom_colormap(val) for val in row] for row in values]
    ax.imshow(colors)

    if mix_scores is not None:
        mix_names = [f"{name} (score={score:.3f})" for name, score in zip(mix_names, mix_scores)]

    if sig_clusters is not None:
        # Find indices where the significance cluster changes, then add a vertical line
        change_indices = np.where(sig_clusters[:-1] != sig_clusters[1:])[0] + 1
        for idx in change_indices:
            ax.axvline(x=idx - 0.5, color="red", linestyle="--", linewidth=1, alpha=0.5)
            ax.axhline(y=idx - 0.5, xmin=0, xmax=1, color="red", linestyle="--", linewidth=1, alpha=0.5)

    ax.set_xticks(range(len(mix_names)))
    ax.set_yticks(range(len(mix_names)))

    if not plot_clean:
        ax.set_xticklabels(mix_names, rotation=45, ha="right", fontsize=12)
        ax.set_yticklabels(mix_names, fontsize=12)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # Add colorbar only for the viridis range
    norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap("viridis"), norm=norm)
    cbar = plt.colorbar(sm, ax=ax, fraction=0.05, pad=0.04)
    label = r"$p$" + "-values " + r"$\alpha$=" + f"{alpha}"

    if len(values) < 15 or plot_clean:
        label = r"$p$" + "-values"

    cbar.set_label(label)

    # Add value annotations with smaller font
    if not plot_clean:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if not mask[i, j]:
                    ax.text(j, i, f"{values[i, j]:.2f}".lstrip("0"), ha="center", va="center", fontsize=10)

    return ax


def assign_color(label):
    if label not in LABEL_COLOR_MAP:
        available_colors = list(mcolors.TABLEAU_COLORS.keys())
        assigned_color = available_colors[COLOR_IDX["col"] % len(available_colors)]
        LABEL_COLOR_MAP[label] = assigned_color
        COLOR_IDX["col"] += 1
    return LABEL_COLOR_MAP[label]


def lighten_color(color, amount=0.2):
    r, g, b = mcolors.to_rgb(color)
    new_r = min(r + (1 - r) * amount, 1)
    new_g = min(g + (1 - g) * amount, 1)
    new_b = min(b + (1 - b) * amount, 1)
    return new_r, new_g, new_b
