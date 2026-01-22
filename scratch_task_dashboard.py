import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from pathlib import Path
import numpy as np
from ace_tools import display_dataframe_to_user

# Construct data set fresh
models_data = [
    {"model": "olmo2-1b", "accuracy": 0.584, "params": 1e9, "tokens": 4e12},
    {"model": "olmo2-7b", "accuracy": 0.687, "params": 7e9, "tokens": 4e12},
    {"model": "olmo2-13b", "accuracy": 0.73,  "params": 13e9, "tokens": 5e12},
    {"model": "meta-llama/Llama-2-13b-hf", "accuracy": 0.749, "params": 13e9, "tokens": 2e12},
    {"model": "deepseek-ai/deepseek-llm-7b-base", "accuracy": 0.723, "params": 7e9, "tokens": 2e12},
    {"model": "Qwen/Qwen2.5-7B", "accuracy": 0.679, "params": 7e9, "tokens": 18e12},
    {"model": "allenai/DataDecide-dclm-baseline-1B", "accuracy": 0.614, "params": 1e9, "tokens": 1e11},
    {"model": "allenai/DataDecide-dclm-baseline-50p-dolma1.7-50p-1B", "accuracy": 0.604, "params": 1e9, "tokens": 1e11},
    {"model": "allenai/DataDecide-dolma1_7-1B", "accuracy": 0.561, "params": 1e9, "tokens": 1e11},
    {"model": "olmo2-1b-step20000", "accuracy": 0.58,  "params": 1e9, "tokens": 42e9},
    {"model": "olmo2-1b-step21000", "accuracy": 0.596, "params": 1e9, "tokens": 45e9},
    {"model": "olmo2-1b-step22000", "accuracy": 0.599, "params": 1e9, "tokens": 47e9},
    {"model": "olmo2-1b-step23000", "accuracy": 0.598, "params": 1e9, "tokens": 49e9},
    {"model": "meta-llama/Meta-Llama-3-8B", "accuracy": 0.751, "params": 8e9, "tokens": 15e12},
    {"model": "meta-llama/Meta-Llama-3.1-8B", "accuracy": 0.749, "params": 8e9, "tokens": 15e12},
]

df = pd.DataFrame(models_data)

# Metrics
df["compute"] = 6 * df["params"] * df["tokens"]
df["t2p_ratio"] = df["tokens"] / df["params"]

# Label renaming
def rename(model):
    mapping = {
        "allenai/DataDecide-dclm-baseline-1B": "DataDecide-DCLM",
        "allenai/DataDecide-dclm-baseline-50p-dolma1.7-50p-1B": "DataDecide-DCLM-Dolma-even-mix",
        "allenai/DataDecide-dolma1_7-1B": "DataDecide-Dolma",
        "meta-llama/Meta-Llama-3-8B": "Llama-3-8B",
        "meta-llama/Meta-Llama-3.1-8B": "Llama-3.1-8B",
    }
    if model in mapping:
        return mapping[model]
    return model.split("/")[-1]

df["label"] = df["model"].apply(rename)

# Marker mapping
marker_map = {
    "DataDecide-DCLM": '^',
    "DataDecide-DCLM-Dolma-even-mix": '^',
    "DataDecide-Dolma": '^',
    "olmo2-1b": 's',
    "olmo2-7b": 's',
    "olmo2-13b": 's',
    "Llama-2-13b-hf": 'o',
    "deepseek-llm-7b-base": 'X',
    "Qwen2.5-7B": 'D',
    "Llama-3-8B": 'v',
    "Llama-3.1-8B": 'P'
}

# Filter checkpoints
ckpt_mask = df["model"].str.contains("olmo2-1b-step")
plot_df = df[~ckpt_mask].copy()

# Compute accuracy std for olmo2-1b
acc_std = df.loc[ckpt_mask, "accuracy"].append(pd.Series([plot_df.loc[plot_df["model"] == "olmo2-1b", "accuracy"].iloc[0]])).std(ddof=0)

# Color normalization
norm = Normalize(vmin=plot_df["t2p_ratio"].min(), vmax=plot_df["t2p_ratio"].max())
cmap = cm.get_cmap("viridis")

plt.figure(figsize=(8,6))

# Scatter each point
for _, row in plot_df.iterrows():
    label = row["label"]
    marker = marker_map.get(label, '*')
    color = cmap(norm(row["t2p_ratio"]))
    plt.scatter(row["compute"], row["accuracy"], marker=marker, color=color, s=70, edgecolor='k')
    plt.annotate(label, (row["compute"], row["accuracy"]), fontsize=8, xytext=(4,2), textcoords='offset points')

# Error bar
olmo1b = plot_df[plot_df["model"] == "olmo2-1b"].iloc[0]
plt.errorbar(olmo1b["compute"], olmo1b["accuracy"], yerr=acc_std, fmt=marker_map["olmo2-1b"],
             color=cmap(norm(olmo1b["t2p_ratio"])), markersize=7, capsize=5, markeredgecolor='k')

# Connect olmo2 line
olmo_series = plot_df[plot_df["model"].isin(["olmo2-1b","olmo2-7b","olmo2-13b"])].sort_values("compute")
plt.plot(olmo_series["compute"], olmo_series["accuracy"], linestyle='--', color='black', alpha=0.6)

# Colorbar
cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap))
cbar.set_label("Token / Parameter Ratio")

plt.xscale("log")
plt.xlabel("Compute (6ND, FLOPs, log scale)")
plt.ylabel("LAMBADA Accuracy")
plt.tight_layout()

plot_path = Path("/mnt/data/compute_vs_lambada_final.png")
plt.savefig(plot_path, dpi=300)
plt.close()

display_dataframe_to_user("Final plot data", plot_df[["label","accuracy","compute","t2p_ratio"]])

plot_path
