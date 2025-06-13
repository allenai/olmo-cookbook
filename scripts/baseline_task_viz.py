#!/usr/bin/env python3
"""
plot_tasks.py  –  Compute-vs-Metric visualiser.

• Macro-averages families:
      mmlu_*, basic_skills_*, minerva_math_*, mt_mbpp_*
      – For mt_mbpp:<subtask>:<metric>  ⭢  mt_mbpp:<metric>
      – For mt_mbpp:<subtask>          ⭢  mt_mbpp

• NaN / null scores dropped per-task (model kept on others).

• Non-overlapping labels with a *simple, silent* fallback (adjustText OFF).

• Outputs
    baseline_task_viz/
    ├─ summary_grid_ALL.pdf
    ├─ summary_grid_<metric>.pdf     (main, bpb, rc, …)
    └─ subplots/<task>.pdf           (6×4 inch each)
"""

from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# ------------------------------------------------------------------ #
#  Fixed parameter / token counts                                    #
# ------------------------------------------------------------------ #
BASE_INFO = {
    "olmo2-1b":  dict(params=1e9,  tokens=4e12),
    "olmo2-7b":  dict(params=7e9,  tokens=4e12),
    "olmo2-13b": dict(params=13e9, tokens=5e12),

    "Llama-2-13b-hf":                        dict(params=13e9, tokens=2e12),
    "Meta-Llama-3-8B":                       dict(params= 8e9, tokens=15e12),
    "Meta-Llama-3.1-8B":                     dict(params= 8e9, tokens=15e12),
    "deepseek-llm-7b-base":                  dict(params= 7e9, tokens=2e12),
    "Qwen2.5-7B":                            dict(params= 7e9, tokens=18e12),
    "DataDecide-dclm-baseline-1B":           dict(params= 1e9, tokens=1e11),
    "DataDecide-dclm-baseline-50p-dolma1.7-50p-1B": dict(params=1e9, tokens=1e11),
    "DataDecide-dolma1_7-1B":                dict(params= 1e9, tokens=1e11),
}

LABEL_MAP = {
    "Meta-Llama-3-8B":   "Llama-3-8B",
    "Meta-Llama-3.1-8B": "Llama-3.1-8B",
    "DataDecide-dclm-baseline-1B":           "DataDecide-DCLM",
    "DataDecide-dclm-baseline-50p-dolma1.7-50p-1B": "DCLM-Dolma-mix",
    "DataDecide-dolma1_7-1B":                "DataDecide-Dolma",
}

MARKERS = {
    "olmo2-1b": "s", "olmo2-7b": "s", "olmo2-13b": "s",
    "DataDecide-DCLM": "^", "DCLM-Dolma-mix": "^", "DataDecide-Dolma": "^",
    "Llama-2-13b-hf": "o", "deepseek-llm-7b-base": "X",
    "Qwen2.5-7B": "D", "Llama-3-8B": "v", "Llama-3.1-8B": "P",
}

FAMILY_PREFIXES = ("mmlu_", "basic_skills_", "minerva_math_", "mt_mbpp_")

# ------------------------------------------------------------------ #
def load_json(fp: Path) -> dict:
    return json.loads(fp.read_text())

# ------------------------------------------------------------------ #
def macro_average(tasks: dict) -> dict:
    """Collapse each family into one macro-average per metric suffix."""
    buckets: dict[str, list[dict]] = {}

    for task, res in tasks.items():
        key = task
        for pref in FAMILY_PREFIXES:
            if task.startswith(pref):
                remainder = task[len(pref):]            # after “family_”
                parts = remainder.split(":")
                # Decide metric suffix
                if pref == "mt_mbpp_":
                    # mt_mbpp:<subtask>[:<metric>] – drop subtask
                    metric = ":".join(parts[1:]) if len(parts) > 1 else ""
                else:
                    metric = ":".join(parts[1:]) if len(parts) > 1 else ""
                base = pref[:-1]                        # strip trailing "_"
                key  = base if metric == "" else f"{base}:{metric}"
                break
        buckets.setdefault(key, []).append(res)

    merged: dict[str, dict] = {}
    for task, lst in buckets.items():
        if len(lst) == 1 and task.split(":")[0] not in [p[:-1] for p in FAMILY_PREFIXES]:
            merged[task] = lst[0]
            continue

        # Macro-avg values per model
        model_vals: dict[str, list] = {}
        for res in lst:
            for m, v in res.items():
                model_vals.setdefault(m, []).append(v)

        merged[task] = {
            m: (None if all(v is None or (isinstance(v, float) and math.isnan(v)) for v in vs)
                else float(np.nanmean([x for x in vs if x not in (None, np.nan)])))
            for m, vs in model_vals.items()
        }
    return merged

# ------------------------------------------------------------------ #
def build_df(scores: dict) -> pd.DataFrame:
    ck = {s: [] for s in ("olmo2-1b", "olmo2-7b", "olmo2-13b")}
    singles = {}
    for m, v in scores.items():
        if v is None or (isinstance(v, float) and math.isnan(v)): continue
        if m.startswith("allenai--OLMo-2-0425-1B"):   ck["olmo2-1b"].append(v)
        elif m.startswith("allenai--OLMo-2-1124-7B"): ck["olmo2-7b"].append(v)
        elif m.startswith("allenai--OLMo-2-1124-13B"):ck["olmo2-13b"].append(v)
        else: singles[m] = v

    rows = []
    for size, arr in ck.items():
        if not arr: continue
        p, t = BASE_INFO[size].values()
        rows.append(dict(label=size,
                         accuracy=float(np.mean(arr)),
                         std=float(np.std(arr, ddof=0)),
                         params=p, tokens=t))
    for m, v in singles.items():
        p, t = BASE_INFO[m].values()
        rows.append(dict(label=LABEL_MAP.get(m, m.split("/")[-1]),
                         accuracy=v, std=0.0,
                         params=p, tokens=t))

    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["compute"] = 6 * df["params"] * df["tokens"]
    df["ratio"]   = df["tokens"] / df["params"]
    return df

# ------------------------------------------------------------------ #
def place_labels(ax, df):
    """Simple, silent de-overlap (no adjustText)."""
    texts = [ax.text(r.compute, r.accuracy, r.label,
                     fontsize=6, ha="center", va="bottom")
             for r in df.itertuples()]
    renderer = ax.figure.canvas.get_renderer()
    placed   = []
    for txt in texts:
        x0, y0 = txt.get_position()
        offset = 0.0
        while True:
            txt.set_position((x0, y0 + offset))
            bbox = txt.get_window_extent(renderer)
            if not any(bbox.overlaps(b) for b in placed):
                placed.append(bbox)
                ax.annotate("",
                            xy=(x0, y0), xytext=(x0, y0 + offset),
                            arrowprops=dict(arrowstyle="-", lw=0.4, color="gray"))
                break
            offset += 0.003   # move label up a bit more each try

# ------------------------------------------------------------------ #
def plot_task(ax, df, task):
    norm = Normalize(vmin=df["ratio"].min(), vmax=df["ratio"].max())
    cmap = plt.get_cmap("viridis")

    # shaded OLMo band
    olmo = df[df.label.str.startswith("olmo2")].sort_values("compute")
    if not olmo.empty:
        ax.fill_between(olmo["compute"],
                        olmo["accuracy"] - olmo["std"],
                        olmo["accuracy"] + olmo["std"],
                        color="lightgrey", alpha=0.3, zorder=0)
        ax.plot(olmo["compute"], olmo["accuracy"],
                "--", lw=1, color="black", zorder=1)

    # scatter
    for r in df.itertuples():
        ax.scatter(r.compute, r.accuracy,
                   marker=MARKERS.get(r.label, "*"),
                   color=cmap(norm(r.ratio)),
                   s=70, edgecolor="k", zorder=2)

    place_labels(ax, df)

    ax.set_xscale("log")
    ax.set_xlabel("Compute (6 N D FLOPs, log)")
    ax.set_ylabel(task)
    ax.grid(True, ls=":", lw=0.3)
    return norm, cmap

# ------------------------------------------------------------------ #
def save_plots(task_dfs: dict[str, pd.DataFrame], out: Path):
    sub = out / "subplots"
    sub.mkdir(parents=True, exist_ok=True)

    # individual 6×4 PDFs
    for task, df in task_dfs.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        n, c = plot_task(ax, df, task)
        cb = fig.colorbar(cm.ScalarMappable(norm=n, cmap=c), ax=ax)
        cb.set_label("Token / Parameter Ratio")
        fig.tight_layout()
        fig.savefig(sub / f"{task}.pdf")
        plt.close(fig)

    # summary grids
    buckets: dict[str, list[str]] = {}
    for t in task_dfs:
        metric = t.split(":", 1)[1] if ":" in t else "main"
        buckets.setdefault(metric, []).append(t)

    def grid(pdf: Path, tasks: list[str]):
        cols = math.ceil(math.sqrt(len(tasks)))
        rows = math.ceil(len(tasks) / cols)
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(4 * cols, 3 * rows),
                                 squeeze=False)
        for ax in axes.flat[len(tasks):]:
            ax.axis("off")
        for ax, t in zip(axes.flat, tasks):
            n, c = plot_task(ax, task_dfs[t], t)
            cb = fig.colorbar(cm.ScalarMappable(norm=n, cmap=c),
                              ax=ax, pad=0.01)
            cb.set_label("Token / Param Ratio")
        fig.tight_layout()
        fig.savefig(pdf)
        plt.close(fig)

    for m, ts in buckets.items():
        grid(out / f"summary_grid_{m}.pdf", ts)

    grid(out / "summary_grid_ALL.pdf", list(task_dfs))

# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_file", type=Path, help="Input multi-task JSON")
    ap.add_argument("--outdir", type=Path, default=Path("baseline_task_viz"),
                    help="Output directory (default baseline_task_viz/)")
    args = ap.parse_args()

    raw       = load_json(args.json_file)
    collapsed = macro_average(raw)
    task_dfs  = {t: build_df(sc) for t, sc in collapsed.items()}
    task_dfs  = {t: df for t, df in task_dfs.items() if not df.empty}

    if not task_dfs:
        print("No plottable data.")
        return

    args.outdir.mkdir(parents=True, exist_ok=True)
    save_plots(task_dfs, args.outdir)
    print("Plots written to", args.outdir.resolve())

# ------------------------------------------------------------------ #
if __name__ == "__main__":
    main()
