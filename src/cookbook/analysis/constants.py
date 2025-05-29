import os
from pathlib import Path

from platformdirs import user_cache_dir

PLOT_DIR = "/tmp/cookbook/analysis/plots"


def get_cache_path(dashboard) -> Path:
    cache_dir = user_cache_dir("cookbook")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    path = Path(cache_dir) / dashboard
    return path


def get_title_from_task(task):
    if isinstance(task, list):
        title_mapping = {
            "mmlu_pro_": "mmlu_pro",
            "mmlu_abstract_algebra:mc": "mmlu_mc",
            "mmlu": "mmlu",
            "minerva": "minerva",
            "agi_eval": "agi_eval",
            "bbh": "bbh",
            "codex": "codex",
            "arc_challenge:mc": "olmes_core9_mc",
            "arc_challenge": "olmes_core9",
            "drop": "olmes_gen",
        }
        for key, title in title_mapping.items():
            if key in task[0]:
                return title

        return "aggregate"

    return task


def get_task_sets(all_tasks) -> list[list[str] | str]:
    mmlu: list[str] = [t for t in all_tasks if "mmlu" in t and ":" not in t and "_pro_" not in t]
    minerva: list[str] = [t for t in all_tasks if "minerva" in t and ":" not in t]
    mmlu_pro: list[str] = [t for t in all_tasks if "_pro_" in t and ":rc" in t]
    mmlu_mc: list[str] = [t for t in all_tasks if "mmlu" in t and ":mc" in t and "_pro_" not in t]
    olmes = [
        "arc_challenge",
        "arc_easy",
        "boolq",
        "csqa",
        "hellaswag",
        "openbookqa",
        "piqa",
        "socialiqa",
        "winogrande",
    ]
    olmes_mc = [f"{task}:mc" for task in olmes]
    olmes_gen = ["drop", "gsm8k", "jeopardy", "naturalqs", "squad", "triviaqa"]
    agi_eval: list[str] = [t for t in all_tasks if "agi_eval" in t and ":" not in t]
    bbh: list[str] = [t for t in all_tasks if "bbh" in t and ":" not in t]
    paloma: list[str] = [t for t in all_tasks if "paloma" in t]

    selected_tasks = (
        olmes + olmes_mc + olmes_gen + [olmes, mmlu, olmes_mc, mmlu_mc, olmes_gen, minerva, agi_eval, bbh, paloma]
    )
    selected_tasks += ["gsm_symbolic_main", mmlu_pro, "autobencher", "autobencher:mc"]
    selected_tasks += [
        "minerva_math_500",
        "mbpp",
        "mbppplus",
        "codex_humaneval",
        "codex_humanevalplus",
        "copycolors:mc",
    ]
    selected_tasks.append(
        [
            "codex_humaneval",
            "minerva_math_500",
            "mbpp",
            "mbppplus",
            "codex_humanevalplus",
            "copycolors:mc",
        ]
    )

    selected_tasks = [
        task for task in selected_tasks if len(task) > 0
    ]  # exclude tasks where we don't see them in the df

    return selected_tasks
