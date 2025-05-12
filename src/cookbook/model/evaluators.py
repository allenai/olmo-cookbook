from enum import Enum
from typing import Dict, List

from olmo_eval import list_tasks

OLMO2_DEV_1B_TASKS = [
    # OLMES Core 9(-ish) RC
    # "arc_challenge_test_rc_5shot",
    # "arc_easy_test_rc_5shot",
    # "hellaswag_rc_5shot",  # 1K subset of HellaSwag
    # "winogrande_val_rc_5shot",  # Helpful after 750M-5xC scale
    # "csqa_val_rc_5shot",
    # "piqa_val_rc_5shot",
    # "socialiqa_val_rc_5shot",
    # MMLU RC
    "mmlu_stem_val_rc_5shot",
    "mmlu_humanities_val_rc_5shot",
    "mmlu_social_sciences_val_rc_5shot",
    "mmlu_other_val_rc_5shot",
    "mmlu_stem_test_rc_5shot",
    "mmlu_humanities_test_rc_5shot",
    "mmlu_social_sciences_test_rc_5shot",
    "mmlu_other_test_rc_5shot",
    # Gen tasks BPB
    # "gsm8k_gold_bpb_5shot",
    # "minerva_math_algebra_gold_bpb_0shot",
    # "minerva_math_counting_and_probability_gold_bpb_0shot",
    # "minerva_math_geometry_gold_bpb_0shot",
    # "minerva_math_intermediate_algebra_gold_bpb_0shot",
    # "minerva_math_number_theory_gold_bpb_0shot",
    # "minerva_math_prealgebra_gold_bpb_0shot",
    # "minerva_math_precalculus_gold_bpb_0shot",
    # "codex_humaneval_gold_bpb_0shot",
    # "codex_mbpp_gold_bpb_0shot",
    # Sanity check for MCQA ability
    # "copycolors_10way",
]

TASK_GROUPS: Dict[str, List[str]] = {
    "all": list(list_tasks()),
    "olmo2_dev_1b": OLMO2_DEV_1B_TASKS,
}


ALL_TASKS_MAP = {task.upper(): task for task in list_tasks()}

DownstreamEvaluator = Enum(
    "DownstreamEvaluator",
    {
        item[0].upper(): item[1] if isinstance(item[1], list) else [item[1]]
        for item in {**TASK_GROUPS, **ALL_TASKS_MAP}.items()
    },
)


def get_tasks_for_groups(groups: List[str]) -> List[str]:
    """Return all tasks in a group"""
    tasks = []
    for group in groups:
        if group in TASK_GROUPS:
            tasks.extend(TASK_GROUPS[group])
        elif group.upper() in ALL_TASKS_MAP:
            tasks.append(ALL_TASKS_MAP[group.upper()])
        else:
            raise ValueError(f"Group or task '{group}' not found")

    return list(set(tasks))
