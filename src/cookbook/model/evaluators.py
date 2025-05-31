from enum import Enum
from typing import Dict, List

from olmo_eval import list_tasks

OLMO2_DEV_1B_TASKS = [
    # OLMES Core 9(-ish) RC
    "arc_challenge_test_rc_5shot",
    "arc_easy_test_rc_5shot",
    "hellaswag_rc_5shot",  # 1K subset of HellaSwag
    "winogrande_val_rc_5shot",  # Helpful after 750M-5xC scale
    "csqa_val_rc_5shot",
    "piqa_val_rc_5shot",
    "socialiqa_val_rc_5shot",
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
    "gsm8k_gold_bpb_5shot",
    "minerva_math_algebra_gold_bpb_0shot",
    "minerva_math_counting_and_probability_gold_bpb_0shot",
    "minerva_math_geometry_gold_bpb_0shot",
    "minerva_math_intermediate_algebra_gold_bpb_0shot",
    "minerva_math_number_theory_gold_bpb_0shot",
    "minerva_math_prealgebra_gold_bpb_0shot",
    "minerva_math_precalculus_gold_bpb_0shot",
    "codex_humaneval_gold_bpb_0shot",
    "codex_mbpp_gold_bpb_0shot",
    # Sanity check for MCQA ability
    "copycolors_10way",
    # Basic Skills rc 5shot
    "basic_skills_arithmetic_rc_5shot",
    "basic_skills_coding_rc_5shot",
    "basic_skills_common_knowledge_rc_5shot",
    "basic_skills_logical_reasoning_rc_5shot",
    "basic_skills_pattern_rc_5shot",
    "basic_skills_string_operations_rc_5shot",
]

# these have to match aliases in oe-eval, NOT, display names in cookbook
OLMO3_DEV_1B_TASKS = [
    # Arc tasks
    "arc_challenge:rc::olmes:full",
    "arc_easy:rc::olmes:full",
    # Basic Skills rc 5shot
    "basic_skills_arithmetic:rc::olmes",
    "basic_skills_coding:rc",
    "basic_skills_common_knowledge:rc::olmes",
    "basic_skills_logical_reasoning:rc::olmes",
    "basic_skills_pattern:rc::olmes",
    "basic_skills_string_operations:rc::olmes",
    #
    "codex_humaneval:3shot:bpb::none",
    "coqa:rc::gen2mc",
    "csqa:rc::olmes:full",
    "drop:rc::gen2mc",
    "hellaswag:rc::olmes:full",
    "jeopardy:rc::gen2mc",
    "lambada",
    "mbpp:3shot:bpb::none",
    # minerva math tasks
    "minerva_math_500::olmes",
    "minerva_math_algebra::olmes",
    "minerva_math_counting_and_probability::olmes",
    "minerva_math_geometry::olmes",
    "minerva_math_intermediate_algebra::olmes",
    "minerva_math_number_theory::olmes",
    "minerva_math_prealgebra::olmes",
    "minerva_math_precalculus::olmes",
    # mmlu tasks
    "mmlu_abstract_algebra:rc::olmes",
    "mmlu_anatomy:rc::olmes",
    "mmlu_astronomy:rc::olmes",
    "mmlu_business_ethics:rc::olmes",
    "mmlu_clinical_knowledge:rc::olmes",
    "mmlu_college_biology:rc::olmes",
    "mmlu_college_chemistry:rc::olmes",
    "mmlu_college_computer_science:rc::olmes",
    "mmlu_college_mathematics:rc::olmes",
    "mmlu_college_medicine:rc::olmes",
    "mmlu_college_physics:rc::olmes",
    "mmlu_computer_security:rc::olmes",
    "mmlu_conceptual_physics:rc::olmes",
    "mmlu_econometrics:rc::olmes",
    "mmlu_electrical_engineering:rc::olmes",
    "mmlu_elementary_mathematics:rc::olmes",
    "mmlu_formal_logic:rc::olmes",
    "mmlu_global_facts:rc::olmes",
    "mmlu_high_school_biology:rc::olmes",
    "mmlu_high_school_chemistry:rc::olmes",
    "mmlu_high_school_computer_science:rc::olmes",
    "mmlu_high_school_european_history:rc::olmes",
    "mmlu_high_school_geography:rc::olmes",
    "mmlu_high_school_government_and_politics:rc::olmes",
    "mmlu_high_school_macroeconomics:rc::olmes",
    "mmlu_high_school_mathematics:rc::olmes",
    "mmlu_high_school_microeconomics:rc::olmes",
    "mmlu_high_school_physics:rc::olmes",
    "mmlu_high_school_psychology:rc::olmes",
    "mmlu_high_school_statistics:rc::olmes",
    "mmlu_high_school_us_history:rc::olmes",
    "mmlu_high_school_world_history:rc::olmes",
    "mmlu_human_aging:rc::olmes",
    "mmlu_human_sexuality:rc::olmes",
    "mmlu_international_law:rc::olmes",
    "mmlu_jurisprudence:rc::olmes",
    "mmlu_logical_fallacies:rc::olmes",
    "mmlu_machine_learning:rc::olmes",
    "mmlu_management:rc::olmes",
    "mmlu_marketing:rc::olmes",
    "mmlu_medical_genetics:rc::olmes",
    "mmlu_miscellaneous:rc::olmes",
    "mmlu_moral_disputes:rc::olmes",
    "mmlu_moral_scenarios:rc::olmes",
    "mmlu_nutrition:rc::olmes",
    "mmlu_philosophy:rc::olmes",
    "mmlu_prehistory:rc::olmes",
    "mmlu_professional_accounting:rc::olmes",
    "mmlu_professional_law:rc::olmes",
    "mmlu_professional_medicine:rc::olmes",
    "mmlu_professional_psychology:rc::olmes",
    "mmlu_public_relations:rc::olmes",
    "mmlu_security_studies:rc::olmes",
    "mmlu_sociology:rc::olmes",
    "mmlu_us_foreign_policy:rc::olmes",
    "mmlu_virology:rc::olmes",
    "mmlu_world_religions:rc::olmes",
    # mt tasks
    "mt_mbpp:bash:3shot:bpb::none",
    "mt_mbpp:c:3shot:bpb::none",
    "mt_mbpp:cpp:3shot:bpb::none",
    "mt_mbpp:csharp:3shot:bpb::none",
    "mt_mbpp:go:3shot:bpb::none",
    "mt_mbpp:haskell:3shot:bpb::none",
    "mt_mbpp:java:3shot:bpb::none",
    "mt_mbpp:javascript:3shot:bpb::none",
    "mt_mbpp:matlab:3shot:bpb::none",
    "mt_mbpp:php:3shot:bpb::none",
    "mt_mbpp:python:3shot:bpb::none",
    "mt_mbpp:r:3shot:bpb::none",
    "mt_mbpp:ruby:3shot:bpb::none",
    "mt_mbpp:rust:3shot:bpb::none",
    "mt_mbpp:scala:3shot:bpb::none",
    "mt_mbpp:swift:3shot:bpb::none",
    "mt_mbpp:typescript:3shot:bpb::none",
    #
    "naturalqs:rc::gen2mc",
    "piqa:rc::olmes:full",
    "sciq::olmo1",
    "socialiqa:rc::olmes:full",
    "squad:rc::gen2mc",
    "winogrande:rc::olmes:full",
]

TASK_GROUPS: Dict[str, List[str]] = {
    "all": list(list_tasks()),
    "olmo2_dev_1b": OLMO2_DEV_1B_TASKS,
    "olmo3:dev:1b": OLMO3_DEV_1B_TASKS,
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

    tasks = list(set(tasks))
    tasks.sort()

    return tasks
