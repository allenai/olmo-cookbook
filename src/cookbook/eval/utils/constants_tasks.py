RC_TASKS_OLMES = [
    "arc_challenge:rc::olmes:full",
    "arc_easy:rc::olmes:full",
    "boolq:rc::olmes:full",
    "csqa:rc::olmes:full",
    "hellaswag:rc::olmes:full",
    "openbookqa:rc::olmes:full",
    "piqa:rc::olmes:full",
    "socialiqa:rc::olmes:full",
    "winogrande:rc::olmes:full",
    "mmlu_abstract_algebra:rc::olmes:full",
    "mmlu_anatomy:rc::olmes:full",
    "mmlu_astronomy:rc::olmes:full",
    "mmlu_business_ethics:rc::olmes:full",
    "mmlu_clinical_knowledge:rc::olmes:full",
    "mmlu_college_biology:rc::olmes:full",
    "mmlu_college_chemistry:rc::olmes:full",
    "mmlu_college_computer_science:rc::olmes:full",
    "mmlu_college_mathematics:rc::olmes:full",
    "mmlu_college_medicine:rc::olmes:full",
    "mmlu_college_physics:rc::olmes:full",
    "mmlu_computer_security:rc::olmes:full",
    "mmlu_conceptual_physics:rc::olmes:full",
    "mmlu_econometrics:rc::olmes:full",
    "mmlu_electrical_engineering:rc::olmes:full",
    "mmlu_elementary_mathematics:rc::olmes:full",
    "mmlu_formal_logic:rc::olmes:full",
    "mmlu_global_facts:rc::olmes:full",
    "mmlu_high_school_biology:rc::olmes:full",
    "mmlu_high_school_chemistry:rc::olmes:full",
    "mmlu_high_school_computer_science:rc::olmes:full",
    "mmlu_high_school_european_history:rc::olmes:full",
    "mmlu_high_school_geography:rc::olmes:full",
    "mmlu_high_school_government_and_politics:rc::olmes:full",
    "mmlu_high_school_macroeconomics:rc::olmes:full",
    "mmlu_high_school_mathematics:rc::olmes:full",
    "mmlu_high_school_microeconomics:rc::olmes:full",
    "mmlu_high_school_physics:rc::olmes:full",
    "mmlu_high_school_psychology:rc::olmes:full",
    "mmlu_high_school_statistics:rc::olmes:full",
    "mmlu_high_school_us_history:rc::olmes:full",
    "mmlu_high_school_world_history:rc::olmes:full",
    "mmlu_human_aging:rc::olmes:full",
    "mmlu_human_sexuality:rc::olmes:full",
    "mmlu_international_law:rc::olmes:full",
    "mmlu_jurisprudence:rc::olmes:full",
    "mmlu_logical_fallacies:rc::olmes:full",
    "mmlu_machine_learning:rc::olmes:full",
    "mmlu_management:rc::olmes:full",
    "mmlu_marketing:rc::olmes:full",
    "mmlu_medical_genetics:rc::olmes:full",
    "mmlu_miscellaneous:rc::olmes:full",
    "mmlu_moral_disputes:rc::olmes:full",
    "mmlu_moral_scenarios:rc::olmes:full",
    "mmlu_nutrition:rc::olmes:full",
    "mmlu_philosophy:rc::olmes:full",
    "mmlu_prehistory:rc::olmes:full",
    "mmlu_professional_accounting:rc::olmes:full",
    "mmlu_professional_law:rc::olmes:full",
    "mmlu_professional_medicine:rc::olmes:full",
    "mmlu_professional_psychology:rc::olmes:full",
    "mmlu_public_relations:rc::olmes:full",
    "mmlu_security_studies:rc::olmes:full",
    "mmlu_sociology:rc::olmes:full",
    "mmlu_us_foreign_policy:rc::olmes:full",
    "mmlu_virology:rc::olmes:full",
    "mmlu_world_religions:rc::olmes:full",
]

MC_TASKS_OLMES          = [task.replace(":rc", ":mc") for task in RC_TASKS_OLMES]
PARA_TASKS_OLMES        = [task.replace(":rc", ":para") for task in RC_TASKS_OLMES]
ENLARGE_TASKS_OLMES     = [task.replace(":rc", ":enlarge") for task in RC_TASKS_OLMES if 'winogrande' not in task and 'mmlu' not in task]
DISTRACTORS_TASKS_OLMES = [task.replace(":rc", ":distractors") for task in RC_TASKS_OLMES if 'winogrande' not in task and 'mmlu' not in task]

MC_TASKS_COPY_COLORS = [
    "copycolors_2way:mc::none",
    "copycolors_cyclic_2way:mc::none",
    "copycolors_4way:mc::none",
    "copycolors_cyclic_4way:mc::none",
    "copycolors_8way:mc::none",
    "copycolors_cyclic_8way:mc::none",
]

PALOMA = [
    "paloma_4chan_meta_sep::paloma",
    # "paloma_c4_100_domains::paloma", # 28K
    "paloma_c4_en::paloma",
    "paloma_dolma_100_programing_languages::paloma",
    # "paloma_dolma_100_subreddits::paloma", # 92K
    "paloma_dolma-v1_5::paloma",
    "paloma_falcon-refinedweb::paloma",
    "paloma_gab::paloma",
    "paloma_m2d2_s2orc_unsplit::paloma",
    "paloma_m2d2_wikipedia_unsplit::paloma",
    "paloma_manosphere_meta_sep::paloma",
    "paloma_mc4::paloma",
    "paloma_ptb::paloma",
    "paloma_redpajama::paloma",
    # "paloma_twitterAAE_HELM_fixed::paloma", # 100K
    "paloma_wikitext_103::paloma",
]
LLM_COMPRESSION = [
    "arxiv_math::llm_compression",
    "cc::llm_compression",
    "python::llm_compression",
]
CUSTOM_LOSS = [
    'sky_t1::custom_loss', 
    'numia_math::custom_loss', 
    'tulu_if::custom_loss'
]

GEN_TASKS_OLMES = [
    # Core generation-based benchmarks
    # "coqa::olmes:full", # <- coqa is not setup properly (no few shot examples)
    "drop::olmes:full",
    "gsm8k::olmes:full",
    "jeopardy::olmes:full",
    "naturalqs::olmes:full",
    "squad::olmes:full",
    "triviaqa::olmes:full",
]
GEN_TASKS_OLMES_PERTURB_RC = [task.replace('::olmes', ':perturb_rc::olmes') for task in GEN_TASKS_OLMES]

MMLU_PRO_MC = [
    "mmlu_pro_math:mc::none",
    "mmlu_pro_health:mc::none",
    "mmlu_pro_physics:mc::none",
    "mmlu_pro_business:mc::none",
    "mmlu_pro_biology:mc::none",
    "mmlu_pro_chemistry:mc::none",
    "mmlu_pro_computer science:mc::none",
    "mmlu_pro_economics:mc::none",
    "mmlu_pro_engineering:mc::none",
    "mmlu_pro_philosophy:mc::none",
    "mmlu_pro_other:mc::none",
    "mmlu_pro_history:mc::none",
    "mmlu_pro_psychology:mc::none",
    "mmlu_pro_law:mc::none",
]
MMLU_PRO_RC          = [task.replace(":mc::none", ":rc::none") for task in MMLU_PRO_MC]
MMLU_PRO_COT         = [task.replace(":mc::none", ":cot::none") for task in MMLU_PRO_MC]
# MMLU_PRO_COT         = [task.replace(":mc::none", ":cot::llama3.1") for task in MMLU_PRO_MC] # <- broken on base models
# MMLU_PRO_PERTURB_COT = [task.replace(":mc", ":perturb_cot") for task in MMLU_PRO_MC] # <- does not exist yet!

AGI_EVAL_MC = [
    # AGI Eval MC
    "agi_eval_lsat-ar::olmes:full",
    "agi_eval_lsat-lr::olmes:full",
    "agi_eval_lsat-rc::olmes:full",
    "agi_eval_logiqa-en::olmes:full",
    "agi_eval_sat-math::olmes:full",
    "agi_eval_sat-en::olmes:full",
    "agi_eval_aqua-rat::olmes:full",
    "agi_eval_sat-en-without-passage::olmes:full",
    "agi_eval_gaokao-english::olmes:full",
]
AGI_EVAL_RC = [task.replace("::olmes:full", ":rc::none") for task in AGI_EVAL_MC]
AGI_EVAL_COT = [task.replace("::olmes:full", ":cot::none") for task in AGI_EVAL_MC] # ::tulu3 does not work on base models. only this config works currently

MINERVA_MC = [
    # Minerva does not have MC
]

MINERVA_COT = [
    # Minerva CoT (there's also a tulu and llama config)
    "minerva_math_algebra::olmes:full",
    "minerva_math_counting_and_probability::olmes:full",
    "minerva_math_geometry::olmes:full",
    "minerva_math_intermediate_algebra::olmes:full",
    "minerva_math_number_theory::olmes:full",
    "minerva_math_prealgebra::olmes:full",
    "minerva_math_precalculus::olmes:full",
]

BBH_COT = [
    # BBH COT (generation-based)
    "bbh_boolean_expressions:cot::olmes:full",
    "bbh_causal_judgement:cot::olmes:full",
    "bbh_date_understanding:cot::olmes:full",
    "bbh_disambiguation_qa:cot::olmes:full",
    "bbh_dyck_languages:cot::olmes:full",
    "bbh_formal_fallacies:cot::olmes:full",
    "bbh_geometric_shapes:cot::olmes:full",
    "bbh_hyperbaton:cot::olmes:full",
    "bbh_logical_deduction_five_objects:cot::olmes:full",
    "bbh_logical_deduction_seven_objects:cot::olmes:full",
    "bbh_logical_deduction_three_objects:cot::olmes:full",
    "bbh_movie_recommendation:cot::olmes:full",
    "bbh_multistep_arithmetic_two:cot::olmes:full",
    "bbh_navigate:cot::olmes:full",
    "bbh_object_counting:cot::olmes:full",
    "bbh_penguins_in_a_table:cot::olmes:full",
    "bbh_reasoning_about_colored_objects:cot::olmes:full",
    "bbh_ruin_names:cot::olmes:full",
    "bbh_salient_translation_error_detection:cot::olmes:full",
    "bbh_snarks:cot::olmes:full",
    "bbh_sports_understanding:cot::olmes:full",
    "bbh_temporal_sequences:cot::olmes:full",
    "bbh_tracking_shuffled_objects_five_objects:cot::olmes:full",
    "bbh_tracking_shuffled_objects_seven_objects:cot::olmes:full",
    "bbh_tracking_shuffled_objects_three_objects:cot::olmes:full",
    "bbh_web_of_lies:cot::olmes:full",
    "bbh_word_sorting:cot::olmes:full",
]

BBH_MC = [
    # QA Variants of BBH Tasks
    "bbh_boolean_expressions:qa::none",
    "bbh_causal_judgement:qa::none",
    "bbh_date_understanding:qa::none",
    "bbh_disambiguation_qa:qa::none",
    "bbh_dyck_languages:qa::none",
    "bbh_formal_fallacies:qa::none",
    "bbh_geometric_shapes:qa::none",
    "bbh_hyperbaton:qa::none",
    "bbh_logical_deduction_five_objects:qa::none",
    "bbh_logical_deduction_seven_objects:qa::none",
    "bbh_logical_deduction_three_objects:qa::none",
    "bbh_movie_recommendation:qa::none",
    "bbh_multistep_arithmetic_two:qa::none",
    "bbh_navigate:qa::none",
    "bbh_object_counting:qa::none",
    "bbh_penguins_in_a_table:qa::none",
    "bbh_reasoning_about_colored_objects:qa::none",
    "bbh_ruin_names:qa::none",
    "bbh_salient_translation_error_detection:qa::none",
    "bbh_snarks:qa::none",
    "bbh_sports_understanding:qa::none",
    "bbh_temporal_sequences:qa::none",
    "bbh_tracking_shuffled_objects_five_objects:qa::none",
    "bbh_tracking_shuffled_objects_seven_objects:qa::none",
    "bbh_tracking_shuffled_objects_three_objects:qa::none",
    "bbh_web_of_lies:qa::none",
    "bbh_word_sorting:qa::none",
]

PERTURB_COT_TASKS = [
    'agi_eval_aqua-rat:perturb_cot::olmes', 
    'agi_eval_gaokao-english:perturb_cot::olmes', 
    'agi_eval_logiqa-en:perturb_cot::olmes', 
    'agi_eval_lsat-ar:perturb_cot::olmes', 
    'agi_eval_lsat-lr:perturb_cot::olmes', 
    'agi_eval_lsat-rc:perturb_cot::olmes', 
    'agi_eval_sat-en-without-passage:perturb_cot::olmes', 
    'agi_eval_sat-en:perturb_cot::olmes', 
    'agi_eval_sat-math:perturb_cot::olmes', 
    
    'bbh_boolean_expressions:perturb_cot::olmes', 
    'bbh_causal_judgement:perturb_cot::olmes', 
    'bbh_date_understanding:perturb_cot::olmes', 
    'bbh_disambiguation_qa:perturb_cot::olmes', 
    'bbh_dyck_languages:perturb_cot::olmes', 
    'bbh_formal_fallacies:perturb_cot::olmes', 
    'bbh_geometric_shapes:perturb_cot::olmes', 
    'bbh_hyperbaton:perturb_cot::olmes', 
    'bbh_logical_deduction_five_objects:perturb_cot::olmes', 
    'bbh_logical_deduction_seven_objects:perturb_cot::olmes', 
    'bbh_logical_deduction_three_objects:perturb_cot::olmes', 
    'bbh_movie_recommendation:perturb_cot::olmes', 
    'bbh_multistep_arithmetic_two:perturb_cot::olmes', 
    'bbh_navigate:perturb_cot::olmes', 
    'bbh_object_counting:perturb_cot::olmes', 
    'bbh_penguins_in_a_table:perturb_cot::olmes', 
    'bbh_reasoning_about_colored_objects:perturb_cot::olmes', 
    'bbh_ruin_names:perturb_cot::olmes', 
    'bbh_salient_translation_error_detection:perturb_cot::olmes', 
    'bbh_snarks:perturb_cot::olmes', 
    'bbh_sports_understanding:perturb_cot::olmes', 
    'bbh_temporal_sequences:perturb_cot::olmes', 
    'bbh_tracking_shuffled_objects_five_objects:perturb_cot::olmes', 
    'bbh_tracking_shuffled_objects_seven_objects:perturb_cot::olmes', 
    'bbh_tracking_shuffled_objects_three_objects:perturb_cot::olmes', 
    'bbh_web_of_lies:perturb_cot::olmes', 
    'bbh_word_sorting:perturb_cot::olmes', 
    
    'gsm8k:perturb_cot::olmes', 
    
    'minerva_math_algebra:perturb_cot::olmes', 
    'minerva_math_counting_and_probability:perturb_cot::olmes', 
    'minerva_math_geometry:perturb_cot::olmes', 
    'minerva_math_intermediate_algebra:perturb_cot::olmes', 
    'minerva_math_number_theory:perturb_cot::olmes', 
    'minerva_math_prealgebra:perturb_cot::olmes', 
    # TODO: Add the final minerva tasks
]

ADDITIONAL_TASKS_TULU_3 = [
    # (Additional Tulu tasks I've excluded for now)

    # CoT (exact match)
    "tydiqa_english::tulu",
    "ifeval::tulu",
    "truthfulqa::tulu",
    "alpaca_eval_v2::tulu",

    # CoT Long Context (exact match)
    "zero_scrolls_gov_report::tulu",
    "zero_scrolls_summ_screen_fd::tulu",
    "zero_scrolls_qmsum::tulu",
    "zero_scrolls_qasper::tulu",
    "zero_scrolls_narrative_qa::tulu",
    "zero_scrolls_quality::tulu",

    # Code benchmarks
    "bigcodebench::tulu",
    "bigcodebench_hard::tulu",
    "codex_humaneval:temp0.1",
    "codex_humaneval:temp0.8",
    "codex_humaneval::tulu",
    "codex_humanevalplus::tulu"
]

AGI_EVAL_TULU_3 = [
    # AGI Eval CoT (only ::tulu3 has proper configs) -- broken
    "agi_eval_lsat-ar:0shot_cot::tulu3",
    "agi_eval_lsat-lr:0shot_cot::tulu3",
    "agi_eval_lsat-rc:0shot_cot::tulu3",
    "agi_eval_logiqa-en:0shot_cot::tulu3",
    "agi_eval_sat-math:0shot_cot::tulu3",
    "agi_eval_sat-en:0shot_cot::tulu3",
    "agi_eval_aqua-rat:0shot_cot::tulu3",
    "agi_eval_sat-en-without-passage:0shot_cot::tulu3",
    "agi_eval_gaokao-english:0shot_cot::tulu3",
]

MISSING_EVALS = [
    ("falcon_and_cc_eli5_oh_top20p-1B-5xC", ["aime::none"]),
    ("prox_fineweb_pro-1B-5xC", ["aime::none"]),
    ("falcon-1B-5xC", ["aime::none"]),
    ("fineweb_edu_dedup-1B-5xC", ["aime::none"]),
    ("falcon_and_cc_og_eli5_oh_top10p-1B-5xC", ["aime::none"]),
    ("falcon_and_cc-1B-5xC", ["aime::none"]),
    ("falcon_and_cc_tulu_qc_top10-1B-5xC", ["aime::none"]),
    ("falcon_and_cc_eli5_oh_top10p-1B-5xC", ["aime::none"]),
    ("redpajama-1B-5xC", ["aime::none"]),
    ("c4-1B-5xC", ["aime::none"]),
]