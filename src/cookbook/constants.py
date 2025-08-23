import json

TRANSFORMERS_GIT_URL = "https://github.com/huggingface/transformers.git"
TRANSFORMERS_COMMIT_HASH = "241c04d36867259cdf11dbb4e9d9a60f9cb65ebc"  # v4.47.1

OLMOE_CONVERSION_SCRIPT = "src/transformers/models/olmoe/convert_olmoe_weights_to_hf.py"
OLMO2_CONVERSION_SCRIPT = "src/transformers/models/olmo2/convert_olmo2_weights_to_hf.py"

AI2_OLMO_GIT_URL = "https://github.com/allenai/OLMo.git"
AI2_OLMO_CORE_GIT_URL = "https://github.com/allenai/OLMo-core.git"

OLMOE_COMMIT_HASH = "04a2da53db172bd9a0450705592ed50888bdcaa7"
OLMOE_UNSHARD_SCRIPT = "scripts/unshard.py"

OLMO2_COMMIT_HASH = "69362b95c66655191d513e9c1420d54aa8477d92"
OLMO2_UNSHARD_SCRIPT = "scripts/unshard.py"

OLMO_CORE_COMMIT_HASH = "9bad23d9a78e62101699a585a8fde3d69dba5616"
OLMO_CORE_V2_COMMIT_HASH = "1662d0d4f3e628ebb68591e311cce68737c094c4"
OLMO_CORE_UNSHARD_CONVERT_SCRIPT = "src/examples/huggingface/convert_checkpoint_to_hf.py"
OLMO_CORE_CONVERT_DTYPES = ["float32", "bfloat16", "float16"]
OLMO_CORE_CONVERT_FROM_HF_SCRIPT = "src/examples/huggingface/convert_checkpoint_from_hf.py"

DEFAULT_OLMOE_TOKENIZER = "allenai/eleuther-ai-gpt-neox-20b-pii-special"
DEFAULT_OLMO2_TOKENIZER = "allenai/dolma2-tokenizer"
DEFAULT_OLMO_CORE_TOKENIZER = "allenai/OLMo-2-1124-7B"

BEAKER_DEFAULT_WORKSPACE = "ai2/oe-data"
BEAKER_DEFAULT_BUDGET = "ai2/oe-data"
BEAKER_DEFAULT_PRIORITY = "normal"

BEAKER_PY_MAX_VERSION = "1.34.1"
BEAKER_GANTRY_MAX_VERSION = "1.14.1"

OLMO_TYPES = ["olmoe", "olmo2", "olmo-core", "olmo-core-v2"]

WEKA_MOUNTS = [
    "ai1-default",
    "climate-default",
    "dfive-default",
    "mosaic-default",
    "nora-default",
    "nsf-uchicago-apto",
    "oe-adapt-default",
    "oe-data-default",
    "oe-eval-default",
    "oe-training-default",
    "prior-default",
    "reviz-default",
    "skylight-default",
]

BEAKER_KNOWN_CLUSTERS = {
    "aus": [
        "ai2/jupiter-cirrascale-2",
        "ai2/neptune-cirrascale",
        "ai2/saturn-cirrascale",
        "ai2/ceres-cirrascale",
    ],
    "aus80g": [
        "ai2/jupiter-cirrascale-2",
        "ai2/saturn-cirrascale",
        "ai2/ceres-cirrascale",
    ],
    "goog": ["ai2/augusta-google-1"],
    "h100": [
        "ai2/augusta-google-1",
        "ai2/jupiter-cirrascale-2",
        "ai2/ceres-cirrascale",
    ],
    "a100": [
        "ai2/saturn-cirrascale",
    ],
    "l40": [
        "ai2/neptune-cirrascale",
    ],
    "80g": [
        "ai2/augusta-google-1",
        "ai2/jupiter-cirrascale-2",
        "ai2/saturn-cirrascale",
        "ai2/ceres-cirrascale",
    ],
}

MMLU_CATEGORIES = [
    "mmlu_abstract_algebra",
    "mmlu_anatomy",
    "mmlu_astronomy",
    "mmlu_business_ethics",
    "mmlu_clinical_knowledge",
    "mmlu_college_biology",
    "mmlu_college_chemistry",
    "mmlu_college_computer_science",
    "mmlu_college_mathematics",
    "mmlu_college_medicine",
    "mmlu_college_physics",
    "mmlu_computer_security",
    "mmlu_conceptual_physics",
    "mmlu_econometrics",
    "mmlu_electrical_engineering",
    "mmlu_elementary_mathematics",
    "mmlu_formal_logic",
    "mmlu_global_facts",
    "mmlu_high_school_biology",
    "mmlu_high_school_chemistry",
    "mmlu_high_school_computer_science",
    "mmlu_high_school_european_history",
    "mmlu_high_school_geography",
    "mmlu_high_school_government_and_politics",
    "mmlu_high_school_macroeconomics",
    "mmlu_high_school_mathematics",
    "mmlu_high_school_microeconomics",
    "mmlu_high_school_physics",
    "mmlu_high_school_psychology",
    "mmlu_high_school_statistics",
    "mmlu_high_school_us_history",
    "mmlu_high_school_world_history",
    "mmlu_human_aging",
    "mmlu_human_sexuality",
    "mmlu_international_law",
    "mmlu_jurisprudence",
    "mmlu_logical_fallacies",
    "mmlu_machine_learning",
    "mmlu_management",
    "mmlu_marketing",
    "mmlu_medical_genetics",
    "mmlu_miscellaneous",
    "mmlu_moral_disputes",
    "mmlu_moral_scenarios",
    "mmlu_nutrition",
    "mmlu_philosophy",
    "mmlu_prehistory",
    "mmlu_professional_accounting",
    "mmlu_professional_law",
    "mmlu_professional_medicine",
    "mmlu_professional_psychology",
    "mmlu_public_relations",
    "mmlu_security_studies",
    "mmlu_sociology",
    "mmlu_us_foreign_policy",
    "mmlu_virology",
    "mmlu_world_religions",
]

MMLU_SUBCATEGORIES = {
    "stem": ['abstract_algebra', 'astronomy', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_physics', 'computer_security', 'conceptual_physics', 'electrical_engineering', 'elementary_mathematics', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_mathematics', 'high_school_physics', 'high_school_statistics', 'machine_learning'],
    "humanities": ['formal_logic', 'high_school_european_history', 'high_school_us_history', 'high_school_world_history', 'international_law', 'jurisprudence', 'logical_fallacies', 'moral_disputes', 'moral_scenarios', 'philosophy', 'prehistory', 'professional_law', 'world_religions'],
    "social_sciences": ['econometrics', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_microeconomics', 'high_school_psychology', 'human_sexuality', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy'],
    "other": ['anatomy', 'business_ethics', 'clinical_knowledge', 'college_medicine', 'global_facts', 'human_aging', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'nutrition', 'professional_accounting', 'professional_medicine', 'virology']
}

MMLU_PRO_CATEGORIES = [
    "mmlu_pro_math",
    "mmlu_pro_health",
    "mmlu_pro_physics",
    "mmlu_pro_business",
    "mmlu_pro_biology",
    "mmlu_pro_chemistry",
    "mmlu_pro_computer science",
    "mmlu_pro_economics",
    "mmlu_pro_engineering",
    "mmlu_pro_philosophy",
    "mmlu_pro_other",
    "mmlu_pro_history",
    "mmlu_pro_psychology",
    "mmlu_pro_law",
]

HELMET_SUITES = {
    "helmet_cite__131072::suite": ["helmet_alce_asqa_700__131072::std", "helmet_alce_qampari_700__131072::std"],
    "helmet_cite__16384::suite": ["helmet_alce_asqa_75__16384::std", "helmet_alce_qampari_75__16384::std"],
    "helmet_cite__32768::suite": ["helmet_alce_asqa_165__32768::std", "helmet_alce_qampari_165__32768::std"],
    "helmet_cite__65536::suite": ["helmet_alce_asqa_345__65536::std", "helmet_alce_qampari_345__65536::std"],
    "helmet_cite__8192::suite": ["helmet_alce_asqa_30__8192::std", "helmet_alce_qampari_30__8192::std"],
    "helmet_icl__131072::suite": [
        "helmet_icl_banking77_5900shot_balance__131072::std",
        "helmet_icl_clinic150_7050shot_balance__131072::std",
        "helmet_icl_nlu_8296shot_balance__131072::std",
        "helmet_icl_trec_coarse_6600shot_balance__131072::std",
        "helmet_icl_trec_fine_6400shot_balance__131072::std",
    ],
    "helmet_icl__16384::suite": [
        "helmet_icl_banking77_720shot_balance__16384::std",
        "helmet_icl_clinic150_880shot_balance__16384::std",
        "helmet_icl_nlu_1020shot_balance__16384::std",
        "helmet_icl_trec_coarse_800shot_balance__16384::std",
        "helmet_icl_trec_fine_800shot_balance__16384::std",
    ],
    "helmet_icl__32768::suite": [
        "helmet_icl_banking77_1450shot_balance__32768::std",
        "helmet_icl_clinic150_1750shot_balance__32768::std",
        "helmet_icl_nlu_2040shot_balance__32768::std",
        "helmet_icl_trec_coarse_1600shot_balance__32768::std",
        "helmet_icl_trec_fine_1600shot_balance__32768::std",
    ],
    "helmet_icl__65536::suite": [
        "helmet_icl_banking77_2900shot_balance__65536::std",
        "helmet_icl_clinic150_3525shot_balance__65536::std",
        "helmet_icl_nlu_4080shot_balance__65536::std",
        "helmet_icl_trec_coarse_3300shot_balance__65536::std",
        "helmet_icl_trec_fine_3200shot_balance__65536::std",
    ],
    "helmet_icl__8192::suite": [
        "helmet_icl_banking77_360shot_balance__8192::std",
        "helmet_icl_clinic150_440shot_balance__8192::std",
        "helmet_icl_nlu_510shot_balance__8192::std",
        "helmet_icl_trec_coarse_400shot_balance__8192::std",
        "helmet_icl_trec_fine_400shot_balance__8192::std",
    ],
    "helmet_longqa__131072::suite": [
        "helmet_infbench_choice_eng_130862__131072::std",
        "helmet_infbench_qa_eng_130862__131072::std",
        "helmet_narrativeqa_130772__131072::std",
    ],
    "helmet_longqa__16384::suite": [
        "helmet_infbench_choice_eng_16174__16384::std",
        "helmet_infbench_qa_eng_16174__16384::std",
        "helmet_narrativeqa_16084__16384::std",
    ],
    "helmet_longqa__32768::suite": [
        "helmet_infbench_choice_eng_32558__32768::std",
        "helmet_infbench_qa_eng_32558__32768::std",
        "helmet_narrativeqa_32468__32768::std",
    ],
    "helmet_longqa__65536::suite": [
        "helmet_infbench_choice_eng_65326__65536::std",
        "helmet_infbench_qa_eng_65326__65536::std",
        "helmet_narrativeqa_65236__65536::std",
    ],
    "helmet_longqa__8192::suite": [
        "helmet_infbench_choice_eng_7982__8192::std",
        "helmet_infbench_qa_eng_7982__8192::std",
        "helmet_narrativeqa_7892__8192::std",
    ],
    "helmet_niah__131072::suite": [
        "helmet_ruler_cwe__131072::std",
        "helmet_ruler_fwe__131072::std",
        "helmet_ruler_niah_mk_1__131072::std",
        "helmet_ruler_niah_mk_2__131072::std",
        "helmet_ruler_niah_mk_3__131072::std",
        "helmet_ruler_niah_mq__131072::std",
        "helmet_ruler_niah_mv__131072::std",
        "helmet_ruler_niah_s_1__131072::std",
        "helmet_ruler_niah_s_2__131072::std",
        "helmet_ruler_niah_s_3__131072::std",
        "helmet_ruler_qa_1__131072::std",
        "helmet_ruler_qa_2__131072::std",
        "helmet_ruler_vt__131072::std",
    ],
    "helmet_niah__65536::suite": [
        "helmet_ruler_cwe__65536::std",
        "helmet_ruler_fwe__65536::std",
        "helmet_ruler_niah_mk_1__65536::std",
        "helmet_ruler_niah_mk_2__65536::std",
        "helmet_ruler_niah_mk_3__65536::std",
        "helmet_ruler_niah_mq__65536::std",
        "helmet_ruler_niah_mv__65536::std",
        "helmet_ruler_niah_s_1__65536::std",
        "helmet_ruler_niah_s_2__65536::std",
        "helmet_ruler_niah_s_3__65536::std",
        "helmet_ruler_qa_1__65536::std",
        "helmet_ruler_qa_2__65536::std",
        "helmet_ruler_vt__65536::std",
    ],
    "helmet_rag__131072::suite": [
        "helmet_kilt_hotpotqa__131072::std",
        "helmet_kilt_nq__131072::std",
        "helmet_kilt_popqa_3__131072::std",
        "helmet_kilt_triviaqa__131072::std",
    ],
    "helmet_rag__16384::suite": [
        "helmet_kilt_hotpotqa__16384::std",
        "helmet_kilt_nq__16384::std",
        "helmet_kilt_popqa_3__16384::std",
        "helmet_kilt_triviaqa__16384::std",
    ],
    "helmet_rag__32768::suite": [
        "helmet_kilt_hotpotqa__32768::std",
        "helmet_kilt_nq__32768::std",
        "helmet_kilt_popqa_3__32768::std",
        "helmet_kilt_triviaqa__32768::std",
    ],
    "helmet_rag__65536::suite": [
        "helmet_kilt_hotpotqa__65536::std",
        "helmet_kilt_nq__65536::std",
        "helmet_kilt_popqa_3__65536::std",
        "helmet_kilt_triviaqa__65536::std",
    ],
    "helmet_rag__8192::suite": [
        "helmet_kilt_hotpotqa__8192::std",
        "helmet_kilt_nq__8192::std",
        "helmet_kilt_popqa_3__8192::std",
        "helmet_kilt_triviaqa__8192::std",
    ],
    "helmet_recall__131072::suite": [
        "helmet_json_kv__131072::std",
        "helmet_recall_ruler_niah_mk_2__131072::std",
        "helmet_recall_ruler_niah_mk_3__131072::std",
        "helmet_recall_ruler_niah_mv__131072::std",
    ],
    "helmet_recall__16384::suite": [
        "helmet_json_kv__16384::std",
        "helmet_ruler_niah_mk_2__16384::std",
        "helmet_ruler_niah_mk_3__16384::std",
        "helmet_ruler_niah_mv__16384::std",
    ],
    "helmet_recall__32768::suite": [
        "helmet_json_kv__32768::std",
        "helmet_ruler_niah_mk_2__32768::std",
        "helmet_ruler_niah_mk_3__32768::std",
        "helmet_ruler_niah_mv__32768::std",
    ],
    "helmet_recall__65536::suite": [
        "helmet_json_kv__65536::std",
        "helmet_recall_ruler_niah_mk_2__65536::std",
        "helmet_recall_ruler_niah_mk_3__65536::std",
        "helmet_recall_ruler_niah_mv__65536::std",
    ],
    "helmet_recall__8192::suite": [
        "helmet_json_kv__8192::std",
        "helmet_ruler_niah_mk_2__8192::std",
        "helmet_ruler_niah_mk_3__8192::std",
        "helmet_ruler_niah_mv__8192::std",
    ],
    "helmet_rerank__131072::suite": ["helmet_msmarco_rerank_psg__131072::std"],
    "helmet_rerank__16384::suite": ["helmet_msmarco_rerank_psg__16384::std"],
    "helmet_rerank__32768::suite": ["helmet_msmarco_rerank_psg__32768::std"],
    "helmet_rerank__65536::suite": ["helmet_msmarco_rerank_psg__65536::std"],
    "helmet_rerank__8192::suite": ["helmet_msmarco_rerank_psg__8192::std"],
    "helmet_summ__131072::suite": [
        "helmet_infbench_sum_eng_129672__131072::std",
        "helmet_multi_lexsum_130372__131072::std",
    ],
    "helmet_summ__16384::suite": [
        "helmet_infbench_sum_eng_14984__16384::std",
        "helmet_multi_lexsum_15684__16384::std",
    ],
    "helmet_summ__32768::suite": [
        "helmet_infbench_sum_eng_31368__32768::std",
        "helmet_multi_lexsum_32068__32768::std",
    ],
    "helmet_summ__65536::suite": [
        "helmet_infbench_sum_eng_64136__65536::std",
        "helmet_multi_lexsum_64836__65536::std",
    ],
    "helmet_summ__8192::suite": ["helmet_infbench_sum_eng_6792__8192::std", "helmet_multi_lexsum_7492__8192::std"],
}

ALL_CORE_TASKS = [
    "arc_easy",
    "arc_challenge",
    "boolq",
    "csqa",
    "hellaswag",
    "openbookqa",
    "piqa",
    "socialiqa",
    "winogrande",
]

ARC_TASKS = [
    "arc_easy",
    "arc_challenge",
]


ALL_GEN_TASKS = [
    "coqa::olmes",
    "squad::olmes",
    "jeopardy::olmes",
    "naturalqs::olmes",
    "drop::olmes",
    "gsm8k::olmo1",
]

ALL_GEN_XLARGE_TASKS = [
    "coqa::xlarge",
    "squad::xlarge",
    "jeopardy::xlarge",
    "naturalqs::xlarge",
    "drop::xlarge",
]

ALL_MINERVA_TASKS = [
    "minerva_math_algebra",
    "minerva_math_counting_and_probability",
    "minerva_math_geometry",
    "minerva_math_intermediate_algebra",
    "minerva_math_number_theory",
    "minerva_math_prealgebra",
    "minerva_math_precalculus",
]

ALL_GSM_SYMB_TASKS = [
    "gsm_symbolic::olmo3", 
    "gsm_symbolic:p1::olmo3",
    "gsm_symbolic:p2::olmo3"
]

BASIC_SKILLS = [
    "basic_skills_arithmetic",
    "basic_skills_coding",
    "basic_skills_common_knowledge",
    "basic_skills_logical_reasoning",
    "basic_skills_string_operations",
    "basic_skills_pattern",
]

AGI_EVAL_ENGLISH_TASKS = [
    "lsat-ar",
    "lsat-lr",
    "lsat-rc",
    "logiqa-en",
    "sat-math",
    "sat-en",
    "aqua-rat",
    "sat-en-without-passage",
    "gaokao-english",
]

BBH_TASKS = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
]


ALL_CODEX_TASKS = [
    "codex_humaneval:temp0.8",
    "codex_humanevalplus:temp0.8",
    "mbpp::none",
    "mbppplus::none",
    "bigcodebench::none",
    "bigcodebench_hard::none",
]

STARCODER_CODEX_TASKS = [
    "codex_humaneval::starcoder_pass@1",
    "codex_humaneval::starcoder_pass@10",
    "mbpp::starcoder_pass@1",
    "mbpp::starcoder_pass@10",
]

STARCODER_PASS_AT_1_TASKS = [
    "codex_humaneval::starcoder_pass@1",
    "mbpp::starcoder_pass@1",
]

STARCODER_INFILLING = {
    "context_kwargs": {"lead_token": "<fim_prefix>", "center_token": "<fim_suffix>", "end_token": "<fim_middle>"},
    "generation_kwargs": {
        "stop_sequences": ["<|eot_id|>", "<|endoftext|>", "<|filename|>", "<file_sep>"],
    },
}

SANTACODER_INFILLING = {
    "context_kwargs": {"lead_token": "<fim-prefix>", "center_token": "<fim-suffix>", "end_token": "<fim-middle>"},
    "generation_kwargs": {
        "stop_sequences": ["<|eot_id|>", "<|endoftext|>", "<|filename|>", "<file_sep>"],
    },
}

DEEPSEEK_CODER_INFILLING = {
    "context_kwargs": {
        "lead_token": "<｜fim▁begin｜>",
        "center_token": "<｜fim▁hole｜>",
        "end_token": "<｜fim▁end｜>",
    },
    "generation_kwargs": {"stop_sequences": ["<|eot_id|>", "<|endoftext|>", "<|EOT|>"]},
}

L2C_INFILLING = {
    "context_kwargs": {
        "lead_token": "<|fim_prefix|>",
        "center_token": "<|fim_suffix|>",
        "end_token": "<|fim_middle|>",
    },
    "generation_kwargs": {
        "stop_sequences": ["<|endoftext|>", "<|filename|>", "<|file_sep|>"],
    },
}

FIM_TOKENS = {
    "santacoder": SANTACODER_INFILLING,
    "starcoder": STARCODER_INFILLING,
    "deepseek": DEEPSEEK_CODER_INFILLING,
    "l2c": L2C_INFILLING,
}

FIM_TASKS = [
    "codex_humanevalfim_single",
    "codex_humanevalfim_multi",
    "codex_humanevalfim_random",
]


CRUX_EVAL_TASKS = [
    "cruxeval_input:pass@5",
    "cruxeval_output:pass@5",
]

MULTILINGUAL_MBPP_TASKS = [
    "mt_mbpp:bash",
    "mt_mbpp:c",
    "mt_mbpp:cpp",
    "mt_mbpp:csharp",
    "mt_mbpp:go",
    "mt_mbpp:haskell",
    "mt_mbpp:java",
    "mt_mbpp:javascript",
    "mt_mbpp:matlab",
    "mt_mbpp:php",
    "mt_mbpp:python",
    "mt_mbpp:r",
    "mt_mbpp:ruby",
    "mt_mbpp:rust",
    "mt_mbpp:scala",
    "mt_mbpp:swift",
    "mt_mbpp:typescript",
]

MULTILINGUAL_MBPP_TASKS_V2 = [
    "mt_mbpp_v2fix:bash",
    "mt_mbpp_v2fix:c",
    "mt_mbpp_v2fix:cpp",
    "mt_mbpp_v2fix:csharp",
    "mt_mbpp_v2fix:go",
    "mt_mbpp_v2fix:haskell",
    "mt_mbpp_v2fix:java",
    "mt_mbpp_v2fix:javascript",
    "mt_mbpp_v2fix:matlab",
    "mt_mbpp_v2fix:php",
    "mt_mbpp_v2fix:python",
    "mt_mbpp_v2fix:r",
    "mt_mbpp_v2fix:ruby",
    "mt_mbpp_v2fix:rust",
    "mt_mbpp_v2fix:scala",
    "mt_mbpp_v2fix:swift",
    "mt_mbpp_v2fix:typescript",
]

MULTIPL_E_HE_TASKS = [
    'multipl_e_humaneval:cpp::olmo3', 
    'multipl_e_humaneval:java::olmo3', 
    'multipl_e_humaneval:js::olmo3', 
    'multipl_e_humaneval:php::olmo3', 
    'multipl_e_humaneval:rs::olmo3', 
    'multipl_e_humaneval:sh::olmo3', 
]

MULTIPL_E_MBPP_TASKS = [
    'multipl_e_mbpp:cpp::olmo3', 
    'multipl_e_mbpp:java::olmo3', 
    'multipl_e_mbpp:js::olmo3', 
    'multipl_e_mbpp:php::olmo3', 
    'multipl_e_mbpp:rs::olmo3',
]

MT_EVAL_TASKS = [
    "refinement_single",
    "refinement_multi",
    "expansion_single",
    "expansion_multi",
    "follow-up_single",
    "follow-up_multi",
    "recollection_single_cls",
    "recollection_multi_cls",
    "recollection_single_global-inst",
    "recollection_multi_global-inst",
]

IFBENCH_MT_TASKS = [
    "wildchat_unused_withRewrite",
    "ood_wildchat_unused_withRewrite",
]

IFEVAL_MT_TASKS = [
    "wildchat_unused_withoutType",
    "wildchat_unused_withRewrite",
    "ood_wildchat_unused_withoutType",
    "ood_wildchat_unused_withRewrite",
]

MULTITURN_ALPACAEVAL_TASKS = [
    "context_switch",
    "related_context",
    "repeated",
    "paraphrase",
    "add_constraint",
    "just_constraint",
    "user_dependent",
    "assistant_dependent",
    "self_compare_context_switch",
    "self_compare_related_context",
    "self_compare_repeated",
    "self_compare_paraphrase",
    "self_compare_add_constraint",
    "self_compare_just_constraint",
    "self_compare_user_dependent",
    "self_compare_assistant_dependent",
    "single_turn_add_constraint",
    "single_turn_just_constraint",
    "single_turn_user_dependent",
    "single_turn_assistant_dependent",
    "weak_context_repeated",
    "weak_context_paraphrase",
    "weak_context_add_constraint",
    "weak_context_just_constraint",
    "split_prompts",
    "single_turn_split_prompts",
    "self_compare_split_prompts",
]

# For now, adapt has disabled some of the styled subtasks
# https://github.com/allenai/oe-eval-internal/pull/601
STYLED_TASKS = [
    "missense",
    "leetspeak",
    "grammar",
    "random_case",
    # "lower",
    # "upper",
    "keywords",
    "phonetic",
    "mechanical",
    "aae",
    "informal",
    "afrikaans_backtranslated",
    # "japanese_backtranslated",
    # "swahili_backtranslated",
]

STYLED_TASKS_POPQA = [
    "leetspeak",
    "random_case",
    # "lower",
    # "upper",
    "phonetic",
    "mechanical",
]

OMEGA_SUB_CATEGORIES = {
    # Compositional has train, test splits
    "compositional": [
        "comp_polynomial_gcd",
        "comp_n_gon",
        "comp_circles_algebra",
        "comp_parametric_intersection",
        "comp_matrix_rank",
        "comp_vertex_color",
        "comp_grid_chips",
    ],
    "explorative": [
        "algebra_func_area",
        "algebra_func_derivative_sign",
        "algebra_func_extrema",
        "algebra_func_extrema_coords",
        "algebra_func_intersection",
        "algebra_func_intersection_coords",
        "algebra_func_zeros",
        "algebra_linear_equation",
        "algebra_polynomial_roots",
        "arithmetic_gcd",
        "arithmetic_list_prime_factors",
        "arithmetic_mixed",
        "arithmetic_matrix_determinant",
        "arithmetic_matrix_eigenvalues",
        "arithmetic_matrix_inverse",
        "arithmetic_matrix_multiplication",
        "arithmetic_matrix_power",
        "arithmetic_matrix_rank",
        "arithmetic_matrix_svd",
        "combinatory_distribution",
        "combinatory_pattern_matching",
        "combinatory_probability_at_least_n_specific_fixed",
        "combinatory_probability_exactly_n_specific_fixed",
        "combinatory_probability_no_fixed_points",
        "combinatory_probability_no_specific_letter_fixed",
        "geometry_basic",
        "geometry_circle",
        "geometry_perpendicular_intersection",
        "geometry_polygon",
        "geometry_polygon_rotation",
        "geometry_triangle",
        "geometry_polygon_chords",
        "geometry_polygon_color",
        "logic_gridworld_blocked",
        "logic_gridworld_knight_move",
        "logic_gridworld_rookmove",
        "logic_zebralogic",
        "logic_puzzles_grid_chip",
        "numbertheory_lte_qr",
        "numbertheory_ordered_lte",
        "numbertheory_qr_sum",
    ],

    # Transformative has train, test splits
    "transformative": [
        "trans_matrix_rank",
        "trans_func_intersection",
        "trans_de_moivre",
        "trans_prob_letter",
        "trans_integrations",
        "trans_gridworld",
        "trans_circles",
    ]
}


OE_EVAL_GIT_URL = "git@github.com:allenai/oe-eval-internal.git"
OE_EVAL_COMMIT_HASH = None
OE_EVAL_LAUNCH_COMMAND = "oe_eval/launch.py"
BEAKER_PRIORITIES = ["low", "normal", "high", "urgent"]


# # # # # # # # # # # # # LEGACY NAMED GROUPS # # # # # # # # # # # # #
# The following named/display groups are preserved as reference only. #
# Do NOT uncomment them, use new named_tasks.py instead.              #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# ALL_NAMED_GROUPS = {
#     "mmlu:rc": [f"{category}:rc::olmes" for category in MMLU_CATEGORIES],
#     "mmlu:mc": [f"{category}:mc::olmes" for category in MMLU_CATEGORIES],
#     "core:rc": [f"{task}:rc::olmes" for task in ALL_CORE_TASKS],
#     "core:mc": [f"{task}:mc::olmes" for task in ALL_CORE_TASKS],
#     "basic:rc": [f"{task}:rc::olmes" for task in BASIC_SKILLS],
#     "basic:mc": [f"{task}:mc::olmes" for task in BASIC_SKILLS],
#     "mmlu_pro:mc": [f"{category}:mc::none" for category in MMLU_PRO_CATEGORIES],
#     "gen": ALL_GEN_TASKS,
#     "gen-no-jp": [task for task in ALL_GEN_TASKS if task != "jeopardy::olmes"],
#     "minerva": ALL_MINERVA_TASKS,
#     "math": ALL_MATH_TASKS,
#     "gsm-symb": ALL_GSM_SYMB_TASKS,
#     "code": ALL_CODEX_TASKS,
#     "agi_eval": ALL_AGI_EVAL_TASKS,
#     "starcoder": STARCODER_CODEX_TASKS,
#     "starcoder::pass@1": STARCODER_PASS_AT_1_TASKS,
#     "code-no-bcb": [task for task in ALL_CODEX_TASKS if "bigcodebench" not in task],
#     "fim": FIM_TASKS,
#     "mt_mbpp": [f"mt_mbpp:{language}" for language in MULTILINGUAL_MBPP_LANGUAGES],
#     "mt_mbpp_v2fix": [f"mt_mbpp_v2fix:{language}" for language in MULTILINGUAL_MBPP_LANGUAGES],
# }

# for helmet_length in (int(2**i) for i in range(13, 18)):
#     ALL_NAMED_GROUPS[f"helmet:{helmet_length // 2 ** 10}k"] = list(
#         set(
#             task
#             for group_name, tasks in HELMET_SUITES.items()
#             for task in tasks
#             if group_name.endswith(f"__{helmet_length}::suite") and not group_name.startswith("helmet_all")
#         )
#     )


# ALL_DISPLAY_TASKS = {
#     "olmo2:paper": [
#         r"arc_challenge:(rc|mc)::olmes$",
#         r"hellaswag:(rc|mc)::olmes$",
#         r"winogrande:(rc|mc)::olmes$",
#         r"naturalqs::olmes$",
#         r"drop::olmes$",
#         r"agieval.*::olmes$",
#         r"^gsm8k::olmes$",
#         r"^mmlu:rc$",
#         r"^mmlu:mc$",
#         r"^mmlu_pro:mc$",
#         r"^agi_eval$",
#         r"^triviaqa::olmes$",
#     ],
#     "olmo2:dev:7b": [
#         r"arc_challenge:mc.*",
#         r"arc_easy:mc.*",
#         r"hellaswag:mc.*",
#         r"naturalqs.*",
#         r"^gsm8k::olmo1$",
#         r"^mmlu:mc$",
#         r"^core:mc$",
#         r"^gen$",
#     ],
#     "olmo2:dev:1b": [
#         r"arc_challenge:rc.*",
#         r"arc_easy:rc.*",
#         r"hellaswag:rc.*",
#         r"^gsm8k::olmo1$",
#         r"^mmlu:rc$",
#         r"^core:rc$",
#     ],
#     "olmo3:dev:1b": [
#         "^arc_challenge.*olmes",     # should return mc, rc, bpb variants, full or not
#         "^arc_easy.*olmes",
#         "^basic_skills.*olmes",
#         "^codex_humaneval.*3shot",
#         "^coqa.*gen2mc",
#         "^csqa.*olmes",
#         "^drop.*gen2mc",
#         "^hellaswag.*olmes",
#         "^jeopardy.*gen2mc",
#         "^lab_bench.*",
#         "^lambada.*",
#         "^mbpp.*3shot",
#         "^medmcqa.*none",
#         "^medqa.*none",
#         "^minerva.*olmes$",     # doesn't return average
#         # "minerva",
#         "^mmlu.*olmes$",    # doesn't return average
#         # "mmlu:rc",
#         "^mt_mbpp_v2fix.*",           # still returns average
#         "^naturalqs.*gen2mc",
#         "^piqa.*olmes",
#         "^qasper_yesno.*olmes",
#         "^sciq.*olmo3",
#         "^sciriff_yesno.*olmes",
#         "^socialiqa.*olmes",
#         "^squad.*gen2mc",
#         "^winogrande.*olmes",
#         "ultrachat_masked_ppl",
#         "wildchat_masked_ppl",
#     ],
#     "olmo3:dev:1b:mini": [
#         "^arc_challenge:rc.*",
#         "^arc_easy:rc.*",
#         "^codex_humaneval.*3shot",
#         "^hellaswag:rc.*olmes",
#         "^mbpp.*3shot",
#         "^minerva$",
#         "^mt_mbpp_v2fix$",
#         "^winogrande:rc.*olmes",
#         "basic:rc",
#         "core:rc"
#         "mmlu:bpb",
#         "mmlu:rc",
#     ],
#     "olmo3:dev:7b:mini": [
#         "^arc_challenge:mc::olmes:full",
#         "^arc_easy:mc::olmes:full",
#         "^hellaswag:rc.*olmes",
#         "^codex_humaneval::olmo3",
#         "^mbpp:3shot::olmo3",
#         "^gsm-symb$",
#         "^minerva$",
#         "^mt_mbpp_v2fix$",
#         "^core:mc$",
#         "^mmlu:mc$",
#     ],
#     "helmet:8k": [r"^helmet:8k$"],
#     "helmet:16k": [r"^helmet:16k$"],
#     "helmet:32k": [r"^helmet:32k$"],
#     "helmet:64k": [r"^helmet:64k$"],
#     "helmet:128k": [r"^helmet:128k$"],
# }


# SHORT_NAMES = {
#      r"::olmes$": "",
#      r"::olmes:full$": "",
#      r"^gsm8k::olmo1$": "GSM*",
#      r"^naturalqs": "NQ",
#      r"^(arc\_\w)\w+": r"\1",
#      r"^hellaswag": "HSwag",


#      r"^winogrande": "WinoG",
#      r"^codex_humaneval": "humaneval",
#      r"::olmo3$": "",
#      r"::none$": "",
#      r":3shot": "",
#      r"::full$": "",
#  }


# ALL_EVAL_TASKS = {
#     "olmo2:paper": [
#         "olmo_2_heldout::olmes",
#         "olmo_2_generative::olmes",
#         "core_9mcqa::olmes", # evaluates both mc and rc
#         "mmlu:mc::olmes",
#         "mmlu:rc::olmes",
#         "triviaqa::olmes"
#     ],
#     "olmo3:dev:1b:vllm": [
#         "arc_challenge:rc::olmes:full",
#         "arc_easy:rc::olmes:full",
#         "basic_skills:rc::olmes",
#         "codex_humaneval:3shot:bpb::none",
#         "coqa:rc::gen2mc",
#         "csqa:rc::olmes:full",
#         "drop:rc::gen2mc",
#         "hellaswag:rc::olmes:full",
#         "jeopardy:rc::gen2mc",
#         "lab_bench_dbqa",
#         "lab_bench_protocolqa",
#         "lambada",
#         "mbpp:3shot:bpb::none",
#         "medmcqa:rc::none",
#         "medqa_en:rc::none",
#         "minerva_math::olmes",
#         "mmlu:rc::olmes",
#         "mt_mbpp_v2fix",
#         "naturalqs:rc::gen2mc",
#         "piqa:rc::olmes:full",
#         "qasper_yesno:rc::olmes",
#         "sciq:rc::olmo3",
#         "sciriff_yesno:rc::olmes",
#         "socialiqa:rc::olmes:full",
#         "squad:rc::gen2mc",
#         "winogrande:rc::olmes:full",
#     ],
#     "olmo3:dev:1b:hf": [
#         "ultrachat_masked_ppl",
#         "wildchat_masked_ppl",
#     ],
#     "olmo3:dev:7b:vllm": [
#         "arc_challenge:mc::xlarge",
#         "arc_easy:mc::xlarge",
#         "basic_skills:rc::olmes",
#         "bigcodebench:3shot::olmo3",
#         "codex_humaneval:3shot::olmo3",
#         "codex_humanevalfim_multi:temp0.2",
#         "codex_humanevalfim_random:temp0.2",
#         "codex_humanevalfim_single:temp0.2",
#         "coqa::xlarge",
#         "coqa:mc::gen2mc",
#         "cruxeval_input:pass@5",
#         "cruxeval_output:pass@5",
#         "csqa:mc::xlarge",
#         "deepseek_leetcode::olmo3",
#         "drop::xlarge",
#         "drop:mc::gen2mc",
#         "ds1000:3shot::olmo3",
#         "gsm_symbolic::olmo3",
#         "gsm_symbolic:p1::olmo3",
#         "gsm_symbolic:p2::olmo3",
#         "gsm8k::olmes",
#         "hellaswag:rc::xlarge",
#         "jeopardy::xlarge",
#         "jeopardy:mc::gen2mc",
#         "lab_bench_dbqa:mc",
#         "lab_bench_protocolqa:mc",
#         "lambada",
#         "mbpp:3shot::olmo3",
#         "medmcqa:mc::none",
#         "medqa_en:mc::none",
#         "minerva_math::olmes",
#         "mmlu:mc::olmes",
#         "mt_mbpp_v2fix",
#         "multipl_e:6lang::olmo3",
#         "naturalqs::xlarge",
#         "naturalqs:mc::gen2mc",
#         "openbookqa:mc::xlarge",
#         "piqa:mc::xlarge",
#         "qasper_yesno:mc::olmes",
#         "sciq:mc::xlarge",
#         "sciriff_yesno:mc::olmes",
#         "socialiqa:mc::xlarge",
#         "squad::xlarge",
#         "squad:mc::gen2mc",
#         "winogrande:rc::xlarge",
#     ],
#     "olmo3:dev:7b:hf": [
#         "ultrachat_masked_ppl",
#         "wildchat_masked_ppl",
#     ],
# }
