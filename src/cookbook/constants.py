import json

from cookbook.utils.average_weights import WEIGHTED_AVERAGES

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

ALL_MINERVA_TASKS = [
    "minerva_math_algebra::olmes",
    "minerva_math_counting_and_probability::olmes",
    "minerva_math_geometry::olmes",
    "minerva_math_intermediate_algebra::olmes",
    "minerva_math_number_theory::olmes",
    "minerva_math_prealgebra::olmes",
    "minerva_math_precalculus::olmes",
]

ALL_GSM_SYMB_TASKS = [
    "gsm_symbolic::olmo3",
    "gsm_symbolic:p1::olmo3",
    "gsm_symbolic:p2::olmo3",
]

ALL_MATH_TASKS = [*ALL_MINERVA_TASKS, "gsm8k::olmo1", "gsm8k::olmes"]


BASIC_SKILLS = [
    "basic_skills_arithmetic",
    "basic_skills_coding",
    "basic_skills_common_knowledge",
    "basic_skills_logical_reasoning",
    "basic_skills_string_operations",
    "basic_skills_pattern",
]

ALL_AGI_EVAL_TASKS = [
    "agi_eval_lsat-ar:1shot::olmes",
    "agi_eval_lsat-lr:1shot::olmes",
    "agi_eval_lsat-rc:1shot::olmes",
    "agi_eval_logiqa-en:1shot::olmes",
    "agi_eval_sat-math:1shot::olmes",
    "agi_eval_sat-en:1shot::olmes",
    "agi_eval_aqua-rat:1shot::olmes",
    "agi_eval_sat-en-without-passage:1shot::olmes",
    "agi_eval_gaokao-english:1shot::olmes",
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
    "codex_humanevalfim_single:temp0.2",
    "codex_humanevalfim_multi:temp0.2",
    "codex_humanevalfim_random:temp0.2",
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

# named groups are things you should able to average; they
# should just contain aliases
ALL_NAMED_GROUPS = {
    "mmlu:rc": [f"{category}:rc::olmes" for category in MMLU_CATEGORIES],
    "mmlu:mc": [f"{category}:mc::olmes" for category in MMLU_CATEGORIES],
    "core:rc": [f"{task}:rc::olmes" for task in ALL_CORE_TASKS],
    "core:mc": [f"{task}:mc::olmes" for task in ALL_CORE_TASKS],
    "arc:rc": [f"{task}:rc::olmes" for task in ARC_TASKS],
    "arc:mc": [f"{task}:mc::olmes" for task in ARC_TASKS],
    "core:rc::full": [f"{task}:rc::olmes:full" for task in ALL_CORE_TASKS],
    "core:mc::full": [f"{task}:mc::olmes:full" for task in ALL_CORE_TASKS],
    "arc:rc::full": [f"{task}:rc::olmes:full" for task in ARC_TASKS],
    "arc:mc::full": [f"{task}:mc::olmes:full" for task in ARC_TASKS],
    "basic:rc": [f"{task}:rc::olmes" for task in BASIC_SKILLS],
    "basic:mc": [f"{task}:mc::olmes" for task in BASIC_SKILLS],
    "mmlu_pro:mc": [f"{category}:mc::none" for category in MMLU_PRO_CATEGORIES],
    "gen": ALL_GEN_TASKS,
    "gen-no-jp": [task for task in ALL_GEN_TASKS if task != "jeopardy::olmes"],
    "minerva": ALL_MINERVA_TASKS,
    "math": ALL_MATH_TASKS,
    "gsm-symb": ALL_GSM_SYMB_TASKS,
    "code": ALL_CODEX_TASKS,
    "agi_eval": ALL_AGI_EVAL_TASKS,
    "starcoder": STARCODER_CODEX_TASKS,
    "starcoder::pass@1": STARCODER_PASS_AT_1_TASKS,
    "code-no-bcb": [task for task in ALL_CODEX_TASKS if "bigcodebench" not in task],
    "fim": FIM_TASKS,
    "mt_mbpp": MULTILINGUAL_MBPP_TASKS
}

for helmet_length in (int(2**i) for i in range(13, 18)):
    ALL_NAMED_GROUPS[f"helmet:{helmet_length // 2 ** 10}k"] = list(
        set(
            task
            for group_name, tasks in HELMET_SUITES.items()
            for task in tasks
            if group_name.endswith(f"__{helmet_length}::suite") and not group_name.startswith("helmet_all")
        )
    )


ALL_DISPLAY_TASKS = {
    "olmo2:paper": [
        r"arc_challenge:mc.*",
        r"hellaswag:mc.*",
        r"winogrande:mc.*",
        r"naturalqs.*",
        r"drop.*",
        r"agieval.*",
        r"^gsm8k::olmes$",
        r"^mmlu:mc$",
        r"^mmlu_pro:mc$",
        r"^agi_eval$",
    ],
    "olmo2:dev:7b": [
        r"arc_challenge:mc.*",
        r"arc_easy:mc.*",
        r"hellaswag:mc.*",
        r"naturalqs.*",
        r"^gsm8k::olmo1$",
        r"^mmlu:mc$",
        r"^core:mc$",
        r"^gen$",
    ],
    "olmo2:dev:1b": [
        r"arc_challenge:rc.*",
        r"arc_easy:rc.*",
        r"hellaswag:rc.*",
        r"^gsm8k::olmo1$",
        r"^mmlu:rc$",
        r"^core:rc$",
    ],
    "olmo3:dev:1b:math:bpb": [
        # Math
        "^minerva.*:bpb::olmes$",
    ],
    "olmo3:dev:1b:code:bpb": [
        # Code
        "^codex_humaneval:3shot:bpb::none",
        "^mbpp:3shot:bpb::none",
        "^mt_mbpp_v2fix.*:bpb$",
    ],
    "olmo3:dev:1b:qa:rc": [
        # Core OLMES
        "^arc:rc::full$",
        "^mmlu:rc$",
        "^csqa:rc::olmes:full",
        "^hellaswag:rc::olmes:full",
        "^winogrande:rc::olmes:full",
        "^socialiqa:rc::olmes:full",
        "^piqa:rc::olmes:full",

        # Gen OLMES
        "^coqa:rc::gen2mc$",
        "^drop:rc::gen2mc$",
        "^jeopardy:rc::gen2mc$",
        "^naturalqs:rc::gen2mc$",
        "^squad:rc::gen2mc$",

        # New OLMo 3
        "^sciq:rc::olmo3",
        "^qasper_yesno:rc::olmes",
        "^basic_skills:rc::olmes",
        "^lab_bench_dbqa$",
        "^lab_bench_protocolqa$",
        "^lambada:rc",
        "^medmcqa:rc::none",
        "^medqa:rc::none",
        "^sciriff_yesno:rc::olmes",
    ],
    "olmo3:dev:1b:micro:rc": [
        # Core OLMES
        "^arc:rc::full$",
        "^mmlu.*:rc::olmes$", # only micro avg
        "^csqa:rc::olmes:full",
        "^hellaswag:rc::olmes:full",
        "^winogrande:rc::olmes:full",
        "^socialiqa:rc::olmes:full",
        "^piqa:rc::olmes:full",

        # Gen OLMES
        "^coqa:rc::gen2mc$",
        "^drop:rc::gen2mc$",
        "^jeopardy:rc::gen2mc$",
        "^naturalqs:rc::gen2mc$",
        "^squad:rc::gen2mc$",

        # New OLMo 3
        "^sciq:rc::olmo3",
        "^qasper_yesno:rc::olmes",
        "^basic_skills:rc::olmes",
        "^lab_bench_dbqa$",
        "^lab_bench_protocolqa$",
        "^lambada:rc",
        "^medmcqa:rc::none",
        "^medqa:rc::none",
        "^sciriff_yesno:rc::olmes",
    ],
    "olmo3:dev:1b:micro:bpb": [
        # Core OLMES
        "^arc_challenge:bpb::olmes:full$",
        "^arc_easy:bpb::olmes:full$",
        "^csqa:bpb::olmes:full",
        "^hellaswag:bpb::olmes:full",
        "^mmlu.*:bpb::olmes$",
        "^winogrande:bpb::olmes:full",
        "^socialiqa:bpb::olmes:full",
        "^piqa:bpb::olmes:full",

        # Gen OLMES
        "^coqa:bpb::gen2mc$",
        "^drop:bpb::gen2mc$",
        "^jeopardy:bpb::gen2mc$",
        "^naturalqs:bpb::gen2mc$",
        "^squad:bpb::gen2mc$",

        # Math
        "^minerva.*:bpb::olmes$",

        # Code
        "^codex_humaneval:3shot:bpb::none",
        "^mbpp:3shot:bpb::none",
        "^mt_mbpp_v2fix.*:bpb$",

        # New OLMo 3
        "^sciq:bpb::olmo3",
        "^qasper_yesno:bpb::olmes",
        "^basic_skills.*:bpb::olmes",
        "^lab_bench_dbqa:bpb$",
        "^lab_bench_protocolqa:bpb$",
        "^lambada:bpb",
        "^medmcqa:bpb::none",
        "^medqa:bpb::none",
        "^sciriff_yesno:bpb::olmes",
        "ultrachat_masked_ppl",
        "wildchat_masked_ppl",
    ],
    "olmo3:dev:1b:macro:bpb": [
        # Core OLMES
        "^arc:bpb::full$",
        "^mmlu:bpb$",
        "^csqa:bpb::olmes:full",
        "^hellaswag:bpb::olmes:full",
        "^winogrande:bpb::olmes:full",
        "^socialiqa:bpb::olmes:full",
        "^piqa:bpb::olmes:full",

        # Gen OLMES
        "^coqa:bpb::gen2mc$",
        "^drop:bpb::gen2mc$",
        "^jeopardy:bpb::gen2mc$",
        "^naturalqs:bpb::gen2mc$",
        "^squad:bpb::gen2mc$",

        # Math
        "^minerva:bpb::olmes$",

        # Code
        "^codex_humaneval:3shot:bpb::none",
        "^mbpp:3shot:bpb::none",
        "^mt_mbpp_v2fix$",

        # New OLMo 3
        "^sciq:bpb::olmo3",
        "^qasper_yesno:bpb::olmes",
        "^basic_skills:bpb::olmes",
        "^lab_bench_dbqa:bpb$",
        "^lab_bench_protocolqa:bpb$",
        "^lambada:bpb",
        "^medmcqa:bpb::none",
        "^medqa:bpb::none",
        "^sciriff_yesno:bpb::olmes",
        "ultrachat_masked_ppl",
        "wildchat_masked_ppl",
    ],
    "olmo3:dev:7b:macro:math": [
        # Math
        "gsm8k::olmes",
        "^gsm-symb$",
        "^minerva$",
    ],
    "olmo3:dev:7b:macro:code": [
        # Code
        "codex_humaneval:3shot::olmo3",
        "cruxeval_input:pass@5$",
        "cruxeval_output:pass@5$",
        "mbpp:3shot::olmo3",
    ],
    "olmo3:dev:7b:macro:qa": [
        # Core OLMES
        "^arc:mc::full$",
        "^mmlu:mc$",
        "csqa:mc::olmes:full",
        "hellaswag:rc::olmes:full",
        "piqa:mc::olmes:full",
        "socialiqa:mc::olmes:full",
        "winogrande:rc::olmes:full",

        # Gen OLMES
        "drop::olmes",
        "jeopardy::olmes",
        "naturalqs::olmes",
        "squad::olmes",
        "coqa::olmes",

        # Gen2MC OLMES
        "coqa:mc::gen2mc",
        "drop:mc::gen2mc",
        "jeopardy:mc::gen2mc",
        "naturalqs:mc::gen2mc",
        "squad:mc::gen2mc",

        # New OLMo 3
        "^basic:rc$",
        "lab_bench_dbqa:mc", # swap to mc?
        "lab_bench_protocolqa:mc", # swap to mc?
        "lambada$",
        "medmcqa:mc::none",
        "medqa_en:mc::none",
        "sciq:mc::olmo3",
    ],
    "olmo3:dev:7b:micro:qa": [
        # Core OLMES
        "^mmlu.*:mc.*$",
        "arc_challenge:mc::olmes:full",
        "arc_easy:mc::olmes:full",
        "csqa:mc::olmes:full",
        "hellaswag:rc::olmes:full",
        "piqa:mc::olmes:full",
        "socialiqa:mc::olmes:full",
        "winogrande:rc::olmes:full",

        # Gen OLMES
        "coqa::olmes",
        "drop::olmes",
        "jeopardy::olmes",
        "naturalqs::olmes",
        "squad::olmes",

        # Gen2MC OLMES
        "coqa:mc::gen2mc",
        "drop:mc::gen2mc",
        "jeopardy:mc::gen2mc",
        "naturalqs:mc::gen2mc",
        "squad:mc::gen2mc",

        # New OLMo 3
        "basic_skills_[^:]*::olmes",
        "lab_bench_dbqa:mc", # swap to mc?
        "lab_bench_protocolqa:mc", # swap to mc?
        "lambada$",
        "medmcqa:mc::none",
        "medqa_en:mc::none",
        "sciq:mc::olmo3",
    ],
    "olmo3:dev:7b:micro:math": [
        # Math
        "^minerva[^:]*::olmes",
        "gsm_symbolic::olmo3",
        "gsm_symbolic:p1::olmo3",
        "gsm_symbolic:p2::olmo3",
        "gsm8k::olmes",
    ],
    "olmo3:dev:7b:micro:code": [
        # Code
        "codex_humaneval:3shot::olmo3",
        "cruxeval_input:pass@5$",
        "cruxeval_output:pass@5$",
        "mbpp:3shot::olmo3",
    ],
    "helmet:8k": [r"^helmet:8k$"],
    "helmet:16k": [r"^helmet:16k$"],
    "helmet:32k": [r"^helmet:32k$"],
    "helmet:64k": [r"^helmet:64k$"],
    "helmet:128k": [r"^helmet:128k$"],
}

ALL_DISPLAY_TASKS.update({
    "olmo3:dev:1b:main": [
        # "^olmo3:dev:1b:macro:bpb$",
        "^olmo3:dev:1b:macro:bpb:w_avg$",
        "^olmo3:dev:1b:math:bpb$",
        "^olmo3:dev:1b:code:bpb$",
        # "^olmo3:dev:1b:macro:rc$",
        "^olmo3:dev:1b:macro:rc:w_avg$",
        "^arc:(rc|bpb)::olmes:full$",
        "^hellaswag:rc::olmes:full",
        "basic:rc",
        "^mt_mbpp_v2fix$",
        "^mmlu:(rc|bpb)$",
        "core:rc"
        "codex_humaneval:3shot:bpb::none",
        "mbpp:3shot:bpb::none",
        "^minerva$",
    ],
    "olmo3:dev:7b:macro": \
        ALL_DISPLAY_TASKS["olmo3:dev:7b:macro:math"] + \
        ALL_DISPLAY_TASKS["olmo3:dev:7b:macro:code"] + \
        ALL_DISPLAY_TASKS["olmo3:dev:7b:macro:qa"],
    "olmo3:dev:7b:micro": \
        ALL_DISPLAY_TASKS["olmo3:dev:7b:micro:math"] + \
        ALL_DISPLAY_TASKS["olmo3:dev:7b:micro:code"] + \
        ALL_DISPLAY_TASKS["olmo3:dev:7b:micro:qa"],
    "olmo3:dev:1b:macro:rc": ALL_DISPLAY_TASKS["olmo3:dev:1b:qa:rc"],
    "olmo3:dev:7b:math": ALL_DISPLAY_TASKS["olmo3:dev:7b:macro:math"],
    "olmo3:dev:7b:code": ALL_DISPLAY_TASKS["olmo3:dev:7b:macro:code"],
    "olmo3:dev:7b:qa": ALL_DISPLAY_TASKS["olmo3:dev:7b:macro:qa"],
    "olmo3:dev:7b:main": [
        "^olmo3:dev:7b:macro:w_avg$",
        "^olmo3:dev:7b:qa$",
        "^olmo3:dev:7b:math$",
        "^olmo3:dev:7b:code$",
        "^arc:mc::full$",
        "^mmlu:mc$",
        "^codex_humaneval:3shot::olmo3$",
        "^mbpp:3shot::olmo3$",
        "^gsm8k::olmes$",
        "^gsm-symb$",
        "^minerva$",
        "^basic:rc$",
    ],
})


SHORT_NAMES = {
    r"::olmes$": "",
    r"::olmes:full$": "",
    r"^gsm8k::olmo1$": "GSM*",
    r"^naturalqs": "NQ",
    r"^(arc\_\w)\w+": r"\1",
    r"^hellaswag": "HSwag",
    r"^winogrande": "WinoG",
    r"^codex_humaneval": "humaneval",
    r"::olmo3$": "",
    r"::none$": "",
    r":3shot": "",
    r"::full$": "",
}


OE_EVAL_GIT_URL = "git@github.com:allenai/oe-eval-internal.git"
OE_EVAL_COMMIT_HASH = None
OE_EVAL_LAUNCH_COMMAND = "oe_eval/launch.py"
BEAKER_PRIORITIES = ["low", "normal", "high", "urgent"]

# use these tasks to launch oe-eval jobs
ALL_EVAL_TASKS = {
    "olmo3:dev:1b:vllm": [
        "arc_challenge:rc::olmes:full",
        "arc_easy:rc::olmes:full",
        "basic_skills:rc::olmes",
        "codex_humaneval:3shot:bpb::none",
        "coqa:rc::gen2mc",
        "csqa:rc::olmes:full",
        "drop:rc::gen2mc",
        "hellaswag:rc::olmes:full",
        "jeopardy:rc::gen2mc",
        "lab_bench_dbqa",
        "lab_bench_protocolqa",
        "lambada",
        "mbpp:3shot:bpb::none",
        "medmcqa:rc::none",
        "medqa_en:rc::none",
        "minerva_math::olmes",
        "mmlu:rc::olmes",
        "mt_mbpp",
        "naturalqs:rc::gen2mc",
        "piqa:rc::olmes:full",
        "qasper_yesno:rc::olmes",
        "sciq:rc::olmo3",
        "sciriff_yesno:rc::olmes",
        "socialiqa:rc::olmes:full",
        "squad:rc::gen2mc",
        "winogrande:rc::olmes:full",
    ],
    "olmo3:dev:1b:hf": [
        "ultrachat_masked_ppl",
        "wildchat_masked_ppl",
    ]
}
