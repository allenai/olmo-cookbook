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

OLMO_CORE_COMMIT_HASH = "c149b73b4b02675140af67e1e32bac9d9bf116f8"
OLMO_CORE_UNSHARD_CONVERT_SCRIPT = "src/examples/huggingface/convert_checkpoint_to_hf.py"

DEFAULT_OLMOE_TOKENIZER = "allenai/eleuther-ai-gpt-neox-20b-pii-special"
DEFAULT_OLMO2_TOKENIZER = "allenai/dolma2-tokenizer"
DEFAULT_OLMO_CORE_TOKENIZER = "allenai/OLMo-2-1124-7B"

BEAKER_DEFAULT_WORKSPACE = "ai2/oe-data"
BEAKER_DEFAULT_BUDGET = "ai2/oe-data"
BEAKER_DEFAULT_PRIORITY = "normal"

OLMO_TYPES = ["olmoe", "olmo2", "olmo-core"]

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

ALL_MATH_TASKS = [
    *ALL_MINERVA_TASKS,
    "gsm8k::olmo1",
    "gsm8k::olmes"
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

SINGLE_FIM_TASK = {"task_name": "codex_humanevalfim_single:temp0.2", "metric_kwargs": {"n_exe_workers": 20}}
MULTI_FIM_TASK = {"task_name": "codex_humanevalfim_multi:temp0.2", "metric_kwargs": {"n_exe_workers": 20}}
RANDOM_FIM_TASK = {"task_name": "codex_humanevalfim_random:temp0.2", "metric_kwargs": {"n_exe_workers": 20}}
INFILLING_STARCODER = [
    json.dumps({**SINGLE_FIM_TASK, **STARCODER_INFILLING}),
    json.dumps({**MULTI_FIM_TASK, **STARCODER_INFILLING}),
    json.dumps({**RANDOM_FIM_TASK, **STARCODER_INFILLING}),
]
INFILLING_SANTACODER = [
    json.dumps({**SINGLE_FIM_TASK, **SANTACODER_INFILLING}),
    json.dumps({**MULTI_FIM_TASK, **SANTACODER_INFILLING}),
    json.dumps({**RANDOM_FIM_TASK, **SANTACODER_INFILLING}),
]
INFILLING_DEEPSEEK_CODER = [
    json.dumps({**SINGLE_FIM_TASK, **DEEPSEEK_CODER_INFILLING}),
    json.dumps({**MULTI_FIM_TASK, **DEEPSEEK_CODER_INFILLING}),
    json.dumps({**RANDOM_FIM_TASK, **DEEPSEEK_CODER_INFILLING}),
]
FIM_TOKEN_TYPES = {
    "santacoder": INFILLING_SANTACODER,
    "starcoder": INFILLING_STARCODER,
    "deepseek-coder": INFILLING_DEEPSEEK_CODER,
}

# TODO: clean up such that tasks kwargs are separate from task aliases.
FIM_TASKS = {f"fim-{name}": tasks for name, tasks in FIM_TOKEN_TYPES.items()}


# named groups are things you should able to average; they
# should just contain aliases
ALL_NAMED_GROUPS = {
    "mmlu:rc": [f"{category}:rc::olmes" for category in MMLU_CATEGORIES],
    "mmlu:mc": [f"{category}:mc::olmes" for category in MMLU_CATEGORIES],
    "core:rc": [f"{task}:rc::olmes" for task in ALL_CORE_TASKS],
    "core:mc": [f"{task}:mc::olmes" for task in ALL_CORE_TASKS],
    "gen": ALL_GEN_TASKS,
    "gen-no-jp": [task for task in ALL_GEN_TASKS if task != "jeopardy::olmes"],
    "minerva": ALL_MINERVA_TASKS,
    "math": ALL_MATH_TASKS,
    "code": ALL_CODEX_TASKS,
    "starcoder": STARCODER_CODEX_TASKS,
    "starcoder::pass@1": STARCODER_PASS_AT_1_TASKS,
    "code-no-bcb": [task for task in ALL_CODEX_TASKS if "bigcodebench" not in task],
}

OE_EVAL_GIT_URL = "git@github.com:allenai/oe-eval-internal.git"
OE_EVAL_COMMIT_HASH = None
OE_EVAL_LAUNCH_COMMAND = "oe_eval/launch.py"
BEAKER_PRIORITIES = ["low", "normal", "high", "urgent"]
