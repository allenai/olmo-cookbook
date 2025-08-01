# a script that extracts a list of item ids and task names from a directory of decon reports

The goal is to build a list of tuples (doc_id int, task_names list of strings) that each represent a benchmark test item that has been marked as contaminated. 

## Input

we have a dir of decon reports with a directory structure as follows within which are several `*.jsonl` files containing reports
```
├── ai2-llm-pretraining-data-sources-cc_all_dressed-all_dressed_v3-weborganizer_ft-dclm_plus2_vigintiles-data
│   ├── adult_content
│   │   ├── vigintile_0007
│   │   ├── vigintile_0008
│   │   ├── vigintile_0009
│   │   ├── vigintile_0010
│   │   ├── vigintile_0011
│   │   ├── vigintile_0012
│   │   ├── vigintile_0013
│   │   ├── vigintile_0014
│   │   ├── vigintile_0015
│   │   ├── vigintile_0016
│   │   ├── vigintile_0017
│   │   ├── vigintile_0018
│   │   └── vigintile_0020
│   ├── art_and_design
│   │   ├── vigintile_0005
│   │   ├── vigintile_0006
│   │   ├── vigintile_0007
│   │   ├── vigintile_0008
│   │   ├── vigintile_0009
│   │   ├── vigintile_0010
│   │   ├── vigintile_0011
│   │   ├── vigintile_0012
│   │   ├── vigintile_0013
│   │   ├── vigintile_0014
│   │   ├── vigintile_0015
│   │   ├── vigintile_0016
│   │   ├── vigintile_0017
│   │   ├── vigintile_0018
│   │   └── vigintile_0020
│   ├── crime_and_law
│   │   ├── vigintile_0002
│   │   ├── vigintile_0003
│   │   ├── vigintile_0004
│   │   ├── vigintile_0005
│   │   ├── vigintile_0006
│   │   ├── vigintile_0007
│   │   ├── vigintile_0008
│   │   ├── vigintile_0009
│   │   ├── vigintile_0010
│   │   ├── vigintile_0011
│   │   ├── vigintile_0012
│   │   ├── vigintile_0013
│   │   ├── vigintile_0014
│   │   ├── vigintile_0015
│   │   ├── vigintile_0016
│   │   ├── vigintile_0017
│   │   ├── vigintile_0018
│   │   └── vigintile_0020
│   ├── education_and_jobs
│   │   ├── vigintile_0003
│   │   ├── vigintile_0004
│   │   ├── vigintile_0005
│   │   ├── vigintile_0006
│   │   ├── vigintile_0007
│   │   ├── vigintile_0008
│   │   ├── vigintile_0009
│   │   ├── vigintile_0010
│   │   ├── vigintile_0011
│   │   ├── vigintile_0012
│   │   ├── vigintile_0013
│   │   ├── vigintile_0014
│   │   ├── vigintile_0015
│   │   ├── vigintile_0016
│   │   ├── vigintile_0017
│   │   ├── vigintile_0018
│   │   └── vigintile_0020
│   ├── electronics_and_hardware
│   │   ├── vigintile_0005
│   │   ├── vigintile_0006
│   │   ├── vigintile_0007
│   │   ├── vigintile_0008
│   │   ├── vigintile_0009
│   │   ├── vigintile_0010
│   │   ├── vigintile_0011
│   │   ├── vigintile_0012
│   │   ├── vigintile_0013
│   │   ├── vigintile_0014
│   │   ├── vigintile_0015
│   │   ├── vigintile_0016
│   │   ├── vigintile_0017
│   │   ├── vigintile_0018
│   │   └── vigintile_0020
│   ├── entertainment
│   │   ├── vigintile_0005
│   │   ├── vigintile_0006
│   │   ├── vigintile_0007
│   │   ├── vigintile_0008
│   │   ├── vigintile_0009
│   │   ├── vigintile_0010
│   │   ├── vigintile_0011
│   │   ├── vigintile_0012
│   │   ├── vigintile_0013
│   │   ├── vigintile_0014
│   │   ├── vigintile_0015
│   │   ├── vigintile_0016
│   │   ├── vigintile_0017
│   │   ├── vigintile_0018
│   │   └── vigintile_0020
│   ├── fashion_and_beauty
│   │   ├── vigintile_0006
│   │   ├── vigintile_0007
│   │   ├── vigintile_0008
│   │   ├── vigintile_0009
│   │   ├── vigintile_0010
│   │   ├── vigintile_0011
│   │   ├── vigintile_0012
│   │   ├── vigintile_0013
│   │   ├── vigintile_0014
│   │   ├── vigintile_0015
│   │   ├── vigintile_0016
│   │   ├── vigintile_0017
│   │   ├── vigintile_0018
│   │   └── vigintile_0020
│   ├── finance_and_business
│   │   ├── vigintile_0002
│   │   ├── vigintile_0003
│   │   ├── vigintile_0004
│   │   ├── vigintile_0005
│   │   ├── vigintile_0006
│   │   ├── vigintile_0007
│   │   ├── vigintile_0008
│   │   ├── vigintile_0009
│   │   ├── vigintile_0010
│   │   ├── vigintile_0011
│   │   ├── vigintile_0012
│   │   ├── vigintile_0013
│   │   ├── vigintile_0014
│   │   ├── vigintile_0015
│   │   ├── vigintile_0016
│   │   ├── vigintile_0017
│   │   ├── vigintile_0018
│   │   └── vigintile_0020
│   ├── food_and_dining
│   │   ├── vigintile_0004
│   │   ├── vigintile_0005
│   │   ├── vigintile_0006
│   │   ├── vigintile_0007
│   │   ├── vigintile_0008
│   │   ├── vigintile_0009
│   │   ├── vigintile_0010
│   │   ├── vigintile_0011
│   │   ├── vigintile_0012
│   │   ├── vigintile_0013
│   │   ├── vigintile_0014
│   │   ├── vigintile_0015
│   │   ├── vigintile_0016
│   │   ├── vigintile_0017
│   │   ├── vigintile_0018
│   │   └── vigintile_0020
│   ├── games
│   │   ├── vigintile_0004
│   │   ├── vigintile_0005
│   │   ├── vigintile_0006
│   │   ├── vigintile_0007
│   │   ├── vigintile_0008
│   │   ├── vigintile_0009
│   │   ├── vigintile_0010
│   │   ├── vigintile_0011
│   │   ├── vigintile_0012
│   │   ├── vigintile_0013
│   │   ├── vigintile_0014
│   │   ├── vigintile_0015
│   │   ├── vigintile_0016
│   │   ├── vigintile_0017
│   │   ├── vigintile_0018
│   │   └── vigintile_0020
│   ├── health
│   │   ├── vigintile_0002
│   │   ├── vigintile_0003
│   │   ├── vigintile_0004
│   │   ├── vigintile_0005
│   │   ├── vigintile_0006
│   │   ├── vigintile_0007
│   │   ├── vigintile_0008
│   │   ├── vigintile_0009
│   │   ├── vigintile_0010
│   │   ├── vigintile_0011
│   │   ├── vigintile_0012
│   │   ├── vigintile_0013
│   │   ├── vigintile_0014
│   │   ├── vigintile_0015
│   │   ├── vigintile_0016
│   │   ├── vigintile_0017
│   │   ├── vigintile_0018
│   │   └── vigintile_0020
│   ├── history_and_geography
│   │   ├── vigintile_0004
│   │   ├── vigintile_0005
│   │   ├── vigintile_0006
│   │   ├── vigintile_0007
│   │   ├── vigintile_0008
│   │   ├── vigintile_0009
│   │   ├── vigintile_0010
│   │   ├── vigintile_0011
│   │   ├── vigintile_0012
│   │   ├── vigintile_0013
│   │   ├── vigintile_0014
│   │   ├── vigintile_0015
│   │   ├── vigintile_0016
│   │   ├── vigintile_0017
│   │   ├── vigintile_0018
│   │   └── vigintile_0020
│   ├── home_and_hobbies
│   │   ├── vigintile_0005
│   │   ├── vigintile_0006
│   │   ├── vigintile_0007
│   │   ├── vigintile_0008
│   │   ├── vigintile_0009
│   │   ├── vigintile_0010
│   │   ├── vigintile_0011
│   │   ├── vigintile_0012
│   │   ├── vigintile_0013
│   │   ├── vigintile_0014
│   │   ├── vigintile_0015
│   │   ├── vigintile_0016
│   │   ├── vigintile_0017
│   │   ├── vigintile_0018
│   │   └── vigintile_0020
│   ├── industrial
│   │   ├── vigintile_0003
│   │   ├── vigintile_0004
│   │   ├── vigintile_0005
│   │   ├── vigintile_0008
│   │   ├── vigintile_0009
│   │   ├── vigintile_0010
│   │   ├── vigintile_0011
│   │   ├── vigintile_0012
│   │   ├── vigintile_0013
│   │   ├── vigintile_0014
│   │   ├── vigintile_0015
│   │   ├── vigintile_0016
│   │   ├── vigintile_0017
│   │   ├── vigintile_0018
│   │   └── vigintile_0020
│   ├── literature
│   │   ├── vigintile_0004
│   │   ├── vigintile_0005
│   │   ├── vigintile_0006
│   │   ├── vigintile_0007
│   │   ├── vigintile_0008
│   │   ├── vigintile_0009
│   │   ├── vigintile_0010
│   │   ├── vigintile_0011
│   │   ├── vigintile_0012
│   │   ├── vigintile_0013
│   │   ├── vigintile_0014
│   │   ├── vigintile_0015
│   │   ├── vigintile_0016
│   │   ├── vigintile_0017
│   │   ├── vigintile_0018
│   │   └── vigintile_0020
│   ├── politics
│   │   ├── vigintile_0002
│   │   ├── vigintile_0003
│   │   ├── vigintile_0004
│   │   ├── vigintile_0005
│   │   ├── vigintile_0006
│   │   ├── vigintile_0007
│   │   ├── vigintile_0008
│   │   ├── vigintile_0009
│   │   ├── vigintile_0010
│   │   ├── vigintile_0011
│   │   ├── vigintile_0012
│   │   ├── vigintile_0013
│   │   ├── vigintile_0014
│   │   ├── vigintile_0015
│   │   ├── vigintile_0016
│   │   ├── vigintile_0017
│   │   ├── vigintile_0018
│   │   └── vigintile_0020
│   ├── religion
│   │   ├── vigintile_0003
│   │   ├── vigintile_0004
│   │   ├── vigintile_0005
│   │   ├── vigintile_0006
│   │   ├── vigintile_0007
│   │   ├── vigintile_0008
│   │   ├── vigintile_0009
│   │   ├── vigintile_0010
│   │   ├── vigintile_0011
│   │   ├── vigintile_0012
│   │   ├── vigintile_0013
│   │   ├── vigintile_0014
│   │   ├── vigintile_0015
│   │   ├── vigintile_0016
│   │   ├── vigintile_0017
│   │   ├── vigintile_0018
│   │   └── vigintile_0020
│   ├── science_math_and_technology
│   │   ├── vigintile_0002
│   │   ├── vigintile_0003
│   │   ├── vigintile_0004
│   │   ├── vigintile_0005
│   │   ├── vigintile_0006
│   │   ├── vigintile_0007
│   │   ├── vigintile_0008
│   │   ├── vigintile_0009
│   │   ├── vigintile_0010
│   │   ├── vigintile_0011
│   │   ├── vigintile_0012
│   │   ├── vigintile_0013
│   │   ├── vigintile_0014
│   │   ├── vigintile_0015
│   │   ├── vigintile_0016
│   │   ├── vigintile_0017
│   │   ├── vigintile_0018
│   │   └── vigintile_0020
│   ├── social_life
│   │   ├── vigintile_0004
│   │   ├── vigintile_0005
│   │   ├── vigintile_0006
│   │   ├── vigintile_0007
│   │   ├── vigintile_0008
│   │   ├── vigintile_0009
│   │   ├── vigintile_0010
│   │   ├── vigintile_0011
│   │   ├── vigintile_0012
│   │   ├── vigintile_0013
│   │   ├── vigintile_0014
│   │   ├── vigintile_0015
│   │   ├── vigintile_0016
│   │   ├── vigintile_0017
│   │   ├── vigintile_0018
│   │   └── vigintile_0020
│   ├── software
│   │   ├── vigintile_0006
│   │   ├── vigintile_0007
│   │   ├── vigintile_0008
│   │   ├── vigintile_0009
│   │   ├── vigintile_0010
│   │   ├── vigintile_0011
│   │   ├── vigintile_0012
│   │   ├── vigintile_0013
│   │   ├── vigintile_0014
│   │   ├── vigintile_0015
│   │   ├── vigintile_0016
│   │   ├── vigintile_0017
│   │   ├── vigintile_0018
│   │   └── vigintile_0020
│   ├── software_development
│   │   ├── vigintile_0004
│   │   ├── vigintile_0005
│   │   ├── vigintile_0006
│   │   ├── vigintile_0007
│   │   ├── vigintile_0008
│   │   ├── vigintile_0009
│   │   ├── vigintile_0010
│   │   ├── vigintile_0011
│   │   ├── vigintile_0012
│   │   ├── vigintile_0013
│   │   ├── vigintile_0014
│   │   ├── vigintile_0015
│   │   ├── vigintile_0016
│   │   ├── vigintile_0017
│   │   ├── vigintile_0018
│   │   └── vigintile_0020
│   ├── sports_and_fitness
│   │   ├── vigintile_0003
│   │   ├── vigintile_0004
│   │   ├── vigintile_0005
│   │   ├── vigintile_0006
│   │   ├── vigintile_0007
│   │   ├── vigintile_0008
│   │   ├── vigintile_0009
│   │   ├── vigintile_0010
│   │   ├── vigintile_0011
│   │   ├── vigintile_0012
│   │   ├── vigintile_0013
│   │   ├── vigintile_0014
│   │   ├── vigintile_0015
│   │   ├── vigintile_0016
│   │   ├── vigintile_0017
│   │   ├── vigintile_0018
│   │   └── vigintile_0020
│   ├── transportation
│   │   ├── vigintile_0005
│   │   ├── vigintile_0006
│   │   ├── vigintile_0007
│   │   ├── vigintile_0008
│   │   ├── vigintile_0009
│   │   ├── vigintile_0010
│   │   ├── vigintile_0011
│   │   ├── vigintile_0012
│   │   ├── vigintile_0013
│   │   ├── vigintile_0014
│   │   ├── vigintile_0015
│   │   ├── vigintile_0016
│   │   ├── vigintile_0017
│   │   ├── vigintile_0018
│   │   └── vigintile_0020
│   └── travel_and_tourism
│       ├── vigintile_0003
│       ├── vigintile_0004
│       ├── vigintile_0005
│       ├── vigintile_0006
│       ├── vigintile_0007
│       ├── vigintile_0008
│       ├── vigintile_0009
│       ├── vigintile_0010
│       ├── vigintile_0011
│       ├── vigintile_0012
│       ├── vigintile_0013
│       ├── vigintile_0014
│       ├── vigintile_0015
│       ├── vigintile_0016
│       ├── vigintile_0017
│       ├── vigintile_0018
│       └── vigintile_0020
├── ai2-llm-pretraining-data-sources-cc_all_dressed-all_dressed_v3-weborganizer_ft-dclm_plus2_vigintiles-vigintile_0018_subset
│   ├── adult_content
│   ├── art_and_design
│   ├── crime_and_law
│   ├── education_and_jobs
│   ├── electronics_and_hardware
│   ├── entertainment
│   ├── fashion_and_beauty
│   ├── finance_and_business
│   ├── food_and_dining
│   ├── games
│   ├── health
│   ├── history_and_geography
│   ├── home_and_hobbies
│   ├── industrial
│   ├── literature
│   ├── politics
│   ├── religion
│   ├── science_math_and_technology
│   ├── social_life
│   ├── software
│   ├── software_development
│   ├── sports_and_fitness
│   ├── transportation
│   └── travel_and_tourism
├── ai2-llm-pretraining-data-sources-cc_all_dressed-all_dressed_v3-weborganizer_ft-dclm_plus2_vigintiles-vigintile_0020_subset
│   ├── adult_content
│   ├── art_and_design
│   ├── crime_and_law
│   ├── education_and_jobs
│   ├── electronics_and_hardware
│   ├── entertainment
│   ├── fashion_and_beauty
│   ├── finance_and_business
│   ├── food_and_dining
│   ├── games
│   ├── health
│   ├── history_and_geography
│   ├── home_and_hobbies
│   ├── industrial
│   ├── literature
│   ├── politics
│   ├── religion
│   ├── science_math_and_technology
│   ├── social_life
│   ├── software
│   ├── software_development
│   ├── sports_and_fitness
│   ├── transportation
│   └── travel_and_tourism
├── ai2-llm-pretraining-data-sources-finemath-finemath-3plus
├── ai2-llm-pretraining-data-sources-midtraining-reasoning-math-meta-reasoning-documents
├── ai2-llm-pretraining-data-sources-stack-edu-documents
│   ├── C
│   ├── CSharp
│   ├── Cpp
│   ├── Go
│   ├── Java
│   ├── JavaScript
│   ├── Markdown
│   ├── PHP
│   ├── Python
│   ├── Ruby
│   ├── Rust
│   ├── SQL
│   ├── Shell
│   ├── Swift
│   └── TypeScript
├── ai2-llm-pretraining-densesub-lowthresh-4omini-rewrite-documents
├── ai2-llm-pretraining-midtraining-reasoning-tinyMath-PoT-processed-data-processed
├── ai2-llm-pretraining-openmath-reasoning-rewrite-full-thoughts
├── ai2-llm-pretraining-stack-edu-fim-documents
│   └── fim_50pct_psm_50pct
│       ├── C
│       ├── CSharp
│       ├── Cpp
│       ├── Go
│       ├── Java
│       ├── JavaScript
│       ├── Markdown
│       ├── PHP
│       ├── Python
│       ├── Ruby
│       ├── Rust
│       ├── SQL
│       ├── Shell
│       ├── Swift
│       └── TypeScript
├── ai2-llm-pretraining-thinking-data-big-reasoning-traces
├── ai2-llm-reddit-rewrites-densesub-highthresh-microanneal-4omini-rewrite-documents
├── ai2-llm-thinking-data-qwq-traces
├── ai2-midtraining-reasoning-flat-dolmino-math
├── megamath_web_pro_max_llama_rewrites
├── midtraining-reasoning-code-meta-reasoning-documents
├── midtraining-reasoning-openmathreasoning-teacher-student-lecture-documents
└── s2pdf_dedupe_minhash_v1_with_no_pii_basic_quality_datadelve_norefs_mdtables_v2_denylisted
    ├── art_design
    │   └── step_final
    ├── crime_law
    │   └── step_final
    ├── education_jobs
    │   └── step_final
    ├── entertainment
    │   └── step_final
    ├── fashion_beauty
    │   └── step_final
    ├── finance_business
    │   └── step_final
    ├── food_dining
    │   └── step_final
    ├── games
    │   └── step_final
    └── health
        └── step_final
```

The reports contain lines like
```
{
  "answer_idf_overlap": 1,
  "answer_overlap_ratio": 1,
  "cluster_token_length": 18,
  "contamination_end_idx": 169,
  "contamination_score": 1,
  "contamination_start_idx": 152,
  "eval_dataset": "sciq_train-1",
  "eval_line": 1272,
  "eval_overlap_text": "what do you call the separation of ions that occurs when a solid ionic compound dissolves",
  "eval_token_length": 18,
  "eval_unique_ngrams": 14,
  "idf_overlap": 1,
  "length_adjusted_question_threshold": 1,
  "matched_answer_tokens": [
    " dissoci",
    "ation"
  ],
  "method": "simple",
  "ngram_match_cnt": 14,
  "overlap_ratio": 1,
  "token_length_delta": 0,
  "training_file": "train-00009-of-00036.jsonl",
  "training_line": 23654,
  "training_overlap_text": "...  that you learned when writing chemical formulas of ionic compounds the subscripts for the ions in the chemical formulas become the coefficients of the respective ions on the product side of the equation shown below are dissociation equations for nacl cano 3 2 and nh 4 3 po 4 q 【 what do you call the separation of ions that occurs when a solid ionic compound dissolves】  choices deformation decomposition dissolution dissociation a the answer is • chosen response dissociation • rejected response sure i can help you with that the correct answer is b dissociation explanation dissociation is the process by which a solid ionic compound breaks apart into its constituent ions when it dissolves in water this is ...",
  "oe-eval-task": null,
  "oe-eval-doc-id": 1146970
}
```

We want to build a file like the following that has tuples for each report line with `oe-eval-doc-id` as the first value and `oe-eval-task` as the second value. If `oe-eval-task` is null just skip that line:
```
(10, ["winogrande:mc"])
(42, ["gsm8k:cot"])
(20, ["arc_challenge:mc", "arc_easy:mc"])
```