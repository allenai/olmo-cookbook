"""
This a dump from the OLMo ladder wandb logs. Once the ladder code is migrated into a stand-alone library, we will move away
from needing this.

data_by_name_tokens = get_step1_data_by_name(configs, 'arc_easy_test_5shot', y_metric=y_metric, moving_avg=1)
"""

DATA_BY_NAME_LADDER = {
    "190M": {
        "ns": [190354176, 190354176, 190354176, 190354176, 190354176],
        "ds": [1908932608, 3812622336, 7619477504, 19041091584, 38076416000],
        "xs": [
            1.3307431936264038,
            1.2346782684326172,
            1.1148797273635864,
            1.0151785612106323,
            1.0056660175323486,
        ],
        "ls": ["0.5xC", "1xC", "2xC", "5xC", "10xC"],
        "fs": [
            3.633445588546093e18,
            7.256911925269758e18,
            1.4502846673534845e19,
            3.624264676869459e19,
            7.247431636038451e19,
        ],
        "mode": "train",
    },
    "370M": {
        "ns": [371262464, 371262464, 371262464, 371262464, 371262464],
        "ds": [3720609792, 7433355264, 14858846208, 37134532608, 74260414464],
        "xs": [
            1.1390928030014038,
            1.0250357389450073,
            0.9110808968544006,
            0.8791245222091675,
            0.8505839109420776,
        ],
        "ls": ["0.5xC", "1xC", "2xC", "5xC", "10xC"],
        "fs": [
            1.2813493428339868e19,
            2.5599902744592777e19,
            5.11727213770986e19,
            1.2788846886340736e20,
            2.5574714520351906e20,
        ],
        "mode": "train",
    },
    "760M": {
        "ns": [758220288, 758220288, 758220288, 758220288, 758220288],
        "ds": [7595622400, 15178137600, 30343168000, 75835637760, 151658168320],
        "xs": [
            1.019742727279663,
            0.9084560871124268,
            0.8210516571998596,
            0.7791905999183655,
            0.7520826458930969,
        ],
        "ls": ["0.5xC", "1xC", "2xC", "5xC", "10xC"],
        "fs": [
            4.841331201744568e19,
            9.6743080787234e19,
            1.9340261832681062e20,
            4.83364522296009e20,
            9.666455013443607e20,
        ],
        "mode": "train",
    },
    "1.3B": {
        "ns": [1279395840, 1279395840, 1279395840, 1279395840, 1279395840],
        "ds": [12810977280, 25604653056, 51192004608, 127955632128, 255895535616],
        "xs": [
            0.8886174559593201,
            0.8153517842292786,
            0.7656160593032837,
            0.7194880247116089,
            0.6943516731262207,
        ],
        "ls": ["0.5xC", "1xC", "2xC", "5xC", "10xC"],
        "fs": [
            1.295070835148587e20,
            2.5883926489114608e20,
            5.175036276437208e20,
            1.2935126160958607e21,
            2.5868662302475656e21,
        ],
        "mode": "train",
    },
    "3.2B": {
        "ns": [3169537280, 3169537280, 3169537280, 3169537280, 3169537280],
        "ds": [31727812608, 63424167936, 126813732864, 316985573376, 633939689472],
        "xs": [
            0.7704519629478455,
            0.7018752694129944,
            0.6705916523933411,
            0.6384084820747375,
            0.6099818348884583,
        ],
        "ls": ["0.5xC", "1xC", "2xC", "5xC", "10xC"],
        "fs": [
            7.287991253247984e20,
            1.4568756657543709e21,
            2.9129564881239933e21,
            7.281271213722383e21,
            1.456181984254954e22,
        ],
        "mode": "train",
    },
    "7B-4T": {
        "ns": [6887575552],
        "ds": [3879731200000],
        "xs": [0.5329778790473938],
        "ls": [""],
        "fs": [1.9170555516032123e23],
        "mode": "eval",
    },
    "13B-5T": {
        "ns": [13202396160],
        "ds": [5000088518656],
        "xs": [0.5153260827064514],
        "ls": [""],
        "fs": [4.5668766253248636e23],
        "mode": "eval",
    },
}

DEFAULT_LADDER_CONFIG = {
    "190M": {
        "paths": [
            "scripts/scaling/data/peteish-moreeval/190M-0.5xC.csv",
            "scripts/scaling/data/peteish-moreeval/190M-1xC-rerun.csv",
            "scripts/scaling/data/peteish-moreeval/190M-2xC.csv",
            "scripts/scaling/data/peteish-moreeval/190M-5xC.csv",
            "scripts/scaling/data/peteish-moreeval/190M-10xC.csv"
        ],
        "mode": "train",
        "n": 190354176,
        "label": "190M",
        "color": "darkred"
    },
    "370M": {
        "paths": [
            "scripts/scaling/data/peteish-moreeval/370M-0.5xC.csv",
            "scripts/scaling/data/peteish-moreeval/370M-1xC-rerun.csv",
            "scripts/scaling/data/peteish-moreeval/370M-2xC.csv",
            "scripts/scaling/data/peteish-moreeval/370M-5xC.csv",
            "scripts/scaling/data/peteish-moreeval/370M-10xC.csv"
        ],
        "mode": "train",
        "n": 371262464,
        "label": "370M",
        "color": "darkorange"
    },
    "760M": {
        "paths": [
            "scripts/scaling/data/peteish-moreeval/760M-0.5xC.csv",
            "scripts/scaling/data/peteish-moreeval/760M-1xC-rerun.csv",
            "scripts/scaling/data/peteish-moreeval/760M-2xC.csv",
            "scripts/scaling/data/peteish-moreeval/760M-5xC.csv",
            "scripts/scaling/data/peteish-moreeval/760M-10xC.csv"
        ],
        "mode": "train",
        "n": 758220288,
        "label": "760M",
        "color": "darkgreen"
    },
    "1.3B": {
        "paths": [
            "scripts/scaling/data/peteish-moreeval/1B-0.5xC.csv",
            "scripts/scaling/data/peteish-moreeval/1B-1xC-rerun.csv",
            "scripts/scaling/data/peteish-moreeval/1B-2xC.csv",
            "scripts/scaling/data/peteish-moreeval/1B-5xC.csv",
            "scripts/scaling/data/peteish-moreeval/1B-10xC.csv"
        ],
        "mode": "train",
        "n": 1279395840,
        "label": "1.3B",
        "color": "teal"
    },
    "3.2B": {
        "paths": [
            "scripts/scaling/data/peteish-moreeval/3B-0.5xC.csv",
            "scripts/scaling/data/peteish-moreeval/3B-1xC-rerun.csv",
            "scripts/scaling/data/peteish-moreeval/3B-2xC.csv",
            "scripts/scaling/data/peteish-moreeval/3B-5xC.csv",
            "scripts/scaling/data/peteish-moreeval/3B-10xC.csv"
        ],
        "mode": "train",
        "n": 3169537280,
        "label": "3.2B",
        "color": "darkblue"
    },
    "7B-4T": {
        "paths": [
            "scripts/scaling/data/peteish-moreeval/peteish7_eval_anneal.csv"
        ],
        "mode": "eval",
        "n": 6887575552,
        "label": "7B-4T",
        "color": "darkviolet"
    },
    "13B-5T": {
        "paths": [
            "scripts/scaling/data/peteish-moreeval/peteish13_eval_final.csv"
        ],
        "mode": "eval",
        "n": 13202396160,
        "label": "13B-5T",
        "color": "mediumvioletred"
    }
}