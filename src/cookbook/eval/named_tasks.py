from functools import cached_property
import re
from typing import Callable, ClassVar, Type, TypeVar, Union

from cookbook import constants
from cookbook.eval.miniframe import MiniFrame


T = TypeVar("T", bound=Type["BaseNamedTasksGroup"])


class NamedTasksGroupRegistry:
    """
    Registry for named tasks.
    """

    _instance: ClassVar[Union["NamedTasksGroupRegistry", None]] = None
    _named_tasks: ClassVar[dict[str, Type["BaseNamedTasksGroup"]]] = {}

    def __new__(cls, *args, **kwargs):
        # singleton pattern
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, task_name: str) -> Callable[[T], T]:

        # instantiate the singleton instance here; it won't get instantiated
        # twice cuz it's a singleton, after all.
        instance = cls()

        def decorator(task: T, _task_name: str = task_name, _instance: "NamedTasksGroupRegistry" = instance) -> T:
            # add to the registry
            _instance._named_tasks[_task_name] = task

            # a little bit of a Python crime, but when registering a group,
            # we replace the class name with the task name for the `.name` property.
            task.name = _task_name  # pyright: ignore
            return task

        return decorator

    @classmethod
    def names(cls) -> list[str]:
        return list(cls._named_tasks.keys())

    @classmethod
    def exists(cls, task_name: str) -> bool:
        return any(cls.search(task_name))

    @classmethod
    def search(cls, task_name: str | re.Pattern) -> list[str]:
        """Return all tasks that match the given pattern."""
        return [
            task
            for task in cls._named_tasks.keys()
            if (task_name.search(task) if isinstance(task_name, re.Pattern) else task_name == task)
        ]

    @classmethod
    def get(cls, task_name: str) -> "BaseNamedTasksGroup":
        assert cls._instance is not None, "NamedTasksGroupRegistry is not initialized"

        if task_name not in cls._named_tasks:
            raise ValueError(f"Task {task_name} not found")
        return cls._named_tasks[task_name]()


class BaseNamedTasksGroup:
    """
    Base class for named tasks.

    Subclasses should define the `tasks` class variable.
    """

    # external, to be defined by subclasses
    tasks: ClassVar[list[Union[str, re.Pattern, "BaseNamedTasksGroup"]]] = []

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @cached_property
    def expanded_tasks(self) -> list[Union[str, re.Pattern]]:
        """
        Return the list of tasks in this group, expanding any nested named groups.
        """
        expanded_tasks: list[Union[str, re.Pattern]] = []
        for task in self.tasks:
            if isinstance(task, BaseNamedTasksGroup):
                expanded_tasks.extend(task.expanded_tasks)
            else:
                expanded_tasks.append(task)
        return expanded_tasks

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tasks={self.tasks})"

    def combine(self, results: MiniFrame) -> MiniFrame | None:
        """
        Combine the metrics of the tasks in the group into a single score.
        If this function returns None, it means that this group of metrics is not supposed to be averaged.
        """
        return None


class BaseAverageNamedTasksGroup(BaseNamedTasksGroup):
    """
    Base class for named tasks groups that are supposed to be averaged.
    """

    def combine(self, results: MiniFrame) -> MiniFrame | None:
        filtered_rows = results.keep_cols(*self.expanded_tasks)
        if len(filtered_rows) == 0:
            return None

        combined_table = MiniFrame(title=results.title)

        # each row here is a model
        for row in filtered_rows.rows:
            # we compute the average of the scores for this model; we set the average to None if
            # there are missing scores or if there are no scores at all.
            average: float | None = None
            if len(row.values) > 0 and all(s is not None for s in row.values):
                filtered_scores = [s for s in row.values if s is not None]
                average = (sum(filtered_scores) / len(filtered_scores)) if filtered_scores else 0.0

            # we add the average to the combined table
            combined_table.add(col=self.name, row=row.name, val=average)

        return combined_table


class BaseAverageOfAveragesNamedTasksGroup(BaseAverageNamedTasksGroup):
    """
    Base class for named tasks groups that include averages of other task groups (e.g., a macro average over all QA tasks, which includes MMLU)
    """

    def combine(self, results: MiniFrame) -> MiniFrame | None:
        out_table = MiniFrame(title=results.title)

        # get task groups (e.g., MMLURCGroup())
        named_group_children: list[BaseAverageNamedTasksGroup] = [
            task for task in self.tasks if isinstance(task, BaseAverageNamedTasksGroup) 
        ]

        # get individual tasks (e.g., "arc_challenge:rc::olmes")
        task_children = [
            task for task in self.tasks if not isinstance(task, BaseAverageNamedTasksGroup) 
        ]

        # calculate the averages for all child task groups
        for named_group in named_group_children:
            combined_table = named_group.combine(results)
            if combined_table is not None:
                # If the named group is able to average all scores, add it!
                combined_table = combined_table.keep_cols(*[named_group.name])
                out_table = combined_table + out_table

        # get the aliases for all task groups
        all_tasks = \
            task_children + \
            [named_group.name for named_group in named_group_children]

        # now get a table of the child tasks and macro averages calculated!
        filtered_rows = results.keep_cols(*all_tasks)
        out_table = filtered_rows + out_table

        if len(out_table) == 0:
            return None

        # each row here is a model
        for row in filtered_rows.rows:

            # we compute the average of the scores for this model; we set the average to None if
            # there are missing scores or if there are no scores at all.
            average: float | None = None
            if len(row.values) > 0 and all(s is not None for s in row.values):
                filtered_scores = [s for s in row.values if s is not None]
                average = (sum(filtered_scores) / len(filtered_scores)) if filtered_scores else 0.0

            # we add the average to the combined table
            out_table.add(col=self.name, row=row.name, val=average)

        return out_table


# # # # # # # # # # # # # # # # NAMED TASK GROUPS # # # # # # # # # # # # # # # # #
#  Typically, you want these to be averaged in some way when displaying results.  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@NamedTasksGroupRegistry.register("mmlu:rc")
class MMLURCGroup(BaseAverageNamedTasksGroup):
    tasks = [f"{category}:rc::olmes" for category in constants.MMLU_CATEGORIES]


@NamedTasksGroupRegistry.register("mmlu:bpb")
class MMLUBpbGroup(BaseAverageNamedTasksGroup):
    tasks = [f"{category}:bpb::olmes" for category in constants.MMLU_CATEGORIES]


@NamedTasksGroupRegistry.register("mmlu:mc")
class MMLUMCGroup(BaseAverageNamedTasksGroup):
    tasks = [f"{category}:mc::olmes" for category in constants.MMLU_CATEGORIES]


@NamedTasksGroupRegistry.register("mmlu:cot::hamish_zs_reasoning")
class MMLUHamishZSReasoningGroup(BaseAverageNamedTasksGroup):
    tasks = [f"{category}:cot::hamish_zs_reasoning" for category in constants.MMLU_CATEGORIES]


@NamedTasksGroupRegistry.register("core:rc")
class CoreRCGroup(BaseAverageNamedTasksGroup):
    tasks = [f"{task}:rc::olmes" for task in constants.ALL_CORE_TASKS]


@NamedTasksGroupRegistry.register("core:mc::full")
class CoreMCFullGroup(BaseAverageNamedTasksGroup):
    tasks = [f"{task}:mc::olmes::full" for task in constants.ALL_CORE_TASKS]


@NamedTasksGroupRegistry.register("core:rc::full")
class CoreRCFullGroup(BaseAverageNamedTasksGroup):
    tasks = [f"{task}:rc::olmes::full" for task in constants.ALL_CORE_TASKS]


@NamedTasksGroupRegistry.register("core:mc")
class CoreMCGroup(BaseAverageNamedTasksGroup):
    tasks = [f"{task}:mc::olmes" for task in constants.ALL_CORE_TASKS]


@NamedTasksGroupRegistry.register("arc:rc")
class ARCRCGroup(BaseAverageNamedTasksGroup):
    tasks = [f"{category}:rc::olmes" for category in constants.ARC_TASKS]


@NamedTasksGroupRegistry.register("arc:mc")
class ARCMCGroup(BaseAverageNamedTasksGroup):
    tasks = [f"{category}:mc::olmes" for category in constants.ARC_TASKS]


@NamedTasksGroupRegistry.register("arc:rc::full")
class ARCRCFullGroup(BaseAverageNamedTasksGroup):
    tasks = [f"{category}:rc::olmes:full" for category in constants.ARC_TASKS]


@NamedTasksGroupRegistry.register("arc:mc::full")
class ARCMCFullGroup(BaseAverageNamedTasksGroup):
    tasks = [f"{category}:mc::olmes:full" for category in constants.ARC_TASKS]


@NamedTasksGroupRegistry.register("arc:rc::xlarge")
class ARCRCXlargeGroup(BaseAverageNamedTasksGroup):
    tasks = [f"{category}:rc::xlarge" for category in constants.ARC_TASKS]


@NamedTasksGroupRegistry.register("arc:mc::xlarge")
class ARCMCXlargeGroup(BaseAverageNamedTasksGroup):
    tasks = [f"{category}:mc::xlarge" for category in constants.ARC_TASKS]


@NamedTasksGroupRegistry.register("basic:rc")
class BasicRCGroup(BaseAverageNamedTasksGroup):
    tasks = [f"{task}:rc::olmes" for task in constants.BASIC_SKILLS]


@NamedTasksGroupRegistry.register("basic:mc")
class BasicMCGroup(BaseAverageNamedTasksGroup):
    tasks = [f"{task}:mc::olmes" for task in constants.BASIC_SKILLS]


@NamedTasksGroupRegistry.register("mmlu_pro:mc")
class MMLUProMCGroup(BaseAverageNamedTasksGroup):
    tasks = [f"{category}:mc::none" for category in constants.MMLU_PRO_CATEGORIES]


@NamedTasksGroupRegistry.register("gen")
class GenGroup(BaseAverageNamedTasksGroup):
    tasks = [task for task in constants.ALL_GEN_TASKS]


@NamedTasksGroupRegistry.register("gen-no-jp")
class GenNoJpGroup(BaseNamedTasksGroup):
    # this is legacy, no need to average it
    tasks = [task for task in constants.ALL_GEN_TASKS if task != "jeopardy::olmes"]


@NamedTasksGroupRegistry.register("gen-no-gsm")
class GenNoGsmGroup(BaseNamedTasksGroup):
    # this is legacy, no need to average it
    tasks = [task for task in constants.ALL_GEN_TASKS if task != "gsm8k::olmo1"]


@NamedTasksGroupRegistry.register("gen::xlarge")
class GenXlargeGroup(BaseAverageNamedTasksGroup):
    tasks = [task for task in constants.ALL_GEN_XLARGE_TASKS]


@NamedTasksGroupRegistry.register("minerva")
class MinervaGroup(BaseAverageNamedTasksGroup):
    tasks = [f"{subtask}::olmes" for subtask in constants.ALL_MINERVA_TASKS]


@NamedTasksGroupRegistry.register("minerva::hamish_zs_reasoning")
class MinervaHamishZSReasoningGroup(BaseAverageNamedTasksGroup):
    tasks = [f"{subtask}::hamish_zs_reasoning" for subtask in constants.ALL_MINERVA_TASKS]


@NamedTasksGroupRegistry.register("math")
class MathGroup(BaseAverageNamedTasksGroup):
    tasks = [
        "gsm8k::olmo1", 
        "gsm8k::olmes",
        [f"{subtask}::olmes" for subtask in constants.ALL_MINERVA_TASKS]
    ]


@NamedTasksGroupRegistry.register("gsm-symb")
class GsmSymbGroup(BaseAverageNamedTasksGroup):
    tasks = [task for task in constants.ALL_GSM_SYMB_TASKS]


@NamedTasksGroupRegistry.register("code")
class CodeGroup(BaseAverageNamedTasksGroup):
    tasks = [task for task in constants.ALL_CODEX_TASKS]


@NamedTasksGroupRegistry.register("agi_eval")
class AgiEvalGroup(BaseAverageNamedTasksGroup):
    tasks = [f"{task}:1shot::olmes" for task in constants.AGI_EVAL_ENGLISH_TASKS]


@NamedTasksGroupRegistry.register("agi_eval_english:0shot_cot::hamish_zs_reasoning")
class AgiEvalEnglishHamishZsReasoningGroup(BaseAverageNamedTasksGroup):
    tasks = [f"agi_eval_{task}:0shot_cot::hamish_zs_reasoning" for task in constants.AGI_EVAL_ENGLISH_TASKS]


@NamedTasksGroupRegistry.register("starcoder")
class StarcoderGroup(BaseAverageNamedTasksGroup):
    tasks = [task for task in constants.STARCODER_CODEX_TASKS]


@NamedTasksGroupRegistry.register("starcoder::pass@1")
class StarcoderPassAt1Group(BaseAverageNamedTasksGroup):
    tasks = [task for task in constants.STARCODER_PASS_AT_1_TASKS]


@NamedTasksGroupRegistry.register("code-no-bcb")
class CodeNoBcbGroup(BaseNamedTasksGroup):
    tasks = [task for task in constants.ALL_CODEX_TASKS if "bigcodebench" not in task]


@NamedTasksGroupRegistry.register("multipl-e-humaneval")
class MultiPlEHEGroup(BaseAverageNamedTasksGroup):
    tasks = [task for task in constants.MULTIPL_E_HE_TASKS]


@NamedTasksGroupRegistry.register("multipl-e-mbpp")
class MultiPlEMBPPGroup(BaseAverageNamedTasksGroup):
    tasks = [task for task in constants.MULTIPL_E_MBPP_TASKS]


@NamedTasksGroupRegistry.register("fim")
class FimGroup(BaseAverageNamedTasksGroup):
    tasks = [task for task in constants.FIM_TASKS]


@NamedTasksGroupRegistry.register("crux-eval")
class CruxEvalGroup(BaseAverageNamedTasksGroup):
    tasks = [task for task in constants.CRUX_EVAL_TASKS]


@NamedTasksGroupRegistry.register("mt_mbpp")
class MtMbppGroup(BaseAverageNamedTasksGroup):
    # this is legacy, no need to average it
    tasks = [task for task in constants.MULTILINGUAL_MBPP_TASKS]


@NamedTasksGroupRegistry.register("mt_mbpp_v2fix")
class MtMbppV2fixGroup(BaseAverageNamedTasksGroup):
    tasks = [task for task in constants.MULTILINGUAL_MBPP_TASKS_V2]


@NamedTasksGroupRegistry.register("bbh:cot::hamish_zs_reasoning")
class BBHHamishZSReasoningGroup(BaseAverageNamedTasksGroup):
    tasks = [f"bbh_{category}:cot::hamish_zs_reasoning" for category in constants.BBH_TASKS]


def make_helmet_group(helmet_length: int) -> Type[BaseAverageNamedTasksGroup]:
    class HelmetGroup(BaseAverageNamedTasksGroup):
        tasks = [
            task
            for group_name, tasks in constants.HELMET_SUITES.items()
            for task in tasks
            if group_name.endswith(f"__{helmet_length}::suite") and not group_name.startswith("helmet_all")
        ]

    return HelmetGroup


for helmet_length in (int(2**i) for i in range(13, 18)):
    NamedTasksGroupRegistry.register(f"helmet:{helmet_length // 2 ** 10}k")(make_helmet_group(helmet_length))


@NamedTasksGroupRegistry.register("minerva:bpb")
class MinervaBpbGroup(BaseAverageNamedTasksGroup):
    tasks = [f"{subtask}:bpb::olmes" for subtask in constants.ALL_MINERVA_TASKS]


# Task macro averages
@NamedTasksGroupRegistry.register("olmo3:dev:1b:math:bpb")
class Olmo3Dev1bMathBpbGroup(BaseAverageOfAveragesNamedTasksGroup):
    tasks = [
        # Math
        MinervaBpbGroup(),
    ]


@NamedTasksGroupRegistry.register("olmo3:dev:1b:code:bpb")
class Olmo3Dev1bCodeBpbGroup(BaseAverageOfAveragesNamedTasksGroup):
    tasks = [
        # Code
        "codex_humaneval:3shot:bpb::none",
        "mbpp:3shot:bpb::none",
        MtMbppV2fixGroup(),
    ]


@NamedTasksGroupRegistry.register("olmo3:dev:1b:qa:rc")
class Olmo3Dev1bQaRcGroup(BaseAverageOfAveragesNamedTasksGroup):
    tasks = [
        # Core OLMES
        ARCRCFullGroup(),
        MMLURCGroup(),
        "csqa:rc::olmes:full",
        "hellaswag:rc::olmes:full",
        "winogrande:rc::olmes:full",
        "socialiqa:rc::olmes:full",
        "piqa:rc::olmes:full",

        # Gen OLMES
        "coqa:rc::gen2mc",
        "drop:rc::gen2mc",
        "jeopardy:rc::gen2mc",
        "naturalqs:rc::gen2mc",
        "squad:rc::gen2mc",

        # New OLMo 3
        "sciq:rc::olmo3",
        "qasper_yesno:rc::olmes",
        "basic_skills:rc::olmes",
        "lab_bench_dbqa",
        "lab_bench_protocolqa",
        "lambada",
        "medmcqa:rc::none",
        "medqa_en:rc::none",
        "sciriff_yesno:rc::olmes",
    ]


@NamedTasksGroupRegistry.register("olmo3:dev:1b:bpb")
class Olmo3Dev1bBpbGroup(BaseAverageOfAveragesNamedTasksGroup):
    tasks = [
        # Core OLMES
        "arc:bpb::full$",
        "mmlu:bpb$",
        "csqa:bpb::olmes:full",
        "hellaswag:bpb::olmes:full",
        "winogrande:bpb::olmes:full",
        "socialiqa:bpb::olmes:full",
        "piqa:bpb::olmes:full",

        # Gen OLMES
        "coqa:bpb::gen2mc",
        "drop:bpb::gen2mc",
        "jeopardy:bpb::gen2mc",
        "naturalqs:bpb::gen2mc",
        "squad:bpb::gen2mc",

        # Math
        MinervaBpbGroup(),

        # Code
        Olmo3Dev1bCodeBpbGroup(),

        # New OLMo 3
        "sciq:bpb::olmo3",
        "qasper_yesno:bpb::olmes",
        "basic_skills:bpb::olmes",
        "lab_bench_dbqa:bpb",
        "lab_bench_protocolqa:bpb",
        "lambada:bpb",
        "medmcqa:bpb::none",
        "medqa_en:bpb::none",
        "sciriff_yesno:bpb::olmes",
        "ultrachat_masked_ppl",
        "wildchat_masked_ppl",
    ]


@NamedTasksGroupRegistry.register("olmo3:dev:7b:math")
class Olmo3Dev7bMathGroup(BaseAverageOfAveragesNamedTasksGroup):
    tasks = [
        # Math
        "gsm8k::olmes",
        GsmSymbGroup(),
        MinervaGroup(),
    ]


@NamedTasksGroupRegistry.register("olmo3:dev:7b:code_gen")
class Olmo3Dev7bCodeGenGroup(BaseAverageOfAveragesNamedTasksGroup):
    tasks = [
        # Code
        "bigcodebench:3shot::olmo3",
        "codex_humaneval:3shot::olmo3",
        "deepseek_leetcode::olmo3",
        "ds1000:3shot::olmo3",
        "mbpp:3shot::olmo3",
        MultiPlEHEGroup(),
        MultiPlEMBPPGroup(),
        # "crux-eval$", # we noticed I/O scores are noisy, so we don't include in the average
    ]


@NamedTasksGroupRegistry.register("olmo3:dev:7b:code_fim")
class Olmo3Dev7bCodeFimGroup(BaseAverageOfAveragesNamedTasksGroup):
    tasks = [
        # Code
        FimGroup(),
    ]


@NamedTasksGroupRegistry.register("olmo3:dev:7b:gen")
class Olmo3Dev7bGenGroup(BaseAverageOfAveragesNamedTasksGroup):
    tasks = [
        "hellaswag:rc::xlarge",
        "winogrande:rc::xlarge",
        "lambada",
        BasicRCGroup(),

        # Gen OLMES
        "drop::xlarge",
        "jeopardy::xlarge",
        "naturalqs::xlarge",
        "squad::xlarge",
        "coqa::xlarge",
    ]


@NamedTasksGroupRegistry.register("olmo3:dev:7b:mcqa")
class Olmo3Dev7bMcqaGroup(BaseAverageOfAveragesNamedTasksGroup):
    tasks = [
        # Core OLMES
        ARCMCXlargeGroup(),
        MMLUMCGroup(),
        "csqa:mc::xlarge",
        "piqa:mc::xlarge",
        "socialiqa:mc::xlarge",

        # Gen2MC OLMES
        "coqa:mc::gen2mc",
        "drop:mc::gen2mc",
        "jeopardy:mc::gen2mc",
        "naturalqs:mc::gen2mc",
        "squad:mc::gen2mc",

        # New OLMo 3
        BasicRCGroup(),
        # "lab_bench_dbqa:mc", # too noisy to include in macro-average
        # "lab_bench_protocolqa:mc", # too noisy to include in macro-average
        "medmcqa:mc::none",
        "medqa_en:mc::none",
        "sciq:mc::xlarge",
    ]


# # # # # # # # # # # # # # # DISPLAY TASK GROUPS # # # # # # # # # # # # # # # # #
#  These are just shortcuts to display many metrics at once. no need to average.  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@NamedTasksGroupRegistry.register("olmo2:paper")
class Olmo2PaperGroup(BaseNamedTasksGroup):
    tasks = [
        "arc_challenge:rc::olmes",
        "arc_challenge:mc::olmes",
        "hellaswag:rc::olmes",
        "hellaswag:mc::olmes",
        "winogrande:rc::olmes",
        "winogrande:mc::olmes",
        "naturalqs::olmes",
        "drop::olmes",
        "agieval.*::olmes",
        "gsm8k::olmes",
        CoreMCGroup(),
        MMLUProMCGroup(),
        "triviaqa::olmes"
    ]


@NamedTasksGroupRegistry.register("olmo2:dev:7b")
class Olmo2Dev7bGroup(BaseAverageOfAveragesNamedTasksGroup):
    tasks = [
        "arc_challenge:mc::olmes",
        "arc_easy:mc::olmes",
        "hellaswag:mc::olmes",
        "naturalqs::olmes",
        "^gsm8k::olmo1",
        MMLUMCGroup(),
        CoreMCGroup(),
        GenGroup(),
    ]


@NamedTasksGroupRegistry.register("olmo2:dev:1b")
class Olmo2Dev1bGroup(BaseAverageOfAveragesNamedTasksGroup):
    tasks = [
        "arc_challenge:rc::olmes",
        "arc_easy:rc::olmes",
        "hellaswag:rc::olmes",
        "gsm8k::olmo1",
        MMLURCGroup(),
        CoreRCGroup(),
    ]


@NamedTasksGroupRegistry.register("olmo3:dev:1b:main")
class Olmo3Dev1bMainGroup(BaseAverageOfAveragesNamedTasksGroup):
    tasks = [
        # re.compile(r"^olmo3:dev:1b:macro:w_avg$"),
        Olmo3Dev1bMathBpbGroup(),
        Olmo3Dev1bCodeBpbGroup(),
        Olmo3Dev1bQaRcGroup(),
        ARCRCFullGroup(),
        "hellaswag:rc::olmes:full",
        BasicRCGroup(),
        MtMbppV2fixGroup(),
        MMLURCGroup(),
        MMLUBpbGroup(),
        CoreRCGroup(),
        "codex_humaneval:3shot:bpb::none",
        "mbpp:3shot:bpb::none",
        MinervaBpbGroup()
    ]


@NamedTasksGroupRegistry.register("olmo3:dev:7b:main")
class Olmo3Dev7bMainGroup(BaseAverageOfAveragesNamedTasksGroup):
    tasks = [
        # re.compile(r"^olmo3:dev:7b:macro:w_avg$"),
        Olmo3Dev7bMcqaGroup(),
        Olmo3Dev7bMathGroup(),
        Olmo3Dev7bCodeGenGroup(),
        Olmo3Dev7bCodeFimGroup(),
        Olmo3Dev7bGenGroup(),
        ARCMCXlargeGroup(),
        MMLUMCGroup(),
        "codex_humaneval:3shot::olmo3",
        "mbpp:3shot::olmo3",
        MultiPlEHEGroup(),
        MultiPlEMBPPGroup(),
        "gsm8k::olmes",
        GsmSymbGroup(),
        MinervaGroup(),
        BasicRCGroup(),
        GenXlargeGroup(),
        CruxEvalGroup(),
    ]


@NamedTasksGroupRegistry.register("olmo3:dev:midtrain:v0")
class Olmo3Dev7bMainGroup(BaseAverageOfAveragesNamedTasksGroup):
    tasks = [
        # Everything in this task set is 0-shot
        "aime::hamish_zs_reasoning",
        "alpaca_eval_v3::hamish_zs_reasoning",
        "codex_humanevalplus:0-shot-chat::tulu-thinker",
        "gpqa:0shot_cot::hamish_zs_reasoning", # requires 4096 context window
        "gsm8k::zs_cot_latex",  #### from adapt: to replace "gsm8k::hamish_zs_reasoning"
        "ifeval::hamish_zs_reasoning",
        "mbppplus:0-shot-chat::tulu-thinker",
        "minerva_math_500::hamish_zs_reasoning",
        "popqa::hamish_zs_reasoning",  #### from adapt: fix and test this guy.
        AgiEvalEnglishHamishZsReasoningGroup(),
        BBHHamishZSReasoningGroup(),
        MinervaHamishZSReasoningGroup(),
        MMLUHamishZSReasoningGroup(),
        "zebralogic::hamish_zs_reasoning"
        "livecodebench_codegeneration::tulu-thinker",

        ### Not yet implemented
        # cruxeval
        # simpleqa
        # gpqa diamond
        # AMC 22/23
        # math OOD
        # turnwise
        # typos eval
        # bcfl v3
        # ace bench
        # appworld

        # all safety
    ]
