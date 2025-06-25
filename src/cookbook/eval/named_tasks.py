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
        out_table = MiniFrame(title=results.title)

        leaf_tasks = []
        for named_group in self.tasks:
            # If a child is a task, compute the macro-average for the task
            if isinstance(named_group, BaseNamedTasksGroup):
                combined_table = named_group.combine(results)
                if combined_table is not None:
                    combined_table = combined_table.keep_cols(*[named_group.name])
                    # we manage to combine! lets put the combined score at the front
                    out_table = combined_table + out_table
            else:
                leaf_tasks += [named_group]

        filtered_rows = results.keep_cols(*leaf_tasks)
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
    tasks = [task for task in constants.ALL_MINERVA_TASKS]


@NamedTasksGroupRegistry.register("math")
class MathGroup(BaseAverageNamedTasksGroup):
    tasks = [task for task in constants.ALL_MATH_TASKS]


@NamedTasksGroupRegistry.register("gsm-symb")
class GsmSymbGroup(BaseAverageNamedTasksGroup):
    tasks = [task for task in constants.ALL_GSM_SYMB_TASKS]


@NamedTasksGroupRegistry.register("code")
class CodeGroup(BaseAverageNamedTasksGroup):
    tasks = [task for task in constants.ALL_CODEX_TASKS]


@NamedTasksGroupRegistry.register("agi_eval")
class AgiEvalGroup(BaseAverageNamedTasksGroup):
    tasks = [task for task in constants.ALL_AGI_EVAL_TASKS]


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
class FimGroup(BaseNamedTasksGroup):
    tasks = [task for task in constants.FIM_TASKS]


@NamedTasksGroupRegistry.register("crux-eval")
class CruxEvalGroup(BaseAverageNamedTasksGroup):
    tasks = [task for task in constants.CRUX_EVAL_TASKS]


@NamedTasksGroupRegistry.register("mt_mbpp")
class MtMbppGroup(BaseNamedTasksGroup):
    # this is legacy, no need to average it
    tasks = [task for task in constants.MULTILINGUAL_MBPP_TASKS]


@NamedTasksGroupRegistry.register("mt_mbpp_v2fix")
class MtMbppV2fixGroup(BaseAverageNamedTasksGroup):
    tasks = [task for task in constants.MULTILINGUAL_MBPP_TASKS_V2]


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
    tasks = [re.compile(r"^minerva.*:bpb::olmes$")]


# Task macro averages
@NamedTasksGroupRegistry.register("olmo3:dev:1b:math:bpb")
class Olmo3Dev1bMathBpbGroup(BaseAverageNamedTasksGroup):
    tasks = [
        # Math
        re.compile(r"^minerva.*:bpb::olmes$"),
    ]


@NamedTasksGroupRegistry.register("olmo3:dev:1b:code:bpb")
class Olmo3Dev1bCodeBpbGroup(BaseAverageNamedTasksGroup):
    tasks = [
        # Code
        re.compile(r"^codex_humaneval:3shot:bpb::none"),
        re.compile(r"^mbpp:3shot:bpb::none"),
        re.compile(r"^mt_mbpp_v2fix.*:bpb$"),
    ]


@NamedTasksGroupRegistry.register("olmo3:dev:1b:qa:rc")
class Olmo3Dev1bQaRcGroup(BaseAverageNamedTasksGroup):
    tasks = [
        # Core OLMES
        re.compile(r"^arc:rc::full$"),
        re.compile(r"^mmlu:rc$"),
        re.compile(r"^csqa:rc::olmes:full"),
        re.compile(r"^hellaswag:rc::olmes:full"),
        re.compile(r"^winogrande:rc::olmes:full"),
        re.compile(r"^socialiqa:rc::olmes:full"),
        re.compile(r"^piqa:rc::olmes:full"),

        # Gen OLMES
        re.compile(r"^coqa:rc::gen2mc$"),
        re.compile(r"^drop:rc::gen2mc$"),
        re.compile(r"^jeopardy:rc::gen2mc$"),
        re.compile(r"^naturalqs:rc::gen2mc$"),
        re.compile(r"^squad:rc::gen2mc$"),

        # New OLMo 3
        re.compile(r"^sciq:rc::olmo3"),
        re.compile(r"^qasper_yesno:rc::olmes"),
        re.compile(r"^basic_skills:rc::olmes"),
        re.compile(r"^lab_bench_dbqa$"),
        re.compile(r"^lab_bench_protocolqa$"),
        re.compile(r"^lambada:rc"),
        re.compile(r"^medmcqa:rc::none"),
        re.compile(r"^medqa:rc::none"),
        re.compile(r"^sciriff_yesno:rc::olmes"),
    ]


@NamedTasksGroupRegistry.register("olmo3:dev:1b:macro:bpb")
class Olmo3Dev1bMacroBpbGroup(BaseAverageNamedTasksGroup):
    tasks = [
        # Core OLMES
        re.compile(r"^arc:bpb::full$"),
        re.compile(r"^mmlu:bpb$"),
        re.compile(r"^csqa:bpb::olmes:full"),
        re.compile(r"^hellaswag:bpb::olmes:full"),
        re.compile(r"^winogrande:bpb::olmes:full"),
        re.compile(r"^socialiqa:bpb::olmes:full"),
        re.compile(r"^piqa:bpb::olmes:full"),

        # Gen OLMES
        re.compile(r"^coqa:bpb::gen2mc$"),
        re.compile(r"^drop:bpb::gen2mc$"),
        re.compile(r"^jeopardy:bpb::gen2mc$"),
        re.compile(r"^naturalqs:bpb::gen2mc$"),
        re.compile(r"^squad:bpb::gen2mc$"),

        # Math
        re.compile(r"^minerva:bpb::olmes$"),

        # Code
        re.compile(r"^codex_humaneval:3shot:bpb::none"),
        re.compile(r"^mbpp:3shot:bpb::none"),
        re.compile(r"^mt_mbpp_v2fix$"),

        # New OLMo 3
        re.compile(r"^sciq:bpb::olmo3"),
        re.compile(r"^qasper_yesno:bpb::olmes"),
        re.compile(r"^basic_skills:bpb::olmes"),
        re.compile(r"^lab_bench_dbqa:bpb$"),
        re.compile(r"^lab_bench_protocolqa:bpb$"),
        re.compile(r"^lambada:bpb"),
        re.compile(r"^medmcqa:bpb::none"),
        re.compile(r"^medqa:bpb::none"),
        re.compile(r"^sciriff_yesno:bpb::olmes"),
        re.compile(r"ultrachat_masked_ppl"),
        re.compile(r"wildchat_masked_ppl"),
    ]


@NamedTasksGroupRegistry.register("olmo3:dev:7b:macro:math")
class Olmo3Dev7bMacroMathGroup(BaseAverageNamedTasksGroup):
    tasks = [
        # Math
        re.compile(r"gsm8k::olmes"),
        re.compile(r"^gsm-symb$"),
        re.compile(r"^minerva$"),
    ]


@NamedTasksGroupRegistry.register("olmo3:dev:7b:macro:code_gen")
class Olmo3Dev7bMacroCodeGenGroup(BaseAverageNamedTasksGroup):
    tasks = [
        # Code
        re.compile(r"bigcodebench:3shot::olmo3"),
        re.compile(r"codex_humaneval:3shot::olmo3"),
        # "crux-eval$", # we noticed I/O scores are noisy, so we don't include in the average
        re.compile(r"deepseek_leetcode::olmo3"),
        re.compile(r"ds1000:3shot::olmo3"),
        re.compile(r"mbpp:3shot::olmo3"),
        re.compile(r"multipl_e:6lang::olmo3"),
    ]


@NamedTasksGroupRegistry.register("olmo3:dev:7b:macro:code_fim")
class Olmo3Dev7bMacroCodeFimGroup(BaseAverageNamedTasksGroup):
    tasks = [
        # Code
        re.compile(r"fim$"),
    ]


@NamedTasksGroupRegistry.register("olmo3:dev:7b:macro:gen")
class Olmo3Dev7bMacroGenGroup(BaseAverageNamedTasksGroup):
    tasks = [
        re.compile(r"hellaswag:rc::xlarge"),
        re.compile(r"winogrande:rc::xlarge"),
        re.compile(r"lambada$"),
        re.compile(r"^basic:rc$"),

        # Gen OLMES
        re.compile(r"drop::xlarge"),
        re.compile(r"jeopardy::xlarge"),
        re.compile(r"naturalqs::xlarge"),
        re.compile(r"squad::xlarge"),
        re.compile(r"coqa::xlarge"),
    ]


@NamedTasksGroupRegistry.register("olmo3:dev:7b:macro:mcqa")
class Olmo3Dev7bMacroMcqaGroup(BaseAverageNamedTasksGroup):
    tasks = [
        # Core OLMES
        # re.compile(r"^arc:mc::xlarge$"),
        ARCMCXlargeGroup(),
        # re.compile(r"^mmlu:mc$"),
        MMLUMCGroup(),
        re.compile(r"csqa:mc::xlarge"),
        re.compile(r"piqa:mc::xlarge"),
        re.compile(r"socialiqa:mc::xlarge"),

        # Gen2MC OLMES
        re.compile(r"coqa:mc::gen2mc"),
        re.compile(r"drop:mc::gen2mc"),
        re.compile(r"jeopardy:mc::gen2mc"),
        re.compile(r"naturalqs:mc::gen2mc"),
        re.compile(r"squad:mc::gen2mc"),

        # New OLMo 3
        BasicRCGroup(),
        # re.compile(r"lab_bench_dbqa:mc"), # too noisy to include in macro-average
        # re.compile(r"lab_bench_protocolqa:mc"), # too noisy to include in macro-average
        re.compile(r"medmcqa:mc::none"),
        re.compile(r"medqa_en:mc::none"),
        re.compile(r"sciq:mc::xlarge"),
    ]


# # # # # # # # # # # # # # # DISPLAY TASK GROUPS # # # # # # # # # # # # # # # # #
#  These are just shortcuts to display many metrics at once. no need to average.  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@NamedTasksGroupRegistry.register("olmo2:paper")
class Olmo2PaperGroup(BaseAverageNamedTasksGroup):
    tasks = [
        re.compile(r"arc_challenge:(rc|mc)::olmes$"),
        re.compile(r"hellaswag:(rc|mc)::olmes$"),
        re.compile(r"winogrande:(rc|mc)::olmes$"),
        re.compile(r"naturalqs::olmes$"),
        re.compile(r"drop::olmes$"),
        re.compile(r"agieval.*::olmes$"),
        re.compile(r"^gsm8k::olmes$"),
        CoreMCGroup(),
        MMLUProMCGroup(),
        re.compile(r"^agi_eval$"),
    ]


@NamedTasksGroupRegistry.register("olmo2:dev:7b")
class Olmo2Dev7bGroup(BaseAverageNamedTasksGroup):
    tasks = [
        re.compile(r"arc_challenge:mc.*"),
        re.compile(r"arc_easy:mc.*"),
        re.compile(r"hellaswag:mc.*"),
        re.compile(r"naturalqs.*"),
        re.compile(r"^gsm8k::olmo1$"),
        MMLUMCGroup(),
        CoreMCGroup(),
        GenGroup(),
    ]


@NamedTasksGroupRegistry.register("olmo2:dev:1b")
class Olmo2Dev1bGroup(BaseAverageNamedTasksGroup):
    tasks = [
        re.compile(r"arc_challenge:rc.*"),
        re.compile(r"arc_easy:rc.*"),
        re.compile(r"hellaswag:rc.*"),
        re.compile(r"^gsm8k::olmo1$"),
        MMLURCGroup(),
        CoreRCGroup(),
    ]


@NamedTasksGroupRegistry.register("olmo3:dev:1b:main")
class Olmo3Dev1bMainGroup(BaseAverageNamedTasksGroup):
    tasks = [
        # re.compile(r"^olmo3:dev:1b:macro:w_avg$"),
        Olmo3Dev1bMathBpbGroup(),
        Olmo3Dev1bCodeBpbGroup(),
        Olmo3Dev1bQaRcGroup(),
        ARCRCFullGroup(),
        re.compile(r"^hellaswag:rc::olmes:full"),
        BasicRCGroup(),
        MtMbppV2fixGroup(),
        MMLURCGroup(),
        MMLUBpbGroup(),
        CoreRCGroup(),
        re.compile(r"codex_humaneval:3shot:bpb::none"),
        re.compile(r"mbpp:3shot:bpb::none"),
        MinervaBpbGroup()
    ]


@NamedTasksGroupRegistry.register("olmo3:dev:7b:main")
class Olmo3Dev7bMainGroup(BaseAverageNamedTasksGroup):
    tasks = [
        # re.compile(r"^olmo3:dev:7b:macro:w_avg$"),
        Olmo3Dev7bMacroMcqaGroup(),
        Olmo3Dev7bMacroMathGroup(),
        Olmo3Dev7bMacroCodeGenGroup(),
        Olmo3Dev7bMacroCodeFimGroup(),
        Olmo3Dev7bMacroGenGroup(),
        ARCMCXlargeGroup(),
        MMLUMCGroup(),
        re.compile(r"^codex_humaneval:3shot::olmo3$"),
        re.compile(r"^mbpp:3shot::olmo3$"),
        MultiPlEHEGroup(),
        MultiPlEMBPPGroup(),
        re.compile(r"^gsm8k::olmes$"),
        GsmSymbGroup(),
        MinervaGroup(),
        BasicRCGroup(),
        GenXlargeGroup(),
        CruxEvalGroup(),
    ]
