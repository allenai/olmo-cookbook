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
        return task_name in cls._named_tasks

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
        tasks_repr = ",".join(str(task) if isinstance(task, str) else repr(task) for task in self.tasks)
        return f"{self.__class__.__name__}({tasks_repr})"


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


# # # # # # # # # # # # # # # # NAMED TASK GROUPS # # # # # # # # # # # # # # # # #
#  Typically, you want these to be averaged in some way when displaying results.  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@NamedTasksGroupRegistry.register("mmlu:rc")
class MMLURCGroup(BaseAverageNamedTasksGroup):
    tasks = [f"{category}:rc::olmes" for category in constants.MMLU_CATEGORIES]


@NamedTasksGroupRegistry.register("mmlu:mc")
class MMLUMCGroup(BaseAverageNamedTasksGroup):
    tasks = [f"{category}:mc::olmes" for category in constants.MMLU_CATEGORIES]


@NamedTasksGroupRegistry.register("core:rc")
class CoreRCGroup(BaseAverageNamedTasksGroup):
    tasks = [f"{task}:rc::olmes" for task in constants.ALL_CORE_TASKS]


@NamedTasksGroupRegistry.register("core:mc")
class CoreMCGroup(BaseAverageNamedTasksGroup):
    tasks = [f"{task}:mc::olmes" for task in constants.ALL_CORE_TASKS]


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


@NamedTasksGroupRegistry.register("fim")
class FimGroup(BaseNamedTasksGroup):
    tasks = [task for task in constants.FIM_TASKS]


@NamedTasksGroupRegistry.register("mt_mbpp")
class MtMbppGroup(BaseNamedTasksGroup):
    # this is legacy, no need to average it
    tasks = [f"mt_mbpp:{language}" for language in constants.MULTILINGUAL_MBPP_LANGUAGES]


@NamedTasksGroupRegistry.register("mt_mbpp_v2fix")
class MtMbppV2fixGroup(BaseAverageNamedTasksGroup):
    tasks = [f"mt_mbpp_v2fix:{language}" for language in constants.MULTILINGUAL_MBPP_LANGUAGES]


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


# # # # # # # # # # # # # # # DISPLAY TASK GROUPS # # # # # # # # # # # # # # # # #
#  These are just shortcuts to display many metrics at once. no need to average.  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@NamedTasksGroupRegistry.register("olmo2:paper")
class Olmo2PaperGroup(BaseNamedTasksGroup):
    tasks = [
        re.compile(r"arc_challenge:mc.*"),
        re.compile(r"hellaswag:mc.*"),
        re.compile(r"winogrande:mc.*"),
        re.compile(r"naturalqs.*"),
        re.compile(r"drop.*"),
        re.compile(r"agieval.*"),
    ]


@NamedTasksGroupRegistry.register("olmo2:dev:7b")
class Olmo2Dev7bGroup(BaseNamedTasksGroup):
    tasks = [
        re.compile(r"arc_challenge:mc.*"),
        re.compile(r"arc_easy:mc.*"),
        re.compile(r"hellaswag:mc.*"),
        re.compile(r"naturalqs.*"),
        re.compile(r"^gsm8k::olmo1$"),
        re.compile(r"^mmlu:mc$"),
        re.compile(r"^core:mc$"),
    ]


@NamedTasksGroupRegistry.register("olmo3:dev:1b")
class Olmo3Dev1bGroup(BaseNamedTasksGroup):
    tasks = [
        re.compile(r"^arc_challenge.*olmes"),  # should return mc, rc, bpb variants, full or not
        re.compile(r"^arc_easy.*olmes"),
        re.compile(r"^basic_skills.*olmes"),
        re.compile(r"^codex_humaneval.*3shot"),
        re.compile(r"^coqa.*gen2mc"),
        re.compile(r"^csqa.*olmes"),
        re.compile(r"^drop.*gen2mc"),
        re.compile(r"^hellaswag.*olmes"),
        re.compile(r"^jeopardy.*gen2mc"),
        re.compile(r"^lab_bench.*"),
        re.compile(r"^lambada.*"),
        re.compile(r"^mbpp.*3shot"),
        re.compile(r"^medmcqa.*none"),
        re.compile(r"^medqa.*none"),
        re.compile(r"^minerva.*olmes$"),  # doesn't return average
        re.compile(r"^mmlu.*olmes$"),  # doesn't return average
        re.compile(r"^mt_mbpp_v2fix.*"),  # still returns average
        re.compile(r"^naturalqs.*gen2mc"),
        re.compile(r"^piqa.*olmes"),
        re.compile(r"^qasper_yesno.*olmes"),
        re.compile(r"^sciq.*olmo3"),
        re.compile(r"^sciriff_yesno.*olmes"),
        re.compile(r"^socialiqa.*olmes"),
        re.compile(r"^squad.*gen2mc"),
        re.compile(r"^winogrande.*olmes"),
        "ultrachat_masked_ppl",
        "wildchat_masked_ppl",
    ]


@NamedTasksGroupRegistry.register("olmo3:dev:1b:mini")
class Olmo3Dev1bMiniGroup(BaseNamedTasksGroup):
    tasks = [
        re.compile(r"^arc_challenge:rc.*"),
        re.compile(r"^arc_easy:rc.*"),
        re.compile(r"^codex_humaneval.*3shot"),
        re.compile(r"^hellaswag:rc.*olmes"),
        re.compile(r"^mbpp.*3shot"),
        re.compile(r"^minerva$"),
        re.compile(r"^mt_mbpp_v2fix$"),
        re.compile(r"^winogrande:rc.*olmes"),
        BasicRCGroup(),
        CoreRCGroup(),
        MMLURCGroup(),
    ]
