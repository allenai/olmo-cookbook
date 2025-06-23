from abc import ABC
import re
from typing import Callable, ClassVar, Generic, Iterable, Type, TypeVar, Union

from cookbook import constants


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
        def decorator(task: T, _task_name: str = task_name) -> T:
            cls._named_tasks[_task_name] = task
            return task

        return decorator

    @classmethod
    def get(cls, task_name: str) -> Type["BaseNamedTasksGroup"]:
        assert cls._instance is not None, "NamedTasksGroupRegistry is not initialized"

        if task_name not in cls._named_tasks:
            raise ValueError(f"Task {task_name} not found")
        return cls._named_tasks[task_name]


class BaseNamedTasksGroup(ABC):
    """
    Base class for named tasks.

    Subclasses should define the `tasks` class variable.
    """

    # external, to be defined by subclasses
    tasks: ClassVar[list[str | re.Pattern | "BaseNamedTasksGroup"]] = []

    def __init__(self):
        assert isinstance(self.tasks, list), "tasks must be a list"
        assert len(self.tasks) > 0, "tasks must not be empty"

        # expand any groups in the tasks list
        i = 0
        while i < len(self.tasks):
            task = self.tasks[i]
            if isinstance(task, BaseNamedTasksGroup):
                self.tasks.pop(i)
                for sub_task in task.tasks[::-1]:
                    self.tasks.insert(i, sub_task)
            else:
                i += 1

    def filter(self, task_names: Iterable[str]) -> list[str]:
        """
        Given a list of task names, filters it to the ones that are in the group.
        """
        known_tasks_names = []
        for received_task_name in task_names:
            for known_task_name in self.tasks:
                if not isinstance(known_task_name, (str, re.Pattern)):
                    # check if expansion during init went wrong
                    raise RuntimeError(
                        f"Tasks are str or re.Pattern after init {known_task_name} is {type(known_task_name)}! "
                        "This should not happen, please report this bug."
                    )
                elif isinstance(known_task_name, str) and known_task_name == received_task_name:
                    # we are dealing with a simple string, so we can just compare them
                    known_tasks_names.append(known_task_name)
                    break

                elif isinstance(known_task_name, re.Pattern) and known_task_name.search(received_task_name):
                    # search for a match anywhere in the received task name
                    known_tasks_names.append(known_task_name)
                    break

        return known_tasks_names

    def get(self) -> list[str]:
        """Return the list of tasks that are strings"""
        return [task for task in self.tasks if isinstance(task, str)]

    def combine(self, tasks_metrics_dict: dict[str, float]) -> float | None:
        """
        Combine the metrics of the tasks in the group into a single score.
        If this function returns None, it means that this group of metrics is not supposed to be averaged.
        """
        return None


class BaseAverageNamedTasksGroup(BaseNamedTasksGroup):
    """
    Base class for named tasks groups that are supposed to be averaged.
    """

    def combine(self, tasks_metrics_dict: dict[str, float]) -> float | None:
        filtered_names = self.filter(tasks_metrics_dict.keys())
        if len(filtered_names) == 0:
            return None
        return sum(tasks_metrics_dict[name] for name in filtered_names) / len(filtered_names)


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
