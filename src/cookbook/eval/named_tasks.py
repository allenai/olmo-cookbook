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
    tasks: ClassVar[list[Union[str, "BaseNamedTasksGroup"]]] = []

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @cached_property
    def expanded_tasks(self) -> list[str]:
        """
        Return the list of tasks in this group, expanding any nested named groups.
        """
        expanded_tasks: list[str] = []
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
    def combine(self, results: MiniFrame) -> MiniFrame:
        # filter results by task names; if we have no results at all, we create a new table.
        filtered = results.keep_cols(*self.expanded_tasks) or MiniFrame(title=results.title)

        # it can be the case that some columns (tasks) have no scores for all models. in that case
        # we iterate over all task names in this named group and add a None value for each model.
        for row in results.rows:
            for metric in self.expanded_tasks:
                filtered.add(col=metric, row=row.name, val=None, overwrite=False)

        # the following code computes the average of the scores for each model.
        # the average is set to None if there are either no scores or if there are missing scores.
        # otherwise, the average is the sum of the scores divided by the number of scores.
        for row in filtered.rows:
            average: float | None = (
                (sum(row.values) / len(row.values))         # pyright: ignore
                if all(s is not None for s in row.values) and len(row.values) > 0
                else None
            )
            filtered.add(col=self.name, row=row.name, val=average)

        return filtered


class BaseAverageOfAveragesNamedTasksGroup(BaseAverageNamedTasksGroup):
    """
    Base class for named tasks groups that include averages of other task groups
    (e.g., a macro average over all QA tasks, which includes MMLU)
    """

    def combine(self, results: MiniFrame) -> MiniFrame:
        filtered_rows = MiniFrame(title=results.title)

        # calculate the averages for all child task groups
        child_task_names: list[str] = []
        for task_or_named_group in self.tasks:
            if isinstance(task_or_named_group, BaseAverageNamedTasksGroup):
                # get task groups (e.g., MMLURCGroup())
                combined_table = task_or_named_group.combine(results)
                # If the named group is able to average all scores, add it!
                named_group_col = combined_table.keep_cols(*[task_or_named_group.name])
                child_task_names.append(task_or_named_group.name)
                filtered_rows = filtered_rows + named_group_col
            elif isinstance(task_or_named_group, str):
                # get individual tasks (e.g., "arc_challenge:rc::olmes")
                task_col = results.keep_cols(*[task_or_named_group])
                child_task_names.append(task_or_named_group)
                filtered_rows = filtered_rows + task_col
            else:
                raise TypeError(f"Task type not yet supported: {type(task_or_named_group)}.")

        # Any tasks that do not exist for all models, add a "None" entry
        for row in results.rows:
            for task in child_task_names:
                if task not in filtered_rows or filtered_rows[(task, row.name)] is None:
                    filtered_rows.add(col=task, row=row.name, val=None)

        # compute the average of averages
        # each row here is a model
        for row in list(filtered_rows.rows):
            # we compute the average of the scores for this model; we set the average to None if
            # there are missing scores or if there are no scores at all.
            average: float | None = None
            if len(row.values) > 0 and all(s is not None for s in row.values):
                filtered_scores = [s for s in row.values if s is not None]
                average = (sum(filtered_scores) / len(filtered_scores)) if filtered_scores else 0.0

            # we add the average to the combined table
            filtered_rows.add(col=self.name, row=row.name, val=average)

        return filtered_rows


class BaseNamedTasksWithNoAverageGroup(BaseAverageOfAveragesNamedTasksGroup):
    """
    Base class for tasks "views". In a task view, only the child tasks are averages

    For example, "olmo3:dev:7b:main" is not a average,
    but contains "olmo3:dev:7b:mcqa" and "mmlu:mc" are task averages.
    """
    def combine(self, results: MiniFrame) -> MiniFrame:
        # Compute all the task averages;
        out_table = super().combine(results)

        # Remove task view as a column in the table
        out_table = out_table.drop_cols(self.name)

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


@NamedTasksGroupRegistry.register("mmlu_stem:mc")
class MMLUStemMCGroup(BaseAverageNamedTasksGroup):
    tasks = [f"mmlu_{category}:mc::olmes" for category in constants.MMLU_SUBCATEGORIES["stem"]]


@NamedTasksGroupRegistry.register("mmlu_humanities:mc")
class MMLUHumanitiesMCGroup(BaseAverageNamedTasksGroup):
    tasks = [f"mmlu_{category}:mc::olmes" for category in constants.MMLU_SUBCATEGORIES["humanities"]]


@NamedTasksGroupRegistry.register("mmlu_social_sciences:mc")
class MMLUSocialSciencesMCGroup(BaseAverageNamedTasksGroup):
    tasks = [f"mmlu_{category}:mc::olmes" for category in constants.MMLU_SUBCATEGORIES["social_sciences"]]


@NamedTasksGroupRegistry.register("mmlu_other:mc")
class MMLUOtherMCGroup(BaseAverageNamedTasksGroup):
    tasks = [f"mmlu_{category}:mc::olmes" for category in constants.MMLU_SUBCATEGORIES["other"]]


@NamedTasksGroupRegistry.register("mmlu:cot::hamish_zs_reasoning")
class MMLUHamishZSReasoningGroup(BaseAverageNamedTasksGroup):
    tasks = [f"{category}:cot::hamish_zs_reasoning" for category in constants.MMLU_CATEGORIES]


@NamedTasksGroupRegistry.register("mmlu:cot::olmo3:thinker")
class MMLUOLMo3ThinkerGroup(BaseAverageNamedTasksGroup):
    tasks = [f"{category}:cot::olmo3:thinker" for category in constants.MMLU_CATEGORIES]


@NamedTasksGroupRegistry.register("mmlu:cot::olmo3:midtrain")
class MMLUMidtrainGroup(BaseAverageNamedTasksGroup):
    tasks = [f"{category}:cot::olmo3:midtrain" for category in constants.MMLU_CATEGORIES]


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


@NamedTasksGroupRegistry.register("arc:bpb::full")
class ARCBPBFullGroup(BaseAverageNamedTasksGroup):
    tasks = [f"{category}:bpb::olmes:full" for category in constants.ARC_TASKS]


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


@NamedTasksGroupRegistry.register("basic:bpb")
class BasicBpbGroup(BaseAverageNamedTasksGroup):
    tasks = [f"{task}:bpb::olmes" for task in constants.BASIC_SKILLS]


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


@NamedTasksGroupRegistry.register("minerva:n4:v2")
class MinervaN4V2Group(BaseAverageNamedTasksGroup):
    tasks = [f"{subtask}::olmes:n4:v2" for subtask in constants.ALL_MINERVA_TASKS]


@NamedTasksGroupRegistry.register("minerva::hamish_zs_reasoning")
class MinervaHamishZSReasoningGroup(BaseAverageNamedTasksGroup):
    tasks = [f"{subtask}::hamish_zs_reasoning" for subtask in constants.ALL_MINERVA_TASKS]


@NamedTasksGroupRegistry.register("minerva_math::olmo3:midtrain")
class MinervaMidtrainReasoningGroup(BaseAverageNamedTasksGroup):
    tasks = [f"{subtask}::olmo3:midtrain" for subtask in constants.ALL_MINERVA_TASKS]


@NamedTasksGroupRegistry.register("deepmind_math::olmo3:heldout")
class DeepmindMathHeldoutGroup(BaseAverageNamedTasksGroup):
    tasks = [f"deepmind_math_{cat}::olmo3:heldout" for cat in constants.DEEPMIND_MATH_CATEGORIES]


@NamedTasksGroupRegistry.register("math")
class MathGroup(BaseAverageOfAveragesNamedTasksGroup):
    tasks = [
        "gsm8k::olmo1",
        "gsm8k::olmes",
        MinervaGroup()
    ]


@NamedTasksGroupRegistry.register("gsm-symb")
class GsmSymbGroup(BaseAverageNamedTasksGroup):
    tasks = [task for task in constants.ALL_GSM_SYMB_TASKS]


@NamedTasksGroupRegistry.register("gsm-symb:n8:v2")
class GsmSymbN8V2Group(BaseAverageNamedTasksGroup):
    tasks = [f'{task}:n8:v2' for task in constants.ALL_GSM_SYMB_TASKS]


@NamedTasksGroupRegistry.register("gsm-symb:n8:v2:pass_at_4")
class GsmSymbN8V2PassAt4Group(BaseAverageNamedTasksGroup):
    tasks = [f'{task}:n8:v2:pass_at_4' for task in constants.ALL_GSM_SYMB_TASKS]


@NamedTasksGroupRegistry.register("code")
class CodeGroup(BaseAverageNamedTasksGroup):
    tasks = [task for task in constants.ALL_CODEX_TASKS]


@NamedTasksGroupRegistry.register("agi_eval")
class AgiEvalGroup(BaseAverageNamedTasksGroup):
    tasks = [f"{task}:1shot::olmes" for task in constants.AGI_EVAL_ENGLISH_TASKS]


@NamedTasksGroupRegistry.register("agi_eval_english:0shot_cot::olmo3:thinker")
class AgiEvalEnglishOLMo3ThinkerGroup(BaseAverageNamedTasksGroup):
    tasks = [f"agi_eval_{task}:0shot_cot::olmo3:thinker" for task in constants.AGI_EVAL_ENGLISH_TASKS]


@NamedTasksGroupRegistry.register("agi_eval_english::olmo3:midtrain")
class AgiEvalEnglishMidtrainGroup(BaseAverageNamedTasksGroup):
    tasks = [f"agi_eval_{task}::olmo3:midtrain" for task in constants.AGI_EVAL_ENGLISH_TASKS]


@NamedTasksGroupRegistry.register("starcoder")
class StarcoderGroup(BaseAverageNamedTasksGroup):
    tasks = [task for task in constants.STARCODER_CODEX_TASKS]


@NamedTasksGroupRegistry.register("starcoder::pass@1")
class StarcoderPassAt1Group(BaseAverageNamedTasksGroup):
    tasks = [task for task in constants.STARCODER_PASS_AT_1_TASKS]


@NamedTasksGroupRegistry.register("code-no-bcb")
class CodeNoBcbGroup(BaseNamedTasksGroup):
    tasks = [task for task in constants.ALL_CODEX_TASKS if "bigcodebench" not in task]


@NamedTasksGroupRegistry.register("multipl-e-humaneval:n32:v2")
class MultiPlEHEN32V2Group(BaseAverageNamedTasksGroup):
    tasks = [f'{task}:n32:v2' for task in constants.MULTIPL_E_HE_TASKS]


@NamedTasksGroupRegistry.register("multipl-e-mbpp:n32:v2")
class MultiPlEMBPPN32V2Group(BaseAverageNamedTasksGroup):
    tasks = [f'{task}:n32:v2' for task in constants.MULTIPL_E_MBPP_TASKS]


@NamedTasksGroupRegistry.register("multipl-e-humaneval:n32:v2:pass_at_16")
class MultiPlEHEN32V2PassAt16Group(BaseAverageNamedTasksGroup):
    tasks = [f'{task}:n32:v2:pass_at_16' for task in constants.MULTIPL_E_HE_TASKS]


@NamedTasksGroupRegistry.register("multipl-e-mbpp:n32:v2:pass_at_16")
class MultiPlEMBPPN32V2PassAt16Group(BaseAverageNamedTasksGroup):
    tasks = [f'{task}:n32:v2:pass_at_16' for task in constants.MULTIPL_E_MBPP_TASKS]


@NamedTasksGroupRegistry.register("fim::olmo3")
class FimOLMo3Group(BaseAverageNamedTasksGroup):
    tasks = [f'{task}::olmo3' for task in constants.FIM_TASKS]


@NamedTasksGroupRegistry.register("crux-eval")
class CruxEvalGroup(BaseAverageNamedTasksGroup):
    tasks = [task for task in constants.CRUX_EVAL_TASKS]


@NamedTasksGroupRegistry.register("mt_mbpp_v2fix")
class MtMbppV2fixGroup(BaseAverageNamedTasksGroup):
    tasks = [task for task in constants.MULTILINGUAL_MBPP_TASKS_V2]


@NamedTasksGroupRegistry.register("bbh:cot::olmo3:thinker")
class BBHOLMo3ThinkerGroup(BaseAverageNamedTasksGroup):
    tasks = [f"bbh_{category}:cot::olmo3:thinker" for category in constants.BBH_TASKS]


@NamedTasksGroupRegistry.register("bbh:cot::olmo3:midtrain")
class BBHMidtrainThinkerGroup(BaseAverageNamedTasksGroup):
    tasks = [f"bbh_{category}:cot::olmo3:midtrain" for category in constants.BBH_TASKS]


@NamedTasksGroupRegistry.register("bbh:cot::olmo3:heldout")
class BBHHeldoutGroup(BaseAverageNamedTasksGroup):
    tasks = [f"bbh_{category}:cot::olmo3:heldout" for category in constants.BBH_TASKS]


@NamedTasksGroupRegistry.register("ifeval_mt::tulu-thinker")
class IFEvalMTThinkerGroup(BaseAverageNamedTasksGroup):
    tasks = [f"ifeval_mt_{task_type}::tulu-thinker" for task_type in constants.IFEVAL_MT_TASKS]


@NamedTasksGroupRegistry.register("multiturn_alpacaeval::tulu")
class AlpacaEvalMTGroup(BaseAverageNamedTasksGroup):
    tasks = [f"multiturn_alpacaeval_{task_type}::tulu" for task_type in constants.MULTITURN_ALPACAEVAL_TASKS]


@NamedTasksGroupRegistry.register("styled_popqa::tulu-thinker")
class StyledPopQAThinkerGroup(BaseAverageNamedTasksGroup):
    tasks = [f"styled_popqa_{task_type}::tulu-thinker" for task_type in constants.STYLED_TASKS_POPQA]


@NamedTasksGroupRegistry.register("styled_math500::tulu-thinker")
class StyledMath500ThinkerGroup(BaseAverageNamedTasksGroup):
    tasks = [f"styled_math500_{task_type}::tulu-thinker" for task_type in constants.STYLED_TASKS]


@NamedTasksGroupRegistry.register("styled_alpacaeval::tulu-thinker")
class StyledAlpacaEvalThinkerGroup(BaseAverageNamedTasksGroup):
    tasks = []
    for task_type in constants.STYLED_TASKS:
        for reference_set in ["og", "new"]:
            tasks += [f"styled_alpacaeval_{task_type}_{reference_set}_ref::tulu-thinker"]


@NamedTasksGroupRegistry.register("omega:0-shot-chat")
class Omega0ShotCoTGroup(BaseAverageNamedTasksGroup):
    tasks = []
    for broad_cate in constants.OMEGA_SUB_CATEGORIES:
        if broad_cate == "explorative":
            target_splits = ["test_in", "test_out"]
        else:
            target_splits = ["test"]
        for sub_cate in constants.OMEGA_SUB_CATEGORIES[broad_cate]:
            for target_split in target_splits:
                tasks += [f"omega_{broad_cate}_{sub_cate}_{target_split}:0-shot-chat"]


@NamedTasksGroupRegistry.register("omega::olmo3:midtrain")
class OmegaMidtrainGroup(BaseAverageNamedTasksGroup):
    tasks = []
    for broad_cate in constants.OMEGA_SUB_CATEGORIES:
        if broad_cate == "explorative":
            target_splits = ["test_in", "test_out"]
        else:
            target_splits = ["test"]
        for sub_cate in constants.OMEGA_SUB_CATEGORIES[broad_cate]:
            for target_split in target_splits:
                tasks += [f"omega_{broad_cate}_{sub_cate}_{target_split}::olmo3:midtrain"]


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


def make_ruler_group(ruler_length: int) -> Type[BaseAverageNamedTasksGroup]:
    class RULERGroup(BaseAverageNamedTasksGroup):
        tasks = [
            task
            for group_name, tasks in constants.RULER_SUITES.items()
            for task in tasks
            if group_name.endswith(f"__{ruler_length}::suite") and not group_name.startswith("ruler_all")
        ]

    return RULERGroup


for ruler_length in (int(2**i) for i in range(12, 18)):
    NamedTasksGroupRegistry.register(f"ruler:{ruler_length // 2 ** 10}k")(make_ruler_group(ruler_length))

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


@NamedTasksGroupRegistry.register("olmo3:dev:1b:qa:bpb")
class Olmo3Dev1bQaBpbGroup(BaseAverageOfAveragesNamedTasksGroup):
    tasks = [
        # Core OLMES
        ARCBPBFullGroup(),
        MMLUBpbGroup(),
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

        # New OLMo 3
        "sciq:bpb::olmo3",
        "qasper_yesno:bpb::olmes",
        BasicBpbGroup(),
        "lab_bench_dbqa:bpb",
        "lab_bench_protocolqa:bpb",
        "lambada:bpb",
        "medmcqa:bpb::none",
        "medqa_en:bpb::none",
        "sciriff_yesno:bpb::olmes",
    ]

@NamedTasksGroupRegistry.register("olmo3:dev:1b:qa:bpb:v2")
class Olmo3Dev1bQaBpbV2Group(BaseAverageOfAveragesNamedTasksGroup):
    tasks = [
        # Core OLMES
        ARCBPBFullGroup(),
        MMLUBpbGroup(),
        "csqa:bpb::olmes:full",
        "hellaswag:bpb::olmes:full",
        "winogrande:bpb::olmes:full",
        "socialiqa:bpb::olmes:full",
        "piqa:bpb::olmes:full",

        # Gen OLMES
        "coqa:bpb::gen2mc:xlarge",
        "drop:bpb::gen2mc:xlarge",
        "jeopardy:bpb::gen2mc:xlarge",
        "naturalqs:bpb::gen2mc:xlarge",
        "squad:bpb::gen2mc:xlarge",

        # New OLMo 3
        "sciq:bpb::olmo3",
        "qasper_yesno:bpb::olmes",
        BasicBpbGroup(),
        "lab_bench_dbqa:bpb",
        "lab_bench_protocolqa:bpb",
        "lambada:bpb",
        "medmcqa:bpb::none",
        "medqa_en:bpb::none",
        "sciriff_yesno:bpb::olmes",
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
        BasicRCGroup(),
        "lab_bench_dbqa",
        "lab_bench_protocolqa",
        "lambada",
        "medmcqa:rc::none",
        "medqa_en:rc::none",
        "sciriff_yesno:rc::olmes",
    ]


@NamedTasksGroupRegistry.register("olmo3:dev:1b:qa:rc:v2")
class Olmo3Dev1bQaRcV2Group(BaseAverageOfAveragesNamedTasksGroup):
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
        "coqa:rc::gen2mc:xlarge",
        "drop:rc::gen2mc:xlarge",
        "jeopardy:rc::gen2mc:xlarge",
        "naturalqs:rc::gen2mc:xlarge",
        "squad:rc::gen2mc:xlarge",

        # New OLMo 3
        "sciq:rc::olmo3",
        "qasper_yesno:rc::olmes",
        BasicRCGroup(),
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
        # QA
        Olmo3Dev1bQaBpbGroup(),

        # Math
        MinervaBpbGroup(),

        # Code
        Olmo3Dev1bCodeBpbGroup(),
    ]

@NamedTasksGroupRegistry.register("olmo3:dev:7b:math:v2")
class Olmo3Dev7bMathV2Group(BaseAverageOfAveragesNamedTasksGroup):
    tasks = [
        # Math
        "gsm8k::olmo3:n8:v2",
        GsmSymbN8V2Group(),
        MinervaN4V2Group(),
    ]


@NamedTasksGroupRegistry.register("olmo3:dev:7b:code_gen:v2")
class Olmo3Dev7bCodeGenV2Group(BaseAverageOfAveragesNamedTasksGroup):
    tasks = [
        "bigcodebench:3shot::olmo3:v2",
        "codex_humaneval:3shot::olmo3:n32:v2",
        "deepseek_leetcode::olmo3:n32:v2",
        "ds1000:3shot::olmo3:v2",
        "mbpp:3shot::olmo3:n32:v2",
        MultiPlEHEN32V2Group(),
        MultiPlEMBPPN32V2Group(),
    ]


@NamedTasksGroupRegistry.register("olmo3:dev:7b:code_gen_mini:v2:n32:pass_at_16")
class Olmo3Dev7bCodeGenMiniV2N32PassAt16Group(BaseAverageOfAveragesNamedTasksGroup):
    tasks = [
        # We only use a subset of code gen benchmarks for pass@k for speed
        "deepseek_leetcode::olmo3:n32:v2:pass_at_16",
        "codex_humaneval:3shot::olmo3:n32:v2:pass_at_16",
        "mbpp:3shot::olmo3:n32:v2:pass_at_16",
        MultiPlEHEN32V2PassAt16Group(),
        MultiPlEMBPPN32V2PassAt16Group(),
    ]


@NamedTasksGroupRegistry.register("olmo3:dev:7b:code_fim")
class Olmo3Dev7bCodeFimGroup(BaseAverageOfAveragesNamedTasksGroup):
    tasks = [
        # Code
        FimOLMo3Group(),
    ]


@NamedTasksGroupRegistry.register("olmo3:dev:7b:gen")
class Olmo3Dev7bGenGroup(BaseAverageOfAveragesNamedTasksGroup):
    tasks = [
        "hellaswag:rc::xlarge",
        "winogrande:rc::xlarge",
        "lambada",
        BasicRCGroup(),
        "drop::xlarge",
        "jeopardy::xlarge",
        "naturalqs::xlarge",
        "squad::xlarge",
        "coqa::xlarge",
    ]


@NamedTasksGroupRegistry.register("olmo3:dev:7b:mcqa:stem")
class Olmo3Dev7bMcqaSTEMGroup(BaseAverageOfAveragesNamedTasksGroup):
    tasks = [
        ARCMCXlargeGroup(),
        MMLUStemMCGroup(),
        "medmcqa:mc::none",
        "medqa_en:mc::none",
        "sciq:mc::xlarge",
        # "lab_bench_dbqa:mc", # too noisy to include in macro-average
        # "lab_bench_protocolqa:mc", # too noisy to include in macro-average
    ]


@NamedTasksGroupRegistry.register("olmo3:dev:7b:mcqa:non_stem")
class Olmo3Dev7bMcqaNonSTEMGroup(BaseAverageOfAveragesNamedTasksGroup):
    tasks = [
        MMLUHumanitiesMCGroup(),
        MMLUSocialSciencesMCGroup(),
        MMLUOtherMCGroup(),
        "csqa:mc::xlarge",
        "piqa:mc::xlarge",
        "socialiqa:mc::xlarge",
        "coqa:mc::gen2mc",
        "drop:mc::gen2mc",
        "jeopardy:mc::gen2mc",
        "naturalqs:mc::gen2mc",
        "squad:mc::gen2mc",
    ]


@NamedTasksGroupRegistry.register("olmo3:dev:7b:mcqa:non_stem:v2")
class Olmo3Dev7bMcqaNonSTEMV2Group(BaseAverageOfAveragesNamedTasksGroup):
    tasks = [
        MMLUHumanitiesMCGroup(),
        MMLUSocialSciencesMCGroup(),
        MMLUOtherMCGroup(),
        "csqa:mc::xlarge",
        "piqa:mc::xlarge",
        "socialiqa:mc::xlarge",
        "coqa:mc::gen2mc:xlarge",
        "drop:mc::gen2mc:xlarge",
        "jeopardy:mc::gen2mc:xlarge",
        "naturalqs:mc::gen2mc:xlarge",
        "squad:mc::gen2mc:xlarge",
    ]


# # # # # # # # # # # # # # # DISPLAY TASK GROUPS # # # # # # # # # # # # # # # # #
#  These are just shortcuts to display many metrics at once. no need to average.  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@NamedTasksGroupRegistry.register("olmo2:paper")
class Olmo2PaperGroup(BaseNamedTasksWithNoAverageGroup):
    tasks = [
        "arc_challenge:rc::olmes",
        "arc_challenge:mc::olmes",
        "hellaswag:rc::olmes",
        "hellaswag:mc::olmes",
        "winogrande:rc::olmes",
        "winogrande:mc::olmes",
        "naturalqs::olmes",
        "drop::olmes",
        AgiEvalGroup(),
        "gsm8k::olmes",
        MMLUMCGroup(),
        MMLURCGroup(),
        CoreMCGroup(),
        MMLUProMCGroup(),
        "triviaqa::olmes"
    ]


@NamedTasksGroupRegistry.register("olmo2:dev:7b")
class Olmo2Dev7bGroup(BaseNamedTasksWithNoAverageGroup):
    tasks = [
        "arc_challenge:mc::olmes",
        "arc_easy:mc::olmes",
        "hellaswag:mc::olmes",
        "naturalqs::olmes",
        "gsm8k::olmo1",
        MMLUMCGroup(),
        CoreMCGroup(),
        GenGroup(),
    ]


@NamedTasksGroupRegistry.register("olmo2:dev:1b")
class Olmo2Dev1bGroup(BaseNamedTasksWithNoAverageGroup):
    tasks = [
        "arc_challenge:rc::olmes",
        "arc_easy:rc::olmes",
        "hellaswag:rc::olmes",
        "gsm8k::olmo1",
        MMLURCGroup(),
        CoreRCGroup(),
    ]


@NamedTasksGroupRegistry.register("olmo3:dev:1b:main")
class Olmo3Dev1bMainGroup(BaseNamedTasksWithNoAverageGroup):
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
        "codex_humaneval:3shot:bpb::none",
        "mbpp:3shot:bpb::none",
        MinervaBpbGroup()
    ]


@NamedTasksGroupRegistry.register("olmo3:dev:1b:main:hf")
class Olmo3Dev1bMainHFGroup(BaseNamedTasksWithNoAverageGroup):
    tasks = [
        "ultrachat_masked_ppl",
        "wildchat_masked_ppl",
    ]


@NamedTasksGroupRegistry.register("olmo3:dev:7b:main:v2")
class Olmo3Dev7bV2MainGroup(BaseNamedTasksWithNoAverageGroup):
    tasks = [
        Olmo3Dev7bMcqaSTEMGroup(),
        Olmo3Dev7bMcqaNonSTEMGroup(),
        Olmo3Dev7bGenGroup(),
        Olmo3Dev7bMathV2Group(),
        Olmo3Dev7bCodeGenV2Group(),
        Olmo3Dev7bCodeGenMiniV2N32PassAt16Group(),
        Olmo3Dev7bCodeFimGroup(),
        ARCMCXlargeGroup(),
        MMLUMCGroup(),
        GenXlargeGroup(),
        BasicRCGroup(),
        "gsm8k::olmo3:n8:v2",
        GsmSymbN8V2Group(),
        GsmSymbN8V2PassAt4Group(),
        MinervaN4V2Group(),
        "minerva_math_500::olmo3:n32:v2",
        "minerva_math_500::olmo3:n32:v2:pass_at_16",
        "codex_humaneval:3shot::olmo3:n32:v2",
        "mbpp:3shot::olmo3:n32:v2",
        MultiPlEHEN32V2Group(),
        MultiPlEMBPPN32V2Group(),
        CruxEvalGroup(),
    ]


@NamedTasksGroupRegistry.register("olmo3:dev:midtrain:v1")
class Olmo3DevMidtrainV1MainGroup(BaseNamedTasksWithNoAverageGroup):
    tasks = [
        # Everything in this task set is 0-shot
        "ifeval::hamish_zs_reasoning",
        StyledMath500ThinkerGroup(),
        "gsm8k::zs_cot_latex",
        MinervaHamishZSReasoningGroup(),
        "minerva_math_500::hamish_zs_reasoning",
        "aime::hamish_zs_reasoning",
        Omega0ShotCoTGroup(),
        "codex_humanevalplus:0-shot-chat::tulu-thinker",
        "mbppplus:0-shot-chat::tulu-thinker",
        "livecodebench_codegeneration::tulu-thinker",
        BBHOLMo3ThinkerGroup(),
        "zebralogic::hamish_zs_reasoning",
        "gpqa:0shot_cot::olmo3:thinker",
        "popqa::olmo3:thinker",
        AgiEvalEnglishOLMo3ThinkerGroup(),
        MMLUOLMo3ThinkerGroup(),
    ]


@NamedTasksGroupRegistry.register("olmo3:dev:midtrain:v2")
class Olmo3DevMidtrainV2MainGroup(BaseNamedTasksWithNoAverageGroup):
    tasks = [
        "ifeval::olmo3:midtrain",
        "gsm8k::olmo3:midtrain",
        MinervaMidtrainReasoningGroup(),
        "minerva_math_500::olmo3:midtrain",
        "aime:2024::olmo3:midtrain",
        "aime:2025::olmo3:midtrain",
        "omega_500::olmo3:midtrain",
        OmegaMidtrainGroup(),
        "codex_humanevalplus::olmo3:midtrain",
        "mbppplus::olmo3:midtrain",
        "livecodebench_codegeneration::olmo3:midtrain",
        BBHMidtrainThinkerGroup(),
        "gpqa::olmo3:midtrain",
        "zebralogic::olmo3:midtrain",
        "popqa::olmo3:midtrain",
        AgiEvalEnglishMidtrainGroup(),
        MMLUMidtrainGroup(),
    ]


@NamedTasksGroupRegistry.register("olmo3:base_heldout")
class Olmo3BaseHeldoutGroup(BaseNamedTasksWithNoAverageGroup):
    tasks = [
        BBHHeldoutGroup(),
        MMLUProMCGroup(),
        DeepmindMathHeldoutGroup(),
        "lbpp::olmo3",
    ]


@NamedTasksGroupRegistry.register("olmo3:paper")
class Olmo3PaperGroup(BaseNamedTasksWithNoAverageGroup):
    tasks = [
        # olmo3:base_easy
        Olmo3Dev1bMathBpbGroup(),
        Olmo3Dev1bCodeBpbGroup(),
        Olmo3Dev1bQaBpbV2Group(),
        Olmo3Dev1bQaRcV2Group(),

        # olmo3:base
        Olmo3Dev7bMcqaSTEMGroup(),
        Olmo3Dev7bMcqaNonSTEMV2Group(),
        Olmo3Dev7bGenGroup(),
        Olmo3Dev7bMathV2Group(),
        Olmo3Dev7bCodeGenV2Group(),
        Olmo3Dev7bCodeFimGroup(),

        # olmo3:base_chat
        Olmo3DevMidtrainV2MainGroup(),

        # olmo3:heldout
        Olmo3BaseHeldoutGroup(),
    ]