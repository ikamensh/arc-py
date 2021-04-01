from .data import train_problems, validation_problems, describe_task_group, train_data_dir, valid_data_dir
from .types import ArcIOPair, ArcProblem
from .consts import ArcColors, colors_rgb
from . import agents
from . import types
from .evaluation import ArcEvaluationResult, evaluate_agent
from .plot import plot_grid

_train_probs_by_id = {p.uid: p for p in train_problems}
_valid_probs_by_id = {p.uid: p for p in train_problems}


def get_train_problem_by_uid(uid: str) -> ArcProblem:
    return _train_probs_by_id[uid]


def get_validation_problem_by_uid(uid: str) -> ArcProblem:
    return _valid_probs_by_id[uid]
