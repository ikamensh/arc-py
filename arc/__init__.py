from .data import train_set, eval_set, describe_task_set
from .types import ArcIOPair, ArcProblem
from .consts import ArcColors, colors_rgb
from . import agents

_train_probs_by_id = {p.uid : p for p in train_set}
_valid_probs_by_id = {p.uid : p for p in train_set}

def get_train_problem_by_uid(uid: str) -> ArcProblem:
    return _train_probs_by_id[uid]

def get_validation_problem_by_uid(uid: str) -> ArcProblem:
    return _valid_probs_by_id[uid]
