import os
import pickle

import appdirs

from arc.version import __version__
from arc.read import parse_dir

here = os.path.dirname(__file__)
train_dir = os.path.join(here, "original", "ARC", "data", "training")
eval_dir = os.path.join(here, "original", "ARC", "data", "evaluation")
cache_dir = appdirs.user_cache_dir("arc-py", "ikamensh")
cache_file = os.path.join(cache_dir, f"arc-py.{__version__}.cache")

if os.path.isfile(cache_file):
    with open(cache_file, "rb") as f:
        train_set, eval_set = pickle.load(f)
else:
    train_set = parse_dir(train_dir)
    eval_set = parse_dir(eval_dir)
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump((train_set, eval_set), f)

assert (
    len(train_set) > 0
), "Training set is empty - this distribution of arc utils is broken."
assert (
    len(eval_set) > 0
), "Evaluation set is empty - this distribution of arc utils is broken."


def describe_task_set(arc_task_set):

    number_examples = [len(x.train_pairs) for x in arc_task_set]
    number_queries = [len(x.test_pairs) for x in arc_task_set]

    from collections import Counter

    print("\nDescribing an ARC task set.")
    print("Number of tasks:", len(arc_task_set))
    print("Number of demonstrations per task:", Counter(number_examples))
    print("Number of tests per task:", Counter(number_queries))


if __name__ == "__main__":

    print("\nTrain")
    describe_task_set(train_set)

    print("\nEval")
    describe_task_set(eval_set)
