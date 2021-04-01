import os
import pickle

import appdirs

from arc.version import __version__
from arc.read import parse_dir

here = os.path.dirname(__file__)
json_sources_folder = os.path.join(here, "original", "ARC", "data")
train_data_dir = os.path.join(json_sources_folder, "training")
valid_data_dir = os.path.join(json_sources_folder, "evaluation")
cache_dir = appdirs.user_cache_dir("arc-py", "ikamensh")
cache_file = os.path.join(cache_dir, f"arc-py.{__version__}.cache")

if os.path.isfile(cache_file):
    with open(cache_file, "rb") as f:
        train_problems, validation_problems = pickle.load(f)
else:
    train_problems = parse_dir(train_data_dir)
    validation_problems = parse_dir(valid_data_dir)
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump((train_problems, validation_problems), f)

assert (
        len(train_problems) > 0
), "Training set is empty - this distribution of arc utils is broken."
assert (
        len(validation_problems) > 0
), "Evaluation set is empty - this distribution of arc utils is broken."


def describe_task_group(arc_task_set):

    number_examples = [len(x.train_pairs) for x in arc_task_set]
    number_queries = [len(x.test_pairs) for x in arc_task_set]

    from collections import Counter

    print("\nDescribing an ARC task set.")
    print("Number of tasks:", len(arc_task_set))
    print("Number of demonstrations per task:", Counter(number_examples))
    print("Number of tests per task:", Counter(number_queries))


if __name__ == "__main__":

    print("\nTrain")
    describe_task_group(train_problems)

    print("\nEval")
    describe_task_group(validation_problems)

    for task in train_problems:
        for pair in task.train_pairs:
            pair.plot()
            break
        break
