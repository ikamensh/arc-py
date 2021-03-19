import os

from read import parse_dir

here = os.path.dirname(__file__)

train_dir = os.path.join(here, "data", "training")
eval_dir = os.path.join(here, "data", "evaluation")

TRAIN_SET = parse_dir(train_dir)
EVAL_SET = parse_dir(eval_dir)


# for i, prob in enumerate(TRAIN_SET):
#     if not i % 100:
#         for case in prob.train_pairs:
#             case.plot()


def describe(lst):
    print(len(lst))

    number_examples = [len(x.train_pairs) for x in lst]
    number_queries = [len(x.test_pairs) for x in lst]

    from collections import Counter

    print("Train:")
    print(Counter(number_examples))
    print(Counter(number_queries))

print("\nTrain")
describe(TRAIN_SET)

print("\nEval")
describe(EVAL_SET)
