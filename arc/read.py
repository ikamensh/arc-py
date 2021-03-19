import os
import json
from typing import List

import numpy as np

from arc.data_model import ArcIOPair, ArcProblem


def parse_json_grid(grid: List[List[int]]) -> np.ndarray:
    rows = []
    for row in grid:
        rows.append(np.array(row))

    return np.vstack(rows)


def parse_dir(d) -> List[ArcProblem]:
    cases = []
    for file in os.listdir(d):
        path = os.path.join(d, file)
        with open(path) as f:
            parsed = json.load(f)

            train = parsed["train"]
            train_pairs = parse_group(train)

            test = parsed["test"]
            test_pairs = parse_group(test)

            cases.append(ArcProblem(train_pairs, test_pairs))
    return cases


def parse_group(group: List[dict]):
    pairs = []
    for pair in group:
        x = parse_json_grid(pair["input"])
        y = parse_json_grid(pair["output"])
        pairs.append(ArcIOPair(x, y))
    return pairs
