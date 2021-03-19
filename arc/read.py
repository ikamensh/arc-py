import os
import json
from typing import List

import numpy as np

from arc.types import ArcIOPair, ArcProblem, ArcGrid, verify_is_arc_grid


def parse_json_grid(grid: List[List[int]]) -> ArcGrid:
    rows = []
    for row in grid:
        rows.append(np.array(row))

    result = np.vstack(rows).astype(np.uint8)
    verify_is_arc_grid(result)
    return result


def parse_dir(d) -> List[ArcProblem]:
    cases = []
    for file in os.listdir(d):
        path = os.path.join(d, file)
        with open(path) as f:
            parsed = json.load(f)

            train = parsed["train"]
            demo_pairs = parse_group(train)

            test = parsed["test"]
            test_pairs = parse_group(test)
            cases.append(
                ArcProblem(
                    demo_pairs=demo_pairs,
                    test_pairs=test_pairs,
                    uid=file.replace(".json", ""),
                )
            )
    return cases


def parse_group(group: List[dict]):
    pairs = []
    for pair in group:
        x = parse_json_grid(pair["input"])
        y = parse_json_grid(pair["output"])
        pairs.append(ArcIOPair(x, y))
    return pairs
