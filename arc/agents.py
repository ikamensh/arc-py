from typing import List
from abc import ABC

import numpy as np

from arc.data import eval_set
from arc.types import ArcIOPair, ArcGrid, ArcPrediction


class ArcAgent(ABC):
    def predict(
        self, demo_pairs: List[ArcIOPair], test_grids: List[ArcGrid]
    ) -> List[ArcPrediction]:
        """given demonstrations, try to solve test tasks.

        given a list of demo grid pairs for the task,
        output for each test task a sorted list of likely answers.
        The most likely answer is the first in the answer list."""

        raise NotImplementedError


class RandomAgent(ArcAgent):
    """Makes random predicitons. Low chance of success. """

    def predict(
        self, demo_pairs: List[ArcIOPair], test_grids: List[ArcGrid]
    ) -> List[ArcPrediction]:
        outputs = []
        for tg in test_grids:
            out_shape = tg.shape
            out1 = np.random.randint(0, 9, out_shape)
            out2 = np.random.randint(0, 9, out_shape)
            out3 = np.random.randint(0, 9, out_shape)
            outputs.append([out1, out2, out3])
        return outputs


class CheatingAgent(ArcAgent):
    """Cheating agent looks at evaluation set and memorizes answers.

    Expected to get perfect score on the evaluation set."""

    def __init__(self):
        self.answers = {}
        for prob in eval_set:
            for pair in prob.test_pairs:
                self.answers[pair.x.tobytes()] = np.copy(pair.y)

    def predict(
        self, demo_pairs: List[ArcIOPair], test_grids: List[ArcGrid]
    ) -> List[ArcPrediction]:
        outputs = []
        for tg in test_grids:
            outputs.append([self.answers[tg.tobytes()]])
        return outputs
