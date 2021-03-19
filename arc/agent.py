from typing import List
from abc import ABC

import numpy as np

from launch import EVAL_SET
from arc.data_model import ArcIOPair


class ArcAgent(ABC):
    def predict(
        self, train_pairs: List[ArcIOPair], test_grids: List[np.ndarray]
    ) -> List[List[np.ndarray]]:
        raise NotImplementedError


class RandomAgent(ArcAgent):
    def predict(
        self, train_pairs: List[ArcIOPair], test_grids: List[np.ndarray]
    ) -> List[List[np.ndarray]]:
        outputs = []
        for tg in test_grids:
            out_shape = tg.shape
            out1 = np.random.randint(0, 9, out_shape)
            out2 = np.random.randint(0, 9, out_shape)
            out3 = np.random.randint(0, 9, out_shape)
            outputs.append([out1, out2, out3])
        return outputs


class CheatingAgent(ArcAgent):

    def __init__(self):
        self.answers = {}
        for prob in EVAL_SET:
            for pair in prob.test_pairs:
                self.answers[pair.x.tobytes()] = np.copy(pair.y)

    def predict(
        self, train_pairs: List[ArcIOPair], test_grids: List[np.ndarray]
    ) -> List[List[np.ndarray]]:
        outputs = []
        for tg in test_grids:
            outputs.append([self.answers[tg.tobytes()]])
        return outputs
