from typing import List, Dict, Set

from arc.data import eval_set
from arc.types import ArcProblem, ArcPrediction
from arc.agents import ArcAgent
from fractions import Fraction
import numpy as np

MAX_ATTEMPTS = 3


class ArcEvaluationResult:
    def __init__(self):
        self.raw_answers: Dict[ArcProblem, ArcPrediction] = {}
        self.correct: Set[ArcProblem] = set()
        self.partially_correct: Set[ArcProblem] = set()
        self.correct_shape: Dict[ArcProblem, Fraction] = {}

    def add_answer(self, prob: ArcProblem, pred: List[ArcPrediction]) -> None:
        assert len(pred) == len(prob.test_pairs), (
            f"The problem has {len(prob.test_pairs)} tests, "
            f"but the agent returned {len(pred)} predictions."
        )
        self.raw_answers[prob] = pred
        tests_successful = []
        right_shapes = []
        for answer_grid, solution in zip(prob.test_outputs, pred):
            success = any(s.shape == answer_grid.shape and np.all(s == answer_grid) for s in solution[:MAX_ATTEMPTS])
            tests_successful.append(success)
            right_shapes.append(
                Fraction(
                    sum(answer_grid.shape == s.shape for s in solution[:MAX_ATTEMPTS]),
                    len(solution[:MAX_ATTEMPTS]),
                )
            )

        self.correct_shape[prob] = sum(right_shapes) / len(prob.test_pairs)
        if all(tests_successful):
            self.correct.add(prob)
        elif any(tests_successful):
            self.partially_correct.add(prob)

    @property
    def accuracy(self):
        N = len(self.raw_answers)
        return len(self.correct) / N

    @property
    def accuracy_any(self):
        N = len(self.raw_answers)
        return self.accuracy + len(self.partially_correct) / N

    @property
    def shape_accuracy(self):
        N = len(self.raw_answers)
        return sum(self.correct_shape.values()) / N

    def __repr__(self):
        return (
            f"ARC results for {len(self.raw_answers)} problems. Stats:"
            f"\n{'Accuracy':<25}: {self.accuracy:.1%}"
            f"\n{'Accuracy(at least one)':<25}: {self.accuracy_any:.1%}"
            f"\n{'Correct answer shape':<25}: {float(self.shape_accuracy):.1%}"
        )


def evaluate_agent(
    agent: ArcAgent, problems: List[ArcProblem] = eval_set
) -> ArcEvaluationResult:
    result = ArcEvaluationResult()
    for prob in problems:
        pred = agent.predict(prob.train_pairs, prob.test_inputs)
        result.add_answer(prob, pred)

    return result


if __name__ == "__main__":
    from arc.agents import RandomAgent, CheatingAgent

    agent = RandomAgent()
    results = evaluate_agent(agent)
    print("\nRandom agent:", results)

    agent = CheatingAgent()
    results = evaluate_agent(agent)
    print("\nCheating agent:", results)
