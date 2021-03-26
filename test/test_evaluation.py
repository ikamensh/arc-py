from arc.agents import RandomAgent, CheatingAgent
from arc.evaluation import evaluate_agent, ArcEvaluationResult
from arc import get_train_problem_by_uid, ArcColors

import numpy as np

def test_shape_matters():
    prob = get_train_problem_by_uid("d631b094")
    result = ArcEvaluationResult()
    guess_1 = np.ones([1,1]) * ArcColors.YELLOW
    guess_2 = np.ones([2,2]) * ArcColors.YELLOW
    guess_2[0,1] = 0
    result.add_answer(prob, [[guess_1, guess_2]])
    assert result.accuracy < 0.01


def test_random_agent():
    agent = RandomAgent()
    results = evaluate_agent(agent)
    assert results.accuracy < 0.5
    assert results.accuracy_any < 0.5


def test_cheating_agent():
    agent = CheatingAgent()
    results = evaluate_agent(agent)
    assert results.accuracy > 0.99
    assert results.accuracy_any > 0.99
    assert results.shape_accuracy > 0.99
