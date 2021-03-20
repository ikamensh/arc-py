from arc.agents import RandomAgent, CheatingAgent
from arc.evaluation import evaluate_agent


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
