from arc.agents import RandomAgent, CheatingAgent
from arc.evaluation import arc_eval


def test_random_agent():
    agent = RandomAgent()
    results = arc_eval(agent)
    assert sum(results) < len(results) // 2


def test_cheating_agent():
    agent = CheatingAgent()
    results = arc_eval(agent)
    assert sum(results) > len(results) // 2
