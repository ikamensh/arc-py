from launch import EVAL_SET
from arc.agent import ArcAgent
import numpy as np

MAX_ATTEMPTS = 3

def eval(agent: ArcAgent):
    results = []
    for prob in EVAL_SET:

        pred = agent.predict(prob.train_pairs, [p.x for p in prob.test_pairs])
        assert len(pred) == len(prob.test_pairs), (
            f"The problem has {len(prob.test_pairs)} tests, "
            f"but the agent returned {len(pred)} predictions."
        )
        problem_scores = []
        for test, solution in zip(prob.test_pairs, pred):
            problem_scores.append( any(np.all(s == test.y) for s in solution[:MAX_ATTEMPTS]) )

        results.append( all(problem_scores) )

    return results


if __name__ == '__main__':
    from arc.agent import RandomAgent, CheatingAgent

    agent = RandomAgent()
    results = eval(agent)

    from collections import Counter
    print("Results for a random agent:", Counter(results))

    agent = CheatingAgent()
    results = eval(agent)

    from collections import Counter

    print("Results for a cheating agent:", Counter(results))

