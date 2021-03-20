# arc-py - types and utilities for dealing with ARC Challenge in python
[![Build Status](https://dev.azure.com/ikamenshchikov/flynt/_apis/build/status/ikamensh.flynt?branchName=master)](https://dev.azure.com/ikamenshchikov/flynt/_build/latest?definitionId=1&branchName=master) ![Coverage](https://img.shields.io/azure-devops/coverage/ikamenshchikov/flynt/1) [![PyPI version](https://badge.fury.io/py/arc-py.svg)](https://badge.fury.io/py/arc-py)  [![Downloads](https://pepy.tech/badge/arc-py)](https://pepy.tech/project/arc-py)  [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


arc-py package provides convenience types and utilities to handle [the ARC Challenge](https://github.com/fchollet/ARC).

It helps you convert the original .json files to numpy arrays, render them (and any edits or generation you might do) with matplotlib. 
It also suggests an interface for an agent competing in ARC Challenge, and offers evaluation function.

Install it by running `pip install arc-py`. 

## Examples

### View training examples:

```python
from arc import train_set, eval_set, describe_task_set


describe_task_set(train_set)
describe_task_set(eval_set)

for n, task in enumerate(train_set, start=1):
    for i, pair in enumerate(task.train_pairs, start=1):
        pair.plot(show=True, title=f"Task {n}: Demo {i}")

    for i, pair in enumerate(task.test_pairs, start=1):
        pair.plot(show=True, title=f"Task {n}: Test {i}")
```
Output example:
![Alt Task 1 example](res/task1_demo1.png?raw=true "Task 1 example")
![Alt Task 1 example](res/task1_demo2.png?raw=true "Task 1 example") 
![Alt Task 1 example](res/task1_test1.png?raw=true "Task 1 example")



> Note: The ARC challenge is designed for the developer not knowing the test problems. If you know those problems, you will overfit to them. We recommend not to view the evaluation set.

### View output of a random agent

```python
from typing import List
import numpy as np
from arc.types import ArcIOPair, ArcGrid, ArcPrediction
from arc.agents import ArcAgent

class RandomAgent(ArcAgent):
    """Makes random predicitons. Low chance of success. """

    def predict(
        self, demo_pairs: List[ArcIOPair], test_grids: List[ArcGrid]
    ) -> List[ArcPrediction]:
        """We are allowed to make up to 3 guesses per challange rules. """
        outputs = []
        for tg in test_grids:
            out_shape = tg.shape
            out1 = np.random.randint(0, 9, out_shape)
            out2 = np.random.randint(0, 9, out_shape)
            out3 = np.random.randint(0, 9, out_shape)
            outputs.append([out1, out2, out3])
        return outputs

    
from arc import train_set

p1 = next(iter(train_set))  # problem #1
agent = RandomAgent()
outs = agent.predict(p1.train_pairs, p1.test_inputs)

for test_pair, predicitons in zip(p1.test_pairs, outs):
    for p in predicitons:
        prediction = ArcIOPair(test_pair.x, p)
        prediction.plot(show=True)
```

### Evaluate the random agent

```python
from arc.agents import RandomAgent, CheatingAgent
from arc.evaluation import evaluate_agent


agent = RandomAgent()
results = evaluate_agent(agent)
print(results)
assert results.accuracy < 0.5
assert results.accuracy_any < 0.5
assert results.shape_accuracy > 0  # random agent guesses the shape of some outputs correctly
```

> ARC results for 400 problems. Stats:\
Accuracy                 : 0.0%\
Accuracy(at least one)   : 0.0%\
Correct answer shape     : 67.5%
