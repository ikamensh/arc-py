# arc-py - types and utilities for dealing with ARC Challenge in python
[![Build Status](https://dev.azure.com/ikamenshchikov/flynt/_apis/build/status/ikamensh.flynt?branchName=master)](https://dev.azure.com/ikamenshchikov/flynt/_build/latest?definitionId=1&branchName=master) ![Coverage](https://img.shields.io/azure-devops/coverage/ikamenshchikov/flynt/1) [![PyPI version](https://badge.fury.io/py/arc-py.svg)](https://badge.fury.io/py/arc-py)  [![Downloads](https://pepy.tech/badge/arc-py)](https://pepy.tech/project/arc-py)  [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


`arc-py` package provides types and utilities to handle [the ARC Challenge](https://github.com/fchollet/ARC).

It helps you convert the original .json files to numpy arrays, view them (and any edits or generation you might do) with matplotlib. 
It also suggests an interface for an agent competing in ARC Challenge, and offers evaluation function.

Install it by running `pip install arc-py`.

## Accessing the data

Upon  installing `arc-py`, you will get the original .json files  locally.
You can find  the  folders containing them as  `from  arc import train_data_dir, eval_data_dir`. 
However, if you are going to be using  python, probably you will find it more convenient to get 
the  data  as 2D numpy arrays, grouped logically under two sets of  problems,  train and validation.
These are available as `from arc import train_problems, validation_problems`.

### Data Types
`arc-py` introduces a few types to structure the data in the challenge:
* `arc.types.ArcProblem` represent one task -  made of demonstrations (`prob.train_pairs`) and tests (`prob.test_pairs`).
* `arc.types.ArcIOPair` is a unit of the demonstrations/tests  making up a  problem  -  it's the  input  grid  (`pair.x`) and corresponding output grid (`pair.y`).
* `arc.types.ArcGrid` is an alias for `np.ndarray`. Specifically, an ArcGrid must be 2D, have dimensions between 1 to 30 each, and only have integers in range [0, 9] as it's elements. To verify some numpy array complies with this spec, a check function is offered: `arc.types.verify_is_arc_grid`.

## Constructing agents and evaluating results

### Agent API
The concept for an agent is the following - it's the program you develop/train by looking on training problems, which then goes on to be scored on validation problems.
Per the rules of original kaggle competition:
1)  a problem can contain multiple test grids. 
2) it's  acceptable  to make up to 3 guesses  as  to what  the answer to a given test grid  could be.

Since the agent must not be shown the answers to test grids, we give it following signature:
```python
    def predict(
        self, demo_pairs: List[ArcIOPair], test_grids: List[ArcGrid]
    ) -> List[ArcPrediction]:
```
Here, `ArcPrediction` is a `list[ArcGrid]` - containing 1 to 3 guesses. First element  of the output list corresponds to the first test grid, and so on (meaning for the return value `result`, you could match inputs and outputs with `x, y in zip(test_grids, result)`).

### Evaluation

`arc-py` offers a class that can keep track  of your accuracy and few auxillary  target metrics: `arc.evaluation.ArcEvaluationResult`. It's best demonstrated with an example:
```python
def evaluate_agent(
    agent: ArcAgent, problems: List[ArcProblem] = validation_problems
) -> ArcEvaluationResult:
    
    result = ArcEvaluationResult()
    for prob in problems:
        pred = agent.predict(prob.train_pairs, prob.test_inputs)
        result.add_answer(prob, pred)

    return result
```

Just print the `ArcEvaluationResult` object to see the results. Sample output:
```text
ARC results for 400 problems. Stats:
Accuracy                 : 2.2%
Accuracy(at least one)   : 2.5%
Correct answer shape     : 52.4%
```

## Examples

for an example of a project using arc-py, see https://github.com/ikamensh/solve_arc

### View training examples:

```python
from arc import train_problems, validation_problems, describe_task_group

describe_task_group(train_problems)
describe_task_group(validation_problems)

for n, task in enumerate(train_problems, start=1):
    for i, pair in enumerate(task.train_pairs, start=1):
        pair.plot(show=True, title=f"Task {n}: Demo {i}")

    for i, pair in enumerate(task.test_pairs, start=1):
        pair.plot(show=True, title=f"Task {n}: Test {i}")
```

![Alt Task 1 example1](res/task1_demo1.png?raw=true "Task 1 example 1")
![Alt Task 1 example2](res/task1_demo2.png?raw=true "Task 1 example 2") 
![Alt Task 1 example3](res/task1_test1.png?raw=true "Task 1 test 1")

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


from arc import train_problems

p1 = next(iter(train_problems))  # problem #1
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
