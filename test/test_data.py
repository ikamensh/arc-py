import os

import pytest


@pytest.fixture()
def no_cache():
    from arc.data import cache_file
    if os.path.isfile(cache_file):
        os.remove(cache_file)


def test_eval_set(no_cache):
    from arc import eval_set, describe_task_set

    assert len(eval_set) == 400
    describe_task_set(eval_set)


def test_train_set(no_cache):
    from arc import train_set, describe_task_set

    assert len(train_set) == 400
    describe_task_set(train_set)
