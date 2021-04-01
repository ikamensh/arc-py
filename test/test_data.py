import os

import pytest


@pytest.fixture()
def no_cache():
    from arc.data import cache_file
    if os.path.isfile(cache_file):
        os.remove(cache_file)


def test_eval_set(no_cache):
    from arc import validation_problems, describe_task_group

    assert len(validation_problems) == 400
    describe_task_group(validation_problems)


def test_train_set(no_cache):
    from arc import train_problems, describe_task_group

    assert len(train_problems) == 400
    describe_task_group(train_problems)
