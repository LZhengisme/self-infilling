from pprint import pprint

from . import (
    ds1000,
    gsm,
    humaneval,
    mbpp,
)

TASK_REGISTRY = {
    **ds1000.create_all_tasks(),
    **humaneval.create_all_tasks(),
    **mbpp.create_all_tasks(),
    **gsm.create_all_tasks(),
}

ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name):
    try:
        return TASK_REGISTRY[task_name]()
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")
