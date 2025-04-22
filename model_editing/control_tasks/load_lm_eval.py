from collections import defaultdict
from typing import Dict, Tuple, Union, Callable
from copy import deepcopy
from datasets import DatasetDict
from lm_eval.tasks import TaskManager, get_task_dict


def load_control_task_dict(control_tasks, editing_tasks, target_splits=None):
    if not control_tasks:
        return None, None, None
    
    assert all(isinstance(_, str) for _ in editing_tasks)
    task_manager = TaskManager()
    task_dict = get_task_dict(
        control_tasks,
        task_manager,
    )
    task_data = dict()

    data_boundaries = {ke_task: defaultdict(dict) for ke_task in editing_tasks}
    task_splits = len(editing_tasks)

    def compute_boundaries(_task, _control_task, _subtask_name=None):
        for split in _task.dataset.keys():
            n = len(_task.dataset[split])
            chunk_size = n // task_splits
            remainder = n % task_splits
            chunk_start = 0
            for i, ke_task in enumerate(editing_tasks):
                # split the control_tasks among the editing_tasks
                chunk_end = chunk_start + chunk_size + (1 if i < remainder else 0)
                data_boundaries[ke_task][(_control_task, _subtask_name)][split] = (chunk_start, chunk_end)
                chunk_start = chunk_end

    # delete all but the target dataset splits
    delete_keys = []
    for control_task, task in task_dict.items():
        if isinstance(task, dict):
            for subtask_name, subtask in task.items():
                for split in subtask.dataset.keys():
                    key = (control_task, subtask_name, split)
                    if key not in target_splits:
                        delete_keys.append(key)
        else:
            for split in task.dataset.keys():
                key = (control_task, split)
                if key not in target_splits:
                    delete_keys.append(key)
    for entry in delete_keys:
            if len(entry) == 2:
                del task_dict[entry[0]].dataset[entry[1]]
            elif len(entry) == 3:
                del task_dict[entry[0]][entry[1]].dataset[entry[2]]
    
    for control_task, task in task_dict.items():
        if isinstance(task, dict):
            for subtask_name, subtask in task.items():
                task_data[(control_task.group, subtask_name)] = deepcopy(subtask.dataset)
                compute_boundaries(_task=subtask, _control_task=control_task, _subtask_name=subtask_name)
        else:
            task_data[(control_task, None)] = deepcopy(task.dataset)
            compute_boundaries(_task=task, _control_task=control_task)
    
    # clean data from task dict after it was separately saved
    delete_keys = []
    for control_task, task in task_dict.items():
        if isinstance(task, dict):
            for subtask_name, subtask in task.items():
                for split in subtask.dataset.keys():
                    key = (control_task, subtask_name, split)
                    delete_keys.append(key)
        else:
            for split in task.dataset.keys():
                key = (control_task, split)
                delete_keys.append(key)
    for entry in delete_keys:
            if len(entry) == 2:
                del task_dict[entry[0]].dataset[entry[1]]
            elif len(entry) == 3:
                del task_dict[entry[0]][entry[1]].dataset[entry[2]]
    
    '''
    print("DEBUG load_control_eval task dict:")
    for control_task, task in task_dict.items():
        if isinstance(task, dict):
            for subtask_name, subtask in task.items():
                print(f"task={control_task}, subtask={subtask_name}, n_splits={len(subtask.dataset)}")
        else:
            print(f"task={control_task}, n_splits={len(task.dataset)}")
    print("DEBUG load_control_eval data_dict entries:")
    for key, value in task_data.items():
        for split, data in value.items():
            print("    ", key, split, len(data))
    print("DEBUG load_control_eval data_boundaries:")
    for ke_task, boundaries in data_boundaries.items():
        for key, splits in boundaries.items():
            for split, _ in splits.items():
                print(f"ke_task={ke_task}, control_task={key}, split={split}, boundary={_}")
    '''

    return task_dict, task_data, data_boundaries
