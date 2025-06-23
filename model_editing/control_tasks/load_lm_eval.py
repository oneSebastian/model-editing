import random
from collections import defaultdict
from typing import Dict, Tuple, Union, Callable
from copy import deepcopy
from datasets import DatasetDict
from lm_eval.tasks import TaskManager, get_task_dict, ConfigurableTask
from model_editing.utils import detached_dataset_copy, detached_select


def recursive_print(data, depth, step=4):
    if isinstance(data, list):
        for i, item in enumerate(data):
            print(f"{' ' * depth}list item {i}:")
            recursive_print(item, depth + step)
    elif isinstance(data, dict):
        for key, value in data.items():
            if 'mmlu_high_school_european_history' not in key:
                continue
            print(f"{' ' * depth}{key}:")
            recursive_print(value, depth + step)
    else:
        print(f"{' ' * depth}{data}")


def compute_task_boundaries(name, task, key_prefix, distribution_dict, task_boundaries=None):
    if isinstance(task, dict):
        for subtask_name, subtask in task.items():
            compute_task_boundaries(subtask_name, subtask, key_prefix + [name.group], distribution_dict, task_boundaries=task_boundaries)
    else:
        assert isinstance(task, ConfigurableTask)
        distribution_keys = list(distribution_dict.keys())
        random.shuffle(distribution_keys)

        # get chunk size for this task depending on minimum split size and number of distribution keys
        if task_boundaries is None:
            minimum_split_size = min(len(task.dataset[split]) for split in task.dataset.keys())
        else:
            minimum_split_size = float('inf')
            for split in task.dataset.keys():
                if task_boundaries[tuple(key_prefix + [name])] is not None and task_boundaries[tuple(key_prefix + [name])][split] is not None:
                    minimum_split_size = min(minimum_split_size, task_boundaries[tuple(key_prefix + [name])][split][1] - task_boundaries[tuple(key_prefix + [name])][split][0])
            # minimum_split_size = min(task_boundaries[tuple(key_prefix + [name])][split][1] - task_boundaries[tuple(key_prefix + [name])][split][0] for split in task.dataset.keys())
        num_filled_chunks = min(minimum_split_size, len(distribution_keys))

        for split in task.dataset.keys():
            if task_boundaries is None:
                split_length = len(task.dataset[split])
            else:
                data_start, data_end = task_boundaries[tuple(key_prefix + [name])][split]
                split_length = data_end - data_start

            chunk_size = split_length // num_filled_chunks
            remainder = split_length % num_filled_chunks
            chunk_start = 0 if task_boundaries is None else data_start
            for i in range(num_filled_chunks):
                chunk_end = chunk_start + chunk_size + (1 if i < remainder else 0)
                if chunk_end > chunk_start:
                    distribution_dict[distribution_keys[i]][tuple(key_prefix + [name])][split] = (chunk_start, chunk_end)
                else:
                    distribution_dict[distribution_keys[i]][tuple(key_prefix + [name])][split] = None
                chunk_start = chunk_end

            '''
            for key in distribution_dict.keys():
                print("Editing Task:", key)
                print("Control Task:", tuple(key_prefix + [name]))
                for k, v in distribution_dict[key][tuple(key_prefix + [name])].items():
                    print("    ", k, v)
            exit()
            '''


def load_control_task_dict(control_tasks, editing_tasks):
    if not control_tasks:
        return None, None, None

    assert all(isinstance(_, str) for _ in editing_tasks)
    task_manager = TaskManager()
    task_dict = get_task_dict(
        control_tasks,
        task_manager,
    )

    # copy task datasets
    task_data = dict()

    def copy_task_data(name, task, key_prefix):
        if isinstance(task, dict):
            for subtask_name, subtask in task.items():
                copy_task_data(subtask_name, subtask, key_prefix + [name.group])
        else:
            assert isinstance(task, ConfigurableTask)
            detached_copy = detached_dataset_copy(task.dataset)
            task_data[tuple(key_prefix + [name])] = detached_copy
    
    for _name, _task in task_dict.items():
        copy_task_data(_name, _task, [])

    # compute distributed task_boundaries
    edit_tasks_distribution_dict = {task: defaultdict(lambda: defaultdict(dict)) for task in editing_tasks}
    for _name, _task in task_dict.items():
        compute_task_boundaries(_name, _task, [], edit_tasks_distribution_dict)
    

    # remove dataset referneces from task_dict. Data is stored separately in task_data
    def remove_task_dataset(name, task, key_prefix):
        if isinstance(task, dict):
            for subtask_name, subtask in task.items():
                remove_task_dataset(subtask_name, subtask, key_prefix + [name.group])
        else:
            assert isinstance(task, ConfigurableTask)
            # cut down dataset to dummy size; we fill it on demand with batch data in the evaluation loop
            splits = list(task.dataset.keys())
            task.dataset = DatasetDict({
                #split: detached_select(detached_copy[split], range(1))
                split: None
                for split in splits
            })
    
    for _name, _task in task_dict.items():
        remove_task_dataset(_name, _task, [])

    print("Loaded control task data and distributed it among editing tasks.")
    return task_dict, task_data, edit_tasks_distribution_dict


def old_load_control_task_dict(control_tasks, editing_tasks, target_splits=None):
    if not control_tasks:
        return None, None, None
    
    assert all(isinstance(_, str) for _ in editing_tasks)
    task_manager = TaskManager()
    print("DEBUG: run get_task_dict")
    task_dict = get_task_dict(
        control_tasks,
        task_manager,
    )
    print("DEBUG: get_task_dict finished")
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
                task_data[(control_task, subtask_name)] = deepcopy(subtask.dataset)
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
    

    return task_dict, task_data, data_boundaries
