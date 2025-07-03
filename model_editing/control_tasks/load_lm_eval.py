import random
import numpy as np
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


def compute_task_boundaries(name, task, key_prefix, distribution_dict, task_boundaries=None, dev_split=None):
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
                data_start = 0
            else:
                data_start, data_end = task_boundaries[tuple(key_prefix + [name])][split]
                split_length = data_end - data_start

            if name in ["lambada", "hellaswag"]:
                assert task_boundaries is None
                split_length = split_length // 2
                if not dev_split:
                    data_start = split_length

            chunk_size = split_length // num_filled_chunks
            remainder = split_length % num_filled_chunks
            chunk_start = data_start
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


def load_control_task_dict(control_tasks, editing_tasks, dev_split=False):
    if not control_tasks:
        return None, None, None
    
    # TODO: For now we have implemented lm eval dev splits only for the tasks lambada and hellaswag

    assert all(isinstance(_, str) for _ in editing_tasks)
    task_manager = TaskManager()
    task_dict = get_task_dict(
        control_tasks,
        task_manager,
    )

    # copy task datasets and set num_fewshot to 0
    task_data = dict()

    def copy_task_data(name, task, key_prefix):
        if isinstance(task, dict):
            for subtask_name, subtask in task.items():
                copy_task_data(subtask_name, subtask, key_prefix + [name.group])
        else:
            assert isinstance(task, ConfigurableTask)
            
            for split, dataset in task.dataset.items():
                dataset_len = len(dataset)
                rng = np.random.RandomState(42)
                perm = rng.permutation(dataset_len)
                shuffled_dataset = dataset.select(perm.tolist())
                task.dataset[split] = shuffled_dataset

            detached_copy = detached_dataset_copy(task.dataset)
            task_data[tuple(key_prefix + [name])] = detached_copy
            task.num_fewshot = 0
    
    for _name, _task in task_dict.items():
        copy_task_data(_name, _task, [])

    # compute distributed task_boundaries
    edit_tasks_distribution_dict = {task: defaultdict(lambda: defaultdict(dict)) for task in editing_tasks}
    for _name, _task in task_dict.items():
        compute_task_boundaries(_name, _task, [], edit_tasks_distribution_dict, dev_split=dev_split)
    

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
    
    #print("DEBUG task dict prior to dataset removal")
    #for k, v in task_dict.items():
    #    print(f"##### TASK {k} ####")
    #    print(v)
    #    print("\n\n")
    #exit()

    for _name, _task in task_dict.items():
        remove_task_dataset(_name, _task, [])

    print("Loaded control task data and distributed it among editing tasks.")
    return task_dict, task_data, edit_tasks_distribution_dict


