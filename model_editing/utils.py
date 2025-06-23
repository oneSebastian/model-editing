from typing import Iterable
from datasets import DatasetDict, Dataset
from lm_eval.tasks import ConfigurableTask


def access_task(_task_dict, _task_key):
    if isinstance(_task_dict, dict):
        #print("DEBUG access_task _task_dict keys:", list(_task_dict.keys()))
        #print("DEBUG access_task _task_key:", _task_key)
        if _task_key[0] in _task_dict:
            return access_task(_task_dict[_task_key[0]], _task_key[1:])
        else:
            for key in _task_dict.keys():
                if isinstance(key, str):
                    assert key != _task_key[0]
                    continue
                #print("Not string", key, key.group, type(key.group), _task_key[0],  _task_key[0] == key.group)
                if key.group == _task_key[0]:
                    return access_task(_task_dict[key], _task_key[1:])
            raise ValueError("One of the keys should match!")
    else:
        assert isinstance(_task_dict, ConfigurableTask)
        assert len(_task_key) == 0
        return _task_dict
    

def detached_dataset_copy(dataset_dict: DatasetDict) -> DatasetDict:
    return DatasetDict({
        split: Dataset.from_dict(ds[:]) for split, ds in dataset_dict.items()
    })


def detached_select(dataset: Dataset, indices: Iterable[int]) -> Dataset:
    subset = dataset.select(indices)
    return Dataset.from_dict(subset[:])
    
