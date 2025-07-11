import os
import requests
import random
from pathlib import Path
from typing import Optional, Union
from itertools import product
from .evaluator import Evaluator
from .editing_tasks.datasets.RippleEdits import RippleEditsDataset
from .editing_tasks.datasets.MQuAKE import MQuAKEDataset
from .editing_tasks.datasets.CounterFact import CounterFactDataset
from .editing_tasks.datasets.zsRE import ZSREDataset
# from control_tasks.tasks import load_tasks_from_names
from .control_tasks.load_lm_eval import load_control_task_dict
from .editing_tasks.util import QueryType


def load_dataset(dataset_base_path, editing_task, split=None, force_query_type=None, dev_split=False):
    if editing_task == "RippleEdits":
        dataset = RippleEditsDataset.from_file(f"{dataset_base_path}/RippleEdits/", split=split, force_query_type=force_query_type, dev_split=dev_split)
    elif editing_task == "MQuAKE":
        dataset = MQuAKEDataset.from_file(f"{dataset_base_path}/MQuAKE/", split=split, force_query_type=force_query_type, dev_split=dev_split)
    elif editing_task == "CounterFact":
        dataset = CounterFactDataset.from_file(f"{dataset_base_path}/CounterFact/", force_query_type=force_query_type, dev_split=dev_split)
    elif editing_task == "zsRE":
        dataset = ZSREDataset.from_file(f"{dataset_base_path}/zsRE/", force_query_type=force_query_type, dev_split=dev_split)
    else:
        raise ValueError("This benchmark only supports the datasets RippleEdits, MQuAKE, CounterFact and zsRE.")
    return dataset


def run_experiment(
        experiment_name,
        model_name,
        editor_name,
        editing_task,
        control_task_dict,
        control_task_data,
        control_data_boundaries,
        dataset,
        dataset_sample_size,
        edit_batch_size,
        evaluate_generate_lengths,
        use_chat_template,
        edit_template_id,
        eval_batch_size,
        device,
        save_path,
        dev_split,
        retriever_type,
        retrieve_k,
        retriever_embedding_model,
):
    print(f"Run experiment: {experiment_name}", flush=True)
    evaluator = Evaluator(
        experiment_name = experiment_name,
        model_name = model_name,
        editor_name = editor_name,
        control_task_dict = control_task_dict,
        control_task_data = control_task_data,
        control_data_boundaries = control_data_boundaries[editing_task] if control_data_boundaries is not None else None,
        dataset = dataset,
        dataset_sample_size=dataset_sample_size,
        edit_batch_size = edit_batch_size,
        evaluate_generate_lengths = evaluate_generate_lengths,
        use_chat_template = use_chat_template,
        edit_template_id = edit_template_id,
        eval_batch_size = eval_batch_size,
        save_path=save_path,
        device=device,
        dev_split=dev_split,
        retriever_type=retriever_type,
        retrieve_k=retrieve_k,
        retriever_embedding_model=retriever_embedding_model,
    )

    # run evaluation for this experiment
    evaluator.evaluate()
    evaluator.save_results()


def download_dataset(dataset, dataset_path):
    if dataset == "zsRE":
        url = "https://rome.baulab.info/data/dsets/zsre_mend_eval.json"
        print(f"Downloading {dataset} from {url}")
        response = requests.get(url)
        response.raise_for_status()
        with open(f"{dataset_path}/zsre_mend_eval.json", "w", encoding="utf-8") as f:
            f.write(response.text)
    elif dataset == "CounterFact":
        url = "https://rome.baulab.info/data/dsets/counterfact.json"
        print(f"Downloading {dataset} from {url}")
        response = requests.get(url)
        response.raise_for_status()
        with open(f"{dataset_path}/counterfact.json", "w", encoding="utf-8") as f:
            f.write(response.text)
    elif dataset == "MQuAKE":
        url = "https://raw.githubusercontent.com/princeton-nlp/MQuAKE/main/datasets/MQuAKE-CF-3k-v2.json"
        print(f"Downloading {dataset} from {url}")
        response = requests.get(url)
        response.raise_for_status()
        with open(f"{dataset_path}/MQuAKE-CF-3k-v2.json", "w", encoding="utf-8") as f:
            f.write(response.text)
    else:
        raise NotImplementedError(f"Download of dataset {dataset} not supported.")


def benchmark_knowledge_editing(
    model_names: list,
    model_editors: list,
    editing_tasks: list,
    control_tasks: list,
    edit_batch_size: int,
    eval_batch_size: int,
    sample_size: Optional[int],
    evaluate_generate_lengths: bool,
    force_query_type: Optional[QueryType],
    use_chat_template: bool,
    edit_template_id: int,
    dev_split: bool,
    dataset_base_path: str,
    save_path: str,
    device: str,
    retriever_type: str,
    retrieve_k: int,
    retriever_embedding_model: str,
    random_seed: int = 42,
):
    # seed random
    random.seed(random_seed)
    
    # check if benchmark datasets are available. If not download them.
    for editing_task in editing_tasks:
        path = Path(dataset_base_path + editing_task + "/")
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            download_dataset(editing_task, path)

    #TODO: handle warning
    '''
    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
            - Avoid using `tokenizers` before the fork if possible
            - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    '''
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # control_tasks should be eavenly spread over all ke tasks to ensure that LM capabilities are tested on all kinds of edits belonging to the different tasks
    # load control_tasks
    # loaded_control_tasks = load_tasks_from_names(control_tasks=control_tasks, control_tasks=control_tasks)


    # TODO: we are checking for emptyness on lm eval evaluation, but that only makes sense if we distinguish between necessary and unnecessary splits
    # Hence I just delete all unnecessary splits for now, which requires manual testing to detect necessary splits for each lm eval task
    target_splits = [
        ("lambada_openai", "test"),
        ("lambada_standard", "test"),
        ("anli_r1", "test_r1"),
        ("anli_r2", "test_r2"),
        ("anli_r3", "test_r3"),
        ("commonsense_qa", "validation"),
        ("hellaswag", "validation"),
        ("wikitext", "test"),
        ("cola", "validation"),
        ("mnli", "validation_matched"),
        ("mnli_mismatch", "validation_mismatched"),
        ("mrpc", "validation"),
        ("qnli", "validation"),
        ("qqp", "validation"),
        ("rte", "validation"),
        ("sst2", "validation"),
        ("wnli", "validation"),
    ]

    control_task_dict, control_task_data, control_data_boundaries = load_control_task_dict(control_tasks=control_tasks, editing_tasks=editing_tasks, dev_split=dev_split)

    '''
    def print_datasets(name, task, key_prefix):
        if isinstance(task, dict):
            for subtask_name, subtask in task.items():
                print_datasets(subtask_name, subtask, key_prefix + [name.group])
        else:
            print(key_prefix, name)
            for split, dataset in task.dataset.items():
                print(split, (dataset if dataset is None else len(dataset)))
    
    for k, v in control_task_dict.items():
        print_datasets(k, v, [])
    exit()
    '''



    for model_name, model_editor, editing_task in product(model_names, model_editors, editing_tasks):
        experiment_name = f"{model_name}_{model_editor}_{editing_task}_{edit_batch_size}{('_' + str(sample_size)) if sample_size else ''}"
        dataset = load_dataset(dataset_base_path, editing_task, force_query_type=force_query_type, dev_split=dev_split)
        run_experiment(
            experiment_name=experiment_name,
            model_name=model_name,
            editor_name=model_editor,
            editing_task=editing_task,
            control_task_dict=control_task_dict,
            control_task_data=control_task_data,
            control_data_boundaries=control_data_boundaries,
            dataset=dataset,
            dataset_sample_size=sample_size,
            edit_batch_size=edit_batch_size,
            evaluate_generate_lengths=evaluate_generate_lengths,
            use_chat_template=use_chat_template,
            edit_template_id=edit_template_id,
            eval_batch_size=eval_batch_size,
            device=device,
            save_path=save_path,
            dev_split=dev_split,
            retriever_type=retriever_type,
            retrieve_k=retrieve_k,
            retriever_embedding_model=retriever_embedding_model,
        )

        


