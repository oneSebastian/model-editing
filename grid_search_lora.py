import itertools
import pandas as pd
from pathlib import Path

from model_editing.benchmark import benchmark_knowledge_editing, load_dataset
from model_editing.evaluator import Evaluator


def get_test_parameters():
    return {
        "model_names": ["mistral_7B"],
        "model_editors": ["lora"],
        "edit_batch_size": [16],
        "lora_alpha": [32],
        "num_steps": [20],
        "learning_rate": [1e-4],
        "datasets": ["CounterFact"],
    }


def get_grid_parameters():
    return {
        "model_names": ["mistral_7B"],
        "model_editors": ["lora"],
        "edit_batch_size": [16, 512],
        "lora_alpha": [8, 32],
        "num_steps": [20, 50],
        "learning_rate": [1e-4, 5e-3],
        "datasets": ["zsRE", "CounterFact", "MQuAKE", "RippleEdits"],
    }


def evaluate_parameters(
        parameters,
        dataset_base_path="/vol/fob-vol3/mi20/pohlseba/projects/model-editing/data/datasets/",
    ):
    edit_batch_size = parameters["edit_batch_size"]
    lora_alpha = parameters["lora_alpha"]
    num_steps = parameters["num_steps"]
    learning_rate = parameters["learning_rate"]

    experiment_name = f"grid_lora_{edit_batch_size}_{lora_alpha}_{num_steps}_{learning_rate}"
    dataset = load_dataset(dataset_base_path, parameters["datasets"], sample_size=None, force_query_type=None, dev_split=True)
    hparams = {
        "lora_alpha": lora_alpha,
        "num_steps": num_steps,
        "lr": learning_rate,
    }
    
    evaluator = Evaluator(
        experiment_name = experiment_name,
        model_name = parameters["model_names"],
        editor_name = parameters["model_editors"],
        control_task_dict = None,
        control_task_data = None,
        control_data_boundaries = None,
        dataset = dataset,
        dataset_sample_size=None,
        edit_batch_size = edit_batch_size,
        evaluate_generate_lengths = False,
        use_chat_template = False,
        eval_batch_size = 16,
        save_path=None,
        device="cuda",
        dev_split=True,
        hparams=hparams,
    )
    evaluator.evaluate()
    df = evaluator.get_aggregate_results().reset_index()
    df["edit_batch_size"] = edit_batch_size
    df["lora_alpha"] = lora_alpha
    df["num_steps"] = num_steps
    df["learning_rate"] = learning_rate
    return df


def run_grid_search(save_path="results/thesis/grid_search/lora/"):
    # grid_parameters = get_test_parameters()
    grid_parameters = get_grid_parameters()
    
    # Iterate parameter grid
    grid_combinations = []
    keys = list(grid_parameters.keys())
    values = list(grid_parameters.values())
    for combination in itertools.product(*values):
        param_dict = dict(zip(keys, combination))
        grid_combinations.append(param_dict)

    # Evaluate all combinations
    df_path = save_path + "results.parquet"
    for i, parameters in enumerate(grid_combinations):
        print(f"Evaluate combination {i+1}: {parameters}")
        result = evaluate_parameters(parameters)
        print(result.to_string())
        if Path(df_path).exists():
            df = pd.read_parquet(file_path)
            df = pd.concat([df, result], ignore_index=True)
            df.to_parquet(df_path)
        else:
            result.to_parquet(df_path)


run_grid_search()
