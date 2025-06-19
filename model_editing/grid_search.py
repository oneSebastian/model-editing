import itertools
import pandas as pd
from pathlib import Path

from model_editing.benchmark import benchmark_knowledge_editing, load_dataset
from model_editing.evaluator import Evaluator


def evaluate_parameters(parameters, config):
    dataset = parameters["datasets"]
    edit_batch_size = parameters["edit_batch_size"]
    lora_alpha = parameters["lora_alpha"]
    num_steps = parameters["num_steps"]
    learning_rate = float(parameters["learning_rate"])
    
    experiment_name = f"grid_lora_{edit_batch_size}_{lora_alpha}_{num_steps}_{learning_rate}"
    dataset = load_dataset(config["dataset_base_path"], dataset, sample_size=None, force_query_type=None, dev_split=True)
    hparams = {
        "lora_alpha": lora_alpha,
        "num_steps": num_steps,
        "lr": learning_rate,
    }

    evaluator = Evaluator(
        experiment_name = experiment_name,
        model_name = config["model"],
        editor_name = config["editor"],
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


def perform_grid_search(config, save_id=None):
    if config["editor"] != "lora":
        raise NotImplementedError("Grid search has so for only been implemented for the LoRA editor.")

    # Iterate parameter grid
    parameters = config["grid_parameters"]
    grid_combinations = []
    keys = list(parameters.keys())
    values = list(parameters.values())
    for combination in itertools.product(*values):
        param_dict = dict(zip(keys, combination))
        grid_combinations.append(param_dict)

    # Evaluate all combinations
    if save_id is not None:
        df_path = config["results_dir"] + f"results_{save_id}.parquet"
    else:
        df_path = config["results_dir"] + "results.parquet"
    for i, parameters in enumerate(grid_combinations):
        print(f"Evaluate combination {i+1}: {parameters}")
        result = evaluate_parameters(parameters, config)
        print(result.to_string())
        if Path(df_path).exists():
            df = pd.read_parquet(df_path)
            df = pd.concat([df, result], ignore_index=True)
            df.to_parquet(df_path)
        else:
            result.to_parquet(df_path)
    print("Grid search complete.")


def show_grid_search_results(config, results_file=None):
    #paths = [
    #    "results/thesis/grid_search/lora/results_0.parquet",
    #    "results/thesis/grid_search/lora/results_1.parquet",
    #]
    dfs = []
    if results_file is not None:
        # for gpt2-xk grid search search id 1 uses the mlp layers for editing without id uses the c_attn
        print("Load results from", results_file)
        dfs.append(pd.read_parquet(results_file))
    else:
        for parquet_file in Path(config["results_dir"]).glob("*.parquet"):
            print("Load results from", parquet_file)
            dfs.append(pd.read_parquet(parquet_file))
    df = pd.concat(dfs, ignore_index=True)

    df = df[["model", "editor", "edit_batch_size", "lora_alpha", "num_steps", "learning_rate", "dataset", "accuracy"]]
    #df = df.groupby(["model", "editor", "edit_batch_size", "lora_alpha", "num_steps", "learning_rate"]).agg("mean")
    #df = df.groupby(["model", "editor", "edit_batch_size", "lora_alpha", "num_steps", "learning_rate", "dataset"]).agg("mean")
    
    #print(df.groupby(["model", "editor", "edit_batch_size", "lora_alpha", "num_steps", "learning_rate", "dataset"]).agg("mean").to_string())

    df = df.pivot_table(
        index=["model", "editor", "edit_batch_size", "lora_alpha", "num_steps", "learning_rate"],
        columns="dataset",
        values="accuracy",
        aggfunc="first"
    ).reset_index()
    df = df[["model", "editor", "edit_batch_size", "lora_alpha", "num_steps", "learning_rate", "zsre", "CounterFact", "MQuAKE", "RippleEdits"]]
    print(df.to_string())

    df_long = df.melt(
        id_vars=["model", "editor", "edit_batch_size", "lora_alpha", "num_steps", "learning_rate"],
        var_name="dataset",
        value_name="accuracy"
    )

    # Create composite column name
    df_long["dataset_lr"] = df_long["dataset"] + "@" + df_long["learning_rate"].astype(str)

    # Pivot so that each dataset@lr becomes a column
    final_df = df_long.pivot_table(
        index=["model", "editor", "edit_batch_size", "lora_alpha", "num_steps"],
        columns="dataset_lr",
        values="accuracy",
        aggfunc="first"  # in case of duplicates
    ).reset_index()
    print(final_df.to_string())
    exit()

    for index, row in df.iterrows():
        print("index:", index)
        print("roe:", row)
        exit()

    # print(df.reset_index().to_csv())