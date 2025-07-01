import sys
import argparse
import yaml
from model_editing.benchmark import benchmark_knowledge_editing
from model_editing.editing_tasks.util import QueryType
from model_editing.analysis import EvalResult
from model_editing.grid_search import perform_grid_search, show_grid_search_results


def int_or_none(arg):
    if arg.lower() == "none":
        return None
    return int(arg)


def parse_query_type(arg):
    if arg.lower() == "arg":
        return QueryType.ARG
    elif arg.lower() == "gen":
        return QueryType.GEN
    elif arg.lower() == "mc":
        return QueryType.MC
    else:
        raise ValueError("Recieved invalid query type.")


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def merge_args_with_config(args, config_dict):
    for key, value in config_dict.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    return args


def evaluate(args):
    benchmark_knowledge_editing(
        model_names = args.models,
        model_editors = args.editors,
        editing_tasks = args.editing_tasks,
        control_tasks = args.control_tasks,
        edit_batch_size = args.edit_batch_size,
        eval_batch_size = args.eval_batch_size,
        sample_size = args.sample_size,
        evaluate_generate_lengths = args.evaluate_generate_lengths,
        force_query_type = args.force_query_type,
        use_chat_template = args.use_chat_template,
        edit_template_id = args.edit_template_id,
        dev_split=args.dev_split,
        dataset_base_path = args.dataset_base_path,
        save_path = args.results_dir,
        device=args.device
    )


def analyze(args):
    result = EvalResult()
    result.load_editing_data(args.results_dir)
    result.aggregate_editing_data(groupby_dimensions=args.groupby_dimension, groupby_dataset_splits=args.groupby_dataset_splits, exclude_fact_queries=not args.include_fact_queries, evaluate_generate_lengths=args.evaluate_generate_lengths)
    if not args.evaluate_generate_lengths:
        print(result.aggregated_editing_data.to_string())
        if args.to_csv:
            csv_data = result.aggregated_editing_data
            if not csv_data.empty:
                csv_data[["accuracy"]].round(3).to_csv(sys.stdout, index=True)
        result.load_aggregated_control_data(args.results_dir)
        df = result.aggregated_control_data
        if len(df) > 1:
            df = df.sort_values(by=['model', 'editor', 'task', 'metric'])
        print(df.to_string())
        if args.to_csv:
            csv_data = result.aggregated_control_data
            if not csv_data.empty:
                csv_data[["model", "editor", "task", "metric", "n-samples", "higher_is_better", "score"]].round(3).to_csv(sys.stdout, index=False)
    else:
        raise NotImplementedError("Analysis of generate lengths not refactored yet")


def grid_search(args):
    config_path = getattr(args, "parameter_config_file", "config/grid_parameters_config.yaml")
    config = load_config(config_path)
    if args.evaluate:
        # perform grid search
        perform_grid_search(config, save_id=args.save_id)
    else:
        # view grid search results
        show_grid_search_results(config, results_file=args.results_file)



def main():
    parser = argparse.ArgumentParser(description="Benchmark model editing.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation")
    eval_parser.set_defaults(func=evaluate)
    eval_parser.add_argument("--models", nargs="+", type=str, required=True, help="Model names (list or single model name)")
    eval_parser.add_argument("--editors", nargs="+", type=str, required=True, help="Editor name (list or single editor)")
    eval_parser.add_argument("--edit_batch_size", type=int, required=True, help="Number of examples per edit batch")
    eval_parser.add_argument("--eval_batch_size", type=int, default=16, help="Evaluation batch size")
    eval_parser.add_argument("--sample_size", type=int_or_none, default=2048, help="Dataset sample size (use 'None' for no down-sampling)")
    eval_parser.add_argument("--force_query_type", type=parse_query_type, default=None, help="Forced query type for the dataset (arg, gen or mc)")
    eval_parser.add_argument("--editing_tasks", nargs="+", type=str, default=["zsRE", "CounterFact", "MQuAKE", "RippleEdits"], help="Editing tasks (list or single task)")
    eval_parser.add_argument("--control_tasks", nargs="+", type=str, default=["lambada", "hellaswag"], help="LM Eval control tasks (list, single task or 'none')")
    eval_parser.add_argument("--evaluate_generate_lengths", action="store_true", help="Add flag to evaluate multiple generate lengths")
    eval_parser.add_argument("--results_dir", type=str, default="results/")
    eval_parser.add_argument("--device", type=str, default="cuda")
    eval_parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to config file")
    eval_parser.add_argument("--use_chat_template", action="store_true", help="Add flag to signify that model is instruction tuned; use chat template for queries")
    eval_parser.add_argument("--edit_template_id", type=int, default=1, help="Id for edit template; relevant only to context editors.")
    eval_parser.add_argument("--dev_split", action="store_true", help="Use dev split of datasets for development or hyper-parameter tuning")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze results")
    analyze_parser.add_argument("--results_dir", type=str, default="results/", help="Directory of evaluation result files")
    analyze_parser.add_argument("--groupby_dimension", action="store_true", help="Add flag to group results by dataset dimension")
    analyze_parser.add_argument("--groupby_dataset_splits", action="store_true", help="Add flag to group results by dataset splits")
    analyze_parser.add_argument("--include_fact_queries", action="store_true", help="Add flag to include fact queries in analysis")
    analyze_parser.add_argument("--evaluate_generate_lengths", action="store_true", help="Add flag to evaluate multiple generate lengths")
    analyze_parser.add_argument("--to_csv", action="store_true", help="Add flag to write results table to csv")
    analyze_parser.set_defaults(func=analyze)

    grid_search_parser = subparsers.add_parser("grid_search", help="Perform hyper-parameter grid search")
    grid_search_parser.add_argument("--parameter_config_file", type=str, default="config/grid_parameters_config.yaml", help="Path to file that holds parameters for grid search")
    grid_search_parser.add_argument("--evaluate", action="store_true", help="Set flag to perform grid grid search. By default we only show results.")
    grid_search_parser.add_argument("--save_id", type=int, default=None, help="Save id for results for parallel gris searches")
    grid_search_parser.add_argument("--results_file", type=str, default=None, help="Path to results file for analysis")
    grid_search_parser.set_defaults(func=grid_search)

    # add arguments from config file
    args = parser.parse_args()
    config_path = getattr(args, "config", "config/default_config.yaml")
    config = load_config(config_path)
    args = merge_args_with_config(args, config)

    # harmonize optional list args
    if args.command == "evaluate":
        list_args =["models", "editors", "editing_tasks", "control_tasks"]
        for list_arg in list_args:
            arg = getattr(args, list_arg, None)
            if arg is None:
                raise ValueError(f"{list_arg} cannot be None")
            if isinstance(arg, str):
                setattr(args, list_arg, [arg])
    
    # possibly no control task
    if args.command == "evaluate":
        arg = getattr(args, "control_tasks", None)
        if len(arg) == 1 and arg[0].lower() == "none":
            setattr(args, "control_tasks", [])

    # force query type generate when evaluating generate lengths
    if args.command == "evaluate" and args.evaluate_generate_lengths:
        setattr(args, "force_query_type", QueryType.GEN)

    args.func(args)
    

if __name__ == "__main__":
    # SEB: on gruenau I should use /home/tmp instead of /tmp
    import os
    os.environ['TMPDIR'] = '/home/tmp'
    main()