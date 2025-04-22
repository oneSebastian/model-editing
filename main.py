import argparse
import yaml
from model_editing.benchmark import benchmark_knowledge_editing
from model_editing.editing_tasks.util import QueryType
from model_editing.analysis import EvalResult


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
        dataset_base_path = args.dataset_base_path,
        save_path = args.results_dir,
        device=args.device
    )


def analyze(args):
    result = EvalResult()
    result.load_editing_data(args.results_dir)
    result.aggregate_editing_data(groupby_dimensions=args.groupby_dimension, groupby_dataset_splits=args.groupby_dataset_splits, exclude_fact_queries=args.exclude_fact_queries, evaluate_generate_lengths=args.evaluate_generate_lengths)
    if not args.evaluate_generate_lengths:
        print(result.aggregated_editing_data.to_string())
        result.load_aggregated_control_data(args.results_dir)
        print(result.aggregated_control_data.to_string())
    else:
        raise NotImplementedError("Analysis of generate lengths not refactored yet")


def main():
    parser = argparse.ArgumentParser(description="Benchmark model editing.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation")
    eval_parser.set_defaults(func=evaluate)
    eval_parser.add_argument("--models", type=str, required=True, help="Model names (list or single model name)")
    eval_parser.add_argument("--editors", nargs="+", type=str, required=True, help="Editor name (list or single editor)")
    eval_parser.add_argument("--edit_batch_size", type=int, required=True, help="Number of examples per edit batch")
    eval_parser.add_argument("--eval_batch_size", type=int, default=16, help="Evaluation batch size")
    eval_parser.add_argument("--sample_size", type=int_or_none, default=2048, help="Dataset sample size (use 'None' for no down-sampling)")
    eval_parser.add_argument("--force_query_type", type=parse_query_type, default=None, help="Forced query type for the dataset (arg, gen or mc)")
    eval_parser.add_argument("--editing_tasks", nargs="+", type=str, default=["zsRE", "CounterFact", "MQuAKE", "RippleEdits"], help="Editing tasks (list or single task)")
    eval_parser.add_argument("--control_tasks", nargs="+", type=str, default=["lambada", "hellaswag"], help="LM Eval control tasks (list, single task or 'none')")
    eval_parser.add_argument("--evaluate_generate_lengths", type=bool, default=False, help="Evaluate multiple generate lengths")
    eval_parser.add_argument("--results_dir", type=str, default=None)
    eval_parser.add_argument("--device", type=str, default="cuda")
    eval_parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to config file")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze results")
    analyze_parser.add_argument("--results_dir", type=str, required=True, help="Directory of evaluation result files")
    analyze_parser.add_argument("--groupby_dimension", type=bool, default=False, help="Group results by dataset dimension")
    analyze_parser.add_argument("--groupby_dataset_splits", type=bool, default=False, help="Group results by dataset splits")
    analyze_parser.add_argument("--exclude_fact_queries", type=bool, default=True, help="Exclude fact queries from analysis")
    analyze_parser.add_argument("--evaluate_generate_lengths", type=bool, default=False, help="Evaluate multiple generate lengths")
    analyze_parser.set_defaults(func=analyze)

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
    main()