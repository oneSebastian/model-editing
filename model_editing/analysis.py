import pandas as pd
import numpy as np
import copy
from tqdm import tqdm
from itertools import product
from collections import defaultdict
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


class EvalResult():
    def __init__(self, editing_data=None, control_data=None):
        self.editing_data = editing_data
        self.control_data = control_data
    
    def load_editing_data(self, results_path):
        # load and aggregate editing_results
        dfs_ke = []
        for path in Path(results_path).iterdir():
            if str(path).endswith("_ke.parquet"):
                dfs_ke.append(pd.read_parquet(path))
                print(f"Loaded data from path={path}")
        self.editing_data = pd.concat(dfs_ke, axis=0, ignore_index=True)
    
    def aggregate_editing_data(self, groupby_dimensions=True, groupby_dataset_splits=False, exclude_fact_queries=False, evaluate_generate_lengths=False):
        self.editing_data["valid_test_case_ratio"] = self.editing_data["valid_test_cases"] / self.editing_data["test_cases"]
        df = self.editing_data.copy()
        if exclude_fact_queries:
            df = df[df["dimension"] != "fact_queries"]

        df["experiment_count"] = 1
        aggregate_by = ["model", "editor", "dataset"]
        if groupby_dataset_splits:
            aggregate_by.append("dataset_split")
        #if groupby_dimensions:
        aggregate_by.append("dimension")
        
        def sum_dicts(dicts):
            result = {}
            for d in dicts:
                for key, value in d.items():
                    result[key] = result.get(key, 0) + (value if value is not None else 0)
            return result

        def mean_dicts(dicts):
            counts = {}
            totals = {}
            for d in dicts:
                for key, value in d.items():
                    totals[key] = totals.get(key, 0.0) + (value if value is not None else 0.0)
                    counts[key] = counts.get(key, 0.0) + 1.0
            return {key: totals[key] / counts[key] if counts[key] > 0.0 else 0.0 for key in totals}

        def divide_dict(d, divisor):
            return {k: v / divisor for k, v in d.items()} if divisor != 0 else {k: 0 for k in d}
        
        df = df.groupby(aggregate_by).agg({
            'test_cases': 'sum',
            'valid_test_cases': 'sum',
            'verify_test_case_time': 'sum',
            'edit_time': 'sum',
            'eval_time': 'sum',
            'accuracy': (lambda x: mean_dicts(x)) if evaluate_generate_lengths else 'mean',
            'valid_test_case_ratio': 'mean',
            'experiment_count': 'sum',
        })
        # we're aggregating accuracies that have already been aggregated over all test cases belonging to a given example,
        #    but not weighing them by the number of test cases for this example (and dimension)
        
        
        if not groupby_dimensions:
            aggregate_by.remove("dimension")
            df = df.groupby(aggregate_by).agg({
                'test_cases': 'sum',
                'valid_test_cases': 'sum',
                'verify_test_case_time': 'sum',
                'edit_time': 'sum',
                'eval_time': 'sum',
                'accuracy': (lambda x: mean_dicts(x)) if evaluate_generate_lengths else 'mean',
                'valid_test_case_ratio': 'mean',
                'experiment_count': 'sum',
            })

        self.aggregated_editing_data = df
        return
        
    @staticmethod
    def extrtact_metrics_from_control_result(doc):
        full_metrics = dict()

        def debug_doc(d):
            for key, val in d.items():
                if isinstance(val, dict):
                    print(f"{key}:")
                    for subkey, subval in val.items():
                        print(f"    {subkey}: {subval}")
                else:
                    print(f"{key}: {val}")

        #print("######## DEBUG DOC START ########")
        #debug_doc(doc)
        #print("######## DEBUG DOC END ########")

        # create aggregation map
        aggregation_map = defaultdict(set)
        
        def build_aggregation_map(group, hierarchy):
            if group in doc["group_subtasks"] and doc["group_subtasks"][group] is not None and len(doc["group_subtasks"][group]) > 0:
                for sub_group in doc["group_subtasks"][group]:
                    build_aggregation_map(sub_group, hierarchy + [group])
            else:
                aggregation_map[group].update(hierarchy)

        for group in doc["group_subtasks"]:
            build_aggregation_map(group, [])
        

        #print("######## DEBUG AGGREGATION MAP START ########")
        #for k, v in aggregation_map.items():
        #    print(k, v)
        #print("######## DEBUG AGGREGATION MAP END ########")

        full_metrics = dict()
        aggregate_metrics = dict()

        for task, samples in doc["n-samples"].items():
            if samples is not None:
                n_samples = samples["effective"]
                higher_is_better = doc["higher_is_better"][task]
                data = doc["results"][task]
                if "alias" in data:
                    del data["alias"]
                metric_names = [m for m in data.keys() if "stderr" not in m]
                metrics = {metric_name.replace(",none", ""): {"score": data[metric_name], "std_err": data[metric_name.replace(",", "_stderr,")], "n-samples": n_samples, "higher_is_better": higher_is_better[metric_name.replace(",none", "")]} for metric_name in metric_names}
                full_metrics[task] = metrics
                for supergroup in aggregation_map[task]:
                    #print(f"DEBUG add to {supergroup} with metrics={metrics}")
                    if supergroup in aggregate_metrics:
                        for metric, result in metrics.items():
                            aggregate_metrics[supergroup][metric]["score"] += (result["score"] * result["n-samples"])
                            if result["std_err"] is not None:
                                if aggregate_metrics[supergroup][metric]["std_err"] is not None:
                                    aggregate_metrics[supergroup][metric]["std_err"] += result["std_err"] * result["n-samples"]
                                else:
                                    aggregate_metrics[supergroup][metric]["std_err"] = result["std_err"] * result["n-samples"]
                            aggregate_metrics[supergroup][metric]["n-samples"] += result["n-samples"]
                    else:
                        _metrics = copy.deepcopy(metrics)
                        for _, result in _metrics.items():
                            result["score"] = result["score"] * result["n-samples"]
                            if result["std_err"] is not None:
                                result["std_err"] = result["std_err"] * metrics["n-samples"]
                        aggregate_metrics[supergroup] = _metrics
                    #print(f"DEBUG aggregate_metrics[{supergroup}]={aggregate_metrics[supergroup]}")
        
        # compute aggregate results
        for task, task_results in aggregate_metrics.items():
            for metric_name, metric_results in task_results.items():
                metric_results["score"] = metric_results["score"] / metric_results["n-samples"]
                if metric_results["std_err"] is not None:
                    metric_results["std_err"] = metric_results["std_err"] / metric_results["n-samples"]
                if metric_name == "acc":
                    assert metric_results["score"] == doc["results"][task]["acc,none"]
            full_metrics[task] = task_results
        
        #print("######## DEBUG EXTRACTED METRICS START ########")
        #for k, v in full_metrics.items():
        #    print(k, v)
        #print("######## DEBUG EXTRACTED METRICS END ########")
        return full_metrics


    def load_aggregated_control_data(self, base_path):
        # for performance reasons we are aggregating the data while loading it
        # df_lm = pd.DataFrame(columns=["model", "editor", "dataset", "batch_id", "eval_time", "task", "metric", "score", "std_err", "n-samples", "higher_is_better"])
        aggregated_data = defaultdict(lambda: {
            "n-samples": 0,
            "weighted_score": 0.0,
            "batch_count": 0,
            "eval_time": 0.0,
            "higher_is_better": set(),
        })

        for path in Path(base_path).iterdir():
            if str(path).endswith("_lm.parquet"):
                df = pd.read_parquet(path)
                for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading from path={path}"):
                    for task, metrics in EvalResult.extrtact_metrics_from_control_result(row).items():
                        if metrics is None:
                            assert task in row["group_subtasks"], "If its None at this point it must be an empty task group"
                            continue
                        for metric, data in metrics.items():
                            key = (row["model"], row["editor"], task, metric)

                            # convert perplexities back to log probabilities for aggregation
                            if "perplexity" in metric:
                                data["score"] = -np.log(data["score"])

                            aggregated_data[key]["n-samples"] += data["n-samples"]
                            aggregated_data[key]["weighted_score"] += data["score"] * data["n-samples"]
                            aggregated_data[key]["batch_count"] += 1
                            aggregated_data[key]["eval_time"] += row["eval_time"]
                            aggregated_data[key]["higher_is_better"].add(data["higher_is_better"])
        self.aggregated_control_data = pd.DataFrame([
            {"model": k[0], "editor": k[1], "task": k[2], "metric": k[3], **v}
            for k, v in aggregated_data.items()
        ])

        if len(self.aggregated_control_data) > 0:
            self.aggregated_control_data["score"] = self.aggregated_control_data["weighted_score"] / self.aggregated_control_data["n-samples"]
            # convert perplexities back to exponentiated values
            self.aggregated_control_data.loc[self.aggregated_control_data["metric"].str.contains("perplexity"), "score"] = np.exp(-self.aggregated_control_data.loc[self.aggregated_control_data["metric"].str.contains("perplexity"), "score"])

    def control_results_to_latex(self):
        df = self.aggregated_control_data.copy()
        df_pivot = df.pivot_table(
            index=["task", "metric"], 
            columns="editor", 
            values="score"
        ).reset_index()
        df_pivot = df_pivot.round(3)
        df_pivot = df_pivot.applymap(lambda x: '{:.3f}'.format(x) if isinstance(x, float) else x)
        print(df_pivot.astype(str).to_latex(index=False).replace("_", " "))

    def editing_results_to_latex(self):
        df = self.aggregated_editing_data.copy().reset_index()
        dataset_order = ["zsre", "CounterFact", "MQuAKE", "RippleEdits"]
        df["dataset"] = pd.Categorical(df["dataset"], categories=dataset_order, ordered=True)
        df_pivot = df.pivot_table(
            index=["dataset", "dimension"], 
            columns="editor", 
            values="accuracy"
        ).reset_index()
        df_pivot = df_pivot.round(3)
        df_pivot = df_pivot.applymap(lambda x: '{:.3f}'.format(x) if isinstance(x, float) else x)
        print(df_pivot.astype(str).to_latex(index=False).replace("_", " "))
