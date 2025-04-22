import pandas as pd
import numpy as np
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
        if groupby_dimensions:
            aggregate_by.append("dimension")
        
        def sum_dicts(dicts):
            result = {}
            for d in dicts:
                for key, value in d.items():
                    result[key] = result.get(key, 0) + (value if value is not None else 0)
            return result

        def divide_dict(d, divisor):
            return {k: v / divisor for k, v in d.items()} if divisor != 0 else {k: 0 for k in d}
        
        df = df.groupby(aggregate_by).agg({
            'test_cases': 'sum',
            'valid_test_cases': 'sum',
            'verify_test_case_time': 'sum',
            'edit_time': 'sum',
            'eval_time': 'sum',
            'accuracy': (lambda x: sum_dicts(x)) if evaluate_generate_lengths else 'sum',
            'valid_test_case_ratio': 'mean',
            'experiment_count': 'sum',
        })
        if evaluate_generate_lengths:
            df["accuracy"] = df.apply(lambda row: divide_dict(row["accuracy"], row["valid_test_cases"]), axis=1)
        else:
            df["accuracy"] = df["accuracy"] / df["valid_test_cases"]

        self.aggregated_editing_data = df
        return
        '''
        if evaluate_generate_lengths:
            df_plot = pd.DataFrame(columns=["editor", "dataset", "x", "y"])
            for index, row in df.iterrows():
                for k, v in row["accuracy"].items():
                    data = {
                        "editor": f"{index[1]}",
                        "dataset": f"{index[2]}",
                        "x": int(k),
                        "y": v,
                    }
                    if data["editor"] != "no-edit":
                        df_plot.loc[len(df_plot)] = data
            df_plot = df_plot.sort_values(by="x")

            # Plot with Plotly Express
            #fig = px.line(df_plot, x="x", y="y", color="group", markers=False,
            #            labels={"x": "Generate Length", "y": "KE Accuracy", "group": "Group"},
            #            title="KE Accuracy per Group")
            #fig.write_image("visualisations/generate_length.png", width=2400, height=1800)
            if groupby_dimensions:
                dimensions = ["attribute", "neighborhood", "default", "fact_queries", "Subject_Aliasing", "Relation_Specificity", "paraphrase", "Logical_Generalization", "Compositionality_II", "Compositionality_I", "Forgetfulness"]
            else:
                dimensions = ["MQuAKE", "RippleEdits", "zsre", "CounterFact"]
            
            editor_colors = {}
            color_palette = px.colors.qualitative.Set1

            color_map = {
                "context-retriever": "blue",
                "in-context": "green",
                "memit": "red",
                "no-edit": "grey"
            }

            fig = make_subplots(rows=2, cols=2, horizontal_spacing=0.1, vertical_spacing=0.1, subplot_titles=dimensions)
            for idx, dimension in enumerate(dimensions):
                filtered_df = df_plot[df_plot["dataset"] == dimension]

                for editor in filtered_df["editor"].unique():
                    if editor not in editor_colors:
                        editor_colors[editor] = color_palette[len(editor_colors) % len(color_palette)]
                    show_legend = (idx == 0)
                    group_df = filtered_df[filtered_df["editor"] == editor]
                    row, col = divmod(idx, 2)  # Get row and column index
                    fig.add_trace(
                        go.Scatter(x=group_df["x"], y=group_df["y"], mode="lines", name=editor, line=dict(color=editor_colors[editor]), showlegend=show_legend),
                        row=row + 1, col=col + 1
                    )

            fig.update_xaxes(title_text="Generate Length", row=2, col=1)
            fig.update_xaxes(title_text="Generate Length", row=2, col=2)
            fig.update_yaxes(title_text="KE Accuracy", row=1, col=1)
            fig.update_yaxes(title_text="KE Accuracy", row=2, col=1)
            fig.update_xaxes(showticklabels=False, row=1, col=1)
            fig.update_xaxes(showticklabels=False, row=1, col=2)
            #fig.update_yaxes(showticklabels=False, row=1, col=2)
            #fig.update_yaxes(showticklabels=False, row=2, col=2)

            fig.update_layout(
                title="KE Accuracy per Group",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
                ),
                margin=dict(l=10, r=10, t=60, b=40),
                meta=dict(mathjax=False),
            )
            fig.write_image("visualisations/generate_length/subplots.pdf", width=450, height=450, engine="kaleido")
        '''
    
    @staticmethod
    def extrtact_metrics_from_control_eval_result(doc):
        full_metrics = dict()
        # print(doc)
        for task in doc["results"].keys():
            if doc["results"][task] is None:
                continue
            higher_is_better = doc["higher_is_better"][task]
            data = doc["results"][task]
            del data["alias"]
            metric_names = [m for m in data.keys() if "stderr" not in m]
            metrics = {metric_name.replace(",none", ""): {"score": data[metric_name], "std_err": data[metric_name.replace(",", "_stderr,")], "n-samples": doc["n-samples"][task]["effective"], "higher_is_better": higher_is_better[metric_name.replace(",none", "")]} for metric_name in metric_names}
            full_metrics[task] = metrics
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
                    for task, metrics in EvalResult.extrtact_metrics_from_control_eval_result(row).items():
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
