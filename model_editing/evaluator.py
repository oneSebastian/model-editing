import time
import math
import random
from tqdm import tqdm
from itertools import chain
from typing import Union
import pandas as pd
import numpy as np
from collections import defaultdict
from lm_eval import evaluate
from .models import load_model
from .model_editor import NoEditModel, InContextModel, MEMITModel, ContextRetrieverModel, LORAModel
from .editing_tasks.util import TestCondition, QueryType
from copy import deepcopy

class DimensionResult:
    def __init__(self, accuracy: Union[float, dict]=0.0, test_cases=0, valid_test_cases=0):
        self.test_cases = test_cases
        self.valid_test_cases = valid_test_cases
        self.accuracy = accuracy
    
    def __str__(self):
        return f"DimensionResult: test_cases={self.test_cases}, valid_test_cases={self.valid_test_cases}, accuracy={self.accuracy}"


class ExampleResult:
    def __init__(self, dataset_split, batch_id, batch_position, example_id):
        self.batch_id = batch_id
        self.dataset_split = dataset_split
        self.batch_position = batch_position
        self.example_id = example_id
        self.successful_edit = 0.0
        self.dimension_results = defaultdict(DimensionResult)
        self.verify_test_case_time = 0.0
        self.edit_time = 0.0
        self.eval_time = 0.0
    
    def __str__(self):
        return f"ExampleResult: dataset_split={self.dataset_split}, example_id={self.example_id}, successful_edit={self.successful_edit}, dimension_results={[(d, str(r)) for d, r in self.dimension_results.items()]}"


class Evaluator:
    def __init__(
            self,
            experiment_name,
            model_name,
            editor_name,
            control_task_dict,
            control_task_data,
            control_data_boundaries,
            dataset, dataset_sample_size=None,
            edit_batch_size=1,
            sequential_editing=False,
            evaluate_generate_lengths=False,
            use_chat_template=False,
            eval_batch_size=8,
            random_seed=42,
            save_path=None,
            dev_split=False,
            device="cuda",
            verbose=False,
            ):
        self.experiment_name
        self.model_name = model_name
        self.device=device
        self.editor_name = editor_name
        self.control_task_dict = control_task_dict
        self.control_task_data = control_task_data
        self.eval_batch_size = eval_batch_size
        self.batch_split_sizes = None
        self.dataset = dataset
        self.random_seed = random_seed
        random.seed(self.random_seed)
        if dataset_sample_size:
            self.dataset.sample(dataset_sample_size, self.random_seed)
        self.edit_batch_size = edit_batch_size
        self.evaluate_generate_lengths = evaluate_generate_lengths
        self.use_chat_template = use_chat_template
        self.save_path = save_path
        self.result_path = f"{self.save_path}/{self.experiment_name}"
        self.editing_results = []
        self.lm_results = []
        self.num_batches = math.ceil(len(self.dataset.examples) / edit_batch_size)
        self.verbose=verbose

        with open(self.result_path + '.txt', "w") as f:
            f.write(f"Experiment name: {self.experiment_name}\n")
            f.write(f"model_name: {self.model_name}\n")
            f.write(f"editor_name: {self.editor_name}\n")
            f.write(f"edit batch size: {self.edit_batch_size}\n")
            f.write(f"use_chat_template: {self.use_chat_template}\n")
            f.write(f"dev split: {dev_split}\n")
            f.write(f"dataset name: {self.dataset.dataset_name}\n")

        if self.evaluate_generate_lengths and self.save_path is None:
            raise ValueError("evaluate_generate_length requires a save_path to write outputs to.")
        if self.evaluate_generate_lengths:
            self.df_generate_lengths = pd.DataFrame(columns=["model", "editor", "dataset", "dimension", "batch_id", "example_id", "query_prompt", "correct_answers", "generated_answer", "query_result"])

        # create batch boundaries of datasets for lm_eval tasks
        if self.control_task_dict is not None:
            self.lm_eval_batch_boundaries = dict()
            self.batch_split_sizes = {i: defaultdict(dict) for i in range(self.num_batches)}

            def compute_lm_batch_boundaries(_task, _task_name, _subtask_name=None):
                for split in self.control_task_data[(_task_name, _subtask_name)].keys():
                    data_start, data_end = control_data_boundaries[(_task_name, _subtask_name)][split]
                    n = data_end - data_start
                    batch_size = n // self.num_batches
                    remainder = n % self.num_batches
                    batch_start = data_start
                    for i in range(self.num_batches):
                        batch_end = batch_start + batch_size + (1 if i < remainder else 0)
                        if batch_end > batch_start:
                            self.batch_split_sizes[i][(_task_name, _subtask_name)][split] = (batch_start, batch_end)
                        else:
                            self.batch_split_sizes[i][(_task_name, _subtask_name)][split] = None
                        batch_start = batch_end

            for task_name, task in self.control_task_dict.items():
                if isinstance(task, dict):
                    for subtask_name, subtask in task.items():
                        compute_lm_batch_boundaries(_task=subtask, _task_name=task_name, _subtask_name=subtask_name)
                else:
                    compute_lm_batch_boundaries(_task=task, _task_name=task_name)
        
        model, tokenizer = load_model(model_name, device=device)
        
        #if editor_name == 'mend':
        #    self.model_editor = MENDModelEditor(query_executor)
        #if editor_name == 'rome':
        #    self.model_editor = ROMEModelEditor(query_executor)
        if self.editor_name == 'no-edit':
            self.lm = NoEditModel(model, model_name, tokenizer, batch_size=self.eval_batch_size, use_chat_template=self.use_chat_template, verbose=self.verbose, log_path=self.result_path + ".txt")
        elif self.editor_name == 'memit':
            self.lm = MEMITModel(model, model_name, tokenizer, batch_size=self.eval_batch_size, use_chat_template=self.use_chat_template, verbose=self.verbose, log_path=self.result_path + ".txt")
        elif self.editor_name == 'in-context':
            self.lm = InContextModel(model, model_name, tokenizer, batch_size=self.eval_batch_size, use_chat_template=self.use_chat_template, verbose=self.verbose, log_path=self.result_path + ".txt")
        elif self.editor_name == 'context-retriever':
            self.lm = ContextRetrieverModel(model, model_name, tokenizer, batch_size=self.eval_batch_size, use_chat_template=self.use_chat_template, verbose=self.verbose, log_path=self.result_path + ".txt")
        elif self.editor_name == 'lora':
            self.lm = LORAModel(model, model_name, tokenizer, batch_size=self.eval_batch_size, use_chat_template=self.use_chat_template, verbose=self.verbose, log_path=self.result_path + ".txt")
        else:
            raise ValueError(f"{editor_name} is not a supported model editor")
    
    def _evaluate_examples(self, batch_id, examples):
        print("DEBUG Verify test cases")
        # check condition queries to determine which test_cases are valid.
        verify_test_case_start_time = time.perf_counter()
        batch_results = []
        queries = defaultdict(list)
        # collect all condition queries for this batch
        for example_idx, example in enumerate(examples):
            batch_results.append(ExampleResult(dataset_split=example.dataset_split, batch_id=batch_id, batch_position=example_idx, example_id=example.example_id))
            example_result = batch_results[example_idx]
            for test_case_idx, test_case in enumerate(example.test_cases):
                example_result.dimension_results[test_case.test_dimension].test_cases += 1
                for query in test_case.condition_queries:
                    queries[query.type].append(((example_idx, test_case_idx), query))
        # TODO: allow for non-generate condition queries
        if len(queries) == 1 and QueryType.GEN not in queries or len(queries) > 1:
            raise NotImplementedError("For now only generate queries are supported as condition queries")
        
        # execute all condition queries for this batch
        generate_results = self.lm.execute_generate_queries([q[1] for q in queries[QueryType.GEN]], answer_length=20, evaluate_generate_lengths=False)

        # evaluate query results on a test case basis
        # TODO: allow for non-generate condition queries
        condition_results = defaultdict(list)
        for i, ((example_idx, test_case_idx), _) in enumerate(queries[QueryType.GEN]):
            condition_results[(example_idx, test_case_idx)].append(generate_results[i])
        for example_idx, example in enumerate(examples):
            example_result = batch_results[example_idx]
            valid_test_cases = []
            for test_case_idx, test_case in enumerate(example.test_cases):
                if all(condition_results[(example_idx, test_case_idx)]):
                    example_result.dimension_results[test_case.test_dimension].valid_test_cases += 1
                    valid_test_cases.append(test_case)
            if self.verbose:
                print(f"DEBUG example {example_idx} has {len(valid_test_cases)}/{len(example.test_cases)} valid test cases")
            example.test_cases = valid_test_cases
        verify_test_case_time = time.perf_counter() - verify_test_case_start_time
        

        # Edit model
        print("DEBUG Edit model")
        start_time = time.perf_counter()
        self.lm.edit_model([fact for _example in examples for fact in _example.facts])
        edit_time = time.perf_counter() - start_time

        # Run Knowledge Editing Evaluation
        test_queries_start_time = time.perf_counter()

        # Run fact queries
        print("DEBUG run fact queries")
        queries = defaultdict(list)
        for example_idx, example in enumerate(examples):
            example_result = batch_results[example_idx]
            assert batch_results[example_idx].batch_position == example_idx, "We should always iterate over the same examples in the same order."
            for fact in example.facts:
                queries[fact.query.type].append(((example_idx, -1), fact.query))
                example_result.dimension_results["fact_queries"].test_cases += 1
                example_result.dimension_results["fact_queries"].valid_test_cases += 1
        results = {
            QueryType.GEN: self.lm.execute_generate_queries([q[1] for q in queries[QueryType.GEN]], answer_length=20, evaluate_generate_lengths=self.evaluate_generate_lengths),
            QueryType.MC: self.lm.execute_options_queries([q[1] for q in queries[QueryType.MC]]),
            QueryType.ARG: self.lm.execute_argmax_queries([q[1] for q in queries[QueryType.ARG]]),
        }
        fact_query_results = defaultdict(list)
        for query_type in QueryType:
            query_type_results = results[query_type]
            if self.evaluate_generate_lengths and query_type == QueryType.GEN:
                for i, ((example_idx, test_case_idx), _) in enumerate(queries[query_type]):
                    fact_query_results[(example_idx, test_case_idx)].append((query_type, (query_type_results[0][i], query_type_results[1][i])))
            else:
                for i, ((example_idx, test_case_idx), _) in enumerate(queries[query_type]):
                    fact_query_results[(example_idx, test_case_idx)].append((query_type, query_type_results[i]))
        for example_idx, example in enumerate(examples):
            example_result = batch_results[example_idx]
            for query_type, result in fact_query_results[(example_idx, -1)]:
                if query_type == QueryType.ARG:
                    assert type(result) == list
                    assert all([isinstance(_, bool) for _ in result])
                    accuracy = sum(result) / len(result)
                else:
                    if self.evaluate_generate_lengths:
                        query_results, query_replies = result
                        data = {
                            'model': self.model_name,
                            'editor': self.editor_name,
                            'dataset': self.dataset.dataset_name,
                            'dimension': "fact_queries",
                            'batch_id': example_result.batch_id,
                            'example_id': example_result.example_id,
                            'query_prompt': query_replies["query"]["prompt"],
                            'correct_answers': query_replies["query"]["answers"],
                            'generated_answer': query_replies["model_answer"],
                            'query_result': {str(k): str(v) for k, v in query_results.items()},
                        }
                        self.df_generate_lengths.loc[len(self.df_generate_lengths)] = data
                        accuracy = {length: 1.0 if length_result else 0.0 for length, length_result in query_results.items()}
                    else:
                        accuracy = 1.0 if result else 0.0
            example_result.dimension_results["fact_queries"].accuracy = accuracy
        
        # Run test queries
        print("DEBUG run test queries")
        queries = defaultdict(list)
        for example_idx, example in enumerate(examples):
            example_result = batch_results[example_idx]
            assert example_result.batch_position == example_idx, "We should always iterate over the same examples in the same order."
            for test_case_idx, test_case in enumerate(example.test_cases):
                for query in test_case.test_queries:
                    queries[query.type].append(((example_idx, test_case_idx), query))
        results = {
            QueryType.GEN: self.lm.execute_generate_queries([q[1] for q in queries[QueryType.GEN]], answer_length=20, evaluate_generate_lengths=self.evaluate_generate_lengths),
            QueryType.MC: self.lm.execute_options_queries([q[1] for q in queries[QueryType.MC]]),
            QueryType.ARG: self.lm.execute_argmax_queries([q[1] for q in queries[QueryType.ARG]]),
        }
        test_results = defaultdict(list)
        for query_type in QueryType:
            query_type_results = results[query_type]
            if self.evaluate_generate_lengths and query_type == QueryType.GEN:
                for i, ((example_idx, test_case_idx), _) in enumerate(queries[query_type]):
                    test_results[(example_idx, test_case_idx)].append((query_type_results[0][i], query_type_results[1][i]))
            else:
                for i, ((example_idx, test_case_idx), _) in enumerate(queries[query_type]):
                    # print(f"DEBUG: add to test_results: query_type={query_type}, example_idx={example_idx}, test_case_idx={test_case_idx}")
                    test_results[(example_idx, test_case_idx)].append(query_type_results[i])
        
        # asses test case results
        for example_idx, example in enumerate(examples):
            example_result = batch_results[example_idx]
            for test_case_idx, test_case in enumerate(example.test_cases):
                dimension = test_case.test_dimension
                test_case_results = test_results[(example_idx, test_case_idx)]
                if self.evaluate_generate_lengths:
                    # report generate outcomes
                    for query_results, query_replies in test_case_results:
                        data = {
                            'model': self.model_name,
                            'editor': self.editor_name,
                            'dataset': self.dataset.dataset_name,
                            'dimension': dimension,
                            'batch_id': example_result.batch_id,
                            'example_id': example_result.example_id,
                            'query_prompt': query_replies["query"]["prompt"],
                            'correct_answers': query_replies["query"]["answers"],
                            'generated_answer': query_replies["model_answer"],
                            'query_result': {str(k): str(v) for k, v in query_results.items()},
                        }
                        self.df_generate_lengths.loc[len(self.df_generate_lengths)] = data

                    # compute test case results
                    length_results = defaultdict(list)
                    for test_case_result, _ in test_case_results:
                        for length, length_result in test_case_result.items():
                            length_results[length].append(length_result)
                    accuracy = {}
                    for length, length_result in length_results.items():
                        if test_case.test_condition == TestCondition.OR and True in length_result:
                            accuracy[length] = 1.0
                        elif test_case.test_condition == TestCondition.AND and False not in length_result:
                            accuracy[length] = 1.0
                        else:
                            accuracy[length] = 0.0
                else:
                    if test_case.test_condition == TestCondition.ACC:
                        assert all([isinstance(result, list) for result in test_case_results])
                        assert query_type == QueryType.ARG
                        query_accuracies = [sum(result) / len(result) for result in test_case_results]
                        if not test_case_results:
                            assert example_result.dimension_results[dimension].valid_test_cases == 0
                            # TODO: does this case ever occur? Should accuracy be 0.0 for correct aggregation?
                            accuracy = 0.0
                        else:
                            accuracy = sum(query_accuracies) / len(query_accuracies)
                    else:
                        assert list not in [type(result) for result in test_case_results]
                        if test_case.test_condition == TestCondition.OR and True in test_case_results:
                            accuracy = 1.0
                        elif test_case.test_condition == TestCondition.AND and False not in test_case_results:
                            accuracy = 1.0
                        else:
                            accuracy= 0.0
                example_result.dimension_results[dimension].accuracy = accuracy
        
        # record editing time results
        test_queries_time = time.perf_counter() - test_queries_start_time
        for example_result in batch_results:
            example_result.verify_test_case_time = verify_test_case_time
            example_result.edit_time = edit_time
            example_result.eval_time = test_queries_time
        
        # run control_tasks on edited model
        print("DEBUG run control tasks")
        if self.control_task_dict is not None:
            # first prepare control_task data for this batch
            batch_task_dict = deepcopy(self.control_task_dict)

            # insert lm eval data for this batch
            empty = True
            skip_tasks = []
            for task_name, task in self.control_task_dict.items():
                if isinstance(task, dict):
                    for subtask_name, subtask in task.items():
                        for split, boundary in self.batch_split_sizes[batch_id][(task_name, subtask_name)].items():
                            if boundary:
                                batch_task_dict[task_name][subtask_name].dataset[split] = self.control_task_data[(task_name.group, subtask_name)][split].select(range(boundary[0], boundary[1]))
                                empty = False
                            else:
                                skip_tasks.append((task_name, subtask_name))
                else:
                    for split, boundary in self.batch_split_sizes[batch_id][(task_name, None)].items():
                        if boundary:
                            batch_task_dict[task_name].dataset[split] = self.control_task_data[(task_name, None)][split].select(range(boundary[0], boundary[1]))
                            empty = False
                        else:
                            skip_tasks.append((task_name,))
            
            # skip empty tasks
            for entry in skip_tasks:
                if len(entry) == 2:
                    del batch_task_dict[entry[0]][entry[1]]
                elif len(entry) == 1:
                    del batch_task_dict[entry[0]]

            # execute control_tasks with data for this batch
            if not empty:
                start_time = time.perf_counter()
                results = evaluate(
                    lm=self.lm,
                    task_dict=batch_task_dict,
                    bootstrap_iters=0,
                    apply_chat_template=self.lm.use_chat_template,
                )
                duration = time.perf_counter() - start_time

                # clean data again for next split
                for task_name, task in self.control_task_dict.items():
                    if isinstance(task, dict):
                        for subtask_name, subtask in task.items():
                            for split, boundary in self.batch_split_sizes[batch_id][(task_name, subtask_name)].items():
                                if boundary:
                                    del batch_task_dict[task_name][subtask_name].dataset[split]
                    else:
                        for split, boundary in self.batch_split_sizes[batch_id][(task_name, None)].items():
                            if boundary:
                                del batch_task_dict[task_name].dataset[split]

                def nan_to_numpy_nan(d):
                    if isinstance(d, dict):
                        return {k: nan_to_numpy_nan(v) for k, v in d.items()}
                    elif isinstance(d, list):
                        return [nan_to_numpy_nan(v) for v in d]
                    elif d == "N/A":
                        return np.nan
                    return d

                lm_batch_results = {
                    "batch_id": batch_id,
                    "eval_time": duration,
                    "results": nan_to_numpy_nan(results["results"]),
                    "higher_is_better": results["higher_is_better"],
                    "n-samples": results["n-samples"],
                    "group_subtasks": results["group_subtasks"],
                }
            else:
                lm_batch_results = None
        else:
            lm_batch_results = None
        print("DEBUG batch done")

        # Restore model
        # TODO: we don't support sequential editing at the moment
        self.lm.restore_model()

        return batch_results, lm_batch_results

    
    def evaluate(self):
        # record evaluation time for entire dataset
        start_time = time.perf_counter()

        # TODO: implement sequential batch editing
        batched_examples = [self.dataset.examples[j * self.edit_batch_size : min((j + 1) * self.edit_batch_size, len(self.dataset.examples))] for j in range(self.num_batches)]
        assert self.num_batches == len(batched_examples)

        for i, examples in enumerate(batched_examples):
            print(f"##### evaluate batch {i + 1}/{len(batched_examples)} #####", flush=True)
            editing_batch_results, lm_batch_results = self._evaluate_examples(i, examples)
            self.editing_results += editing_batch_results
            if lm_batch_results is not None:
                self.lm_results.append(lm_batch_results)

        # end time recording
        end_time = time.perf_counter()
        self.eval_time = end_time - start_time
        
    
    def save_results(self):
        # self.results is a list of ExampleResult objects
        editing_df = pd.DataFrame(columns=[
            'model', 'editor', 'dataset', 'dataset_split', 'batch_id', 'batch_position', 'example_id', 'successful_edit', 'dimension', 'test_cases', 'valid_test_cases',
                'accuracy', 'verify_test_case_time', 'edit_time', 'eval_time',
            ])
        lm_df = pd.DataFrame(columns=['model', 'editor', 'dataset', 'batch_id', 'eval_time', 'results', 'higher_is_better', 'n-samples', 'group_subtasks'])
                                    
        for example_result in self.editing_results:
            for dimension, dimension_result in example_result.dimension_results.items():
                if self.evaluate_generate_lengths:
                    if isinstance(dimension_result.accuracy, dict):
                        accuracy = {str(k): v for k, v in dimension_result.accuracy.items()}
                    else:
                        assert dimension_result.accuracy == 0.0
                        assert dimension_result.valid_test_cases == 0
                        accuracy = {str(k): 0.0 for k in  range(1, 64 + 1)}
                else:
                    accuracy = dimension_result.accuracy
                data = {
                    'model': self.model_name,
                    'editor': self.editor_name,
                    'dataset': self.dataset.dataset_name,
                    'dataset_split': example_result.dataset_split,
                    'batch_id': example_result.batch_id,
                    'batch_position': example_result.batch_position,
                    'example_id': example_result.example_id,
                    'successful_edit': example_result.successful_edit,
                    'dimension': dimension,
                    'test_cases': dimension_result.test_cases,
                    'valid_test_cases': dimension_result.valid_test_cases,
                    'accuracy': accuracy,
                    'verify_test_case_time': example_result.verify_test_case_time,
                    'edit_time': example_result.edit_time,
                    'eval_time': example_result.eval_time,
                }
                editing_df.loc[len(editing_df)] = data

        for result in self.lm_results:
            data = {
                'model': self.model_name,
                'editor': self.editor_name,
                'dataset': self.dataset.dataset_name,
                'batch_id': result['batch_id'],
                'eval_time': result['eval_time'],
                'results': result['results'],
                'higher_is_better': result['higher_is_better'],
                'n-samples': result['n-samples'],
                'group_subtasks': result['group_subtasks'],
            }
            lm_df.loc[(len(lm_df))] = data
    
        # save full results for future use
        editing_df = editing_df.replace("N/A", float("nan"))
        lm_df = lm_df.replace("N/A", float("nan"))
        editing_df.to_parquet(self.result_path + '_ke.parquet')
        lm_df.to_parquet(self.result_path + '_lm.parquet')
        if self.evaluate_generate_lengths:
            self.df_generate_lengths.to_parquet(self.result_path + '_generate_lengths')
        
        with open(self.result_path + '.txt', "a") as f:
            # write out evaluation time
            minutes, seconds = divmod(self.eval_time, 60)
            hours, minutes = divmod(minutes, 60)
            f.write(f"Total evaluation time: {hours}:{minutes}:{seconds}\n\n")
