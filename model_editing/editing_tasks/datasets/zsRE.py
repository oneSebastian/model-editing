import json
from typing import Optional
from tqdm import tqdm
from ..util import Dataset, Example, Fact, Query, TestCase, TestCondition, QueryType


class ZSREDataset(Dataset):
    def __init__(self, dataset_name: str, examples: list):
        super().__init__(dataset_name, examples)
    
    @staticmethod
    def parse_example(example_id: int, data_dict: dict, force_query_type: Optional[QueryType]):
        fact = Fact(
                prompt = data_dict["src"],
                subject = data_dict["subject"],
                target = data_dict["answers"][0],
                original_target = None,
                fact_query = Query(
                    prompt = data_dict["src"],
                    answers = data_dict["answers"],
                    query_type=QueryType.ARG if force_query_type is None else force_query_type,
                ),
            )

        test_cases = []
        test_cases.append(TestCase(
                    test_dimension = "efficacy",
                    test_condition = TestCondition.OR,
                    test_queries = [Query(
                        prompt = data_dict["src"],
                        answers = data_dict["answers"],
                        query_type=QueryType.ARG if force_query_type is None else force_query_type,
                    )],
                    condition_queries = [],
            ))
        test_cases.append(TestCase(
                    test_dimension = "paraphrase",
                    test_condition = TestCondition.OR,
                    test_queries = [Query(
                        prompt = data_dict["rephrase"],
                        answers = data_dict["answers"],
                        query_type=QueryType.ARG if force_query_type is None else force_query_type,
                    )],
                    condition_queries = [],
            ))
        test_cases.append(TestCase(
                    test_dimension = "neighborhood",
                    test_condition = TestCondition.OR,
                    test_queries = [Query(
                        prompt = data_dict["loc"].replace("nq question: ", "") + "?",
                        answers = [data_dict["loc_ans"]],
                        query_type=QueryType.ARG if force_query_type is None else force_query_type,
                    )],
                    condition_queries = [],
            ))

        return Example(
            example_id=example_id,
            example_type="default",
            facts=[fact],
            test_cases=test_cases,
        )

    @staticmethod
    def from_file(data_directory, force_query_type: Optional[QueryType]=None, limit=0):
        with open(f"{data_directory}zsre_mend_eval.json", 'r') as f:
            examples = json.load(f)
        example_list = []
        for i, example_data in tqdm(enumerate(examples), desc=f"Reading data from file: {data_directory}zsre_mend_eval.json"):
            if limit and i >= limit:
                break
            #if i % 100 == 0:
            #    print(f"Read {i} examples from file.", flush=True)
            
            example = ZSREDataset.parse_example(i, example_data, force_query_type)
            example_list.append(example)
        return ZSREDataset("zsre", example_list)

