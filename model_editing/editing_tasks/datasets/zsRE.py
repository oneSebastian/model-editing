import json
import random
from typing import Optional
from tqdm import tqdm
from ..util import Dataset, Example, Fact, Query, TestCase, TestCondition, QueryType
        

class ZSREDataset(Dataset):
    def __init__(self, dataset_name: str, examples: list):
        super().__init__(dataset_name, examples)
    
    @staticmethod
    def parse_example(example_id: int, data_dict: dict, force_query_type: Optional[QueryType]):
        fact = Fact(
                example_id = example_id,
                prompt = data_dict["src"],
                subject = data_dict["subject"],
                target = data_dict["answers"][0],
                original_target = None,
                fact_query = Query(
                    query_id = (example_id, -1),
                    prompt = data_dict["src"],
                    answers = data_dict["answers"],
                    query_type=QueryType.ARG if force_query_type is None else force_query_type,
                ),
            )

        test_cases = []
        test_cases.append(TestCase(
                    test_case_id = 0,
                    test_dimension = "efficacy",
                    test_condition = TestCondition.OR,
                    test_queries = [Query(
                        query_id = (example_id, 0),
                        prompt = data_dict["src"],
                        answers = data_dict["answers"],
                        query_type=QueryType.ARG if force_query_type is None else force_query_type,
                    )],
                    condition_queries = [],
            ))
        test_cases.append(TestCase(
                    test_case_id = 1,
                    test_dimension = "paraphrase",
                    test_condition = TestCondition.OR,
                    test_queries = [Query(
                        query_id = (example_id, 1),
                        prompt = data_dict["rephrase"],
                        answers = data_dict["answers"],
                        query_type=QueryType.ARG if force_query_type is None else force_query_type,
                    )],
                    condition_queries = [],
            ))
        test_cases.append(TestCase(
                    test_case_id = 2,
                    test_dimension = "neighborhood",
                    test_condition = TestCondition.OR,
                    test_queries = [Query(
                        query_id = (example_id, 2),
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
    def from_file(data_directory, force_query_type: Optional[QueryType]=None, dev_split=False, dev_split_size=512):
        with open(f"{data_directory}/zsre_mend_eval.json", 'r') as f:
            examples = json.load(f)
        example_list = []
        for i, example_data in tqdm(enumerate(examples), desc=f"Reading data from file: {data_directory}zsre_mend_eval.json"):
            #if i % 100 == 0:
            #    print(f"Read {i} examples from file.", flush=True)
            example = ZSREDataset.parse_example(i, example_data, force_query_type)
            example_list.append(example)
        
        # sample dev split or its complement
        # use constant local seed for dev split creation
        local_rng = random.Random(42)
        local_rng.shuffle(example_list)
        if dev_split:
            example_list = example_list[:dev_split_size]
        else:
            example_list = example_list[dev_split_size:]
        return ZSREDataset("zsre", example_list)

