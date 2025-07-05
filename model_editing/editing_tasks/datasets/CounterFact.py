import json
import random
from typing import Optional
from tqdm import tqdm
from ..util import Dataset, Example, Fact, Query, TestCase, TestCondition, QueryType


class CounterFactDataset(Dataset):
    def __init__(self, dataset_name: str, examples: list):
        super().__init__(dataset_name, examples)
    
    @staticmethod
    def parse_example(example_id: int, data_dict: dict, force_query_type, max_test_cases_by_dimension=5):
        fact = Fact(
                example_id=example_id,
                prompt = data_dict["requested_rewrite"]["prompt"].format(data_dict["requested_rewrite"]["subject"]),
                subject = data_dict["requested_rewrite"]["subject"],
                target = data_dict["requested_rewrite"]["target_new"]["str"],
                original_target = data_dict["requested_rewrite"]["target_true"]["str"],
                fact_query = Query(
                    query_id=(example_id, -1),
                    prompt = data_dict["requested_rewrite"]["prompt"].format(data_dict["requested_rewrite"]["subject"]),
                    answers = [data_dict["requested_rewrite"]["target_new"]["str"]],
                    query_type=QueryType.MC if force_query_type is None else force_query_type,
                    answer_options=[data_dict["requested_rewrite"]["target_new"]["str"], data_dict["requested_rewrite"]["target_true"]["str"]]
                ),
            )

        test_cases = []
        test_case_id = 0
        for dimension in ["paraphrase", "neighborhood", "attribute"]:
            for i, prompt in enumerate(data_dict[f"{dimension}_prompts"]):
                if i >= max_test_cases_by_dimension:
                    break
                test_cases.append(TestCase(
                    test_case_id = test_case_id,
                    test_dimension = dimension,
                    test_condition = TestCondition.OR,
                    test_queries = [Query(
                        query_id = (example_id, test_case_id),
                        prompt = prompt,
                        answers = [data_dict["requested_rewrite"]["target_true"]["str"] if dimension == "neighborhood" else data_dict["requested_rewrite"]["target_new"]["str"]],
                        query_type=QueryType.MC if force_query_type is None else force_query_type,
                        answer_options=[data_dict["requested_rewrite"]["target_new"]["str"], data_dict["requested_rewrite"]["target_true"]["str"]],
                    )],
                    condition_queries = [],
                ))
                test_case_id += 1

        return Example(
            example_id=data_dict["case_id"],
            example_type="default",
            facts=[fact],
            test_cases=test_cases,
        )

    @staticmethod
    def from_file(data_directory, force_query_type: Optional[QueryType]=None, dev_split=False, dev_split_size=512):
        with open(f"{data_directory}/counterfact.json", 'r') as f:
            examples = json.load(f)
        example_list = []
        for i, example_data in tqdm(enumerate(examples), desc=f"Reading data from file: {data_directory}counterfact.json"):
            example = CounterFactDataset.parse_example(i, example_data, force_query_type)
            example_list.append(example)
        # sample dev split or its complement
        # use constant local seed for dev split creation
        local_rng = random.Random(42)
        local_rng.shuffle(example_list)
        if dev_split:
            example_list = example_list[:dev_split_size]
        else:
            example_list = example_list[dev_split_size:]
        return CounterFactDataset("CounterFact", example_list)
    