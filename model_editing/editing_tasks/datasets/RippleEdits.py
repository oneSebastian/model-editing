import json
from typing import Optional
from tqdm import tqdm
from ..util import Dataset, Example, Fact, Query, TestCase, TestCondition, QueryType


class RippleEditsDataset(Dataset):
    def __init__(self, dataset_name: str, examples: list):
        super().__init__(dataset_name, examples)

    @staticmethod
    def parse_example(i: int, split: str, data_dict: dict, force_query_type: Optional[QueryType]):
        if "original_fact" in data_dict["edit"]:
            assert data_dict["edit"]["fact_prompt"] == data_dict["edit"]["fact_query"]["prompt"] == data_dict["edit"]["original_fact"]["fact_prompt"] == data_dict["edit"]["original_fact"]["fact_query"]["prompt"], "Changed answers should not affect the fact prompt."
        else:
            assert data_dict["edit"]["fact_prompt"] == data_dict["edit"]["fact_query"]["prompt"], "Changed answers should not affect the fact prompt."
        # TODO: then I can get rid of some redundancy
        if "original_fact" in data_dict["edit"]:
            original_target = data_dict["edit"]["original_fact"]["target_label"],
        else:
            original_target = None

        fact = Fact(
            data_dict["edit"]["fact_prompt"],
            data_dict["edit"]["subject_label"],
            data_dict["edit"]["target_label"],
            original_target,
            Query.from_dict(data_dict["edit"]["fact_query"], query_type=QueryType.GEN if force_query_type is None else force_query_type),
        )
        if fact.query is None:
            #print("data_dict:", data_dict)
            #print("#" * 100)
            #print(fact.to_dict())
            #print("#" * 100)
            #print(data_dict["edit"]["fact_query"])
            print(f"Skipping example {i} during loading because of no valid fact query.")
            #exit()
            return None

        # load test cases
        test_cases = []
        test_case_id = 0
        axes = ['Relation_Specificity', 'Logical_Generalization', 'Subject_Aliasing', 'Compositionality_I', 'Compositionality_II', 'Forgetfulness']
        for axis in axes:
            for test_case_data in data_dict[axis]:
                if test_case_data["test_condition"] == "OR":
                    test_condition = TestCondition.OR
                elif test_case_data["test_condition"] == "AND":
                    test_condition = TestCondition.AND
                else:
                    msg = f"test_condition must be OR or AND; {test_case_data['test_condition']} is not a valid value."
                    raise ValueError(msg)
                test_queries = [Query.from_dict(query, query_type=QueryType.GEN if force_query_type is None else force_query_type) for query in test_case_data["test_queries"]]
                test_queries = [query for query in test_queries if query is not None]
                # RippleEdits condition queries are for now always treated as generate queries to ease comparability
                condition_queries = [Query.from_dict(query, query_type=QueryType.GEN) for query in test_case_data["condition_queries"]]
                condition_queries = [query for query in condition_queries if query is not None]
                test_case = TestCase(
                    test_case_id=test_case_id,
                    test_dimension=axis,
                    test_condition=test_condition,
                    test_queries=test_queries,
                    condition_queries=condition_queries,
                )
                test_case_id += 1
                test_cases.append(test_case)

        if len(test_cases) == 0:
            return None
        else:
            return Example(
                example_id=i,
                example_type=data_dict["example_type"],
                facts=[fact],
                test_cases=test_cases,
                dataset_split=split,
            )

    @staticmethod
    def from_file(data_directory, force_query_type: Optional[QueryType]=None, split=None, limit=0, dev_split=False, dev_split_size=512):
        if split is None:
            dataset_name = "RippleEdits"
            splits = ["popular", "random", "recent"]
        else:
            dataset_name = "RippleEdits_" + split
            splits = [split]
        
        # load dataset
        split_dev_limits = dict()
        for split in splits:
            split_dev_limits[split] = dev_split_size // len(splits)
        split_dev_limits[splits[0]] += dev_split_size % len(splits)
        split_limits = dict()
        for split in splits:
            split_limits[split] = limit // len(splits) if limit > 0 else 0
        split_limits[splits[0]] += limit % len(splits)

        example_list = []
        example_id = 0
        for split in splits:
            with open(f"{data_directory}/extended_{split}.json", 'r', encoding='utf-8') as f:
                examples = json.load(f)
            for i, example_data in tqdm(enumerate(examples), desc=f"Reading data from file: {data_directory}extended_{split}.json"):
                example_id += 1
                if dev_split:
                    if i >= split_dev_limits[split]:
                        continue
                else:
                    if i < split_dev_limits[split]:
                        continue
                    elif i >= split_dev_limits[split] + split_limits[split]:
                        continue
                # print(f"dev_split={dev_split}, split={split}, example_id={example_id}, i={i}")
                example = RippleEditsDataset.parse_example(example_id, split, example_data, force_query_type)
                if example is None:
                    print(f"Skipping example {i} during loading because of no valid test cases.")
                    continue
                
                # TODO: This check is taken from the RippleEdits benchmark; memit model editor fails when it is given such empty strings
                if any([fact.subject == '' or fact.target == '' for fact in example.facts]):
                    print(f"Skipping example {i} during loading because of missing labels.")
                    continue
                else:
                    example_list.append(example)
        return RippleEditsDataset(dataset_name, example_list)
    