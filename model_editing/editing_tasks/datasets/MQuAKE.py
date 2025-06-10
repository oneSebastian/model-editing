import json
from typing import Optional
from tqdm import tqdm
from ..util import Dataset, Example, Fact, Query, TestCase, TestCondition, QueryType


class MQuAKEDataset(Dataset):
    def __init__(self, dataset_name: str, examples: list):
        super().__init__(dataset_name, examples)

    
    @staticmethod
    def parse_example(id: int, split: str, data_dict: dict, force_query_type):
        facts = []
        for fact_data in data_dict["requested_rewrite"]:
            fact_query_answers = [fact_data["target_new"]["str"]]
            for single_hop in data_dict["new_single_hops"]:
                if single_hop["answer"] == fact_data["target_new"]["str"]:
                    fact_query_answers = [single_hop["answer"]] + single_hop["answer_alias"]
                    break
            
            facts.append(Fact(
                prompt = fact_data["prompt"].replace("{}", fact_data["subject"]),
                subject = fact_data["subject"],
                target = fact_data["target_new"]["str"],
                original_target = fact_data["target_true"]["str"],
                fact_query = Query(
                    #prompt = fact_data["prompt"].replace("{}", fact_data["subject"]),
                    # seb, since test cases use question answer format I will do the same for fact queries
                    prompt=fact_data["question"],
                    answers = fact_query_answers,
                    query_type=QueryType.GEN if force_query_type is None else force_query_type,
                ),
            ))

        test_queries = []
        for question in data_dict["questions"]:
            test_queries.append(Query(
                prompt = question,
                answers = [data_dict["new_answer"]] + data_dict["new_answer_alias"],
                query_type=QueryType.GEN if force_query_type is None else force_query_type,
            ))

        test_case = TestCase(
                test_case_id=0,
                test_dimension = f"{len(data_dict["new_single_hops"])}-hop",
                test_condition = TestCondition.OR,
                test_queries = test_queries,
                condition_queries = [],
        )
        return Example(
            example_id=data_dict["case_id"],
            example_type="default",
            facts=facts,
            test_cases=[test_case],
            dataset_split=split,
        )

    @staticmethod
    def from_file(data_directory, force_query_type: Optional[QueryType]=None, split=None, limit=0, dev_split=False, dev_split_size=512):
        if split is None:
            dataset_name = "MQuAKE"
            split = "CF-3k-v2"
        else:
            dataset_name = "MQuAKE_" + split
        with open(f"{data_directory}/MQuAKE-{split}.json", 'r') as f:
            examples = json.load(f)
        example_list = []
        if dev_split:
            limit = dev_split_size
        else:
            if limit > 0:
                limit += dev_split_size
        for i, example_data in tqdm(enumerate(examples), desc=f"Reading data from file: {data_directory}MQuAKE-{split}.json"):
            if not dev_split and i < dev_split_size:
                continue
            if limit > 0 and i >= limit:
                break
            example = MQuAKEDataset.parse_example(i, split, example_data, force_query_type)
            example_list.append(example)
        return MQuAKEDataset(dataset_name, example_list)