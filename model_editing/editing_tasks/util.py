import re
import random
from typing import Optional
from enum import Enum


class Dataset():
    def __init__(self, dataset_name: str, examples: list):
        self.dataset_name = dataset_name
        self.examples = examples
    
    def sample(self, k, seed):
        random.seed(seed)
        self.examples = random.sample(self.examples, min(k, len(self.examples)))


def is_wikidata_id(s: str) -> bool:
    return bool(re.fullmatch(r'Q\d+', s))


class QueryType(Enum):
    MC = "multiple_choice"
    GEN = "generation"
    ARG = "argmax"


class Query:
    def __init__(self, prompt: str, answers: list[str], query_type: QueryType, answer_options: Optional[list] = None, phrase: Optional[str] = None):
        # TODO: then I can remove some redundant information?
        assert prompt == phrase or phrase is None, f"DEBUG phrase: prompt={prompt}, answers={answers}, phrase={phrase}"
        self.prompt = prompt
        self.answers = answers
        self.answer_options = answer_options
        self.phrase = phrase
        self.type = query_type
        
    
    def to_dict(self):
        return {
            "prompt": self.prompt,
            "answers": self.answers,
            "answer_options": self.answer_options,
            "phrase": self.phrase,
        }
    
    @staticmethod
    def from_dict(query_dict: dict, query_type: QueryType):
        answer_candidates = [[answer["value"]] + answer["aliases"] for answer in query_dict["answers"]]
        return Query(
            prompt=query_dict["prompt"],
            answers = [[alias for alias in answer if len(alias) > 1 or alias.isdigit()] for answer in answer_candidates],
            query_type=query_type,
            phrase=query_dict["phrase"]
        )

    def __str__(self):
        return f"Query: prompt={self.prompt}, answers={[self.answers]}"


class Fact:
    def __init__(self, prompt: str, subject: str, target: str, original_target: str, fact_query: Query,  question: Optional[str] = None):
        self.prompt = prompt
        self.subject = subject
        self.target = target
        self.original_target = original_target
        self.query = fact_query
        self.question = question
    
    def to_dict(self):
        return {
            "prompt": self.prompt,
            "subject": self.subject,
            "target": self.target,
            "original_target": self.original_target,
            "query": self.query.to_dict(),
            "question": self.question,
        }


class TestCondition(Enum):
    OR = "OR"
    AND = "AND"
    ACC = "ACC"


class TestCase:
    def __init__(self, test_dimension: str, test_condition: TestCondition, test_queries: list[Query], condition_queries: list[Query]):
        self.test_dimension = test_dimension
        self.test_condition = test_condition
        self.test_queries = test_queries
        self.condition_queries = condition_queries
        if QueryType.ARG in [query.type for query in self.test_queries]:
            self.test_condition = TestCondition.ACC
    
    def to_dict(self):
        return {
            "test_dimension": self.test_dimension,
            "test_condition": self.test_condition,
            "test_queries": [query.to_dict() for query in self.test_queries],
            "condition_queries": [query.to_dict() for query in self.condition_queries],
        }
    

class Example:
    def __init__(self, example_id: int, example_type: str, facts: list[Fact], test_cases: list, dataset_split:str = "default"):
        self.example_id = example_id
        self.example_type = example_type
        self.facts = facts
        self.test_cases = test_cases
        self.dataset_split = dataset_split
    
    def to_dict(self):
        return {
            "example_id": self.example_id,
            "example_type": self.example_type,
            "facts": [fact.to_dict() for fact in self.facts],
            "test_cases": [test_case.to_dict() for test_case in self.test_cases],
        }

