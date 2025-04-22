import json
from dataclasses import dataclass


@dataclass
class HyperParams:
    """
    Simple wrapper to store hyperparameters for Python-based rewriting methods.
    """

    @classmethod
    def from_json(cls, fpath):
        import os
        print("DEBUG current working dir:", os.getcwd())
        with open(fpath, "r") as f:
            data = json.load(f)

        return cls(**data)
