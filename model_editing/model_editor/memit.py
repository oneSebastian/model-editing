import sys
import os
import torch
import inspect
from typing import Dict, List, Literal, Optional, Tuple, Union
from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    PreTrainedModel,
)
from .util import EditModel
from ..queryexecutor import QueryExecutor


class MEMITModel(EditModel):
    def __init__(
        self,
        model: PreTrainedModel,
        model_name: str,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        batch_size: Optional[int]=16,
        use_chat_template: bool=False,
        verbose: bool=False,
        log_path: Optional[str]=None,
    ):
        QueryExecutor.__init__(
            self,
            model=model,
            model_name=model_name,
            tokenizer=tokenizer,
            batch_size=batch_size,
            use_chat_template=use_chat_template,
            verbose=verbose,
            log_path=log_path
        )
        self._changed_weights = None
    
    def _format_fact_for_rome(self, fact):
        subject = fact.subject
        target = fact.target
        prompt = fact.prompt.replace(subject, '{}').replace(" " + target, "")
        if self.use_chat_template:
            prompt, _ = self.apply_chat_to_prompt_and_model_answer(prompt, target)
        return {'prompt': prompt, 'subject': subject, 'target_new': {'str': target}}

    def edit_model(self, facts):
        from .rome_style.memit import MEMITHyperParams, apply_memit_to_model

        # TODO: adapt for sequential editing
        requests = [self._format_fact_for_rome(fact) for fact in facts]

        #print("#### DEBUG edit_model START")
        #assert len(facts) == len(requests)
        #for i in range(len(facts)):
        #    print(f"fact {i+1}:")
        #    print(f"    fact={facts[i].to_dict()}")
        #    print(f"    request={requests[i]}")
        #print("#### DEBUG edit_model END")

        if self._model_name == "gpt-j":
            hparams = MEMITHyperParams.from_json(f'model_editing/model_editor/rome_style/hparams/MEMIT/EleutherAI_gpt-j-6B.json')
        else:
            hparams = MEMITHyperParams.from_json(f'model_editing/model_editor/rome_style/hparams/MEMIT/{self._model_name}.json')
        _, self._changed_weights = apply_memit_to_model(self._model, self.tokenizer, requests, hparams, return_orig_weights=True)


    def restore_model(self):
        if self._changed_weights is None:
            return

        from .rome_style.util import nethook

        with torch.no_grad():
            for k, v in self._changed_weights.items():
                nethook.get_parameter(self._model, k)[...] = v.to(self._device)
    
    