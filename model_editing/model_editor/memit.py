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
    ):
        QueryExecutor.__init__(
            self,
            model=model,
            model_name=model_name,
            tokenizer=tokenizer,
            batch_size=batch_size,
            use_chat_template=use_chat_template,
        )
        self._changed_weights = None
    
    @staticmethod
    def _format_fact_for_rome(fact):
        subject = fact.subject
        target = fact.target
        prompt = fact.prompt.replace(subject, '{}').replace(" " + target, "")
        return {'prompt': prompt, 'subject': subject, 'target_new': {'str': target}}

    def edit_model(self, facts):
        from .rome_style.memit import MEMITHyperParams, apply_memit_to_model

        # TODO: adapt for sequential editing
        requests = [self._format_fact_for_rome(fact) for fact in facts]
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
    
    