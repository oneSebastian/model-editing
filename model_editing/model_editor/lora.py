import sys
import os
import torch
import gc
import inspect
from typing import Dict, List, Literal, Optional, Tuple, Union
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from .util import EditModel
from ..queryexecutor import QueryExecutor
from .lora_src.lora_main import apply_lora_to_model
from .lora_src.lora_hparams import LoRAHyperParams
from ..models import load_model


class LORAModel(EditModel):
    def __init__(
        self,
        model: AutoModelForCausalLM,
        model_name: str,
        tokenizer: AutoTokenizer,
        batch_size: Optional[int]=16,
        use_chat_template: bool=False,
        verbose: bool=False,
    ):
        QueryExecutor.__init__(
            self,
            model=model,
            model_name=model_name,
            tokenizer=tokenizer,
            batch_size=batch_size,
            use_chat_template=use_chat_template,
            verbose=verbose,
        )
        self._original_model = None
    

    def _format_fact(self, fact):
        subject = fact.subject
        target = fact.target
        prompt = fact.prompt.replace(" " + target, "")
        if self.use_chat_template:
            prompt, _ = self.apply_chat_to_prompt_and_model_answer(prompt, target)
        return {'prompt': prompt, 'subject': subject, 'target_new': target}


    @staticmethod
    def hparam_map(model_name):
        if model_name == "gpt-j":
            return "model_editing/model_editor/lora_src/hparams/gpt-j-6B"
        elif model_name == "mistral_7B":
            return "model_editing/model_editor/lora_src/hparams/mistral_7B"
        elif model_name == "mistral_7B_instruct":
            return "model_editing/model_editor/lora_src/hparams/mistral_7B_instruct"
        else:
            raise ValueError(f"There are no LORA hyperparameters for model {model_name} yet.")


    def edit_model(self, facts):
        requests = [self._format_fact(fact) for fact in facts]
        # each request needs to be dict with key "prompt" and "target_new"
        hparams = LoRAHyperParams.from_hparams(LORAModel.hparam_map(self._model_name))
        self._model, _ = apply_lora_to_model(
            self._model,
            self.tokenizer,
            requests,
            hparams,
            lora_device=self._device,
        )

        #model: AutoModelForCausalLM,
        #tok: AutoTokenizer,
        #requests: List[Dict],
        #hparams: LoRAHyperParams,
        #copy=False,
        #return_orig_weights=False,
        #keep_original_weight=False,
        #**kwargs: Any,


    def restore_model(self):
        # Visual debugging
        import objgraph
        objgraph.show_backrefs([self._model], filename='model_refs.png')

        # Break possible internal references
        if hasattr(self._model, 'base_model'):
            self._model.base_model = None

        # Move model to CPU before deleting
        try:
            self._model.to('cpu')
        except Exception:
            pass  # not all models have .to()

        # Final memory cleanup
        del self._model
        if '_model' in self.__dict__:
            del self.__dict__['_model']
        gc.collect()
        torch.cuda.empty_cache()

        # Load clean model
        self._model, _ = load_model(self._model_name, self._device)
    
    