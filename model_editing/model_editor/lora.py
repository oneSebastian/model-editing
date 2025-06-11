import sys
import os
import torch
import gc
import inspect
from peft import set_peft_model_state_dict
from typing import Dict, List, Literal, Optional, Tuple, Union
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from copy import deepcopy
from peft import get_peft_model, AdaLoraConfig, TaskType, get_peft_model_state_dict, set_peft_model_state_dict, LoraConfig
from .lora_src.lora_hparams import LoRAHyperParams
from .lora_src.lora_multimodal_hparams import LoRAMultimodalHyperParams
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
        external_hparams: Optional[dict]=None,
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
            log_path=log_path,
        )
        self._original_model = None

        # prepare peft model for editing
        self.hparams = LoRAHyperParams.from_hparams(LORAModel.hparam_map(self._model_name))
        for param, value in external_hparams.items():
            setattr(self.hparams, param, value) 
        
        self._model.config.use_cache = False
        self._model.supports_gradient_checkpointing = True  #
        self._model.gradient_checkpointing_enable()
        self._model.enable_input_require_grads()
        if self.hparams.lora_type == "lora":
            Config = LoraConfig
        elif self.hparams.lora_type == "adalora":
            Config = AdaLoraConfig
        else:
            raise NotImplementedError

        #log lora hyperparamaters
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(f"LoRA hyper parameters:\n")
            for param_name, param_value in vars(self.hparams).items():
                with open(self.log_path, "a") as f:
                    f.write(f"    {param_name}: {param_value}\n")

        peft_config = Config(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.hparams.rank,
            lora_alpha=self.hparams.lora_alpha, lora_dropout=self.hparams.lora_dropout,
            layers_to_transform=self.hparams.layers if len(self.hparams.layers) > 0 else None,
            target_modules=self.hparams.target_modules
        )
        self._model = get_peft_model(self._model, peft_config)
        self.original_lora_weights = deepcopy(get_peft_model_state_dict(self._model))
    

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
        apply_lora_to_model(
            self._model,
            self.tokenizer,
            requests,
            self.hparams,
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
        set_peft_model_state_dict(self._model, self.original_lora_weights)
        return
        self._model = self._model.base_model.model
        gc.collect()
        torch.cuda.empty_cache()
        print("Post restoration model type:", type(self._model))
        return

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
    
    