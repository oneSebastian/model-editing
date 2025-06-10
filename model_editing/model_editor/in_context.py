from typing import Dict, List, Literal, Optional, Tuple, Union
from math import ceil
from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    PreTrainedModel,
)
from torch import cat
from lm_eval.api.instance import Instance
from lm_eval.models.huggingface import HFLM
from .util import EditModel
from ..queryexecutor import QueryExecutor


class InContextModel(EditModel):
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
            editor_applies_chat_template=use_chat_template,
            verbose=verbose,
            log_path=log_path,
        )
        self.edit_context = ""
        self.edit_facts = []
        self.edit_tokens = None

    def edit_model(self, facts):
        self.edit_facts += facts
        contextualised_facts = [f"{fact.prompt} {fact.target}." for fact in self.edit_facts]
        context = "Imagine that " + " ".join(contextualised_facts) + "\n"
        self.edit_context = context
        self.edit_tokens = self.tokenizer.encode(context, return_tensors='pt')

    def restore_model(self):
        self.edit_facts = []
        self.edit_context = ""
        self.edit_tokens = None

    def _edit_generate_until_requests(self, requests):
        if not self.edit_facts:
            return requests
        context_window = self.model.config.max_position_embeddings
        for request in requests:
            assert len(request.arguments) == 2 and isinstance(request.arguments[1], dict)
            argument_length = self.tokenizer.encode(request.arguments[0], return_tensors='pt').shape[-1]
            # reserve self.max_gen_toks space in context_window
            max_edit_context = context_window - argument_length - self.max_gen_toks
            if max_edit_context <= 0:
                edit_context = ""
            elif self.edit_tokens.shape[-1] > max_edit_context:
                edit_context = self.tokenizer.decode(self.edit_tokens[0][:max_edit_context])
            else:
                edit_context = self.edit_context
            request.arguments = (edit_context + request.arguments[0],) + request.arguments[1:]
            
    def _edit_loglikelihood_requests(self, requests):
        if not self.edit_facts:
            return requests
        context_window = self.model.config.max_position_embeddings
        for request in requests:
            if isinstance(request.arguments, tuple) and len(request.arguments) > 1:
                if isinstance(request.arguments[1], str):
                    total_argument_length = self.tokenizer.encode(request.arguments[0] + request.arguments[1], return_tensors='pt').shape[-1]
                    max_edit_context = context_window - total_argument_length
                    if max_edit_context <= 0:
                        edit_context = ""
                    elif self.edit_tokens.shape[-1] > max_edit_context:
                        edit_context = self.tokenizer.decode(self.edit_tokens[0][:max_edit_context])
                    else:
                        edit_context = self.edit_context
                    request.arguments = (edit_context + request.arguments[0],) + request.arguments[1:]
                else:
                    for i, arg in enumerate(request.arguments):
                        print(i, arg)
                    raise NotImplementedError("Second request argument is not a string")
            else:
                assert isinstance(request.arguments, str) or len(request.arguments) == 1
                argument = request.arguments[0] if isinstance(request.arguments, tuple) else request.arguments
                assert isinstance(argument, str)
                argument_length = self.tokenizer.encode(argument, return_tensors='pt').shape[-1]
                max_edit_context = context_window - argument_length
                if max_edit_context <= 0:
                    edit_context = ""
                elif self.edit_tokens.shape[-1] > max_edit_context:
                    edit_context = self.tokenizer.decode(self.edit_tokens[0][:max_edit_context])
                else:
                    edit_context = self.edit_context
                request.arguments = edit_context + argument if isinstance(request.arguments, str) else (edit_context + argument,)

    def generate_until(
        self, requests, disable_tqdm: bool = False
    ) -> List[str]:
        self._edit_generate_until_requests(requests)
        return HFLM.generate_until(self, requests, disable_tqdm)

    def loglikelihood(
        self, requests, disable_tqdm: bool = False
    ) -> List[Tuple[float, bool]]:
        self._edit_loglikelihood_requests(requests)
        return HFLM.loglikelihood(self, requests, disable_tqdm=disable_tqdm)
        
    def loglikelihood_rolling(
        self, requests, disable_tqdm: bool = False
    ) -> List[float]:
        '''
        https://github.com/EleutherAI/lm-evaluation-harness/blob/main/examples/lm-eval-overview.ipynb
        https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/model.py
        loglikelihood_rolling computes loglikelihood for given a sequence conditional on empty context. We can use loglikelihood instead to condition on edit context.
        '''
        context_window = self.model.config.max_position_embeddings
        contextualised_requests = []
        for request in requests:
            assert isinstance(request.arguments, tuple) and len(request.arguments) == 1 and isinstance(request.arguments[0], str)
            argument_tokens = self.tokenizer.encode(request.arguments[0], return_tensors='pt')
            argument_length = argument_tokens.shape[-1]
            if argument_length > context_window:
                # we need to cut rolling loglikelihood continuation to context window otherwise hFLM loglikelihood throws an error
                request.arguments = (self.tokenizer.decode(argument_tokens[0][-context_window:]),)
                argument_length = context_window
            max_edit_context = context_window - argument_length
            if max_edit_context <= 0 or not self.edit_facts:
                edit_context = ""
            elif self.edit_tokens.shape[-1] > max_edit_context:
                edit_context = self.tokenizer.decode(self.edit_tokens[0][:max_edit_context])
            else:
                edit_context = self.edit_context
            contextualised_requests.append(Instance(
                request_type="loglikelihood",
                doc=request.doc,
                arguments=(edit_context, request.arguments[0]),
                idx=request.idx,
            ))
        contextualised_results = HFLM.loglikelihood(self, contextualised_requests, disable_tqdm)
        return [_[0] for _ in contextualised_results]
    
    def create_argmax_inputs(self, queries):
        argmax_inputs = []
        context_window = self.model.config.max_position_embeddings
        for query in queries:
            # To have effective edits: self.edit_context + query.prompt instead of just query.prompt
            edit_ids = self.tokenizer.encode(self.edit_context[:-1], return_tensors='pt') if self.edit_facts else None
            prompt_ids = self.tokenizer.encode((self.edit_context[-1] if self.edit_facts else "") + query.prompt, return_tensors='pt')
            if isinstance(query.answers[0], str):
                answer = query.answers[0]
            else:
                answer = next(item for sublist in query.answers for item in sublist)
            answer_tokens = self.tokenizer.encode(" " + answer, return_tensors='pt')
            answer_length = answer_tokens.shape[-1]

            fill_context = max(0, context_window - prompt_ids.shape[-1] - answer_length)
            if self.edit_facts and fill_context < edit_ids.shape[-1]:
                edit_ids = edit_ids[:, :fill_context]
            prompt_length = edit_ids.shape[-1] + prompt_ids.shape[-1] if self.edit_facts else prompt_ids.shape[-1]

            inputs = cat((edit_ids, prompt_ids, answer_tokens), dim=-1) if self.edit_facts else cat((prompt_ids, answer_tokens), dim=-1)
            assert inputs.shape[-1] <= context_window
            argmax_inputs.append((inputs, prompt_length, answer_length, answer_tokens))
        return argmax_inputs
            
    
        
    