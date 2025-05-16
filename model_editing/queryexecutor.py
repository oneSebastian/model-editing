import torch
from torch.nn.functional import log_softmax
import transformers

from typing import Dict, List, Literal, Optional, Tuple, Union
from lm_eval.models.huggingface import HFLM
from lm_eval.api.model import TemplateLM
from lm_eval.api.instance import Instance
from lm_eval.models.utils import (
    Collator,
    clear_torch_cache,
    configure_pad_token,
    get_dtype,
    handle_stop_sequences,
    pad_and_concat,
    stop_sequences_criteria,
)


class QueryExecutor(HFLM):
    def __init__(
    # custom init (based on HFLM), because we always pass a pretrained model
        self,
        model: transformers.PreTrainedModel,
        model_name: str,
        tokenizer: Union[
            transformers.PreTrainedTokenizer,
            transformers.PreTrainedTokenizerFast,
        ],
        backend: Literal["default", "causal", "seq2seq"] = "causal",
        revision: Optional[str] = "main",
        subfolder: Optional[str] = None,
        truncation: Optional[bool] = False,
        logits_cache: bool = True,
        max_length: Optional[int] = None,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        max_batch_size: Optional[int] = 64,
        trust_remote_code: Optional[bool] = False,
        use_fast_tokenizer: Optional[bool] = True,
        add_bos_token: Optional[bool] = False,
        prefix_token_id: Optional[int] = None,
        # PEFT, delta weights and quantization options
        peft: Optional[str] = None,
        delta: Optional[str] = None,
        autogptq: Optional[Union[bool, str]] = False,
        gptqmodel: Optional[bool] = False,
        gguf_file: Optional[str] = None,
        **kwargs,
    ) -> None:
        TemplateLM.__init__(self)
        self._model = model
        self._model_name = model_name
        self._device = self._model.device
        self._config = self._model.config

        self.argmax_batch_size = batch_size

        # determine which of 'causal' and 'seq2seq' backends to use for HF models
        self._get_backend(
            config=self.config, backend=backend, trust_remote_code=trust_remote_code
        )

        assert isinstance(
            tokenizer, transformers.PreTrainedTokenizer
        ) or isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
        self.tokenizer = tokenizer

        # access self._model through self.model property outside this method
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()
            self.model.tie_weights()

        self.truncation = truncation
        self.logits_cache = logits_cache
        self.vocab_size = self.tokenizer.vocab_size
        # select (or create) a pad token to use
        self.tokenizer = configure_pad_token(self.tokenizer, model_config=self.config)

        self.add_bos_token = add_bos_token
        self._max_length = max_length
        self.pretrained = model
        self.delta = delta
        self.peft = peft
        self.revision = revision
        self.batch_schedule = 1
        self.batch_sizes = {}
        self.max_batch_size = max_batch_size

        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self.batch_size_per_gpu = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self.batch_size_per_gpu = int(batch_size)

        # if a PreTrainedModel was passed into HFLM, we forgo distributed setup.
        self._rank = 0
        self._world_size = 1

        self.custom_prefix_token_id = prefix_token_id
        if prefix_token_id is not None:
            print(f"Loglikelihood prefix token id used in evaluation: {self.prefix_token_id}")
        
        self._max_gen_toks = 256
    
    '''
    # TODO: used to debug test_loglikelihood_rolling: Why are outputs not equal, but only similar when evaluated on different lm instances despice equal parameters?
    def loglikelihood(self, requests, disable_tqdm=None) -> List[Tuple[float, bool]]:
        print("DEBUG loglikelihood requests:")
        for i, r in enumerate(requests):
            print(i, r)
            print(self.tokenizer.encode(r.arguments[0] + r.arguments[1]))
            print(" -" * 50)
        return super().loglikelihood(requests)
    '''
    
    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks
    

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        # build stopping criteria
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )
        #print("DEBUG self._max_gen_toks, _model_generate:", self._max_gen_toks)
        #print("DEBUG max_length:", max_length)
        #print("DEBUG generation_kwargs", generation_kwargs)
        if max_length:
            generation_kwargs["max_new_tokens"] = None
        return self.model.generate(
            input_ids=context,
            max_length=max_length,
            # eos_token_id=None, # set eos_token_id to None for forcing target number of generated tokens
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            **generation_kwargs,
        )
    

    def _create_generate_until_requests(self, queries):
        return [
            Instance(
                request_type="generate_until",
                doc=None,
                arguments=(queries[i].prompt, {"until": [self.tokenizer.eos_token]}),
                idx=i,
            ) for i in range(len(queries))
        ]

    @staticmethod
    def _verify_generate_queries(queries, generated):
        results = []
        for i, query in enumerate(queries):
            correct_answers = query.answers
            model_answer = generated[i]
            if all(isinstance(answer, str) for answer in correct_answers):
                result = any([answer in model_answer for answer in correct_answers])
            elif all(isinstance(answer, list) for answer in correct_answers):
                result = any([any([answer in model_answer for answer in answer_group]) for answer_group in correct_answers])
            else:
                raise ValueError("Unsupported type in correct query answers.")
            results.append(result)
        return results
    
    def _verify_generate_queries_all_lengths(self, queries, generated, max_length):
        results = [{} for _ in queries]
        query_replies = [{} for _ in queries]
        for i, query in enumerate(queries):
            query_replies[i] = {"query": query.to_dict(), "model_answer": generated[i]}
            correct_answers = query.answers
            #print(f"DEBUG: correct_answers={correct_answers}")
            tokenised_answer = self.tokenizer.encode(generated[i])
            for length in range(1, max_length + 1):
                model_answer = self.tokenizer.decode(tokenised_answer[:length])
                #print(f"DEBUG: length={length}, model_answer={model_answer}")
                if all(isinstance(answer, str) for answer in correct_answers):
                    result = any([answer in model_answer for answer in correct_answers])
                elif all(isinstance(answer, list) for answer in correct_answers):
                    result = any([any([answer in model_answer for answer in answer_group]) for answer_group in correct_answers])
                else:
                    raise ValueError("Unsupported type in correct query answers.")
                #print(f"DEBUG: result={result}")
                results[i][length] = result
        return results, query_replies
    
    def execute_generate_queries(self, queries, answer_length=64, evaluate_generate_lengths=False):
        if evaluate_generate_lengths:
            # no answer length given; generate with length 50, but verify for all shorter lengths as well
            answer_length = 64
            tmp_max_gen_toks = self.max_gen_toks
            self._max_gen_toks = answer_length
            requests = self._create_generate_until_requests(queries)
            generated = self.generate_until(requests)
            self._max_gen_toks = tmp_max_gen_toks
            results, query_replies = self._verify_generate_queries_all_lengths(queries, generated, answer_length)
            return results, query_replies
        else:
            assert isinstance(answer_length, int)
            tmp_max_gen_toks = self._max_gen_toks
            self._max_gen_toks = answer_length
            requests = self._create_generate_until_requests(queries)
            # print("DEBUG self._max_gen_toks, execute_generate_queries:", self._max_gen_toks)
            generated = self.generate_until(requests)
            # print("DEBUG generated:", generated)
            self._max_gen_toks = tmp_max_gen_toks
            results = self._verify_generate_queries(queries, generated)
            return results

    def _create_options_requests(self, queries):
        requests = []
        i = 0
        for query in queries:
            assert (query.answer_options is not None) and (len(query.answer_options) > 1), "options queries require multiple answer options"
            for option in query.answer_options:
                assert isinstance(option, str)
                requests.append(
                    Instance(
                    request_type="loglikelihood",
                    doc=None,
                    arguments=(query.prompt, f" {option}"),
                    idx=i,
                ))
                i += 1
        return requests

    def execute_options_queries(self, queries):
        requests = self._create_options_requests(queries)
        responses = self.loglikelihood(requests, disable_tqdm=False)
        assert len(requests) == len(responses)

        # check query responses
        results = []
        i = 0
        for query in queries:
            correct_option = query.answer_options.index(query.answers[0])
            n = len(query.answer_options)
            query_responses = [_[0] for _ in responses[i: i + n]]
            i += n
            max_response = query_responses.index(max(query_responses))
            results.append(correct_option == max_response)
        assert i == len(responses)
        return results
    
    def create_argmax_inputs(self, queries):
        argmax_inputs = []
        for query in queries:
            prompt_ids = self.tokenizer.encode(query.prompt, return_tensors='pt')
            answer_start = prompt_ids.shape[-1]
            if isinstance(query.answers[0], str):
                answer = query.answers[0]
            else:
                answer = next(item for sublist in query.answers for item in sublist)
            answer_tokens = self.tokenizer.encode(" " + answer, return_tensors='pt')
            answer_length = answer_tokens.shape[-1]
            inputs = torch.cat((prompt_ids, answer_tokens), dim=-1)
            argmax_inputs.append((inputs, answer_start, answer_length, answer_tokens))
        return argmax_inputs

    def execute_argmax_queries(self, queries):
        argmax_inputs = self.create_argmax_inputs(queries)
        results = []
        #print("DEBUG self.argmax_batch_size:", self.argmax_batch_size)
        for i in range(0, len(argmax_inputs), self.argmax_batch_size):
            batch_inputs = argmax_inputs[i:min(i + self.argmax_batch_size, len(argmax_inputs))]
            # batch and pad input ids
            m = max(_[0].shape[1] for _ in batch_inputs)
            pad_token = self.tokenizer.encode(self.tokenizer.pad_token)[0]
            padded_inputs = [
                torch.cat([inputs[0], torch.full((1, m - inputs[0].shape[1]), pad_token)], dim=1)
                for inputs in batch_inputs
            ]
            # load tensors to input device in case model is spread
            input_device = next(self._model.parameters()).device
            inputs = torch.cat(padded_inputs, dim=0).to(input_device) 
            attention_mask = torch.where(inputs != pad_token, 1, 0).to(input_device)
            #print("DEBUG inputs shape:", inputs.shape)
            with torch.no_grad():
                outputs = self._model(inputs, attention_mask=attention_mask)
                logits = outputs.logits
                max_tokens = torch.argmax(logits, dim=-1)
                for j, inputs in enumerate(batch_inputs):
                    _, answer_start, answer_length, answer_tokens = inputs
                    query_results = []
                    for k in range(answer_tokens.shape[1]):
                        query_results.append((max_tokens[j][answer_start - 1 + k] == answer_tokens[0][k]).item())
                    results.append(query_results)
            del outputs, logits, max_tokens  # free memory
            torch.cuda.empty_cache()
            '''
            with torch.no_grad():
                outputs = self._model(inputs, attention_mask=attention_mask)
            logits = outputs.logits
            max_tokens = torch.argmax(logits, dim=-1)
            for j, inputs in enumerate(batch_inputs):
                _, answer_start, answer_length, answer_tokens = inputs
                query_results = []
                for k in range(answer_tokens.shape[1]):
                    query_results.append((max_tokens[j][answer_start - 1 + k] == answer_tokens[0][k]).item())
                results.append(query_results)
            '''
        return results
    
    # The base QueryExecutor overwrites loglikelihood_rolling, because in context editors use loglikelihood method instead to compute loglikelihood of request sequence conditional on edit sequence
    # however these functions handle sequences longer than context window differently, hence we use loglikelihood for all editors
    '''
        https://github.com/EleutherAI/lm-evaluation-harness/blob/main/examples/lm-eval-overview.ipynb
        https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/model.py
        loglikelihood_rolling computes loglikelihood for given a sequence conditional on empty context. We can use loglikelihood instead to condition on edit context.
    '''
    def loglikelihood_rolling(
        self, requests, disable_tqdm: bool = False
    ) -> List[float]:
        context_window = self.model.config.max_position_embeddings
        contextualised_requests = []
        for request in requests:
            argument_tokens = self.tokenizer.encode(request.arguments[0], return_tensors='pt')
            if argument_tokens.shape[-1] > context_window:
                # we need to cut rolling loglikelihood continuation to context window otherwise hFLM loglikelihood throws an error
                request.arguments = (self.tokenizer.decode(argument_tokens[0][-context_window:]),)
            contextualised_requests.append(Instance(
                request_type="loglikelihood",
                doc=request.doc,
                # compute loglikelihood conditional on empty sequence for generic editor
                arguments=("", request.arguments[0]),
                idx=request.idx,
            ))
        contextualised_results = HFLM.loglikelihood(self, contextualised_requests, disable_tqdm)
        return [_[0] for _ in contextualised_results]
       
