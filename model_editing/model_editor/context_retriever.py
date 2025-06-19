from typing import Dict, List, Literal, Optional, Tuple, Union
from math import ceil
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    PreTrainedModel,
)
import torch
from lm_eval.api.instance import Instance
from lm_eval.models.huggingface import HFLM
from .util import EditModel
from ..queryexecutor import QueryExecutor


class ContextRetrieverModel(EditModel):
    def __init__(
        self,
        model: PreTrainedModel,
        model_name: str,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        batch_size: Optional[int]=16,
        use_chat_template: bool=False,
        edit_template_id: Union[bool, str]=1,
        verbose: bool=False,
        log_path: Optional[str]=None,
        retrieve_k: int=4,
        embedding_model: str="facebook/contriever-msmarco",
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
            log_path=log_path
        )
        self.embedding_model = AutoModel.from_pretrained(embedding_model).to(self._device)
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.edit_facts = []
        self.fact_embeddings = None
        self.edit_template_id = edit_template_id
        self.retrieve_k = retrieve_k

        self.chat_template_beginning = "<s>[INST]"
        if self.use_chat_template and self._model_name != "mistral_7B_instruct":
            # TODO: generalise implementation
            raise NotImplementedError("In context editing with chat template is currently only implemented for Mistral-7B-Instruct-v0.3. We have a hard coded assumption that the first user quers starts with \"<s>[INST]\"")

        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write("Context retriever parameters:\n")
                f.write(f"    retrieve_k: {self.retrieve_k}\n")
                f.write(f"    embedding_model: {embedding_model}\n")
                f.write(f"    edit_template_id: {self.edit_template_id}\n")


    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def get_sent_embeddings(self, sents, batch_size=32):    
        all_embs = []
        for i in range(0, len(sents), batch_size):
            sent_batch = sents[i: i + batch_size]
            inputs = self.embedding_tokenizer(sent_batch, padding=True, truncation=True, return_tensors='pt').to(self._device)
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                embeddings = self.mean_pooling(outputs[0], inputs['attention_mask'])
            all_embs.append(embeddings.cpu())
        all_embs = torch.vstack(all_embs)
        return all_embs

    def retrieve_facts(self, query):
        inputs = self.embedding_tokenizer([query], padding=True, truncation=True, return_tensors='pt').to(self._device)
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            query_emb = self.mean_pooling(outputs[0], inputs['attention_mask']).cpu()
        sim = (query_emb @ self.fact_embeddings.T)[0]
        knn = sim.topk(min(self.retrieve_k, sim.shape[0]), largest=True)
        return knn.indices

    def edit_model(self, facts):
        self.edit_facts += facts
        fact_statements = [f"{fact.prompt} {fact.target}." for fact in facts]
        embs = self.get_sent_embeddings(fact_statements)
        if self.fact_embeddings is not None:
            self.fact_embeddings = torch.cat((self.fact_embeddings, embs), dim=0)
        else:
            self.fact_embeddings = embs

    def get_edit_context(self, fact_sentences):
        if self.edit_template_id == 0:
            return "Imagine that " + " ".join(fact_sentences) + "\n"
        elif self.edit_template_id == 1:
            instruction = "### Instruction\n\nYour internal knowledge is out of date. The following facts are true. " \
            "Disregard any conflicting knowledge that you might have regarding the named entities. Do not print out any inaccuracies in the provided fact. " \
            "Only complete the request that is provided.\n\n### Updated Facts\n\n"
            facts = "\n".join(["- " + fact_sentence for fact_sentence in fact_sentences])
            pre_prompt = "\n### Request to answer\n\n"
            return instruction + facts + pre_prompt
        elif self.edit_template_id == "test_empty":
            return ""
        else:
            raise ValueError(f"{self.edit_template_id} is not a recognised edit context template id.")

    def get_prompt_context(self, prompt):
        if self.fact_embeddings is None:
            return ""
        fact_ids = self.retrieve_facts(prompt)
        context_facts = [self.edit_facts[i] for i in fact_ids]
        context_fact_sentences = [f"{fact.prompt} {fact.target}." for fact in context_facts]
        return self.get_edit_context(context_fact_sentences)
        #prompt_context = self.instruction + " ".join(context_fact_sentences) + "\n"
        #return prompt_context

    def restore_model(self):
        self.edit_facts = []
        self.fact_embeddings = None

    def _edit_generate_until_requests(self, requests):
        # We may still need to apply the chat template and cannot return immediately
        #if not self.edit_facts:
        #    return requests
        context_window = self.model.config.max_position_embeddings
        for request in requests:
            assert len(request.arguments) == 2 and isinstance(request.arguments[1], dict)
            argument_length = self.tokenizer.encode(request.arguments[0], return_tensors='pt').shape[-1]
            max_edit_context = context_window - argument_length - self.max_gen_toks
            edit_context = self.get_prompt_context(request.arguments[0])
            edit_tokens = self.tokenizer.encode(edit_context, return_tensors='pt')
            if max_edit_context <= 0:
                edit_context = ""
            elif edit_tokens.shape[-1] > max_edit_context:
                edit_context = self.tokenizer.decode(edit_tokens[0][:max_edit_context])
            if self.use_chat_template:
                assert request.arguments[0].startswith(self.chat_template_beginning), "For now we are assuming we're working with Mistral-7B-Instruvt-v0.3"
                prompt = self.chat_template_beginning + edit_context + request.arguments[0][len(self.chat_template_beginning):]
            else:
                prompt = edit_context + request.arguments[0]
            request.arguments = (prompt,) + request.arguments[1:]
            
    def _edit_loglikelihood_requests(self, requests):
        # We may still need to apply the chat template and cannot return immediately
        #if not self.edit_facts:
        #    return requests
        context_window = self.model.config.max_position_embeddings
        for request in requests:
            if isinstance(request.arguments, tuple) and len(request.arguments) > 1:
                if isinstance(request.arguments[1], str):
                    total_argument_length = self.tokenizer.encode(request.arguments[0] + request.arguments[1], return_tensors='pt').shape[-1]
                    max_edit_context = context_window - total_argument_length
                    edit_context = self.get_prompt_context(request.arguments[0])
                    edit_tokens = self.tokenizer.encode(edit_context, return_tensors='pt')
                    if max_edit_context <= 0:
                        edit_context = ""
                    elif edit_tokens.shape[-1] > max_edit_context:
                        edit_context = self.tokenizer.decode(edit_tokens[0][:max_edit_context])
                    if self.use_chat_template:
                        #prompt, response = self.apply_chat_to_prompt_and_model_answer(edit_context + request.arguments[0], request.arguments[1])
                        assert request.arguments[0].startswith(self.chat_template_beginning), f"For now we are assuming we're working with Mistral-7B-Instruvt-v0.3, request.arguments[0]={request.arguments[0]}"
                        prompt = self.chat_template_beginning + edit_context + request.arguments[0][len(self.chat_template_beginning):]
                        response = request.arguments[1]
                    else:
                        prompt = edit_context + request.arguments[0]
                        response = request.arguments[1]
                    request.arguments = (prompt, response) + request.arguments[2:]
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
                edit_context = self.get_prompt_context(request.arguments[0])
                edit_tokens = self.tokenizer.encode(edit_context, return_tensors='pt')
                if max_edit_context <= 0:
                    edit_context = ""
                elif edit_tokens.shape[-1] > max_edit_context:
                    edit_context = self.tokenizer.decode(edit_tokens[0][:max_edit_context])
                if self.use_chat_template:
                    prompt = self.apply_chat_to_single_prompt(edit_context + argument)
                else:
                    prompt = edit_context + argument
                request.arguments = prompt if isinstance(request.arguments, str) else (prompt,)

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
            edit_context = self.get_prompt_context(request.arguments[0])
            edit_tokens = self.tokenizer.encode(edit_context, return_tensors='pt')
            if max_edit_context <= 0 or not self.edit_facts:
                edit_context = ""
            elif edit_tokens.shape[-1] > max_edit_context:
                edit_context = self.tokenizer.decode(edit_tokens[0][:max_edit_context])
            if self.use_chat_template:
                #arguments = self.apply_chat_to_prompt_and_model_answer(edit_context, request.arguments[0])
                #response = request.arguments[1]
                assert request.arguments[0].startswith(self.chat_template_beginning), "For now we are assuming we're working with Mistral-7B-Instruvt-v0.3"
                arguments = (
                    self.chat_template_beginning + edit_context,
                    request.arguments[0][len(self.chat_template_beginning):]
                )
            else:
                arguments = (edit_context, request.arguments[0])
            contextualised_requests.append(Instance(
                request_type="loglikelihood",
                doc=request.doc,
                arguments=arguments,
                idx=request.idx,
            ))
        contextualised_results = HFLM.loglikelihood(self, contextualised_requests, disable_tqdm)
        if not contextualised_results:
            print("DEBGUG empty contextualised_results:")
            for request in contextualised_requests:
                print("DEBUG request arguments:", request.arguments)
        return [_[0] for _ in contextualised_results]
    
    def create_argmax_inputs(self, queries):
        argmax_inputs = []
        context_window = self.model.config.max_position_embeddings
        for query in queries:
            edit_context = self.get_prompt_context(query.prompt)
            if isinstance(query.answers[0], str):
                answer = query.answers[0]
            else:
                answer = next(item for sublist in query.answers for item in sublist)
            if self.use_chat_template:
                prompt, answer = self.apply_chat_to_prompt_and_model_answer((edit_context if self.edit_facts else "") + query.prompt, answer)
            else:
                prompt = (edit_context if self.edit_facts else "") + query.prompt
                answer = " " + answer
            
            prompt_ids = self.tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=False)
            answer_tokens = self.tokenizer.encode(answer, return_tensors='pt', add_special_tokens=False)
            answer_length = answer_tokens.shape[-1]

            prompt_space = max(0, context_window - answer_length)
            if self.edit_facts and prompt_space < prompt_ids.shape[-1]:
                prompt_ids = prompt_ids[:, :prompt_space]
            prompt_length = prompt_ids.shape[-1]
            inputs = torch.cat((prompt_ids, answer_tokens), dim=-1)
            assert inputs.shape[-1] <= context_window
            argmax_inputs.append((inputs, prompt_length, answer_length, answer_tokens))
            '''
            else:
                # To have effective edits: self.edit_context + query.prompt instead of just query.prompt
                edit_context = self.get_prompt_context(query.prompt)
                edit_ids = self.tokenizer.encode(edit_context[:-1], return_tensors='pt') if self.edit_facts else None
                prompt_ids = self.tokenizer.encode((edit_context[-1] if self.edit_facts else "") + query.prompt, return_tensors='pt')
                
                answer_tokens = self.tokenizer.encode(" " + answer, return_tensors='pt')
                answer_length = answer_tokens.shape[-1]

                fill_context = max(0, context_window - prompt_ids.shape[-1] - answer_length)
                if self.edit_facts and fill_context < edit_ids.shape[-1]:
                    edit_ids = edit_ids[:, :fill_context]
                prompt_length = edit_ids.shape[-1] + prompt_ids.shape[-1] if self.edit_facts else prompt_ids.shape[-1]

                inputs = torch.cat((edit_ids, prompt_ids, answer_tokens), dim=-1) if self.edit_facts else torch.cat((prompt_ids, answer_tokens), dim=-1)
                assert inputs.shape[-1] <= context_window
                argmax_inputs.append((inputs, prompt_length, answer_length, answer_tokens))
            '''
        return argmax_inputs
