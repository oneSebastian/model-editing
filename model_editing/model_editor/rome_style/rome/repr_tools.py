"""
Contains utilities for extracting token representations and indices
from string templates. Used in computing the left and right vectors for ROME.
"""

from copy import deepcopy
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, LlamaTokenizerFast

from ..util import nethook


def get_reprs_at_word_tokens(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    context_templates: List[str],
    words: List[str],
    layer: int,
    module_template: str,
    subtoken: str,
    track: str = "in",
) -> torch.Tensor:
    """
    Retrieves the last token representation of `word` in `context_template`
    when `word` is substituted into `context_template`. See `get_last_word_idx_in_template`
    for more details.
    """

    idxs = get_words_idxs_in_templates(tok, context_templates, words, subtoken)
    return get_reprs_at_idxs(
        model,
        tok,
        [context_templates[i].format(words[i]) for i in range(len(words))],
        idxs,
        layer,
        module_template,
        track,
    )


def get_words_idxs_in_templates(
    tok: AutoTokenizer, context_templates: str, words: str, subtoken: str
) -> int:
    """
    Given list of template strings, each with *one* format specifier
    (e.g. "{} plays basketball"), and words to be substituted into the
    template, computes the post-tokenization index of their last tokens.
    """

    assert all(
        tmp.count("{}") == 1 for tmp in context_templates
    ), "We currently do not support multiple fill-ins for context"

    # Compute prefixes and suffixes of the tokenized context
    fill_idxs = [tmp.index("{}") for tmp in context_templates]
    prefixes, suffixes = [
        tmp[: fill_idxs[i]] for i, tmp in enumerate(context_templates)
    ], [tmp[fill_idxs[i] + 2 :] for i, tmp in enumerate(context_templates)]
    words = deepcopy(words)

    # Pre-process tokens
    for i, prefix in enumerate(prefixes):
        if len(prefix) > 0:
            # TODO: When we apply a chat template the prefix doesnt always end in " ". But if we remove the trailing space, that shouldnt be a problem?
            #assert prefix[-1] == " "
            if prefix[-1] == " ":
                prefix = prefix[:-1]

            prefixes[i] = prefix
            words[i] = f" {words[i].strip()}"

    # Tokenize to determine lengths
    assert len(prefixes) == len(words) == len(suffixes)
    n = len(prefixes)
    batch_tok = tok([*prefixes, *words, *suffixes], add_special_tokens=False)

    batch_tok = batch_tok['input_ids']

    prefixes_tok, words_tok, suffixes_tok = [
        batch_tok[i : i + n] for i in range(0, n * 3, n)
    ]

    #print("#### DEBUG get_words_idxs_in_templates START")
    #for i, _toks in enumerate([prefixes_tok, words_tok, suffixes_tok]):
    #    print(f"i={i}")
    #    for j, tokens in enumerate(_toks):
    #        if tokens:
    #            print(f"    j={j}, tokens[0]={tokens[0]}, tok.pad_token_id={tok.pad_token_id}")
    #            print(f"    tokens={tokens}, decoded={tok.decode(tokens)}")
    #print("#### DEBUG get_words_idxs_in_templates END")

    if isinstance(tok, LlamaTokenizer) or isinstance(tok, LlamaTokenizerFast):
        words_tok = [tokens[1:] if tokens[0] == 29871 else tokens for tokens in words_tok]
        suffixes_tok = [tokens[1:] if tokens[0] == 29871 else tokens for tokens in suffixes_tok]

    prefixes_len, words_len, suffixes_len = [
        [len(el) for el in tok_list]
        for tok_list in [prefixes_tok, words_tok, suffixes_tok]
    ]

    # Compute indices of last tokens
    if subtoken == "last" or subtoken == "first_after_last":
        return [
            [
                prefixes_len[i]
                + words_len[i]
                - (1 if subtoken == "last" or suffixes_len[i] == 0 else 0)
            ]
            # If suffix is empty, there is no "first token after the last".
            # So, just return the last token of the word.
            for i in range(n)
        ]
    elif subtoken == "first":
        return [[prefixes_len[i]] for i in range(n)]
    else:
        raise ValueError(f"Unknown subtoken type: {subtoken}")


def get_reprs_at_idxs(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    contexts: List[str],
    idxs: List[List[int]],
    layer: int,
    module_template: str,
    track: str = "in",
) -> torch.Tensor:
    """
    Runs input through model and returns averaged representations of the tokens
    at each index in `idxs`.
    """

    def _batch(n):
        #print(f"in _batch: n={n}, len(contexts)={len(contexts)}")
        for i in range(0, len(contexts), n):
            yield contexts[i : i + n], idxs[i : i + n]

    assert track in {"in", "out", "both"}
    both = track == "both"
    tin, tout = (
        (track == "in" or both),
        (track == "out" or both),
    )
    module_name = module_template.format(layer)
    to_return = {"in": [], "out": []}

    def _process(cur_repr, batch_idxs, key):
        nonlocal to_return
        # 2 lines taken from memit "Key error bug fixing" pull request
        if len(cur_repr)==0: 
            return
        cur_repr = cur_repr[0] if type(cur_repr) is tuple else cur_repr
        #print("cur_repr:", type(cur_repr), cur_repr.shape)
        #print("batch_idxs:", batch_idxs)
        # exit()
        for i, idx_list in enumerate(batch_idxs):
            to_return[key].append(cur_repr[i][idx_list].mean(0))

    for batch_contexts, batch_idxs in _batch(n=128):
        contexts_tok = tok(batch_contexts, padding=True, add_special_tokens=False, return_tensors="pt").to(
            next(model.parameters()).device
        )

        if isinstance(model, LlamaForCausalLM):
            if 'token_type_ids' in contexts_tok:
                del contexts_tok['token_type_ids']

        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=module_name,
                retain_input=tin,
                retain_output=tout,
            ) as tr:
                model(**contexts_tok)
        
        if tin:
            _process(tr.input, batch_idxs, "in")
        if tout:
            _process(tr.output, batch_idxs, "out")

    to_return = {k: torch.stack(v, 0) for k, v in to_return.items() if len(v) > 0}

    if len(to_return) == 1:
    #    return to_return["in"] if tin else to_return["out"]
        # 3 lines taken from memit "Key error bug fixing" pull request   
        single_key = list(to_return.keys())[0]
        dummy_tensor = torch.zeros_like(to_return[single_key], device="cuda")
        return (to_return["in"], dummy_tensor) if (single_key=="in" and tin) else (dummy_tensor, to_return["out"])
    else:
        return to_return["in"], to_return["out"]

