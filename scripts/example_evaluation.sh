#!/bin/bash

python main.py evaluate\
    --model gpt2-xl \
    --editors memit context-retriever in-context \
    --edit_batch_size 16 \
    --sample_size 32 \
    --editing_tasks MQuAKE RippleEdits CounterFact zsRE\
    --control_tasks lambada hellaswag
