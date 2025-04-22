#!/bin/bash

python main.py evaluate\
    --model gpt2-xl \
    --editors in-context \
    --edit_batch_size 16 \
    --sample_size 64 \
    --editing_tasks CounterFact zsRE\
    --control_tasks lambada
