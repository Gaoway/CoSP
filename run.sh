# python evaluation.py --run-baseline --fp16
# python evaluation_log.py --disable-tqdm --disable-tree-attn


#!/bin/sh

nvidia-smi

llama13b="/data0/lygao/model/llama/llama-13b"
alpaca68m="/data0/lygao/model/llama/llama-68m-alpaca-finetuned"
datapath_alpaca="/data0/lygao/dataset/alpaca/data/alpaca_data.json"
datapath_wmt="/data0/amax/git/MCSD/dataset/wmt_ende.json"
datapath_webglm="/data0/lygao/dataset/webglm-qa/data/test.jsonl"

python evaluation_log.py \
    --datapath "$datapath_webglm"  \
    --disable-tqdm \
    --sampling-type myway \
    --gpu-id 7 \

# nohup
