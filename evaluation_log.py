import argparse
import json
import logging
import time
from typing import Literal, Tuple

import torch
from inference.generate_log import Generator, BaseGenerator, SpeculativeGenerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from model.llama_tree_attn import LlamaForCausalLM, LlamaTokenizer

from tqdm import tqdm

from mylog import init_logger, logger

# # Setup logging
# logging.basicConfig(
#     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S",
#     level=logging.INFO,
# )

# logger = logging.getLogger(__name__)
import subprocess

def get_gpu_usage():
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                            stdout=subprocess.PIPE, text=True)
    gpu_usages = result.stdout.strip().split('\n')
    
    # 打印每个 GPU 的使用情况
    logger.info(f"GPU usage: {gpu_usages}")

# device = "cuda:7"
# device_num = 7
# get_gpu_usage()

class JsonData:
    def __init__(self, path, datapath) -> None:
        self.datapath = datapath
        with open(path) as fin:
            if self.datapath == datapath_webglm:
                self.data = [json.loads(line) for line in fin]
            else:
                self.data = json.load(fin)

    def __getitem__(self, index) -> Tuple[str, str]:
        if self.datapath == datapath_wmt:
            return self.data[index]
        elif self.datapath == datapath_alpaca:
            return self.data[index].get("instruction")
        elif self.datapath == datapath_webglm:
            return self.data[index].get("question")

    def __len__(self):
        return len(self.data)


def run_eval(
    draft_model,
    target_model,
    tokenizer,
    k_config: Tuple[int],
    datapath: str,
    max_new_tokens: int = 128,
    replacement=False,
    speculative_sampling=True,
    tree_attn=True,
    # sampling_type: Literal["argmax", "sampling"] = "sampling",
    sampling_type: str= "sampling",
    disable_tqdm: bool = False,
):

    if sampling_type == "argmax":
        target_model_temp = 0
        draft_model_temp = 0
        iter = 16*2
        total_epoch = 2
    elif sampling_type == "sampling":
        target_model_temp = 1
        draft_model_temp = 1
        iter = 16
        total_epoch = 4
    else:
        target_model_temp = 1
        draft_model_temp = 0
        iter = 16
        total_epoch = 4

    
    dataloader = JsonData(datapath, args.datapath)
    generator = SpeculativeGenerator(
        draft_model,
        target_model,
        eos_token_id=tokenizer.eos_token_id,
        k_config=k_config,
        max_new_tokens=max_new_tokens,
        draft_model_temp=draft_model_temp,
        target_model_temp=target_model_temp,
        replacement=replacement,
        speculative_sampling=speculative_sampling,
        tree_attn=tree_attn,
    )

    draft_model.eval()
    target_model.eval()

    start_time = time.time()

    acceptance_count = 0
    draft_token_count = 0
    invocation_count = 0
    acclerate_list = []

    iterator = range(min(iter,len(dataloader)))
    with torch.no_grad():
        for sample_idx in iterator if disable_tqdm else tqdm(iterator):
            
            prompt_text = dataloader[sample_idx]
            # logger.info(f"{sample_idx}, prompt_text: {prompt_text}")

            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
            input_ids = inputs.input_ids
            init_input_len = input_ids.size(-1)
            
            count = 0
            total_generate = 0
            total_invocation = 0
            one_sample_start = 0


            for epoch in range(total_epoch):
                one_epoch_start = time.time()
                output = generator.generate(input_ids)

                acceptance_count += output.acceptance_count
                draft_token_count += output.draft_token_count
                invocation_count += output.invocation_count

                acclerate_list.append((output.sequences.size(-1) - init_input_len)/output.invocation_count)

                if ((output.sequences.size(-1) - init_input_len)/output.invocation_count > 1):
                    one_sample_start += time.time() - one_epoch_start
                    total_generate += (output.sequences.size(-1) - init_input_len)
                    total_invocation += output.invocation_count
                    count += 1 

            if count > 0 and sampling_type != "argmax":
                logger.info(f"sample_idx: {sample_idx}, k_config: {k_config}, good_speed > 1: {total_generate / one_sample_start :.3f}, count: {count/total_epoch}, invocation: {total_generate} / {total_invocation}")
            


    end_time = time.time()

    run_time = end_time - start_time

    latency = run_time / (acceptance_count + invocation_count)
    speed = (acceptance_count + invocation_count) / run_time
    acceptance_rate = acceptance_count / draft_token_count
    block_efficiency = 1 + acceptance_count / invocation_count

    # logger.info("Running time: {:.2f} s".format(run_time))
    logger.info(f"k_config: {k_config}, Speed: {speed:.2f} tokens/s, Acceptance rate: {acceptance_rate:.2f}, Block efficiency: {block_efficiency:.2f}")

    # block_list = []
    # for i in range(1,9):
    #     sum = 0
    #     for j in range(0, len(acclerate_list)-(len(acclerate_list)%i), i):
    #         sum += max(acclerate_list[j:j+i])
    #     sum /= len(acclerate_list) / i
    #     block_list.append(sum)
    # logger.info(f"block_list: {block_list}")


def run_baseline_eval(
    target_model,
    tokenizer,
    datapath: str,
    max_new_tokens: int = 128,
    # sampling_type: Literal["argmax", "sampling"] = "sampling",
    sampling_type: str= "sampling",
    disable_tqdm: bool = False,
):

    if sampling_type == "argmax":
        target_model_temp = 0
    else:
        target_model_temp = 1

    dataloader = JsonData(datapath, args.datapath)
    generator = BaseGenerator(
        target_model,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        temp=target_model_temp,
    )

    target_model.eval()

    start_time = time.time()
    invocation_count = 0

    iterator = range(min(16, len(dataloader)))
    with torch.no_grad():
        for sample_idx in iterator if disable_tqdm else tqdm(iterator):

            prompt_text = dataloader[sample_idx]
            logger.info(f"{sample_idx}, prompt_text: {prompt_text}")

            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
            input_ids = inputs.input_ids
            output = generator.generate(input_ids)

            invocation_count += output.invocation_count
            logger.info(f"samples: {sample_idx}, invocation_count: {output.invocation_count}")
    end_time = time.time()
    run_time = end_time - start_time

    latency = run_time / invocation_count
    speed = invocation_count / run_time

    logger.info("Running time: {:.2f} s".format(run_time))
    logger.info("Token latency: {:.2f} ms".format(latency * 1000))
    logger.info("Speed: {:.2f} tokens/s".format(speed))


def main(args):
    torch_dtype = torch.float16 #if args.fp16 else torch.float32
    # print(torch_dtype)

    logger.info("The full evaluation configuration:\n" + repr(args))

    if args.auto_model and not args.disable_tree_attn:
        logger.warning(
            "Tree Attn is currently not supported for models other than LLaMA. Therefore, "
            "when using '--auto-model', Tree Attn will be disabled."
        )
        args.disable_tree_attn = True

    ModelLoader = AutoModelForCausalLM if args.auto_model else LlamaForCausalLM
    TokenizerLoader = AutoTokenizer if args.auto_model else LlamaTokenizer

    # logger.info("Loading draft model: {}".format(args.draft_model))
    draft_model = ModelLoader.from_pretrained(
        vicuna68m,
        torch_dtype=torch_dtype,
        device_map=device_num,
        use_flash_attention_2=True if args.flash_attn else False,
    )

    draft_model1 = ModelLoader.from_pretrained(
        llama68m,
        torch_dtype=torch_dtype,
        device_map=device_num,
        use_flash_attention_2=True if args.flash_attn else False,
    )

    draft_model2 = ModelLoader.from_pretrained(
        alpaca68m,
        torch_dtype=torch_dtype,
        device_map=device_num,
        use_flash_attention_2=True if args.flash_attn else False,
    )

    draft_model3 = ModelLoader.from_pretrained(
        chat68m,
        torch_dtype=torch_dtype,
        device_map=device_num,
        use_flash_attention_2=True if args.flash_attn else False,
    )

    draft_model_list = [draft_model, draft_model1, draft_model2, draft_model3]

    # logger.info("Loading target model: {}".format(args.target_model))
    target_model = ModelLoader.from_pretrained(
        args.target_model,
        torch_dtype=torch_dtype,
        device_map=device_num,
        use_flash_attention_2=True if args.flash_attn else False,
    )

    tokenizer = TokenizerLoader.from_pretrained(args.tokenizer)

    run_baseline_eval(
                target_model,
                tokenizer=tokenizer,
                datapath=args.datapath,
                max_new_tokens=args.max_new_tokens,
                sampling_type=args.sampling_type,
                disable_tqdm=args.disable_tqdm,
            )
    run_baseline_eval(
                draft_model_list[0],
                tokenizer=tokenizer,
                datapath=args.datapath,
                max_new_tokens=args.max_new_tokens,
                sampling_type=args.sampling_type,
                disable_tqdm=args.disable_tqdm,
    )
    run_baseline_eval(
                draft_model_list[2],
                tokenizer=tokenizer,
                datapath=args.datapath,
                max_new_tokens=args.max_new_tokens,
                sampling_type=args.sampling_type,
                disable_tqdm=args.disable_tqdm,
    )



    k_config_list = [
        # "1,1",
        "1,1,1",
        # "2,1",         
        "2,1,1",
        "4,1,1",
        "3,1,1",
        "2,2,1",
        "1,1,1,1"
        "2,1,1,1",
        "3,1,1,1",
        "2,2,1,1",
        # "3,1",
        # "3,1,1",
        # "2,2",
        # "2,2,1",
        # "2,2,2",
        # "4,1",
        # "5,1",
        # "3,2",
        # "7,1",
        # "4,2",
        # "4,2,2",
        # "3,3",
    ]
    for k_config in k_config_list:
        for i,draft_model in enumerate(draft_model_list):
            logger.info(f"Draft model: {i}")
            run_eval(
                draft_model,
                target_model,
                tokenizer=tokenizer,
                k_config=tuple(map(int, k_config.split(","))),
                datapath=args.datapath,
                max_new_tokens=args.max_new_tokens,
                replacement=args.replacement,
                speculative_sampling=not args.naive_sampling,
                tree_attn=not args.disable_tree_attn,
                sampling_type=args.sampling_type,
                disable_tqdm=args.disable_tqdm,
            )

    while False:
        k_config = input("Please input the k_config: ")
        logger.info(f"k_config: {k_config}")

        if k_config == "-1":
            break
        if k_config == "0":
            run_baseline_eval(
                target_model,
                tokenizer=tokenizer,
                datapath=args.datapath,
                max_new_tokens=args.max_new_tokens,
                sampling_type=args.sampling_type,
                disable_tqdm=args.disable_tqdm,
            )

            run_baseline_eval(
                draft_model,
                tokenizer=tokenizer,
                datapath=args.datapath,
                max_new_tokens=args.max_new_tokens,
                sampling_type=args.sampling_type,
                disable_tqdm=args.disable_tqdm,
            )
            continue

        try: 
            k_config = tuple(map(int, k_config.split(",")))
        except:
            logger.error("Invalid k_config. Please try again.")
            continue

        run_eval(
            draft_model,
            target_model,
            tokenizer=tokenizer,
            k_config=k_config,
            datapath=args.datapath,
            max_new_tokens=args.max_new_tokens,
            replacement=args.replacement,
            speculative_sampling=not args.naive_sampling,
            tree_attn=not args.disable_tree_attn,
            sampling_type=args.sampling_type,
            disable_tqdm=args.disable_tqdm,
        )


if __name__ == "__main__":
    llama13b = "/data0/lygao/model/llama/llama-13b"
    llama8b = "/data0/lygao/model/llama/llama-8b"

    vicuna160m = "/data0/lygao/model/llama/vicuna-160m"
    llama68m = "/data0/lygao/model/llama/llama-68m"
    vicuna68m = "/data0/lygao/model/llama/vicuna-68m"
    alpaca68m = "/data0/lygao/model/llama/llama-68m-alpaca-finetuned"
    chat68m = "/data0/lygao/model/llama/Llama-68M-Chat-v1"

    datapath_wmt = '/data0/amax/git/MCSD/dataset/wmt_ende.json'
    datapath_alpaca = '/data0/lygao/dataset/alpaca/data/alpaca_data.json'
    datapath_webglm = '/data0/lygao/dataset/webglm-qa/data/test.jsonl'


    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--draft-model", type=str,  help="Draft model path.", default= vicuna68m
    )
    parser.add_argument(
        "--target-model", type=str, help="Target model path.", default=llama13b
    )
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer path.")
    parser.add_argument("--fp16", action="store_true", help="use float16 dtype.")

    parser.add_argument(
        "--k-config",
        type=lambda x: tuple(map(int, x.split(","))),
        # required=True,
        help="Use comma separations, e.g. `--k-config 4,2,2`.",
        default= "3,2"
    )

    parser.add_argument(
        "--datapath", type=str,  help="The json data file." ,default= datapath_wmt
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--replacement",
        action="store_true",
        help="Sampling with replacement.",
    )
    parser.add_argument(
        "--naive-sampling",
        action="store_true",
        help="Use multi-candidate naive sampling.",
    )

    parser.add_argument("--disable-tree-attn", action="store_true")

    parser.add_argument(
        "--sampling-type", type=str, default="sampling", 
    )

    parser.add_argument("--disable-tqdm", action="store_true")

    parser.add_argument("--auto-model", action="store_true")
    parser.add_argument("--run-baseline", action="store_true")

    parser.add_argument("--flash-attn", action="store_true")

    parser.add_argument("--gpu-id", type=int, default=0)


    global args
    args = parser.parse_args()

    if args.tokenizer is None:
        args.tokenizer = args.target_model
    gpu_id = args.gpu_id

    global device, device_num
    device = f"cuda:{gpu_id}"
    device_num = gpu_id

    main(args)
