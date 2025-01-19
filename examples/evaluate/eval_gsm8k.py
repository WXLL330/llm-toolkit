import os
import tempfile
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from llmtoolkit import (
    infly_evaluate,
    safe_dict2file,
    print_rank_0,
    prune_magnitude,
)

import argparse 


def find_adapter_model_paths(root_dir):
    matching_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        if "adapter_model.safetensors" in filenames:
            matching_paths.append(dirpath)
    return matching_paths


def eval(
    task: str = "gsm8k",
    base_model_name_or_path: str = None,
    peft_model_name_or_path: str = None,
    load_in_4bit: bool = False,
    sparsity_ratio: float = None,
    structured_sparse: bool = False,
    rank: int = None,
):
    temp_dirs = []  # reserved for temp dirs used in current eval process
    if structured_sparse:
        raise NotImplementedError("structured_sparse is not implemented.")
    if sparsity_ratio:
        assert sparsity_ratio < 1 and sparsity_ratio > 0
        sparse_temp_dir = tempfile.mkdtemp(dir=".")
        temp_dirs.append(sparse_temp_dir)
        model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        prune_magnitude(model, sparsity_ratio)
        model.save_pretrained(sparse_temp_dir)
        tokenizer.save_pretrained(sparse_temp_dir)
        base_model_name_or_path = sparse_temp_dir
        print_rank_0(f"base_model_name_or_path has changed to {sparse_temp_dir}.")

    if not peft_model_name_or_path:
        acc = infly_evaluate(
            task=task,
            model_name_or_path=base_model_name_or_path,
            load_in_4bit=load_in_4bit,
        )
    else:
        base_tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        peft_tokenizer = AutoTokenizer.from_pretrained(peft_model_name_or_path)
        if len(base_tokenizer) != len(peft_tokenizer):
            print_rank_0(
                f"Since the embedding of base model mismatch peft adapter ({len(base_tokenizer)} - {len(peft_tokenizer)}), merging."
            )
            model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path)
            model.resize_token_embeddings(len(peft_tokenizer))
            model = PeftModel.from_pretrained(model, peft_model_name_or_path)
            model = model.merge_and_unload()
            merge_temp_dir = tempfile.mkdtemp(dir=".")
            temp_dirs.append(merge_temp_dir)
            model.save_pretrained(merge_temp_dir)
            peft_tokenizer.save_pretrained(merge_temp_dir)
            del model
            del base_tokenizer
            del peft_tokenizer
            torch.cuda.empty_cache()
            acc = infly_evaluate(
                task=task,
                model_name_or_path=merge_temp_dir,
                load_in_4bit=load_in_4bit,
            )
        else:
            del base_tokenizer
            del peft_tokenizer
            acc = infly_evaluate(
                task=task,
                model_name_or_path=base_model_name_or_path,
                peft_name_or_path=peft_model_name_or_path,
                load_in_4bit=load_in_4bit,
            )
    results = {}
    results["model"] = base_model_name_or_path
    results["peft"] = peft_model_name_or_path
    results["bits"] = 4 if load_in_4bit else 16
    results["rank"] = rank
    results["sparse"] = sparsity_ratio
    results["task"] = task
    results["accuracy"] = acc
    safe_dict2file(results, "eval_result.txt")
    for t in temp_dirs:
        shutil.rmtree(t)


# ckpts = [
#     # "/hpc2hdd/home/lzhang330/llm-toolkit/tmp/metamath/output_lora_rank16_scale1/checkpoint-2250",
# ]
# ranks = [1, 2, 4, 8]
# ranks = [1]
# sparses = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# sparses = [0.7, 0.8, 0.9]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='eval model')
    parser.add_argument('--base', type=str, default="meta-llama/Llama-2-7b-hf", help='peft_model_name_or_path')
    parser.add_argument('--ckpt', type=str, required=True, help='peft_model_name_or_path')
    parser.add_argument('--sparsity_ratio', type=float, default=1.0, help='sparsity_ratio')
    parser.add_argument('--rank', type=int, default=0, help='lora rank')

    parser.add_argument('--eval_base', action='store_true', help='if eval base model')
    parser.add_argument('--eval_lora', action='store_true', help='if eval lora model')
    parser.add_argument('--eval_sparse', action='store_true', help='if eval sparse model')

    args = parser.parse_args()

    base_model = args.base
    ckpt = args.ckpt
    sparsity_ratio = args.sparsity_ratio
    rank = args.rank

    # base model eval 16-bit and 4-bit
    if args.eval_base:
        print_rank_0(f"eval base model: {base_model} in 16bit... ...")
        eval(
            base_model_name_or_path=base_model,
            load_in_4bit=False,
        )
        print_rank_0(f"eval base model: {base_model} in 4bit... ...")
        eval(
            base_model_name_or_path=base_model,
            load_in_4bit=True,
        )
    
    # lora eval 16-bit and 4-bit
    if args.eval_lora:
        print_rank_0(f"eval ckpt.lora.rank{rank}: {ckpt} in 16bit... ...")
        eval(
            base_model_name_or_path=base_model,
            peft_model_name_or_path=ckpt,
            load_in_4bit=False,
            rank=rank,
        )
        print_rank_0(f"eval ckpt.lora.rank{rank}: {ckpt} in 4bit... ...")
        eval(
            base_model_name_or_path=base_model,
            peft_model_name_or_path=ckpt,
            load_in_4bit=True,
            rank=rank,
        )

    # sparse eval 16-bit and 4-bit
    # it is suggest to keep the sparsity_ratio the same as the checkpoint
    if args.eval_sparse:
        print_rank_0(f"eval ckpt.sparse{sparsity_ratio}: {ckpt} in 16bit... ...")
        eval(
            base_model_name_or_path=base_model,
            peft_model_name_or_path=ckpt,
            sparsity_ratio=sparsity_ratio,
            load_in_4bit=False,
            rank=rank,
        )
        # torch.distributed.destroy_process_group()
        # print_rank_0(f"eval ckpt.sparse{sparsity_ratio}: {ckpt} in 4bit... ...")
        # eval(
        #     base_model_name_or_path=base_model,
        #     peft_model_name_or_path=ckpt,
        #     sparsity_ratio=sparsity_ratio,
        #     load_in_4bit=True,
        #     rank=rank,
        # )
    
