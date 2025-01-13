import os
import tempfile
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from llmtoolkit import (
    infly_evaluate,
    safe_dict2file,
)


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
):
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
            model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path)
            model.resize_token_embeddings(len(peft_tokenizer))
            model = PeftModel.from_pretrained(model, peft_model_name_or_path)
            model = model.merge_and_unload()
            temp_dir = tempfile.mkdtemp(dir=".")
            model.save_pretrained(temp_dir)
            peft_tokenizer.save_pretrained(temp_dir)
            del model
            del base_tokenizer
            del peft_tokenizer
            torch.cuda.empty_cache()
            acc = infly_evaluate(
                task=task,
                model_name_or_path=temp_dir,
                load_in_4bit=load_in_4bit,
            )
            shutil.rmtree(temp_dir)
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
    results["task"] = task
    results["accuracy"] = acc
    safe_dict2file(results, "eval_result.txt")


ckpts = [
    # "/hpc2hdd/home/lzhang330/llm-toolkit/tmp/metamath/output_lora_rank16_scale1/checkpoint-2250",
]

ranks = ["1", "2", "4", "8"]
sparses = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]

if __name__ == "__main__":
    for rank in ranks:
        for sparse in sparses:
            path = f"/workspace/llm-toolkit/examples/finetune/llama2_7b.metamath40k.rank{rank}.sparse{sparse}.lora.output/checkpoint-3375"
            ckpts.append(path)

    for ckpt in ckpts:
        if not os.path.exists(ckpt):
            print(f"{ckpt} not exist!")


    # for ckpt in ckpts:
    #     eval(
    #         model_name_or_path="meta-llama/Llama-2-7b-hf",
    #         peft_name_or_path=ckpt,
    #         load_in_4bit=False,
    #     )

    # for ckpt in ckpts:
    #     eval(
    #         model_name_or_path="meta-llama/Llama-2-7b-hf",
    #         peft_name_or_path=ckpt,
    #         load_in_4bit=True,
    #     )
