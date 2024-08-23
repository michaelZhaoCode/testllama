import os
import torch
import torch.distributed as dist
from transformers import BertTokenizerFast, AutoTokenizer
import argparse

def setup(rank, world_size):
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"Process {rank}/{world_size} is initialized and using GPU {rank}")
    dist.barrier()
    print(f"Process {rank} has passed the barrier.")

def cleanup():
    dist.destroy_process_group()

def load_LM_model_parallel(LM_name, rank, world_size, HF_TOKEN=None):
    setup(rank, world_size)

    # Assume custom_LlamaForCausalLM is your large model
    model_path = os.path.join(os.getcwd(), "LMs", LM_name)
    LM = custom_LlamaForCausalLM.from_pretrained(model_path, token=HF_TOKEN, output_hidden_states=True)

    # Split the model layers across GPUs/nodes for model parallelism
    if rank == 0:
        LM.part1 = LM.part1.to(rank)
    elif rank == 1:
        LM.part2 = LM.part2.to(rank)

    LM_tokenizer = AutoTokenizer.from_pretrained(LM_name, token=HF_TOKEN)
    LM_tokenizer.pad_token = LM_tokenizer.eos_token

    return LM, LM_tokenizer

def run_inference_model_parallel(LM, LM_tokenizer, rank, world_size):
    texts = ["This is a test.", "Model parallel inference with PyTorch."]
    inputs = LM_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(rank)

    LM.eval()
    with torch.no_grad():
        if rank == 0:
            outputs_part1 = LM.part1(inputs)
            dist.send(tensor=outputs_part1, dst=1)
        elif rank == 1:
            inputs_part2 = torch.zeros_like(inputs).to(rank)
            dist.recv(tensor=inputs_part2, src=0)
            outputs_part2 = LM.part2(inputs_part2)
            print(f"Inference results (last hidden states from rank {rank}):")
            print(outputs_part2)

def main():
    parser = argparse.ArgumentParser(description="Model Parallel Hugging Face Model Inference")
    parser.add_argument('--model_name', type=str, required=True, help="Hugging Face model name or path")
    parser.add_argument('--hf_token', type=str, required=False, help="Hugging Face authentication token")
    parser.add_argument('--local-rank', type=int, help="Local rank passed from distributed launcher")

    args = parser.parse_args()

    world_size = 2  # total number of processes (2 nodes with 1 GPU each)
    rank = args.local_rank  # using local rank as the rank of the current process

    LM, LM_tokenizer = load_LM_model_parallel(args.model_name, rank, world_size, HF_TOKEN=args.hf_token)
    run_inference_model_parallel(LM, LM_tokenizer, rank, world_size)
    cleanup()

if __name__ == "__main__":
    main()
