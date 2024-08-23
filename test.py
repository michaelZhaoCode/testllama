import os
import torch
import torch.distributed as dist
from transformers import BertTokenizerFast, AutoTokenizer
from customllama import custom_LlamaForCausalLM
import argparse

def setup(rank, world_size):
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"Process {rank}/{world_size} is initialized and using GPU {rank}")

def cleanup():
    dist.destroy_process_group()

def load_LM_model_parallel(LM_name, rank, world_size, HF_TOKEN=None):
    setup(rank, world_size)

    # Load the model and assign layers to different GPUs
    model_path = os.path.join(os.getcwd(), "LMs", LM_name)
    LM = custom_LlamaForCausalLM.from_pretrained(model_path, token=HF_TOKEN, output_hidden_states=True)

    # Split the model layers across GPUs for model parallelism
    num_layers = len(LM.model.layers)
    layers_per_rank = num_layers // world_size

    if rank == 0:
        # Assign first half of the layers to rank 0
        LM.model.layers = LM.model.layers[:layers_per_rank]
        LM.model.to(rank)
    elif rank == 1:
        # Assign second half of the layers to rank 1
        LM.model.layers = LM.model.layers[layers_per_rank:]
        LM.model.to(rank)

    LM_tokenizer = AutoTokenizer.from_pretrained(LM_name, token=HF_TOKEN)
    LM_tokenizer.pad_token = LM_tokenizer.eos_token

    return LM, LM_tokenizer


def run_inference_model_parallel(LM, LM_tokenizer, rank, world_size):
    # Sample text input for inference
    texts = ["This is a test.", "Model parallel inference with PyTorch."]

    # Tokenize input texts
    inputs = LM_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(rank)

    # Set model in evaluation mode
    LM.eval()

    with torch.no_grad():
        # Rank 0 processes the first part of the model
        if rank == 0:
            # Process through the layers assigned to rank 0
            hidden_states = LM.model(inputs.input_ids)

            # Send intermediate hidden states to rank 1
            dist.send(tensor=hidden_states, dst=1)

        # Rank 1 receives the intermediate activations and processes further
        elif rank == 1:
            # Receive intermediate hidden states from rank 0
            intermediate_hidden_states = torch.zeros_like(inputs.input_ids).to(rank)
            dist.recv(tensor=intermediate_hidden_states, src=0)

            # Process through the remaining layers on rank 1
            final_output = LM.model(intermediate_hidden_states)

            # Print final output on rank 1
            print(f"Inference results from rank {rank}:")
            print(final_output)


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
