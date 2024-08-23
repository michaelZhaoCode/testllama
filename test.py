import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BertTokenizerFast, AutoTokenizer
import argparse


def setup(rank, world_size):
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def load_LM_distributed(LM_name, rank, world_size, HF_TOKEN=None):
    setup(rank, world_size)

    if "bert" in LM_name:
        LM = custom_BertModel.from_pretrained(LM_name, output_hidden_states=True, use_auth_token=HF_TOKEN)
        LM_tokenizer = BertTokenizerFast.from_pretrained(LM_name, use_auth_token=HF_TOKEN)
    elif "llama" in LM_name.lower():
        model_path = os.path.join(os.getcwd(), "LMs", LM_name)
        LM = custom_LlamaForCausalLM.from_pretrained(
            model_path, token=HF_TOKEN, output_hidden_states=True, use_auth_token=HF_TOKEN
        )
        LM_tokenizer = AutoTokenizer.from_pretrained(LM_name, token=HF_TOKEN)
        LM_tokenizer.pad_token = LM_tokenizer.eos_token

    # Wrap the model in DistributedDataParallel
    LM = LM.to(rank)
    LM = DDP(LM, device_ids=[rank])

    return LM, LM_tokenizer


def run_inference(LM, LM_tokenizer, rank):
    # Sample text input for inference
    texts = ["This is a test.", "Distributed inference with PyTorch."]

    # Tokenize input texts
    inputs = LM_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(rank)

    # Set model in evaluation mode
    LM.eval()

    with torch.no_grad():  # Disable gradient computation for inference
        # Perform inference
        outputs = LM(**inputs)

        # Access the model outputs
        hidden_states = outputs.hidden_states[-1]  # The last hidden state

    # Print results (only by the rank 0 process to avoid duplicate printing)
    if rank == 0:
        print("Inference results (last hidden states):")
        print(hidden_states)


def main():
    parser = argparse.ArgumentParser(description="Distributed Hugging Face Model Inference")

    parser.add_argument('--model_name', type=str, required=True, help="Hugging Face model name or path")
    parser.add_argument('--hf_token', type=str, required=False, help="Hugging Face authentication token")

    args = parser.parse_args()

    world_size = 2  # total number of processes (2 nodes with 1 GPU each)
    rank = int(os.environ['RANK'])  # rank of the current process (0 for master, 1 for worker)

    # Load the model and tokenizer with the specified Hugging Face folder and token
    LM, LM_tokenizer = load_LM_distributed(args.model_name, rank, world_size, HF_TOKEN=args.hf_token)

    # Run inference
    run_inference(LM, LM_tokenizer, rank)

    # Cleanup the distributed environment
    cleanup()


if __name__ == "__main__":
    main()
