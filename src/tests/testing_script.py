import gc

import psutil
import torch
from transformers import BertModel
from transformers import BertTokenizerFast


def flush():
    gc.collect()
    torch.cuda.empty_cache()
    # torch.cuda.reset_peak_memory_stats()


def bytes_to_kilo_bytes(bytes):
    return bytes / 1024


# Function to measure memory usage
def memory_usage(message: str = ""):
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"{message}Memory used: {mem_info.rss / (1024 ):.2f} KB")

    return mem_info.rss / (1024)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# batch : list of str, a list of the input sentences
batch = ["hello world"]

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
tokenized_batch = tokenizer(batch, padding=False, return_tensors="pt").to(device)


LM = BertModel.from_pretrained(
    "bert-base-uncased", output_attentions=False, output_hidden_states=False
).to(device)

with torch.no_grad():
    LM.eval()
    output1 = LM(**tokenized_batch)

del LM
flush()

LM = BertModel.from_pretrained(
    "bert-base-uncased", output_attentions=False, output_hidden_states=True
).to(device)
# Measure memory before inference
before_1 = memory_usage("Before Inference: ")
# run LM on tokenized batch
with torch.no_grad():
    LM.eval()
    output1 = LM(**tokenized_batch)

# Measure memory after inference
after_1 = memory_usage("After Inference: ")

del LM
flush()
bytes_to_kilo_bytes(torch.cuda.max_memory_allocated())

LM = BertModel.from_pretrained(
    "bert-base-uncased", output_attentions=False, output_hidden_states=False
).to(device)
# Measure memory before inference
before_2 = memory_usage("Before Inference: ")
# run LM on tokenized batch
with torch.no_grad():
    LM.eval()
    output2 = LM(**tokenized_batch)

# Measure memory after inference
after_2 = memory_usage("After Inference: ")

print(f"first LM {after_1 - before_1}")
print(f"second LM {after_2 - before_2}")

del LM
flush()
bytes_to_kilo_bytes(torch.cuda.max_memory_allocated())


x = 0
