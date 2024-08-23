from transformers import AutoConfig
from transformers import LlamaForCausalLM
from transformers import LlamaTokenizer

# Path to the directory containing the LLaMA model files
model_directory = (
    "/Users/moabdelwahab/PycharmProjects/PhD_research/llama3/Meta-Llama-3-8B"
)

# Load the tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_directory)

# Load the configuration with output_hidden_states set to True
config = AutoConfig.from_pretrained(model_directory, output_hidden_states=True)

# Load the tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_directory)

# Load the model with the updated configuration
model = LlamaForCausalLM.from_pretrained(model_directory, config=config)

# Example text to encode
text = "Once upon a time"

# Encode the text
inputs = tokenizer(text, return_tensors="pt")

# Generate predictions
outputs = model(**inputs)

# Access hidden states
hidden_states = outputs.hidden_states

print(hidden_states)
