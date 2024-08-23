import logging
import os

logger = logging.getLogger(__name__)

probed_LM = "meta-llama/Meta-Llama-3-8B"
extracted_layers = [5]
truncation_method = "PCA"
truncated_feature_size = 16
HF_TOKEN = "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"
# implement PCA

dataset_name = "ambition"
dataset_path = os.path.join(
    os.getcwd(),
    "data",
    "concept_probing",
    "generation",
    "human_labelled",
    "ambition_no_neg.csv",
)

num_output_labels = 2

num_train_subsets = 1  # number of subsets to which the training set is split
num_validation_subsets = 1  # number of subsets to which the validation set is split
num_test_subsets = 1  # number of subsets to which the test set is split

train_split = 0.8  # training split
validation_split = 0.1  # validation split
test_split = 0.1  # testing split

LM_batch_size = 128

# training hyperparameters
probe_batch_size = 64
learning_rate = 0.005
EPOCHS = 3000
early_stopping = True


