import os

# path for example sentences
sentence_path = os.path.join(
    os.getcwd(), "data", "POS_probing", "input_dataset", "input_sentences_sample.csv"
)

# path for corresponding POS tags for each example sentence
POS_path = os.path.join(
    os.getcwd(), "data", "POS_probing", "input_dataset", "output_POS_sample.csv"
)

# cap on sentence length
max_sent_len = 80

# extracted layers
extracted_layers = list(range(13))
# extracted_layers = [0, 2, 4]

# LM name
LM_name = "bert-base-uncased"


num_train_subsets = 10  # number of subsets to which the training set is split
num_validation_subsets = 1  # number of subsets to which the validation set is split
num_test_subsets = 1  # number of subsets to which the test set is split

train_split = 0.9  # training split
validation_split = 0.05  # validation split
test_split = 0.05  # testing split

LM_batch_size = 128

num_shift_positions = 0  # for POS tags only

# training hyperparameters
probe_batch_size = 512
learning_rate = 0.001
EPOCHS = 15
early_stopping = True
