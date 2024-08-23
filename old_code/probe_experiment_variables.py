# variables that are frequently edited

label_shift = 0  # number of positions by which the POS tags are shifted
shift_direction = "left"  # direction to which the POS tags are shifted
LM_probed_layers = []  # layers of the language model that are desired to be probed
probe_batch_size = 512  # batch size used for training the probe
LM_batch_size = 128  # batch size used for inferring the language model


# unknown parameters
test_only = False
randomize_labels = False


# variables that are not generally edited

# LM used: "gpt2", "bert-base-uncased"
LM_name = "bert-base-uncased"  # name of language model that is desired to be probed
learning_rate = 0.001  # learning rate used for training the probe
epochs = 15  # number of epochs used for training the probe
early_stopping = True  # determine whether or not to use early stopping
use_control_dataset = False  # determine whether or not to include the control dataset. The control dataset is a
# non-shifted dataset that includes the same number of examples as that of the shifted dataset
# LM_embedding_size = 768     # embedding size (and similarly, hidden layer size) of the probed language model
# LM_num_layers = 13          # number of layers of the probed language model (includes the embedding layer)
num_train_subsets = 10  # number of subsets to which the training set is split
num_validation_subsets = 1  # number of subsets to which the validation set is split
num_test_subsets = 1  # number of subsets to which the test set is split
train_split = 0.9  # training split
validation_split = 0.05  # validation split
test_split = 0.05  # testing split
sentence_len_cap = 80  # max sentence length that a sentence can have in order to be kept in the dataset


if LM_name == "bert-base-uncased" or LM_name == "gpt2":
    LM_embedding_size = 768  # embedding size (and similarly, hidden layer size) of the probed language model
    LM_num_layers = 13  # number of layers of the probed language model (includes the embedding layer)
elif LM_name == "gpt2-medium":
    LM_embedding_size = 1024
    LM_num_layers = 25
elif LM_name == "gpt2-large":
    LM_embedding_size = 1280
    LM_num_layers = 37
else:
    raise Exception("chosen model is not within available options")
