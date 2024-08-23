## Introduction
This repository is used for probing language models for abstract concepts. It includes scripts for the following tasks:
1. generating datasets composed of sentences that infer a certain higher-level abstract concept.
2. Extracting hidden states from a chosen language model that represent the example sentences
3. training linear probes using the dataset of hidden states with their corresponding labels

## Usage
To use this repository:
1. clone the repository, and cd into its directory
2. create and activate the conda environment:
```conda env create -f environment.yml```

### Dataset Generation
For generating a dataset for a certain concept:
1. create a directory for storing the prompts:

```mkdir src/DatasetCreation/prompts/dataset_generation/{investigator's first name}```
2. create a prompt for the desired concept and name it: {concept}_{prompt_version}.yaml

   (for generating negative examples, use not_{concept}_{prompt_version}.yaml)

3. run the file dataset_generation_main.py:
```
 python -m src.dataset_generation_main --help
usage: dataset_generation_main.py [-h] [--investigated_concept INVESTIGATED_CONCEPT] [--num_examples NUM_EXAMPLES]
                                        [--investigator_name INVESTIGATOR_NAME] [--prompt_version PROMPT_VERSION]
                                        [--whole_dataset WHOLE_DATASET]

Creating and preprocessing a dataset for training probes

optional arguments:
  -h, --help            show this help message and exit
  --investigated_concept INVESTIGATED_CONCEPT
                        concept inferred from dataset. For -ve examples, input "not_{concept}"
  --num_examples NUM_EXAMPLES
                        number of examples to be generated
  --investigator_name INVESTIGATOR_NAME
                        name of the person running the script for naming the file storing the generated examples
  --prompt_version PROMPT_VERSION
                        prompt version used for generation
  --whole_dataset WHOLE_DATASET
                        (boolean) use both prompts to generate +ve & -ve examples in one go, and shuffle them
```

Note: "whole_dataset" is a boolean variable, so not including it in the arguments makes it false, and including it
makes it true
