import torch

from src.Probing.hidden_state_extractor import hidden_state_extractor


class POS_hidden_state_extractor(hidden_state_extractor):
    def __init__(
        self, LM, LM_tokenizer, LM_name, extracted_layers, device, num_shift_positions=0
    ):
        super().__init__(LM, LM_tokenizer, LM_name, extracted_layers, device)

        # num_shift_positions: int, absolute value denotes number of positions by which the POS label is shifted
        # Sign denotes direction of shifting (+ve means shift to right, -ve means shift to left)
        self.num_shift_positions = num_shift_positions

    def get_last_subword_hidden_state(
        self, batch, processed_hidden_states, tokenized_batch
    ):
        """
        Description:
            every word in the sentence is split into a subword, each having its own token and hidden state vector.
            For probing, there needs to be only 1 hidden state per output label, so we select the hidden state
            of the last subword of a word as the representative hidden state for the whole word

        Parameters:
            batch : list of str, a list of the input sentences
            processed_hidden_states: hidden states vector after removing [cls], [sep], and padding vectors
            tokenized_batch: batch input sentences after tokenization

        Returns:
            last_subword_hidden_states: hidden states for the last subword of each word in the sentences of the batch
        """

        token_counter = 0
        last_subword_indices = (
            []
        )  # list that stores index of the last subword of the words in every sentence
        for i, input_example in enumerate(batch):
            input_example = input_example.split(" ")
            # iterate over every word in the sentence and exclude the last u words corresponding to label_shift = u
            # we still have to iterate over every word in the sentence to properly increment first_subword_index
            #           and last_subword_index
            for j, word in enumerate(input_example):
                # in case the word isn't placed at the beginning of the sentence, it is prepended with a space. This
                # gives it a different token ID when tokenized with a GPT2 tokenizer than when tokenized without a space
                # (BERT tokenizers neglect this space)
                if j > 0:
                    word = " " + word

                if "bert" in self.LM_name.lower():
                    # get subword tokenization while removing [cls] and [sep] token
                    tokenized_word = self.LM_tokenizer(word).data["input_ids"][1:-1]
                else:
                    # get subword tokenization
                    tokenized_word = self.LM_tokenizer(word).data["input_ids"]

                first_subword_index = (
                    token_counter  # first subword index in processed_hidden_states
                )
                last_subword_index = (
                    token_counter + len(tokenized_word) - 1
                )  # last subword index in processed_hidden_states

                # only add the index if it falls within the shifted window
                if (
                    (
                        self.num_shift_positions < 0
                        and j < (len(input_example) + self.num_shift_positions)
                    )
                    or (self.num_shift_positions > 0 and j >= self.num_shift_positions)
                    or self.num_shift_positions == 0
                ):
                    last_subword_indices.append(
                        last_subword_index
                    )  # equivalent to index of hidden state of last subword
                token_counter = last_subword_index + 1

                # testing only
                # check that first and last indices computed actually correspond to first and last indices
                # in the hidden states vector by checking that the subword tokens are the same
                # only for verification, get a list of the subword tokens for all words in all sentences in the batch
                if "bert" in self.LM_name.lower():
                    # create a mask to remove [cls], [sep], and padding tokens
                    LM_mask = self.create_MLM_mask(
                        tokenized_batch
                    )  # shape: (batch_size*seq_len)
                else:
                    # create a mask to remove padding tokens
                    LM_mask = self.create_pad_mask(
                        tokenized_batch
                    )  # shape: (batch_size*seq_len)

                # this is stacked into a 1-D vector, and the [cls], [sep], and padding tokens are removed from it
                processed_subword_tokens = torch.mul(
                    tokenized_batch.data["input_ids"].view(-1), LM_mask
                )  # shape: (batch_size*seq_len)
                # remove zeros from tensor
                processed_subword_tokens = processed_subword_tokens[
                    processed_subword_tokens != 0
                ]

                tokenized_word = torch.tensor(tokenized_word).to(self.device)
                assert torch.all(
                    torch.eq(
                        processed_subword_tokens[
                            first_subword_index : last_subword_index + 1
                        ],
                        tokenized_word,
                    )
                ), "extracted indices don't match the indices in the adjusted subword tokens vector"
                ##################################################################################

        # extract the rows that have the last subword hidden states
        last_subword_hidden_states = torch.squeeze(
            processed_hidden_states[[last_subword_indices], ...]
        )

        return last_subword_hidden_states
