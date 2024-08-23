import torch


class hidden_state_extractor:
    def __init__(self, LM, LM_tokenizer, LM_name, extracted_layers):
        self.LM = LM  # language model from which hidden states should be extracted
        self.LM_tokenizer = LM_tokenizer  # LM's tokenizer
        self.LM_name = LM_name  # str, name of the LM
        self.extracted_layers = self.check_extracted_layers(extracted_layers)
        self.testing = True

    def check_extracted_layers(self, extracted_layers):
        assert not (
            extracted_layers is None or len(extracted_layers) == 0
        ), "extracted_layers must be a non-empty list/tensor"
        if not (
            isinstance(extracted_layers, list)
            or isinstance(extracted_layers, torch.Tensor)
        ):
            raise Exception("extracted_layers must either be a list or a torch tensor")
        elif isinstance(extracted_layers, list):
            return torch.tensor(
                extracted_layers
            )  # list of layers for which hidden states are extracted
        else:
            return extracted_layers

    def change_extracted_layers(self, new_extracted_layers):
        self.extracted_layers = self.check_extracted_layers(new_extracted_layers)
        return

    def infer_LM(self, input_batch):
        """
        Run LM on input batch
        """
        # Convert input examples into tokens
        tokenized_batch = self.LM_tokenizer(
            input_batch, padding=True, return_tensors="pt"
        )

        # Run the model in evaluation mode
        self.LM.eval()

        # Move the tokenized batch to the appropriate device(s) based on model parallelism
        with torch.no_grad():
            output = self.LM(
                input_ids=tokenized_batch['input_ids'],
                attention_mask=tokenized_batch['attention_mask'],
                extracted_layers=self.extracted_layers
            )

        return output, tokenized_batch, tokenized_batch['attention_mask']

    def create_pad_mask(self, tokenized_batch):
        """
        Description:
            create a mask to zero out padding tokens

        Parameters:
            tokenized_batch : batch of sentences after tokenization

        Returns:
            padding_mask: torch tensor, mask that zeros out all padding tokens
        """

        padding_mask = tokenized_batch.data["attention_mask"].view(-1)

        return padding_mask

    def create_MLM_mask(self, tokenized_batch):
        """
        Description:
            create a mask to zero out any [cls], [sep], and padding tokens (used for masked-language models (MLMs))

        Parameters:
            tokenized_batch : batch of sentences after tokenization

        Returns:
            MLM_mask: torch tensor, mask that zeros out all unnecessary tokens
        """

        # use the padding mask to zero out any padded hidden representations
        padding_mask = self.create_pad_mask(tokenized_batch)
        # for MLM models, zero out the [cls] and [sep] by adding their positions to the padding mask
        # masks have to have float32 datatype to avoid allocating extra memory when being multiplied by any tensor
        cls_mask = (tokenized_batch.data["input_ids"].view(-1) != 101).float()
        sep_mask = (tokenized_batch.data["input_ids"].view(-1) != 102).float()
        # combine the 3 masks
        MLM_mask = torch.mul(torch.mul(cls_mask, sep_mask), padding_mask)

        return MLM_mask

    def process_LM_hidden_states(self, LM_hidden_states, tokenized_batch):
        """
        Description:
            remove all [cls], [sep], and padding vectors from hidden states vector

        Parameters:
            LM_hidden_states : output of the LM for a batch. S
                                hape: (batch_size, seq_len, len(LM_probed_layers), embedding_size)
            tokenized_batch: batch input sentences after tokenization
            device: device which is used for computation (CPU/GPU/TPU)

        Returns:
            processed_hidden_states: processed hidden states vector
        """

        # stack the hidden states for the words of all sentences, and stack the number of layers with the
        #           embedding size.
        # this means that there is a 1-D vector for every subword
        temp_hidden_states = LM_hidden_states.view(
            LM_hidden_states.shape[0] * LM_hidden_states.shape[1], -1
        )  # shape: (batch_size*seq_len, num_layers*embedding_size)

        if "bert" in self.LM_name.lower():
            # create a mask to remove [cls], [sep], and padding tokens
            hidden_state_mask = self.create_MLM_mask(
                tokenized_batch
            )  # shape: (batch_size*seq_len)
        else:
            # create a mask to remove padding tokens
            hidden_state_mask = self.create_pad_mask(
                tokenized_batch
            )  # shape: (batch_size*seq_len)

        # repeat the mask so that it has the same shape as that of the hidden states
        # "repeat" allocates double the memory required for the tensor for some reason, "expand" doesn't allocate
        #           any new memory at all (just gets references to the original tensor)
        # adjusted_mask has to be of the same datatype of hidden_states so as not to allocate more memory
        #           than necessary in the elementwise multiplication
        adjusted_mask = torch.unsqueeze(hidden_state_mask, -1).expand(
            temp_hidden_states.shape[0], temp_hidden_states.shape[-1]
        )  # shape: (batch_size*seq_len, num_layers*embedding_size)
        # apply the mask to the hidden representation tensor
        processed_hidden_states = torch.mul(
            temp_hidden_states, adjusted_mask
        )  # shape: (batch_size*seq_len, num_layers*embedding_size)
        # remove rows with all zero columns ([cls], [sep], and padded hidden representations)
        processed_hidden_states = processed_hidden_states[
            torch.any(processed_hidden_states != 0, dim=-1)
        ]

        return processed_hidden_states

    def get_last_subword_hidden_state(
        self, batch, processed_hidden_states, tokenized_batch
    ):
        """
        Description:
            every word in the sentence is split into a subword, each having its own token and hidden state vector.
            For probing, there needs to be only 1 hidden state per output label, so we select the hidden state of
            the last subword of a word as the representative hidden state for the whole word

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
            # iterate over every word in the sentence
            # we still have to iterate over every word in the sentence to properly increment first_subword_index and
            #       last_subword_index
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

    def get_last_token_hidden_state(
        self, batch, processed_hidden_states, tokenized_batch
    ):
        """
        Description:
            function extracts hidden state of last token in an input example

        Parameters:
            batch : list of str, a list of the input sentences
            processed_hidden_states: hidden states vector after removing [cls], [sep], and padding vectors
            tokenized_batch: batch input sentences after tokenization

        Returns:
            last_token_hidden_states: hidden states for the last tokens in the sentences of the batch
        """

        token_counter = 0
        last_token_indices = (
            []
        )  # list that stores index of the last subword of the words in every sentence
        for i, input_example in enumerate(batch):
            if "bert" in self.LM_name.lower():
                # get example tokenization while removing [cls] and [sep] token
                tokenized_example = self.LM_tokenizer(input_example).data["input_ids"][
                    1:-1
                ]
            else:
                # get example tokenization
                tokenized_example = self.LM_tokenizer(input_example).data["input_ids"]

            last_token_index = (
                token_counter + len(tokenized_example) - 1
            )  # last token index in processed_hidden_states

            last_token_indices.append(
                last_token_index
            )  # equivalent to index of hidden state of last token
            token_counter = last_token_index + 1

        # extract the rows that have the last token hidden states
        last_token_hidden_states = torch.squeeze(
            processed_hidden_states[[last_token_indices], ...]
        )

        assert (
            token_counter == processed_hidden_states.shape[0]
        ), "error in extracting last token hidden states"

        return last_token_hidden_states

    def extract_word_hidden_states(
        self, batch, processed_hidden_states, tokenized_batch, extraction_method="last"
    ):
        """
        Description:
            get the hidden representation of a word from its subword token representations. This is done either
            by selecting last subword hidden state or by averaging the hidden states for all subwords

        Parameters:
            batch: list of str, a list of the input sentences
            processed_hidden_states: torch tensor, hidden states tensor for whole batch after removing [cls], [sep]
                                    representations and all padded representations
            tokenized_batch: input batch after tokenization
            extraction_method : str, denotes whether a word should be represented by it last subword
                                            representation or average of all corresponding subword representations

        Returns:
            word_hidden_states: torch tensor, hidden states for each word in the sentences of the batch
        """

        if "last" in extraction_method.lower():
            # for each word, assign the hidden state of the last subword as the hidden state for the whole word
            word_hidden_states = self.get_last_subword_hidden_state(
                batch, processed_hidden_states, tokenized_batch
            )
        else:
            # for each word, assign the average of the hidden states of the subword as the hidden state
            #       for the whole word
            raise Exception("hidden states averaging not implemented yet")

        return word_hidden_states

    def extract_sentence_hidden_states(
        self,
        batch,
        processed_hidden_states,
        tokenized_batch,
        extraction_method="last",
        attention_mask=None,
    ):
        """
        Description:
            get the hidden state of a whole sentence. This is done either by selecting the hidden state of the last
            subword in the sentence or by averaging the hidden states for all subwords in the sentence

        Parameters:
            batch: list of str, a list of the input sentences
            processed_hidden_states: torch tensor, hidden states tensor for whole batch after removing [cls], [sep]
                                    representations and all padded representations
            tokenized_batch: input batch after tokenization
            extraction_method : str, denotes whether a sentence should be represented by it last subword
                                            hidden state or average of all corresponding subword hidden states

        Returns:
            sent_hidden_states: torch tensor, hidden states for each sentence in the batch
        """

        # reshape hidden states vector to concatenate hidden states for all layers for each example
        processed_hidden_states = processed_hidden_states.view(
            processed_hidden_states.shape[0], processed_hidden_states.shape[1], -1
        )  # shape: (batch_size, seq_len, num_layers*embedding_size)

        # get sequence length of each example in the batch
        example_lens = torch.argmin(attention_mask, dim=-1)
        # the example (row) with the biggest length is assigned with zero in example_lens (argmin)
        #       since it has only ones in its columns
        # adjust for the row with all ones
        example_lens[example_lens == 0] = attention_mask.shape[-1]

        # For Llama models: padded tokens get non-zero hidden states ==> multiply hidden states matrix
        #                   by attention mask to zero them out
        attention_mask = attention_mask.unsqueeze(-1)
        processed_hidden_states = torch.mul(processed_hidden_states, attention_mask)

        if "last" in extraction_method.lower() and self.extracted_layers != [0]:
            last_token_index = example_lens - 1
            # Extract the specified indices from each row in processed_hidden_states
            sent_hidden_states = processed_hidden_states[
                torch.arange(processed_hidden_states.shape[0]), last_token_index
            ]
            if self.testing:
                for i, example in enumerate(batch):
                    hidden_state_example, *_ = self.infer_LM(example)
                    hidden_state_example = hidden_state_example.hidden_states[0]
                    last_hidden_state = hidden_state_example[0, -1, :]
                    assert torch.equal(
                        last_hidden_state, sent_hidden_states[i]
                    ), "Error in extracting last token hidden state"

            # For (i, example) loop here, loop through all the example in the batch (list of sentences)
            # infer_LM(example) - returns a vector, dimension should be 1
            # (batch size, seq_len, num_layers, embedding_size)
            # last hidden state = output(...-1...)
            # compare with sent_hidden_states
        else:
            # for each sentence, assign the average of the hidden states of the subwords as the feature vector
            #       for the whole sentence
            # Reshape example_lens to enable broadcasting
            example_lens = example_lens.view(-1, 1)
            sent_hidden_states = (
                torch.sum(processed_hidden_states, dim=1) / example_lens
            )

            if self.testing:
                for i, example in enumerate(batch):
                    hidden_state_example, *_ = self.infer_LM(example)
                    hidden_state_example = hidden_state_example.hidden_states[0]
                    avg = (
                        torch.sum(hidden_state_example, dim=1)
                        / hidden_state_example[0].shape[0]
                    )
                    avg = torch.squeeze(avg)
                    assert torch.equal(
                        avg, sent_hidden_states[i]
                    ), "Error in extracting average of hidden states"

        return sent_hidden_states

    def extract_hidden_states(
        self, batch, extraction_type="word-level", extraction_method="last"
    ):
        """
        Description:
            get the hidden representations of the LM for each input sentence for a single batch

        Parameters:
            batch: list of str, a list of the input sentences
            extraction_type: str, denotes whether word-level or sentence-level hidden states are to be extracted

        Returns:
            processed_hidden_states: processed hidden states tensor
            tokenized_batch: input batch after tokenization
        """

        # run the LM in inference mode
        LM_output, tokenized_batch, attention_mask = self.infer_LM(batch)

        # get hidden states before going into extraction functions
        hidden_states_tuple = (
            LM_output.hidden_states
        )  # tuple of tensors. tuple size = num_layers, each layer is a tensor with shape
        #                                                                (batch_size, seq_len, embedding_size)
        # convert hidden states from tuple of tensors to tensor of tensors (place num_layers dimension beside
        #                           the embedding_size dimension so that they can be later joined into one dimension)
        hidden_states = torch.squeeze(
            torch.stack(hidden_states_tuple, dim=-2)
        )  # shape: (batch_size, seq_len, num_layers, embedding_size)
        if len(batch) == 1:  # shape: (seq_len, num_layers, embedding_size)
            hidden_states = hidden_states.unsqueeze(
                0
            )  # shape: (batch_size, seq_len, num_layers, embedding_size)

        if "word" in extraction_type.lower():
            # remove any padded tokens, [cls], [sep]
            processed_hidden_states = self.process_LM_hidden_states(
                hidden_states, tokenized_batch
            )
            extracted_hidden_states = self.extract_word_hidden_states(
                batch, processed_hidden_states, tokenized_batch, extraction_method
            )
        else:
            extracted_hidden_states = self.extract_sentence_hidden_states(
                batch, hidden_states, tokenized_batch, extraction_method, attention_mask
            )

        return extracted_hidden_states
