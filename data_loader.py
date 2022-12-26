import os
import logging

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, Dataset

from tqdm.auto import tqdm

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class MRCLoader:
    def __init__(self,
                 config):
        self.raw_datasets = None
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path)
        self.pad_on_right = self.tokenizer.padding_side == "right"
        self.question_column_name = "question"
        self.answer_column_name = "answers"
        self.context_column_name = "context"

    def get_dataset(self, evaluate=False, output_examples=False):
        dataset_type = "validation" if evaluate else "train"
        cached_file_name = f"cached_{self.config.dataset_name}_{dataset_type}"
        cached_features_file = os.path.join(self.config.data_dir, cached_file_name)

        if os.path.exists(cached_features_file):
            logger.info(f"Loading features from cached file {cached_features_file}")
            examples_and_dataset = torch.load(cached_features_file)
            examples, dataset = (
                examples_and_dataset["examples"],
                examples_and_dataset["dataset"]
            )
        else:
            if self.config.dataset_name == "docent":
                self.raw_datasets = load_dataset(self.config.data_dir)

            else:
                self.raw_datasets = load_dataset(
                    self.config.dataset_name,
                    self.config.task if self.config.dataset_name == "klue" else None
                )

                if self.config.dataset_name.startswith("squad"):
                    self.raw_datasets = self.raw_datasets.rename_column("id", "guid")

            logger.info(f"Creating features from dataset file at {self.config.data_dir}")

            examples, dataset = self._create_dataset(dataset_type)

            logger.info(f"Saving features into cached file {cached_features_file}")
            torch.save({"dataset": dataset, "examples": examples}, cached_features_file)

        if output_examples:
            return examples, dataset
        return dataset

    def _create_dataset(self, dataset_type):
        feature_preparation = self.prepare_validation_features if dataset_type == "validation" else self.prepare_train_features
        examples = self.raw_datasets[dataset_type]

        if self.config.dataset_name == "docent":
            examples = self.make_dataset_from_docent(examples, dataset_type)

        dataset = examples.map(
            feature_preparation,
            batched=True,
            remove_columns=examples.column_names,
            desc=f"Running tokenizer on {dataset_type} dataset",
        )

        return examples, dataset

    # Training preprocessing
    @staticmethod
    def make_dataset_from_docent(docent_dataset, dataset_type):
        examples = []
        mode = True if dataset_type == "train" else False
        for q in tqdm(docent_dataset):
            context_id = q["id"]
            context = q["explain"]
            title = q["title"]

            for qa in q["q&a"]:
                id_ = f'{context_id}-{qa["QnAID"]}'
                question = qa["Questions"]
                answer_text = qa["Answer"].strip()
                question_type = qa["Type"]
                start_position_character = qa["StartPoint"]

                answers = {"text": [answer_text], "answer_start": [start_position_character]}

                if not mode:
                    answer_text = None
                    start_position_character = None

                example = {
                    "question_type": question_type,
                    "guid": id_,
                    "question": question,
                    "answers": answers,
                    "context": context,
                    "answer_text": answer_text,
                    "start_position_character": start_position_character,
                    "title": title,
                    "is_impossible": False,
                }

                examples.append(example)
        return Dataset.from_list(examples)

    def prepare_train_features(self, examples):
        examples[self.question_column_name] = [q.lstrip() for q in examples[self.question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples[self.question_column_name if self.pad_on_right else self.context_column_name],
            examples[self.context_column_name if self.pad_on_right else self.question_column_name],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.config.max_seq_length,
            stride=self.config.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=True if self.config.use_token_types else False,
            padding="max_length" if self.config.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[self.answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if self.pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if self.pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    # Validation preprocessing

    def prepare_validation_features(self, examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[self.question_column_name] = [q.lstrip() for q in examples[self.question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples[self.question_column_name if self.pad_on_right else self.context_column_name],
            examples[self.context_column_name if self.pad_on_right else self.question_column_name],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.config.max_seq_length,
            stride=self.config.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=True if self.config.use_token_types else False,
            padding="max_length" if self.config.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if self.pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["guid"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples
    # Docent-processing
