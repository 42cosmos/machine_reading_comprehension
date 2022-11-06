import os

import yaml
import argparse
from typing import Any, Dict, List, Union

from easydict import EasyDict
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.data.processors.squad import SquadExample, squad_convert_examples_to_features

import logging

logger = logging.getLogger(__name__)


class KlueMRCExample(SquadExample):
    def __init__(self, question_type: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.question_type = question_type


class KlueMRCProcessor:
    def __init__(self):
        with open("config.yaml") as f:
            saved_hparams = yaml.load(f, Loader=yaml.FullLoader)
            self.hparams = EasyDict(saved_hparams)["CFG"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.PLM, use_fast=False)

    def get_dataset(self, evaluate=False, output_examples=False):
        dataset_type = "validation" if evaluate else "train"
        cached_file_name = f"cached_{self.hparams.task}_{self.hparams.max_seq_length}_{self.hparams.dset_name}_{dataset_type}"
        cached_features_file = os.path.join(self.hparams.data_dir, cached_file_name)

        if os.path.exists(cached_features_file):
            logger.info(f"Loading features from cached file {cached_features_file}")
            features_and_dataset = torch.load(cached_features_file)
            features, dataset, examples = (
                features_and_dataset["features"],
                features_and_dataset["dataset"],
                features_and_dataset["examples"],
            )

        else:
            logger.info(f"Creating features from dataset file at {self.hparams.data_dir}")

            if evaluate:
                examples = self._create_examples(is_training=False)
            else:
                examples = self._create_examples(is_training=True)

            features, dataset = self._create_dataset(examples, dataset_type)

            logger.info(f"Saving features into cached file {cached_features_file}")
            torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

        if output_examples:
            return dataset, examples, features
        return dataset

    def _create_dataset(self, examples, dataset_type: str):
        is_training = dataset_type == "train"
        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=self.tokenizer,
            max_seq_length=self.hparams.max_seq_length,
            doc_stride=self.hparams.doc_stride,
            max_query_length=self.hparams.max_query_length,
            is_training=is_training,
            return_dataset="pt",
            threads=self.hparams.threads,
        )

        if not is_training:
            data = getattr(self.hparams, "data", {})
            data[dataset_type] = {"examples": examples, "features": features}
            setattr(self.hparams, "data", data)

        return features, dataset

    def _create_examples(self, is_training: bool = True):
        examples = []
        mode = "train" if is_training else "validation"
        data = load_dataset(self.hparams.dset_name, self.hparams.task, split=mode)
        for q in tqdm(data):
            context = q["context"]
            question = q["question"]
            question_type = q["question_type"]
            id_ = q["guid"]
            answer_impossibility = q["is_impossible"]
            if not is_training:
                answers = q["answers"]
                answer_text = None
                start_position_character = None
            elif is_training and not answer_impossibility:
                answers = []
                answer_text = q["answers"]["text"][0]
                start_position_character = q["answers"]["answer_start"][0]
            elif is_training and answer_impossibility:
                answers = []
                answer_text = ""
                start_position_character = -1

            examples.append(
                KlueMRCExample(
                    question_type=question_type,
                    qas_id=id_,
                    question_text=question,
                    context_text=context,
                    answer_text=answer_text,
                    start_position_character=start_position_character,
                    title=q["title"],
                    is_impossible=answer_impossibility,
                    answers=answers,
                )
            )

        return examples
