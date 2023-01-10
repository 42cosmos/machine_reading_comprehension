import os
import logging
import argparse

import yaml
import random
import numpy as np
from easydict import EasyDict

from tqdm import tqdm, trange
import torch

from transformers import (
    set_seed,
    Trainer,
    TrainingArguments,
    AutoConfig,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    default_data_collator,
    EarlyStoppingCallback
)
from trainer import QuestionAnsweringTrainer
from transformers.trainer_utils import get_last_checkpoint

import evaluate
from seqeval.metrics import classification_report


from data_loader import MRCLoader
from metrics import post_processing_function, compute_metrics

logger = logging.getLogger(__name__)


def f1_by_character(predictions: dict):
    f1 = 0

    for q_id, value in predictions.items():
        ground_truth = value["original_text"][0]
        pred = value["predictions"][0]
        f1 += f1_score(pred, ground_truth)

    return {"char-f1": f1 / len(predictions)}


if __name__ == '__main__':
    with open("config.yaml", "r") as f:
        saved_config = yaml.load(f, Loader=yaml.FullLoader)
        hparams = EasyDict(saved_config)

    set_seed(hparams.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    if os.path.exists(hparams.output_dir) and len(os.listdir(hparams.output_dir)) > 1:
        hparams.model_name_or_path = hparams.output_dir
        logger.info(f"***** Load Model from {hparams.model_name_or_path} *****")
    else:
        logger.info(f"No checkpoint found, training from scratch: {hparams.model_name_or_path}")

    set_seed(hparams.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        hparams.model_name_or_path,
        revision=hparams.model_revision,
        use_fast=True,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        hparams.model_name_or_path,
        revision=hparams.model_revision,
    )

    data_collator = (
        default_data_collator
        if hparams.pad_to_max_length
        else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if hparams.fp16 else None)
    )


    loader = MRCLoader(hparams)

    test_examples, test_dataset = loader.get_dataset(dataset_type="test", output_examples=True)

    training_args = TrainingArguments(
        output_dir=hparams.output_dir,
        do_train=False,
        do_eval=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=hparams.train_batch_size,
        per_device_eval_batch_size=hparams.eval_batch_size,
        learning_rate=float(hparams.learning_rate),
        adam_epsilon=float(hparams.adam_epsilon),
        num_train_epochs=hparams.num_train_epochs,
        weight_decay=hparams.weight_decay,
        logging_steps=hparams.logging_steps,
        seed=hparams.seed,
        fp16=hparams.fp16,
        warmup_steps=hparams.warmup_steps,
        max_steps=hparams.max_steps,
        log_level="info",
    )

    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        eval_examples=test_examples,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )


    result = trainer.evaluate(eval_dataset=test_dataset, eval_examples=test_examples)
    for k, v in result.items():
        logger.info(f"{k}, {v}")
