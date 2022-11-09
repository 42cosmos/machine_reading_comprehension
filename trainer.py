import os
import timeit

import torch
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    get_linear_schedule_with_warmup,
    EvalPrediction,
    set_seed, )

from transformers.data.metrics.squad_metrics import (
    compute_predictions_logits,
    squad_evaluate,
)

from transformers.data.processors.squad import SquadResult
from accelerate import Accelerator

import logging
from tqdm.auto import tqdm
from typing import List
from dataclasses import dataclass
from metrics import BaseMetric, klue_mrc_em, mrc_f1

logger = logging.getLogger(__name__)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


@dataclass
class QAResults:
    results: List[SquadResult]


class Trainer(object):
    def __init__(self,
                 config,
                 train_dataset=None,
                 test_dataset=None):
        self.config = config

        self.accelerator = Accelerator(fp16=config.fp16)
        self.device = self.accelerator.device
        self.metrics = BaseMetric(klue_mrc_em)

        self.num_train_epochs = config.num_train_epochs
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model_type = config.model_name_or_path.split("/")[1].split("-")[0]

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(config.model_name_or_path)

        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler,
                                      batch_size=self.config.train_batch_size)

        if self.config.max_steps > 0:
            t_total = self.config.max_steps
            self.num_train_epochs = self.config.max_steps // (
                    len(train_dataloader) // self.config.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.config.gradient_accumulation_steps * self.num_train_epochs

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]

        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=float(self.config.learning_rate),
                          eps=float(self.config.adam_epsilon))
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.config.warmup_steps,
                                                    num_training_steps=t_total)

        if os.path.isfile(os.path.join(self.config.model_name_or_path, "optimizer.pt")) and os.path.isfile(
                os.path.join(self.config.model_name_or_path, "scheduler.pt")
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(self.config.model_name_or_path, "optimizer.pt")))
            scheduler.load_state_dict(
                torch.load(os.path.join(self.config.model_name_or_path, "scheduler.pt")))

        self.model, optimizer, train_dataloader = self.accelerator.prepare(self.model, optimizer, train_dataloader)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.config.train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    self.config.train_batch_size * self.accelerator.num_processes * self.config.gradient_accumulation_steps)
        logger.info("  Gradient Accumulation steps = %d", self.config.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 1
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        if os.path.exists(self.config.model_name_or_path):
            try:
                checkpoint_suffix = self.config.model_name_or_path.split("-")[-1].split("/")[0]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (len(train_dataloader) // self.config.gradient_accumulation_steps)
                steps_trained_in_current_epoch = global_step % (
                        len(train_dataloader) // self.config.gradient_accumulation_steps)

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info(f"  Continuing training from epoch {epochs_trained}")
                logger.info(f"  Continuing training from global step {global_step}")
                logger.info(f"  Will skip the first {steps_trained_in_current_epoch} steps in the first epoch")
            except ValueError:
                logger.info("  Starting fine-tuning.")

        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()

        train_iterator = tqdm(range(epochs_trained, int(self.num_train_epochs)), desc="Epoch")
        set_seed(self.config.seed)

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "start_positions": batch[3],
                    "end_positions": batch[4],
                }
                # if self.config.version_2_with_negative:
                # check is_impossible values in dataset
                # inputs.update({"is_impossible": batch[5]})

                if self.model_type in ["distilbert", "roberta"]:
                    del inputs["token_type_ids"]

                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                self.model.train()
                outputs = self.model(**inputs)

                loss = outputs[0]
                if self.config.gradient_accumulation_steps > 1:
                    loss = loss / self.config.gradient_accumulation_steps

                self.accelerator.backward(loss)

                tr_loss += loss.item()

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(optimizer),
                                                   self.config.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()

                    wandb.log({"loss": loss})

                    global_step += 1

                    # Log Metrics
                    if self.config.logging_steps > 0 and global_step % self.config.logging_steps == 0:
                        if self.config.evaluate_during_training:
                            logger.info("***** Running Evaluation *****")
                            results = self.evaluate()

                        logging_loss = tr_loss

                    if self.config.save_steps > 0 and global_step % self.config.save_steps == 0:
                        output_dir = os.path.join(self.config.output_dir, f"checkpoint-{global_step}")
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
                        model_to_save.save_pretrained(output_dir)
                        self.tokenizer.save_pretrained(output_dir)

                        logger.info(f"Saving model checkpoint to {output_dir}")

                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info(f"Saving optimizer and scheduler states to {output_dir}")

                if 0 < self.config.max_steps < global_step:
                    epoch_iterator.close()
                    break
            if 0 < self.config.max_steps < global_step:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def evaluate(self):
        dataset, examples, features = self.test_dataset
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.config.eval_batch_size)

        logger.info("***** Running evaluation *****")
        logger.info(f"  Num examples = {len(dataset)}")
        logger.info(f"  Batch size = {self.config.eval_batch_size}")

        all_result = []
        start_time = timeit.default_timer()

        for batch in tqdm(eval_dataloader, desc="Evaluation"):
            self.model.eval()

            with torch.no_grad():
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }

                if self.model_type in ["distilbert", "roberta"]:
                    del inputs["token_type_ids"]

                outputs = self.model(**inputs)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits

            feature_indices = batch[3].tolist()

            for i, feature_index in enumerate(feature_indices):
                unique_id = int(features[feature_index].unique_id)
                single_example_start_logits = to_list(start_logits[i])
                single_example_end_logits = to_list(end_logits[i])

                result = SquadResult(unique_id, single_example_start_logits, single_example_end_logits)
                all_result.append(result)

        eval_time = timeit.default_timer() - start_time
        logger.info(
            f"  Evaluation done in total {eval_time} secs {eval_time / len(self.test_dataset)} sec per example)")

        output_prediction_file = os.path.join(self.config.output_dir, "predictions.json")
        output_nbest_file = os.path.join(self.config.output_dir, "nbest_predictions.json")

        if self.config.version_2_with_negative:
            output_null_log_odds_file = os.path.join(self.config.output_dir, "null_odds.json")
        else:
            output_null_log_odds_file = None

        predictions = compute_predictions_logits(
            all_examples=examples,
            all_features=features,
            all_results=all_result,
            n_best_size=self.config.n_best_size,
            max_answer_length=self.config.max_answer_length,
            do_lower_case=self.config.do_lower_case,
            tokenizer=self.tokenizer,
            output_null_log_odds_file=output_null_log_odds_file,
            null_score_diff_threshold=self.config.null_score_diff_threshold,
            version_2_with_negative=self.config.version_2_with_negative,
            verbose_logging=self.config.verbose_logging,
            output_prediction_file=output_prediction_file,
            output_nbest_file=output_nbest_file,
        )

        f1_result = mrc_f1(predictions=predictions, examples=examples)

        logger.info("***** EM / F1 results *****")
        for key, value in f1_result.items():
            logger.info(f"  {key} = {value}")

        wandb.log(f1_result)
        return f1_result
