import timeit

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    get_scheduler,
    default_data_collator,
    set_seed)
from tqdm.auto import tqdm

from accelerate import Accelerator

import logging

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, config, train_dataset=None, test_dataset=None):
        self.config = config
        self.num_train_epochs = config.num_train_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForQuestionAnswering.from_pretrained(config.PLM)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.to(device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler,
                                      batch_size=self.config.train_batch_size)
        t_total = len(train_dataloader) * self.num_train_epochs

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
        accelerator = Accelerator()
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate, eps=self.config.adam_epsilon)
        scheduler = get_scheduler(self.model.parameters(), lr=self.config.learning_rate, num_training_steps=t_total)

        if self.config.fp16:
            self.model, optimizer, train_dataloader = accelerator.prepare(self.model, optimizer, train_dataloader)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.config.train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    self.config.train_batch_size * accelerator.num_processes * self.config.gradient_accumulation_steps)
        logger.info("  Gradient Accumulation steps = %d", self.config.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 1
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()

        train_iterator = tqdm(range(epochs_trained, int(self.num_train_epochs)), desc="Epoch")
        set_seed(self.config.seed)

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

            self.model.train()
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)

            loss = outputs[0]
            if self.config.gradient_accumulation_steps > 1:
                loss = loss / self.config.gradient_accumulation_steps

            if self.config.fp16:
                accelerator.backward(loss)

            tr_loss += loss.item()
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.fp16:
                    torch.nn.utils.clip_grad_norm_(accelerator.master_params(optimizer), self.config.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                self.model.zero_grad()
                global_step += 1

                if self.config.logging_steps > 0 and global_step % self.config.logging_steps == 0:
                    if self.config.evaluate_during_training:
                        results = self.evaluate()
                        for key, value in results.items():
                            logger.info("  %s = %s", key, value)

    def evaluate(self):
        eval_sampler = SequentialSampler(self.test_dataset)
        eval_dataloader = DataLoader(self.test_dataset, sampler=eval_sampler, batch_size=self.config.eval_batch_size)

        logger.info("***** Running evaluation *****")
        logger.info(f"  Num examples = {len(self.test_dataset)}")
        logger.info(f"  Batch size = {self.config.eval_batch_size}")

        all_result = []
        start_time = timeit.default_timer()

        for batch in tqdm(eval_dataloader, desc="Evaluation"):
            self.model.eval()
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = self.model(**batch)

            for i in range(len(outputs.start_logits)):
                start_logits = outputs.start_logits[i].detach().cpu().numpy()
                end_logits = outputs.end_logits[i].detach().cpu().numpy()
                all_result.append((start_logits, end_logits))

        eval_time = timeit.default_timer() - start_time
        logger.info(
            f"  Evaluation done in total {eval_time} secs {eval_time / len(self.test_dataset)} sec per example)")

        predictions = compute_predictions_logits(
            self.test_dataset,
            all_result,
            self.config.n_best_size,
            self.config.max_answer_length,
            self.config.do_lower_case,
            tokenizer
        )

        result = squad_evaluate(self.test_dataset, predictions)
        return result

