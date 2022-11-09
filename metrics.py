import re
import string
import logging
import numpy as np

import torch
from difflib import SequenceMatcher
from typing import List, Dict, Tuple, Sequence, Any, Optional, Callable
from pytorch_lightning.metrics import Metric
import evaluate
from transformers import EvalPrediction

from klue_data_loader import KlueMRCExample

NORMALIZE_CHAR_PATTERN = re.compile(r"[\'\"《》<>〈〉]\(\)\‘\’")
PUNCTUATION_SET = set(string.punctuation)
KLUE_MRC_NUM_QUESTION_TYPE = 3

logger = logging.getLogger(__name__)


class BaseMetric(Metric):
    """Base class for metrics."""

    def __init__(
            self,
            metric_fn: Callable,
            compute_on_step: bool = True,
            dist_sync_on_step: bool = False,
            process_group: Optional[Any] = None,
            dist_sync_fn: Optional[Callable] = None,
            device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)

        self.metric_fn = metric_fn
        self.device = device

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Updates state with predictions and targets.
        Args:
            preds: Predictions from model
            targets: Ground truth values
        """

        self.preds.append(preds)
        self.targets.append(targets)

    def compute(self) -> Any:
        """Computes metric value over state."""

        preds = self.preds
        targets = self.targets

        if type(preds[0]) == torch.Tensor:
            preds = torch.cat(preds, dim=0)
            preds = preds.cpu().numpy()
        if type(targets[0]) == torch.Tensor:
            targets = torch.cat(targets, dim=0)
            targets = targets.cpu().numpy()

        score = self.metric_fn(preds, targets)
        score = torch.tensor(score).to(self.device)
        return score


def normalize_answer_for_klue_mrc(answer: str) -> str:
    """Excludes useless characters in answer string.
    Args:
        answer: The raw text of answer.
    Returns:
        The normalized answer.
    """
    answer = NORMALIZE_CHAR_PATTERN.sub(" ", answer.lower())
    answer = "".join(c for c in answer if c not in PUNCTUATION_SET)
    answer = " ".join(answer.split())
    return answer


def rouge_w_score_for_klue_mrc(pred: str, label: str, beta: int = 1) -> float:
    """Calculates character level ROUGE-W score https://en.wikipedia.org/wiki/ROUGE_(metric)"""
    if label == "":
        return float(pred == label)

    matcher = SequenceMatcher(None, pred, label)
    longest_common_consecutive_sequence_length = matcher.find_longest_match(0, len(pred), 0, len(label)).size

    precision = longest_common_consecutive_sequence_length / len(pred) if len(pred) else 0.0
    recall = longest_common_consecutive_sequence_length / len(label) if len(label) else 0.0

    if precision + recall == 0.0:
        return 0.0

    return (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall)


def compute_em_and_rouge_w_score_for_klue_mrc(pred: str, labels: List[str]) -> Tuple[float, float]:
    """Calculates Exact Match(EM) and ROUGE-W scores for single example.
    The maximum EM and ROUGE-W scores will be returned among the multiple labels.
    Args:
        pred: The predicted answer of single example.
        label: The ground truth answer of single example.
    Returns:
        Exact Match(EM), ROUGE-W score
    """
    em_scores, rouge_scores = [0.0], [0.0]

    for label in labels:
        em_scores.append(float(pred == label))
        rouge_scores.append(rouge_w_score_for_klue_mrc(pred, label))

    return max(em_scores), max(rouge_scores)


def klue_mrc_em(preds: List[Dict[str, str]], examples: List[List[KlueMRCExample]], mrc_num_question_type=5) -> Any:
    """KLUE-MRC Exact Match (EM)"""
    preds, examples = preds[0], examples[0]

    em_scores_per_question_type: List[List[float]] = [[] for _ in range(mrc_num_question_type)]
    for example in examples:
        prediction = normalize_answer_for_klue_mrc(preds[example.qas_id])
        ground_truths = [normalize_answer_for_klue_mrc(answer) for answer in example.answers["text"]]
        # For unanswerable questions, only correct answer is empty string
        if not ground_truths:
            ground_truths = [""]

        em_score, _ = compute_em_and_rouge_w_score_for_klue_mrc(prediction, ground_truths)
        em_scores_per_question_type[example.question_type - 1].append(em_score)

    logger.info("** Exact Match(EM) scores by type **")
    for question_type in range(mrc_num_question_type):
        question_type_em_scores = em_scores_per_question_type[question_type]
        avg_em_score = np.mean(question_type_em_scores) * 100.0
        logger.info(f"type{question_type + 1} ({len(question_type_em_scores)}): {avg_em_score:.4f}")

    total_em_scores = [score for scores in em_scores_per_question_type for score in scores]
    return np.mean(total_em_scores) * 100.0

#
# def mrc_f1(p: EvalPrediction):
#     """KLUE-MRC F1 score"""
#     metric = evaluate.load("squad")
#     return metric.compute(predictions=p.predictions, references=p.label_ids)

def mrc_f1(examples, predictions):
    metric = evaluate.load("squad")
    formatted_preds = [
        {"id": k, "prediction_text": v} for k, v in predictions.items()
    ]
    references = [{"id": ex.qas_id, "answers": ex.answers} for ex in examples]
    return metric.compute(predictions=formatted_preds, references=references)
