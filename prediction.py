import argparse
import os
import json
import yaml
from easydict import EasyDict

from metrics import f1_score
from data_loader import MRCLoader


def f1_by_character(predictions: dict):
    f1 = 0

    for q_id, value in predictions.items():
        ground_truth = value["original_text"][0]
        pred = value["predictions"][0]
        f1 += f1_score(pred, ground_truth)

    return {"char-f1": f1 / len(predictions)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluation')
    parser.add_argument('--prediction_file', help='Prediction File')  # model/{dset_name}_predictions.json
    args = parser.parse_args()

    with open("config.yaml") as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    dset_name = args.prediction_file.split('/')[-1].split('_')[0]
    config.dataset_name = dset_name

    with open(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)

    print(f1_by_character(predictions))
