import argparse
import os
import json
import yaml
from easydict import EasyDict

from metrics import f1_score
from data_loader import MRCLoader


def f1_by_character(examples, predictions: dict):
    f1 = 0

    for ex in examples:
        q_id = ex["guid"]
        ground_truth = ex["answers"]["text"][0]
        if q_id in predictions.keys():
            pred = predictions[q_id]
            f1 += f1_score(pred, ground_truth)

    return {"char-f1": f1 / examples.num_rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluation')
    parser.add_argument('--prediction_file', help='Prediction File')  # output/{dset_name}_predictions.json
    args = parser.parse_args()

    with open("config.yaml") as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    dset_name = args.prediction_file.split('/')[-1].split('_')[0]
    config.dataset_name = dset_name

    loader = MRCLoader(config)
    examples, _ = loader.get_dataset(evaluate=True, output_examples=True)

    with open(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)

    print(f1_by_character(examples=examples, predictions=predictions))
