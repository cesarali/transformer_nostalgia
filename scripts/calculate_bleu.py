from collections import defaultdict
from email.policy import default
from statistics import mean
from typing import List
from unittest import result
import click
import csv
import string
import pandas as pd
import glob
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import os


@click.command()
@click.option("--reference-file", "-r", type=click.Path(exists=True), help="Reference file.", required=True, default="/rdata/results/generated_atomic2/real_test_set.csv")
@click.option("--hypothesis-root", "-hyp", type=click.Path(exists=True), help="Hypothesis root dir.", required=True, default="/rdata/results/generated_atomic2/")
@click.option("--n-gram", "-n", type=int, default=2, help="N-grap.", required=True)
def main(reference_file: str, hypothesis_root: str, n_gram: int):
    result = defaultdict(list)
    for hypothesis_file in glob.glob(os.path.join(hypothesis_root, '**', '*.csv'), recursive=True):
        if hypothesis_file == reference_file or 'results' in os.path.split(hypothesis_file)[-1]:
            continue
        path, sample_name = os.path.split(hypothesis_file)
        print(sample_name)
        folder_name = os.path.split(path)[-1]
        result['folder'].append(folder_name)
        result['name'].append(sample_name)
        reference_agg, reference_cat = load_file(reference_file, 1. / 3.)
        hypothesis_agg, hypothesis_cat = load_file(hypothesis_file, 1.1)

        # aggregated
        data = merge_files(reference_agg, hypothesis_agg)
        assert len(reference_agg) == len(data)

        score_agg = bleuscore(n_gram, data)
        print(f"BLEU {n_gram}-GRAM-Smoothing1 AGG {score_agg:.4f}")
        result['agg'].append(score_agg)
        # category

        for relation in reference_cat.keys():
            ref_relation = reference_cat[relation]
            data = merge_files(ref_relation, hypothesis_cat[relation])
            assert len(ref_relation) == len(data)

            score = bleuscore(n_gram, data)
            print(f"BLEU {n_gram}-GRAM-Smoothing1 {relation} {score:.4f}")
            result[relation].append(score)

    result = pd.DataFrame(result)
    result.to_csv(os.path.join(hypothesis_root, 'results.csv'), index=False)


def bleuscore(n_gram, data):
    weights = [1. / n_gram] * n_gram
    chencherry = SmoothingFunction()
    score = 0
    for x in data:
        instance_score = []
        for j in range(len(x[2])):
            instance_score.append(sentence_bleu(
                x[1], x[2][j], weights=weights, smoothing_function=chencherry.method1))
        instance_score = mean(instance_score)
        score += instance_score
        # print(instance_score)

    return score / len(data)


def merge_files(referece: dict, hypothesis: dict) -> List[tuple]:
    result = []
    for key, ref in referece.items():
        result.append((key, ref, hypothesis[key]))
    return result


def load_file(reference_file: str, proportion: float):
    values_agg = dict()
    values_category = defaultdict(dict)
    with open(reference_file, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            key = row[0]
            key = key.replace(" 's", "'s").replace(" n't", "n't").replace(" ' ", "'").replace(" , ", ", ")
            relation = row[1]
            ref = row[2:]
            if is_valid_reference(ref, proportion):
                ref = [process_text(r) for r in ref]
                values_category[relation][key] = ref
                values_agg[key + relation] = ref
    return values_agg, values_category


def remove_punctuation(x: str) -> str:
    return x.translate(str.maketrans('', '', string.punctuation))


def normalize_person(x: str) -> str:
    return x.replace('person x', 'personx').replace('person y', 'persony')


def process_text(text: str) -> List[str]:
    text = text.lower()
    text = remove_punctuation(text)
    text = normalize_person(text)
    return text.split()


def is_valid_reference(ref: list, proportion: float = 1./3.):
    total_len = len(ref)
    count = sum([i.lower() == 'none' for i in ref])

    return (count / float(total_len)) < proportion


if __name__ == "__main__":
    main()
