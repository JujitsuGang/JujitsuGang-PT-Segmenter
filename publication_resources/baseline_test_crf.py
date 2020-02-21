import typing as t

import argparse
import collections
import itertools

import scipy.stats
import sklearn_crfsuite
import sklearn
import sklearn.model_selection
import nltk
import numpy as np

import baseline_utils
import approx_recall_and_precision


def word_to_feat(word: str) -> dict[str, t.Any]:
    return {
        "lower": word.lower(),
        "isart": word.lower() in {"art", "artigo"},
        "ispar": word.lower() in {"parágrafo", "§"},
        "sig": "".join("D" if c.isdigit() else ("c" if c.islower() else "C") for c in word),
        "length": len(word) if len(word) < 4 else ("long" if len(word) > 6 else "normal"),
        "islower": word.islower(),
        "isupper": word.isupper(),
        "istitle": word == word.capitalize(),
        "isdigit": word.isdigit(),
    }


def dataset_to_feats(docs):
    feats = []
    labels = []
    all_tokens = []

    for doc in docs:
        cur_feats = []
        cur_labels = []
        cur_tokens = []
        for sent in doc:
            tokens = nltk.tokenize.word_tokenize(sent, language="portuguese")
            if tokens:
                cur_feats.extend([word_to_feat(tok) for tok in tokens])
                cur_labels.extend(len(tokens) * ["NO-OP"])
                cur_labels[-len(tokens)] = "START"
                cur_tokens.extend(tokens)

        cur_labels[0] = "NO-OP"

        assert len(cur_feats) == len(cur_labels) == len(cur_tokens)

        feats.append(cur_feats)
        labels.append(cur_labels)
        all_tokens.append(cur_tokens)

    return (feats, labels, all_tokens)


def get_stats(labels):
    class_counter = collections.Counter()
    for yi in labels:
        class_counter.update(yi)
    return class_counter


def build_sents(tokens_per_doc, preds_per_doc):
    assert len(tokens_per_doc) == len(preds_per_doc)

    sents = []
    is_start_count = 0
    n_docs = len(tokens_per_doc)

    for cur_tokens, cur_labels in zip(tokens_per_doc, preds_per_doc, strict=True):
        cur_sent = []
        for tok, lab in zip(cur_tokens, cur_labels, strict=True):
            if lab == "START":
                sents.append(" ".join(cur_sent))
                cur_sent.clear()
                is_start_count += 1

            cur_sent.append(tok)

        if cur_sent:
            sents.append(" ".join(cur_sent))

    assert len(sents) == is_start_count + n_docs

    return sents


def run(redo_hparam_search: bool = False):
    # Data info: https://github.com/ulysses-camara/ulysses-segmenter?tab=readme-ov-file#train-and-evaluation-data
    # Tokenizer info: https://github.com/ulysses-camara/ulysses-segmenter?tab=readme-ov-file#available-models

    train_docs = baseline_utils.load_ground_truth_sentences(
        test_uri="data/dataset_ulysses_segmenter_v2_active_learning_curated_only",
        tokenizer_uri="tokenizers/6000_subwords",
        split="train",
        group_by_document=True,
    )

    test_docs = baseline_utils.load_ground_truth_sentences(
        test_uri="data/dataset_ulysses_segmenter_v2_active_learning_curated_only",
        tokenizer_uri="tokenizers/6000_subwords",
        split="test",
        group_by_document=True,
    )

    train_feats, train_labels, _ = dataset_to_feats(train_docs)
    test_feats, _, test_tokens = dataset_to_feats(test_docs)
    flatten_test_docs = list(itertools.chain(*test_docs))

    if not redo_hparam_search:
        best_score = 0.610011381785407
        best_config = {"c1": 0.0717248764429347, "c2": 0.019492472144742503, "min_freq": 7}

    else:
        best_score = -np.inf
        best_config = None

        param_distributions = {
            "c1": scipy.stats.expon(scale=0.5),
            "c