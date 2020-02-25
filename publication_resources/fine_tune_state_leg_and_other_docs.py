import collections
import itertools
import json
import os

import matplotlib.pyplot as plt
import seaborn as sns
import datasets
import torch
import pandas as pd
import regex as re
import numpy as np
import sklearn.model_selection
import segmentador
import tqdm

import utils


STATES = {
    "São Paulo": "SP",
    "Minas Gerais": "MG",
    "Espírito Santo": "ES",
    "Rio de Janeiro": "RJ",
    "Rio Grande do Sul": "RS",
    "Santa Catarina": "SC",
    "Paraná": "PR",
    "Goiás": "GO",
    "Mato Grosso": "MT",
    "Mato Grosso do Sul": "MS",
    "Bahia": "BA",
    "Pernambuco": "PE",
    "Ceará": "CE",
    "Paraíba": "PB",
    "Rio Grande do Norte": "RN",
    "Maranhão": "MA",
    "Alagoas": "AL",
    "Sergipe": "SE",
    "Piauí": "PI",
    "Acre": "AC",
    "Amazonas": "AM",
    "Amapá": "AP",
    "Roraima": "RR",
    "Rondônia": "RO",
    "Pará": "PA",
    "Tocantins": "TO",
}


assert len(STATES) == 26


def load():
    reg_states = re.compile("(?<=ESTADO D[AEO] )" + "(?:" + "|".join(sorted(STATES.keys())).upper() + ")")

    tsv = pd.read_csv("data/dataset_ulysses_segmenter_train_v3.tsv", sep="\t", index_col=0)
    tsv = tsv.groupby("document_id").agg(lambda x: " ".join(x).upper())

    dist_by_state = collections.Counter()
    indices_by_state = collections.defaultdict(list)

    for i, (content,) in tsv.iterrows():
        detected_states = reg_states.findall(content)
        if not detected_states:
            continue
        state, freqs = np.unique(detected_states, return_counts=True)

        if state.size > 1 and freqs[0] == freqs[1]:
            continue

        dist_by_state[state[0]] += 1
        indices_by_state[state[0]].append(i)

    selected_states = {k for k, v in dist_by_state.most_common(100) if v >= 19}
    indices_by_state = {k: v for k, v in indices_by_state.items() if k in selected_states}

    dt = datasets.Dataset.load_from_disk("data/dataset_ulysses_segmenter_train_v3")

    return (dt, indices_by_state)


def check_misclass_quantiles(dt, indices_by_state):
    test_inference_kwargs = {
        "return_logits": True,
        "show_progress_bar": True,
        "window_shift_size": 1.0,
        "moving_window_size": 1024,
        "apply_postprocessing": False,
        "batch_size": 16,
    }

    segmenter = segmentador.BERTSegmenter(uri_model="4_layer_6000_vocab_size_bert_v2", device="cuda:0")
    misclass_fracs = []

    for inds in tqdm.tqdm(indices_by_state.values()):
        for i in inds[:20]:
            dt_cur = dt.select([i]).to_dict()
            labels_true = np.asarray(dt_cur.pop("labels")[0])
            labels_pred = segmenter(dt_cur, **test_inference_kwargs, return_labels=True).labels
            assert labels_true.shape == labels_pred.shape
            labels_pred = labels_pred[labels_true != -100]
            labels_true = labels_true[labels_true != -100]
            assert labels_pred.size == labels_true.size
            misclass_inds_frac = np.flatnonzero(labels_pred != labels_true) / labels_true.size
            misclass_fracs.extend(misclass_inds_frac.tolist())

    fig, ax = plt.subplots(1, figsize=(10, 5), layout="tight")
    sns.histplot(y=misclass_fracs, bins=50, ax=ax, stat="probability", kde=False)
    ax.set_ylabel("Misclassification relative location (w.r.t. instance size)")
    sns.despine(ax=ax)
    fig.savefig("misclass_state_leg_dist.pdf", format="pdf", bbox_inches="tight")
    # plt.show()


def undersample_inds(inds_by_class, rng, m: int):
    return sorted({k: rng.choice(v, size=m, replace=False) for k, v in inds_by_class.items()}.items())


def compute_undersampling_stats(dt, indices_by_state, m):
    rng = np.random.RandomState(1485)
    token_counts = []

    for _ in tqdm.tqdm(np.arange(100)):
        undersampled_inds = undersample_inds(indices_by_state, rng, m=m)
        token_count_per_state = [list(itertools.chain(map(len, dt.select(inds)["input_ids"]))) for _, inds in undersampled_inds]
   