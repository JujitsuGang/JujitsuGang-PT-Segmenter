import collections
import os
import json
import functools

import torch
import datasets
import numpy as np
import tqdm

import utils


def run_few_shot_finetuning(input_: dict[str, list[int]], lang: str, *, random_init: bool = False):
    if random_init:
        lang = f"{lang}-random"

    n = len(input_["labels"])
    rng = np.random.RandomState(19888012)

    output_dir = f"results/few_shot_fine_tuni