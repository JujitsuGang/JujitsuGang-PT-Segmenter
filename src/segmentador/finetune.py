"""Module for fine-tuning pretrained segmenter models in new documents."""
import typing as t
import functools
import copy

import numpy as np
import torch
import torch.nn
import tqdm


def _label_noise_tokens(
    input_ids: t.List[int],
    labels: t.List[int],
    tokens: t.List[str],
    noise_start_id: int,
    noise_end_id: int,
) -> t.Tuple[t.List[str], t.List[int], t.List[int]]:
    """Place labels to noise sequences enclosed by `noise_start_id` and `noise_end_id`."""
    input_ids_np = np.array(input_ids, dtype=int)
    noise_start_inds = np.flatnonzero(input_ids_np == noise_start_id)

    if noise_start_inds.size > 0:
        noise_end_inds = np.hstack((np.flatnonzero(input_ids_np == noise_end_id), input_ids_np.size))

        for i_start, i_end in zip(noise_start_inds, noise_end_inds):
            if i_end > i_start + 1:
                labels[i_start + 1] = 2 if labels[i_start + 1] != -100 else -100
                if i_end + 1 < input_ids_np.size:
                    labels[i_end + 1] = 3 if labels[i_end + 1] != -100 else -100

            input_ids[i_start] = -1
            if i_end < input_ids_np.size:
                input_ids[i_end] = -1

        labels = [lab for lab, i in zip(labels, input_ids) if i >= 0]
        tokens = [tok for tok, i in zip(tokens, input_ids) if i >= 0]
        input_ids = [i for i in input_ids if i >= 0]

    assert len(input_ids) == len(labels)

    return (tokens, input_ids, labels)


def text_to_ids(
    segments: t.List[t.List[str]],
    tokenizer,
    noise_start_token: str,
    noise_end_token: str,
) -> t.Tuple[t.List[t.List[int]], t.List[t.List[int]]]:
    """Convert text segments to tokenized input ids."""
    input_ids: t.List[t.List[int]] = []
    labels: t.List[t.List[int]] = []

    tokenizer = copy.deepcopy(tokenizer)
    tokenizer.add_tokens([noise_start_token, noise_end_token], special_tokens=True)

    (noise_start_id, noise_end_id) = tokenizer.encode(
        f"{noise_start_token} {noise_en