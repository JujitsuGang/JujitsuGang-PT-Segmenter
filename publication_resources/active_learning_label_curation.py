import argparse
import datetime

import segmentador
import datasets
import scipy.special
import numpy as np
import tqdm


def sample_instances(
    dt: datasets.Dataset,
    segmenter: segmentador.Segmenter,
    n: int,
    q: float = 0.01,
    *,
    ignore_label: int = -100,
) -> list[int]:
    instance_quantile_margins = np.full(len(dt), fill_value=np.inf)

    for i in tqdm.tqdm(range(len(dt))):
        inst = dt[i]

        token_logits = segmenter(
            inst,
            return_logits=True,
            regex_justificativa="__IGNORE__",  # Do not suppress Justification or Annex parts.
        ).logits

        probs = scipy.special.softmax(token_logits, axis=-1)
        token_margins = np.diff(np.sort(probs, axis=-1)[:, [-2, -1]], axis=-1).ravel()
        true_labels = np.asarray(inst["labels"], dtype=int)

        assert len(token_logits) == len(inst["input_ids"])
        assert len(token_logits) == len(true_labels)

        try:
            is_not_middle_subword = true_labels != ignore_label  # Ignore margins from middle subwords; they don't have labels.
      