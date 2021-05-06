import functools
import warnings

import torch
import torch.nn
import transformers
import datasets
import sklearn.metrics
import numpy as np
import segmentador


def fn_compute_metrics(labels, logits):
    assert labels.size
    assert logits.size

    preds = logits.argmax(-1).astype(int, copy=False)

    assert labels.ndim == 1
    assert preds.ndim == 1
    assert labels.size == preds.size, (labe