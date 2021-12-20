"""Tests that are expected to fail, raise expections or warnings."""
import pytest

import segmentador
import segmentador.optimize

from . import paths


@pytest.mark.parametrize("batch_size", (0, -1, -100))
def test_invalid_batch_size(
    fixture_model_bert_2_layers: segmentador.Segmenter,
    fixture_legal_text_short: str,
    batch_size: int,
):
    with pytest.raises(ValueError):
        fixture_model_bert_2_layers(fixture_legal_text_short, batch_size=batch_size)


@pytest.mark.parametrize("moving_window_size", (0, -1, -100))
def test_invalid_moving_window_size(
    fixture_model_bert_2_layers: se