"""Tests related to quantization and quantized segmenter models."""
import timeit
import os
import shutil

import segmentador
import segmentador.optimize

import pytest

from . import paths


@pytest.mark.skip(reason="Support for ONNX newer versions has been dropped.")
def test_inference_lstm_onnx(
    fixture_quantized_model_lstm_onnx: segmentador.optimize.ONNXLSTMSegmenter,
    fixture_legal_text_short: str,
):
    output = fixture_quantized_model_lstm_onnx(
        fixture_legal_text_short,
        return_labels=True,
        return_logits=True,
    )

    num_segments = len(output.segments)

    assert num_segments and output.logits.shape == (output.labels.size, 4)


@pytest.mark.skip(reason="Support for ONNX newer versions has been dropped.")
def test_inference_bert_onnx(
    fixture_quantized_model_bert_onnx: segmentador.optimize.ONNXBERTSegmenter,
    fixture_legal_text_short: str,
):
    output = fixture_quantized_model_bert_onnx(
        fixture_legal_text_short,
        return_labels=True,
        return_logits=True,
    )

    num_segments = len(output.segments)

    assert num_segments and output.logits.shape == (output.labels.size, 4)


@pytest.mark.skip(reason="Hardware dependant.")
def test_model_lstm_inference_time_standard_vs_quantized_torch(
    fixture_quantized_model_lstm_torch: segmentador.LSTMSegmenter,
    fixture_model_lstm_1_layer: segmentador.LSTMSegmenter,
    fixture_legal_text_long: str,
):
    common_kwargs = {
        "number": 10,
        "repeat": 3,
        "globals": {
            "fixture_quantized_model_lstm_torch": fixture_quantized_model_lstm_torch,
            "fixture_legal_text_long": fixture_legal_text_long,
            "fixture_model_lstm_1_layer": fixture_model_lstm_1_layer,
        },
    }

    times_quantized = timeit.repeat(
        "fixture_quantized_model_lstm_torch(fixture_legal_text_long, batch_size=4)",
        **common_kwargs,
    )
    times_standard = timeit.repeat(
        "fixture_model_lstm_1_layer(fixture_legal_text_long, batch_size=4)",
        **common_kwargs,
    )

    best_time_quantized = min(times_quantized)
    best_time_standard = min(times_standard)

    assert best_time_quantized < best_time_standard


@pytest.mark.skip(reason="Support for ONNX newer versions has been dropped.")
def test_model_lstm_inference_time_standard_vs_quantized_onnx(
    fixture_quantized_model_lstm_onnx: segmentador.optimize.ONNXLSTMSegmenter,
    fixture_model_lstm_1_layer: segmentador.LSTMSegmenter,
    fixture_legal_text_long: str,
):
    common_kwargs = {
        "number": 10,
        "repeat": 3,
        "globals": {
            "fixture_quantized_model_lstm_onnx": fixture_quantized_model_lstm_onn