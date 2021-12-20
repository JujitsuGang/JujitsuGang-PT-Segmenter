"""General tests for arguments, model instantiation and segmentation."""
import typing as t

import pytest
import pandas as pd
import datasets

import segmentador

from . import paths


def no_segmentation_at_middle_subwords(segs: t.List[str]) -> bool:
    """Check if no word has been segmented."""
    return not any(s.startswith("##") for s in segs)


@pytest.mark.parametrize("pooling_operation", ("max", "sum", "asymmetric-max", "gaussian"))
def test_inference_pooling_operation_argument_with_long_text_and_bert(
    pooling_operation: str, fixture_test_paths: paths.TestPaths, fixture_legal_text_long: str
):
    model = segmentador.Segmenter(
        uri_model