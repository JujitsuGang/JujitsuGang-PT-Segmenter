import itertools

import nltk

import baseline_utils
import approx_recall_and_precision


NATURAL_TEXT_SEG_ABBRV = {
    "art",
    "arts",
    "profa",
    "profᵃ",
    "dep",
    "sr",
    "sra",
    "srᵃ",
    "s.exª",
    "s.exa",
    "v.em.ª",
    "v.ex.ª",
    "v.mag.ª",
    "v.em.a",
    "v.ex.a",
    "v.mag.a",
    "v.m",
    "v.rev.ª",
    "v.rev.a",
    "v.s",
    "v.s.ª",
    "v.s.a",
    "v.a",
    "v.emª",
    "v.exª",
    "v.ema",
    