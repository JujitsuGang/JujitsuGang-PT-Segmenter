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
    "Santa Catari