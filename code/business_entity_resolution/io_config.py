"""Shared TSV reading configuration for the Amazon ML Challenge 2026 entity-resolution code.

Every loader in this package (profiling checks, feature builders, training/inference data
loaders, ...) MUST read the challenge TSVs through ``READ_KWARGS`` / ``read_tsv`` so that all
code sees identical values:

* ``sep="\\t"``                  - files are tab separated;
* ``dtype=str``                 - IDs such as ``S2-000123`` must never become numbers;
* ``keep_default_na=False``     - empty fields stay ``""`` (an empty ``matched_entity_ids``
                                  is a Source-1 singleton, not a missing value), and strings
                                  like ``"NA"``/``"null"`` in names are kept literally;
* ``quoting=csv.QUOTE_NONE``    - quote characters inside business names are data, not
                                  field delimiters.

Use ``usecols`` / ``chunksize`` to bound memory on the multi-million-row files. Note that pandas
pads rows with too few fields with NaN; callers must detect and report such malformed rows (see
``profiling.ground_truth_checks.scan_tsv_structure``) rather than hide them.
"""
from __future__ import annotations

import csv

import pandas as pd

READ_KWARGS = dict(sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)


def read_tsv(path, usecols=None, chunksize=None, **overrides):
    """``pd.read_csv`` with the shared ``READ_KWARGS``; extra keyword arguments are passed through
    (they may add options such as ``on_bad_lines`` but should not change the four above)."""
    kwargs = dict(READ_KWARGS)
    kwargs.update(overrides)
    return pd.read_csv(path, usecols=usecols, chunksize=chunksize, **kwargs)
