"""
Unit tests for Feature Schema Adapter (63 vs 78 compatibility).
"""

from __future__ import annotations
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from core.preprocessing.schema_adapter import FeatureSchemaAdapter


def test_schema_adapter_63_to_78():
    adapter = FeatureSchemaAdapter()
    df_63 = pd.DataFrame(np.random.randn(10, 63))
    arr_78 = adapter.align_to_78_features(df_63)
    assert arr_78.shape == (10, 78)
    assert np.all(arr_78[:, 63:] == 0.0)


def test_schema_adapter_78_direct():
    adapter = FeatureSchemaAdapter()
    df_78 = pd.DataFrame(np.random.randn(10, 78))
    arr_78 = adapter.align_to_78_features(df_78)
    assert arr_78.shape == (10, 78)


if __name__ == "__main__":
    test_schema_adapter_63_to_78()
    test_schema_adapter_78_direct()
    print("All preprocessing tests passed.")
