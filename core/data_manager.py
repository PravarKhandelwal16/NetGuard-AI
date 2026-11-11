"""
Data Management and Caching Service.
Loads sample data, parses user CSV uploads, and handles session caching.
"""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import streamlit as st
from config.settings import SAMPLES_DIR

SAMPLE_CSV_PATH = SAMPLES_DIR / "cicids_sample_stream.csv"


@st.cache_data
def load_default_dataset() -> pd.DataFrame:
    """Loads the pre-generated representative CICIDS2017 streaming dataset."""
    if SAMPLE_CSV_PATH.exists():
        return pd.read_csv(SAMPLE_CSV_PATH)
    # Fallback minimal dataframe if missing
    return pd.DataFrame({"Destination Port": [80, 443, 22], "Label": ["BENIGN", "BENIGN", "PortScan"]})


def get_active_dataset() -> pd.DataFrame:
    """Returns uploaded dataset if available in session_state, else loads default."""
    if "custom_dataset" in st.session_state and st.session_state["custom_dataset"] is not None:
        return st.session_state["custom_dataset"]
    return load_default_dataset()


def set_custom_dataset(df: pd.DataFrame, source_name: str = "Uploaded CSV") -> None:
    """Stores user uploaded dataframe in session state."""
    st.session_state["custom_dataset"] = df
    st.session_state["dataset_source_name"] = source_name

