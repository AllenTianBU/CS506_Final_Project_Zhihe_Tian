# Test functions are working
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from linear_regression_predictor import load_and_clean, score_to_label, FEATURES

def test_load_and_clean():
    df = load_and_clean()
    assert len(df) > 0
    assert 'happiness' in df.columns
    assert all(col in df.columns for col in FEATURES)
    assert df.isnull().sum().sum() == 0

def test_score_to_label():
    assert score_to_label(2.0) == "Very Happy"
    assert score_to_label(1.5) == "Pretty Happy"
    assert score_to_label(0.5) == "Not Too Happy"
