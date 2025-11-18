"""Feature engineering stubs for order book features."""
import numpy as np
import pandas as pd


def compute_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with example engineered features (midprice, spread)."""
    # Expect df to include best bid/ask columns: bid_price, ask_price, bid_size, ask_size
    out = df.copy()
    out['mid_price'] = (out['bid_price'] + out['ask_price']) / 2
    out['spread'] = out['ask_price'] - out['bid_price']
    return out