import pandas as pd
import numpy as np


def preprocess_data(df):
    """
    Cleans data, parses dates, and creates interaction matrix.
    """
    print("Preprocessing data...")

    # 1. Handle Duplicates
    df = df.drop_duplicates(["user_id", "product_id"], keep="last")

    # 2. Parse Timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 3. Create User-Item Matrix
    user_item_matrix = df.pivot(
        index="user_id", columns="product_id", values="rating"
    ).fillna(0)

    return df, user_item_matrix
