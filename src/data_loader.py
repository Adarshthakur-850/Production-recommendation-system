import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta


def load_data(filepath="data/ecommerce_data.csv"):
    """
    Loads data. Generates synthetic data if file is missing.
    """
    if not os.path.exists(filepath):
        print(f"File {filepath} not found. Generating detailed synthetic data...")
        generate_synthetic_data(filepath)

    return pd.read_csv(filepath)


def generate_synthetic_data(filepath):
    """
    Generates a realistic e-commerce dataset.
    """
    np.random.seed(42)
    random.seed(42)

    n_users = 100
    n_products = 50
    n_interactions = 2000

    categories = ["Electronics", "Clothing", "Home", "Books", "Beauty"]

    # Generate Products
    products = []
    for pid in range(1, n_products + 1):
        cat = random.choice(categories)
        products.append(
            {
                "product_id": pid,
                "product_name": f"Product {pid} - {cat}",
                "category": cat,
                "description": f"This is a great {cat.lower()} product with features X, Y, Z.",
            }
        )
    products_df = pd.DataFrame(products)

    # Generate Interactions
    data = []
    start_date = datetime.now() - timedelta(days=365)

    for _ in range(n_interactions):
        user_id = random.randint(1, n_users)
        prod = products_df.sample(1).iloc[0]

        rating = random.choices([1, 2, 3, 4, 5], weights=[0.05, 0.05, 0.15, 0.35, 0.4])[
            0
        ]

        # Random timestamp within last year
        random_days = random.randint(0, 365)
        timestamp = start_date + timedelta(days=random_days)

        data.append(
            {
                "user_id": user_id,
                "product_id": prod["product_id"],
                "product_name": prod["product_name"],
                "category": prod["category"],
                "rating": rating,
                "review_text": f"Product {prod['product_id']} is {random.choice(['good', 'bad', 'okay', 'excellent', 'horrible'])}.",
                "description": prod["description"],
                "timestamp": timestamp,
            }
        )

    df = pd.DataFrame(data)

    # Ensure dir exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Synthetic data (shape {df.shape}) saved to {filepath}")
