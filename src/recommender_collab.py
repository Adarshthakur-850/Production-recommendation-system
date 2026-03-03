from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np


class CollaborativeRecommender:
    def __init__(self, n_components=20):
        self.model = None
        self.user_item_matrix = None
        self.n_components = n_components

    def fit(self, df):
        # Create user-item matrix
        self.user_item_matrix = df.pivot(
            index="user_id", columns="product_id", values="rating"
        ).fillna(0)

        # SVD
        self.model = TruncatedSVD(
            n_components=min(self.n_components, self.user_item_matrix.shape[1] - 1),
            random_state=42,
        )
        self.matrix_reduced = self.model.fit_transform(self.user_item_matrix)
        self.corr_matrix = np.corrcoef(self.matrix_reduced)

    def recommend(self, user_id, all_product_ids, n=5):
        if user_id not in self.user_item_matrix.index:
            return []

        # Reconstruct matrix approximation
        reconstructed_matrix = np.dot(self.matrix_reduced, self.model.components_)

        # Get user index
        user_idx = self.user_item_matrix.index.get_loc(user_id)

        # Get predicted ratings for this user
        predicted_ratings = reconstructed_matrix[user_idx]

        # Create series
        pred_series = pd.Series(predicted_ratings, index=self.user_item_matrix.columns)

        # Filter already rated items (optional, but good practice)
        # rated_items = self.user_item_matrix.loc[user_id]
        # pred_series[rated_items > 0] = 0

        return pred_series.sort_values(ascending=False).head(n).index.tolist()

    def predict_rating(self, user_id, product_id):
        if (
            user_id not in self.user_item_matrix.index
            or product_id not in self.user_item_matrix.columns
        ):
            return 0  # Default fallback

        user_idx = self.user_item_matrix.index.get_loc(user_id)
        prod_idx = self.user_item_matrix.columns.get_loc(product_id)

        # Recompute single element (inefficient for heavy use, but fine for demo)
        pred = np.dot(
            self.matrix_reduced[user_idx], self.model.components_[:, prod_idx]
        )
        return pred
