import pandas as pd


class PopularityRecommender:
    def __init__(self):
        self.popular_items = None

    def fit(self, df):
        """
        Calculates top rated products based on average rating and vote count.
        """
        # Calculate mean rating and count
        agg = df.groupby("product_id").agg({"rating": ["mean", "count"]})
        agg.columns = ["mean_rating", "count"]

        # Simple weighted score: (R * v + C * m) / (v + m)
        # R = average for the item
        # v = number of votes for the item
        # m = minimum votes required to be listed
        # C = the mean vote across the whole report

        C = agg["mean_rating"].mean()
        m = agg["count"].quantile(0.5)  # Top 50%

        self.popular_items = agg.loc[agg["count"] >= m].copy()
        v = self.popular_items["count"]
        R = self.popular_items["mean_rating"]

        self.popular_items["score"] = (v / (v + m) * R) + (m / (m + v) * C)
        self.popular_items = self.popular_items.sort_values("score", ascending=False)

    def recommend(self, n=5):
        if self.popular_items is None:
            raise Exception("Model not fitted")
        return self.popular_items.head(n).index.tolist()
