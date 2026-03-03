class HybridRecommender:
    def __init__(self, content_rec, collab_rec):
        self.content_rec = content_rec
        self.collab_rec = collab_rec

    def recommend(self, user_id, product_id, all_product_ids, n=5):
        """
        Combines content predictions (based on current/last product) and Collab predictions (based on user).
        Weighted approach or simple interleaving. Here we use interleaving for diversity.
        """
        content_recs = self.content_rec.recommend(product_id, n=n)
        collab_recs = self.collab_rec.recommend(user_id, all_product_ids, n=n)

        combined = []
        # Interleave
        for c, co in zip(content_recs, collab_recs):
            if c not in combined:
                combined.append(c)
            if co not in combined:
                combined.append(co)

        return combined[:n]
