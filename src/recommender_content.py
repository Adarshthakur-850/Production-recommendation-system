import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class ContentRecommender:
    def __init__(self):
        self.tfidf_matrix = None
        self.indices = None
        self.df_products = None
        
    def fit(self, df):
        # Create a product catalogue
        self.df_products = df[['product_id', 'product_name', 'category', 'description']].drop_duplicates('product_id').reset_index(drop=True)
        # Combine features
        self.df_products['soup'] = self.df_products['description'] + " " + self.df_products['category']
        
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.df_products['soup'])
        
        self.indices = pd.Series(self.df_products.index, index=self.df_products['product_id']).drop_duplicates()
        
    def recommend(self, product_id, n=5):
        if product_id not in self.indices:
            return []
            
        idx = self.indices[product_id]
        
        # Calculate cosine similarity of this product with all others
        sim_scores = linear_kernel(self.tfidf_matrix[idx:idx+1], self.tfidf_matrix).flatten()
        
        # Get indices of top scores
        sim_indices = sim_scores.argsort()[-(n+1):-1][::-1]
        
        return self.df_products['product_id'].iloc[sim_indices].tolist()
