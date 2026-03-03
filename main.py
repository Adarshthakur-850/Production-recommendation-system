import sys
import os
import pandas as pd
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.eda import run_eda
from src.recommender_popularity import PopularityRecommender
from src.recommender_content import ContentRecommender
from src.recommender_collab import CollaborativeRecommender
from src.recommender_hybrid import HybridRecommender
from src.evaluation import get_rmse
from src.visualization import plot_heatmap

def main():
    print("Starting Product Recommendation System Pipeline...")
    
    # 1. Load Data
    df = load_data()
    print(f"Data Loaded: {df.shape}")
    
    # 2. Preprocessing
    df, user_item_matrix = preprocess_data(df)
    
    # 3. EDA
    run_eda(df)
    plot_heatmap(user_item_matrix.iloc[:50, :50]) # Plot subset for speed
    
    # 4. Train Models
    print("Training models...")
    
    # Popularity
    pop_model = PopularityRecommender()
    pop_model.fit(df)
    print(f"Top 5 Popular Items: {pop_model.recommend(5)}")
    
    # Content-Based
    content_model = ContentRecommender()
    content_model.fit(df)
    # Recommend items similar to Product 1
    print(f"Content Recs for Product 1: {content_model.recommend(1, 5)}")
    
    # Collaborative
    collab_model = CollaborativeRecommender()
    collab_model.fit(df)
    # Evaluate RMSE on full set (just for demo)
    rmse = get_rmse(collab_model, df)
    print(f"Collaborative Model RMSE: {rmse:.4f}")
    
    # Hybrid
    hybrid_model = HybridRecommender(content_model, collab_model)
    # Recommend for User 1, based on Product 1 being currently viewed
    all_pids = df['product_id'].unique()
    hybrid_recs = hybrid_model.recommend(user_id=1, product_id=1, all_product_ids=all_pids, n=5)
    print(f"Hybrid Recs for User 1 (viewing Prod 1): {hybrid_recs}")
    
    print("\nPipeline completed successfully.")

if __name__ == "__main__":
    main()
