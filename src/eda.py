import matplotlib.pyplot as plt
import seaborn as sns
import os


def run_eda(df, plots_dir="plots"):
    """
    Generates and saves EDA plots.
    """
    print("Running EDA...")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # 1. Rating Distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(x="rating", data=df)
    plt.title("Rating Distribution")
    plt.savefig(f"{plots_dir}/rating_distribution.png")
    plt.close()

    # 2. Top Products by Rating Count
    top_products = df["product_name"].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_products.values, y=top_products.index)
    plt.title("Top 10 Most Rated Products")
    plt.savefig(f"{plots_dir}/top_products.png")
    plt.close()

    print(f"EDA plots saved to {plots_dir}")
