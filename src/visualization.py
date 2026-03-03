import seaborn as sns
import matplotlib.pyplot as plt

def plot_heatmap(user_item_matrix, save_path="plots/heatmap.png"):
    plt.figure(figsize=(12, 8))
    sns.heatmap(user_item_matrix, cmap='viridis', robust=True)
    plt.title('User-Item Interaction Matrix')
    plt.savefig(save_path)
    plt.close()
    print(f"Heatmap saved to {save_path}")
