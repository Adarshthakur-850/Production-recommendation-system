<<<<<<< HEAD
# Product Recommendation System

A production-quality recommendation engine using Collaborative, Content-Based, and Hybrid filtering.

## Project Structure
- `data/`: Contains the dataset (synthetic data generated if missing).
- `models/`: Stores trained models.
- `plots/`: Stores EDA and visualization plots.
- `src/`: Source code modules.
- `main.py`: Main pipeline orchestrator.

## Features
- **Data Engineering**: Synthetic data generation, timestamp parsing, interaction matrix creation.
- **EDA**: Visualizes rating distribution and top products.
- **Recommenders**:
    - **Popularity**: Baseline based on weighted rating.
    - **Content-Based**: TF-IDF on product descriptions.
    - **Collaborative**: Matrix Factorization (SVD) using `surpriselib`.
    - **Hybrid**: Combines Content and Collaborative results.
- **Evaluation**: RMSE metric.
- **Visualization**: User-Item interaction heatmap.

## Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the full pipeline:
```bash
python main.py
```
This will:
1. Load/Generate data.
2. clean and preprocess.
3. Generate EDA plots in `plots/`.
4. Train all recommender models.
5. Output recommendations and RMSE.

## Future Improvements
- Implement precision@k and recall@k.
- Wrap `main.py` in a FastAPI interface.
- dockerize the application.
=======
# Production-recommendation-system
ml project
>>>>>>> f0fc56d7e78ebc5238dedf6c7bc2db47e8e9de9a
