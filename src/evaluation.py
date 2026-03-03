import numpy as np
from sklearn.metrics import mean_squared_error

def get_rmse(model, test_df):
    """
    Calculates RMSE for Collaborative Filtering model.
    """
    predictions = []
    actuals = []
    
    for _, row in test_df.iterrows():
        # Use predict_rating method
        pred = model.predict_rating(row['user_id'], row['product_id'])
        predictions.append(pred)
        actuals.append(row['rating'])
        
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    return rmse
